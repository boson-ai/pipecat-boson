# ruff: noqa: CPY001
from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import warnings
from types import SimpleNamespace

import pytest
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    AggregatedTextFrame,
    EndFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallResultProperties,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMConfigureOutputFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMSetToolsFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import NOT_GIVEN as LLM_CONTEXT_NOT_GIVEN
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.realtime import events
from pipecat.tests.utils import run_test
from pipecat.utils import deprecation as pipecat_deprecation
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

import pipecat_boson.realtime.llm as realtime_llm
from pipecat_boson.realtime import BosonRealtimeLLMService
from pipecat_boson.realtime._compat import HANDLE_CONTEXT_TAKES_DIRECTION, SENTENCE_TEXT, forget_answered_call
from pipecat_boson.realtime.config import DEFAULT_TURN_DETECTION


# pipecat-ai >= 1.8.0 gives run_test() a 1-second default budget for the pipeline
# to start, which a cold CI runner can exceed. It is only an upper bound, so raise
# it where supported; older run_test() has no start timeout.
RUN_TEST_START_KWARGS = {"start_timeout": 10.0} if "start_timeout" in inspect.signature(run_test).parameters else {}


class CapturingBosonRealtimeLLMService(BosonRealtimeLLMService):
    """Fake that captures outgoing client events, pushed frames, and errors."""

    def __init__(self, **kwargs):
        super().__init__(
            url="ws://localhost:1234/v1/realtime/",
            api_key="test-key",
            model="model-1",
            **kwargs,
        )
        self.sent: list[dict] = []
        self.pushed: list[Frame] = []
        self.errors: list[str] = []

    async def _ws_send(self, realtime_message):
        self.sent.append(realtime_message)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        self.pushed.append(frame)
        return None

    async def push_error(self, error_msg: str, exception: Exception | None = None, fatal: bool = False):
        self.errors.append(error_msg)

    async def start_processing_metrics(self):
        return None

    async def start_ttfb_metrics(self):
        return None

    async def stop_processing_metrics(self):
        return None

    async def stop_all_metrics(self):
        return None


class RaceResponseCreatedBosonRealtimeLLMService(CapturingBosonRealtimeLLMService):
    async def _ws_send(self, realtime_message):
        if realtime_message["type"] == "response.create":
            event_id = realtime_message["event_id"]
            assert event_id in self._boson_pending_response_client_event_ids
            self._handle_evt_response_created(make_response_created("resp_race", event_id))
        self.sent.append(realtime_message)


class CapturingErrorBosonRealtimeLLMService(BosonRealtimeLLMService):
    """Fake with real _ws_send/push_frame, for websocket-level tests."""

    def __init__(self, **kwargs):
        super().__init__(
            url="ws://localhost:1234/v1/realtime/",
            api_key="test-key",
            model="model-1",
            **kwargs,
        )
        self.errors: list[str] = []
        self.metrics_stopped = False

    async def push_error(self, error_msg: str, exception: Exception | None = None, fatal: bool = False):
        self.errors.append(error_msg)

    async def stop_all_metrics(self):
        self.metrics_stopped = True


class FakeWebSocket:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class ClosingOKWebSocket(FakeWebSocket):
    async def send(self, payload: str):
        raise ConnectionClosedOK(None, None, None)


class ClosingErrorWebSocket(FakeWebSocket):
    async def send(self, payload: str):
        raise ConnectionClosedError(None, None, None)


class ExhaustedWebSocket(FakeWebSocket):
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class TerminalThenClosedWebSocket(FakeWebSocket):
    def __init__(self, terminal_event_type: str):
        super().__init__()
        self._terminal_event_type = terminal_event_type
        self._sent_terminal_event = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._sent_terminal_event:
            self._sent_terminal_event = True
            return f'{{"type": "{self._terminal_event_type}"}}'
        raise ConnectionClosedError(None, None, None)


def make_error_event(error_type: str, message: str = "", code: str | None = None) -> dict:
    error: dict = {"type": error_type, "message": message}
    if code is not None:
        error["code"] = code
    return {"type": "error", "error": error}


def make_response_created(response_id: str, client_event_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        response=SimpleNamespace(id=response_id, metadata=SimpleNamespace(client_event_id=client_event_id))
    )


def make_response_done(response_id: str | None = None, *, status: str = "completed") -> SimpleNamespace:
    response = SimpleNamespace(usage=None, status=status, output=[])
    if response_id is not None:
        response.id = response_id
    return SimpleNamespace(response=response)


@pytest.mark.asyncio
async def test_create_response_sends_metadata_only_response_payload():
    service = CapturingBosonRealtimeLLMService()
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False

    await service._create_response()

    payload = service.sent[-1]
    event_id = payload["event_id"]
    assert payload["type"] == "response.create"
    assert payload["response"] == {"metadata": {"client_event_id": event_id}}
    assert "output_modalities" not in payload["response"]
    assert "modalities" not in payload["response"]


@pytest.mark.asyncio
async def test_send_client_event_maps_pipecat_response_output_modalities():
    service = CapturingBosonRealtimeLLMService()

    await service.send_client_event(
        events.ResponseCreateEvent(response=events.ResponseProperties(output_modalities=["text"]))
    )

    payload = service.sent[-1]
    assert payload["type"] == "response.create"
    assert payload["response"]["modalities"] == ["text"]
    assert "output_modalities" not in payload["response"]


@pytest.mark.asyncio
async def test_send_tool_result_does_not_double_encode_serialized_json():
    service = CapturingBosonRealtimeLLMService()
    result = json.dumps(
        {
            "ok": True,
            "results": [{"title": "Current result", "url": "https://example.com"}],
        },
        ensure_ascii=False,
    )

    await service._send_tool_result("call_search", result)

    payload = service.sent[-1]
    assert payload["type"] == "conversation.item.create"
    assert payload["item"]["type"] == "function_call_output"
    assert payload["item"]["call_id"] == "call_search"
    assert payload["item"]["output"] == result
    assert json.loads(payload["item"]["output"]) == {
        "ok": True,
        "results": [{"title": "Current result", "url": "https://example.com"}],
    }


@pytest.mark.asyncio
async def test_pipecat_aggregator_tool_result_is_encoded_once_on_boson_wire():
    result = {
        "ok": True,
        "results": [{"title": "Current result", "url": "https://example.com"}],
    }
    context = LLMContext()
    assistant_aggregator = LLMContextAggregatorPair(context).assistant()

    _, upstream_frames = await run_test(
        assistant_aggregator,
        frames_to_send=[
            FunctionCallInProgressFrame(
                function_name="search",
                tool_call_id="call_search",
                arguments={"query": "current result"},
                cancel_on_interruption=True,
            ),
            FunctionCallResultFrame(
                function_name="search",
                tool_call_id="call_search",
                arguments={"query": "current result"},
                result=result,
            ),
        ],
        expected_up_frames=[LLMContextFrame],
        **RUN_TEST_START_KWARGS,
    )

    service = CapturingBosonRealtimeLLMService()
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False
    service._context = LLMContext([context.get_messages()[0]])
    await service._handle_context(upstream_frames[0].context, FrameDirection.UPSTREAM)

    payload = service.sent[-2]
    assert payload["type"] == "conversation.item.create"
    assert payload["item"]["type"] == "function_call_output"
    assert payload["item"]["output"] == json.dumps(result, ensure_ascii=False)
    assert json.loads(payload["item"]["output"]) == result


@pytest.mark.asyncio
async def test_session_update_omits_unspecified_temperature():
    service = CapturingBosonRealtimeLLMService()

    await service._send_session_update()

    payload = service.sent[-1]
    assert payload["type"] == "session.update"
    assert "temperature" not in payload["session"]
    assert payload["session"]["audio"]["output"]["speed"] == 1.0


def test_constructor_uses_public_realtime_model_by_default():
    service = BosonRealtimeLLMService(url="ws://localhost:1234/v1/realtime/")

    assert service._boson_model == "higgs-realtime"


def test_constructor_uses_openai_output_modalities_name():
    service = CapturingBosonRealtimeLLMService(output_modalities=["text"])

    assert service._boson_output_modalities == ["text"]
    assert service._settings.session_properties.output_modalities == ["text"]


@pytest.mark.asyncio
async def test_constructor_tools_are_advertised_on_initial_session_update():
    async def get_weather(params, location: str = "San Francisco") -> None:
        """Get weather.

        Args:
            location: City or place name.
        """

    service = CapturingBosonRealtimeLLMService(tools=[get_weather])

    await service._send_session_update()

    payload = service.sent[-1]
    assert payload["type"] == "session.update"
    assert payload["session"]["tools"][0]["name"] == "get_weather"


def test_initializes_pipecat_realtime_typed_audio_settings():
    service = CapturingBosonRealtimeLLMService(
        input_audio_transcription={"model": "asr-v1", "language": "zh"},
        input_audio_noise_reduction="near_field",
    )

    audio_input = service._settings.session_properties.audio.input

    assert audio_input.transcription.model == "asr-v1"
    assert audio_input.transcription.language == "zh"
    assert audio_input.noise_reduction.type == "near_field"


@pytest.mark.asyncio
async def test_update_settings_sends_synchronized_boson_session_update():
    service = CapturingBosonRealtimeLLMService(max_output_tokens=64)

    changed = await service._update_settings(
        service.Settings(
            model="model-2",
            system_instruction="Be brief.",
            temperature=0.2,
            max_tokens=999999,
            session_properties=events.SessionProperties(
                output_modalities=["text"],
                tool_choice="none",
                audio=events.AudioConfiguration(
                    input=events.AudioInput(
                        transcription=events.InputAudioTranscription(model="asr-v1", prompt="Ignore me")
                    ),
                    output=events.AudioOutput(voice="voice-2", speed=1.25),
                ),
            ),
        )
    )

    assert {"model", "system_instruction", "temperature", "max_tokens", "session_properties"} <= set(changed)
    payload = service.sent[-1]
    session = payload["session"]
    assert payload["type"] == "session.update"
    assert session["model"] == "model-2"
    assert session["instructions"] == "Be brief."
    assert session["output_modalities"] == ["text"]
    assert session["audio"]["input"]["transcription"] == {"model": "asr-v1"}
    assert session["audio"]["output"]["voice"] == "voice-2"
    assert session["audio"]["output"]["speed"] == 1.25
    assert session["temperature"] == 0.2
    assert session["max_output_tokens"] == 4096
    assert session["tool_choice"] == "none"


@pytest.mark.asyncio
async def test_update_settings_can_clear_audio_input_transcription_and_noise_reduction():
    service = CapturingBosonRealtimeLLMService(
        input_audio_transcription={"model": "asr-v1"},
        input_audio_noise_reduction="near_field",
    )

    await service._update_settings(
        service.Settings(
            session_properties=events.SessionProperties(audio=events.AudioConfiguration(input=events.AudioInput())),
        )
    )

    # Nothing changed, so nothing is sent.
    assert service.sent == []

    await service._update_settings(service.Settings(session_properties=events.SessionProperties(tool_choice="none")))

    unchanged_input = service.sent[-1]["session"]["audio"]["input"]
    assert unchanged_input["transcription"] == {"model": "asr-v1"}
    assert unchanged_input["noise_reduction"] == {"type": "near_field"}

    await service._update_settings(
        service.Settings(
            session_properties=events.SessionProperties(
                audio=events.AudioConfiguration(input=events.AudioInput(transcription=None, noise_reduction=None))
            ),
        )
    )

    cleared_input = service.sent[-1]["session"]["audio"]["input"]
    assert cleared_input["transcription"] is None
    assert cleared_input["noise_reduction"] is None
    assert service.user_transcription_enabled is False


@pytest.mark.asyncio
async def test_set_tools_frame_updates_advertised_session_tools():
    service = CapturingBosonRealtimeLLMService()
    tool = FunctionSchema(
        name="get_weather",
        description="Get weather.",
        properties={"location": {"type": "string"}},
        required=["location"],
    )
    frame = LLMSetToolsFrame(tools=[tool])

    await service.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert service.sent[-1]["type"] == "session.update"
    assert service.sent[-1]["session"]["tools"][0]["name"] == "get_weather"
    assert frame in service.pushed


@pytest.mark.asyncio
async def test_set_tools_frame_can_clear_advertised_session_tools():
    service = CapturingBosonRealtimeLLMService()
    tool = FunctionSchema(
        name="get_weather",
        description="Get weather.",
        properties={"location": {"type": "string"}},
        required=["location"],
    )

    await service.process_frame(LLMSetToolsFrame(tools=[tool]), FrameDirection.DOWNSTREAM)
    await service.process_frame(LLMSetToolsFrame(tools=LLM_CONTEXT_NOT_GIVEN), FrameDirection.DOWNSTREAM)

    assert service.sent[-1]["type"] == "session.update"
    assert service.sent[-1]["session"]["tools"] == []


@pytest.mark.asyncio
async def test_session_created_marks_session_ready_and_runs_pending_llm():
    service = CapturingBosonRealtimeLLMService()
    service._run_llm_when_api_session_ready = True
    service._llm_needs_conversation_setup = False

    await service._handle_evt_session_created(SimpleNamespace())

    assert service._api_session_ready is True
    assert service.sent[-1]["type"] == "response.create"


@pytest.mark.asyncio
async def test_messages_append_sends_text_conversation_item_and_creates_response():
    service = CapturingBosonRealtimeLLMService()
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False
    service._context = LLMContext([])

    await service._handle_messages_append(
        LLMMessagesAppendFrame(messages=[{"role": "user", "content": "hello"}], run_llm=True)
    )

    assert [event["type"] for event in service.sent[-2:]] == [
        "conversation.item.create",
        "response.create",
    ]
    assert service.sent[-2]["item"]["role"] == "user"
    assert service.sent[-2]["item"]["content"][0]["text"] == "hello"
    assert service._context.get_messages()[-1] == {"role": "user", "content": "hello"}


@pytest.mark.asyncio
async def test_text_input_transcription_is_forwarded():
    service = CapturingBosonRealtimeLLMService()
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False
    service._context = LLMContext([])

    await service._handle_messages_append(
        LLMMessagesAppendFrame(messages=[{"role": "user", "content": "hello"}], run_llm=True)
    )
    text_item_id = service.sent[-2]["item"]["id"]

    await service.handle_evt_input_audio_transcription_completed(
        SimpleNamespace(item_id=text_item_id, transcript="hello")
    )

    transcriptions = [frame for frame in service.pushed if isinstance(frame, TranscriptionFrame)]
    assert len(transcriptions) == 1
    assert transcriptions[0].text == "hello"


@pytest.mark.asyncio
async def test_audio_transcription_completed_is_forwarded_once():
    service = CapturingBosonRealtimeLLMService()

    event = SimpleNamespace(item_id="audio_item_1", transcript="hello from voice")
    await service.handle_evt_input_audio_transcription_completed(event)
    await service.handle_evt_input_audio_transcription_completed(event)

    transcriptions = [frame for frame in service.pushed if isinstance(frame, TranscriptionFrame)]
    assert len(transcriptions) == 1
    assert transcriptions[0].text == "hello from voice"


@pytest.mark.asyncio
async def test_completed_transcription_dedup_cache_is_bounded():
    service = CapturingBosonRealtimeLLMService()

    for index in range(realtime_llm._BOSON_TRANSCRIPTION_DEDUP_CACHE_SIZE + 1):
        await service.handle_evt_input_audio_transcription_completed(
            SimpleNamespace(item_id=f"audio_item_{index}", transcript=f"transcript {index}")
        )

    assert len(service._boson_completed_transcription_item_ids) == realtime_llm._BOSON_TRANSCRIPTION_DEDUP_CACHE_SIZE
    assert "audio_item_0" not in service._boson_completed_transcription_item_ids


@pytest.mark.asyncio
async def test_input_audio_transcription_failed_logs_warning_and_continues(monkeypatch):
    service = CapturingBosonRealtimeLLMService()
    warnings = []
    monkeypatch.setattr(realtime_llm.logger, "warning", warnings.append)

    should_continue = await service._dispatch_boson_server_event(
        {
            "type": "conversation.item.input_audio_transcription.failed",
            "item_id": "audio_item_1",
            "error": {"message": "ASR unavailable"},
        }
    )

    assert should_continue is True
    assert warnings == ["Boson input audio transcription failed for item audio_item_1: ASR unavailable"]


@pytest.mark.asyncio
async def test_messages_append_cancels_active_response_before_creating_text_response():
    service = CapturingBosonRealtimeLLMService()
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False
    service._context = LLMContext([])
    await service._create_response()

    await service._handle_messages_append(
        LLMMessagesAppendFrame(messages=[{"role": "user", "content": "interrupt"}], run_llm=True)
    )

    assert [event["type"] for event in service.sent[-3:]] == [
        "response.cancel",
        "conversation.item.create",
        "response.create",
    ]
    assert sum(isinstance(frame, LLMFullResponseStartFrame) for frame in service.pushed) == 2
    assert sum(isinstance(frame, LLMFullResponseEndFrame) for frame in service.pushed) == 1


@pytest.mark.asyncio
async def test_response_created_can_arrive_while_response_create_is_sending():
    service = RaceResponseCreatedBosonRealtimeLLMService()
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False

    await service._create_response()

    assert service._boson_pending_response_client_event_ids == set()
    assert service._boson_active_response_ids == {"resp_race"}


@pytest.mark.asyncio
async def test_function_call_output_echo_clears_manual_id_tracking():
    service = CapturingBosonRealtimeLLMService()
    service._messages_added_manually["item_tool"] = True

    await service._handle_evt_conversation_item_added(
        SimpleNamespace(item=SimpleNamespace(id="item_tool", type="function_call_output"))
    )

    assert "item_tool" not in service._messages_added_manually
    assert not any(isinstance(frame, LLMFullResponseStartFrame) for frame in service.pushed)


@pytest.mark.asyncio
async def test_manual_assistant_message_echo_is_deduped_by_item_id():
    # Contract: the server preserves client-supplied item ids and echoes them on
    # conversation.item.added, so id-based dedup covers manually added messages.
    service = CapturingBosonRealtimeLLMService()
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False
    service._context = LLMContext([])

    await service._handle_messages_append(
        LLMMessagesAppendFrame(messages=[{"role": "assistant", "content": "hello"}], run_llm=False)
    )
    item = service.sent[-1]["item"]
    assert item["id"]
    assert service._messages_added_manually == {item["id"]: True}

    await service._handle_evt_conversation_item_added(
        SimpleNamespace(
            item=SimpleNamespace(
                id=item["id"],
                type="message",
                role="assistant",
                content=[SimpleNamespace(type="text", text="hello")],
            )
        )
    )

    assert service._messages_added_manually == {}
    assert not any(isinstance(frame, LLMFullResponseStartFrame) for frame in service.pushed)
    assert service._current_assistant_response is None


@pytest.mark.asyncio
async def test_conversation_item_added_tracks_function_calls():
    service = CapturingBosonRealtimeLLMService()
    item = SimpleNamespace(id="item_fc", type="function_call", call_id="call_1")

    await service._handle_evt_conversation_item_added(SimpleNamespace(item=item))

    assert service._pending_function_calls["call_1"] is item


@pytest.mark.asyncio
async def test_assistant_conversation_item_added_does_not_duplicate_manual_response_start():
    service = CapturingBosonRealtimeLLMService()
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False
    await service._create_response()

    assistant_item = SimpleNamespace(id="item_assistant", type="message", role="assistant")
    await service._handle_evt_conversation_item_added(SimpleNamespace(item=assistant_item))

    start_frames = [frame for frame in service.pushed if isinstance(frame, LLMFullResponseStartFrame)]
    assert len(start_frames) == 1
    assert service._current_assistant_response is assistant_item


@pytest.mark.asyncio
async def test_assistant_conversation_item_added_starts_server_generated_response():
    service = CapturingBosonRealtimeLLMService()

    assistant_item = SimpleNamespace(id="item_assistant", type="message", role="assistant")
    await service._handle_evt_conversation_item_added(SimpleNamespace(item=assistant_item))

    assert any(isinstance(frame, LLMFullResponseStartFrame) for frame in service.pushed)
    assert service._current_assistant_response is assistant_item


@pytest.mark.asyncio
async def test_response_not_active_error_is_ignored_for_idempotent_cancel():
    service = CapturingBosonRealtimeLLMService()
    service._boson_pending_response_client_event_ids.add("stale")
    service._boson_active_response_ids.add("resp_stale")

    await service._dispatch_boson_server_event(
        make_error_event("invalid_request_error", "No active response to cancel.", code="response_not_active")
    )

    assert service.errors == []
    assert service._boson_pending_response_client_event_ids == set()
    assert service._boson_active_response_ids == set()


@pytest.mark.asyncio
async def test_nonfatal_boson_error_keeps_receive_loop_alive():
    service = CapturingBosonRealtimeLLMService()

    should_continue = await service._dispatch_boson_server_event(
        make_error_event("voice_output_task_ongoing", "Voice output task is ongoing. Skipping.")
    )

    assert should_continue is True
    assert service.errors == []


@pytest.mark.asyncio
async def test_conversation_item_not_found_fails_pending_retrieve_without_stopping_receive_loop():
    service = CapturingBosonRealtimeLLMService()
    future = asyncio.get_running_loop().create_future()
    service._retrieve_conversation_item_futures["missing_item"] = [future]

    should_continue = await service._dispatch_boson_server_event(
        make_error_event("conversation_item_not_found", "Conversation item not found: missing_item")
    )

    assert should_continue is True
    with pytest.raises(Exception, match="missing_item"):
        await future


@pytest.mark.asyncio
async def test_conversation_item_not_found_without_pending_retrieve_is_nonfatal():
    service = CapturingBosonRealtimeLLMService()

    should_continue = await service._dispatch_boson_server_event(
        make_error_event("conversation_item_not_found", "Conversation item not found: missing_item")
    )

    assert should_continue is True
    assert service.errors == []


@pytest.mark.asyncio
async def test_unhandled_error_event_stops_receive_loop():
    service = CapturingBosonRealtimeLLMService()

    should_continue = await service._dispatch_boson_server_event(
        make_error_event("invalid_request_error", "Bad request.", code="invalid_value")
    )

    assert should_continue is False
    assert len(service.errors) == 1
    assert "Bad request" in service.errors[0]


@pytest.mark.asyncio
async def test_messages_append_cancels_after_stale_cancelled_response_done():
    service = CapturingBosonRealtimeLLMService()
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False
    service._context = LLMContext([])

    await service._create_response()
    await service._cancel_active_response()
    await service._create_response()
    await service._handle_evt_response_done(make_response_done(status="cancelled"))

    await service._handle_messages_append(
        LLMMessagesAppendFrame(messages=[{"role": "user", "content": "interrupt again"}], run_llm=True)
    )

    assert [event["type"] for event in service.sent[-3:]] == [
        "response.cancel",
        "conversation.item.create",
        "response.create",
    ]


@pytest.mark.asyncio
async def test_stale_cancelled_response_done_does_not_end_new_response():
    service = CapturingBosonRealtimeLLMService()
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False

    await service._create_response()
    service._handle_evt_response_created(make_response_created("resp_a", service.sent[-1]["event_id"]))
    await service._cancel_active_response()
    await service._create_response()
    service._handle_evt_response_created(make_response_created("resp_b", service.sent[-1]["event_id"]))

    await service._handle_evt_response_done(make_response_done("resp_a", status="cancelled"))

    assert service._boson_active_response_ids == {"resp_b"}
    assert service._boson_response_start_frame_active is True
    assert sum(isinstance(frame, LLMFullResponseStartFrame) for frame in service.pushed) == 2
    assert sum(isinstance(frame, LLMFullResponseEndFrame) for frame in service.pushed) == 1


@pytest.mark.asyncio
async def test_cancelled_response_created_after_cancel_filters_late_deltas():
    service = CapturingBosonRealtimeLLMService()
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False

    await service._create_response()
    response_create = service.sent[-1]
    await service._cancel_active_response()
    service._handle_evt_response_created(make_response_created("resp_cancelled", response_create["event_id"]))
    pushed_count = len(service.pushed)

    await service._handle_evt_text_delta(SimpleNamespace(response_id="resp_cancelled", delta="old text"))
    await service._handle_evt_audio_transcript_delta(
        SimpleNamespace(response_id="resp_cancelled", delta="old transcript")
    )
    await service._handle_evt_audio_delta(
        SimpleNamespace(
            response_id="resp_cancelled",
            item_id="item_cancelled",
            content_index=0,
            output_index=0,
            delta="AA==",
        )
    )

    assert service._boson_active_response_ids == set()
    assert "resp_cancelled" in service._boson_cancelled_response_ids
    assert len(service.pushed) == pushed_count


@pytest.mark.asyncio
async def test_interruption_tracks_cancelled_response_and_filters_late_events():
    service = CapturingBosonRealtimeLLMService()
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False

    await service._create_response()
    service._handle_evt_response_created(make_response_created("resp_a", service.sent[-1]["event_id"]))

    await service._handle_interruption()

    assert service.sent[-1]["type"] == "response.cancel"
    assert service._boson_active_response_ids == set()
    assert service._boson_cancelled_response_ids == {"resp_a"}
    assert service._boson_response_start_frame_active is False
    assert sum(isinstance(frame, LLMFullResponseEndFrame) for frame in service.pushed) == 1

    pushed_count = len(service.pushed)
    await service._handle_evt_text_delta(SimpleNamespace(response_id="resp_a", delta="old text"))
    await service._handle_evt_response_done(make_response_done("resp_a", status="cancelled"))

    assert len(service.pushed) == pushed_count
    assert service._boson_cancelled_response_ids == set()
    assert sum(isinstance(frame, LLMFullResponseEndFrame) for frame in service.pushed) == 1


@pytest.mark.asyncio
async def test_server_vad_speech_started_keeps_response_active_until_server_done():
    service = CapturingBosonRealtimeLLMService()
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False
    broadcasted = []

    async def broadcast_frame(frame_cls):
        broadcasted.append(frame_cls)

    async def broadcast_interruption():
        broadcasted.append(InterruptionFrame)

    service.broadcast_frame = broadcast_frame
    service.broadcast_interruption = broadcast_interruption

    await service._create_response()
    service._handle_evt_response_created(make_response_created("resp_a", service.sent[-1]["event_id"]))

    await service._handle_evt_speech_started(SimpleNamespace())

    assert [event["type"] for event in service.sent] == ["response.create"]
    assert service._boson_active_response_ids == {"resp_a"}
    assert service._boson_cancelled_response_ids == set()
    assert service._boson_response_start_frame_active is True
    assert not any(isinstance(frame, LLMFullResponseEndFrame) for frame in service.pushed)
    assert broadcasted == [UserStartedSpeakingFrame, InterruptionFrame]

    await service._handle_evt_text_delta(SimpleNamespace(response_id="resp_a", delta="still valid"))

    assert any(isinstance(frame, LLMTextFrame) and frame.text == "still valid" for frame in service.pushed)


@pytest.mark.asyncio
async def test_response_done_without_active_response_does_not_push_unpaired_end_frame():
    service = CapturingBosonRealtimeLLMService()

    await service._handle_evt_response_done(make_response_done("resp_empty"))

    assert not any(isinstance(frame, LLMFullResponseEndFrame) for frame in service.pushed)


@pytest.mark.asyncio
async def test_cancel_without_done_does_not_leak_active_response_state():
    service = CapturingBosonRealtimeLLMService()
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False
    service._context = LLMContext([])

    await service._create_response()
    await service._cancel_active_response()
    await service._create_response()
    service._handle_evt_response_created(make_response_created("resp_b", service.sent[-1]["event_id"]))
    await service._handle_evt_response_done(make_response_done("resp_b"))

    await service._handle_messages_append(
        LLMMessagesAppendFrame(messages=[{"role": "user", "content": "next"}], run_llm=True)
    )

    assert [event["type"] for event in service.sent[-2:]] == [
        "conversation.item.create",
        "response.create",
    ]


@pytest.mark.asyncio
async def test_messages_append_waits_for_initial_context_before_creating_response():
    service = CapturingBosonRealtimeLLMService()
    service._api_session_ready = True
    tool = FunctionSchema(
        name="get_weather",
        description="Get weather.",
        properties={"location": {"type": "string"}},
        required=["location"],
    )
    context = LLMContext([], [tool])

    await service._handle_messages_append(
        LLMMessagesAppendFrame(messages=[{"role": "user", "content": "weather"}], run_llm=True)
    )

    assert service._context is None
    assert service.sent == []

    await service._handle_context(context, FrameDirection.DOWNSTREAM)

    assert service._context is context
    assert service._context.get_messages()[-1] == {"role": "user", "content": "weather"}
    assert [event["type"] for event in service.sent[-3:]] == [
        "conversation.item.create",
        "session.update",
        "response.create",
    ]
    assert service.sent[-2]["session"]["tools"][0]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_pending_messages_append_without_run_llm_syncs_context_without_creating_response():
    service = CapturingBosonRealtimeLLMService()
    service._api_session_ready = True

    await service._handle_messages_append(
        LLMMessagesAppendFrame(messages=[{"role": "user", "content": "prefill only"}], run_llm=False)
    )
    await service._handle_context(LLMContext([]), FrameDirection.DOWNSTREAM)

    assert service._context.get_messages()[-1] == {"role": "user", "content": "prefill only"}
    assert [event["type"] for event in service.sent] == [
        "conversation.item.create",
        "session.update",
    ]
    assert service.sent[0]["item"]["content"][0]["text"] == "prefill only"

    await service._handle_messages_append(
        LLMMessagesAppendFrame(messages=[{"role": "user", "content": "now run"}], run_llm=True)
    )

    assert [event["type"] for event in service.sent[-2:]] == [
        "conversation.item.create",
        "response.create",
    ]
    assert service.sent[-2]["item"]["content"][0]["text"] == "now run"


@pytest.mark.asyncio
async def test_initial_context_with_direct_function_advertises_tool():
    service = CapturingBosonRealtimeLLMService()
    service._api_session_ready = True

    async def get_weather(params, location: str = "San Francisco") -> None:
        """Get weather.

        Args:
            location: City or place name.
        """

    await service._handle_context(LLMContext([], [get_weather]), FrameDirection.DOWNSTREAM)

    assert service.sent[-2]["type"] == "session.update"
    assert service.sent[-2]["session"]["tools"][0]["name"] == "get_weather"
    assert service.sent[-1]["type"] == "response.create"


@pytest.mark.asyncio
async def test_messages_append_does_not_send_unsupported_skip_tts_override():
    service = CapturingBosonRealtimeLLMService()
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False
    service._context = LLMContext([])

    await service.process_frame(LLMConfigureOutputFrame(skip_tts=True), FrameDirection.DOWNSTREAM)
    await service._handle_messages_append(
        LLMMessagesAppendFrame(messages=[{"role": "user", "content": "text only"}], run_llm=True)
    )

    assert service.sent[-1]["type"] == "response.create"
    assert "modalities" not in service.sent[-1]["response"]
    assert "output_modalities" not in service.sent[-1]["response"]


@pytest.mark.asyncio
async def test_completed_function_call_output_creates_follow_up_response():
    service = CapturingBosonRealtimeLLMService()
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False
    service._context = LLMContext(
        [
            {"role": "user", "content": "weather"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
                ],
            },
        ]
    )

    await service._handle_context(
        LLMContext(
            [
                *service._context.get_messages(),
                {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 20 degrees."},
            ]
        ),
        FrameDirection.UPSTREAM,
    )

    assert [event["type"] for event in service.sent[-2:]] == [
        "conversation.item.create",
        "response.create",
    ]
    assert service.sent[-2]["item"]["type"] == "function_call_output"
    assert service.sent[-2]["item"]["call_id"] == "call_1"
    assert service.sent[-2]["item"]["output"] == "Sunny, 20 degrees."


@pytest.mark.asyncio
@pytest.mark.skipif(
    not HANDLE_CONTEXT_TAKES_DIRECTION,
    reason="before pipecat-ai 1.13, Pipecat replies to any context with a new result",
)
async def test_downstream_context_sends_function_call_output_without_creating_response():
    # Pipecat asks for a reply by sending the context upstream; a downstream context only syncs it.
    service = CapturingBosonRealtimeLLMService()
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False
    service._context = LLMContext(
        [
            {"role": "user", "content": "weather"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
                ],
            },
        ]
    )

    await service._handle_context(
        LLMContext(
            [
                *service._context.get_messages(),
                {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 20 degrees."},
            ]
        ),
        FrameDirection.DOWNSTREAM,
    )

    assert [event["type"] for event in service.sent] == ["conversation.item.create"]
    assert service.sent[0]["item"]["type"] == "function_call_output"
    assert service.sent[0]["item"]["call_id"] == "call_1"


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_before_created", [False, True])
async def test_started_response_takes_in_function_call_output(cancel_before_created):
    # Any response that starts, even one already cancelled, sees the tool result, so it is no longer awaiting a reply.
    service = CapturingBosonRealtimeLLMService()
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False
    service._context = LLMContext(
        [
            {"role": "user", "content": "weather"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
                ],
            },
        ]
    )
    context = LLMContext(
        [
            *service._context.get_messages(),
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 20 degrees."},
        ]
    )

    await service._handle_context(context, FrameDirection.UPSTREAM)
    response_create = service.sent[-1]
    assert response_create["type"] == "response.create"
    if cancel_before_created:
        await service._cancel_active_response()
    service._handle_evt_response_created(make_response_created("resp_1", response_create["event_id"]))
    await service._handle_evt_response_done(
        make_response_done("resp_1", status="cancelled" if cancel_before_created else "completed")
    )
    sent_count = len(service.sent)

    # The assistant aggregator pushes the context upstream again, with nothing new in it.
    await service._handle_context(LLMContext(list(context.get_messages())), FrameDirection.UPSTREAM)

    assert service.sent[sent_count:] == []


@pytest.mark.asyncio
async def test_function_call_output_waits_until_session_created_to_create_response():
    service = CapturingBosonRealtimeLLMService()
    service._api_session_ready = False
    service._llm_needs_conversation_setup = False
    service._context = LLMContext(
        [
            {"role": "user", "content": "weather"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
                ],
            },
        ]
    )

    await service._handle_context(
        LLMContext(
            [
                *service._context.get_messages(),
                {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 20 degrees."},
            ]
        ),
        FrameDirection.UPSTREAM,
    )

    assert service.sent[-1]["type"] == "conversation.item.create"
    assert service.sent[-1]["item"]["type"] == "function_call_output"
    assert service._run_llm_when_api_session_ready is True
    assert not any(event["type"] == "response.create" for event in service.sent)

    await service._handle_evt_session_created(SimpleNamespace())

    assert service.sent[-1]["type"] == "response.create"


@pytest.mark.asyncio
async def test_pending_messages_append_does_not_send_unsupported_skip_tts_override():
    service = CapturingBosonRealtimeLLMService()
    service._api_session_ready = True

    await service.process_frame(LLMConfigureOutputFrame(skip_tts=True), FrameDirection.DOWNSTREAM)
    await service._handle_messages_append(
        LLMMessagesAppendFrame(messages=[{"role": "user", "content": "text only"}], run_llm=True)
    )
    await service.process_frame(LLMConfigureOutputFrame(skip_tts=False), FrameDirection.DOWNSTREAM)
    await service._handle_context(LLMContext([]), FrameDirection.DOWNSTREAM)

    assert service.sent[-1]["type"] == "response.create"
    assert "modalities" not in service.sent[-1]["response"]
    assert "output_modalities" not in service.sent[-1]["response"]


@pytest.mark.asyncio
async def test_messages_append_frame_is_consumed_by_realtime_service():
    service = CapturingBosonRealtimeLLMService()
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False
    service._context = LLMContext([])

    frame = LLMMessagesAppendFrame(messages=[{"role": "user", "content": "hello"}], run_llm=True)
    await service.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert frame not in service.pushed
    assert [event["type"] for event in service.sent[-2:]] == [
        "conversation.item.create",
        "response.create",
    ]


@pytest.mark.asyncio
async def test_send_client_event_treats_normal_websocket_close_as_disconnect():
    service = CapturingErrorBosonRealtimeLLMService()
    service._websocket = ClosingOKWebSocket()
    service._api_session_ready = True

    await service.send_client_event({"type": "input_audio_buffer.commit"})

    assert service.errors == []
    assert service._websocket is None
    assert service._api_session_ready is False
    assert service.metrics_stopped is True


@pytest.mark.asyncio
async def test_send_client_event_terminal_session_close_suppresses_error():
    service = CapturingErrorBosonRealtimeLLMService()
    service._websocket = ClosingErrorWebSocket()
    service._api_session_ready = True
    service._boson_terminal_session_event_type = "session.idle_timeout"

    await service.send_client_event({"type": "input_audio_buffer.commit"})

    assert service.errors == []
    assert service._websocket is None
    assert service._api_session_ready is False
    assert service.metrics_stopped is True


@pytest.mark.asyncio
async def test_mark_websocket_disconnected_closes_existing_websocket():
    service = CapturingErrorBosonRealtimeLLMService()
    websocket = FakeWebSocket()
    service._websocket = websocket
    service._api_session_ready = True

    await service._mark_websocket_disconnected()

    assert websocket.closed is True
    assert service._websocket is None


@pytest.mark.asyncio
async def test_receive_task_end_marks_websocket_disconnected():
    service = CapturingErrorBosonRealtimeLLMService()
    service._websocket = ExhaustedWebSocket()
    service._api_session_ready = True

    await service._receive_task_handler()

    assert service.errors == []
    assert service._websocket is None
    assert service._api_session_ready is False
    assert service.metrics_stopped is True


@pytest.mark.asyncio
async def test_receive_task_disconnect_closes_active_response_frames():
    service = CapturingBosonRealtimeLLMService()
    service._websocket = ExhaustedWebSocket()
    service._api_session_ready = True
    service._boson_response_start_frame_active = True
    service._current_assistant_response = SimpleNamespace(id="assistant_item")

    await service._receive_task_handler()

    assert any(isinstance(frame, LLMFullResponseEndFrame) for frame in service.pushed)
    assert service._current_assistant_response is None
    assert service._current_audio_response is None
    assert service._boson_response_start_frame_active is False


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_event_type", ["session.idle_timeout", "session.max_duration_reached"])
async def test_receive_task_terminal_session_events_suppress_unexpected_close_error(terminal_event_type: str):
    service = CapturingErrorBosonRealtimeLLMService()
    service._websocket = TerminalThenClosedWebSocket(terminal_event_type)
    service._api_session_ready = True

    await service._receive_task_handler()

    assert service.errors == []
    assert service._boson_terminal_session_event_type == terminal_event_type
    assert service._websocket is None
    assert service._api_session_ready is False
    assert service.metrics_stopped is True


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_event_type", ["session.idle_timeout", "session.max_duration_reached"])
async def test_terminal_session_event_notifies_application_handler(terminal_event_type: str):
    service = CapturingBosonRealtimeLLMService()
    calls = []

    async def capture(event_name, *args):
        calls.append((event_name, args))

    service._call_event_handler = capture

    should_continue = await service._dispatch_boson_server_event({"type": terminal_event_type})

    assert should_continue is True
    assert service._boson_terminal_session_event_type == terminal_event_type
    assert calls[0][0] == "on_session_terminated"
    assert calls[0][1][0] == terminal_event_type


@pytest.mark.asyncio
async def test_should_end_call_event_notifies_application_handler():
    service = CapturingBosonRealtimeLLMService()
    calls = []

    async def capture(event_name, *args):
        calls.append((event_name, args))

    service._call_event_handler = capture

    should_continue = await service._dispatch_boson_server_event({"type": "should_end_call", "response_id": "resp_1"})

    assert should_continue is True
    assert calls[0][0] == "on_should_end_call"
    assert calls[0][1][0].response_id == "resp_1"
    assert service._boson_terminal_session_event_type is None


@pytest.mark.asyncio
async def test_session_created_event_notifies_application_handler():
    service = CapturingBosonRealtimeLLMService()
    received_session_ids = []

    @service.event_handler("on_session_created")
    async def capture(service, event):
        received_session_ids.append(event.session.id)

    should_continue = await service._dispatch_boson_server_event(
        {
            "type": "session.created",
            "session": {"id": "24198f53"},
        }
    )

    assert should_continue is True
    assert received_session_ids == ["24198f53"]


@pytest.mark.asyncio
async def test_reply_transcript_is_reported_as_spoken_text():
    service = CapturingBosonRealtimeLLMService()

    await service._push_output_transcript_text_frames("The capital is Rome.")

    llm_text, announced, spoken = service.pushed
    # bot-llm-text for RTVI, kept out of the context (the TTS text goes there).
    assert type(llm_text) is LLMTextFrame
    assert llm_text.append_to_context is False
    # RTVI 2 "new": opens the assistant message in the playground.
    assert type(announced) is AggregatedTextFrame
    assert announced.will_be_spoken is True
    assert announced.append_to_context is False
    # RTVI 2 "completed": the text the caller heard, added to the context.
    assert type(spoken) is TTSTextFrame
    assert spoken.will_be_spoken is True
    assert spoken.append_to_context is not False
    assert announced.text == spoken.text == "The capital is Rome."


def session_with_turn_detection(turn_detection, **input_fields):
    return events.SessionProperties(
        audio=events.AudioConfiguration(input=events.AudioInput(turn_detection=turn_detection, **input_fields))
    )


def test_merge_model_overlays_only_the_fields_an_update_set():
    current = events.SessionProperties(
        tool_choice="none",
        audio=events.AudioConfiguration(
            input=events.AudioInput(
                transcription=events.InputAudioTranscription(model="asr-v1"),
                turn_detection=events.TurnDetection(threshold=0.6, silence_duration_ms=900),
            )
        ),
    )

    merged = realtime_llm._merge_model(current, session_with_turn_detection(events.TurnDetection(threshold=0.7)))

    assert merged.tool_choice == "none"
    assert merged.audio.input.transcription.model == "asr-v1"
    assert merged.audio.input.turn_detection.threshold == 0.7
    assert merged.audio.input.turn_detection.silence_duration_ms == 900
    # The stored copy is left as it was.
    assert current.audio.input.turn_detection.threshold == 0.6


@pytest.mark.parametrize("replacement", [events.SemanticTurnDetection(), False])
def test_merge_model_replaces_a_value_of_a_different_type(replacement):
    current = session_with_turn_detection(events.TurnDetection(threshold=0.6, silence_duration_ms=900))

    merged = realtime_llm._merge_model(current, session_with_turn_detection(replacement))

    assert merged.audio.input.turn_detection == replacement


def test_merge_model_clears_a_field_set_to_none():
    current = session_with_turn_detection(
        events.TurnDetection(threshold=0.6), transcription=events.InputAudioTranscription(model="asr-v1")
    )
    update = events.SessionProperties(audio=events.AudioConfiguration(input=events.AudioInput(transcription=None)))

    merged = realtime_llm._merge_model(current, update)

    assert merged.audio.input.transcription is None
    assert merged.audio.input.turn_detection.threshold == 0.6


def test_merge_model_replaces_tools_and_keeps_the_objects_as_given():
    old_tools = ToolsSchema(standard_tools=[FunctionSchema("a", "a", {}, [])])
    new_tools = ToolsSchema(standard_tools=[FunctionSchema("b", "b", {}, [])])

    merged = realtime_llm._merge_model(
        events.SessionProperties(tools=old_tools), events.SessionProperties(tools=new_tools)
    )

    assert merged.tools is new_tools


@pytest.mark.asyncio
async def test_unrelated_settings_update_keeps_pipecats_stored_session_properties():
    tools = ToolsSchema(standard_tools=[FunctionSchema("weather", "weather", {}, [])])
    service = CapturingBosonRealtimeLLMService(
        tools=tools, output_modalities=["text"], input_audio_transcription={"model": "asr-v1"}
    )

    await service._update_settings(service.Settings(session_properties=events.SessionProperties(tool_choice="none")))

    stored = service._settings.session_properties
    assert stored.tool_choice == "none"
    assert stored.tools is tools
    assert stored.output_modalities == ["text"]
    assert stored.audio.input.transcription.model == "asr-v1"
    # Pipecat's own readers see the stored copy, not just Boson's.
    assert service._is_modality_enabled("audio") is False


@pytest.mark.asyncio
async def test_settings_update_leaves_the_callers_delta_unchanged():
    service = CapturingBosonRealtimeLLMService()
    session_properties = events.SessionProperties(tool_choice="none")
    delta = service.Settings(session_properties=session_properties)

    await service._update_settings(delta)

    # Pipecat forwards the frame carrying this delta to other services.
    assert delta.session_properties is session_properties
    assert session_properties.model_fields_set == {"tool_choice"}


@pytest.mark.asyncio
async def test_max_tokens_update_wins_over_an_earlier_max_output_tokens_update():
    service = CapturingBosonRealtimeLLMService()

    await service._update_settings(
        service.Settings(session_properties=events.SessionProperties(max_output_tokens=4096))
    )
    await service._update_settings(service.Settings(max_tokens=100))

    assert service.sent[-1]["session"]["max_output_tokens"] == 100


def make_turn_detection_delta(service, turn_detection):
    return service.Settings(
        session_properties=events.SessionProperties(
            audio=events.AudioConfiguration(input=events.AudioInput(turn_detection=turn_detection))
        )
    )


def sent_turn_detection(service):
    return service.sent[-1]["session"]["audio"]["input"]["turn_detection"]


@pytest.mark.asyncio
async def test_default_turn_detection_survives_an_unrelated_settings_update():
    service = CapturingBosonRealtimeLLMService()
    await service._send_session_update()
    before = sent_turn_detection(service)

    await service._update_settings(service.Settings(system_instruction="new prompt"))

    # Pipecat's TurnDetection model has no create_response/interrupt_response, so a
    # round trip through it used to drop them from every later session.update.
    assert sent_turn_detection(service) == before
    assert before["create_response"] is True
    assert before["interrupt_response"] is True


@pytest.mark.asyncio
async def test_boson_only_turn_detection_fields_survive_an_unrelated_settings_update():
    service = CapturingBosonRealtimeLLMService(
        turn_detection={"type": "server_vad", "threshold": 0.6, "min_speech_duration": 0.2}
    )

    await service._update_settings(service.Settings(system_instruction="new prompt"))

    assert sent_turn_detection(service) == {
        "type": "server_vad",
        "threshold": 0.6,
        "min_speech_duration": 0.2,
    }


@pytest.mark.asyncio
async def test_turn_detection_update_applies_changes_and_keeps_boson_only_fields():
    service = CapturingBosonRealtimeLLMService(
        turn_detection={"type": "server_vad", "threshold": 0.6, "min_speech_duration": 0.2}
    )

    await service._update_settings(make_turn_detection_delta(service, events.TurnDetection(threshold=0.7)))

    turn_detection = sent_turn_detection(service)
    assert turn_detection["threshold"] == 0.7
    assert turn_detection["min_speech_duration"] == 0.2


@pytest.mark.asyncio
async def test_turn_detection_can_still_be_disabled_at_runtime():
    service = CapturingBosonRealtimeLLMService()

    await service._update_settings(make_turn_detection_delta(service, False))

    assert sent_turn_detection(service) is None
    # The wire payload and Pipecat's own copy have to agree: None disables server VAD
    # on the wire, while the base class reads False to drive local turn handling.
    assert service._is_turn_detection_disabled() is True


@pytest.mark.asyncio
async def test_switching_vad_type_replaces_rather_than_merges():
    service = CapturingBosonRealtimeLLMService(
        turn_detection={"type": "server_vad", "threshold": 0.6, "min_speech_duration": 0.2}
    )

    await service._update_settings(make_turn_detection_delta(service, events.SemanticTurnDetection()))

    # server_vad fields must not leak into a semantic_vad payload
    assert sent_turn_detection(service) == {"type": "semantic_vad"}


@pytest.mark.asyncio
async def test_turn_detection_update_without_a_type_keeps_the_configured_type():
    service = CapturingBosonRealtimeLLMService(
        turn_detection={"type": "server_vad", "threshold": 0.6, "min_speech_duration": 0.2}
    )

    # Pipecat's TurnDetection.type is optional, so the type has to come from the model
    # class rather than the dumped fields, or an update would erase it.
    await service._update_settings(make_turn_detection_delta(service, events.TurnDetection(type=None, threshold=0.7)))

    turn_detection = sent_turn_detection(service)
    assert turn_detection["type"] == "server_vad"
    assert turn_detection["threshold"] == 0.7
    assert turn_detection["min_speech_duration"] == 0.2


def test_constructor_rejects_an_eagerness_boson_cannot_accept():
    with pytest.raises(ValueError, match="eagerness"):
        CapturingBosonRealtimeLLMService(turn_detection={"type": "semantic_vad", "eagerness": "high"})


@pytest.mark.asyncio
async def test_runtime_eagerness_is_dropped_instead_of_killing_the_session():
    service = CapturingBosonRealtimeLLMService(turn_detection={"type": "semantic_vad"})

    await service._update_settings(make_turn_detection_delta(service, events.SemanticTurnDetection(eagerness="high")))

    # Raising mid-call would break a live session, so the value is dropped instead.
    assert sent_turn_detection(service) == {"type": "semantic_vad"}


@pytest.mark.asyncio
async def test_manual_turn_mode_survives_an_unrelated_settings_update():
    service = CapturingBosonRealtimeLLMService(turn_detection=False)

    assert service._is_turn_detection_disabled() is True

    await service._update_settings(service.Settings(session_properties=events.SessionProperties(tool_choice="none")))

    # The delta replaced Pipecat's copy of the session properties, which no longer
    # says False, but Boson still has server VAD off — so the application has to keep
    # driving turns or the session stops responding altogether.
    assert sent_turn_detection(service) is None
    assert service._is_turn_detection_disabled() is True


@pytest.mark.asyncio
async def test_server_vad_sessions_report_turn_detection_as_enabled():
    service = CapturingBosonRealtimeLLMService()

    assert service._is_turn_detection_disabled() is False

    await service._update_settings(make_turn_detection_delta(service, False))

    assert service._is_turn_detection_disabled() is True


async def weather_tool(params, location: str = "San Francisco") -> None:
    """Get the weather.

    Args:
        location: City or place name.
    """


@pytest.mark.asyncio
async def test_tools_survive_a_settings_update_that_replaces_session_properties():
    service = CapturingBosonRealtimeLLMService(tools=[weather_tool])

    await service._update_settings(make_turn_detection_delta(service, events.TurnDetection(threshold=0.7)))

    assert [tool["name"] for tool in service.sent[-1]["session"]["tools"]] == ["weather_tool"]


@pytest.mark.asyncio
async def test_service_tools_survive_a_settings_update_so_handlers_are_kept():
    service = CapturingBosonRealtimeLLMService(tools=[weather_tool])

    await service._update_settings(make_turn_detection_delta(service, events.TurnDetection(threshold=0.7)))

    # Pipecat prunes every registered tool handler when this returns nothing.
    assert service._service_tools() is not None


@pytest.mark.asyncio
async def test_tools_can_still_be_cleared_with_a_set_tools_frame():
    service = CapturingBosonRealtimeLLMService(tools=[weather_tool])

    # Pipecat's SessionProperties stores tools=[] as None, so a settings update cannot
    # express "no tools"; LLMSetToolsFrame is the path that can.
    await service.process_frame(LLMSetToolsFrame(tools=[]), FrameDirection.DOWNSTREAM)

    assert service.sent[-1]["session"]["tools"] == []


@pytest.mark.asyncio
async def test_partial_turn_detection_update_keeps_unmentioned_fields():
    service = CapturingBosonRealtimeLLMService(
        turn_detection={
            "type": "server_vad",
            "threshold": 0.6,
            "prefix_padding_ms": 100,
            "silence_duration_ms": 1200,
            "min_speech_duration": 0.2,
        }
    )

    await service._update_settings(make_turn_detection_delta(service, events.TurnDetection(threshold=0.7)))

    # Pipecat's model carries defaults for padding and silence, which must not be
    # mistaken for a caller asking to change them.
    assert sent_turn_detection(service) == {
        "type": "server_vad",
        "threshold": 0.7,
        "prefix_padding_ms": 100,
        "silence_duration_ms": 1200,
        "min_speech_duration": 0.2,
    }


@pytest.mark.asyncio
async def test_disabling_then_re_enabling_restores_the_configured_payload():
    configured = {
        "type": "server_vad",
        "threshold": 0.6,
        "prefix_padding_ms": 100,
        "min_speech_duration": 0.2,
    }
    service = CapturingBosonRealtimeLLMService(turn_detection=dict(configured))

    await service._update_settings(make_turn_detection_delta(service, False))
    assert sent_turn_detection(service) is None

    await service._update_settings(make_turn_detection_delta(service, events.TurnDetection(threshold=0.7)))

    assert sent_turn_detection(service) == {**configured, "threshold": 0.7}


@pytest.mark.asyncio
async def test_a_type_less_server_vad_update_after_a_semantic_switch_does_not_mix_fields():
    service = CapturingBosonRealtimeLLMService(turn_detection={"type": "server_vad", "threshold": 0.6})
    await service._update_settings(make_turn_detection_delta(service, events.SemanticTurnDetection()))
    assert sent_turn_detection(service) == {"type": "semantic_vad"}

    # TurnDetection means server VAD even when its type field is left out.
    await service._update_settings(make_turn_detection_delta(service, events.TurnDetection(type=None, threshold=0.7)))

    assert sent_turn_detection(service) == {"type": "server_vad", "threshold": 0.7}


@pytest.mark.asyncio
async def test_update_to_a_payload_without_a_type_keeps_boson_only_fields():
    # Boson defaults a missing type itself, so a caller may never set one; that must
    # not read as a VAD type change on the first update.
    service = CapturingBosonRealtimeLLMService(turn_detection={"threshold": 0.6, "min_speech_duration": 0.2})

    await service._update_settings(make_turn_detection_delta(service, events.TurnDetection(threshold=0.7)))

    assert sent_turn_detection(service) == {
        "type": "server_vad",
        "threshold": 0.7,
        "min_speech_duration": 0.2,
    }


@pytest.mark.asyncio
async def test_re_enabling_with_true_uses_the_defaults_when_nothing_was_configured():
    service = CapturingBosonRealtimeLLMService(turn_detection=False)

    await service._update_settings(make_turn_detection_delta(service, True))

    assert sent_turn_detection(service) == DEFAULT_TURN_DETECTION
    assert service._is_turn_detection_disabled() is False


@pytest.mark.asyncio
async def test_the_remembered_turn_detection_payload_is_not_aliased():
    service = CapturingBosonRealtimeLLMService(turn_detection={"type": "server_vad", "threshold": 0.6})

    await service._update_settings(make_turn_detection_delta(service, events.TurnDetection(threshold=0.7)))

    assert service._boson_turn_detection is not service._boson_last_enabled_turn_detection


@pytest.mark.asyncio
async def test_turn_detection_none_keeps_detection_rather_than_disabling_it():
    # In Pipecat, None means "use the default detection"; only False opts out.
    service = CapturingBosonRealtimeLLMService(turn_detection={"type": "server_vad", "threshold": 0.6})

    await service._update_settings(make_turn_detection_delta(service, None))

    assert sent_turn_detection(service) == {"type": "server_vad", "threshold": 0.6}
    assert service._is_turn_detection_disabled() is False


@pytest.mark.asyncio
async def test_turn_detection_false_still_disables_detection():
    service = CapturingBosonRealtimeLLMService(turn_detection={"type": "server_vad", "threshold": 0.6})

    await service._update_settings(make_turn_detection_delta(service, False))

    assert sent_turn_detection(service) is None
    assert service._is_turn_detection_disabled() is True


@pytest.mark.asyncio
async def test_switching_to_semantic_vad_drops_untyped_server_vad_fields():
    # Boson reads a payload without a type as server VAD.
    service = CapturingBosonRealtimeLLMService(turn_detection={"threshold": 0.6, "silence_duration_ms": 900})

    await service._update_settings(make_turn_detection_delta(service, events.SemanticTurnDetection()))

    assert sent_turn_detection(service) == {"type": "semantic_vad"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "turn_detection", [events.TurnDetection(threshold=0.7), False, events.SemanticTurnDetection()]
)
async def test_turn_detection_is_fixed_once_the_session_has_started(monkeypatch, turn_detection):
    # Boson rejects any turn-detection change after the first session.update, which goes out on connect.
    warnings = []
    monkeypatch.setattr(realtime_llm.logger, "warning", warnings.append)
    service = CapturingBosonRealtimeLLMService(turn_detection={"type": "server_vad", "threshold": 0.6})
    service._websocket = object()

    await service._update_settings(make_turn_detection_delta(service, turn_detection))

    assert sent_turn_detection(service) == {"type": "server_vad", "threshold": 0.6}
    assert service._is_turn_detection_disabled() is False
    assert len(warnings) == 1


@pytest.mark.asyncio
async def test_turn_detection_update_that_changes_nothing_is_quiet_mid_session(monkeypatch):
    warnings = []
    monkeypatch.setattr(realtime_llm.logger, "warning", warnings.append)
    service = CapturingBosonRealtimeLLMService(turn_detection={"type": "server_vad", "threshold": 0.6})
    service._websocket = object()

    await service._update_settings(make_turn_detection_delta(service, None))

    assert sent_turn_detection(service) == {"type": "server_vad", "threshold": 0.6}
    assert warnings == []


@pytest.mark.asyncio
async def test_turn_detection_can_change_again_after_a_disconnect():
    # Each connection starts a new Boson session, which accepts any turn detection.
    service = CapturingBosonRealtimeLLMService(turn_detection={"type": "server_vad", "threshold": 0.6})
    service._websocket = object()
    await service._update_settings(make_turn_detection_delta(service, events.TurnDetection(threshold=0.7)))
    assert sent_turn_detection(service) == {"type": "server_vad", "threshold": 0.6}

    await service._mark_websocket_disconnected()
    await service._update_settings(make_turn_detection_delta(service, events.TurnDetection(threshold=0.7)))

    assert service._boson_turn_detection == {"type": "server_vad", "threshold": 0.7}


@pytest.mark.asyncio
async def test_ignored_turn_detection_update_leaves_the_callers_delta_unchanged():
    # With Pipecat's stored audio cleared, the merge replaces it with the update's own models.
    service = CapturingBosonRealtimeLLMService(turn_detection={"type": "server_vad", "threshold": 0.6})
    await service._update_settings(service.Settings(session_properties=events.SessionProperties(audio=None)))
    service._websocket = object()
    delta = make_turn_detection_delta(service, events.TurnDetection(threshold=0.7))

    await service._update_settings(delta)

    assert delta.session_properties.audio.input.turn_detection.threshold == 0.7


async def other_tool(params, query: str = "") -> None:
    """Search for something.

    Args:
        query: What to search for.
    """


def sent_tool_names(service):
    return [tool["name"] for tool in service.sent[-1]["session"].get("tools") or []]


def registered_tool_names(service):
    return sorted(name for name in service._functions if name)


@pytest.mark.asyncio
async def test_tools_set_by_a_settings_update_survive_a_later_unrelated_update():
    service = CapturingBosonRealtimeLLMService(tools=[weather_tool])

    await service._update_settings(service.Settings(session_properties=events.SessionProperties(tools=[other_tool])))
    assert sent_tool_names(service) == ["other_tool"]

    await service._update_settings(service.Settings(session_properties=events.SessionProperties(tool_choice="none")))
    assert service.sent[-1]["session"]["tool_choice"] == "none"
    assert sent_tool_names(service) == ["other_tool"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("frame_tools", "expected"), [([other_tool], ["other_tool"]), ([], [])])
async def test_handler_registry_follows_tools_set_by_frame(frame_tools, expected):
    service = CapturingBosonRealtimeLLMService(tools=[weather_tool])
    await service.process_frame(LLMSetToolsFrame(tools=frame_tools), FrameDirection.DOWNSTREAM)

    # What Pipecat does on a context frame that carries no tools.
    service._sync_registered_tool_handlers(None)

    assert sent_tool_names(service) == expected
    assert registered_tool_names(service) == expected


@pytest.mark.asyncio
async def test_wire_and_handler_registry_agree_across_a_sequence_of_updates():
    # Whatever Boson is offered must have a handler, through every way tools can change.
    service = CapturingBosonRealtimeLLMService(tools=[weather_tool])

    async def agree():
        await service._send_session_update()
        service._sync_registered_tool_handlers(None)
        return sent_tool_names(service) == registered_tool_names(service)

    assert await agree()

    await service._update_settings(service.Settings(session_properties=events.SessionProperties(tools=[other_tool])))
    assert await agree()

    await service._update_settings(service.Settings(system_instruction="unrelated"))
    assert await agree()

    await service.process_frame(LLMSetToolsFrame(tools=[weather_tool]), FrameDirection.DOWNSTREAM)
    assert await agree()

    await service._update_settings(service.Settings(session_properties=events.SessionProperties(tool_choice="none")))
    assert await agree()


@pytest.mark.asyncio
async def test_server_executed_tool_calls_are_not_run_or_answered():
    # A server tool (web_search) arrives as a function_call with executor "server".
    # The server runs it and adds the output itself; the client must do nothing.
    service = CapturingBosonRealtimeLLMService()
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False
    service._context = LLMContext([])
    dispatched = []

    async def run_function_calls(function_calls):
        dispatched.extend(call.function_name for call in function_calls)

    service.run_function_calls = run_function_calls
    item = {
        "id": "item_1",
        "type": "function_call",
        "name": "web_search",
        "call_id": "call_server",
        "arguments": "",
        "executor": "server",
    }
    sent = len(service.sent)

    await service._dispatch_boson_server_event(
        {"type": "response.output_item.added", "response_id": "r", "item": item}
    )
    await service._dispatch_boson_server_event({"type": "conversation.item.added", "item": item})
    await service._dispatch_boson_server_event(
        {"type": "response.function_call_arguments.done", "call_id": "call_server", "arguments": '{"queries": ["x"]}'}
    )

    assert "call_server" not in service._pending_function_calls
    assert dispatched == []
    assert not [e for e in service.sent[sent:] if e["type"] in ("conversation.item.create", "response.create")]


@pytest.mark.asyncio
async def test_client_executed_tool_calls_are_still_tracked():
    service = CapturingBosonRealtimeLLMService()
    item = {
        "id": "item_2",
        "type": "function_call",
        "name": "get_time",
        "call_id": "call_client",
        "executor": "client",
    }

    await service._dispatch_boson_server_event(
        {"type": "response.output_item.added", "response_id": "r", "item": item}
    )

    assert "call_client" in service._pending_function_calls


def _dispatch_recorder(service):
    dispatched = []

    async def run_function_calls(function_calls):
        dispatched.extend(call.function_name for call in function_calls)

    service.run_function_calls = run_function_calls
    return dispatched


@pytest.mark.asyncio
async def test_client_executed_tool_calls_are_dispatched():
    service = CapturingBosonRealtimeLLMService()
    service._context = LLMContext([])
    dispatched = _dispatch_recorder(service)
    item = {
        "id": "item_2",
        "type": "function_call",
        "name": "get_time",
        "call_id": "call_client",
        "executor": "client",
    }

    await service._dispatch_boson_server_event({"type": "conversation.item.added", "item": item})
    await service._dispatch_boson_server_event(
        {"type": "response.function_call_arguments.done", "call_id": "call_client", "arguments": "{}"}
    )

    assert dispatched == ["get_time"]


@pytest.mark.asyncio
@pytest.mark.parametrize("marked_on_output_item", [True, False])
@pytest.mark.parametrize("output_item_first", [True, False])
async def test_a_call_marked_server_by_either_item_event_is_not_dispatched(marked_on_output_item, output_item_first):
    # The executor may sit on only one of the two item events, in either order.
    service = CapturingBosonRealtimeLLMService()
    service._context = LLMContext([])
    dispatched = _dispatch_recorder(service)
    marked = {"id": "item_1", "type": "function_call", "name": "web_search", "call_id": "call_s", "executor": "server"}
    unmarked = {key: value for key, value in marked.items() if key != "executor"}
    events = [
        {
            "type": "response.output_item.added",
            "response_id": "r",
            "item": marked if marked_on_output_item else unmarked,
        },
        {"type": "conversation.item.added", "item": unmarked if marked_on_output_item else marked},
    ]
    for event in events if output_item_first else events[::-1]:
        await service._dispatch_boson_server_event(event)
    await service._dispatch_boson_server_event(
        {"type": "response.function_call_arguments.done", "call_id": "call_s", "arguments": "{}"}
    )

    assert "call_s" not in service._pending_function_calls
    assert dispatched == []


@pytest.mark.asyncio
async def test_server_call_ids_are_forgotten_on_a_new_connection(monkeypatch):
    service = CapturingBosonRealtimeLLMService()
    await service._dispatch_boson_server_event(
        {
            "type": "conversation.item.added",
            "item": {"id": "i", "type": "function_call", "name": "web_search", "call_id": "c", "executor": "server"},
        }
    )
    assert "c" in service._boson_server_call_ids

    async def fake_connect(**kwargs):
        return object()

    async def no_session_update():
        return None

    monkeypatch.setattr("pipecat_boson.realtime.llm.websocket_connect", fake_connect)
    monkeypatch.setattr(service, "create_task", lambda coroutine: coroutine.close())
    monkeypatch.setattr(service, "_send_session_update", no_session_update)
    service._websocket = None
    await service._connect()

    assert "c" not in service._boson_server_call_ids


@pytest.mark.asyncio
async def test_speech_stopped_opens_the_callers_turn_before_the_reply():
    # The server's transcript of a turn can arrive after the reply starts. An
    # empty interim transcript at speech_stopped lets clients place the turn first.
    service = CapturingBosonRealtimeLLMService(input_audio_transcription_model="higgs-stt-3.1")

    await service._handle_evt_speech_stopped(SimpleNamespace(type="input_audio_buffer.speech_stopped"))

    interim = [f for f in service.pushed if isinstance(f, InterimTranscriptionFrame)]
    assert len(interim) == 1 and interim[0].text == ""


@pytest.mark.asyncio
async def test_speech_stopped_sends_no_interim_transcript_without_transcription():
    service = CapturingBosonRealtimeLLMService()

    await service._handle_evt_speech_stopped(SimpleNamespace(type="input_audio_buffer.speech_stopped"))

    assert not [f for f in service.pushed if isinstance(f, InterimTranscriptionFrame)]


@pytest.mark.asyncio
async def test_reply_transcript_frames_are_labelled_as_a_sentence_without_deprecation_warnings(monkeypatch):
    # pipecat-ai 1.13 renamed the label; the old names still work there, but warn. Pipecat
    # forces its deprecation warnings to "always", so record them rather than raising, and
    # it warns once per call site per process, so clear that record (1.13 only).
    monkeypatch.setattr(pipecat_deprecation, "_warned_sites", set(), raising=False)
    service = CapturingBosonRealtimeLLMService()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await service._push_output_transcript_text_frames("The capital is Rome.")

    assert [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)] == []
    _, announced, spoken = service.pushed
    label_field = next(iter(SENTENCE_TEXT))
    assert getattr(announced, label_field).value == getattr(spoken, label_field).value == "sentence"


def test_forget_answered_call_removes_only_that_call():
    service = SimpleNamespace(_open_function_calls={"call_1", "call_2"})

    forget_answered_call(service, "call_1")

    assert service._open_function_calls == {"call_2"}


def test_forget_answered_call_is_harmless_before_pipecat_tracks_open_calls():
    # Before pipecat-ai 1.13 there is no _open_function_calls; this must not raise.
    forget_answered_call(SimpleNamespace(), "call_1")


@pytest.mark.asyncio
@pytest.mark.skipif(
    not hasattr(BosonRealtimeLLMService, "_handle_function_call_cancel"),
    reason="pipecat-ai 1.13 added the open-call tracking this relies on",
)
async def test_cancel_after_a_tool_result_sends_nothing_more():
    service = CapturingBosonRealtimeLLMService()
    service._open_function_calls.add("call_1")
    await service._send_tool_result("call_1", "Sunny, 20 degrees.")
    sent_count = len(service.sent)

    await service._handle_function_call_cancel(
        FunctionCallCancelFrame(function_name="get_weather", tool_call_id="call_1")
    )

    assert len(service.sent) == sent_count


# Real-pipeline tests. Pipecat 1.13 changed how context frames and tool results reach
# the service, so these run the real service between Pipecat's real aggregators, on
# every supported version. Only the websocket is replaced; Boson's side is played back
# as server events through the service's own dispatcher.


async def _close_nothing():
    return None


class PipelineBosonService(BosonRealtimeLLMService):
    """The real service for a real pipeline, with the websocket replaced by a capture."""

    def __init__(self, **kwargs):
        super().__init__(url="ws://localhost:1234/v1/realtime/", api_key="test-key", model="model-1", **kwargs)
        self.sent: list[dict] = []

    async def _connect(self):
        if self._websocket:
            return
        self._websocket = SimpleNamespace(close=_close_nothing)
        await self._send_session_update()
        await self._dispatch_boson_server_event({"type": "session.created", "session": {"id": "session_1"}})

    async def _disconnect(self):
        self._websocket = None

    async def _ws_send(self, realtime_message):
        self.sent.append(realtime_message)


def sent_types(service):
    return [
        f"{message['type']}:{message['item']['type']}"
        if message["type"] == "conversation.item.create"
        else message["type"]
        for message in service.sent
    ]


async def wait_until(condition, timeout=5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        assert asyncio.get_running_loop().time() < deadline, "timed out waiting for the pipeline"
        await asyncio.sleep(0.01)


@contextlib.asynccontextmanager
async def running_pipeline(service, context):
    pair = LLMContextAggregatorPair(context)
    task = PipelineTask(Pipeline([pair.user(), service, pair.assistant()]))
    runner = asyncio.create_task(PipelineRunner(handle_sigint=False).run(task))
    try:
        await wait_until(lambda: service._api_session_ready)
        await task.queue_frame(LLMContextFrame(context))
        await wait_until(lambda: "response.create" in sent_types(service))
        yield task
    finally:
        await task.queue_frame(EndFrame())
        await asyncio.wait_for(runner, 10)


async def finish_last_response(service, response_id):
    """Boson's events for the last response.create: created, then done."""

    create = [message for message in service.sent if message["type"] == "response.create"][-1]
    metadata = {"client_event_id": create["event_id"]}
    await service._dispatch_boson_server_event(
        {"type": "response.created", "response": {"id": response_id, "metadata": metadata}}
    )
    await service._dispatch_boson_server_event(
        {"type": "response.done", "response": {"id": response_id, "status": "completed", "output": [], "usage": None}}
    )


async def call_tool(service, name, arguments):
    """Boson's events for the model calling a tool: the call item, then its arguments."""

    await service._dispatch_boson_server_event(
        {
            "type": "conversation.item.added",
            "item": {"id": "item_1", "type": "function_call", "call_id": "call_1", "name": name},
        }
    )
    await service._dispatch_boson_server_event(
        {
            "type": "response.function_call_arguments.done",
            "item_id": "item_1",
            "call_id": "call_1",
            "name": name,
            "arguments": json.dumps(arguments),
        }
    )


@pytest.mark.asyncio
async def test_a_context_frame_in_a_real_pipeline_starts_one_response():
    service = PipelineBosonService()
    context = LLMContext([{"role": "user", "content": "What's the weather in Paris?"}])

    async with running_pipeline(service, context):
        await asyncio.sleep(0.2)

    assert sent_types(service).count("response.create") == 1
    messages = [message["item"] for message in service.sent if message["type"] == "conversation.item.create"]
    assert [item["content"][0]["text"] for item in messages] == ["What's the weather in Paris?"]


@pytest.mark.asyncio
async def test_a_tool_call_in_a_real_pipeline_sends_one_result_and_one_follow_up_response():
    service = PipelineBosonService()

    async def get_weather(params):
        await params.result_callback({"conditions": "sunny"})

    service.register_function("get_weather", get_weather)
    context = LLMContext([{"role": "user", "content": "What's the weather in Paris?"}])

    async with running_pipeline(service, context):
        await finish_last_response(service, "resp_0")
        before = len(service.sent)
        await call_tool(service, "get_weather", {"location": "Paris"})
        await wait_until(lambda: "response.create" in sent_types(service)[before:])
        await asyncio.sleep(0.2)

    after = service.sent[before:]
    assert [message["type"] for message in after] == ["conversation.item.create", "response.create"]
    assert after[0]["item"]["type"] == "function_call_output"
    assert after[0]["item"]["call_id"] == "call_1"
    assert after[0]["item"]["output"] == json.dumps({"conditions": "sunny"})


@pytest.mark.asyncio
async def test_a_tool_progress_update_in_a_real_pipeline_never_reaches_boson():
    # Pipecat 1.13 would send it as a system item, which Boson rejects, ending the session.
    service = PipelineBosonService()

    async def search_flights(params):
        await params.result_callback(
            {"progress": "searching"}, properties=FunctionCallResultProperties(is_final=False)
        )
        await params.result_callback({"flights": 3})

    # Only an async tool (cancel_on_interruption=False) may report progress.
    service.register_function("search_flights", search_flights, cancel_on_interruption=False)
    context = LLMContext([{"role": "user", "content": "Find me a flight."}])

    async with running_pipeline(service, context):
        await finish_last_response(service, "resp_0")
        before = len(service.sent)
        await call_tool(service, "search_flights", {"to": "Rome"})
        await wait_until(lambda: "response.create" in sent_types(service)[before:])
        await asyncio.sleep(0.2)

    after = service.sent[before:]
    items = [message["item"] for message in after if message["type"] == "conversation.item.create"]
    assert [item.get("role") for item in items if item["type"] == "message"] == []
    assert [message["type"] for message in after].count("response.create") == 1
    assert [item["output"] for item in items if item["type"] == "function_call_output"] == [json.dumps({"flights": 3})]
