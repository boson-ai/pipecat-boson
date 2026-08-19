# ruff: noqa: CPY001
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import (
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InterruptionFrame,
    LLMConfigureOutputFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMSetToolsFrame,
    LLMTextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import NOT_GIVEN as LLM_CONTEXT_NOT_GIVEN
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.services.openai.realtime import events
from pipecat.tests.utils import run_test
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

import pipecat_boson.realtime.llm as realtime_llm
from pipecat_boson.realtime import BosonRealtimeLLMService


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
    )

    service = CapturingBosonRealtimeLLMService()
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False
    service._context = LLMContext([context.get_messages()[0]])
    await service._handle_context(upstream_frames[0].context)

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
async def test_audio_transcript_length_event_is_forwarded_as_rtvi_server_message():
    service = CapturingBosonRealtimeLLMService()
    event = {
        "event_id": "event_1",
        "type": "response.output_audio_transcript.length",
        "response_id": "response_1",
        "item_id": "item_1",
        "output_index": 0,
        "content_index": 0,
        "delta": "Hello",
        "length_ms": 320,
    }

    should_continue = await service._dispatch_boson_server_event(event)

    assert should_continue is True
    server_messages = [frame for frame in service.pushed if isinstance(frame, RTVIServerMessageFrame)]
    assert len(server_messages) == 1
    assert server_messages[0].data == event


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
    await service._handle_evt_audio_transcript_length(
        SimpleNamespace(
            type="response.output_audio_transcript.length",
            response_id="resp_cancelled",
            delta="old transcript",
            length_ms=320,
        )
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

    await service._handle_context(context)

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
    await service._handle_context(LLMContext([]))

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

    await service._handle_context(LLMContext([], [get_weather]))

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
        )
    )

    assert [event["type"] for event in service.sent[-2:]] == [
        "conversation.item.create",
        "response.create",
    ]
    assert service.sent[-2]["item"]["type"] == "function_call_output"
    assert service.sent[-2]["item"]["call_id"] == "call_1"
    assert service.sent[-2]["item"]["output"] == "Sunny, 20 degrees."


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
        )
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
    await service._handle_context(LLMContext([]))

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
