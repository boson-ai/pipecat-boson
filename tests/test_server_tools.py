# ruff: noqa: CPY001
"""Server tools: Higgs Realtime runs ``web_search`` itself, so the service only declares it."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import copy
from types import SimpleNamespace

import pytest
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.frames.frames import (
    EndFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TTSAudioRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.realtime import events

from pipecat_boson.realtime import BosonRealtimeLLMService


WEB_SEARCH = {"type": "web_search", "allowed_domains": ["weather.com"], "location": {"country": "US"}}
PCM = base64.b64encode(b"\x01\x00" * 480).decode()


async def get_weather(params: FunctionCallParams, location: str) -> None:
    """Get the current weather for a location.

    Args:
        location: City or place name.
    """
    await params.result_callback({"location": location, "condition": "sunny"})


def readme_tools():
    # The declaration the README shows: a Python function next to the server tool.
    return ToolsSchema(standard_tools=[get_weather], custom_tools={AdapterType.OPENAI: [copy.deepcopy(WEB_SEARCH)]})


async def _close_nothing():
    return None


class PipelineBosonService(BosonRealtimeLLMService):
    """The real service, with the websocket replaced by a list of sent events."""

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


class FrameRecorder(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.frames = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            self.frames.append(frame)
        await self.push_frame(frame, direction)


async def wait_until(condition, timeout=5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        assert asyncio.get_running_loop().time() < deadline, "timed out waiting for the pipeline"
        await asyncio.sleep(0.01)


@contextlib.asynccontextmanager
async def running_pipeline(service, context, recorder):
    pair = LLMContextAggregatorPair(context)
    task = PipelineTask(Pipeline([pair.user(), service, recorder, pair.assistant()]))
    runner = asyncio.create_task(PipelineRunner(handle_sigint=False).run(task))
    try:
        await wait_until(lambda: service._api_session_ready)
        await task.queue_frame(LLMContextFrame(context))
        await wait_until(lambda: any(message["type"] == "response.create" for message in service.sent))
        yield task
    finally:
        await task.queue_frame(EndFrame())
        await asyncio.wait_for(runner, 10)


def spoken_item(response_id, item_id, text):
    """Boson's events for one spoken assistant item: the item, the events opening it, then its audio."""

    item = {"id": item_id, "type": "message", "role": "assistant", "content": []}
    part = {"response_id": response_id, "item_id": item_id, "output_index": 0, "content_index": 0}
    opening = [
        {"type": "response.output_item.added", "response_id": response_id, "output_index": 0, "item": item},
        {"type": "conversation.item.added", "item": item},
        {"type": "response.content_part.added", **part, "part": {"type": "audio", "transcript": ""}},
    ]
    audio = [
        {"type": "response.output_audio_transcript.delta", **part, "delta": text},
        {"type": "response.output_audio.delta", **part, "delta": PCM},
        {"type": "response.output_audio_transcript.length", **part},
        {"type": "response.output_audio_transcript.done", **part, "transcript": text},
        {"type": "response.output_audio.done", **part},
        {"type": "response.content_part.done", **part, "part": {"type": "audio", "transcript": text}},
        {"type": "response.output_item.done", "response_id": response_id, "output_index": 0, "item": item},
    ]
    return item, opening, audio


def web_search_call(response_id):
    """Boson's events for the model calling web_search, which Boson runs itself."""

    call = {
        "id": "item_call",
        "type": "function_call",
        "name": "web_search",
        "call_id": "call_ws",
        "executor": "server",
    }
    arguments = '{"objective": "Weather in Paris today", "queries": ["Paris weather today"]}'
    ids = {"response_id": response_id, "item_id": "item_call", "call_id": "call_ws"}
    return call, [
        {"type": "response.output_item.added", "response_id": response_id, "output_index": 1, "item": call},
        {"type": "conversation.item.added", "item": call},
        {"type": "response.function_call_arguments.delta", **ids, "delta": arguments},
        {"type": "response.function_call_arguments.done", **ids, "name": "web_search", "arguments": arguments},
        {"type": "response.output_item.done", "response_id": response_id, "output_index": 1, "item": call},
    ]


@pytest.mark.asyncio
async def test_web_search_reaches_boson_unchanged_and_survives_a_settings_update():
    service = PipelineBosonService(tools=readme_tools())

    await service._send_session_update()
    await service._update_settings(service.Settings(session_properties=events.SessionProperties(instructions="Hi.")))

    sent_tools = [message["session"]["tools"] for message in service.sent if message["type"] == "session.update"]
    assert len(sent_tools) == 2
    for tools in sent_tools:
        assert WEB_SEARCH in tools
        assert [tool["name"] for tool in tools if tool["type"] == "function"] == ["get_weather"]


@pytest.mark.asyncio
@pytest.mark.parametrize("call_before_filler_audio", [False, True])
async def test_a_web_search_in_a_real_pipeline_is_left_to_boson_and_its_answer_is_played(call_before_filler_audio):
    # The model says a filler and calls web_search, which ends that reply. Boson adds the
    # result and later starts a reply of its own with the answer. The client sends nothing.
    # Boson can open the filler at the call and send its audio after the call's events, as it
    # did against a local server from voice-chat main.
    tools = readme_tools()
    service = PipelineBosonService(tools=tools)
    context = LLMContext([{"role": "user", "content": "What's the weather in Paris today?"}], tools=tools)
    recorder = FrameRecorder()
    filler, filler_opening, filler_audio = spoken_item("resp_0", "item_filler", "One moment.")
    call, call_events = web_search_call("resp_0")
    first_reply = filler_opening + (
        call_events + filler_audio if call_before_filler_audio else filler_audio + call_events
    )
    answer, answer_opening, answer_audio = spoken_item("resp_1", "item_answer", "It's sunny in Paris.")

    async with running_pipeline(service, context, recorder):
        create = [message for message in service.sent if message["type"] == "response.create"][-1]
        metadata = {"client_event_id": create["event_id"]}
        before = len(service.sent)
        for event in [
            {"type": "response.created", "response": {"id": "resp_0", "metadata": metadata}},
            *first_reply,
            {
                "type": "response.done",
                "response": {"id": "resp_0", "status": "completed", "output": [filler, call], "usage": None},
            },
            {
                "type": "conversation.item.added",
                "item": {"id": "item_out", "type": "function_call_output", "call_id": "call_ws", "output": "Sunny."},
            },
            {"type": "response.created", "response": {"id": "resp_1", "metadata": None}},
            *answer_opening,
            *answer_audio,
            {
                "type": "response.done",
                "response": {"id": "resp_1", "status": "completed", "output": [answer], "usage": None},
            },
        ]:
            await service._dispatch_boson_server_event(event)
        await wait_until(lambda: any(m.get("content") == "It's sunny in Paris." for m in context.get_messages()))
        await asyncio.sleep(0.1)
        sent_after_request = service.sent[before:]

    assert any(
        WEB_SEARCH in message["session"]["tools"] for message in service.sent if message["type"] == "session.update"
    )
    assert sent_after_request == []
    played = [frame for frame in recorder.frames if isinstance(frame, TTSAudioRawFrame)]
    starts = [frame for frame in recorder.frames if isinstance(frame, LLMFullResponseStartFrame)]
    ends = [frame for frame in recorder.frames if isinstance(frame, LLMFullResponseEndFrame)]
    assert (len(played), len(starts), len(ends)) == (2, 2, 2)
    assert [message["content"] for message in context.get_messages()[1:]] == ["One moment.", "It's sunny in Paris."]
