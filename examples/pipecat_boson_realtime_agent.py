# ruff: noqa: CPY001
"""Browser example for Boson's Pipecat realtime service with function calling."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from collections import deque
from typing import Any

import pipecat.processors.frameworks.rtvi.models as RTVI
from dotenv import load_dotenv
from loguru import logger
from pipecat.frames.frames import (
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.rtvi import (
    RTVIObserver,
    RTVIObserverParams,
    RTVIProcessor,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

from pipecat_boson.realtime import BosonMessagesAppendRelay, BosonRealtimeLLMService


load_dotenv(override=True)

DEFAULT_ASR_MODEL = "higgs-stt-3.1"
DEFAULT_INSTRUCTIONS = (
    "You are a concise voice assistant. Greet the user briefly when the session starts. "
    "When the user asks about weather, call get_weather and then summarize the returned result."
)
DISABLED_ENV_VALUES = {"", "0", "false", "none", "off", "disabled"}
TEXT_ECHO_TTL_SECONDS = 15.0

transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    "websocket": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        serializer=ProtobufFrameSerializer(),
    ),
}


class BosonExampleRTVIObserver(RTVIObserver):
    """RTVI observer used by this example to avoid duplicate typed-message display.

    The default RTVI client echoes typed input locally. Boson also returns final
    user transcriptions for text and voice turns. This observer keeps voice
    transcriptions visible while dropping matching typed echoes and duplicate
    user context messages.
    """

    def __init__(self, *args, text_echo_ttl_seconds: float = TEXT_ECHO_TTL_SECONDS, **kwargs):
        super().__init__(*args, **kwargs)
        self._text_echo_ttl_seconds = text_echo_ttl_seconds
        self._pending_text_echoes: deque[tuple[str, float]] = deque()
        self._pending_user_transcriptions: deque[tuple[str, float]] = deque()

    async def on_push_frame(self, data):
        frame = data.frame
        if isinstance(frame, LLMFullResponseStartFrame):
            self._bot_transcription = ""
        if data.source is self._rtvi and isinstance(frame, LLMMessagesAppendFrame):
            self._remember_typed_text(frame)
        await super().on_push_frame(data)
        if isinstance(frame, LLMFullResponseEndFrame):
            await self._flush_bot_transcription()

    async def _handle_context(self, frame: LLMContextFrame):
        text = _last_user_message_text(frame)
        if text and self._consume_typed_echo(text):
            logger.debug(f"Skipping duplicate typed user LLM text: {text!r}")
            return
        if text and self._consume_user_transcription_echo(text):
            logger.debug(f"Skipping duplicate transcribed user LLM text: {text!r}")
            return
        await super()._handle_context(frame)

    async def _handle_user_transcriptions(self, frame):
        if isinstance(frame, TranscriptionFrame) and self._consume_typed_echo(frame.text):
            logger.debug(f"Skipping duplicate typed user transcription: {frame.text!r}")
            return
        if isinstance(frame, TranscriptionFrame):
            self._remember_user_transcription(frame.text)
        await super()._handle_user_transcriptions(frame)

    async def _flush_bot_transcription(self) -> None:
        text = self._bot_transcription
        if not text:
            return
        await self.send_rtvi_message(RTVI.BotTranscriptionMessage(data=RTVI.TextMessageData(text=text)))
        self._bot_transcription = ""

    def _remember_typed_text(self, frame: LLMMessagesAppendFrame) -> None:
        for message in frame.messages:
            if message.get("role") != "user":
                continue
            text = _message_content_text(message.get("content"))
            normalized = _normalize_text_echo(text)
            if normalized:
                self._pending_text_echoes.append((normalized, time.monotonic() + self._text_echo_ttl_seconds))

    def _consume_typed_echo(self, text: str) -> bool:
        self._drop_expired_echoes()
        normalized = _normalize_text_echo(text)
        if not normalized:
            return False

        for index, (pending_text, _) in enumerate(self._pending_text_echoes):
            if pending_text == normalized:
                del self._pending_text_echoes[index]
                return True
        return False

    def _remember_user_transcription(self, text: str) -> None:
        normalized = _normalize_text_echo(text)
        if normalized:
            self._pending_user_transcriptions.append((normalized, time.monotonic() + self._text_echo_ttl_seconds))

    def _consume_user_transcription_echo(self, text: str) -> bool:
        self._drop_expired_echoes()
        normalized = _normalize_text_echo(text)
        if not normalized:
            return False

        for index, (pending_text, _) in enumerate(self._pending_user_transcriptions):
            if pending_text == normalized:
                del self._pending_user_transcriptions[index]
                return True
        return False

    def _drop_expired_echoes(self) -> None:
        now = time.monotonic()
        while self._pending_text_echoes and self._pending_text_echoes[0][1] <= now:
            self._pending_text_echoes.popleft()
        while self._pending_user_transcriptions and self._pending_user_transcriptions[0][1] <= now:
            self._pending_user_transcriptions.popleft()


class BosonExampleRTVIProcessor(RTVIProcessor):
    def create_rtvi_observer(self, *, params: RTVIObserverParams | None = None, **kwargs):
        return BosonExampleRTVIObserver(self, params=params, **kwargs)


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for part in content:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            parts.append(part["text"])
    return " ".join(parts)


def _last_user_message_text(frame: LLMContextFrame) -> str:
    messages = frame.context.get_messages()
    if not messages:
        return ""

    message = messages[-1]
    if not isinstance(message, dict) or message.get("role") != "user":
        return ""
    return _message_content_text(message.get("content"))


def _normalize_text_echo(text: str) -> str:
    return " ".join(text.split())


def _env_or_default(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def _json_output_modalities() -> list[str] | None:
    value = os.environ.get("BOSON_OUTPUT_MODALITIES")
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _configure_log_level() -> None:
    level = os.environ.get("BOSON_PIPECAT_LOG_LEVEL", "INFO").upper()
    logger.remove()
    logger.add(sys.stderr, level=level)


def _input_audio_transcription() -> dict[str, str] | None:
    transcription_model = os.environ.get("BOSON_ASR_MODEL", DEFAULT_ASR_MODEL)
    if transcription_model.lower() in DISABLED_ENV_VALUES:
        return None

    transcription = {"model": transcription_model}
    if os.environ.get("BOSON_ASR_LANGUAGE"):
        transcription["language"] = os.environ["BOSON_ASR_LANGUAGE"]
    return transcription


def _client_state(client) -> str:
    pc = getattr(client, "pc", None)
    if pc is None:
        return "pc=unknown"
    return (
        f"pc_id={getattr(client, 'pc_id', 'unknown')} "
        f"connection={pc.connectionState} ice={pc.iceConnectionState} gathering={pc.iceGatheringState}"
    )


async def get_weather(params: FunctionCallParams, location: str = "San Francisco") -> None:
    """Get the current weather for a city or place.

    Args:
        location: City or place name, for example "San Francisco".
    """

    logger.info(f"get_weather tool called location={location!r}")
    result: dict[str, Any] = {
        "location": location,
        "condition": "sunny",
        "temperature_f": 72,
        "humidity_percent": 45,
        "wind_mph": 6,
        "summary": f"The weather in {location} is sunny and 72 F.",
    }
    await params.result_callback(result)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting Boson Pipecat realtime bot")

    llm = BosonRealtimeLLMService(
        url=os.environ.get("BOSON_REALTIME_URL", "wss://api.boson.ai/v1/realtime/"),
        api_key=os.environ.get("BOSON_API_KEY"),
        model=os.environ.get("BOSON_REALTIME_MODEL", "higgs-realtime"),
        voice=os.environ.get("BOSON_REALTIME_VOICE", "default"),
        instructions=_env_or_default("BOSON_REALTIME_INSTRUCTIONS", DEFAULT_INSTRUCTIONS),
        output_modalities=_json_output_modalities(),
        tools=[get_weather],
        input_audio_transcription=_input_audio_transcription(),
        input_audio_noise_reduction=os.environ.get("BOSON_NOISE_REDUCTION") or None,
    )

    context = LLMContext([], [get_weather])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context, realtime_service_mode=True)
    rtvi = BosonExampleRTVIProcessor()

    pipeline = Pipeline(
        [
            transport.input(),
            # Preserve run_llm for typed text / LLMMessagesAppendFrame past the aggregator.
            BosonMessagesAppendRelay(),
            user_aggregator,
            llm,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True),
        rtvi_processor=rtvi,
        rtvi_observer_params=RTVIObserverParams(),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )
    client_connected = False

    @rtvi.event_handler("on_client_ready")
    async def on_rtvi_client_ready(rtvi):
        # Let PipelineWorker's default RTVI bot-ready handler reach the client
        # before the greeting starts the realtime LLM connection.
        await asyncio.sleep(0.5)
        if not client_connected:
            logger.debug("Skipping greeting because browser client disconnected")
            return
        logger.info("RTVI client ready; queueing greeting LLMRunFrame")
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        nonlocal client_connected
        client_connected = True
        logger.info(f"Browser client connected ({_client_state(client)})")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        nonlocal client_connected
        client_connected = False
        logger.info(f"Browser client disconnected ({_client_state(client)})")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    _configure_log_level()
    transport = await create_transport(
        runner_args,
        transport_params,
    )
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
