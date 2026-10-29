# ruff: noqa: CPY001
"""Boson realtime LLM service for Pipecat pipelines."""

from __future__ import annotations

import asyncio
import copy
import json
import uuid
from collections import deque
from types import SimpleNamespace
from typing import Any, Literal

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    AggregatedTextFrame,
    Frame,
    FunctionCallResultFrame,
    InterimTranscriptionFrame,
    LLMConfigureOutputFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMSetToolsFrame,
    LLMTextFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import (
    LLMContext,
)
from pipecat.processors.aggregators.llm_context import (
    is_given as is_context_value_given,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.utils.time import time_now_iso8601
from pydantic import BaseModel
from websockets.asyncio.client import connect as websocket_connect
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK

from ._compat import (
    HANDLE_CONTEXT_TAKES_DIRECTION,
    SENTENCE_TEXT,
    assert_given,
    forget_answered_call,
    is_given,
    resolved,
)
from .config import (
    DEFAULT_TURN_DETECTION,
    SUPPORTED_SEMANTIC_EAGERNESS,
    UNSET,
    build_input_audio_transcription,
    build_noise_reduction,
    build_session_update_payload,
    build_turn_detection,
    input_audio_transcription_enabled,
    normalize_ws_url,
    resolve_output_modalities,
)


_BOSON_TERMINAL_SESSION_EVENT_TYPES = {
    "session.idle_timeout",
    "session.max_duration_reached",
}

# Boson rejects some client events per-request while the session stays healthy.
# These errors must not kill the receive loop.
_BOSON_NONFATAL_ERROR_TYPES = {
    "conversation_item_not_found",
    "invalid_previous_item_id",
    "voice_output_task_ongoing",
}

_BOSON_NONFATAL_ERROR_CODES = {
    "response_id_mismatch",
}

# Some request-level Boson errors currently arrive with only code="400" and a
# human-readable message. Keep this allow-list narrow so unrelated bad requests
# remain fatal to the realtime session.
_BOSON_NONFATAL_400_MESSAGES = {
    "No user input",
}

_BOSON_SERVER_EVENT_HANDLERS = {
    "session.created": "_handle_evt_session_created",
    "session.updated": "_handle_evt_session_updated",
    "response.output_audio.delta": "_handle_evt_audio_delta",
    "input_audio_buffer.speech_started": "_handle_evt_speech_started",
    "input_audio_buffer.speech_stopped": "_handle_evt_speech_stopped",
    "conversation.item.input_audio_transcription.delta": "_handle_evt_input_audio_transcription_delta",
    "conversation.item.input_audio_transcription.completed": "handle_evt_input_audio_transcription_completed",
    "conversation.item.input_audio_transcription.failed": "_handle_evt_input_audio_transcription_failed",
    "conversation.item.added": "_handle_evt_conversation_item_added",
    "conversation.item.retrieved": "_handle_conversation_item_retrieved",
    "response.output_text.delta": "_handle_evt_text_delta",
    "response.output_audio_transcript.delta": "_handle_evt_audio_transcript_delta",
    "response.function_call_arguments.done": "_handle_evt_function_call_arguments_done",
    "response.created": "_handle_evt_response_created",
    "response.done": "_handle_evt_response_done",
    "should_end_call": "_handle_evt_should_end_call",
}

_BOSON_IGNORED_SERVER_EVENT_TYPES = {
    "conversation.created",
    "conversation.item.deleted",
    "conversation.item.truncated",
    "input_audio_buffer.committed",
    "input_audio_buffer.cleared",
    "response.output_item.done",
    "response.content_part.added",
    "response.content_part.done",
    "response.function_call_arguments.delta",
    "response.output_text.done",
    "response.output_audio.done",
    "response.output_audio_transcript.length",
    "response.output_audio_transcript.done",
    "rate_limits.updated",
    "latency_testing",
    "conversation.context.summarized",
    "state.changed",
}

_BOSON_TRANSCRIPTION_DEDUP_CACHE_SIZE = 4096


class BosonRealtimeLLMService(OpenAIRealtimeLLMService):
    """Pipecat realtime LLM service backed by Boson's WebSocket API.

    The service intentionally maps to Boson's OpenAI-compatible realtime API.
    Boson-only session extensions such as states/scripted responses are not part
    of this first Pipecat surface.
    """

    def __init__(
        self,
        *,
        url: str,
        api_key: str | None = None,
        model: str = "higgs-realtime",
        voice: str = "default",
        instructions: str = "You are a helpful AI assistant.",
        output_modalities: list[Literal["text", "audio"]] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | Literal["inf"] = "inf",
        tools: Any = UNSET,
        tool_choice: Any = "auto",
        speed: float = 1.0,
        turn_detection: Any = UNSET,
        input_audio_transcription: Any = UNSET,
        input_audio_transcription_model: str = "",
        input_audio_transcription_language: str | None = None,
        input_audio_noise_reduction: Any = UNSET,
        truncation: Literal["auto", "disabled"] = "auto",
        query_params: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        """Create a Boson realtime service.

        Args:
            url: Boson Realtime API WebSocket URL.
            api_key: Optional bearer token for the Boson Realtime API.
            model: Boson realtime model id sent in ``session.update``.
            voice: Boson voice preset or voice id.
            instructions: System instructions for the realtime session.
            output_modalities: Exactly one output modality, either
                ``["audio"]`` or ``["text"]``. Defaults to audio.
            temperature: Optional sampling temperature sent to Boson. Omit it to
                use the server default.
            max_output_tokens: Maximum output tokens, or ``"inf"``.
            tools: Optional tools to advertise in ``session.update``.
            tool_choice: OpenAI-compatible tool choice value.
            turn_detection: Server VAD config. Omit for the default server VAD;
                pass ``False`` or ``None`` to use local/manual turn boundaries.
            input_audio_transcription: Optional transcription payload. Passing
                ``None`` or omitting a model suppresses client-facing user
                transcript events.
            input_audio_transcription_model: Convenience transcription model.
            input_audio_transcription_language: Convenience transcription language.
            input_audio_noise_reduction: Optional noise reduction config or type.
            truncation: Boson input truncation mode, ``"auto"`` or ``"disabled"``.
            query_params: Optional query parameters to merge into ``url``.
            **kwargs: Additional arguments passed to Pipecat's OpenAI realtime
                base service.
        """

        output_modalities = resolve_output_modalities(output_modalities)
        turn_detection_config = build_turn_detection(turn_detection)
        transcription_config = build_input_audio_transcription(
            input_audio_transcription=input_audio_transcription,
            model=input_audio_transcription_model,
            language=input_audio_transcription_language,
        )
        noise_reduction_config = build_noise_reduction(input_audio_noise_reduction)

        # Pipecat's OpenAI service uses False, not None, to select local/manual
        # turn handling. Boson receives None on the wire when server VAD is
        # disabled, but base Pipecat logic must see False.
        pipecat_turn_detection = _to_pipecat_turn_detection(turn_detection_config)
        pipecat_transcription = _to_pipecat_transcription(transcription_config)
        pipecat_noise_reduction = _to_pipecat_noise_reduction(noise_reduction_config)
        session_tools = None if tools is UNSET else tools
        settings = self.Settings(
            model=model,
            system_instruction=instructions,
            temperature=temperature,
            max_tokens=max_output_tokens if max_output_tokens != "inf" else None,
            session_properties=events.SessionProperties(
                model=model,
                instructions=instructions,
                output_modalities=output_modalities,
                tools=session_tools,
                tool_choice=tool_choice,
                audio=events.AudioConfiguration(
                    input=events.AudioInput(
                        format=events.PCMAudioFormat(),
                        turn_detection=pipecat_turn_detection,
                        transcription=pipecat_transcription,
                        noise_reduction=pipecat_noise_reduction,
                    ),
                    output=events.AudioOutput(format=events.PCMAudioFormat(), voice=voice, speed=speed),
                ),
            ),
        )

        super().__init__(
            api_key=api_key or "boson",
            base_url=normalize_ws_url(url, query_params),
            settings=settings,
            **kwargs,
        )

        # OpenAIRealtimeLLMService appends ?model=... to base_url. Boson
        # receives the model only in session.update, so replace it after parent
        # initialization.
        self.api_key = api_key
        self.base_url = normalize_ws_url(url, query_params)

        self._register_event_handler("on_should_end_call")
        self._register_event_handler("on_session_created", sync=True)
        self._register_event_handler("on_session_terminated")

        self._boson_model = model
        self._boson_voice = voice
        self._boson_instructions = instructions
        self._boson_output_modalities = output_modalities
        self._boson_temperature = temperature
        self._boson_max_output_tokens = max_output_tokens
        self._boson_tool_choice = tool_choice
        self._boson_speed = speed
        self._boson_turn_detection = turn_detection_config
        self._boson_last_enabled_turn_detection = turn_detection_config
        self._boson_input_audio_transcription = transcription_config
        self._boson_input_audio_noise_reduction = noise_reduction_config
        self._boson_truncation = truncation
        self._boson_pending_response_client_event_ids: set[str] = set()
        self._boson_active_response_ids: set[str] = set()
        self._boson_cancelled_response_ids: set[str] = set()
        self._boson_cancelled_response_client_event_ids: set[str] = set()
        self._boson_runtime_tools: Any = UNSET
        self._boson_pending_message_appends: list[LLMMessagesAppendFrame] = []
        self._boson_completed_transcription_item_ids: set[str] = set()
        self._boson_completed_transcription_item_id_order: deque[str] = deque()
        # call_ids of tool calls the server runs itself (server tools such as
        # web_search). They are not ours to run or answer.
        self._boson_server_call_ids: deque[str] = deque(maxlen=_BOSON_TRANSCRIPTION_DEDUP_CACHE_SIZE)
        self._boson_terminal_session_event_type: str | None = None
        self._boson_response_start_frame_active = False

        if session_tools is not None:
            self._sync_registered_tool_handlers(session_tools)

    @property
    def provider(self) -> str:
        """Provider name reported by Pipecat metrics and logs."""

        return "Boson Realtime API"

    @property
    def user_transcription_enabled(self) -> bool:
        """Whether this service is configured to emit user transcription frames."""

        return input_audio_transcription_enabled(self._boson_input_audio_transcription)

    @property
    def audio_output_enabled(self) -> bool:
        """Whether this service is configured to emit realtime audio output frames."""

        return "audio" in self._boson_output_modalities

    @property
    def accepts_intermediate_function_call_results(self) -> bool:
        """Boson has no channel for a tool's partial results: it rejects system items."""

        return False

    def _is_turn_detection_disabled(self) -> bool:
        """Whether the application, rather than Boson, drives turn taking.

        Read from the Boson payload: a settings update replaces Pipecat's copy
        wholesale, which would flip this answer without Boson being told.
        """

        return self._boson_turn_detection is None

    async def _connect(self):
        try:
            if self._websocket:
                return

            self._boson_terminal_session_event_type = None
            # call_ids belong to one session; a new connection starts with none.
            self._boson_server_call_ids.clear()
            headers = {"User-Agent": "Pipecat Boson realtime service"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            self._websocket = await websocket_connect(uri=self.base_url, additional_headers=headers)
            self._receive_task = self.create_task(self._receive_task_handler())
            await self._send_session_update()
        except Exception as e:  # noqa: BLE001 - Surface connection setup failures as Pipecat ErrorFrames.
            await self.push_error(error_msg=f"Error connecting to Boson realtime API: {e}", exception=e)
            await self._mark_websocket_disconnected()

    async def send_client_event(self, event: events.ClientEvent | dict[str, Any]):
        """Send an OpenAI-compatible realtime client event to Boson."""

        if hasattr(event, "model_dump"):
            payload = event.model_dump(exclude_none=True)
        else:
            payload = event
        payload = _normalize_response_create_payload(payload)
        await self._ws_send(payload)

    async def _handle_function_call_result(self, frame: FunctionCallResultFrame) -> None:
        """Send a tool's result as pipecat-ai 1.13 produces it, dropping progress updates Boson can't take."""

        is_final = frame.properties.is_final if frame.properties else True
        if not is_final and not self.accepts_intermediate_function_call_results:
            logger.warning(
                f"Boson takes one result per tool call; dropping a progress update for {frame.function_name}"
            )
            return
        await super()._handle_function_call_result(frame)

    async def _send_tool_result(self, tool_call_id: str, result: str):
        """Send Pipecat's already-serialized tool result on the Boson wire."""

        logger.debug(f"Sending tool result to Boson realtime for tool_call_id={tool_call_id}")
        item = events.ConversationItem(
            type="function_call_output",
            call_id=tool_call_id,
            output=result,
        )
        forget_answered_call(self, tool_call_id)
        await self.send_client_event(events.ConversationItemCreateEvent(item=item))

    async def _ws_send(self, realtime_message):
        try:
            if self._disconnecting or not self._websocket:
                return
            await self._websocket.send(json.dumps(realtime_message))
        except ConnectionClosedOK:
            logger.info("Boson realtime API websocket closed normally; dropping client event")
            await self._mark_websocket_disconnected()
        except ConnectionClosed as exc:
            await self._mark_websocket_disconnected()
            if self._boson_terminal_session_event_type:
                logger.info(
                    "Boson realtime API websocket closed after terminal session event "
                    f"{self._boson_terminal_session_event_type}; dropping client event"
                )
                return
            await self.push_error(
                error_msg=f"Boson realtime API websocket closed while sending client event: {exc}",
                exception=exc,
            )
        except Exception as exc:  # noqa: BLE001 - Match Pipecat's base websocket error behavior.
            if self._disconnecting or not self._websocket:
                return
            await self.push_error(error_msg=f"Error sending client event: {exc}", exception=exc)

    async def _mark_websocket_disconnected(self) -> None:
        if self._websocket is None and not self._api_session_ready:
            return

        websocket = self._websocket
        self._websocket = None
        self._api_session_ready = False
        self._run_llm_when_api_session_ready = False
        self._boson_pending_response_client_event_ids.clear()
        self._boson_active_response_ids.clear()
        self._boson_cancelled_response_ids.clear()
        self._boson_cancelled_response_client_event_ids.clear()
        self._boson_completed_transcription_item_ids.clear()
        self._boson_completed_transcription_item_id_order.clear()
        await self._close_active_response_frames()
        await self.stop_all_metrics()
        if websocket is not None:
            try:
                await websocket.close()
            except Exception as exc:  # noqa: BLE001 - Best-effort cleanup of a possibly failing connection.
                logger.debug(f"Error closing Boson realtime websocket during cleanup: {exc}")

        receive_task = self._receive_task
        if not receive_task:
            return

        if receive_task is asyncio.current_task():
            self._receive_task = None
            return

        await self.cancel_task(receive_task, timeout=1.0)
        self._receive_task = None

    async def _create_response(self):
        if not self._api_session_ready:
            self._run_llm_when_api_session_ready = True
            return

        await self._setup_conversation_if_needed()

        logger.debug("Creating response")

        event_id = str(uuid.uuid4())
        await self._push_response_start_frame_if_needed()
        await self.start_processing_metrics()
        await self.start_ttfb_metrics()
        response = {"metadata": {"client_event_id": event_id}}
        self._boson_pending_response_client_event_ids.add(event_id)
        await self.send_client_event({"event_id": event_id, "type": "response.create", "response": response})

    async def _setup_conversation_if_needed(self) -> None:
        if not self._api_session_ready or not self._llm_needs_conversation_setup:
            return

        adapter = self.get_llm_adapter()

        logger.debug(
            "Setting up conversation on Boson Realtime LLM service with initial messages: "
            f"{adapter.get_messages_for_logging(self._context)}"
        )

        llm_invocation_params = await resolved(adapter.get_llm_invocation_params(self._context))
        messages = llm_invocation_params["messages"]
        for item in messages:
            evt = events.ConversationItemCreateEvent(item=item)
            self._messages_added_manually[evt.item.id] = True
            await self.send_client_event(evt)

        await self._send_session_update()
        self._llm_needs_conversation_setup = False

    async def _send_session_update(self):
        instructions = self._current_instructions()
        # _current_tools already prefers runtime tools; the flag only decides whether the context may override.
        runtime_tools_set = self._boson_runtime_tools is not UNSET
        tools = self._current_tools()

        if self._context:
            adapter = self.get_llm_adapter()
            llm_invocation_params = await resolved(
                adapter.get_llm_invocation_params(
                    self._context,
                    system_instruction=instructions,
                )
            )
            if not runtime_tools_set and llm_invocation_params["tools"]:
                tools = llm_invocation_params["tools"]
            if llm_invocation_params["system_instruction"]:
                instructions = llm_invocation_params["system_instruction"]

        payload = build_session_update_payload(
            event_id=f"session_update_{uuid.uuid4().hex}",
            model=self._current_model(),
            voice=self._boson_voice,
            instructions=instructions,
            output_modalities=self._boson_output_modalities,
            temperature=self._current_temperature(),
            max_output_tokens=self._boson_max_output_tokens,
            tool_choice=self._boson_tool_choice,
            tools=tools,
            speed=self._boson_speed,
            turn_detection=self._boson_turn_detection,
            input_audio_transcription=self._boson_input_audio_transcription,
            input_audio_noise_reduction=self._boson_input_audio_noise_reduction,
            truncation=self._boson_truncation,
        )
        logger.debug(f"Sending Boson session.update with tools={_tool_names(tools)}")
        await self.send_client_event(payload)

    async def _update_settings(self, delta):
        update = delta.session_properties
        if is_given(update):
            # Pipecat's apply_update replaces session_properties wholesale; merge so unmentioned fields survive.
            delta = copy.copy(delta)
            delta.session_properties = _merge_model(assert_given(self._settings.session_properties), update)
        changed = await super(OpenAIRealtimeLLMService, self)._update_settings(delta)
        handled = {"model", "session_properties", "system_instruction", "temperature", "max_tokens"}
        if changed.keys() & handled:
            self._sync_boson_settings_from_pipecat(update if is_given(update) else None)
            await self._send_session_update()
        self._warn_unhandled_updated_settings(changed.keys() - handled)
        return changed

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process Pipecat control frames that require Boson-specific handling."""

        if isinstance(frame, LLMMessagesAppendFrame):
            await super(OpenAIRealtimeLLMService, self).process_frame(frame, direction)
            await self._handle_messages_append(frame)
            return

        if isinstance(frame, LLMSetToolsFrame):
            await super(OpenAIRealtimeLLMService, self).process_frame(frame, direction)
            self._boson_runtime_tools = frame.tools
            self._sync_registered_tool_handlers(frame.tools)
            await self._send_session_update()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMConfigureOutputFrame) and frame.skip_tts:
            logger.warning("Boson realtime API does not support per-response skip_tts; ignoring")

        await super().process_frame(frame, direction)

    async def _handle_context(self, context: LLMContext, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        if self._context is None and self._boson_pending_message_appends:
            self._context = context
            await self._process_completed_function_calls(send_new_results=False)
            run_llm = False
            for frame in self._boson_pending_message_appends:
                self._context.add_messages(frame.messages)
                run_llm = run_llm or frame.run_llm
            self._boson_pending_message_appends.clear()
            await self._setup_conversation_if_needed()
            if run_llm:
                await self._create_response()
            return
        if HANDLE_CONTEXT_TAKES_DIRECTION:
            await super()._handle_context(context, direction)
        else:
            await super()._handle_context(context)

    async def _handle_messages_append(self, frame: LLMMessagesAppendFrame) -> None:
        if self._context is None:
            self._boson_pending_message_appends.append(frame)
            return

        self._context.add_messages(frame.messages)

        if frame.run_llm:
            await self._cancel_active_response()

        if self._llm_needs_conversation_setup:
            await self._setup_conversation_if_needed()
        else:
            await self._send_conversation_messages(frame.messages)

        if frame.run_llm:
            await self._create_response()

    async def _send_conversation_messages(self, messages: list[dict[str, Any]]) -> None:
        adapter = self.get_llm_adapter()
        llm_invocation_params = await resolved(adapter.get_llm_invocation_params(LLMContext(list(messages))))
        for item in llm_invocation_params["messages"]:
            evt = events.ConversationItemCreateEvent(item=item)
            self._messages_added_manually[evt.item.id] = True
            await self.send_client_event(evt)

    async def _cancel_active_response(self) -> None:
        await self._mark_active_response_cancelled(send_cancel=True)

    async def _mark_active_response_cancelled(self, *, send_cancel: bool) -> None:
        if not (
            self._boson_pending_response_client_event_ids
            or self._boson_active_response_ids
            or self._current_assistant_response
            or self._current_audio_response
            or self._boson_response_start_frame_active
        ):
            return
        if send_cancel:
            await self.send_client_event(events.ResponseCancelEvent())
        self._boson_cancelled_response_client_event_ids.update(self._boson_pending_response_client_event_ids)
        self._boson_cancelled_response_ids.update(self._boson_active_response_ids)
        self._boson_pending_response_client_event_ids.clear()
        self._boson_active_response_ids.clear()
        await self._close_active_response_frames()

    async def _receive_task_handler(self):
        try:
            websocket = self._websocket
            if websocket is None:
                return

            async for message in websocket:
                try:
                    event = _loads_ws_message(message)
                    should_continue = await self._dispatch_boson_server_event(event)
                    if not should_continue:
                        return
                except Exception as exc:  # noqa: BLE001 - Keep the receive loop from dying on one bad event.
                    await self.push_error(error_msg=f"Error handling Boson realtime event: {exc}", exception=exc)
        except ConnectionClosedOK:
            logger.info("Boson realtime API websocket closed normally")
        except ConnectionClosed as exc:
            if self._boson_terminal_session_event_type:
                logger.info(
                    "Boson realtime API websocket closed after terminal session event "
                    f"{self._boson_terminal_session_event_type}: {exc}"
                )
                return
            await self.push_error(error_msg=f"Boson realtime API websocket closed unexpectedly: {exc}", exception=exc)
        finally:
            await self._mark_websocket_disconnected()

    async def _dispatch_boson_server_event(self, event: dict[str, Any]) -> bool:
        event_type = event.get("type")
        evt = _to_attr(event)

        handler_name = _BOSON_SERVER_EVENT_HANDLERS.get(event_type)
        if handler_name:
            result = getattr(self, handler_name)(evt)
            if asyncio.iscoroutine(result):
                await result
        elif event_type == "response.output_item.added":
            self._track_function_call(getattr(evt, "item", None))
        elif event_type == "error":
            return await self._handle_error_event(evt)
        elif event_type in _BOSON_TERMINAL_SESSION_EVENT_TYPES:
            self._boson_terminal_session_event_type = event_type
            logger.info(f"Boson realtime API terminal session event received: {event_type}")
            await self._call_event_handler("on_session_terminated", event_type, evt)
        elif event_type in _BOSON_IGNORED_SERVER_EVENT_TYPES:
            return True
        else:
            logger.debug(f"Ignoring unsupported Boson realtime event type: {event_type}")
        return True

    async def _handle_evt_session_created(self, evt):
        logger.debug("Boson realtime API session created")
        await self._call_event_handler("on_session_created", evt)
        await self._mark_api_session_ready()

    async def _handle_evt_session_updated(self, evt):
        await self._mark_api_session_ready()

    async def _handle_evt_speech_started(self, evt):
        await self._truncate_current_audio_response()
        await self.broadcast_frame(UserStartedSpeakingFrame)
        await self.broadcast_interruption()

    async def _handle_evt_conversation_item_added(self, evt) -> None:
        item = evt.item
        item_id = getattr(item, "id", None)

        self._track_function_call(item)

        if item_id:
            await self._call_event_handler("on_conversation_item_created", item_id, item)

        if item_id and self._messages_added_manually.get(item_id):
            del self._messages_added_manually[item_id]
            return

        if getattr(item, "role", None) == "assistant":
            self._current_assistant_response = item
            await self._push_response_start_frame_if_needed()

    async def _handle_evt_should_end_call(self, evt) -> None:
        await self._call_event_handler("on_should_end_call", evt)

    async def _handle_interruption(self):
        if self._is_turn_detection_disabled():
            await self.send_client_event(events.InputAudioBufferClearEvent())
            await self._replay_user_audio_preroll()
        await self._truncate_current_audio_response()
        await self._mark_active_response_cancelled(send_cancel=True)
        await self.stop_all_metrics()

    async def _push_response_start_frame_if_needed(self) -> None:
        if self._boson_response_start_frame_active:
            return
        self._boson_response_start_frame_active = True
        await self.push_frame(LLMFullResponseStartFrame())

    async def _close_active_response_frames(self) -> None:
        should_push_end = (
            self._boson_response_start_frame_active
            or self._current_assistant_response is not None
            or self._current_audio_response is not None
        )
        if not should_push_end:
            self._current_assistant_response = None
            self._current_audio_response = None
            self._boson_response_start_frame_active = False
            return

        await self.stop_processing_metrics()
        if self._current_audio_response is not None:
            await self.push_frame(TTSStoppedFrame())
        await self.push_frame(LLMFullResponseEndFrame())
        self._current_assistant_response = None
        self._current_audio_response = None
        self._boson_response_start_frame_active = False

    async def _mark_api_session_ready(self):
        self._api_session_ready = True
        if self._run_llm_when_api_session_ready:
            self._run_llm_when_api_session_ready = False
            await self._create_response()

    async def handle_evt_input_audio_transcription_completed(self, evt):
        """Forward final user transcriptions while de-duplicating repeated server events."""

        item_id = getattr(evt, "item_id", None)
        if item_id and not self._mark_transcription_item_completed(item_id):
            return
        await super().handle_evt_input_audio_transcription_completed(evt)

    async def _handle_evt_speech_stopped(self, evt) -> None:
        """Mark the caller's turn as soon as the server ends it.

        The server's transcript of a turn can arrive after the reply to it has
        started, so clients that order the conversation by arrival (the Pipecat
        playground) put the caller's words under the reply. An empty interim
        transcript now opens the caller's turn first; the final transcript then
        fills it in.
        """
        await super()._handle_evt_speech_stopped(evt)
        if self.user_transcription_enabled:
            await self.push_frame(InterimTranscriptionFrame("", "", time_now_iso8601()), FrameDirection.UPSTREAM)

    async def _handle_evt_input_audio_transcription_failed(self, evt) -> None:
        error = getattr(evt, "error", None)
        message = (getattr(error, "message", None) if error is not None else None) or "transcription failed"
        item_id = getattr(evt, "item_id", None)
        if item_id:
            logger.warning(f"Boson input audio transcription failed for item {item_id}: {message}")
        else:
            logger.warning(f"Boson input audio transcription failed: {message}")

    def _mark_transcription_item_completed(self, item_id: str) -> bool:
        if item_id in self._boson_completed_transcription_item_ids:
            return False

        self._boson_completed_transcription_item_ids.add(item_id)
        self._boson_completed_transcription_item_id_order.append(item_id)
        while len(self._boson_completed_transcription_item_id_order) > _BOSON_TRANSCRIPTION_DEDUP_CACHE_SIZE:
            expired_item_id = self._boson_completed_transcription_item_id_order.popleft()
            self._boson_completed_transcription_item_ids.discard(expired_item_id)
        return True

    async def _handle_error_event(self, evt) -> bool:
        """Handle a Boson error event. Returns False when the error is fatal."""

        error = getattr(evt, "error", None)
        error_type = getattr(error, "type", None) if error is not None else None
        error_code = getattr(error, "code", None) if error is not None else None
        message = (getattr(error, "message", None) if error is not None else None) or "Boson realtime API error"

        if error_code == "response_not_active":
            # Idempotent response.cancel: the server had nothing active to cancel.
            self._boson_pending_response_client_event_ids.clear()
            self._boson_active_response_ids.clear()
            self._boson_cancelled_response_ids.clear()
            self._boson_cancelled_response_client_event_ids.clear()
            await self._close_active_response_frames()
            logger.debug("Ignoring response_not_active from idempotent response.cancel")
            return True

        if error_type == "conversation_item_not_found":
            self._fail_pending_conversation_item_retrievals(error, message)

        if (
            error_type in _BOSON_NONFATAL_ERROR_TYPES
            or error_code in _BOSON_NONFATAL_ERROR_CODES
            or (error_code == "400" and message in _BOSON_NONFATAL_400_MESSAGES)
        ):
            logger.warning(f"Ignoring non-fatal Boson realtime error: {message}")
            return True

        if error_code and await self._maybe_handle_evt_retrieve_conversation_item_error(evt):
            return True

        await self._handle_evt_error(evt)
        return False

    def _fail_pending_conversation_item_retrievals(self, error: Any, message: str) -> None:
        # Boson retrieve errors do not echo the client's "rci_{item_id}" event_id,
        # so fall back to parsing the item id from the error message.
        item_id = _conversation_item_id_from_error(error, message)
        item_ids = [item_id] if item_id else list(self._retrieve_conversation_item_futures.keys())
        for current_item_id in item_ids:
            futures = self._retrieve_conversation_item_futures.pop(current_item_id, None)
            for future in futures or []:
                future.set_exception(Exception(message))

    def _handle_evt_response_created(self, evt) -> None:
        # Pipecat's handler clears it: a started response takes in everything sent before it.
        self._results_awaiting_response = False
        response = evt.response
        response_id = getattr(response, "id", None)
        if not response_id:
            return

        client_event_id = _response_client_event_id(response)
        if client_event_id:
            if client_event_id in self._boson_cancelled_response_client_event_ids:
                self._boson_cancelled_response_client_event_ids.discard(client_event_id)
                self._boson_cancelled_response_ids.add(response_id)
                return
            if client_event_id not in self._boson_pending_response_client_event_ids:
                return
            self._boson_pending_response_client_event_ids.discard(client_event_id)

        self._boson_active_response_ids.add(response_id)

    async def _handle_evt_audio_delta(self, evt):
        if self._is_stale_response_event(evt):
            return
        await super()._handle_evt_audio_delta(evt)

    async def _handle_evt_text_delta(self, evt):
        if self._is_stale_response_event(evt):
            return
        await super()._handle_evt_text_delta(evt)

    async def _push_output_transcript_text_frames(self, text: str):
        """Report the reply transcript as text that is being spoken.

        Boson speaks this text itself. The base class pushes it as a
        ``TTSTextFrame`` without ``will_be_spoken``, so RTVI 2 clients (the
        Pipecat playground, the voice UI kit) file every reply as unspoken and
        leave the assistant's message empty. Announce each piece the way a TTS
        service does instead: an ``AggregatedTextFrame`` that will be spoken
        (RTVI ``spoken_status: new``), then the ``TTSTextFrame`` once it is
        (``completed``).
        """
        llm_text_frame = LLMTextFrame(text)
        llm_text_frame.append_to_context = False
        await self.push_frame(llm_text_frame)

        announced = AggregatedTextFrame(text, **SENTENCE_TEXT)
        announced.append_to_context = False
        announced.will_be_spoken = True
        announced.includes_inter_frame_spaces = True
        await self.push_frame(announced)

        spoken = TTSTextFrame(text, **SENTENCE_TEXT)
        spoken.will_be_spoken = True
        spoken.includes_inter_frame_spaces = True
        await self.push_frame(spoken)

    async def _handle_evt_audio_transcript_delta(self, evt):
        if self._is_stale_response_event(evt):
            return
        await super()._handle_evt_audio_transcript_delta(evt)

    async def _handle_evt_response_done(self, evt):
        response = evt.response
        response_id = getattr(response, "id", None)
        client_event_id = _response_client_event_id(response)
        if self._is_stale_response_done(response, response_id=response_id, client_event_id=client_event_id):
            return

        if response_id:
            self._boson_active_response_ids.discard(response_id)
        if client_event_id:
            self._boson_pending_response_client_event_ids.discard(client_event_id)

        usage = getattr(response, "usage", None)
        if usage:
            input_details = getattr(usage, "input_token_details", None)
            cached_tokens = getattr(input_details, "cached_tokens", None) if input_details else None
            tokens = LLMTokenUsage(
                prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
                completion_tokens=getattr(usage, "output_tokens", 0) or 0,
                total_tokens=getattr(usage, "total_tokens", 0) or 0,
                cache_read_input_tokens=cached_tokens,
            )
            await self.start_llm_usage_metrics(tokens)

        await self._close_active_response_frames()

        if getattr(response, "status", None) == "failed":
            await self.push_error(error_msg=_response_error_message(response))
            return

        for item in getattr(response, "output", []) or []:
            await self._call_event_handler("on_conversation_item_updated", item.id, item)

    def _is_stale_response_done(
        self,
        response: Any,
        *,
        response_id: str | None,
        client_event_id: str | None,
    ) -> bool:
        if client_event_id and client_event_id in self._boson_cancelled_response_client_event_ids:
            self._boson_cancelled_response_client_event_ids.discard(client_event_id)
            if response_id:
                self._boson_cancelled_response_ids.discard(response_id)
            logger.debug(f"Ignoring stale response.done for cancelled client event {client_event_id}")
            return True

        if response_id and response_id in self._boson_cancelled_response_ids:
            self._boson_cancelled_response_ids.discard(response_id)
            logger.debug(f"Ignoring stale response.done for cancelled response {response_id}")
            return True

        if response_id and self._boson_active_response_ids and response_id not in self._boson_active_response_ids:
            logger.debug(f"Ignoring stale response.done for inactive response {response_id}")
            return True

        if client_event_id and client_event_id in self._boson_pending_response_client_event_ids:
            return False

        if (
            not response_id
            and getattr(response, "status", None) == "cancelled"
            and (self._boson_active_response_ids or self._boson_pending_response_client_event_ids)
        ):
            logger.debug("Ignoring stale cancelled response.done without response id")
            return True

        return False

    def _is_stale_response_event(self, evt) -> bool:
        response_id = getattr(evt, "response_id", None)
        if not response_id:
            return False
        if response_id in self._boson_cancelled_response_ids:
            logger.debug(f"Ignoring stale event for cancelled response {response_id}")
            return True
        if self._boson_active_response_ids and response_id not in self._boson_active_response_ids:
            logger.debug(f"Ignoring stale event for inactive response {response_id}")
            return True
        return False

    def _track_function_call(self, item: Any) -> None:
        """Track a function_call item as a call to run here, unless the server runs it.

        Server tools (``{"type": "web_search"}``) show up as ordinary
        ``function_call`` items with ``executor: "server"``. The server runs
        them and adds the ``function_call_output`` itself; the result is spoken
        in the next turn, or in a reply the server starts after a pause. A
        client must not run them, answer them or request a response. Either
        item event may carry the executor, so a call once marked stays the
        server's and leaves the pending calls.
        """
        if getattr(item, "type", None) != "function_call":
            return
        call_id = getattr(item, "call_id", None)
        if getattr(item, "executor", None) == "server" or call_id in self._boson_server_call_ids:
            if call_id and call_id not in self._boson_server_call_ids:
                self._boson_server_call_ids.append(call_id)
                logger.debug(f"Server is running tool {getattr(item, 'name', None)!r} (call_id={call_id})")
            self._pending_function_calls.pop(call_id, None)
            return
        if call_id and call_id not in self._pending_function_calls:
            self._pending_function_calls[call_id] = item

    async def _handle_evt_function_call_arguments_done(self, evt):
        if getattr(evt, "call_id", None) in self._boson_server_call_ids:
            return
        await super()._handle_evt_function_call_arguments_done(evt)

    def _current_model(self) -> str:
        if is_given(self._settings.model) and self._settings.model:
            return self._settings.model
        return self._boson_model

    def _current_instructions(self) -> str:
        if is_given(self._settings.system_instruction) and self._settings.system_instruction:
            return self._settings.system_instruction
        return self._boson_instructions

    def _current_temperature(self) -> float | None:
        if is_given(self._settings.temperature):
            return self._settings.temperature
        return self._boson_temperature

    def _service_tools(self) -> ToolsSchema | list[Any] | None:
        """Tools Pipecat registers handlers for.

        Same precedence as the wire payload, so the handler registry and the
        tools Boson is offered agree.
        """

        if self._boson_runtime_tools is not UNSET:
            return self._boson_runtime_tools
        return assert_given(self._settings.session_properties).tools

    def _current_tools(self) -> list[dict[str, Any]]:
        return self._tools_from_value(self._service_tools())

    def _tools_from_value(self, tools: Any) -> list[dict[str, Any]]:
        if tools is None or not is_given(tools) or not is_context_value_given(tools):
            return []
        adapter = self.get_llm_adapter()
        if isinstance(tools, ToolsSchema):
            return adapter.from_standard_tools(tools) or []
        if isinstance(tools, list) and any(isinstance(tool, FunctionSchema) or callable(tool) for tool in tools):
            return adapter.from_standard_tools(ToolsSchema(standard_tools=tools)) or []
        return [_dump_tool(tool) for tool in tools]

    def _sync_boson_settings_from_pipecat(self, session_properties: events.SessionProperties | None) -> None:
        self._boson_model = self._current_model()
        self._boson_instructions = self._current_instructions()
        self._boson_temperature = self._current_temperature()
        if is_given(self._settings.max_tokens):
            self._boson_max_output_tokens = self._settings.max_tokens or "inf"
        if session_properties is None:
            return
        if session_properties.output_modalities:
            self._boson_output_modalities = resolve_output_modalities(session_properties.output_modalities)
        if session_properties.tool_choice is not None:
            self._boson_tool_choice = session_properties.tool_choice
        if session_properties.max_output_tokens is not None:
            self._boson_max_output_tokens = session_properties.max_output_tokens
        if session_properties.audio and session_properties.audio.output:
            output = session_properties.audio.output
            if output.voice is not None:
                self._boson_voice = output.voice
            if output.speed is not None:
                self._boson_speed = output.speed
        if session_properties.audio and session_properties.audio.input:
            audio_input = session_properties.audio.input
            if _field_was_set(audio_input, "turn_detection"):
                merged = _drop_unsupported_eagerness(
                    _merge_turn_detection(
                        self._boson_turn_detection or self._boson_last_enabled_turn_detection,
                        audio_input.turn_detection,
                    )
                )
                if self._websocket is not None and merged != self._boson_turn_detection:
                    # Connecting sends the first session.update, and Boson rejects later turn-detection changes.
                    logger.warning(
                        "Boson fixes turn detection when the session starts; ignoring this turn-detection update."
                    )
                    merged = self._boson_turn_detection
                    # Keep Pipecat's copy in step too, or repeating this request would look like no change.
                    stored_input = assert_given(self._settings.session_properties).audio.input
                    stored_input.turn_detection = _to_pipecat_turn_detection(merged)
                if merged is not None:
                    # A copy: the live payload must not alias what we remember.
                    self._boson_last_enabled_turn_detection = dict(merged)
                self._boson_turn_detection = merged
            if _field_was_set(audio_input, "transcription"):
                transcription = _dump_model(audio_input.transcription)
                self._boson_input_audio_transcription = (
                    build_input_audio_transcription(input_audio_transcription=transcription)
                    if isinstance(transcription, dict)
                    else transcription
                )
            if _field_was_set(audio_input, "noise_reduction"):
                self._boson_input_audio_noise_reduction = _dump_model(audio_input.noise_reduction)


def _merge_model(current: BaseModel, update: BaseModel) -> BaseModel:
    """``current`` with the fields ``update`` set laid over it.

    Nested models of the same type merge field by field; anything else, such as a
    list or a different VAD type, replaces the stored value.
    """

    changes = {}
    for name in update.model_fields_set:
        value = getattr(update, name)
        stored = getattr(current, name, None)
        if isinstance(value, BaseModel):
            value = _merge_model(stored, value) if type(value) is type(stored) else value.model_copy(deep=True)
        changes[name] = value
    return current.model_copy(update=changes)


def _drop_unsupported_eagerness(turn_detection: dict[str, Any] | None) -> dict[str, Any] | None:
    """Strip an eagerness value Boson would refuse.

    Boson accepts only ``"auto"`` and ends the session otherwise. Construction
    rejects a bad value; dropping it here keeps a live session alive.
    """

    if not isinstance(turn_detection, dict):
        return turn_detection
    eagerness = turn_detection.get("eagerness")
    if eagerness is None or eagerness in SUPPORTED_SEMANTIC_EAGERNESS:
        return turn_detection
    logger.warning(
        f"Boson realtime API accepts only eagerness={SUPPORTED_SEMANTIC_EAGERNESS[0]!r}; "
        f"dropping eagerness={eagerness!r} from the turn detection update."
    )
    sanitized = dict(turn_detection)
    sanitized.pop("eagerness")
    return sanitized


def _turn_detection_fields(value: Any) -> dict[str, Any]:
    """The turn-detection fields a caller actually set, type excluded.

    ``exclude_unset`` stops Pipecat's model defaults overwriting values the caller
    never mentioned; this service's own model passes absent fields as None.
    """

    if hasattr(value, "model_dump"):
        dumped = value.model_dump(exclude_unset=True)
    elif isinstance(value, dict):
        dumped = dict(value)
    else:
        return {}
    return {name: field for name, field in dumped.items() if field is not None and name != "type"}


def _turn_detection_type(value: Any) -> str | None:
    """The VAD type a value implies, which its dump may omit."""

    if isinstance(value, events.SemanticTurnDetection):
        return "semantic_vad"
    if isinstance(value, events.TurnDetection):
        return "server_vad"
    if isinstance(value, dict):
        return value.get("type")
    return getattr(value, "type", None)


def _merge_turn_detection(current: dict[str, Any] | None, value: Any) -> dict[str, Any] | None:
    """Overlay the fields a caller set onto the Boson payload.

    Pipecat's models cannot express every field Boson accepts, so replacing the
    payload would drop them on every settings update.
    """

    if value is False:
        return None
    if value is None or value is True:
        # Only False opts out (Pipecat's convention); None and True re-enable the last configuration.
        return dict(current) if isinstance(current, dict) else dict(DEFAULT_TURN_DETECTION)
    fields = _turn_detection_fields(value)
    vad_type = _turn_detection_type(value)
    # Boson reads a missing type as server VAD.
    current_type = (current.get("type") or "server_vad") if isinstance(current, dict) else None
    if not isinstance(current, dict) or (vad_type and current_type != vad_type):
        # A different VAD type shares no fields: start fresh.
        return {**({"type": vad_type} if vad_type else {}), **fields}
    merged = dict(current)
    merged.update(fields)
    if vad_type:
        merged["type"] = vad_type
    return merged


def _loads_ws_message(message: Any) -> dict[str, Any]:
    if isinstance(message, bytes):
        message = message.decode("utf-8")
    event = json.loads(message)
    if not isinstance(event, dict):
        raise TypeError("Boson realtime event must be a JSON object")
    return event


def _to_attr(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _to_attr(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_attr(item) for item in value]
    return value


def _dump_tool(tool: Any) -> dict[str, Any]:
    if hasattr(tool, "model_dump"):
        return tool.model_dump(exclude_none=True)
    return dict(tool)


def _tool_names(tools: list[dict[str, Any]] | None) -> list[str]:
    names: list[str] = []
    for tool in tools or []:
        name = tool.get("name")
        if isinstance(name, str):
            names.append(name)
    return names


def _dump_model(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return dict(value)
    return value


def _field_was_set(value: Any, field: str) -> bool:
    fields_set = getattr(value, "model_fields_set", None)
    if fields_set is not None:
        return field in fields_set
    if isinstance(value, dict):
        return field in value
    return getattr(value, field, None) is not None


def _response_client_event_id(response: Any) -> str | None:
    metadata = getattr(response, "metadata", None)
    if isinstance(metadata, dict):
        client_event_id = metadata.get("client_event_id")
    else:
        client_event_id = getattr(metadata, "client_event_id", None)
    return client_event_id if isinstance(client_event_id, str) else None


def _conversation_item_id_from_error(error: Any, message: str) -> str | None:
    event_id = getattr(error, "event_id", None) if error is not None else None
    if isinstance(event_id, str) and event_id.startswith("rci_"):
        return event_id.split("_", 1)[1]

    prefix = "Conversation item not found: "
    if message.startswith(prefix):
        return message[len(prefix) :]
    return None


def _to_pipecat_turn_detection(
    value: dict[str, Any] | None,
) -> events.TurnDetection | events.SemanticTurnDetection | bool:
    if value is None:
        return False

    turn_detection_type = value.get("type")
    if turn_detection_type == "semantic_vad":
        return events.SemanticTurnDetection(
            eagerness=value.get("eagerness"),
            create_response=value.get("create_response"),
            interrupt_response=value.get("interrupt_response"),
        )

    return events.TurnDetection(
        threshold=value.get("threshold"),
        prefix_padding_ms=value.get("prefix_padding_ms"),
        silence_duration_ms=value.get("silence_duration_ms"),
    )


def _to_pipecat_transcription(value: dict[str, Any] | object) -> events.InputAudioTranscription | None:
    if not isinstance(value, dict) or not value.get("model"):
        return None
    return events.InputAudioTranscription(
        model=value.get("model"),
        language=value.get("language"),
    )


def _to_pipecat_noise_reduction(value: dict[str, Any] | None) -> events.InputAudioNoiseReduction | None:
    if value is UNSET or not value:
        return None
    return events.InputAudioNoiseReduction(type=value.get("type"))


def _normalize_response_create_payload(payload: dict[str, Any]) -> dict[str, Any]:
    # This service's own response.create payloads never carry modalities, but
    # base-class pydantic events routed through send_client_event use OpenAI's
    # "output_modalities" naming, which Boson expects as "modalities".
    if payload.get("type") != "response.create":
        return payload

    response = payload.get("response")
    if not isinstance(response, dict):
        return payload
    if "output_modalities" not in response:
        return payload

    normalized_response = dict(response)
    output_modalities = normalized_response.pop("output_modalities")
    if "modalities" not in normalized_response:
        normalized_response["modalities"] = output_modalities
    return {**payload, "response": normalized_response}


def _response_error_message(response: Any) -> str:
    details = getattr(response, "status_details", None)
    error = getattr(details, "error", None) if details is not None else None
    if error is None and isinstance(details, dict):
        error = details.get("error")
    if isinstance(error, dict):
        return error.get("message") or "Boson realtime response failed"
    message = getattr(error, "message", None)
    return message or "Boson realtime response failed"
