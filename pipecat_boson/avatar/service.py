# ruff: noqa: CPY001
"""Pipecat processors that let a Boson Avatar speak the bot's audio."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass

import aiohttp
from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .api import AvatarAudioSocket, BosonAvatarAPI, BosonAvatarError
from .playback import SAMPLE_RATE, AvatarPlayback


_REPLY_METADATA_KEY = "boson_avatar_reply"


class AvatarAudioOutput(FrameProcessor):
    """Send the bot's speech to an avatar that plays it together with video.

    Place it right after the speech service. It consumes ``TTSAudioRawFrame``
    (the avatar publishes the audio, so the output transport must not play it
    as well) and reports ``BotStartedSpeakingFrame`` / ``BotStoppedSpeakingFrame``
    from the avatar's playback rather than from generation.

    Pass it as ``BosonRealtimeLLMService(audio_playback=...)``: an interruption
    then truncates the reply at the avatar's playback position. Audio from
    other speech services is accepted too, one reply per ``TTSStoppedFrame``.
    """

    def __init__(self, playback: AvatarPlayback, **kwargs):
        super().__init__(**kwargs)
        self._playback = playback
        self._reply: tuple[str, int] | None = None
        self._untracked_reply: str | None = None
        self._speaking = False

    def track_audio(self, frame: TTSAudioRawFrame, *, response_id: str, item_id: str, content_index: int) -> bool:
        """``BosonRealtimeLLMService`` hook: register generated audio before it is pushed."""
        generation = self._playback.observe(response_id, item_id, content_index, len(frame.audio))
        if generation is None:
            return False
        frame.metadata[_REPLY_METADATA_KEY] = (response_id, generation)
        return True

    async def interrupt(self) -> list[tuple[str, int, int]]:
        """``BosonRealtimeLLMService`` hook: stop the avatar and report unheard audio."""
        truncations = await self._playback.interrupt()
        self._reply = self._untracked_reply = None
        await self._stopped_speaking()
        return truncations

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        try:
            if isinstance(frame, StartFrame):
                await self._open()
            elif isinstance(frame, EndFrame | CancelFrame):
                await self._close()
            elif isinstance(frame, TTSAudioRawFrame):
                await self._write(frame)
            elif isinstance(frame, TTSStoppedFrame):
                await self._finish()
            elif isinstance(frame, InterruptionFrame):
                # A no-op when BosonRealtimeLLMService already interrupted.
                await self.interrupt()
        except Exception as exc:  # noqa: BLE001 - Report avatar failures as fatal pipeline errors.
            await self.push_error(f"Avatar output failed: {exc}", exception=exc, fatal=True)
        if not isinstance(frame, TTSAudioRawFrame):
            await self.push_frame(frame, direction)

    async def _open(self) -> None:
        pass

    async def _close(self) -> None:
        pass

    async def _write(self, frame: TTSAudioRawFrame) -> None:
        if frame.sample_rate != SAMPLE_RATE or frame.num_channels != 1:
            raise ValueError(f"The avatar expects {SAMPLE_RATE} Hz mono PCM16 output audio")
        reply = frame.metadata.get(_REPLY_METADATA_KEY)
        if reply is None:
            if self._untracked_reply is None:
                self._untracked_reply = f"tts-{uuid.uuid4().hex}"
            generation = self._playback.observe(self._untracked_reply, self._untracked_reply, 0, len(frame.audio))
            if generation is None:
                return
            reply = (self._untracked_reply, generation)
        if await self._playback.write(*reply, frame.audio):
            self._reply = reply
            if not self._speaking:
                self._speaking = True
                await self.broadcast_frame(BotStartedSpeakingFrame)

    async def _finish(self) -> None:
        reply, self._reply, self._untracked_reply = self._reply, None, None
        if reply is not None and await self._playback.finish(*reply):
            await self._stopped_speaking()

    async def _stopped_speaking(self) -> None:
        if self._speaking:
            self._speaking = False
            await self.broadcast_frame(BotStoppedSpeakingFrame)


@dataclass(frozen=True)
class DailyAvatarRoom:
    """A Daily room that the avatar joins as its own participant.

    Parameters:
        url: Daily room URL.
        token: Meeting token minted for the avatar participant.
        identity: ``user_id`` of that token, used to recognize the avatar.
    """

    url: str
    token: str
    identity: str = "boson-avatar"

    def transport(self) -> dict:
        return {
            "type": "daily",
            "room_url": self.url,
            "participant_token": self.token,
            "participant_identity": self.identity,
            "audio_source": "websocket",
        }


class BosonAvatarService(AvatarAudioOutput):
    """A Boson Avatar participant that speaks the bot's audio in the call.

    On ``StartFrame`` it creates an Avatar session in ``room`` and connects the
    session's audio channel, then emits ``on_session_started(service,
    session_id)``. ``EndFrame`` or ``CancelFrame`` ends the session. If the
    avatar ends the session first, the processor pushes a fatal error.
    """

    def __init__(
        self,
        *,
        api_url: str,
        api_key: str,
        avatar_id: str,
        room: DailyAvatarRoom,
        width: int | None = None,
        height: int | None = None,
        max_duration_seconds: int | None = None,
        idempotency_key: str | None = None,
        aiohttp_session: aiohttp.ClientSession | None = None,
        **kwargs,
    ):
        """Create the avatar output.

        Args:
            api_url: Avatar session API base URL.
            api_key: Bearer key for the Avatar session API.
            avatar_id: Avatar to render.
            room: Room the avatar joins.
            width: Optional output width; requires ``height``.
            height: Optional output height; requires ``width``.
            max_duration_seconds: Optional upper bound for the session.
            idempotency_key: Optional stable key for session creation.
            aiohttp_session: Optional HTTP session; one is created otherwise.
            **kwargs: Passed to ``FrameProcessor``.
        """
        if (width is None) != (height is None):
            raise ValueError("width and height must be provided together")
        self._socket = AvatarAudioSocket()
        super().__init__(AvatarPlayback(self._socket), **kwargs)
        self._body: dict = {"avatar_id": avatar_id, "transport": room.transport()}
        if width is not None:
            self._body["output"] = {"width": width, "height": height}
        if max_duration_seconds is not None:
            self._body["max_duration_seconds"] = max_duration_seconds
        self._idempotency_key = idempotency_key or str(uuid.uuid4())
        self._api_url = api_url
        self._api_key = api_key
        self._http = aiohttp_session
        self._owns_http = aiohttp_session is None
        self._api: BosonAvatarAPI | None = None
        self._session_id: str | None = None
        self._closing = False
        self._ended_task: asyncio.Task | None = None
        self._register_event_handler("on_session_started")

    @property
    def session_id(self) -> str | None:
        """Avatar session ID while the session is live."""
        return self._session_id

    async def _open(self) -> None:
        if self._http is None:
            self._http = aiohttp.ClientSession()
        self._api = BosonAvatarAPI(api_url=self._api_url, api_key=self._api_key, session=self._http)
        session = await self._api.create_session(self._body, idempotency_key=self._idempotency_key)
        self._session_id = session["id"]
        try:
            audio = session.get("audio_input") or {}
            if audio.get("type") != "websocket" or not audio.get("url") or not audio.get("token"):
                raise BosonAvatarError(201, "invalid_response", "Avatar session has no audio channel")
            await self._socket.connect(self._http, audio["url"], audio["token"])
        except BaseException:
            await self._end_session()
            raise
        self._socket.reader.add_done_callback(self._audio_channel_closed)
        logger.info(f"{self}: Avatar session {self._session_id} started")
        await self._call_event_handler("on_session_started", self._session_id)

    def _audio_channel_closed(self, task: asyncio.Task) -> None:
        if self._closing or task.cancelled():
            return
        self._ended_task = asyncio.get_running_loop().create_task(
            self.push_error("Avatar session ended", exception=task.exception(), fatal=True)
        )

    async def _close(self) -> None:
        if self._closing:
            return
        self._closing = True
        try:
            await self._playback.aclose()
            await self._socket.aclose()
            await self._end_session()
        finally:
            if self._owns_http and self._http is not None:
                await self._http.close()

    async def _end_session(self) -> None:
        session_id, self._session_id = self._session_id, None
        if session_id is None:
            return
        try:
            await self._api.end_session(session_id)
        except Exception as exc:  # noqa: BLE001 - The session API also ends sessions on socket loss.
            logger.warning(f"{self}: failed to end Avatar session {session_id}: {exc}")
