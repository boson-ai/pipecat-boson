# ruff: noqa: CPY001
"""Boson Avatar session API client and the per-session audio channel.

The session API follows the public LiveKit Avatar plugin contract. One
resource serves every room provider; ``transport.type`` selects it::

    POST   {api_url}/sessions        Idempotency-Key header
           {"avatar_id", "transport": {"type", ..., "audio_source"}, "output", "max_duration_seconds"}
    DELETE {api_url}/sessions/{id}

When ``transport.audio_source`` is ``"websocket"``, the created session returns
``audio_input = {"type": "websocket", "url", "token"}``. Its messages are::

    client -> avatar  {"type": "start", "reply_id", "generation"}, binary PCM16 24 kHz mono (<= 1 s each),
                      {"type": "end", "reply_id", "generation"}, {"type": "interrupt", "generation"}
    avatar -> client  {"type": "playback_finished", "reply_id", "generation",
                       "playback_position", "interrupted", "clock_source"}

``interrupt`` advances the generation by exactly one. Closing the socket ends
the Avatar session, and the avatar closes it when the session ends first.
"""

from __future__ import annotations

import asyncio
import json

import aiohttp


_MAX_PCM_MESSAGE_BYTES = 48000  # one second of 24 kHz PCM16


class BosonAvatarError(Exception):
    """The Avatar session API rejected a request."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(f"{message} (HTTP {status}, {code})")
        self.status = status
        self.code = code


class BosonAvatarAPI:
    """Create and end Avatar sessions."""

    def __init__(self, *, api_url: str, api_key: str, session: aiohttp.ClientSession):
        if not api_url or not api_key:
            raise ValueError("api_url and api_key are required")
        self._url = api_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
        self._session = session

    async def create_session(self, body: dict, *, idempotency_key: str) -> dict:
        headers = {**self._headers, "Idempotency-Key": idempotency_key}
        async with self._session.post(f"{self._url}/sessions", json=body, headers=headers) as response:
            payload = await _read_json(response)
            if response.status not in (200, 201) or payload.get("status") != "active" or not payload.get("id"):
                raise _api_error(response.status, payload)
            return payload

    async def end_session(self, session_id: str) -> None:
        async with self._session.delete(f"{self._url}/sessions/{session_id}", headers=self._headers) as response:
            if response.status not in (200, 204):
                raise _api_error(response.status, await _read_json(response))


class AvatarAudioSocket:
    """Playback channel for ``AvatarPlayback`` over a session's audio WebSocket."""

    def __init__(self):
        self.reader: asyncio.Task | None = None
        self._socket: aiohttp.ClientWebSocketResponse | None = None
        self._callback = None
        self._active: tuple[str, int] | None = None
        self._last_completed: tuple[str, int] | None = None
        self._clear_key: tuple[str, int] | None = None

    async def connect(self, session: aiohttp.ClientSession, url: str, token: str) -> None:
        self._socket = await session.ws_connect(url, headers={"Authorization": f"Bearer {token}"}, max_msg_size=65536)
        self.reader = asyncio.create_task(self._read())

    def bind_receipts(self, callback) -> None:
        self._callback = callback

    def unbind_receipts(self) -> None:
        self._callback = None

    async def open_reply(self, response_id: str, generation: int):
        if self._socket is None:
            raise RuntimeError("Avatar audio channel is not connected")
        self._active = (response_id, generation)
        await self._socket.send_json({"type": "start", "reply_id": response_id, "generation": generation})
        return _ReplyWriter(self._socket, response_id, generation)

    async def clear(self, generation: int) -> None:
        self._clear_key = self._active or self._last_completed
        await self._socket.send_json({"type": "interrupt", "generation": generation})

    async def _read(self) -> None:
        async for message in self._socket:
            payload = json.loads(message.data)
            if payload.get("type") != "playback_finished":
                raise RuntimeError("Unexpected Avatar audio channel message")
            key = (payload.get("reply_id"), payload.get("generation"))
            expected = self._clear_key if payload.get("interrupted") else self._active
            if key != expected:
                raise RuntimeError("Playback receipt does not match the active reply")
            if self._callback is not None and self._callback(payload) != "ok":
                raise RuntimeError("Rejected Avatar playback receipt")
            self._active = None
            self._last_completed = key
            if payload.get("interrupted"):
                self._clear_key = None
        raise ConnectionError("Avatar audio channel closed")

    async def aclose(self) -> None:
        if self.reader is not None:
            self.reader.cancel()
            await asyncio.gather(self.reader, return_exceptions=True)
        if self._socket is not None:
            await self._socket.close()


class _ReplyWriter:
    def __init__(self, socket, response_id: str, generation: int):
        self._socket = socket
        self._end = {"type": "end", "reply_id": response_id, "generation": generation}

    async def write(self, pcm: bytes) -> None:
        for offset in range(0, len(pcm), _MAX_PCM_MESSAGE_BYTES):
            await self._socket.send_bytes(pcm[offset : offset + _MAX_PCM_MESSAGE_BYTES])

    async def aclose(self) -> None:
        await self._socket.send_json(self._end)


async def _read_json(response: aiohttp.ClientResponse) -> dict:
    text = await response.text()
    try:
        payload = json.loads(text) if text else {}
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _api_error(status: int, payload: dict) -> BosonAvatarError:
    error = payload.get("error")
    if isinstance(error, dict):
        return BosonAvatarError(status, str(error.get("code") or "error"), str(error.get("message") or ""))
    return BosonAvatarError(status, "invalid_response", "Avatar session API returned an unexpected response")
