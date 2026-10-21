# ruff: noqa: CPY001
"""Reply audio and playback receipts for an avatar that plays speech itself.

Generation, delivery and playback are different clocks: a realtime response can
finish generating while the avatar still has seconds of it to play. This ledger
keeps each reply's identity until the avatar reports where its playback ended,
so an interruption truncates exactly the audio the user did not hear.

No Pipecat imports. The channel is injected; it is any object with::

    bind_receipts(callback)      # callback(payload: dict) -> "ok" | "reject"
    unbind_receipts()
    await open_reply(response_id, generation) -> writer with write(pcm) / aclose()
    await clear(generation)      # stop playback; a receipt with interrupted=True follows
"""

from __future__ import annotations

import asyncio
import math
from collections import deque
from dataclasses import dataclass, field


SAMPLE_RATE = 24000
BYTES_PER_SECOND = SAMPLE_RATE * 2


@dataclass
class _Item:
    item_id: str
    content_index: int
    size: int = 0


@dataclass
class _Reply:
    response_id: str
    generation: int
    items: dict[tuple[str, int], _Item] = field(default_factory=dict)
    sent_bytes: int = 0
    played_bytes: int = 0
    finished: asyncio.Future | None = None

    def truncations(self) -> list[tuple[str, int, int]]:
        remaining = self.played_bytes
        result = []
        for item in self.items.values():
            played = min(remaining, item.size)
            remaining -= played
            if played < item.size:
                result.append((item.item_id, item.content_index, played * 1000 // BYTES_PER_SECOND))
        return result


class AvatarPlayback:
    """Keep generated, delivered and played audio of each reply distinct.

    One reply streams to the avatar at a time; the next waits for the previous
    playback receipt. An interruption advances the generation so late audio of
    the interrupted reply can never enter the next one.
    """

    def __init__(self, channel, *, clear_timeout: float = 5.0):
        self.channel = channel
        self.clear_timeout = clear_timeout
        self.generation = 0
        self.replies: dict[str, _Reply] = {}
        self._active: _Reply | None = None
        self._writer = None
        self._lock = asyncio.Lock()
        self._interrupt_lock = asyncio.Lock()
        self._clear_receipt: asyncio.Future | None = None
        self._closed = False
        self._retired = deque(maxlen=256)
        self.channel.bind_receipts(self._playback_finished)

    def observe(self, response_id: str, item_id: str, content_index: int, size: int) -> int | None:
        """Register generated audio; return its generation, or None for a retired reply."""
        if self._closed:
            raise RuntimeError("Avatar playback is closed")
        if not response_id or not item_id or size < 0 or size % 2:
            raise ValueError("Expected identified PCM16 audio")
        if response_id in self._retired:
            return None
        reply = self.replies.setdefault(response_id, _Reply(response_id, self.generation))
        key = (item_id, content_index)
        item = reply.items.setdefault(key, _Item(*key))
        item.size += size
        return reply.generation

    async def write(self, response_id: str, generation: int, pcm: bytes) -> bool:
        if len(pcm) % 2:
            raise ValueError("PCM16 requires complete samples")
        async with self._lock:
            if self._closed or generation != self.generation:
                return False
            reply = self.replies[response_id]
            if self._active is not None and self._active is not reply:
                raise RuntimeError("Previous Avatar reply has not finished playback")
            if self._writer is None:
                reply.finished = asyncio.get_running_loop().create_future()
                self._active = reply
                self._writer = await self.channel.open_reply(response_id, generation)
            # Own immutable bytes; neither a later frame nor graph replay can
            # overwrite the data handed to the channel.
            await self._writer.write(bytes(pcm))
            reply.sent_bytes += len(pcm)
            return True

    async def finish(self, response_id: str, generation: int) -> bool:
        """End the reply's audio and wait until the avatar has played it."""
        async with self._lock:
            if self._closed or generation != self.generation:
                return False
            reply = self.replies.get(response_id)
            if reply is None:
                return False
            if self._active is not reply or self._writer is None:
                raise RuntimeError("Avatar reply ended without an open audio stream")
            await self._writer.aclose()
            self._writer = None
        # Pipecat cancellation may discard the waiting coroutine, but must not
        # cancel the future resolved by the independent receipt handler.
        timeout = reply.sent_bytes / BYTES_PER_SECOND + 30.0
        await asyncio.wait_for(asyncio.shield(reply.finished), timeout)
        async with self._lock:
            if generation != self.generation:
                return False
            self.replies.pop(response_id, None)
            self._retired.append(response_id)
            if self._active is reply:
                self._active = None
        return True

    def _playback_finished(self, payload) -> str:
        if self._closed:
            return "reject"
        try:
            position = float(payload["playback_position"])
            interrupted = payload["interrupted"]
            if not math.isfinite(position) or position < 0 or not isinstance(interrupted, bool):
                return "reject"
        except (KeyError, TypeError, ValueError):
            return "reject"
        reply = self._active
        if interrupted:
            if self._clear_receipt is None or self._clear_receipt.done():
                return "reject"
            self._clear_receipt.set_result(position)
        if reply is not None and reply.finished is not None and not reply.finished.done():
            reply.played_bytes = min(reply.sent_bytes, int(position * SAMPLE_RATE) * 2)
            reply.finished.set_result(interrupted)
        return "ok"

    async def interrupt(self) -> list[tuple[str, int, int]]:
        """Fence old audio, clear the avatar, and return what was not heard.

        Returns ``(item_id, content_index, audio_end_ms)`` for every item whose
        audio was not fully played, including replies that never reached the
        avatar (truncated to zero).
        """
        async with self._interrupt_lock:
            if self._closed or not self.replies:
                return []
            old = self.replies
            self._retired.extend(old)
            self.replies = {}
            self.generation += 1
            async with self._lock:
                if self._active is not None:
                    self._clear_receipt = asyncio.get_running_loop().create_future()
                    try:
                        if self._writer is not None:
                            await self._writer.aclose()
                            self._writer = None
                        async with asyncio.timeout(self.clear_timeout):
                            await self.channel.clear(self.generation)
                            await asyncio.shield(self._clear_receipt)
                    except BaseException:
                        # A missing receipt cannot be replaced with wall time or
                        # the generated duration. Stop using this playback.
                        self._closed = True
                        raise
                    finally:
                        self._clear_receipt = None
                        self._active = None
            return [truncation for reply in old.values() for truncation in reply.truncations()]

    async def aclose(self) -> None:
        self._closed = True
        self.generation += 1
        self.channel.unbind_receipts()
        try:
            async with self._lock:
                if self._writer is not None:
                    await self._writer.aclose()
        finally:
            self._writer = None
            for reply in self.replies.values():
                if reply.finished is not None and not reply.finished.done():
                    reply.finished.cancel()
            self.replies.clear()
            self._active = None
