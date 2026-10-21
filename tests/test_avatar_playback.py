# ruff: noqa: CPY001
from __future__ import annotations

import asyncio

import pytest

from pipecat_boson.avatar.playback import AvatarPlayback


class Writer:
    def __init__(self, channel, response_id, generation):
        self.channel = channel
        self.key = (response_id, generation)
        self.chunks: list[bytes] = []
        self.closed = False

    async def write(self, data):
        self.channel.write_started.set()
        if self.channel.write_gate is not None:
            await self.channel.write_gate.wait()
        self.chunks.append(bytes(data))

    async def aclose(self):
        self.closed = True


class Channel:
    """In-memory playback channel; tests deliver the avatar's receipts explicitly."""

    def __init__(self):
        self.callback = None
        self.writers: list[Writer] = []
        self.clears: list[int] = []
        self.clear_position = 0.0
        self.clear_gate = None
        self.write_gate = None
        self.write_started = asyncio.Event()
        self.send_clear_receipt = True

    def bind_receipts(self, callback):
        self.callback = callback

    def unbind_receipts(self):
        self.callback = None

    async def open_reply(self, response_id, generation):
        writer = Writer(self, response_id, generation)
        self.writers.append(writer)
        return writer

    async def clear(self, generation):
        self.clears.append(generation)
        if self.clear_gate is not None:
            await self.clear_gate.wait()
        if self.send_clear_receipt:
            self.receipt(self.clear_position, True)

    def receipt(self, position, interrupted):
        return self.callback({"playback_position": position, "interrupted": interrupted})


@pytest.fixture
def playback():
    channel = Channel()
    return AvatarPlayback(channel, clear_timeout=0.05), channel


def observe(playback, response="r1", item="i1", size=48000):
    return playback.observe(response, item, 0, size)


@pytest.mark.asyncio
async def test_reply_audio_is_delivered_once_and_finish_waits_for_playback(playback):
    playback, channel = playback
    pcm = bytes(range(256)) * 8
    generation = observe(playback, size=len(pcm))
    assert await playback.write("r1", generation, bytearray(pcm))
    finish = asyncio.create_task(playback.finish("r1", generation))
    await asyncio.sleep(0)
    assert not finish.done()
    assert channel.writers[0].closed
    assert b"".join(channel.writers[0].chunks) == pcm
    assert channel.receipt(len(pcm) / 48000, False) == "ok"
    assert await finish
    assert not playback.replies
    await playback.aclose()
    assert channel.callback is None


@pytest.mark.asyncio
async def test_interruption_after_generation_finished_truncates_at_receipt(playback):
    playback, channel = playback
    generation = observe(playback, size=48000 * 4)
    await playback.write("r1", generation, b"\0\0" * 24000 * 4)
    finish = asyncio.create_task(playback.finish("r1", generation))
    await asyncio.sleep(0)
    channel.clear_position = 0.25
    assert await playback.interrupt() == [("i1", 0, 250)]
    assert not await finish
    assert not await playback.write("r1", generation, b"\0\0")
    assert not await playback.finish("r1", generation)
    assert observe(playback, size=2) is None  # A late delta of the interrupted reply.
    assert await playback.interrupt() == []
    assert channel.clears == [1]


@pytest.mark.asyncio
async def test_multiple_items_and_undelivered_replies_are_truncated(playback):
    playback, channel = playback
    generation = observe(playback, item="first")
    observe(playback, item="second")
    observe(playback, response="r2", item="future")
    await playback.write("r1", generation, b"\0\0" * 48000)
    channel.clear_position = 1.25
    assert await playback.interrupt() == [("second", 0, 250), ("future", 0, 0)]


@pytest.mark.asyncio
async def test_interruption_before_first_output_fences_queued_audio(playback):
    playback, channel = playback
    generation = observe(playback)
    assert await playback.interrupt() == [("i1", 0, 0)]
    assert not await playback.write("r1", generation, b"\0\0")
    assert channel.clears == []
    assert not channel.writers


@pytest.mark.asyncio
async def test_next_reply_waits_for_the_clear_receipt(playback):
    playback, channel = playback
    generation = observe(playback, size=2)
    await playback.write("r1", generation, b"\0\0")
    channel.clear_gate = asyncio.Event()
    clear = asyncio.create_task(playback.interrupt())
    await asyncio.sleep(0)
    new_generation = observe(playback, response="r2", item="i2", size=2)
    new_write = asyncio.create_task(playback.write("r2", new_generation, b"\1\0"))
    await asyncio.sleep(0)
    assert not new_write.done()
    channel.clear_gate.set()
    await clear
    assert await new_write
    assert [writer.closed for writer in channel.writers] == [True, False]
    assert channel.writers[1].chunks == [b"\1\0"]


@pytest.mark.asyncio
async def test_inflight_write_is_cleared_before_its_successor(playback):
    playback, channel = playback
    generation = observe(playback, size=2)
    channel.write_gate = asyncio.Event()
    write = asyncio.create_task(playback.write("r1", generation, b"\0\0"))
    await channel.write_started.wait()
    clear = asyncio.create_task(playback.interrupt())
    await asyncio.sleep(0)
    channel.write_gate.set()
    await write
    assert await clear == [("i1", 0, 0)]
    assert channel.clears == [1]


@pytest.mark.asyncio
async def test_cancelled_finish_does_not_cancel_the_receipt(playback):
    playback, _ = playback
    generation = observe(playback, size=2)
    await playback.write("r1", generation, b"\0\0")
    finish = asyncio.create_task(playback.finish("r1", generation))
    await asyncio.sleep(0)
    finish.cancel()
    with pytest.raises(asyncio.CancelledError):
        await finish
    assert not playback.replies["r1"].finished.cancelled()
    assert await playback.interrupt() == [("i1", 0, 0)]


@pytest.mark.asyncio
async def test_missing_clear_receipt_fails_instead_of_guessing_progress(playback):
    playback, channel = playback
    generation = observe(playback, size=2)
    await playback.write("r1", generation, b"\0\0")
    channel.send_clear_receipt = False
    with pytest.raises(TimeoutError):
        await playback.interrupt()
    with pytest.raises(RuntimeError, match="closed"):
        observe(playback, response="r2")
    await playback.aclose()


@pytest.mark.asyncio
async def test_invalid_receipts_are_rejected(playback):
    playback, channel = playback
    generation = observe(playback, size=2)
    await playback.write("r1", generation, b"\0\0")
    assert channel.receipt(float("nan"), False) == "reject"
    assert channel.receipt(-1, False) == "reject"
    assert channel.receipt(0, True) == "reject"  # No clear is pending.
    assert channel.callback({"interrupted": False}) == "reject"
    assert not playback.replies["r1"].finished.done()
    await playback.aclose()


@pytest.mark.asyncio
async def test_second_reply_cannot_overlap_unfinished_playback(playback):
    playback, _ = playback
    first = observe(playback, size=2)
    second = observe(playback, response="r2", item="i2", size=2)
    await playback.write("r1", first, b"\0\0")
    with pytest.raises(RuntimeError, match="has not finished"):
        await playback.write("r2", second, b"\0\0")
