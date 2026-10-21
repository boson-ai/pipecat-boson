# ruff: noqa: CPY001
from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace

import aiohttp
import pytest
import pytest_asyncio
from aiohttp import web
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    TTSAudioRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner
from websockets.asyncio.server import serve

from pipecat_boson.avatar import BosonAvatarService, DailyAvatarRoom
from pipecat_boson.avatar.api import BosonAvatarError
from pipecat_boson.realtime import BosonRealtimeLLMService


class HookedService(BosonRealtimeLLMService):
    """Captures client events and errors; frames go to the real push_frame."""

    def __init__(self, **kwargs):
        super().__init__(url="ws://localhost:1234/v1/realtime/", api_key="test-key", model="model-1", **kwargs)
        self.sent: list[dict] = []
        self.errors: list[tuple[str, bool]] = []

    async def _ws_send(self, realtime_message):
        self.sent.append(realtime_message)

    async def push_error(self, error_msg: str, exception: Exception | None = None, fatal: bool = False):
        self.errors.append((error_msg, fatal))

    async def start_processing_metrics(self):
        return None

    async def start_ttfb_metrics(self):
        return None

    async def stop_ttfb_metrics(self):
        return None

    async def stop_processing_metrics(self):
        return None

    async def stop_all_metrics(self):
        return None


class RecordingPlayback:
    def __init__(self, *, accept=True, truncations=(), error=None):
        self.accept = accept
        self.truncations = list(truncations)
        self.error = error
        self.tracked: list[tuple] = []
        self.interrupts = 0

    def track_audio(self, frame, *, response_id, item_id, content_index):
        self.tracked.append((response_id, item_id, content_index, len(frame.audio)))
        frame.metadata["recorded"] = True
        return self.accept

    async def interrupt(self):
        self.interrupts += 1
        if self.error is not None:
            raise self.error
        return self.truncations


@pytest.fixture
def pushed(monkeypatch):
    frames: list[Frame] = []

    async def capture(self, frame, direction=FrameDirection.DOWNSTREAM):
        frames.append(frame)

    monkeypatch.setattr(FrameProcessor, "push_frame", capture)
    return frames


def audio_delta(response_id: str, item_id: str, pcm: bytes) -> SimpleNamespace:
    return SimpleNamespace(
        response_id=response_id,
        item_id=item_id,
        content_index=0,
        output_index=0,
        delta=base64.b64encode(pcm).decode(),
    )


def truncate_events(llm: HookedService) -> list[tuple[str, int, int]]:
    return [
        (event["item_id"], event["content_index"], event["audio_end_ms"])
        for event in llm.sent
        if event.get("type") == "conversation.item.truncate"
    ]


@pytest.mark.asyncio
async def test_audio_playback_tracks_identity_and_truncates_after_generation_finished(pushed):
    playback = RecordingPlayback(truncations=[("item-1", 0, 375)])
    llm = HookedService(audio_playback=playback)
    llm._handle_evt_response_created(SimpleNamespace(response=SimpleNamespace(id="r1")))
    pcm = b"\1\0" * 24000
    await llm._handle_evt_audio_delta(audio_delta("r1", "item-1", pcm))
    await llm._handle_evt_response_done(SimpleNamespace(response=SimpleNamespace(id="r1", status="completed")))

    audio = [frame for frame in pushed if isinstance(frame, TTSAudioRawFrame)]
    assert [frame.audio for frame in audio] == [pcm]
    assert audio[0].metadata["recorded"]
    assert playback.tracked == [("r1", "item-1", 0, len(pcm))]

    # Generation has finished but the reply is still playing: the player, not
    # the cleared generation state, decides what to truncate.
    await llm._handle_evt_speech_started(SimpleNamespace())
    assert playback.interrupts == 1
    assert truncate_events(llm) == [("item-1", 0, 375)]


@pytest.mark.asyncio
async def test_audio_playback_drops_audio_it_rejects(pushed):
    llm = HookedService(audio_playback=RecordingPlayback(accept=False))
    await llm._handle_evt_audio_delta(audio_delta("r1", "item-1", b"\0\0" * 10))
    assert not [frame for frame in pushed if isinstance(frame, TTSAudioRawFrame)]


@pytest.mark.asyncio
async def test_bot_stopped_speaking_keeps_generation_identity_with_audio_playback(pushed):
    llm = HookedService(audio_playback=RecordingPlayback())
    await llm._handle_evt_audio_delta(audio_delta("r2", "item-2", b"\0\0" * 10))
    await llm._handle_bot_stopped_speaking()
    assert llm._current_audio_response is not None
    without_playback = HookedService()
    await without_playback._handle_evt_audio_delta(audio_delta("r2", "item-2", b"\0\0" * 10))
    await without_playback._handle_bot_stopped_speaking()
    assert without_playback._current_audio_response is None


@pytest.mark.asyncio
async def test_missing_playback_position_is_a_fatal_error(pushed):
    llm = HookedService(audio_playback=RecordingPlayback(error=TimeoutError("no receipt")))
    await llm._handle_evt_audio_delta(audio_delta("r1", "item-1", b"\0\0" * 10))
    await llm._handle_evt_speech_started(SimpleNamespace())
    assert not truncate_events(llm)
    assert len(llm.errors) == 1 and llm.errors[0][1] is True


class FakeAvatarAPI:
    """Session API plus an avatar that reports playback on request."""

    def __init__(self):
        self.created: list[tuple[dict, dict]] = []
        self.deleted: list[str] = []
        self.messages: list = []
        self.pcm: dict[str, bytearray] = {}
        self.create_status = 201
        self.audio_token = "audio-token"
        self.hold_receipts = False
        self.clear_position = 0.0
        self.sockets: list[web.WebSocketResponse] = []
        self._active = None
        self._last = None
        self.url = ""

    async def start(self):
        app = web.Application()
        app.router.add_post("/v1/sessions", self.create)
        app.router.add_delete("/v1/sessions/{sid}", self.delete)
        app.router.add_get("/v1/sessions/{sid}/audio", self.audio)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}/v1"

    async def stop(self):
        await self.runner.cleanup()

    async def create(self, request):
        body = await request.json()
        self.created.append((body, dict(request.headers)))
        if self.create_status != 201:
            error = {"error": {"code": "capacity_unavailable", "message": "No warm GPU", "retryable": True}}
            return web.json_response(error, status=self.create_status)
        audio_url = self.url.replace("http", "ws", 1) + "/sessions/s1/audio"
        return web.json_response(
            {
                "id": "s1",
                "object": "avatar.daily.session",
                "status": "active",
                "avatar_identity": body["transport"]["participant_identity"],
                "audio_input": {"type": "websocket", "url": audio_url, "token": self.audio_token},
            },
            status=201,
        )

    async def delete(self, request):
        self.deleted.append(request.match_info["sid"])
        return web.Response(status=204)

    async def audio(self, request):
        if request.headers.get("Authorization") != "Bearer audio-token":
            raise web.HTTPForbidden()
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        self.sockets.append(socket)
        async for message in socket:
            if message.type == aiohttp.WSMsgType.BINARY:
                self.pcm[self._active[0]] += message.data
                continue
            event = message.json()
            self.messages.append(event)
            if event["type"] == "start":
                self._active = (event["reply_id"], event["generation"])
                self.pcm.setdefault(event["reply_id"], bytearray())
            elif event["type"] == "end" and not self.hold_receipts:
                await self.receipt(len(self.pcm[event["reply_id"]]) / 48000, False)
            elif event["type"] == "interrupt":
                await self.receipt(self.clear_position, True)
        return socket

    async def receipt(self, position, interrupted):
        key = self._active or self._last
        self._last, self._active = key, None
        await self.sockets[-1].send_json(
            {
                "type": "playback_finished",
                "reply_id": key[0],
                "generation": key[1],
                "playback_position": position,
                "interrupted": interrupted,
                "clock_source": "test",
            }
        )


@pytest_asyncio.fixture
async def avatar_api():
    api = FakeAvatarAPI()
    await api.start()
    yield api
    await api.stop()


def make_avatar(api: FakeAvatarAPI, **kwargs) -> BosonAvatarService:
    return BosonAvatarService(
        api_url=api.url,
        api_key="avatar-key",
        avatar_id="local",
        room=DailyAvatarRoom(url="https://example.daily.co/room", token="avatar-room-token"),
        width=448,
        height=768,
        **kwargs,
    )


class Recorder(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.frames: list[Frame] = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        self.frames.append(frame)
        await self.push_frame(frame, direction)


async def eventually(predicate, timeout=5.0):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_pipeline_speaks_through_avatar_and_truncates_at_its_playback(avatar_api):
    requests: asyncio.Queue[dict] = asyncio.Queue()
    connected = asyncio.get_running_loop().create_future()

    async def boson(ws):
        connected.set_result(ws)
        await ws.send(json.dumps({"type": "session.created", "session": {"id": "test"}}))
        async for payload in ws:
            await requests.put(json.loads(payload))

    async def request_of_type(kind):
        async with asyncio.timeout(5):
            while True:
                request = await requests.get()
                if request["type"] == kind:
                    return request

    async def reply(ws, response_id, pcm):
        item = f"item-{response_id}"
        await ws.send(json.dumps({"type": "response.created", "response": {"id": response_id}}))
        await ws.send(
            json.dumps(
                {
                    "type": "response.output_audio.delta",
                    "response_id": response_id,
                    "item_id": item,
                    "content_index": 0,
                    "output_index": 0,
                    "delta": base64.b64encode(pcm).decode(),
                }
            )
        )
        await ws.send(
            json.dumps({"type": "response.done", "response": {"id": response_id, "status": "completed", "output": []}})
        )

    async with serve(boson, "127.0.0.1", 0) as endpoint:
        port = endpoint.sockets[0].getsockname()[1]
        avatar = make_avatar(avatar_api)
        llm = BosonRealtimeLLMService(url=f"ws://127.0.0.1:{port}", api_key="fake", audio_playback=avatar)
        recorder = Recorder()
        user, assistant = LLMContextAggregatorPair(LLMContext(), realtime_service_mode=True)
        worker = PipelineWorker(
            Pipeline([user, llm, avatar, recorder, assistant]),
            params=PipelineParams(audio_in_sample_rate=24000, audio_out_sample_rate=24000),
            enable_rtvi=False,
            idle_timeout_secs=None,
        )
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        run = asyncio.create_task(runner.run())
        try:
            await worker.queue_frame(LLMRunFrame())
            await request_of_type("response.create")
            ws = await connected
            body, headers = avatar_api.created[0]
            assert body["transport"] == {
                "type": "daily",
                "room_url": "https://example.daily.co/room",
                "participant_token": "avatar-room-token",
                "participant_identity": "boson-avatar",
                "audio_source": "websocket",
            }
            assert body["output"] == {"width": 448, "height": 768}
            assert headers["Authorization"] == "Bearer avatar-key" and headers["Idempotency-Key"]

            # Generation completes long before the avatar has played the reply.
            avatar_api.hold_receipts = True
            first = b"\1\0" * 48000
            await reply(ws, "r1", first)
            await eventually(lambda: any(m["type"] == "end" for m in avatar_api.messages))
            assert bytes(avatar_api.pcm["r1"]) == first
            assert any(isinstance(frame, BotStartedSpeakingFrame) for frame in recorder.frames)

            avatar_api.clear_position = 0.25
            await ws.send(json.dumps({"type": "input_audio_buffer.speech_started", "audio_start_ms": 0}))
            truncate = await request_of_type("conversation.item.truncate")
            assert (truncate["item_id"], truncate["audio_end_ms"]) == ("item-r1", 250)
            assert {"type": "interrupt", "generation": 1} in avatar_api.messages
            await eventually(lambda: any(isinstance(frame, BotStoppedSpeakingFrame) for frame in recorder.frames))

            # A typed turn goes straight to the realtime service: the user
            # aggregator would only forward a context snapshot.
            avatar_api.hold_receipts = False
            await llm.queue_frame(InterruptionFrame())
            await llm.queue_frame(LLMMessagesAppendFrame(messages=[{"role": "user", "content": "Hi"}], run_llm=True))
            await request_of_type("response.create")
            second = b"\2\0" * 2400
            await reply(ws, "r2", second)
            await eventually(lambda: bytes(avatar_api.pcm.get("r2", b"")) == second)
            await eventually(lambda: sum(isinstance(frame, BotStoppedSpeakingFrame) for frame in recorder.frames) >= 2)
            starts = [m for m in avatar_api.messages if m["type"] == "start"]
            assert starts == [
                {"type": "start", "reply_id": "r1", "generation": 0},
                {"type": "start", "reply_id": "r2", "generation": 1},
            ]
            await worker.queue_frame(EndFrame())
            await asyncio.wait_for(run, 5)
        finally:
            if not run.done():
                await runner.cancel()
                await asyncio.gather(run, return_exceptions=True)
    assert avatar_api.deleted == ["s1"]


@pytest.mark.asyncio
async def test_rejected_session_creation_raises_api_error(avatar_api):
    avatar_api.create_status = 503
    avatar = make_avatar(avatar_api)
    with pytest.raises(BosonAvatarError) as error:
        await avatar._open()
    assert (error.value.status, error.value.code) == (503, "capacity_unavailable")
    await avatar._close()
    assert avatar_api.deleted == []


@pytest.mark.asyncio
async def test_audio_channel_failure_ends_the_created_session(avatar_api):
    avatar_api.audio_token = "not-what-the-server-accepts"
    avatar = make_avatar(avatar_api)
    with pytest.raises(aiohttp.WSServerHandshakeError):
        await avatar._open()
    assert avatar_api.deleted == ["s1"]
    await avatar._close()


@pytest.mark.asyncio
async def test_avatar_ending_the_session_is_a_fatal_pipeline_error(avatar_api, monkeypatch):
    errors = []

    async def push_error(self, error_msg, exception=None, fatal=False):
        errors.append((error_msg, fatal))

    monkeypatch.setattr(BosonAvatarService, "push_error", push_error)
    avatar = make_avatar(avatar_api)
    await avatar._open()
    await eventually(lambda: avatar_api.sockets)
    await avatar_api.sockets[0].close()
    await eventually(lambda: errors)
    assert errors == [("Avatar session ended", True)]
    await avatar._close()
    assert avatar_api.deleted == ["s1"]


def test_width_and_height_go_together():
    with pytest.raises(ValueError):
        BosonAvatarService(
            api_url="http://localhost",
            api_key="key",
            avatar_id="local",
            room=DailyAvatarRoom(url="https://example.daily.co/room", token="token"),
            width=448,
        )
