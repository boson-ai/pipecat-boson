# ruff: noqa: CPY001
"""Pipecat bot with Higgs Realtime voice and a Boson Avatar in a Daily room.

The bot hears the browser microphone through Pipecat's Daily transport and
publishes no audio itself. The avatar joins the same room as its own
participant and publishes the reply as synchronized audio and video. Every
participant gets its own meeting token and ``user_id``.

By default the example runs one conversation and serves a page to join it at
``http://HOST:PORT/``. Without ``DAILY_ROOM_URL`` it creates a private room and
deletes it on exit; a supplied room is never deleted.

With ``--serve`` it is a bot launcher for your own web page instead:
``POST /sessions`` creates a room, starts a bot and answers once the avatar
is in the room, with ``{"id", "room_url", "token", "avatar_session_id"}`` for
the page to join. ``DELETE /sessions/{id}`` stops the bot. Each bot leaves
when the viewer leaves, or after a minute if nobody joins, and deletes its
room. Optional JSON fields: ``avatar_id``, ``voice``, ``instructions``,
``width`` and ``height``.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode, urlsplit

import aiohttp
from aiohttp import web
from dotenv import load_dotenv
from loguru import logger
from pipecat.frames.frames import (
    InterruptionFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    OutputTransportMessageUrgentFrame,
    TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.workers.runner import WorkerRunner

from pipecat_boson.avatar import BosonAvatarService, DailyAvatarRoom
from pipecat_boson.avatar.api import BosonAvatarError
from pipecat_boson.realtime import BosonRealtimeLLMService


load_dotenv(override=True)

DAILY_API = "https://api.daily.co/v1"
AVATAR_ID, BOT_ID, VIEWER_ID = "boson-avatar", "pipecat-bot", "viewer"
INSTRUCTIONS = "You are a concise voice assistant. Greet the user briefly when the session starts."
PLAYER = Path(__file__).with_name("avatar_player.html")
AVATAR_START_TIMEOUT_S = 90
VIEWER_JOIN_TIMEOUT_S = 60


class ReplyTranscripts(FrameProcessor):
    """Send the bot's reply text to the browser page as Daily app messages."""

    def __init__(self):
        super().__init__()
        self._reply_id: str | None = None

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame):
            self._reply_id = uuid.uuid4().hex
        elif isinstance(frame, TTSTextFrame) and self._reply_id:
            message = {"type": "transcript", "role": "assistant", "text": frame.text, "reply_id": self._reply_id}
            await self.push_frame(OutputTransportMessageUrgentFrame(message=message))
        await self.push_frame(frame, direction)


async def daily_request(http: aiohttp.ClientSession, method: str, path: str, **kwargs) -> dict:
    headers = {"Authorization": f"Bearer {os.environ['DAILY_API_KEY']}"}
    async with http.request(method, f"{DAILY_API}{path}", headers=headers, **kwargs) as response:
        if method == "DELETE" and response.status == 404:
            return {}  # Already gone, for example expired.
        response.raise_for_status()
        return await response.json()


async def create_room(http: aiohttp.ClientSession) -> dict:
    properties = {"exp": int(time.time()) + 3600, "eject_at_room_exp": True}
    return await daily_request(http, "POST", "/rooms", json={"privacy": "private", "properties": properties})


async def meeting_token(http: aiohttp.ClientSession, room_name: str, user_id: str, user_name: str) -> str:
    properties = {"room_name": room_name, "user_id": user_id, "user_name": user_name, "exp": int(time.time()) + 3600}
    return (await daily_request(http, "POST", "/meeting-tokens", json={"properties": properties}))["token"]


@dataclass
class Bot:
    runner: WorkerRunner
    avatar: BosonAvatarService
    viewer_token: str
    viewer_joined: asyncio.Event = field(default_factory=asyncio.Event)
    errors: list[Exception] = field(default_factory=list)


async def create_bot(
    http: aiohttp.ClientSession,
    room_url: str,
    *,
    avatar_id: str,
    voice: str,
    instructions: str,
    width: int | None = None,
    height: int | None = None,
    handle_signals: bool = True,
) -> Bot:
    """Mint one token per participant and build the bot for ``room_url``; ``bot.runner.run()`` starts it."""
    room_name = urlsplit(room_url).path.strip("/")
    avatar_token = await meeting_token(http, room_name, AVATAR_ID, "Boson Avatar")
    bot_token = await meeting_token(http, room_name, BOT_ID, "Pipecat bot")
    viewer_token = await meeting_token(http, room_name, VIEWER_ID, "Viewer")

    transport = DailyTransport(
        room_url,
        bot_token,
        "Pipecat bot",
        # The avatar publishes the bot's audio and video; the bot publishes nothing.
        DailyParams(
            audio_in_enabled=True,
            audio_in_user_tracks=True,
            audio_out_enabled=False,
            microphone_out_enabled=False,
            camera_out_enabled=False,
        ),
    )
    avatar = BosonAvatarService(
        api_url=os.environ["BOSON_AVATAR_API_URL"],
        api_key=os.getenv("BOSON_AVATAR_API_KEY") or os.environ["BOSON_API_KEY"],
        avatar_id=avatar_id,
        room=DailyAvatarRoom(url=room_url, token=avatar_token, identity=AVATAR_ID),
        width=width,
        height=height,
        aiohttp_session=http,
    )
    llm = BosonRealtimeLLMService(
        url=os.environ["BOSON_REALTIME_URL"],
        api_key=os.environ["BOSON_API_KEY"],
        model=os.getenv("BOSON_REALTIME_MODEL", "higgs-realtime"),
        voice=voice,
        instructions=instructions,
        audio_playback=avatar,
    )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(LLMContext(), realtime_service_mode=True)
    worker = PipelineWorker(
        Pipeline(
            [
                transport.input(),
                user_aggregator,
                llm,
                avatar,
                ReplyTranscripts(),
                transport.output(),
                assistant_aggregator,
            ]
        ),
        params=PipelineParams(audio_in_sample_rate=24000, audio_out_sample_rate=24000),
        idle_timeout_secs=None,
    )
    runner = WorkerRunner(handle_sigint=handle_signals, handle_sigterm=handle_signals)
    bot = Bot(runner=runner, avatar=avatar, viewer_token=viewer_token)
    viewer: str | None = None

    @worker.event_handler("on_pipeline_error")
    async def on_pipeline_error(_worker, frame):
        if frame.exception is not None:
            bot.errors.append(frame.exception)

    @transport.event_handler("on_participant_joined")
    async def on_participant_joined(_transport, participant):
        nonlocal viewer
        if viewer is None and participant.get("info", {}).get("userId") not in (AVATAR_ID, BOT_ID):
            viewer = participant["id"]
            bot.viewer_joined.set()
            await worker.queue_frame(LLMRunFrame())

    @transport.event_handler("on_participant_left")
    async def on_participant_left(_transport, participant, _reason):
        if participant["id"] == viewer:
            await runner.cancel()

    @transport.event_handler("on_app_message")
    async def on_app_message(_transport, message, sender):
        text = str(message.get("text", "")).strip() if isinstance(message, dict) else ""
        if sender == viewer and message.get("type") == "chat" and text:
            # A typed turn goes straight to the realtime service. Interrupting
            # first stops the avatar and truncates the reply at what was heard.
            await llm.queue_frame(InterruptionFrame())
            await llm.queue_frame(LLMMessagesAppendFrame(messages=[{"role": "user", "content": text}], run_llm=True))

    @llm.event_handler("on_session_terminated")
    async def on_session_terminated(_service, _event_type, _event):
        await runner.cancel()

    await runner.add_workers(worker)
    return bot


def default_options() -> dict:
    return {
        "avatar_id": os.environ["BOSON_AVATAR_ID"],
        "voice": os.getenv("BOSON_REALTIME_VOICE", "default"),
        "instructions": os.getenv("BOSON_REALTIME_INSTRUCTIONS") or INSTRUCTIONS,
    }


async def run_one(http: aiohttp.ClientSession, args: argparse.Namespace) -> None:
    room_url, owned_room = os.getenv("DAILY_ROOM_URL"), None
    if not room_url:
        room = await create_room(http)
        room_url, owned_room = room["url"], room["name"]
    try:
        bot = await create_bot(http, room_url, **default_options(), width=args.width, height=args.height)
        app = web.Application()
        app.router.add_get("/", lambda _: web.FileResponse(PLAYER))
        page = web.AppRunner(app)
        await page.setup()
        await web.TCPSite(page, args.host, args.port).start()
        try:
            url = f"http://{args.host}:{args.port}/#" + urlencode({"url": room_url, "token": bot.viewer_token})
            if args.player_file:
                fd = os.open(args.player_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "w") as stream:
                    stream.write(url + "\n")
                logger.info(f"Player URL written to {args.player_file}")
            else:
                logger.info(f"Open the player: {url}")
            await bot.runner.run()
        finally:
            await page.cleanup()
    finally:
        if owned_room:
            await daily_request(http, "DELETE", f"/rooms/{owned_room}")


async def serve(http: aiohttp.ClientSession, args: argparse.Namespace) -> None:
    bots: dict[str, Bot] = {}
    tasks: set[asyncio.Task] = set()

    async def run_bot(session_id: str, bot: Bot, room_name: str) -> None:
        async def leave_if_nobody_joins():
            try:
                await asyncio.wait_for(bot.viewer_joined.wait(), VIEWER_JOIN_TIMEOUT_S)
            except TimeoutError:
                logger.info(f"Session {session_id}: nobody joined; stopping")
                await bot.runner.cancel()

        watchdog = asyncio.create_task(leave_if_nobody_joins())
        try:
            await bot.runner.run()
        finally:
            watchdog.cancel()
            bots.pop(session_id, None)
            await daily_request(http, "DELETE", f"/rooms/{room_name}")
            logger.info(f"Session {session_id} ended")

    async def start(request: web.Request) -> web.Response:
        options = {**default_options(), **(await request.json() if request.can_read_body else {})}
        room = await create_room(http)
        try:
            bot = await create_bot(
                http,
                room["url"],
                **{key: options[key] for key in ("avatar_id", "voice", "instructions")},
                width=options.get("width"),
                height=options.get("height"),
                handle_signals=False,
            )
        except BaseException:
            await daily_request(http, "DELETE", f"/rooms/{room['name']}")
            raise
        session_id = uuid.uuid4().hex
        started = asyncio.get_running_loop().create_future()

        @bot.avatar.event_handler("on_session_started")
        async def on_session_started(_avatar, avatar_session_id):
            if not started.done():
                started.set_result(avatar_session_id)

        task = asyncio.create_task(run_bot(session_id, bot, room["name"]))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        await asyncio.wait({started, task}, timeout=AVATAR_START_TIMEOUT_S, return_when=asyncio.FIRST_COMPLETED)
        if not started.done():
            await bot.runner.cancel()
            error = next((e for e in bot.errors if isinstance(e, BosonAvatarError)), None)
            status, code = (error.status, error.code) if error else (502, "avatar_start_failed")
            message = str(error) if error else "The avatar did not join the room; see the launcher log"
            return web.json_response({"error": {"code": code, "message": message}}, status=status)
        bots[session_id] = bot
        payload = {
            "id": session_id,
            "room_url": room["url"],
            "token": bot.viewer_token,
            "avatar_session_id": started.result(),
        }
        return web.json_response(payload, status=201)

    async def stop(request: web.Request) -> web.Response:
        bot = bots.get(request.match_info["session_id"])
        if bot is not None:
            await bot.runner.cancel()
        return web.Response(status=204)

    app = web.Application()
    app.router.add_post("/sessions", start)
    app.router.add_delete("/sessions/{session_id}", stop)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, args.host, args.port).start()
    logger.info(f"Bot launcher listening on http://{args.host}:{args.port}")
    try:
        await asyncio.Event().wait()
    finally:
        for bot in list(bots.values()):
            await bot.runner.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await runner.cleanup()


async def main(args: argparse.Namespace) -> None:
    async with aiohttp.ClientSession() as http:
        await (serve(http, args) if args.serve else run_one(http, args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--serve", action="store_true", help="Run as a bot launcher for your own page.")
    parser.add_argument("--host", default="127.0.0.1", help="Address for the page or the launcher.")
    parser.add_argument("--port", type=int, help="Port for the page (default 7861) or the launcher (default 7860).")
    parser.add_argument("--player-file", help="Write the player URL, which contains a room token, to this file.")
    parser.add_argument("--width", type=int, help="Avatar output width; requires --height.")
    parser.add_argument("--height", type=int, help="Avatar output height; requires --width.")
    arguments = parser.parse_args()
    arguments.port = arguments.port or (7860 if arguments.serve else 7861)
    asyncio.run(main(arguments))
