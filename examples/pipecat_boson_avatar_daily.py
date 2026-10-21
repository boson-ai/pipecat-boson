# ruff: noqa: CPY001
"""Pipecat bot with Higgs Realtime voice and a Boson Avatar in a Daily room.

The bot hears the browser microphone through Pipecat's Daily transport and
publishes no audio itself. The avatar joins the same room as its own
participant and publishes the reply as synchronized audio and video. A minimal
page to join the room is served at ``http://HOST:PORT/``.

Without ``DAILY_ROOM_URL`` the example creates a private one-hour room and
deletes it on exit; a supplied room is never deleted. Every participant gets
its own meeting token and ``user_id``.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
import uuid
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
from pipecat_boson.realtime import BosonRealtimeLLMService


load_dotenv(override=True)

DAILY_API = "https://api.daily.co/v1"
AVATAR_ID, BOT_ID, VIEWER_ID = "boson-avatar", "pipecat-bot", "viewer"
INSTRUCTIONS = "You are a concise voice assistant. Greet the user briefly when the session starts."
PLAYER = Path(__file__).with_name("avatar_player.html")


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


async def meeting_token(http: aiohttp.ClientSession, room_name: str, user_id: str, user_name: str) -> str:
    properties = {"room_name": room_name, "user_id": user_id, "user_name": user_name, "exp": int(time.time()) + 3600}
    return (await daily_request(http, "POST", "/meeting-tokens", json={"properties": properties}))["token"]


async def serve_player(host: str, port: int) -> web.AppRunner:
    app = web.Application()
    app.router.add_get("/", lambda _: web.FileResponse(PLAYER))
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, host, port).start()
    return runner


async def run_bot(http: aiohttp.ClientSession, room_url: str, args: argparse.Namespace) -> None:
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
        avatar_id=os.environ["BOSON_AVATAR_ID"],
        room=DailyAvatarRoom(url=room_url, token=avatar_token, identity=AVATAR_ID),
        width=args.width,
        height=args.height,
        aiohttp_session=http,
    )
    llm = BosonRealtimeLLMService(
        url=os.environ["BOSON_REALTIME_URL"],
        api_key=os.environ["BOSON_API_KEY"],
        model=os.getenv("BOSON_REALTIME_MODEL", "higgs-realtime"),
        voice=os.getenv("BOSON_REALTIME_VOICE", "default"),
        instructions=os.getenv("BOSON_REALTIME_INSTRUCTIONS") or INSTRUCTIONS,
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
    runner = WorkerRunner()
    viewer: str | None = None

    @transport.event_handler("on_participant_joined")
    async def on_participant_joined(_transport, participant):
        nonlocal viewer
        if viewer is None and participant.get("info", {}).get("userId") not in (AVATAR_ID, BOT_ID):
            viewer = participant["id"]
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

    player = await serve_player(args.host, args.port)
    try:
        url = f"http://{args.host}:{args.port}/#" + urlencode({"url": room_url, "token": viewer_token})
        if args.player_file:
            fd = os.open(args.player_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as stream:
                stream.write(url + "\n")
            logger.info(f"Player URL written to {args.player_file}")
        else:
            logger.info(f"Open the player: {url}")
        await runner.add_workers(worker)
        await runner.run()
    finally:
        await player.cleanup()


async def main(args: argparse.Namespace) -> None:
    async with aiohttp.ClientSession() as http:
        room_url = os.getenv("DAILY_ROOM_URL")
        owned_room = None
        if not room_url:
            room = await daily_request(
                http, "POST", "/rooms", json={"privacy": "private", "properties": {"exp": int(time.time()) + 3600}}
            )
            room_url, owned_room = room["url"], room["name"]
        try:
            await run_bot(http, room_url, args)
        finally:
            if owned_room:
                await daily_request(http, "DELETE", f"/rooms/{owned_room}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1", help="Address for the player page.")
    parser.add_argument("--port", type=int, default=7861, help="Port for the player page.")
    parser.add_argument("--player-file", help="Write the player URL, which contains a room token, to this file.")
    parser.add_argument("--width", type=int, help="Avatar output width; requires --height.")
    parser.add_argument("--height", type=int, help="Avatar output height; requires --width.")
    asyncio.run(main(parser.parse_args()))
