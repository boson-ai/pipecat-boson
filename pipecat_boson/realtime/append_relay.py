# ruff: noqa: CPY001
"""Relay so LLMMessagesAppendFrame still reaches BosonRealtimeLLMService.

In a standard Pipecat pipeline the user context aggregator sits before the
realtime service. It consumes ``LLMMessagesAppendFrame``, updates the shared
context, and only pushes an ``LLMContextFrame`` when ``run_llm`` is true. The
service never sees the append frame or its ``run_llm`` flag.

Place ``BosonMessagesAppendRelay`` immediately before the user aggregator. It
emits a ``BosonMessagesAppendIntent`` that aggregators forward unchanged, then
forwards the original append so the context stays in sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pipecat.frames.frames import DataFrame, Frame, LLMMessagesAppendFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


@dataclass
class BosonMessagesAppendIntent(DataFrame):
    """Append intent that survives the user context aggregator.

    Parameters:
        messages: Context messages from the original ``LLMMessagesAppendFrame``.
            Prefer the same message objects so identity-based dedup works.
        run_llm: Whether Boson should create a response after sending the items.
    """

    messages: list[Any]
    run_llm: bool = False


class BosonMessagesAppendRelay(FrameProcessor):
    """Forward append frames to Boson while still updating the LLM context.

    Pipeline order::

        transport.input() → BosonMessagesAppendRelay() → user_aggregator → llm
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMMessagesAppendFrame) and direction == FrameDirection.DOWNSTREAM:
            await self.push_frame(
                BosonMessagesAppendIntent(
                    messages=frame.messages,
                    run_llm=bool(frame.run_llm),
                ),
                direction,
            )
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)
