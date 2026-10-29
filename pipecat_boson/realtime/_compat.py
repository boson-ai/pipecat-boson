# ruff: noqa: CPY001
"""Differences between the Pipecat versions this package supports (pipecat-ai 1.4 to 1.14).

Each one is detected once, at import, from Pipecat itself rather than from its version
number, so ``llm.py`` never branches on a version. No side effects beyond the imports.
"""

from __future__ import annotations

import inspect
from typing import Any

from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService


try:  # pipecat-ai 1.8 moved these from pipecat.services.settings.
    from pipecat.utils.types import assert_given, is_given
except ImportError:
    from pipecat.services.settings import assert_given, is_given

try:  # pipecat-ai 1.13 renamed AggregationType to TextType, and aggregated_by to text_type.
    from pipecat.frames.frames import TextType
except ImportError:
    from pipecat.utils.text.base_text_aggregator import AggregationType

    SENTENCE_TEXT: dict[str, Any] = {"aggregated_by": AggregationType.SENTENCE}
else:
    SENTENCE_TEXT = {"text_type": TextType.SENTENCE}


# pipecat-ai 1.13 tells _handle_context which way the context frame was travelling.
HANDLE_CONTEXT_TAKES_DIRECTION = "direction" in inspect.signature(OpenAIRealtimeLLMService._handle_context).parameters
__all__ = [
    "HANDLE_CONTEXT_TAKES_DIRECTION",
    "SENTENCE_TEXT",
    "assert_given",
    "forget_answered_call",
    "is_given",
    "resolved",
]


async def resolved(value: Any) -> Any:
    """``value``, awaited if it is awaitable; the realtime adapter is async from pipecat-ai 1.13."""

    return await value if inspect.isawaitable(value) else value


def forget_answered_call(service: Any, tool_call_id: str) -> None:
    """Drop an answered call from pipecat-ai 1.13's record of calls."""

    open_calls = getattr(service, "_open_function_calls", None)
    if open_calls is not None:
        open_calls.discard(tool_call_id)
