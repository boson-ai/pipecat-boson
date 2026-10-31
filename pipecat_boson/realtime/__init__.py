# ruff: noqa: CPY001, N999
"""Realtime Pipecat service for Boson."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .append_relay import BosonMessagesAppendIntent as BosonMessagesAppendIntent
    from .append_relay import BosonMessagesAppendRelay as BosonMessagesAppendRelay
    from .llm import BosonRealtimeLLMService as BosonRealtimeLLMService


__all__ = [
    "BosonMessagesAppendIntent",
    "BosonMessagesAppendRelay",
    "BosonRealtimeLLMService",
]


def __getattr__(name: str):
    if name == "BosonRealtimeLLMService":
        from .llm import BosonRealtimeLLMService

        return BosonRealtimeLLMService
    if name == "BosonMessagesAppendIntent":
        from .append_relay import BosonMessagesAppendIntent

        return BosonMessagesAppendIntent
    if name == "BosonMessagesAppendRelay":
        from .append_relay import BosonMessagesAppendRelay

        return BosonMessagesAppendRelay
    raise AttributeError(name)
