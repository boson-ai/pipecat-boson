# ruff: noqa: CPY001, N999
"""Boson Avatar output for Pipecat pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .playback import AvatarPlayback as AvatarPlayback
    from .service import AvatarAudioOutput as AvatarAudioOutput
    from .service import BosonAvatarService as BosonAvatarService
    from .service import DailyAvatarRoom as DailyAvatarRoom


__all__ = ["AvatarAudioOutput", "AvatarPlayback", "BosonAvatarService", "DailyAvatarRoom"]


def __getattr__(name: str):
    if name == "AvatarPlayback":
        from .playback import AvatarPlayback

        return AvatarPlayback
    if name in {"AvatarAudioOutput", "BosonAvatarService", "DailyAvatarRoom"}:
        from . import service

        return getattr(service, name)
    raise AttributeError(name)
