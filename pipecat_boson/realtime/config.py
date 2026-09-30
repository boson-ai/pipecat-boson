# ruff: noqa: CPY001
"""Configuration helpers for Boson's OpenAI-compatible realtime session payload."""

from __future__ import annotations

from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


SAMPLE_RATE = 24000
SUPPORTED_PCM_RATES = (8000, 16000, 24000, 48000)
DEFAULT_OUTPUT_MODALITIES = ["audio"]
VALID_OUTPUT_MODALITIES = ("text", "audio")
MAX_OUTPUT_TOKENS_LIMIT = 4096

UNSET = object()

DEFAULT_TURN_DETECTION: dict[str, Any] = {
    "type": "server_vad",
    "create_response": True,
    "interrupt_response": True,
    "prefix_padding_ms": 300,
    "silence_duration_ms": 500,
    "threshold": 0.55,
}


def resolve_pcm_rate(rate: int | None) -> int:
    """Return a PCM rate Boson accepts, falling back to 24 kHz."""
    return int(rate) if rate in SUPPORTED_PCM_RATES else SAMPLE_RATE


def normalize_ws_url(url: str, query_params: dict[str, str] | None = None) -> str:
    """Normalize an HTTP(S) or WebSocket URL and merge optional query params."""

    parsed = urlparse(url)
    scheme = parsed.scheme
    if scheme == "http":
        scheme = "ws"
    elif scheme == "https":
        scheme = "wss"

    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if query_params:
        query.update(query_params)
    return urlunparse((scheme, parsed.netloc, parsed.path, parsed.params, urlencode(query), parsed.fragment))


def resolve_output_modalities(output_modalities: list[str] | None) -> list[str]:
    """Resolve and validate Boson's single-output-modality realtime setting."""

    if output_modalities is None:
        return list(DEFAULT_OUTPUT_MODALITIES)
    if len(output_modalities) != 1 or output_modalities[0] not in VALID_OUTPUT_MODALITIES:
        raise ValueError(
            "output_modalities must be exactly one of ['text'] or ['audio'] "
            f"(got {output_modalities!r}); mixed and empty lists are not supported."
        )
    return list(output_modalities)


def normalize_max_output_tokens(value: int | Literal["inf"]) -> int | Literal["inf"]:
    """Clamp numeric max output tokens to Boson's current supported upper bound."""

    if value == "inf":
        return value
    return min(value, MAX_OUTPUT_TOKENS_LIMIT)


def build_turn_detection(value: Any = UNSET) -> dict[str, Any] | None:
    """Build the Boson turn-detection payload.

    Omitted means default server VAD. Explicit ``None`` or ``False`` disables
    server VAD, matching the existing LiveKit plugin while still allowing a
    Pipecat local-VAD/manual-commit mode.
    """

    if value is UNSET:
        return dict(DEFAULT_TURN_DETECTION)
    if value is None or value is False:
        return None
    return dict(value)


def build_input_audio_transcription(
    *,
    input_audio_transcription: Any = UNSET,
    model: str = "",
    language: str | None = None,
) -> dict[str, Any] | object:
    """Build the optional input audio transcription payload for session.update."""

    has_convenience_options = any(value is not None and value != "" for value in (model, language))
    if input_audio_transcription is UNSET and not has_convenience_options:
        return UNSET
    if input_audio_transcription is None and not has_convenience_options:
        return UNSET

    transcription = dict(input_audio_transcription) if input_audio_transcription not in (UNSET, None) else {}
    transcription.pop("temperature", None)
    transcription.pop("prompt", None)
    if model:
        transcription["model"] = model
    if language is not None:
        transcription["language"] = language
    return transcription


def input_audio_transcription_enabled(transcription: dict[str, Any] | object) -> bool:
    """Whether Boson will emit client-facing user transcript events."""

    if transcription is UNSET or not isinstance(transcription, dict):
        return False
    return bool(transcription.get("model"))


def build_noise_reduction(value: Any = UNSET) -> dict[str, Any] | None | object:
    """Build Boson's OpenAI-shaped input noise reduction payload."""

    if value is UNSET:
        return UNSET
    if value is None:
        return None
    if isinstance(value, str):
        return {"type": value}
    return dict(value)


def tool_choice_to_boson(tool_choice: Any) -> Any:
    """Convert Pipecat/OpenAI tool choice values to Boson's accepted payload."""

    if tool_choice is None:
        return "auto"
    if isinstance(tool_choice, str):
        return tool_choice
    function = tool_choice.get("function", {})
    name = function.get("name")
    if name:
        return {"type": "function", "name": name}
    return "auto"


def build_session_update_payload(
    *,
    event_id: str,
    model: str,
    voice: str,
    instructions: str,
    output_modalities: list[str],
    temperature: float | None,
    max_output_tokens: int | Literal["inf"],
    tool_choice: Any,
    tools: list[dict[str, Any]] | None,
    speed: float,
    turn_detection: dict[str, Any] | None,
    input_audio_transcription: dict[str, Any] | object,
    input_audio_noise_reduction: dict[str, Any] | None | object,
    truncation: Literal["auto", "disabled"] = "auto",
    input_sample_rate: int = SAMPLE_RATE,
) -> dict[str, Any]:
    """Build a complete Boson realtime ``session.update`` client event."""

    audio_input: dict[str, Any] = {
        "format": {"type": "audio/pcm", "rate": input_sample_rate},
        "turn_detection": turn_detection,
    }
    if input_audio_transcription is not UNSET:
        audio_input["transcription"] = input_audio_transcription
    if input_audio_noise_reduction is not UNSET:
        audio_input["noise_reduction"] = input_audio_noise_reduction

    payload = {
        "type": "session.update",
        "event_id": event_id,
        "session": {
            "type": "realtime",
            "model": model,
            "instructions": instructions,
            "output_modalities": list(output_modalities),
            "audio": {
                "input": audio_input,
                "output": {
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                    "voice": voice,
                    "speed": speed,
                },
            },
            "tools": tools or [],
            "tool_choice": tool_choice_to_boson(tool_choice),
            "max_output_tokens": normalize_max_output_tokens(max_output_tokens),
            "truncation": truncation,
        },
    }
    if temperature is not None:
        payload["session"]["temperature"] = temperature
    return payload
