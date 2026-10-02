# ruff: noqa: CPY001
from __future__ import annotations

import pytest

from pipecat_boson.realtime.config import (
    SUPPORTED_SEMANTIC_EAGERNESS,
    UNSET,
    build_input_audio_transcription,
    build_noise_reduction,
    build_session_update_payload,
    build_turn_detection,
    input_audio_transcription_enabled,
    normalize_max_output_tokens,
    normalize_ws_url,
    resolve_output_modalities,
)


def test_realtime_package_imports_without_optional_runtime_dependencies():
    from pipecat_boson import realtime

    assert "BosonRealtimeLLMService" in realtime.__all__


def test_normalize_ws_url_converts_http_and_merges_query_params():
    assert (
        normalize_ws_url(
            "https://voice.example/v1/realtime/?foo=bar",
            {"foo": "override", "trace": "abc"},
        )
        == "wss://voice.example/v1/realtime/?foo=override&trace=abc"
    )


def test_resolve_output_modalities_defaults_to_audio_and_accepts_text_only():
    assert resolve_output_modalities(None) == ["audio"]
    assert resolve_output_modalities(["audio"]) == ["audio"]
    assert resolve_output_modalities(["text"]) == ["text"]


@pytest.mark.parametrize("output_modalities", [[], ["audio", "text"], ["text", "audio"], ["video"]])
def test_resolve_output_modalities_rejects_invalid_lists(output_modalities):
    with pytest.raises(ValueError):
        resolve_output_modalities(output_modalities)


def test_turn_detection_defaults_to_server_vad_and_explicit_none_or_false_disables():
    default = build_turn_detection()
    assert default["type"] == "server_vad"
    assert default["create_response"] is True
    assert build_turn_detection(None) is None
    assert build_turn_detection(False) is None


def test_input_audio_transcription_event_gate_requires_model():
    assert input_audio_transcription_enabled(UNSET) is False
    assert input_audio_transcription_enabled({}) is False
    assert input_audio_transcription_enabled({"language": "zh"}) is False
    assert input_audio_transcription_enabled({"model": ""}) is False
    assert input_audio_transcription_enabled({"model": "whisper-1"}) is True


def test_build_input_audio_transcription_omits_none_without_convenience_fields():
    assert build_input_audio_transcription(input_audio_transcription=None) is UNSET
    assert build_input_audio_transcription(input_audio_transcription={"language": "zh"}) == {"language": "zh"}
    assert build_input_audio_transcription(input_audio_transcription=None, model="whisper-1") == {"model": "whisper-1"}
    assert build_input_audio_transcription(
        input_audio_transcription={"model": "whisper-1", "temperature": 0.2, "prompt": "Ignore me"}
    ) == {"model": "whisper-1"}


def test_noise_reduction_normalizes_string_and_dict():
    assert build_noise_reduction() is UNSET
    assert build_noise_reduction(None) is None
    assert build_noise_reduction("near_field") == {"type": "near_field"}
    assert build_noise_reduction({"type": "far_field"}) == {"type": "far_field"}


def test_normalize_max_output_tokens_caps_explicit_int_but_not_inf():
    assert normalize_max_output_tokens("inf") == "inf"
    assert normalize_max_output_tokens(1024) == 1024
    assert normalize_max_output_tokens(999999) == 4096


def test_build_session_update_payload_matches_openai_compatible_subset():
    payload = build_session_update_payload(
        event_id="evt_1",
        model="higgs-realtime",
        voice="voice_123",
        instructions="Be brief.",
        output_modalities=["text"],
        temperature=0.7,
        max_output_tokens=999999,
        tool_choice="auto",
        tools=[{"type": "function", "name": "get_weather", "description": "", "parameters": {}}],
        speed=1.25,
        turn_detection={"type": "server_vad"},
        input_audio_transcription={"model": "whisper-1"},
        input_audio_noise_reduction={"type": "near_field"},
        truncation="disabled",
    )

    session = payload["session"]
    assert payload["type"] == "session.update"
    assert session["model"] == "higgs-realtime"
    assert session["output_modalities"] == ["text"]
    assert session["audio"]["output"] == {
        "format": {"type": "audio/pcm", "rate": 24000},
        "voice": "voice_123",
        "speed": 1.25,
    }
    assert session["audio"]["input"]["noise_reduction"] == {"type": "near_field"}
    assert session["audio"]["input"]["transcription"] == {"model": "whisper-1"}
    assert session["truncation"] == "disabled"
    assert session["max_output_tokens"] == 4096
    assert session["temperature"] == 0.7
    assert "states" not in session
    assert "scripted_response" not in session
    assert "model" not in session["audio"]["output"]
    assert "temperature" not in session["audio"]["output"]


def test_build_session_update_payload_omits_unspecified_temperature():
    payload = build_session_update_payload(
        event_id="evt_1",
        model="higgs-realtime",
        voice="voice_123",
        instructions="Be brief.",
        output_modalities=["audio"],
        temperature=None,
        max_output_tokens="inf",
        tool_choice="auto",
        tools=None,
        speed=1.0,
        turn_detection=None,
        input_audio_transcription=UNSET,
        input_audio_noise_reduction=UNSET,
    )

    assert "temperature" not in payload["session"]


def test_build_session_update_payload_can_omit_or_clear_noise_reduction():
    base_kwargs = {
        "event_id": "evt_1",
        "model": "higgs-realtime",
        "voice": "voice_123",
        "instructions": "Be brief.",
        "output_modalities": ["audio"],
        "temperature": 0.7,
        "max_output_tokens": "inf",
        "tool_choice": "auto",
        "tools": None,
        "speed": 1.0,
        "turn_detection": None,
        "input_audio_transcription": UNSET,
        "truncation": "auto",
    }

    omitted = build_session_update_payload(input_audio_noise_reduction=UNSET, **base_kwargs)
    cleared = build_session_update_payload(input_audio_noise_reduction=None, **base_kwargs)

    assert "noise_reduction" not in omitted["session"]["audio"]["input"]
    assert cleared["session"]["audio"]["input"]["noise_reduction"] is None


def test_build_turn_detection_rejects_an_eagerness_boson_cannot_accept():
    # Boson terminates the session for any value other than "auto", so fail loudly at
    # construction rather than on the wire.
    for eagerness in ("low", "medium", "high"):
        with pytest.raises(ValueError, match="eagerness"):
            build_turn_detection({"type": "semantic_vad", "eagerness": eagerness})


def test_build_turn_detection_keeps_the_supported_eagerness():
    for eagerness in SUPPORTED_SEMANTIC_EAGERNESS:
        assert build_turn_detection({"type": "semantic_vad", "eagerness": eagerness}) == {
            "type": "semantic_vad",
            "eagerness": eagerness,
        }
