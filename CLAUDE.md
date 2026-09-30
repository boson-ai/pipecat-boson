# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`pipecat-boson` is a small, standalone Python package that exposes the Boson
Realtime API to [Pipecat](https://github.com/pipecat-ai/pipecat) pipelines as a
single speech-to-speech `LLMService`: `BosonRealtimeLLMService`. It is
intentionally isolated from Boson's backend; the only runtime contract is the
Boson realtime WebSocket protocol (Boson's OpenAI-compatible realtime interface).

## Commands

Uses `uv`. Run everything from the repository root.

```bash
uv sync --extra dev                       # install deps + dev tools into .venv
uv run --extra dev python -m pytest -q    # run the full test suite
uv run --extra dev python -m pytest tests/test_realtime_llm.py::<test_name>  # single test
uv run --extra dev ruff check .           # lint
uv run --extra dev ruff format .          # format (line-length 119, 4-space indent)
```

Running the WebRTC browser example needs the heavier `webrtc` extra (pulls in
`opencv-python`) and a `.env` (copy from `.env.example`):

```bash
uv run --extra webrtc --python 3.12 python examples/pipecat_boson_realtime_agent.py \
  -t webrtc --host 127.0.0.1 --port 7860
```

Tests use explicit `@pytest.mark.asyncio` markers (there is no `asyncio_mode`
config), so new async tests must carry the marker.

## Architecture

The whole package is two source modules under `pipecat_boson/realtime/`:

- **`config.py`** — pure, side-effect-free helpers that build Boson wire
  payloads and validate options. No Pipecat/websocket imports. This is where the
  Boson session-shape rules live: single output modality (`["audio"]` or
  `["text"]`, never mixed), 24 kHz PCM (`SAMPLE_RATE`), default server VAD
  (`DEFAULT_TURN_DETECTION`), OpenAI-shaped noise reduction, max-token clamping,
  and `session.update` assembly (`build_session_update_payload`). Prefer adding
  payload logic here and keeping it independently unit-testable
  (`tests/test_realtime_config.py`).

- **`llm.py`** — `BosonRealtimeLLMService`, which subclasses Pipecat's
  `OpenAIRealtimeLLMService` and adapts it to Boson's protocol variations.

### The key adaptation pattern in `llm.py`

Boson is OpenAI-compatible but not identical, so the service maintains a
**parallel set of `_boson_*` fields** (model, voice, instructions, modalities,
turn detection, transcription, etc.) alongside Pipecat's base `_settings`. There
are two representations on purpose:

- `_boson_*` values hold what actually goes on the Boson wire.
- Base Pipecat logic expects OpenAI conventions, so `__init__` translates via
  the `_to_pipecat_*` helpers. Notably: server-VAD-disabled is `None` on the
  Boson wire but must be `False` for Pipecat's base turn-detection logic.

When Pipecat pushes runtime setting changes (`_update_settings`), they are
mirrored back into the `_boson_*` fields by `_sync_boson_settings_from_pipecat`,
then a fresh `session.update` is sent. `_current_model` / `_current_instructions`
/ `_current_temperature` / `_current_tools` resolve "use the runtime override if
given, else the constructed default."

### Server events and response lifecycle

`llm.py` dispatches incoming server events through three module-level tables:
`_BOSON_SERVER_EVENT_HANDLERS` (type → method name), plus
`_BOSON_IGNORED_SERVER_EVENT_TYPES` and `_BOSON_TERMINAL_SESSION_EVENT_TYPES`.
When adding support for a new Boson event, register it in the appropriate table
rather than adding ad-hoc branches in `_dispatch_boson_server_event`.

Much of the complexity is **stale-response tracking**: Boson can send events for
responses that were cancelled or superseded (barge-in, idempotent
`response.cancel`, etc.). The service tags each `response.create` with a
`client_event_id` in `metadata` and tracks sets of pending / active / cancelled
response ids (`_boson_pending_response_client_event_ids`,
`_boson_active_response_ids`, `_boson_cancelled_response_ids`, and their
client-event-id counterparts). `_is_stale_response_event` /
`_is_stale_response_done` gate audio/text/transcript deltas so stale output is
dropped. When touching response handling, keep these sets consistent — they are
cleared together in `_mark_websocket_disconnected` and `_cancel_active_response`.

Terminal session events (`session.idle_timeout`, `session.max_duration_reached`)
set `_boson_terminal_session_event_type` so a subsequent websocket close is
logged as expected rather than surfaced as an error frame.

### Boson-vs-OpenAI wire quirks handled here

- Model is sent only in `session.update`, never as a `?model=` query param
  (the base class appends one; `__init__` overwrites `base_url` afterward).
- `response.create` payloads: `output_modalities` is rewritten to `modalities`
  by `_normalize_response_create_payload` before send.
- `SimpleNamespace`-based attribute access (`_to_attr`) is used to read incoming
  JSON events uniformly.

## Conventions

- Ruff config in `pyproject.toml`: line length 119, `preview = true`, includes
  the `CPY` (copyright) rule — source files carry `# ruff: noqa: CPY001` headers.
- Public entry point is `from pipecat_boson.realtime import BosonRealtimeLLMService`;
  it is lazily imported via `__getattr__` in `realtime/__init__.py` so importing
  the package doesn't pull Pipecat until the service is actually used.
- Targets `pipecat-ai>=1.4.0,<2` and Python 3.11+. CI tests the newest Pipecat plus
  the older versions listed in the `compat` job in `.github/workflows/ci.yml`.
- `assert_given` and `is_given` are imported with a `try`/`except` in `llm.py`
  because Pipecat 1.8.0 moved them. Keep the fallback while the minimum supported
  Pipecat is below 1.8.0.
