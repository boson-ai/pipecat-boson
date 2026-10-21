# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Leave tool calls marked `executor: "server"` to the server. Server tools such as `web_search` arrive as ordinary `function_call` items; the service tracked them as calls to run locally and could answer a call the server had already answered. The server runs them and adds the output; the result is spoken in the next turn or in a reply the server starts after a pause.
- Cap `pipecat-ai` below 1.13. Pipecat 1.13.0 added a `direction` argument to `OpenAIRealtimeLLMService._handle_context` and made `get_llm_invocation_params` async, which broke the service on the first context frame and when sending conversation messages.
- Report the reply transcript as text that is being spoken. RTVI 2 clients, such as the Pipecat playground (prebuilt 1.13) and the voice UI kit, showed every assistant message empty because the text arrived as unspoken.
- Import correctly on `pipecat-ai` 1.8.0 and newer, which moved `assert_given` from `pipecat.services.settings` to `pipecat.utils.types`. The package previously failed to import on those versions. Older supported versions continue to work.
- Omit `temperature` from realtime session updates unless callers configure it explicitly, allowing the server's default to apply.

## [0.1.3] - 2026-08-03

### Fixed

- Excluded nltk 3.10.1 in the development environment via a uv resolution constraint; its import-security hook broke test collection with in-project virtual environments. No runtime changes for package consumers.

## [0.1.2] - 2026-07-31

### Changed

- Aligned the default Higgs Realtime and STT model identifiers with the public API.
- Renamed the public `modalities` option to the OpenAI-compatible `output_modalities`.
- Removed the unsupported transcription prompt setting from the public API.

### Fixed

- Send Pipecat tool results to Boson without JSON-encoding the already-serialized output a second time.

## [0.1.1] - 2026-07-30

### Added

- Added `BosonRealtimeLLMService` for using the Boson Realtime API from Pipecat pipelines.
- Added support for audio and text output modalities, server VAD, user transcripts, function calling, and text-only responses.
- Added a WebRTC RTVI example with user transcripts and function calling.
- Added unit tests for session configuration, realtime event handling, function-call follow-up responses, terminal session events, and package metadata.
