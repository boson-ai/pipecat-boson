# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Forward `response.output_audio_transcript.length` events to RTVI clients as server messages.

### Fixed

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
