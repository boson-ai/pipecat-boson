# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Support `pipecat-ai` 1.13 and 1.14 alongside 1.4 to 1.12; the cap moves from `<1.13` to `<1.15`. Pipecat 1.13.0 added a `direction` argument to `OpenAIRealtimeLLMService._handle_context` and made `get_llm_invocation_params` async, which broke the service on the first context frame and when sending conversation messages. It also renamed `AggregationType` to `TextType`, which made the reply transcript print deprecation warnings. On 1.13, a tool's progress updates (results with `is_final=False`) are dropped with a warning: Pipecat sends them as system messages, which Boson rejects, and that ended the session.

## [0.1.4] - 2026-10-22

### Changed

- Merge a runtime settings update into the current session properties instead of replacing them, so fields the update does not mention keep their values. Setting a field to `None` still clears it, which now also works for `tools`. An update that changes nothing no longer sends a `session.update`. Pipecat's own settings update replaces the session properties wholesale.
- Raised the default server VAD `threshold` from `0.55` to `0.85`, matching the Boson Realtime server default. Quiet or distant speech is less likely to start a turn; pass a lower `threshold` in `turn_detection` for far-field input.

### Fixed

- Keep the session running when an application changes turn detection after the session has started. Boson fixes turn detection when the session starts and rejects later changes to it, which ended the session; the service now logs a warning and keeps the current turn detection.
- Keep the application driving turn taking after a runtime settings update. With server VAD disabled, any settings change that carried `session_properties` left Pipecat believing the server drove turns while Boson still had detection off, so neither side did: the session stopped responding and barge-in lost the start of the user's speech.
- Keep tools, and the Python handlers registered for them, across a runtime settings update. A settings change that carried `session_properties` previously advertised no tools on the next `session.update` and made Pipecat unregister every auto-registered tool handler. Tools set with an `LLMSetToolsFrame` now keep their handlers too; previously the next context removed them while Boson still offered the tools.
- Apply a turn-detection update made before the session starts field by field. Previously an update rebuilt the whole payload from Pipecat's model, dropping fields it cannot express (`min_speech_duration`, and `create_response` / `interrupt_response` for server VAD) and resetting fields the caller never mentioned to Pipecat's defaults. Re-enabling turn detection restores the configured payload, `turn_detection=None` keeps detection on as Pipecat defines it, and switching from a server VAD payload without a `type` to semantic VAD no longer carries server VAD fields over.
- Reject a `turn_detection` eagerness other than `"auto"` when constructing the service, and drop one that arrives in a runtime settings update. Boson accepts only `"auto"` and terminates the session for any other value, while Pipecat's semantic VAD model also offers `low`, `medium` and `high`.
- Stop an earlier `max_output_tokens` update from overriding a later `max_tokens` update.
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
