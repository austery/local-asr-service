---
specId: SPEC-014
title: Apple SpeechAnalyzer Local ASR Integration and Diarization Boundary Probe
status: 📝 草案 (Draft)
priority: P1 - Core Feature
creationDate: 2026-07-03
lastUpdateDate: 2026-07-04
owner: User (AI-Assisted)
relatedSpecs:
  - SPEC-011
  - SPEC-012
  - SPEC-013
tags:
  - asr
  - apple-silicon
  - speechanalyzer
  - speechtranscriber
  - dictation
  - diarization
  - swift-sidecar
  - openai-compatible-api
---

# SPEC-014: Apple SpeechAnalyzer Local ASR Integration and Diarization Boundary Probe

## 1. Goal

Add an Apple-native local ASR path to `local-asr-service` by wrapping Apple SpeechAnalyzer / SpeechTranscriber / DictationTranscriber behind a Swift sidecar worker, while keeping the existing Python service as the OpenAI-compatible gateway.

The immediate goal is to create a production-candidate local dictation / transcription baseline for Apple Silicon machines running macOS 26+:

```text
audio file or live microphone stream
  -> apple-speech-worker Swift sidecar
  -> Apple SpeechAnalyzer + SpeechTranscriber or DictationTranscriber
  -> normalized transcript JSON
  -> existing /v1/audio/transcriptions response contract
```

The diarization goal is deliberately narrower:

```text
audio
  -> Apple SpeechAnalyzer transcript + timing metadata
  -> optional existing diarization runtime, e.g. Sortformer, if requested
  -> speaker-labeled segments only if coverage and consistency gates pass
```

Apple SpeechAnalyzer must not be represented as a native speaker diarization solution unless real API/runtime evidence proves speaker labels are available. The initial Apple path is an ASR engine, not a diarized-ASR engine.

## 2. Background

### 2.1 Current project pain

The user's real pain is not only transcription accuracy. The harder product problem is multi-person dialogue: producing readable, trustworthy speaker-labeled transcripts.

Previous work already established that ASR and diarization should be treated as separate tasks and aligned on the audio timeline. SPEC-011's architecture was valid, but FireRed was a bad fit because model strength was confused with Apple Silicon runtime viability. The lesson is:

> strongest model ≠ strongest usable model on this hardware.

The previous Qwen3 + ForcedAligner + Sortformer path was implemented as an explicit experiment, but real meeting probes showed that it was not production-reliable: speaker-labeled output lost transcript coverage, fragmented heavily, produced Unknown segments, and had poor resource tradeoffs on a 64 GB Apple Silicon machine.

### 2.2 Apple SpeechAnalyzer opportunity

Apple's new SpeechAnalyzer API is exposed to apps on iOS 26 / macOS 26 class systems. It provides an on-device speech-to-text model via modules such as SpeechTranscriber and DictationTranscriber.

Relevant Apple API properties to verify and use:

- SpeechAnalyzer manages an analysis session.
- Transcriber modules receive audio buffers and return text plus metadata asynchronously.
- Results are ordered by audio timeline timecodes.
- SpeechTranscriber is designed for long-form and distant audio such as lectures, meetings, and conversations.
- The model is on-device and its assets are managed by the system through AssetInventory.
- The model is retained in system storage and runs outside the app memory space.
- Volatile results can provide low-latency interim text; finalized results should be used for durable transcript output.
- `audioTimeRange` should be requested and empirically validated for segment alignment.
- DictationTranscriber can bias recognition using short contextual phrases through AnalysisContext / contextualStrings.

This is a better fit for local dictation than Cohere Transcribe because the user has already validated Apple SpeechAnalyzer indirectly through Spokenly in daily Chinese/English mixed dictation.

### 2.3 `speech-swift` reference boundary

`/Users/leipeng/Documents/Projects/speech-swift` is a useful reference project, but it is not the implementation template for this spec.

What this spec should borrow:

- task separation: ASR, forced alignment, and diarization are separate contracts rather than one overloaded transcription model
- model registry thinking: aliases should resolve to explicit runtime capabilities and requestable surfaces
- OpenAI-compatible API discipline: preserve the client-facing `/v1/audio/transcriptions` contract and keep error responses clear
- audio handling lessons: decode/resample/format conversion belongs behind a narrow worker or adapter boundary

What this spec must not copy:

- do not turn `local-asr-service` into a Swift multi-model framework
- do not add a second HTTP server for the first implementation
- do not import `speech-swift` as a dependency just to reach Apple SpeechAnalyzer
- do not copy CLI behavior where human logs and JSON share stdout; this worker's stdout is reserved for machine-parseable JSON and logs go to stderr
- do not copy `speech-swift` diarization claims into the Apple SpeechAnalyzer path; Apple remains ASR-only until runtime evidence proves otherwise

No direct `SpeechAnalyzer`, `SpeechTranscriber`, or `DictationTranscriber` implementation was found in the local `speech-swift` checkout during spec review, so Apple API feasibility must be proven through a standalone Swift probe in this repo's own worker boundary.

## 3. Design Decision

### Chosen approach

Implement a small Swift sidecar binary and call it from the existing Python service.

```text
FastAPI / Python service
  -> AppleSpeechEngine adapter
  -> apple-speech-worker Swift binary
  -> Apple Speech framework
  -> JSON response
  -> normalize into existing OpenAI-compatible contract
```

### Rationale

1. Apple SpeechAnalyzer is a Swift-first system API; a Swift worker is cleaner than trying to bridge the new async API through Python.
2. The current Python service should remain the local gateway: API compatibility, queueing, routing, model selection, lifecycle isolation, and response normalization.
3. A sidecar preserves the project boundary: do not rewrite the whole service in Swift and do not turn the service into a speech model framework.
4. Apple model assets run outside the app memory space, which may reduce the memory lifecycle pain currently seen with MLX / PyTorch workers.
5. Apple SpeechAnalyzer is a realistic local default for dictation and short/medium transcription, but it must not be treated as native diarization unless validated.

### Alternatives considered

| Alternative | Pros | Cons | Decision |
|---|---|---|---|
| Direct Python bridge to SpeechAnalyzer via PyObjC | Single-language service | New Swift async APIs, asset management, and audio buffer streaming are likely brittle from Python | ❌ Rejected for first implementation |
| Full Swift rewrite of local-asr-service | Best native Apple integration | Large migration, breaks working Python/FastAPI/OpenAI-compatible gateway | ❌ Rejected |
| Add Apple SpeechAnalyzer through a Swift sidecar | Clean API boundary, minimal service rewrite, realistic runtime isolation | Adds a second build artifact and IPC protocol | ✅ Chosen |
| Keep only Qwen/FunASR/Sortformer | Existing code path | Does not exploit Apple's system ASR model already proving useful in Spokenly | ❌ Insufficient |

## 4. Scope

### In scope

- Create `apple-speech-worker` Swift CLI or daemon.
- Support macOS 26+ SpeechAnalyzer runtime detection.
- Support locale discovery and availability checks.
- Support AssetInventory allocation/download for requested locales.
- Support file transcription from extracted WAV/M4A-compatible audio inputs.
- Support live/streaming transcription as a later phase if file mode is validated.
- Support both SpeechTranscriber and DictationTranscriber modes if available.
- Request and normalize timing metadata, especially audio time ranges.
- Normalize Apple results into the existing OpenAI-compatible response structure.
- Add model registry entries such as `apple-speech`, `apple-speech-transcriber`, and `apple-dictation-transcriber`.
- Add quality gates for monotonic timeline, transcript coverage, duplicate volatile handling, and missing time ranges.
- Add optional contextual vocabulary for dictation mode.
- Add an explicit diarization probe that combines Apple ASR timing with an existing diarization engine only if requested.

### Out of scope

- Replacing system-wide Apple Dictation or Siri.
- Accessing Apple model weights directly.
- Fine-tuning Apple's model.
- Claiming Apple SpeechAnalyzer provides speaker diarization without evidence.
- Rewriting existing MLX/FunASR/Qwen pipelines.
- Solving long-gap same-speaker reconciliation in this spec.
- Making 5-hour diarized batch transcription a first-phase acceptance gate.
- Implementing a new diarization model from scratch.

## 5. Runtime and capability assumptions

### Required runtime

- macOS 26+.
- Xcode / Swift toolchain compatible with the SpeechAnalyzer APIs.
- Local Speech framework availability.
- Runtime checks must determine whether a given device supports SpeechTranscriber for the requested locale.

### Supported transcriber modes

```text
speech-transcriber
  Primary mode for file transcription, long-form audio, meeting-like audio, and general ASR.

dictation-transcriber
  Primary mode for short dictation, keyboard-like input, command text, and contextual vocabulary experiments.
```

### Locale policy

The first implementation must support at minimum:

```text
zh-CN
zh-Hans or Chinese locale variants reported by Apple runtime
en-US
en-CA if available
```

The worker must expose a discovery command so the Python service never hardcodes unsupported locales.

### Timing policy

The worker must request timing metadata. If the Apple API only returns segment-level time ranges rather than word-level timestamps, the response must preserve that truth.

The service must not synthesize fake word timestamps unless a separate forced-alignment stage is explicitly run.

## 6. Worker protocol

### 6.1 Capability discovery

Command:

```bash
apple-speech-worker capabilities --json
```

Response:

```json
{
  "runtime": "apple-speech",
  "platform": "macOS",
  "osVersion": "26.x",
  "supported": true,
  "supportedLocales": ["zh-CN", "en-US"],
  "modules": {
    "speechTranscriber": true,
    "dictationTranscriber": true,
    "speechDetector": true
  },
  "notes": []
}
```

### 6.2 Asset preparation

Command:

```bash
apple-speech-worker prepare --locale zh-CN --module speechTranscriber --json
```

Response:

```json
{
  "locale": "zh-CN",
  "module": "speechTranscriber",
  "supported": true,
  "allocated": true,
  "downloaded": true,
  "durationMs": 1234
}
```

If the asset is unsupported or cannot be downloaded, the worker must fail clearly.

### 6.3 File transcription

Command:

```bash
apple-speech-worker transcribe \
  --input /tmp/audio.wav \
  --locale zh-CN \
  --module speechTranscriber \
  --audio-time-ranges true \
  --volatile false \
  --json
```

Request fields from Python adapter:

```json
{
  "jobId": "uuid",
  "inputPath": "/tmp/audio.wav",
  "locale": "zh-CN",
  "module": "speechTranscriber",
  "mode": "file",
  "reportingOptions": {
    "volatileResults": false
  },
  "attributeOptions": {
    "audioTimeRange": true,
    "confidence": true
  },
  "contextualStrings": [
    "local-asr-service",
    "Qwen3-ASR",
    "FunASR",
    "SpeechAnalyzer",
    "Soniox",
    "ElevenLabs",
    "Spokenly"
  ]
}
```

Response:

```json
{
  "jobId": "uuid",
  "engine": "apple-speech",
  "module": "speechTranscriber",
  "locale": "zh-CN",
  "text": "...",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.42,
      "text": "...",
      "isFinal": true,
      "confidence": null,
      "speaker": null
    }
  ],
  "metadata": {
    "local": true,
    "appleApi": true,
    "volatileIncluded": false,
    "timingGranularity": "segment|word|unknown",
    "assetManagedBySystem": true,
    "durationMs": 12345
  }
}
```

### 6.4 Live transcription, later phase

Live mode should not be first implementation unless file transcription is validated.

The future command can be either daemon IPC or stdin/stdout framed JSON:

```text
Python service / UI client
  -> audio buffer stream
  -> apple-speech-worker daemon
  -> volatile and final segment events
```

Live response events:

```json
{
  "type": "transcript.delta",
  "isFinal": false,
  "start": 12.1,
  "end": 13.8,
  "text": "rough text"
}
```

```json
{
  "type": "transcript.final",
  "isFinal": true,
  "start": 12.1,
  "end": 14.2,
  "text": "final text"
}
```

## 7. Python service integration

### 7.1 Registry

Add model entries:

```yaml
apple-speech:
  engine: apple-speech
  module: speechTranscriber
  local: true
  realtime: possible
  diarization: false
  timestamps: true_if_runtime_provides
  recommendedFor:
    - local dictation baseline
    - Chinese/English mixed daily input
    - short and medium transcription
    - privacy-sensitive local transcription
  notRecommendedFor:
    - speaker-labeled transcript without external diarization
    - legacy macOS
    - guaranteed word-level timestamps until validated
```

```yaml
apple-dictation:
  engine: apple-speech
  module: dictationTranscriber
  local: true
  contextualVocabulary: true
  recommendedFor:
    - short dictation
    - technical vocabulary biasing
    - Spokenly-like input
```

### 7.2 Adapter boundary

Create a narrow Python adapter and core engine boundary that match this repo's existing layout:

```text
src/core/apple_speech_port.py
src/core/apple_speech_engine.py
src/adapters/apple_speech_worker_client.py
```

Boundary ownership:

- `src/core/apple_speech_port.py`: typed request/response dataclasses and literals
- `src/core/apple_speech_engine.py`: `ASREngine` implementation that maps service calls to Apple worker requests
- `src/adapters/apple_speech_worker_client.py`: subprocess execution, timeout, stderr capture, stdout JSON parsing, and worker error mapping
- `apple-speech-worker/`: Swift package and CLI binary

Do not place an `AppleSpeechEngine` implementation under `src/adapters`; existing engines live in `src/core`, and `src/core/factory.py` is the creation boundary for ASR engines.

Core port types:

```python
@dataclass
class AppleSpeechRequest:
    input_path: str
    locale: str
    module: Literal["speechTranscriber", "dictationTranscriber"]
    contextual_strings: list[str]
    include_audio_ranges: bool = True
    include_volatile: bool = False

@dataclass
class AppleSpeechSegment:
    start: float | None
    end: float | None
    text: str
    is_final: bool
    confidence: float | None = None

@dataclass
class AppleSpeechResult:
    text: str
    segments: list[AppleSpeechSegment]
    timing_granularity: Literal["none", "segment", "word", "unknown"]
    metadata: dict[str, object]
```

### 7.3 Response normalization

For this repo's existing `/v1/audio/transcriptions` JSON behavior, return:

```json
{
  "text": "...",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.42,
      "text": "...",
      "speaker": null
    }
  ],
  "language": "zh-CN",
  "duration": 123.4,
  "model": "apple-speech"
}
```

No speaker field should be non-null unless an explicit diarization stage passes quality gates.

Do not change the existing default JSON behavior as part of this spec. `speech-swift` uses a stricter OpenAI shape where default `json` returns only `{"text": "..."}` and rich metadata belongs to `verbose_json`; this repo already returns rich JSON by default. Apple integration should preserve the local contract unless a separate API compatibility spec changes it.

## 8. Diarization boundary and probe design

### 8.1 Truthful product statement

Apple SpeechAnalyzer is an ASR engine in this service.

It may improve the speaker pipeline indirectly by producing better transcript text and more reliable time ranges, but the speaker labels still require a separate diarization stage.

### 8.2 Probe pipeline

```text
audio
  -> Apple SpeechTranscriber transcript + audio time ranges
  -> existing diarization runtime, initially Sortformer if still available
  -> timeline alignment
  -> quality gates
  -> speaker-labeled transcript or clear failure
```

### 8.3 Alignment rules

If Apple returns word-level time ranges:

```text
assign each word to speaker turn by midpoint or max overlap
merge adjacent words by same speaker and short gap threshold
```

If Apple returns segment-level time ranges only:

```text
assign segment to speaker only when one speaker dominates the segment duration
otherwise split only if a separate forced-alignment stage exists
otherwise mark speaker = null or fail diarization request
```

If Apple returns no usable time ranges:

```text
no diarization alignment should be attempted
return ASR-only result
```

### 8.4 Speaker quality gates

A diarized Apple pipeline is not production-eligible unless all of these pass on representative samples:

- transcript coverage loss <= 1% normalized words versus top-level ASR text
- Unknown speaker duration <= 5% of speech duration
- segments shorter than 0.5s <= 10% of all speaker segments
- timeline is monotonic
- no tail timestamp collapse
- cross-chunk speaker label remapping does not create obvious identity swaps
- failure returns an explicit error instead of misleading speaker labels

## 9. Implementation phases

### Phase 0: API feasibility probe

- [x] Create a minimal Swift package outside the Python service integration path.
- [x] Build and run on the target Mac.
- [x] Print SpeechTranscriber / DictationTranscriber availability.
- [x] Print supported locales.
- [x] Allocate `zh-CN` and `en-US` model assets.
- [ ] Transcribe one 30-second Chinese/English mixed sample.
- [x] Verify whether audio time ranges are returned, and at what granularity.
- [x] Produce one deterministic JSON fixture that matches the proposed worker response envelope.

Acceptance: a standalone Swift program can produce JSON for one local file and records an explicit go / no-go / blocked result for Phase 1. Python service integration must not start before this acceptance is met.

Phase 0 evidence from 2026-07-03:

- Swift package: `apple-speech-worker/`
- Target SDK/toolchain: Apple Swift 6.3.3, macOS SDK 26.5, runtime OS `Version 26.5.1 (Build 25F80)`
- `capabilities --json`: `supported=true`, 54 locales, `speechTranscriber=true`, `dictationTranscriber=true`, `speechDetector=true`
- `prepare --locale en-US --module speechTranscriber --json`: `supported=true`, `allocated=true`, `downloaded=true`
- `prepare --locale zh-CN --module speechTranscriber --json`: `supported=true`, `allocated=true`, `downloaded=true`
- `transcribe --input /Users/leipeng/Documents/Projects/speech-swift/Tests/AudioServerTests/Resources/test_audio.wav --locale en-US --module speechTranscriber --audio-time-ranges true --volatile false --json`: produced one final segment with segment-level timing and no stderr
- Fixture: `docs/fixtures/apple-speech-transcribe-en-US.json`
- Python subprocess client probe: parsed capabilities, prepare, and transcribe output from the built Swift binary

Phase 0 decision: **GO for Phase 1 CLI and Python worker-client boundary**. Full service registry/API integration remains out of scope until a project-owned Chinese/English mixed fixture is added or the English-only probe is explicitly accepted as sufficient.

Environment caveat: running the built worker inside the Codex filesystem sandbox returned no Speech framework locales. Running the same binary outside the sandbox returned the expected locale inventory. Real worker verification must run outside sandboxed shells or under the final app's normal macOS permissions.

### Phase 1: `apple-speech-worker` CLI

- [x] Implement `capabilities` command.
- [x] Implement `prepare` command.
- [x] Implement `transcribe` command.
- [x] Convert input audio to the analyzer's required format when needed.
- [x] Return deterministic JSON.
- [x] Separate stdout JSON from stderr logs.
- [x] Add timeout and structured error codes.
- [x] Add CLI contract tests that assert stdout is valid JSON with no human log lines.

Acceptance: Python can call the CLI and parse stable JSON.

Phase 1 evidence from 2026-07-04:

- Swift package builds in a non-sandboxed shell.
- `swift run --package-path apple-speech-worker apple-speech-worker-contract-tests` prints `contract-tests: passed`.
- `tests/unit/test_apple_speech_worker_client.py` verifies Python subprocess JSON parsing, stderr failure handling, invalid stdout rejection, timeout handling, and missing binary errors.
- `tests/unit/test_apple_speech_worker_source_contracts.py` verifies the live runtime cancels its result collection task on early exit.

Phase 1 decision: **GO for Phase 2 Python service integration**. Runtime verification for Apple Speech framework capability discovery and real transcription must continue outside the Codex filesystem sandbox.

### Phase 2: Python adapter and registry

- [x] Add `AppleSpeechEngine` in `src/core/apple_speech_engine.py`.
- [x] Add `AppleSpeechWorkerClient` in `src/adapters/apple_speech_worker_client.py`.
- [x] Add registry entries for `apple-speech` and `apple-dictation`.
- [x] Add model discovery to existing capabilities endpoint.
- [x] Add OpenAI-compatible `/v1/audio/transcriptions` support.
- [x] Preserve existing Qwen/FunASR routes.
- [x] Preserve existing default JSON response shape for this repo.

Acceptance: `model=apple-speech` works through the same HTTP API used by Spokenly / puresubs.

Phase 2 implementation note from 2026-07-04:

- The Python service routes Apple Speech requests through a direct sidecar path rather than through `src/workers/model_worker.py`; the Swift CLI is already the process boundary for Apple Speech framework access.
- `apple-speech` and `apple-dictation` preserve the existing local JSON response shape and do not emit non-null speaker labels.
- `GET /v1/models` advertises Apple aliases as requestable, but real runtime use remains macOS 26+ and final Speech framework checks still run in the sidecar.
- Current automated acceptance covers registry, service routing, and API response-shape compatibility with mocked worker clients. Real sidecar smoke still must run from a non-sandboxed shell with a project-owned fixture before broader bilingual quality claims.

### Phase 3: Batch transcription quality probe

Run at least these samples:

```text
1. 30-60s Chinese daily dictation with English technical terms
2. 5min Chinese/English mixed monologue
3. 5min English meeting audio
4. 10-20min real multi-person meeting audio
5. 5min noisy/distant audio if available
```

Compare against:

```text
apple-speech
qwen3-asr
funasr / paraformer
soniox API if available
elevenlabs API if available
```

Metrics:

```text
subjective readability
technical term preservation
Chinese/English switching errors
punctuation quality
segment timing availability
runtime duration
peak process memory
failure rate
```

Acceptance: Apple SpeechAnalyzer has a clear recommended role, even if it is ASR-only.

### Phase 4: Dictation vocabulary mode

- [ ] Add configurable contextual vocabulary list.
- [ ] Include user/project terms: `local-asr-service`, `PureSubs`, `FunASR`, `Qwen3-ASR`, `SpeechAnalyzer`, `Spokenly`, `Soniox`, `ElevenLabs`, `WhisperX`, `pyannote`, `Playwright`, `Spring Boot`, `Spinnaker`, `GitHub Actions`, `Obsidian`.
- [ ] Validate whether DictationTranscriber improves short input over SpeechTranscriber.

Acceptance: `apple-dictation` is either recommended for short dictation or kept as fallback only.

### Phase 5: Diarization integration probe

- [ ] Feed Apple time-ranged output into the existing segment alignment layer.
- [ ] Run external diarization, initially the existing Sortformer route if it remains available.
- [ ] Align Apple transcript units to speaker turns.
- [ ] Apply quality gates.
- [ ] Compare against previous `qwen3-sortformer` outputs on the same 1:1 meeting sample.

Acceptance: either a truthful speaker-labeled path emerges, or the Apple path remains ASR-only and the spec explicitly blocks diarized promotion.

### Phase 6: Production decision

Possible outcomes:

```text
A. Promote apple-speech as default local dictation / transcription engine.
B. Promote apple-dictation for short dictation only.
C. Keep Apple ASR as optional local engine.
D. Use Apple transcript + external diarization as an experimental speaker pipeline.
E. Do not promote Apple diarization integration because timing granularity or speaker quality gates fail.
```

## 10. Acceptance criteria

- [ ] AC-1: Swift worker builds and runs on the target macOS 26+ machine.
- [ ] AC-2: Worker reports supported locales and module availability without crashing.
- [ ] AC-3: Worker can allocate/download required speech assets for `zh-CN` and `en-US` when supported.
- [ ] AC-4: File transcription returns stable JSON with finalized text.
- [ ] AC-5: If timing is requested, the worker reports the real timing granularity: none, segment, word, or unknown.
- [ ] AC-6: Python adapter exposes `model=apple-speech` through `/v1/audio/transcriptions`.
- [ ] AC-7: No fake speaker labels are emitted by Apple-only mode.
- [ ] AC-8: Diarized output is only returned when coverage and consistency gates pass.
- [ ] AC-9: Unsupported OS, unsupported locale, missing asset, permission failure, and missing timing metadata all fail clearly.
- [ ] AC-10: Benchmark results document where Apple beats or loses to Qwen3-ASR, FunASR, Soniox, and ElevenLabs.

## 11. Known risks

### Risk 1: Apple API availability varies by OS, hardware, and locale

Mitigation: runtime discovery first. No hardcoded assumption that `zh-CN` or a given module works on every machine.

### Risk 2: Timing granularity may not be enough for diarization

Mitigation: preserve truth. If only coarse segment ranges are available, do not claim word-level alignment.

### Risk 3: Volatile results cause duplicate text

Mitigation: default file mode should use finalized results only. Live mode must explicitly replace volatile results with final results by audio range.

### Risk 4: Apple SpeechAnalyzer improves ASR but does not solve speaker identity

Mitigation: keep Apple ASR and diarization stages separate. Speaker labels must come from a separate diarization runtime.

### Risk 5: Swift sidecar adds build and deployment complexity

Mitigation: keep the worker small, testable, and JSON-driven. Do not migrate the Python service.

## 12. Affected areas

| File / Area | Change Type | Intent |
|---|---|---|
| `apple-speech-worker/` | Add | Swift sidecar package |
| `src/core/apple_speech_port.py` | Add | Typed request/response contract |
| `src/core/apple_speech_engine.py` | Add | ASR engine implementation that wraps the Apple worker client |
| `src/adapters/apple_speech_worker_client.py` | Add | Subprocess client for worker CLI, JSON parsing, timeout, and error mapping |
| `src/core/model_registry.py` | Modify | Add Apple model profiles |
| `src/core/factory.py` | Modify | Add `apple-speech` engine creation path |
| `src/api/routes.py` | Modify | Route `/v1/audio/transcriptions` to Apple engine |
| `src/services/transcription.py` | Modify | Normalize Apple transcript and metadata |
| `src/adapters/segment_alignment.py` | Modify / probe | Consume Apple timing metadata for diarization probe |
| `tests/unit/test_apple_speech_engine.py` | Add | Engine request mapping and result normalization |
| `tests/unit/test_apple_speech_worker_client.py` | Add | Worker stdout/stderr, timeout, and structured error handling |
| `tests/integration/test_apple_speech_worker.py` | Add | Real worker smoke tests, gated by macOS 26 |
| `MODELS.md` | Modify | Document Apple engine capabilities and limits |

## 13. Agent implementation instructions

1. Do not rewrite the service in Swift.
2. Do not attempt a PyObjC bridge first.
3. Start with a standalone Swift CLI proof.
4. Separate `capabilities`, `prepare`, and `transcribe` commands.
5. Treat Apple SpeechAnalyzer as ASR-only until speaker evidence exists.
6. Do not synthesize fake word timestamps.
7. Do not output speaker labels unless an external diarization stage passes gates.
8. Keep stdout as machine-parseable JSON; send logs to stderr.
9. Make all runtime assumptions discoverable and testable.
10. Preserve existing Qwen/FunASR routes.
11. Use `speech-swift` only as a reference for boundaries and testing discipline; do not vendor or depend on it for this Apple system API path.
12. Keep Phase 0 independent from Python service integration; do not add registry or API changes until the Swift feasibility probe returns a usable JSON result.

## 14. Reference implementation notes

### `speech-swift` lessons to apply

- Keep ASR, forced alignment, and diarization as separate runtime contracts.
- Keep model selection registry-driven and capability-aware.
- Keep OpenAI-compatible behavior test-covered.
- Keep audio decode/resample details hidden behind a narrow boundary.

### `speech-swift` lessons not to apply

- Do not add a Swift HTTP server; Python FastAPI remains the gateway.
- Do not copy the large multi-model package shape.
- Do not mix human CLI logs with JSON stdout.
- Do not use Sortformer or speaker identity code as evidence for Apple SpeechAnalyzer diarization.

## 15. Initial command examples

```bash
# Probe capability
apple-speech-worker capabilities --json

# Prepare Chinese asset
apple-speech-worker prepare --locale zh-CN --module speechTranscriber --json

# Transcribe a file
apple-speech-worker transcribe \
  --input /tmp/sample.wav \
  --locale zh-CN \
  --module speechTranscriber \
  --audio-time-ranges true \
  --json
```

Expected service call:

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@/tmp/sample.wav" \
  -F "model=apple-speech" \
  -F "language=zh-CN" \
  -F "response_format=verbose_json"
```

## 16. Status history

| Date | Status | Note |
|------|--------|------|
| 2026-07-03 | 📝 草案 (Draft) | Initial Apple SpeechAnalyzer sidecar probe draft |
| 2026-07-03 | 📝 草案 (Draft) | Added `speech-swift` reference boundary, narrowed Phase 0, and aligned Python file ownership with this repo's core engine layout |

## 17. Final recommendation

Implement Apple SpeechAnalyzer as the next local ASR engine before spending more time on Cohere Transcribe.

For product truthfulness:

```text
apple-speech = likely best local dictation / ASR baseline
apple-speech + external diarization = probe only
cohere-transcribe = watchlist / English single-language candidate
qwen3-sortformer = experimental, not production-recommended
funasr / paraformer = still relevant for Chinese diarized pipeline, but English quality remains a known weakness
```
