---
specId: SPEC-011
title: Runtime-Aware ASR Model Registry and Decoupled Diarization Pipeline
status: 📝 草案 (Draft)
priority: P1 - Core Feature
creationDate: 2026-04-30
lastUpdateDate: 2026-04-30
owner: User (AI-Assisted)
relatedSpecs:
  - SPEC-002
  - SPEC-005
  - SPEC-007
  - SPEC-008
  - SPEC-009
tags:
  - asr
  - model-registry
  - runtime-contract
  - mlx
  - funasr
  - qwen3-asr
  - parakeet
  - sortformer
  - diarization
  - apple-silicon
---

# SPEC-011: Runtime-Aware ASR Model Registry and Decoupled Diarization Pipeline

## 1. Goal

Enable future ASR model upgrades to be added by model registration whenever the runtime contract is compatible, while supporting a decoupled Qwen3-ASR/Parakeet + Sortformer pipeline for higher-quality Apple Silicon transcription and speaker diarization.

## 2. Background

The project currently exposes an OpenAI Whisper-compatible HTTP API backed by a small set of local ASR engines:

- `FunASREngine`: wraps Alibaba FunASR `AutoModel`, backed by PyTorch MPS or CPU.
- `MlxAudioEngine`: wraps `mlx-audio` STT APIs, backed by MLX Metal on Apple Silicon.
- `ModelSpec`: maps user-facing aliases such as `paraformer`, `qwen3-asr`, and `sensevoice-small` to an engine type, model ID, description, and declared capabilities.

The earlier SPEC-011 FireRed exploration correctly identified the need for decoupled ASR and diarization, but chose the wrong ASR runtime for this Mac-focused product. FireRedASR2S may be a strong model family, but the official runtime path is CUDA/CPU-oriented and does not fit the Apple Silicon constraint. The archived branch `archive/firered-runtime-exploration-2026-04-26` remains useful as an implementation reference for ports, pipeline profiles, segment alignment, serial lifecycle management, and cancellation-safe cleanup. It must not be reused as-is because its FireRed runtime assumption was rejected.

This replacement SPEC clarifies the semantic layers that were previously conflated under the word "model". The goal is to make future upgrades routine: if a new Qwen3-ASR release uses the same `mlx-audio` API, adding it should be a registry change, not an architecture change.

## 3. Semantic Model

The project must use these terms consistently:

| Layer | Meaning | Examples | Code Owner |
|-------|---------|----------|------------|
| Model family / weights | The learned ASR or diarization model and its checkpoint format | Qwen3-ASR, Parakeet, Paraformer, SenseVoice, Sortformer | `ModelSpec.model_id` |
| Task capability | What the model can produce | transcription, timestamps, speaker diarization, VAD, language detection, emotion tags | `EngineCapabilities` |
| Runtime library | Python or Swift package used to load and call the model | `funasr`, `mlx-audio`, `parakeet-mlx`, Speech Swift | Engine implementation |
| Compute backend | Hardware acceleration framework used by the runtime | MLX Metal, PyTorch MPS, CoreML/ANE, CPU | Runtime dependency |
| Engine adapter | Project-local wrapper that normalizes lifecycle and outputs | `FunASREngine`, `MlxAudioEngine`, future `ParakeetMlxEngine` if needed | `src/core/*_engine.py` |
| Pipeline profile | Composition of multiple task models | `qwen3-sortformer`, future `parakeet-sortformer` | Pipeline registry |
| API contract | External HTTP request/response shape | `/v1/audio/transcriptions`, `/v1/models` | `src/api/routes.py` |

### 3.1 Registration Rule

Adding a model must be a registry-only change when all of these are true:

- The model uses an existing runtime library already wrapped by an engine.
- The model can be loaded with the existing engine's load function.
- The model can be invoked with the existing engine's transcription method.
- The model returns output that the existing normalization code can handle.
- Its declared capabilities can be expressed with `EngineCapabilities`.

Adding a model requires a new engine adapter when any of these are true:

- It uses a different Python or Swift package with a different API.
- It needs a different lifecycle model, such as a separate server, streaming context, or CoreML/ANE session.
- It returns output that cannot be normalized by the existing engine without model-specific branches that would make the engine misleading.
- It performs a different task, such as diarization-only or VAD-only, rather than ASR.

### 3.2 Examples

| Candidate | Runtime contract | Expected change |
|-----------|------------------|-----------------|
| Future `mlx-community/Qwen3-ASR-3.5B-8bit` that works with `mlx_audio.stt.load_model()` and `generate_transcription()` | Same as current `MlxAudioEngine` | Register a new `ModelSpec` only |
| `SenseVoiceLarge` that works with `funasr.AutoModel` | Same as `FunASREngine` | Register a new `ModelSpec` only |
| `mlx-audio` Parakeet model that works with `generate_transcription()` and current JSON normalization | Same as `MlxAudioEngine` | Register a new `ModelSpec` after validation |
| `parakeet-mlx` package using `from_pretrained(...).transcribe(...)` | Different runtime API despite MLX backend | Add `ParakeetMlxEngine` or a dedicated runtime adapter |
| Sortformer diarization | Diarization-only task, not ASR | Add `DiarizationPort` and a Sortformer adapter |
| Speech Swift | Separate Swift framework/server, may use MLX/CoreML/ANE | Reference architecture only unless realtime/ANE use case becomes product scope |

## 4. Design Decision

**Chosen approach**: build a runtime-aware model registry plus a serial decoupled pipeline using Apple Silicon-native ASR models and a dedicated diarization adapter.

**Rationale**:

1. The main architectural boundary is the runtime contract, not the vendor or model name.
2. Qwen3-ASR via `mlx-audio` is the primary quality-first ASR path for Chinese, English, and mixed-language local transcription.
3. Parakeet via MLX is a candidate throughput-first ASR path for English or supported European-language audio, but it must be validated before it becomes a default route.
4. Sortformer diarization should be modeled as a separate task port, not forced into the ASR engine abstraction.
5. Decoupled execution must be serial by default to avoid co-resident large models exhausting unified memory on M-series machines.
6. The public API should remain stable; model and pipeline complexity belongs behind `ModelSpec`, engine adapters, capabilities, and pipeline profiles.

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| Keep Paraformer as the primary multi-speaker path | Stable, already supports CAM++ diarization | English-heavy audio can hallucinate Chinese; diarization quality is tied to FunASR path | ❌ Rejected as the long-term quality path |
| Restore FireRed + Sortformer | Reuses archived code; strong model reputation | Official runtime is CUDA/CPU-oriented and mismatched with Apple Silicon | ❌ Rejected |
| Qwen3-ASR + Sortformer | Strong multilingual ASR fit for Apple Silicon; preserves useful decoupled architecture | Requires Sortformer API validation and alignment orchestration | ✅ Chosen |
| Parakeet + Sortformer | Fast English/European-language transcription with timestamps | Parakeet does not provide diarization; package/runtime choice needs validation | 🟡 Candidate follow-up |
| Speech Swift migration | Strong Apple platform integration; possible realtime/ANE path | Different API/server/language stack; not needed for current batch API | ⏸️ Deferred |

## 5. Target Architecture

### 5.1 Model Registry

`src/core/model_registry.py` remains the single source of truth for user-facing ASR aliases.

The registry should grow in two directions:

- Built-in ASR aliases: `qwen3-asr`, future `qwen3-asr-3.5`, validated `parakeet-*`, `paraformer`, `sensevoice-*`.
- Pipeline aliases: decoupled profiles such as `qwen3-sortformer`, kept separate from single-model aliases if they combine multiple engines.

`ModelSpec.engine_type` should mean "which project-local engine adapter knows this runtime contract", not "which company made the model".

### 5.2 Engine Adapters

`FunASREngine` owns the FunASR runtime contract:

- Load through `funasr.AutoModel`.
- Run through `AutoModel.generate()`.
- Use PyTorch MPS where available, with CPU fallback.
- Normalize FunASR sentence output and speaker fields.

`MlxAudioEngine` owns the `mlx-audio` STT runtime contract:

- Load through `mlx_audio.stt.utils.load_model()`.
- Run through `mlx_audio.stt.generate.generate_transcription()`.
- Use MLX Metal through `mlx-audio`.
- Normalize text, segment, sentence, SRT, VTT, and JSON outputs.
- Pass model-specific generation kwargs only through a validated mapping.

A new engine is required only if a model cannot fit these contracts without making the adapter lie about its runtime.

### 5.3 Language Semantics

Qwen3-ASR does not use the same language contract as the OpenAI form field. The HTTP API accepts values like `auto`, `en`, `zh`, and `yue`. The Qwen3-ASR model prompt expects language names such as `English`, `Chinese`, or `Cantonese`.

The MLX engine must include a language normalization layer:

| API value | Qwen3-ASR language value |
|-----------|--------------------------|
| `auto` | `English` initially, until automatic routing is implemented |
| `en` | `English` |
| `zh` | `Chinese` |
| `yue` | `Cantonese` |
| Already-supported language name | Preserve canonical language name |
| Unknown value | Pass through only if explicitly allowed by the model's `support_languages`, otherwise fail clearly |

The current code accepts `language` but does not forward it into `generate_transcription()`. Phase 1 fixes that.

### 5.4 Diarization Port

Speaker diarization is a separate task:

```text
audio -> ASR model -> transcript segments
audio -> diarization model -> speaker turns
transcript segments + speaker turns -> aligned speaker-labeled transcript
```

The project should reintroduce a small `DiarizationPort` from the archived branch:

- `SpeakerTurn(speaker: str, start: float, end: float)`
- `load()`
- `diarize_file(file_path: str) -> list[SpeakerTurn]`
- `release()`

Sortformer is the first target implementation. Its actual Python import path and output schema must be validated before implementation because the currently installed `mlx-audio` package in this workspace does not expose a Sortformer/VAD module under `mlx_audio.*`.

### 5.5 Pipeline Profiles

Pipeline profiles are not the same as models. A pipeline profile combines task-specific components:

| Profile | ASR component | Diarization component | Status |
|---------|---------------|-----------------------|--------|
| `qwen3-sortformer` | `qwen3-asr` | `sortformer-diar` | Target |
| `parakeet-sortformer` | validated Parakeet alias | `sortformer-diar` | Candidate follow-up |
| `firered-sortformer` | FireRed | Sortformer | Rejected/archive only |

Pipeline execution must be serial:

1. Load ASR model.
2. Transcribe audio with timestamped segments.
3. Release or switch away from the ASR model if needed.
4. Load diarization model.
5. Produce speaker turns.
6. Align speaker turns onto transcript segments.
7. Restore the resident model or let idle offload reclaim memory.

The service must not keep two large ML models resident in the same worker longer than required.

## 6. Implementation Phases

### Phase 0: Runtime Validation — Target: before implementation

- [ ] Confirm the installed or target `mlx-audio` version exposes the required Qwen3-ASR language and JSON segment behavior.
- [ ] Confirm Qwen3-ASR supported language list for the exact model IDs used in the registry.
- [ ] Confirm whether Parakeet should use `mlx-audio` or the independent `parakeet-mlx` package.
- [ ] Confirm Sortformer Python package, import path, model ID, and output schema.
- [ ] Record validation results in this SPEC or an accompanying session log before enabling public pipeline aliases.

**Acceptance**: every runtime chosen for Phase 1-3 has a known load function, inference function, output schema, memory expectation, and package dependency.

### Phase 1: Qwen3-ASR Language Forwarding

- [ ] Add a typed language normalization helper for `MlxAudioEngine`.
- [ ] Forward normalized language into `generate_transcription()`.
- [ ] Preserve existing behavior for Whisper and unknown MLX models unless a model-specific mapping is validated.
- [ ] Add unit tests proving `language=en` becomes `English`, `language=zh` becomes `Chinese`, and unsupported values fail or pass through according to the selected rule.

**Acceptance**: Qwen3-ASR receives the intended language prompt value during transcription, and existing MLX model tests still pass.

### Phase 2: Runtime-Aware Registry Updates

- [ ] Update registry descriptions to distinguish model family, runtime, backend, and capability.
- [ ] Register validated Parakeet alias only if it works with an existing engine contract.
- [ ] Add registry tests for the registration-only rule: Qwen3/Parakeet via `mlx-audio` should use `engine_type="mlx"`, FunASR/SenseVoice should use `engine_type="funasr"`.
- [ ] Keep custom full-path inference conservative; unknown `mlx-community/...` models should not claim capabilities until validated.

**Acceptance**: `/v1/models` communicates which engine contract each alias uses and does not overstate diarization support.

### Phase 3: Decoupled Pipeline Primitives

- [ ] Reintroduce a `DiarizationPort` for diarization-only adapters.
- [ ] Reintroduce `segment_alignment.align_speakers()` as a pure function using maximum overlap.
- [ ] Add a pipeline registry or equivalent profile resolver for `qwen3-sortformer`.
- [ ] Add a dedicated diarization worker job kind / IPC path before any pipeline profile becomes requestable.
- [ ] Keep pipeline aliases non-requestable until Sortformer runtime validation and integration tests pass.

**Acceptance**: alignment and profile resolution are fully unit-tested without loading real models.

### Phase 4: Serial Pipeline Orchestration

- [ ] Add a decoupled execution path in `TranscriptionService`.
- [ ] Serialize standard requests and pipeline requests to avoid worker attachment races.
- [ ] Ensure model switch/release boundaries prevent ASR and diarization models from remaining resident together.
- [ ] Restore the previous resident model after pipeline completion.
- [ ] Use cancellation-safe cleanup for gate release and resident model restore.
- [ ] Degrade to transcript-only if diarization or alignment fails, except for worker lifecycle failures.

**Acceptance**: integration tests prove orchestration order, degradation behavior, queue/gate behavior, and restoration behavior.

### Phase 5: Public Enablement and Documentation

- [ ] Lift the public request gate for `qwen3-sortformer` only after real-model validation.
- [ ] Update `README.md`, `MODELS.md`, and `/v1/models` examples.
- [ ] Document model-selection guidance:
  - Qwen3-ASR for Chinese, English, and mixed-language quality.
  - Parakeet for validated English/European-language throughput paths.
  - Qwen3 + Sortformer for multi-speaker diarization.
  - Paraformer remains supported but is no longer the preferred English-heavy route.
- [ ] Add benchmark notes for M1 Max specifically; do not rely on M4 Pro/A100 numbers for product decisions.

**Acceptance**: users can choose the right alias/profile from docs without knowing the internal runtime/backend details.

## 7. Acceptance Criteria

- [ ] SPEC-011 no longer describes FireRed as the chosen live implementation.
- [ ] The spec defines the semantic distinction between model family, runtime library, compute backend, engine adapter, capability, and pipeline profile.
- [ ] A future compatible Qwen3-ASR model can be added by `ModelSpec` registration only.
- [ ] A future incompatible runtime, such as independent `parakeet-mlx`, has a clear rule for when to add a new engine adapter.
- [ ] Qwen3-ASR language forwarding is implemented and tested.
- [ ] Sortformer integration is gated by actual package/API validation, not assumed from research notes.
- [ ] `qwen3-sortformer` uses serial execution and does not keep ASR and diarization models resident together.
- [ ] Pipeline failure modes degrade to transcript-only when safe.
- [ ] Public docs explain when to use Qwen3-ASR, Parakeet, Paraformer, SenseVoice, and decoupled diarization profiles.

## 8. Status History

| Date | Status | Note |
|------|--------|------|
| 2026-04-25 | 📝 草案 (Draft) | Original SPEC-011 drafted around FireRed + Sortformer exploration on a separate branch |
| 2026-04-26 | ⏸️ 暂缓 (Deferred) | FireRed runtime route re-evaluated as unsuitable for Apple Silicon live path |
| 2026-04-30 | 📝 草案 (Draft) | Rewritten as runtime-aware registry + Qwen3/Parakeet + Sortformer direction |

## 9. Related

- **Current code**: `src/core/model_registry.py`, `src/core/factory.py`, `src/core/mlx_engine.py`, `src/core/funasr_engine.py`, `src/services/transcription.py`, `src/workers/model_worker.py`
- **Archived reference branch**: `archive/firered-runtime-exploration-2026-04-26`
- **Rejected archived code**: `src/core/firered_engine.py`, `firered-asr`, `firered-sortformer`
- **Reusable archived ideas**: `DiarizationPort`, `SpeakerTurn`, `segment_alignment`, `PipelineProfile`, serial pipeline orchestration, submission gate, cancellation-safe cleanup
- **Research notes**:
  - `2026-04-27 local-asr-service retrospective and diarization roadmap`
  - `2026-04-27 Apple Silicon ASR model landscape`
