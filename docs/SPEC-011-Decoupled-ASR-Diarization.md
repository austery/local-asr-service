---
specId: SPEC-011
title: Decoupled ASR and Diarization Pipeline
status: 🚧 进行中 (In Progress)
priority: P1 - Core Feature
creationDate: 2026-04-25
lastUpdateDate: 2026-05-01
owner: User (AI-Assisted)
relatedSpecs:
  - SPEC-002
  - SPEC-007
  - SPEC-008
tags:
  - asr
  - diarization
  - mlx
  - firered-asr
  - sortformer
---

# SPEC-011: Decoupled ASR and Diarization Pipeline

## 1. Goal

Enable a decoupled transcription and diarization pipeline so the service can combine a stronger bilingual ASR model with a dedicated speaker-segmentation model without turning this small project into a heavyweight architecture refactor.

## 2. Background

The current service mostly relies on integrated model adapters such as `Paraformer` (FunASR) to provide both transcription and speaker diarization in one pass.

Current pain points:
- `Cam++` speaker attribution is weaker in English-heavy audio.
- `Paraformer` is no longer the best option for mixed Chinese-English recognition quality in 2026.
- Large model residency and hot-switching overhead compete with other local workloads on Apple Silicon.

### Architecture fit

This repository is better described as **lightweight hexagonal / ports-and-adapters** than strict Clean Architecture:
- `src/api` acts as inbound adapters for HTTP contracts and request validation.
- `src/services` acts as the application orchestration layer.
- `src/core/base_engine.py` defines stable engine-facing contracts.
- `src/core/*_engine.py` contains concrete model adapters.
- `src/core/diarization_port.py` defines the diarization port interface (SPEC-011).
- `src/adapters/` contains pure functions: text cleaning, audio chunking, and segment alignment.

SPEC-011 extends that existing shape instead of turning a feature delivery task into a full architecture rewrite.

## 3. Design Decision

**Chosen approach**: **Serial modular pipeline inside the current lightweight ports-and-adapters structure**.

**Rationale**:
1. **FireRedASR2-AED (1.1B)** is a better fit for mixed Chinese-English transcription and provides timestamps needed for alignment.
2. **Sortformer** is a better fit for speaker diarization than the current CAM++ path, especially for English-heavy multi-speaker audio.
3. **Serial execution** (`load ASR -> infer -> release -> load diarization -> infer -> release`) keeps memory bounded and aligns with the service's existing subprocess + release-first design.
4. **Separate ports** keep responsibilities clear: transcription and diarization are independent capabilities, not one forced "do everything" engine abstraction.
5. **No architecture rewrite**: the new pipeline is added through clear ports, adapters, and orchestration logic rather than a broad directory refactor.

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| **VibeVoice (9B)** | Integrated semantics and diarization | Too slow (RTF ~0.92) and memory-heavy for this service | ❌ Rejected |
| **Qwen3.5-Omni (30B)** | Strong multimodal reasoning | Far too large for this local service profile | ❌ Rejected |
| **Strict Clean Architecture refactor** | Terminology purity and explicit layers | Too heavy for a small service and distracts from the feature goal | ❌ Rejected |
| **Decoupled FireRed + Sortformer pipeline** | Better ASR, better diarization, controlled memory, fits current architecture | Requires alignment logic and pipeline orchestration | ✅ Chosen |

## 4. Implementation Phases

### Phase 1: Ports and model registration — ✅ Landed

- [x] Introduce explicit transcription and diarization ports, instead of treating both as one mandatory engine capability.
  - `src/core/diarization_port.py` defines the `DiarizationPort` protocol.
- [x] Register `firered-asr` and `sortformer-diar` in `src/core/model_registry.py`.
  - Both are registered as `requestable=False` — they are adapter targets for the decoupled pipeline, not directly usable via POST today.
- [x] Implement a `FireRedEngine` adapter for transcription (`src/core/firered_engine.py`).
- [x] Implement a `SortformerEngine` adapter for diarization (`src/core/sortformer_engine.py`).
- [x] Register `firered-sortformer` pipeline profile in `src/core/pipeline_registry.py`.
  - The profile is `requestable=False` and is listed in `GET /v1/models` as a discoverable but not-yet-enabled entry.
- [x] Keep existing integrated adapters (`FunASREngine`, `MlxAudioEngine`) working without regression.
- [x] Add `segment_alignment.py` pure function adapter to `src/adapters/`.

**Public API boundary**: `POST /v1/audio/transcriptions` with `model=firered-sortformer` returns `501 Not Implemented`. This is intentional — the decoupled runtime is not yet publicly enabled. The profile is visible in `GET /v1/models` for tooling and client discovery, but cannot be invoked via POST until Phase 2 is complete.

**Acceptance**: Both new adapters can load, run independently, and release resources cleanly through the same lifecycle expectations used by existing engines. ✅

### Phase 2: Pipeline orchestration — ⬜ Planned

- [ ] Add a decoupled execution path in `TranscriptionService` while keeping the existing integrated path intact.
- [ ] Ensure the service runs transcription and diarization serially, with an explicit release boundary between stages.
- [ ] Represent combined pipeline capabilities as a pipeline/profile concern rather than a single-model concern.
- [ ] Lift the 501 gate on `firered-sortformer` once the runtime is verified end-to-end.

**Acceptance**: The service can execute a serial decoupled pipeline without changing the public API contract for existing clients.

### Phase 3: Alignment and degradation — ⬜ Planned

- [ ] Implement alignment that merges transcript segments and diarization speaker turns.
  - Pure function skeleton already exists at `src/adapters/segment_alignment.py`.
- [ ] Define transcript-only degradation as a first-class success path when diarization fails.
- [ ] Define transcript-only degradation as a first-class success path when alignment fails.
- [ ] Add focused tests for orchestration order, degradation behavior, and compatibility with current models.

**Acceptance**: The decoupled path returns speaker-labeled output when both stages succeed, and still returns transcript-only output when diarization or alignment fails.

## 5. Acceptance Criteria

- [x] **Architecture clarity**: SPEC-011 is framed as a lightweight ports-and-adapters extension, not a strict Clean Architecture refactor.
- [x] **Phase 1 plumbing landed**: FireRed and Sortformer adapters exist; pipeline registry maps `firered-sortformer` to both components.
- [x] **Discovery-only gate**: `firered-sortformer` is visible in `GET /v1/models` (`requestable: false`) but POST returns `501` until runtime enablement.
- [x] **Compatibility**: Existing `paraformer` and `qwen3-asr` behavior remains compatible at the current API surface. (186/186 tests pass)
- [ ] **High-accuracy transcription**: FireRed-based transcription benchmarks better than the current default path on the project's mixed Chinese-English evaluation fixtures.
- [ ] **Better diarization quality**: Sortformer-based diarization benchmarks better than the current CAM++ path on the project's English-heavy multi-speaker evaluation fixtures.
- [ ] **Resource control**: The service never keeps the ASR model and diarization model resident at the same time during the decoupled path.
- [ ] **Graceful degradation**: Diarization failure or alignment failure still produces a successful transcript-only result.

## 6. Status History

| Date | Status | Note |
|------|--------|------|
| 2026-04-25 | 📝 草案 (Draft) | Initial draft based on April 2026 model research |
| 2026-04-25 | 📝 草案 (Draft) | Refined architecture framing to lightweight ports-and-adapters and expanded implementation phases |
| 2026-05-01 | 🚧 进行中 (In Progress) | Phase 1 landed: FireRed + Sortformer adapters, pipeline_registry, diarization_port, segment_alignment adapter. firered-sortformer discoverable but 501-gated at POST boundary. |

## 7. Related

- **Code**: `src/core/base_engine.py`, `src/core/model_registry.py`, `src/core/pipeline_registry.py`, `src/core/factory.py`, `src/core/firered_engine.py`, `src/core/sortformer_engine.py`, `src/core/diarization_port.py`, `src/services/transcription.py`, `src/adapters/segment_alignment.py`
- **Docs**: [2026 语音服务升级建议](../../research-notes/04-resource/2026-04-25_语音服务模型升级建议.md)
