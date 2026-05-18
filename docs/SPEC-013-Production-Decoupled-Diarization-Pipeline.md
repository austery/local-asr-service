---
specId: SPEC-013
title: Production Decoupled Diarization Pipeline
status: 🔵 待办 (Backlog)
priority: P1 - Core Feature
creationDate: 2026-05-17
lastUpdateDate: 2026-05-17
owner: User (AI-Assisted)
relatedSpecs:
  - SPEC-011
  - SPEC-012
  - SPEC-008
  - SPEC-009
tags:
  - asr
  - diarization
  - pipeline
  - multiprocessing
  - worker-ipc
  - apple-silicon
---

# SPEC-013: Production Decoupled Diarization Pipeline

## 1. Goal

Expose a real decoupled ASR + diarization pipeline through the existing service without violating Apple Silicon memory constraints or returning speaker-labeled success responses from placeholder logic.

## 2. Background

`SPEC-011` hardened the current contract and kept the scaffolding truthful.

`SPEC-012` is responsible for proving whether the chosen Apple-native diarization runtime is actually viable on this machine and in this repo's Python environment.

This SPEC deliberately comes **after** that proof. It should not absorb runtime uncertainty and orchestration design into one branch.

The service already has useful primitives:

- model and pipeline registries
- segment alignment helper
- subprocess worker isolation
- API gating for discovery-only pipeline profiles

What it still lacks for production pipeline enablement:

- a validated diarization runtime contract
- a dedicated diarization worker execution path
- deterministic resident-model restore behavior under concurrency
- end-to-end truthful request handling for pipeline profiles

## 3. Design Decision

**Chosen approach**: add a dedicated diarization execution path and explicit pipeline orchestration on top of the existing subprocess model, but only after `SPEC-012` returns `go` or a clearly bounded `conditional go`.

**Rationale**:

1. Diarization is not a transcription job and should not continue to be modeled as one.
2. The current subprocess architecture is the right place to manage Apple Silicon memory pressure and model lifecycle.
3. Pipeline requests must either produce real speaker-labeled output or fail clearly.
4. Restore semantics must become deterministic before any requestable pipeline profile is exposed.

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| Keep the current placeholder pipeline path and just flip `requestable=True` later | Minimal effort | High risk of fake-success behavior and resident-model drift | ❌ Rejected |
| Build a side-channel diarization path outside the worker lifecycle | Smaller local change | Breaks architecture consistency and weakens memory guarantees | ❌ Rejected |
| Add explicit diarization worker support and orchestration after runtime validation | Correct contract boundary, consistent lifecycle, safer production path | More implementation work | ✅ Chosen |

## 4. Scope

### In scope

- define the worker-side diarization job contract
- implement a dedicated diarization execution path
- orchestrate serial ASR then diarization execution
- make resident-model restore deterministic for pipeline requests
- add integration and failure-mode coverage
- enable requestable pipeline profiles only after these guarantees hold

### Out of scope

- runtime discovery or model selection experiments
- realtime / websocket expansion
- unrelated engine abstraction rewrites

## 5. Entry Criteria

This SPEC must not enter active implementation until `SPEC-012` produces either:

- `go`, or
- `conditional go` with constraints explicit enough to incorporate here

If `SPEC-012` returns `no-go`, this SPEC is blocked and must be rewritten around a different runtime or a different product decision.

Current intake constraints from `SPEC-012`:

- use a tested `mlx-audio` version that exposes `mlx_audio.vad`; probe validated `0.4.3`
- target the validated runtime contract: `load("mlx-community/diar_sortformer_4spk-v1-fp16")` plus `model.generate(audio_path, threshold=0.5)`
- add a thin mapping layer from `DiarizationSegment` to local `SpeakerTurn`

## 6. Implementation Phases

### Phase 1: Runtime Contract Intake

- [ ] import the validated runtime assumptions from `SPEC-012`
- [ ] define the exact worker request and response contract for diarization jobs
- [ ] define how diarization results map into `SpeakerTurn`

**Acceptance**: one explicit diarization runtime contract exists for the service to implement against.

### Phase 2: Worker IPC Extension

- [ ] add a dedicated diarization job kind
- [ ] add result handling for diarization output
- [ ] keep transcription and diarization execution paths explicit rather than overloaded
- [ ] add unit coverage for diarization job lifecycle and error propagation

**Acceptance**: the subprocess worker can execute diarization work and return typed results.

### Phase 3: Pipeline Orchestration

- [ ] implement serial ASR then diarization orchestration in `TranscriptionService`
- [ ] prevent concurrent requests from leaving the service resident on the wrong model
- [ ] make restore behavior deterministic rather than best-effort
- [ ] preserve cancellation-safe cleanup semantics

**Acceptance**: pipeline execution leaves the service in a correct and predictable post-request state.

### Phase 4: Public Enablement

- [ ] enable requestable pipeline profiles only after integration coverage passes
- [ ] return real speaker-labeled output on success
- [ ] fail clearly on diarization or orchestration failure
- [ ] update public model and pipeline documentation

**Acceptance**: the service exposes a truthful requestable pipeline profile backed by real diarization execution.

## 7. Acceptance Criteria

- [ ] AC-1: a dedicated diarization worker path exists
- [ ] AC-2: the pipeline no longer depends on placeholder `NotImplementedError` behavior
- [ ] AC-3: resident-model restoration becomes deterministic for pipeline requests
- [ ] AC-4: requestable pipeline profiles return real speaker-labeled results or fail clearly
- [ ] AC-5: integration and concurrency tests cover orchestration and failure modes

## 8. Affected Areas

| File / Area | Change Type | Intent |
|-------------|-------------|--------|
| `src/workers/model_worker.py` | Modify | Add diarization job support |
| `src/services/transcription.py` | Modify | Implement orchestration and deterministic restore |
| `src/core/diarization_port.py` | Modify / confirm | Lock runtime contract |
| `src/core/pipeline_registry.py` | Modify | Enable only truthful requestable profiles |
| `src/api/routes.py` | Modify | Public pipeline enablement |
| `tests/unit/test_service.py` | Modify | Orchestration semantics |
| `tests/integration/*` | Modify | End-to-end pipeline behavior |

## 9. Status History

| Date | Status | Note |
|------|--------|------|
| 2026-05-17 | 🔵 待办 (Backlog) | Created as the production follow-up gated by SPEC-012 |

## 10. Related

- **Specs**: [SPEC-011](./SPEC-011-Decoupled-ASR-Diarization.md)
- **Specs**: [SPEC-012](./SPEC-012-Apple-Native-Diarization-Capability-Probe.md)
- **Specs**: [SPEC-008](./SPEC-008-Dynamic-Model-Switching.md)
- **Specs**: [SPEC-009](./SPEC-009-Idle-Model-Offload.md)
