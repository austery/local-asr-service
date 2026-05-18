---
specId: SPEC-013
title: Production Qwen3 + Sortformer Speaker Separation Pipeline
status: 📝 重新切分中 (Rescoping)
priority: P1 - Core Feature
creationDate: 2026-05-17
lastUpdateDate: 2026-05-18
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
  - forced-alignment
---

# SPEC-013: Production Qwen3 + Sortformer Speaker Separation Pipeline

## 1. Goal

Expose a truthful Apple-native batch speaker-separation pipeline through the
existing OpenAI-compatible service without violating Apple Silicon memory
constraints or returning misleading speaker labels.

The target production pipeline is:

```text
audio
  -> Qwen3-ASR transcript text
  -> Qwen3-ForcedAligner word timestamps
  -> Sortformer speaker turns
  -> speaker-labeled transcript segments
```

This replaces the earlier two-stage assumption:

```text
audio -> Qwen3-ASR segments -> Sortformer speaker turns
```

Local E2E validation showed that Qwen3-ASR's native segments are chunk-level for
the tested English samples, so they are too coarse for reliable speaker-label
assignment.

## 2. Background

`SPEC-011` hardened the current contract and kept the scaffolding truthful.

`SPEC-012` is responsible for proving whether the chosen Apple-native diarization runtime is actually viable on this machine and in this repo's Python environment.

This SPEC deliberately comes **after** that proof. It should not absorb runtime uncertainty and orchestration design into one branch.

The service already has useful gateway primitives:

- model and pipeline registries
- segment alignment helper
- subprocess worker isolation
- API gating for discovery-only pipeline profiles
- OpenAI-compatible HTTP contract for Spokenly and puresubs

What it still lacks for production pipeline enablement:

- a validated diarization runtime contract
- a validated forced-alignment runtime contract
- a dedicated diarization worker execution path
- a dedicated forced-alignment worker execution path or an explicit combined
  pipeline worker job
- deterministic resident-model restore behavior under concurrency
- end-to-end truthful request handling for pipeline profiles

This repo should not reimplement lower-level model internals from `mlx-audio`
or `speech-swift`. Its role is the local gateway: API, queueing, model/runtime
isolation, lifecycle management, and response normalization.

## 3. Design Decision

**Chosen approach**: add a production pipeline that wraps proven `mlx-audio`
runtime contracts for Qwen3-ASR, Qwen3-ForcedAligner, and Sortformer, while this
service owns orchestration and OpenAI-compatible response shaping.

**Rationale**:

1. Diarization is not a transcription job and should not continue to be modeled as one.
2. Qwen3-ASR's native segment timestamps are not sufficient for speaker-label assignment.
3. Forced alignment is the missing bridge between high-quality Qwen3 text and Sortformer speaker turns.
4. The current subprocess architecture is the right place to manage Apple Silicon memory pressure and model lifecycle.
5. Pipeline requests must either produce real speaker-labeled output or fail clearly.
6. Restore semantics must become deterministic before any requestable pipeline profile is exposed.

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| Keep the current two-stage Qwen3-ASR + Sortformer alignment | Minimal conceptual complexity | E2E proved Qwen3-ASR segments are too coarse, producing misleading speaker labels | ❌ Rejected |
| Build a side-channel diarization path outside the worker lifecycle | Smaller local change | Breaks architecture consistency and weakens memory guarantees | ❌ Rejected |
| Reimplement VAD, forced alignment, and diarization post-processing locally | Maximum control | Turns this gateway into a model framework and repeats upstream work | ❌ Rejected |
| Wrap `mlx-audio` Qwen3-ASR + ForcedAligner + Sortformer in an explicit batch pipeline | Best fit for Python service, preserves gateway boundary, truthful output path | More orchestration and model lifecycle work | ✅ Chosen |

## 4. Scope

### In scope

- define the worker-side diarization job contract
- define the worker-side forced-alignment job contract, or a combined pipeline job
- implement a dedicated diarization execution path
- orchestrate serial ASR, forced alignment, then diarization execution
- group aligned words into speaker-labeled transcript segments
- make resident-model restore deterministic for pipeline requests
- add integration and failure-mode coverage
- enable requestable pipeline profiles only after these guarantees hold

### Out of scope

- runtime discovery or model selection experiments
- implementing forced-alignment model internals
- implementing Sortformer model internals
- replacing `mlx-audio` with a local model framework
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

Additional intake from 2026-05-18 E2E debugging:

- `qwen3-sortformer` two-stage output is not production-truthful because
  Qwen3-ASR produced one full-audio segment on both 60s and 10min samples
- forced chunking produced multiple segments, but they were chunk-level and
  duplicated overlap text
- `mlx-audio` includes Qwen3-ForcedAligner and demonstrates the intended
  two-pass Qwen3-ASR + forced-alignment workflow
- a local 60s experiment produced 153 word timestamps from
  `mlx-community/Qwen3-ForcedAligner-0.6B-8bit`
- word midpoint assignment against Sortformer turns produced plausible
  speaker-labeled text segments on the 60s English sample

## 6. Implementation Phases

### Phase 1: Runtime Contract Intake

- [ ] import the validated runtime assumptions from `SPEC-012`
- [ ] define the exact worker request and response contract for diarization jobs
- [ ] define the exact worker request and response contract for forced-alignment jobs
- [ ] define how diarization results map into `SpeakerTurn`
- [ ] define how forced-alignment results map into local word timestamp objects

**Acceptance**: explicit ASR, forced-alignment, and diarization runtime contracts
exist for the service to implement against.

### Phase 2: Worker IPC Extension

- [ ] add a dedicated diarization job kind
- [ ] add a dedicated forced-alignment job kind or a single composite pipeline job kind
- [ ] add result handling for diarization output
- [ ] add result handling for aligned word output
- [ ] keep transcription and diarization execution paths explicit rather than overloaded
- [ ] add unit coverage for diarization job lifecycle and error propagation

**Acceptance**: the subprocess worker can execute the required pipeline stages
and return typed results.

### Phase 3: Pipeline Orchestration

- [ ] implement serial ASR, forced alignment, then diarization orchestration in `TranscriptionService`
- [ ] assign words to speakers using overlap or midpoint rules
- [ ] merge adjacent words into readable speaker-labeled transcript segments
- [ ] prevent concurrent requests from leaving the service resident on the wrong model
- [ ] make restore behavior deterministic rather than best-effort
- [ ] preserve cancellation-safe cleanup semantics

**Acceptance**: pipeline execution produces speaker-labeled transcript segments
from word timestamps and leaves the service in a correct post-request state.

### Phase 4: Public Enablement

- [ ] enable requestable pipeline profiles only after integration coverage passes
- [ ] return real speaker-labeled output on success
- [ ] fail clearly on ASR, forced-alignment, diarization, or orchestration failure
- [ ] update public model and pipeline documentation

**Acceptance**: the service exposes a truthful requestable pipeline profile backed by real diarization execution.

## 7. Acceptance Criteria

- [ ] AC-1: a dedicated diarization worker path exists
- [ ] AC-2: a forced-alignment stage exists between Qwen3-ASR text and Sortformer speaker turns
- [ ] AC-3: resident-model restoration becomes deterministic for pipeline requests
- [ ] AC-4: requestable pipeline profiles return real speaker-labeled results or fail clearly
- [ ] AC-5: integration and concurrency tests cover orchestration and failure modes

## 8. Affected Areas

| File / Area | Change Type | Intent |
|-------------|-------------|--------|
| `src/workers/model_worker.py` | Modify | Add diarization plus forced-alignment job support, or one composite pipeline job |
| `src/services/transcription.py` | Modify | Implement three-stage orchestration and deterministic restore |
| `src/core/diarization_port.py` | Modify / confirm | Lock runtime contract |
| `src/core/alignment_port.py` | Add | Lock forced-alignment word timestamp contract |
| `src/core/pipeline_registry.py` | Modify | Enable only truthful requestable profiles |
| `src/api/routes.py` | Modify | Public pipeline enablement |
| `tests/unit/test_service.py` | Modify | Orchestration semantics |
| `tests/integration/*` | Modify | End-to-end pipeline behavior |

## 9. Status History

| Date | Status | Note |
|------|--------|------|
| 2026-05-17 | 🔵 待办 (Backlog) | Created as the production follow-up gated by SPEC-012 |
| 2026-05-18 | 📝 重新切分中 (Rescoping) | E2E showed two-stage Qwen3-ASR + Sortformer is too coarse; rescope to Qwen3-ASR + Qwen3-ForcedAligner + Sortformer |

## 10. Related

- **Specs**: [SPEC-011](./SPEC-011-Decoupled-ASR-Diarization.md)
- **Specs**: [SPEC-012](./SPEC-012-Apple-Native-Diarization-Capability-Probe.md)
- **Specs**: [SPEC-008](./SPEC-008-Dynamic-Model-Switching.md)
- **Specs**: [SPEC-009](./SPEC-009-Idle-Model-Offload.md)
