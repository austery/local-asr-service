---
specId: SPEC-013
title: Experimental Qwen3 + Sortformer Speaker Separation Pipeline
status: 🧪 实验保留 (Experimental)
priority: P1 - Core Feature
creationDate: 2026-05-17
lastUpdateDate: 2026-05-21
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

# SPEC-013: Experimental Qwen3 + Sortformer Speaker Separation Pipeline

## 1. Goal

Evaluate whether a truthful Apple-native batch speaker-separation pipeline can
fit the existing OpenAI-compatible service without violating the project's
Apple Silicon memory and gateway-boundary constraints.

The pipeline explored here is:

```text
audio
  -> Qwen3-ASR transcript text
  -> Qwen3-ForcedAligner word timestamps
  -> Sortformer speaker turns
  -> speaker-labeled transcript segments
```

This replaced the earlier two-stage assumption:

```text
audio -> Qwen3-ASR segments -> Sortformer speaker turns
```

Local E2E validation showed that Qwen3-ASR's native segments are chunk-level for
the tested English samples, so they are too coarse for reliable speaker-label
assignment. The three-stage path was implemented and made explicitly
requestable for evaluation, but a later real English manager 1:1 probe showed
that it is not currently a production recommendation: speaker-labeled segments
lost transcript coverage, fragmented heavily, and cost too much memory for the
value delivered by this local gateway.

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

At the original production-candidate stage, it still lacked:

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

**Chosen approach**: evaluate an explicit pipeline that wraps proven
`mlx-audio` runtime contracts for Qwen3-ASR, Qwen3-ForcedAligner, and
Sortformer, while this service owns orchestration and OpenAI-compatible response
shaping.

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
- add long-form chunk planning for ASR, forced alignment, and diarization
- stitch chunk-level outputs into one global timeline with deterministic offsets
- support truthful 5-hour offline batch processing as the original production-candidate gate
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
- 10min sample alignment currently collapses near tail (max ~245s) when
  force-aligning full audio in one pass
- the current `AudioChunkingService` only applies in `MlxAudioEngine` ASR path
  and only when audio exceeds `MAX_AUDIO_DURATION_MINUTES` (default 50), so
  forced alignment currently receives unchunked long files
- forced-align prompt length for long-form input can exceed model context
  (`max_position_embeddings=8192`), so the original long-form production
  candidate required explicit chunk-and-stitch orchestration in the service
  layer

## 6. Implementation Phases

### Phase 1: Runtime Contract Intake

- [x] import the validated runtime assumptions from `SPEC-012`
- [x] define the exact worker request and response contract for diarization jobs
- [x] define the exact worker request and response contract for forced-alignment jobs
- [x] define how diarization results map into `SpeakerTurn`
- [x] define how forced-alignment results map into local word timestamp objects

**Acceptance**: explicit ASR, forced-alignment, and diarization runtime contracts
exist for the service to implement against.

### Phase 2: Worker IPC Extension

- [x] add a dedicated diarization job kind
- [x] add a dedicated forced-alignment job kind
- [x] add result handling for diarization output
- [x] add result handling for aligned word output
- [x] keep transcription, forced-alignment, and diarization execution paths explicit rather than overloaded
- [x] add unit coverage for diarization and forced-alignment job lifecycle and error propagation

**Acceptance**: the subprocess worker can execute the required pipeline stages
and return typed results.

### Phase 3: Pipeline Orchestration

- [x] implement serial ASR, forced alignment, then diarization orchestration in `TranscriptionService`
- [x] assign words to speakers using overlap or midpoint rules
- [x] merge adjacent words into readable speaker-labeled transcript segments
- [x] prevent concurrent requests from leaving the service resident on the wrong model in unit-covered orchestration
- [ ] validate restore behavior under real pipeline load
- [x] preserve cancellation-safe cleanup semantics for composite request IDs

**Acceptance**: pipeline execution produces speaker-labeled transcript segments
from word timestamps and leaves the service in a correct post-request state.

### Phase 4: Long-Form Chunked Pipeline (5h Production Gate)

- [x] design a shared chunk plan (time windows + overlap + absolute offsets)
- [x] chunk ASR execution with deterministic global offset mapping
- [x] chunk forced alignment per ASR chunk and stitch aligned words globally
- [x] chunk diarization and merge speaker turns into a global timeline
- [ ] add cross-chunk speaker reconciliation rules
- [x] enforce quality guardrails (no tail timestamp collapse, monotonic timeline)
- [ ] validate one 5-hour batch sample end-to-end under real runtime

**Acceptance**: the pipeline can process a 5-hour sample with chunked ASR +
forced alignment + diarization and produce globally coherent speaker-labeled
segments without tail timestamp collapse.

### Phase 5: Experimental Reachability

- [x] keep the pipeline reachable only through explicit `model=qwen3-sortformer` selection
- [ ] promote the profile to a recommended production path
- [x] fail clearly on ASR, forced-alignment, diarization, or orchestration failure
- [x] update public model and pipeline documentation

**Acceptance**: the service preserves an explicit evaluation path without
presenting it as the stable English meeting-transcript answer.

## 7. Acceptance Criteria

- [x] AC-1: a dedicated diarization worker path exists
- [x] AC-2: a forced-alignment stage exists between Qwen3-ASR text and Sortformer speaker turns
- [ ] AC-3: resident-model restoration is validated under real pipeline requests
- [ ] AC-4: requestable pipeline profiles return trustworthy speaker-labeled results or fail clearly
- [x] AC-5: unit and integration tests cover orchestration and failure modes
- [ ] AC-6: long-form chunked pipeline supports truthful 5-hour batch requests
- [ ] AC-7: no timestamp-tail collapse in final aligned words for long-form English

## 7.1 Current Implementation Status

The current branch implements the non-public three-stage code path:

- `AlignmentPort` and `AlignedWord` define the forced-alignment boundary.
- `qwen3-forced-aligner` maps to `mlx-community/Qwen3-ForcedAligner-0.6B-8bit`.
- `model_worker` supports explicit `transcribe`, `align`, and `diarize` job kinds.
- `TranscriptionService` runs Qwen3-ASR text, forced alignment, Sortformer turns,
  then groups adjacent aligned words by speaker.
- Long-form requests now use real extracted audio chunks for per-chunk ASR,
  per-chunk forced alignment, and per-chunk diarization before stitching back to
  a global timeline.
- Alignment quality gates fail loudly on non-monotonic words or severe tail
  timestamp collapse.
- `qwen3-sortformer` is currently explicitly requestable for evaluation, but it
  is no longer a production recommendation after the 2026-05-20 real English
  manager 1:1 probe.

## 7.2 Current Gaps (2026-05-18 Reality Check)

The current branch is functional but not a recommended production path. The
remaining gaps are:

1. Cross-chunk speaker identity now has a lightweight overlap-based remap, but
   this only reconciles adjacent chunks when speakers are present in the overlap.
   It does not yet solve long-gap same-speaker returns.
2. 10-minute real-runtime output reaches full timeline coverage. After tuning
   Sortformer runtime parameters, `Unknown` coverage on the Blair 10-minute probe
   dropped from ~275.44s to ~17.41s while max timeline coverage stayed at
   ~596.76s.
3. Word-to-speaker assignment still falls back to `Unknown` when overlap or
   midpoint matching misses a speaker turn boundary.
4. AC-3/AC-4/AC-6/AC-7 remain open and explain why this path should not be a
   production recommendation.
5. `qwen3-sortformer` may remain explicitly reachable for evaluation, but
   restore stability and speaker-consistency gaps remain unresolved.
6. 5-hour long-form validation has not run yet. Public enablement still requires
   truthful long-form evidence, not just the 10-minute probe.
7. Align and diarize jobs still share the same worker process as the resident
   ASR engine. That contributes to the poor resource tradeoff for this local
   gateway; the current decision is to avoid more local worker-topology work
   until upstream capability changes the direction.
8. Worker-originated errors now preserve the remote exception type in the IPC
   tuple, but that hardening does not turn the route into a reliable speaker
   transcript product.

Immediate project priority is no longer to deepen local speaker-recovery logic.
Keep the route as an explicit experiment, prefer stronger upstream or
open-source local diarized-ASR capabilities, and treat local embedding fallback
or more complex reconciliation as out of bounds unless a new product decision
reopens this line.

## 7.3 Real English 1:1 Reassessment (2026-05-20)

A roughly 22-minute English 1:1 conversation between Lei and his manager was a
more representative local-tool probe than the earlier promising long-form
interview run.

Observed `qwen3-sortformer` behavior:

- top-level Qwen3 transcript text was materially more readable than the
  Paraformer English transcript;
- speaker-labeled `segments` were not trustworthy enough for the target use
  case: top-level `text` contained 3,601 normalized words while `segments`
  contained 2,540;
- the response fragmented into 182 speaker segments, including 30 `Unknown`
  segments and 59 segments shorter than 0.5 seconds;
- Activity Monitor showed a peak near 28 GB while the local path combined ASR,
  forced alignment, diarization, and service-side merge work.

Observed Paraformer behavior on the same conversation:

- top-level `text` and `segments` stayed structurally consistent at 3,279
  normalized words;
- the speaker segments did not include `Unknown` for the sample;
- English text quality was visibly worse than Qwen3 and did not meet the
  intended English meeting transcript bar.

Decision from this probe: `qwen3-sortformer` remains an experimental reachable
path and a deletion candidate. A 64 GB Apple Silicon machine being able to run
the pipeline does not make the current design a good fit for this local speech
gateway.

## 8. Affected Areas

| File / Area | Change Type | Intent |
|-------------|-------------|--------|
| `src/workers/model_worker.py` | Modify | Add diarization plus forced-alignment job support, or one composite pipeline job |
| `src/services/transcription.py` | Modify | Implement three-stage orchestration and deterministic restore |
| `src/adapters/audio_chunking.py` | Modify | Expose reusable chunk plan for pipeline orchestration |
| `src/adapters/segment_alignment.py` | Modify | Support chunk-aware global timeline and speaker reconciliation |
| `src/core/diarization_port.py` | Modify / confirm | Lock runtime contract |
| `src/core/alignment_port.py` | Add | Lock forced-alignment word timestamp contract |
| `src/core/pipeline_registry.py` | Modify | Enable only truthful requestable profiles |
| `src/api/routes.py` | Modify | Public pipeline enablement |
| `tests/unit/test_service.py` | Modify | Orchestration semantics |
| `tests/unit/test_worker.py` | Modify | Chunked job contract and failure behavior |
| `tests/integration/*` | Modify | End-to-end pipeline behavior |

## 9. Status History

| Date | Status | Note |
|------|--------|------|
| 2026-05-17 | 🔵 待办 (Backlog) | Created as the production follow-up gated by SPEC-012 |
| 2026-05-18 | 📝 重新切分中 (Rescoping) | E2E showed two-stage Qwen3-ASR + Sortformer is too coarse; rescope to Qwen3-ASR + Qwen3-ForcedAligner + Sortformer |
| 2026-05-18 | 🟡 实现中 (Implementation) | Added forced-alignment port, worker align job, and unit-tested three-stage orchestration; profile remains discovery-only pending real-model E2E |
| 2026-05-18 | 🟡 实现中 (Long-form planning) | Added 5-hour production gate with chunked ASR/align/diarization and timestamp-collapse prevention requirements |
| 2026-05-18 | 🟡 实现中 (Chunked implementation) | Added real chunk extraction, per-chunk ASR/alignment/diarization stitching, and alignment quality gates; still blocked on real-runtime probe and speaker reconciliation |
| 2026-05-18 | 🟡 实现中 (Gap alignment) | Added explicit remaining blockers and prioritized lightweight cross-chunk speaker reconciliation before public enablement |
| 2026-05-18 | 🟡 实现中 (Reconciliation probe) | Added overlap-based speaker label reconciliation and reran 10-minute probe; speaker-consistency gap remains open |
| 2026-05-18 | 🟡 实现中 (Sortformer tuning) | Tuned Sortformer runtime parameters; Blair 10-minute probe now reaches ~596.76s max end with `Unknown` reduced from ~275.44s to ~17.41s |
| 2026-05-18 | 🟡 实现中 (Review hardening) | Added chunk sub-request cleanup, typed worker errors, pipeline quality 422 responses, and stricter runtime contract validation |
| 2026-05-21 | 🧪 实验保留 (Experimental) | Real English manager 1:1 probe showed better Qwen3 text did not translate into trustworthy speaker-labeled output at an acceptable resource cost; stop deeper local recovery work and keep the route as a deletion candidate |

## 10. Related

- **Specs**: [SPEC-011](./SPEC-011-Decoupled-ASR-Diarization.md)
- **Specs**: [SPEC-012](./SPEC-012-Apple-Native-Diarization-Capability-Probe.md)
- **Specs**: [SPEC-008](./SPEC-008-Dynamic-Model-Switching.md)
- **Specs**: [SPEC-009](./SPEC-009-Idle-Model-Offload.md)
