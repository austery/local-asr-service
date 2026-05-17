---
specId: SPEC-011
title: Pipeline Foundation Hardening for Decoupled Diarization
status: 📝 草案 (Draft)
priority: P1 - Core Feature
creationDate: 2026-04-30
lastUpdateDate: 2026-05-17
owner: User (AI-Assisted)
relatedSpecs:
  - SPEC-008
  - SPEC-009
tags:
  - asr
  - qwen3-asr
  - diarization
  - runtime-contract
  - pipeline
  - model-registry
  - apple-silicon
---

# SPEC-011: Pipeline Foundation Hardening for Decoupled Diarization

## 1. Goal

Restore contract safety and truthful scaffolding around the future decoupled diarization pipeline, so the current service can merge incremental groundwork without silently changing Qwen behavior or pretending that an unimplemented pipeline already works.

## 2. Background

The existing branch direction is still useful, but it currently mixes three different verification stories:

- API and model contract correctness for already-supported requests
- runtime feasibility of Apple-native diarization
- production orchestration for a decoupled ASR + diarization pipeline

Those are not one review unit.

The current branch already contains one real regression that affects today's service behavior:

- the API still defaults `language=auto`
- `qwen3-asr` still advertises `language_detect=True`
- but the new language forwarding path coerces `auto` to `English`

That is a semantic regression for Chinese or mixed-language audio, independent of any future pipeline work.

The branch also contains useful scaffolding that must remain truthful:

- pipeline profiles exist in discovery
- alignment helpers and profile resolution exist
- `_diarize_with_alias()` is still unimplemented
- resident-model restoration is not yet deterministic under concurrency

This SPEC narrows scope to hardening the current contract and keeping the scaffolding honest. Runtime validation and production enablement remain future work, but are intentionally out of scope for this PR.

## 3. Design Decision

**Chosen approach**: merge only the safe foundation pieces now, and make all incomplete pipeline behavior fail closed until Apple-native diarization is proven and implemented end-to-end.

**Rationale**:

1. A regression on current Qwen requests is more urgent than enabling future diarization.
2. Discovery-only pipeline profiles are useful only if they cannot accidentally produce fake-success results.
3. Runtime feasibility should be proven before service orchestration is generalized.
4. The current branch should be reviewable as one contract-hardening story.

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| Keep the umbrella SPEC and finish the pipeline in one branch | One large narrative | Mixes contract fixes, runtime research, and orchestration into one oversized review surface | ❌ Rejected |
| Roll back all pipeline groundwork and revisit later | Removes short-term complexity | Discards useful registry/profile/alignment groundwork | ❌ Rejected |
| Narrow `SPEC-011` to contract hardening and truthful gating within the current PR | Safe merge boundary, clear verification target, preserves useful groundwork | Leaves later runtime and orchestration work for follow-up | ✅ Chosen |

## 4. Scope

### In scope

- Restore correct Qwen language semantics for the current API contract
- Keep pipeline discovery metadata accurate
- Make unimplemented diarization paths fail closed rather than degrade silently
- Clarify what a requestable pipeline profile must guarantee before enablement
- Preserve useful groundwork for future work without overstating capability

### Out of scope

- Real Sortformer runtime validation
- New diarization worker job kinds or IPC protocol extensions
- Deterministic resident-model restore under concurrency
- Publicly requestable decoupled pipeline profiles

## 5. Implementation Phases

### Phase 1: Qwen Contract Hardening — Target: current branch

- [ ] Preserve `language=auto` semantics for Qwen requests instead of coercing them to `English`
- [ ] Keep explicit language mapping only for validated user-specified values such as `en`, `zh`, and `yue`
- [ ] Keep registry capability claims aligned with runtime behavior
- [ ] Add unit tests for Qwen language normalization and forwarding behavior

**Acceptance**: default Qwen requests do not silently change language behavior, and explicit language forwarding remains typed and test-covered.

### Phase 2: Pipeline Truthfulness Hardening — Target: current branch

- [ ] Keep `qwen3-sortformer` discovery-only until real diarization runtime and orchestration work are complete
- [ ] Treat unimplemented diarization execution as a hard failure for any future requestable profile
- [ ] Remove transcript-only fallback for "diarization path not implemented"
- [ ] Document that requestable pipeline profiles require a working diarization execution path and deterministic orchestration semantics

**Acceptance**: the service cannot return `200` for a requestable pipeline profile unless diarization is actually implemented for that profile.

### Phase 3: Merge Boundary Clarification — Target: current branch

- [ ] Keep pipeline discovery metadata, alignment helpers, and profile resolution if they remain accurate
- [ ] Record in status history that the original umbrella spec was resliced by verification unit

**Acceptance**: this spec can be reviewed as one contract-hardening story rather than as an umbrella architecture program.

## 6. Acceptance Criteria

- [ ] AC-1: `qwen3-asr` no longer turns default `language=auto` into `English`
- [ ] AC-2: Qwen registry/API capability claims remain semantically consistent with actual forwarding behavior
- [ ] AC-3: non-requestable pipeline profiles stay gated at the API layer
- [ ] AC-4: unimplemented diarization execution cannot degrade to fake-success behavior for requestable profiles
- [ ] AC-5: useful scaffolding for pipeline profiles remains available without overstating present capability

## 7. Affected Areas

| File / Area | Change Type | Intent |
|-------------|-------------|--------|
| `src/core/mlx_engine.py` | Modify | Correct Qwen language semantics |
| `src/core/model_registry.py` | Review / possibly modify | Keep declared capabilities truthful |
| `src/api/routes.py` | Review / possibly modify | Preserve API gate behavior |
| `src/services/transcription.py` | Modify | Fail closed for unimplemented diarization path |
| `tests/unit/test_mlx_engine.py` | Modify | Cover corrected Qwen behavior |
| `tests/unit/test_service.py` | Modify | Lock in hard-failure semantics for unimplemented diarization |
| `tests/integration/*` | Review / possibly modify | Keep public contract aligned |

## 8. Status History

| Date | Status | Note |
|------|--------|------|
| 2026-04-30 | 📝 草案 (Draft) | Original umbrella draft created |
| 2026-05-17 | 📝 草案 (Draft) | Resliced as foundation hardening so the current PR can merge on a smaller verification unit |

