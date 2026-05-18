---
specId: SPEC-012
title: Apple-Native Diarization Capability Probe
status: ✅ 已完成 (Completed)
priority: P1 - Core Feature
creationDate: 2026-05-17
lastUpdateDate: 2026-05-17
owner: User (AI-Assisted)
relatedSpecs:
  - SPEC-011
  - SPEC-013
tags:
  - apple-silicon
  - diarization
  - mlx
  - coreml
  - sortformer
  - capability-probe
---

# SPEC-012: Apple-Native Diarization Capability Probe

## 1. Goal

Prove whether `mlx-audio` Sortformer is a viable Apple-native diarization runtime for this service, using local fixture audio and a realistic multi-speaker sample, and end with an explicit `go`, `conditional go`, or `no-go` decision.

## 2. Background

`SPEC-011` intentionally stopped at contract hardening and truthful pipeline scaffolding. It did **not** attempt to prove that the next-stage diarization runtime is actually suitable for Apple Silicon or for this repo's Python service architecture.

That separation is deliberate. The previous FireRed exploration already showed the failure mode:

- the architecture idea was not the problem
- the runtime selection was the problem
- research quality failed before implementation quality even mattered

For the next stage, the question is no longer:

> "Do Apple-native diarization implementations exist?"

That question is already answered by external references such as `speech-swift`, `FluidAudio`, and the `mlx-audio` model ecosystem.

The real question is narrower and more useful:

> "Should this repo continue down the `mlx-audio` Sortformer path for offline batch diarization on Apple Silicon?"

This SPEC exists to answer that question with real local evidence rather than assumptions.

## 3. Design Decision

**Chosen approach**: run a thin, execution-oriented capability probe with `mlx-audio` Sortformer as the primary target, and use `speech-swift` Sortformer only as a reference comparison point when output semantics or architectural expectations need cross-checking.

**Rationale**:

1. `mlx-audio` is the closest runtime match to the current Python service.
2. `speech-swift` is a strong Apple-native reference, but it is not the target stack.
3. The probe should use both a smoke-test fixture and a realistic multi-speaker sample, so the result is not distorted by either toy input or excessively long audio.
4. The outcome must be explicit enough to gate `SPEC-013`.

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| Start implementing worker IPC and pipeline orchestration before runtime proof | Visible code progress | Risks repeating the FireRed mistake with a different runtime | ❌ Rejected |
| Probe only with the 60-second fixture | Fastest execution | Too weak to justify a production decision | ❌ Rejected |
| Probe only with a long real recording | More realistic | Conflates runtime viability with long-audio scaling and failure handling | ❌ Rejected |
| Probe `mlx-audio` with layered samples and use `speech-swift` as a comparison reference | Smallest meaningful verification unit with realistic evidence | Requires some upfront experimental discipline | ✅ Chosen |

## 4. Scope

### In scope

- validate the `mlx-audio` Sortformer runtime on the target Mac
- identify the exact import path, model ID, and invocation style
- verify output structure and speaker-turn usability
- observe latency, memory, and obvious stability limits on two local samples
- compare any surprising behavior against `speech-swift` documentation or local commands when useful
- record one explicit decision: `go`, `conditional go`, or `no-go`

### Out of scope

- service/API integration
- worker IPC design
- enabling pipeline profiles
- solving long-audio chunking generically
- full production rollout criteria for `SPEC-013`

## 5. Probe Inputs

### Layer 1: Smoke Test Fixture

File:

- `tests/fixtures/two_speakers_60s.wav`

Purpose:

- confirm the runtime loads and returns a usable diarization shape on a short, known two-speaker sample
- validate the basic "can this runtime produce speaker turns at all?" question

### Layer 2: Realistic Multi-Speaker Sample

Primary file:

- `tests/fixtures/Blair_FEB_09_last10min.wav`

Reference file:

- `tests/fixtures/Paul Rosolie and Lex Fridman.normalized.mp3`

Purpose:

- validate that the runtime behaves plausibly on a more realistic multi-speaker Q&A segment
- use the Blair clip as the default realistic test artifact because it is closer to the target product scenario than the podcast-style reference sample

### Sample Notes

- The Blair sample is a trimmed final segment from a VP presentation that includes audience Q&A and speaker handoff.
- This keeps the probe focused on diarization viability under realistic multi-speaker conditions rather than turning it into a long-audio systems investigation.
- The Paul sample remains available only as a secondary conversational reference if probe behavior needs comparison.

## 6. Probe Questions

The probe must answer these questions explicitly:

1. What is the exact `mlx-audio` entrypoint for Sortformer diarization in this environment?
2. What model ID and package version are actually used?
3. What does the returned speaker-turn structure look like?
4. Does the runtime complete successfully on both probe layers?
5. Are the results at least directionally plausible for a future alignment pipeline?
6. Is there any immediate blocker that would make `SPEC-013` irresponsible to start?

## 7. Implementation Phases

### Phase 1: Runtime Entry Validation

- [x] identify the exact Python import path and callable entrypoint for `mlx-audio` Sortformer
- [x] record the exact package version and model ID
- [x] confirm whether the runtime expects file input, waveform input, or another representation
- [x] identify whether the output is already close to `list[SpeakerTurn]` or requires a normalization adapter

**Acceptance**: the probe has a reproducible runtime entrypoint and a known expected output contract.

### Phase 2: Layer 1 Smoke Probe

- [x] run the probe on `tests/fixtures/two_speakers_60s.wav`
- [x] capture whether the runtime loads and completes inference successfully
- [x] capture a sample of the diarization output shape
- [x] record rough latency and any obvious warnings/errors

**Acceptance**: the short fixture either proves basic viability or reveals an immediate no-go issue.

### Phase 3: Layer 2 Realistic Probe

- [x] run the probe on `tests/fixtures/Blair_FEB_09_last10min.wav`
- [x] record whether inference completes successfully
- [x] capture approximate latency and memory observations
- [x] note whether the result is directionally plausible for conversational multi-speaker audio
- [x] if output semantics are confusing, cross-check against `speech-swift` reference behavior or documentation

**Acceptance**: the repo has realistic evidence for or against the target runtime path.

### Phase 4: Decision Record

- [x] classify the outcome as `go`, `conditional go`, or `no-go`
- [x] if `go`, state what `SPEC-013` may assume about the runtime contract
- [x] if `conditional go`, list the exact conditions or risks that remain
- [x] if `no-go`, recommend the next fallback path instead of proceeding blindly

**Acceptance**: the probe closes with one explicit decision that can gate the next spec.

## 8. Decision Gates

### `go`

Use `go` only if all of the following are true:

- the runtime loads successfully
- both probe layers complete successfully
- output can be normalized into a `SpeakerTurn`-like structure without unreasonable ambiguity
- no immediate Apple Silicon runtime blocker is discovered

### `conditional go`

Use `conditional go` if the runtime works but at least one meaningful caveat remains, such as:

- output needs a thin normalization adapter
- the realistic sample works but shows quality uncertainty
- memory or latency are acceptable for probe purposes but need explicit guardrails before `SPEC-013`

### `no-go`

Use `no-go` if any of the following occur:

- runtime entrypoint is not actually usable in this environment
- one or both probe layers fail in a way that undermines the target architecture
- output semantics are too weak or too ambiguous to justify moving forward
- the runtime shows an Apple Silicon mismatch similar in spirit to the old FireRed failure

## 9. Acceptance Criteria

- [x] AC-1: `mlx-audio` Sortformer is exercised locally on the target Mac
- [x] AC-2: the exact runtime entrypoint, package version, and model ID are recorded
- [x] AC-3: both probe layers are attempted and their outcomes documented
- [x] AC-4: output shape is documented in terms relevant to a future `SpeakerTurn` contract
- [x] AC-5: the result ends with one of: `go`, `conditional go`, `no-go`
- [x] AC-6: the decision is strong enough to either permit or block `SPEC-013`

## 10. Deliverables

| Deliverable | Purpose |
|-------------|---------|
| Probe script or reproducible command sequence | Make the experiment repeatable |
| Short probe report | Capture observations and the final decision |
| Runtime contract summary | Feed validated assumptions into `SPEC-013` |

## 11. Relationship to SPEC-013

`SPEC-013` must not start production implementation until this SPEC has produced one explicit result.

- If this SPEC returns `go`, `SPEC-013` may proceed using the recorded runtime contract.
- If this SPEC returns `conditional go`, `SPEC-013` may proceed only with the listed constraints written into its scope.
- If this SPEC returns `no-go`, `SPEC-013` is blocked and must be re-scoped or replaced.

## 12. Probe Result

**Decision**: `conditional go`

**Validated runtime contract**:

- tested runtime: `mlx-audio 0.4.3`
- Python entrypoint: `from mlx_audio.vad import load`
- tested model ID: `mlx-community/diar_sortformer_4spk-v1-fp16`
- tested call shape: `model.generate(audio_path, threshold=0.5, verbose=False)`
- accepted input shape: file path, `np.ndarray`, or `mx.array`
- returned structure: `DiarizationOutput`
  - `segments: List[DiarizationSegment]`
  - `speaker_probs`
  - `num_speakers`
  - `text` as RTTM-style output

**Immediate repo-level blocker discovered**:

- the repo currently pins `mlx-audio 0.3.1`
- `mlx-audio 0.3.1` does **not** expose `mlx_audio.vad`
- `SPEC-013` must therefore treat a runtime dependency bump as an explicit prerequisite rather than assuming the current lockfile is sufficient

**Layer 1 evidence**:

- audio: `tests/fixtures/two_speakers_60s.wav`
- outcome: success
- model load + download time: `26.071s`
- inference time: `1.715s`
- detected speakers: `2`
- emitted segments: `21`
- `speaker_probs.shape`: `[714, 4]`
- output was directly usable as diarization segments without architectural reinterpretation

**Layer 2 evidence**:

- audio: `tests/fixtures/Blair_FEB_09_last10min.wav`
- outcome: success
- warm model load time: `1.835s`
- inference time: `3.543s`
- peak memory observed: `9.531 GB`
- detected speakers: `4`
- emitted segments: `569`
- `speaker_probs.shape`: `[7476, 4]`
- output looked directionally plausible for a realistic Q&A segment and did not show an Apple Silicon runtime mismatch

**Conditions carried forward into `SPEC-013`**:

- upgrade to a tested `mlx-audio` version that exposes `mlx_audio.vad`; this probe validated `0.4.3`
- add a thin adapter from `DiarizationSegment(speaker: int, start: float, end: float)` to local `SpeakerTurn`
- keep the public pipeline gated until worker IPC and restore semantics are implemented
- treat diarization quality tuning and threshold calibration as follow-up work, not as part of this probe

**Reproduction commands**:

```bash
uv run --with 'mlx-audio>=0.4.3' python -c "from mlx_audio.vad import load; model = load('mlx-community/diar_sortformer_4spk-v1-fp16'); result = model.generate('tests/fixtures/two_speakers_60s.wav', threshold=0.5); print(result.text)"
```

```bash
uv run --with 'mlx-audio>=0.4.3' python -c "from mlx_audio.vad import load; model = load('mlx-community/diar_sortformer_4spk-v1-fp16'); result = model.generate('tests/fixtures/Blair_FEB_09_last10min.wav', threshold=0.5); print(result.text)"
```

## 13. Status History

| Date | Status | Note |
|------|--------|------|
| 2026-05-17 | 📝 草案 (Draft) | Created as the execution-oriented runtime probe that follows SPEC-011 |
| 2026-05-17 | ✅ 已完成 (Completed) | Executed locally; returned `conditional go` with dependency bump and adapter prerequisites |

## 14. Related

- **Specs**: [SPEC-011](./SPEC-011-Decoupled-ASR-Diarization.md)
- **Specs**: [SPEC-013](./SPEC-013-Production-Decoupled-Diarization-Pipeline.md)
- **Fixtures**: `tests/fixtures/two_speakers_60s.wav`
- **Fixtures**: `tests/fixtures/Blair_FEB_09_last10min.wav`
- **Fixtures**: `tests/fixtures/Paul Rosolie and Lex Fridman.normalized.mp3`
- **Reference**: local `speech-swift` Sortformer docs and commands
