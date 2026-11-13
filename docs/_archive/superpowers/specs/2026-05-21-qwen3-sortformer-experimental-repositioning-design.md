# Qwen3 Sortformer Experimental Repositioning Design

Date: 2026-05-21

## Context

`local-asr-service` is intended to remain a usable local Apple Silicon speech
gateway. It owns the OpenAI-compatible HTTP boundary, serial execution, worker
isolation, response normalization, and a small set of runtime-backed model
choices. It is not intended to become a long-lived diarization research
platform.

The `qwen3-sortformer` profile was recently made requestable as an explicit
batch opt-in path after a promising long-form English probe. That enablement
followed the three-stage pipeline direction:

```text
Qwen3-ASR text
  -> Qwen3-ForcedAligner word timestamps
  -> Sortformer speaker turns
  -> service-side speaker-labeled segments
```

The current documentation is now inconsistent:

- `README.md`, `MODELS.md`, and `ADR-002` describe an opt-in reachable batch
  path with production-adjacent language.
- `SPEC-013`, the 2026-05-19 long-form defect note, and the Research Notes
  project MOC retain more conservative enablement gates.
- The latest real-media evidence changes the cost-benefit evaluation.

## Evidence

The 2026-05-20 real probe used a roughly 22-minute English 1:1 conversation
recording between Lei and his manager.

For `qwen3-sortformer`:

- The top-level Qwen3 transcript text was materially more readable than the
  Paraformer English transcript.
- Speaker-labeled `segments` were not trustworthy enough for the target use
  case. The response had 3,601 normalized words in top-level `text` but only
  2,540 words in `segments`.
- Speaker segmentation was unstable for the sample: 182 segments included 30
  `Unknown` segments and 59 segments shorter than 0.5 seconds.
- The runtime footprint was surprising for a local gateway path. Activity
  Monitor showed a peak near 28 GB while the pipeline combined ASR, forced
  alignment, diarization, and service-side orchestration.

For `paraformer` on the same conversation:

- Top-level `text` and `segments` were structurally consistent at 3,279
  normalized words.
- Speaker output had no `Unknown` segments for the sample.
- English transcription quality was visibly worse than Qwen3 text, with word
  corruption and poor phrase recognition that make it unattractive for this
  English meeting use case.

This evidence changes the interpretation of the Sortformer path. It preserved
Qwen3's English ASR strength, but it did not show that the service can reliably
turn that text into a high-quality English diarized transcript at an acceptable
project cost.

## Decision

Reposition `qwen3-sortformer` as an **experimental reachable deletion
candidate**.

The alias may remain explicitly reachable for evaluation through
`model=qwen3-sortformer`, but documentation must not present it as a production
recommendation or as the stable answer for English multi-speaker meetings.

The project should not continue expanding local diarization recovery logic to
rescue this path. Future improvement should prefer stronger upstream or
open-source diarized ASR capabilities that reduce local orchestration and memory
cost. If upstream progress does not make the path materially simpler and more
trustworthy, removing `qwen3-sortformer` is an acceptable future outcome.

## Documentation Shape

### Public repo docs

`README.md` should make the user-facing positioning hard to misread:

- Keep `qwen3-asr` as the quality-first transcription path.
- Keep `paraformer` as the current diarization-capable alternative when callers
  accept its English quality limits.
- Mark `qwen3-sortformer` as an experimental opt-in evaluation path, not a
  recommended local speech workflow.
- State that the current pipeline is high cost and did not produce trustworthy
  speaker segments on the latest real English 1:1 probe.

`MODELS.md` should distinguish **reachable** from **recommended**:

- The pipeline table should explain that requestable status only keeps the
  experiment callable.
- The model selection guide should stop treating `qwen3-sortformer` as the
  preferred Apple-native English meeting route.
- The notes should capture the observed tradeoff: Qwen3 text quality survives,
  but speaker segmentation quality, service complexity, and memory footprint
  currently fail the cost-benefit bar.

### Decision and specification docs

`ADR-002` should keep the lightweight-gateway boundary but revise the
`qwen3-sortformer` wording:

- Replace production-reachability framing with experimental reachability.
- Make deletion an explicit acceptable outcome.
- Reinforce that further speaker reconciliation, alignment cleanup, and
  model-specific diarization postprocessing are outside the current project
  direction.

`SPEC-013` should be updated as a reality record rather than an active
production promise:

- Record the 2026-05-20 real 1:1 probe findings.
- Clarify that the route is now experimental and not a public production target
  unless upstream capability changes the cost-benefit picture.
- Preserve the technical history of why forced alignment was attempted and why
  it did not close the final product gap for this project.

The 2026-05-19 long-form defect note can retain the earlier probe evidence, but
it should point forward to the newer real-meeting reassessment if touched.

### Research Notes

Research Notes should capture the project lesson, not only the implementation
status:

- Update the local ASR MOC so it no longer describes `qwen3-sortformer` as the
  next enablement path.
- Add a research note for the real 1:1 reassessment. The note should explain
  that the route is being downgraded because the local gateway cost-benefit is
  poor, not because a 64 GB Mac mini cannot execute it.
- Keep the note connected to the earlier boundary and long-form probe notes so
  future Lei can see how an initially promising path was rejected by more
  representative evidence.

## Non-Goals

- Do not add new alignment coverage repair, speaker reconciliation, embedding,
  clustering, or chunk-boundary cleanup logic.
- Do not change the default model or dynamic model-switching behavior as part of
  the documentation repositioning.
- Do not remove the pipeline in this change. Removal remains a later product
  decision after the experiment has been documented clearly.
- Do not claim `paraformer` solves English meeting transcription quality. It is
  only the more structurally coherent diarization output on this probe.

## Verification

The documentation change should be checked for consistency across:

- `README.md`
- `MODELS.md`
- `docs/ADR-002-Lightweight-Local-Speech-Gateway-Boundary.md`
- `docs/SPEC-013-Production-Decoupled-Diarization-Pipeline.md`
- the local ASR Research Notes MOC
- the new Research Notes reassessment note

The final wording should preserve one product truth:

> This route is not being downgraded because the local machine cannot run it.
> It is being downgraded because its current resource cost, orchestration
> complexity, and unreliable speaker-segment output are a poor fit for a usable
> local speech gateway.
