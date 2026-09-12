# Changelog

This file separates merged runtime changes from work that has not yet landed.

## Unreleased

- Resolve passthrough requests once under the admission lock, validate that selected
  model before queuing, and return its identity with the transcript (SPEC-016 Phase
  2). Concurrent switches no longer leave response metadata or timestamp gating
  tied to an earlier model. Explicit requests remain pinned and validate immediately.

- Remove the retired three-stage pipeline orchestration from production request
  handling (SPEC-016). Keep independent experimental adapters, worker domains,
  historical evidence, and active native diarization. Replace the broken pipeline
  probe with a retirement notice. A single spawn lock now protects resident model
  selection and enqueue; inference runs after that lock is released.

- Retire the `qwen3-sortformer` public profile: remove discovery and reject
  explicit requests with 400. Standalone `qwen3-asr`, internal pipeline
  infrastructure, historical evidence, and downloaded weights are retained.
- Update Swagger's manual model instructions, README, and model reference for
  `moss-transcribe-diarize`, explicit English, the 1,800-second input limit,
  speaker segments, and 400/422 behavior.
- Record the [PureSubs MOSS integration roadmap](docs/plans/2026-09-07-puresubs-moss-integration.md).
  Caller duration-aware chunking and chunk-scoped speaker labels are planned;
  no PureSubs implementation is included here.

These changes require branch review and integration before the running main
service exposes the updated `/docs` and model list.

## 2026-09-07 — Runtime upgrades and bounded MOSS merged

- [PR #28](https://github.com/austery/local-asr-service/pull/28): upgrade
  `mlx-audio` from 0.4.3 to 0.5.1 with a frozen dependency baseline.
- [PR #29](https://github.com/austery/local-asr-service/pull/29): upgrade
  FunASR from 1.2.7 to 1.4.14 with NumPy 1.26.4. Preserve the existing
  Paraformer/VAD/punctuation/CAM++ checkpoints and the CAM++ patch.
- [PR #30](https://github.com/austery/local-asr-service/pull/30): add opt-in
  `moss-transcribe-diarize`, using the evaluated pinned checkpoint. Accepted
  scope: English continuous speech up to 30 minutes per request. No automatic
  splitting, cross-recording speaker reconciliation, or default-model change.
- All three PRs merged after Lei confirmed external review acceptance. Main
  reached `55909f1d9eeddc80b6beefcc04bc99f944873464`; local main and its frozen
  environment were synchronized on request. The post-sync unit, integration,
  and reliability suite passed all 353 tests. Previous real-model evidence is
  recorded in the [admission report](.scratch/runtime-upgrades-moss/evidence/2026-09-07-moss-adapter-admission.md).

Local source/environment synchronization did not itself restart the service.
Historical pre-merge evidence reports describe their original execution state;
use this changelog and the [roadmap](docs/plans/2026-09-07-runtime-upgrades-and-moss-evaluation.md)
for current delivery status.
