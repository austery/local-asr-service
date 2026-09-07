# Runtime Upgrades and MOSS Adoption Roadmap

Status: Runtime/adoption delivery merged; follow-up work tracked below.
Updated: 2026-09-07
Decision record: [Runtime Upgrade and MOSS Adoption Decision Map](../../.scratch/runtime-upgrades-moss/map.md).
Delivery history: [Changelog](../../CHANGELOG.md).

## Delivered baseline

| Work | State | Evidence |
| --- | --- | --- |
| Baseline and MLX Audio 0.5.1 | Merged, PR #28 | [Runtime comparison](../../.scratch/runtime-upgrades-moss/evidence/2026-09-07-runtime-validation.md) |
| FunASR 1.4.14 / NumPy 1.26.4 | Merged, PR #29 | [FunASR validation](../../.scratch/runtime-upgrades-moss/evidence/2026-09-07-funasr-preflight.md) |
| Bounded English MOSS adapter | Merged, PR #30 | [Adapter admission](../../.scratch/runtime-upgrades-moss/evidence/2026-09-07-moss-adapter-admission.md) |
| Local main and dependency synchronization | Completed at `55909f1`; 353 tests passed | [Changelog](../../CHANGELOG.md) |

Lei accepted the listening results, confirmed external review acceptance, and
requested all three merges followed by local synchronization. The CAM++ patch,
existing specialist roles, and default Paraformer model remain unchanged.
The synchronization backed up the old environment and conflicting drafts and
preserved the user's untracked `transcript.json`. It did not restart the service.

## Accepted MOSS scope

- Exact alias: `moss-transcribe-diarize`; explicit English, normally `language=en`.
- One continuous-speech recording per call, at most 1,800 seconds. Inputs above
  this hard bound return 400. There is no automatic MOSS splitting.
- One upstream MLX call, 32,768 output-token budget, 900-second isolated-worker
  inference deadline, and the pinned evaluated checkpoint revision.
- Native speaker labels and timestamps; output checks reject detectable
  truncation, malformed output, or uncovered gaps over 10 seconds with 422.
  The gap check can reject legitimate silence and cannot certify every word.
- Speaker IDs are local to each recording. Numerical WER/CER and speaker
  accuracy remain unmeasured. Forty/sixty-minute probes stopped early; the
  30-minute policy is not a universal model maximum.

See [model usage](../../MODELS.md), the [adapter contract](2026-09-07-moss-adapter-contract.md),
and [duration evidence](../../.scratch/runtime-upgrades-moss/evidence/2026-09-07-moss-duration-validation.md).

## Current follow-up work

| Task | Owner / state | Completion boundary |
| --- | --- | --- |
| Refresh model docs and Swagger manual testing | This ASR follow-up branch; pending review/integration | Model examples include MOSS and its language/duration/errors; examples reflect the active registry |
| Retire `qwen3-sortformer` public entry | This ASR follow-up branch; pending review/integration | Omitted from discovery, rejected before queuing, standalone Qwen3 preserved |
| Add MOSS duration-aware caller chunking | PureSubs; planned, not implemented | Each actual chunk satisfies both the existing byte limit and 1,800-second bound |
| Preserve speaker scope across caller chunks | Same PureSubs PR; planned | Saved results distinguish chunk-local identities; no assumed global speaker match |
| Validate greater-than-30-minute caller workflow | Same PureSubs PR; planned | Real recording completes with explicit chunk boundaries and reviewed failure handling |

The [PureSubs handoff](2026-09-07-puresubs-moss-integration.md) records the inspected
caller commit, exact code paths, the 24 MiB versus duration mismatch, acceptance
checks, and rollback. Documenting that task does not implement it.

## Retirement scope and boundaries

Lei requested retirement of the unused Qwen3 + Sortformer experiment after the
MOSS review. The public `qwen3-sortformer` registration and recommendations are
removed in this follow-up. Its standalone `qwen3-asr` component stays active.
No direct use of the retired alias was found in the inspected PureSubs source.

Internal alignment, diarization, chunking, and worker machinery remains tested;
its removal would require a separate reference audit. Historical evidence and
model caches are retained. MOSS is only a bounded replacement for the accepted
English workload, not unrestricted long-form or global speaker identity.

## Working and verification boundary

Continue development in independent worktrees. Leave the running main checkout
and service alone until the reviewed change is explicitly integrated. Use uv
with the frozen lock, a single inference worker, and no competing model runs.
For this follow-up, verify OpenAPI, model discovery, retired-alias rejection,
retained public models, and the unit/integration/reliability suite. No new model
or inference algorithm is introduced; the recorded real-model acceptance remains
the baseline. A server restart is required to expose updated application docs.

Preserve the [lightweight gateway boundary](../ADR-002-Lightweight-Local-Speech-Gateway-Boundary.md):
no model forks, custom speaker reconciliation, automatic routing, or broad
pipeline refactor. Revert the retirement commit if the old public alias must
be restored; retain the accepted source/lock/checkpoint combination for runtime
rollback. Show new diffs before any push.

## Historical evaluation records

The [archived initial roadmap](2026-09-07-runtime-upgrades-and-moss-evaluation-initial.md) preserves the original source preflight, evaluation rubric, and reasoning.
The original supplied roadmap was a 527-line concluding attachment, not the
complete preceding ChatGPT conversation. It led to five resolved decisions in
the linked map. Keep those decisions and the evidence reports as historical
records; their pre-merge statements describe the time they were written, not
current delivery state. The next work is the caller integration above, not
another baseline or runtime evaluation.
