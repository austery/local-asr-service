# Local ASR Runtime and Model Roadmap

Status: Runtime upgrades and SPEC-016 Phases 1-3 merged; next-phase priorities proposed for review.
Updated: 2026-10-24
Source baseline: `4a031c6121177c9f75931b53d6f6f845cb80442d` (PR #38 merge; local main synchronized).

## Recommendation and decision boundary

Complete operational acceptance of the merged gateway and Qwen automatic-language
upgrade before expanding model support. Then run **one** isolated model experiment
at a time. Prefer the already drafted Nemotron diarization probe if meeting speaker attribution is the
main unmet need. Choose Fun-ASR-Nano instead if Chinese text accuracy or domain
terms are the immediate problem. Neither experiment commits us to integration.
Keep Paraformer as the configured default and preserve the current specialist roles.

This update authorizes no dependency synchronization, service restart, model
weights, inference, API/registry change, or production pipeline revival. It is a
reviewable roadmap update. Follow ADR-002 and SPEC-016 for subsequent work.

## Verified baseline and remaining uncertainty

| Area | Observed on 2026-10-24 | What remains unproven |
| --- | --- | --- |
| MLX Audio 0.6.1 / Qwen auto | PR #48 merge is in local main; adapter maps Qwen `auto` to `None`; lock and installed metadata show mlx-audio 0.6.1, MLX/MLX-Metal 0.32.4 | Current-process module versions and real HTTP silent-first-chunk/bilingual acceptance |
| Running gateway | Earlier read-only inspection on Oct 24 found healthy FunASR/Paraformer and the five active aliases; the process started Oct 22. The later main synchronization did not restart it | This is a dated snapshot, not a fresh health check. PR #38 activation and Qwen HTTP acceptance remain unverified |
| FunASR | Lock and installed metadata show 1.4.14; CAM++ patch retained | 1.4.16 compatibility/performance on this machine |
| Execution and worker lifecycle | [PR #38](https://github.com/austery/local-asr-service/pull/38) merged as `4a031c6`; main synchronized without restarting the gateway. SPEC-016 Phases 2/3 are delivered | Operational acceptance of execution identity, switching, shutdown and offload in the owner-controlled service |
| Retired pipeline | Public profile retirement and production orchestration removal are already in main (`8e8d043`, `174b338`) | No new production orchestration is proposed |
| MOSS | Registered opt-in English model, <=1,800 s; current main still does not send an English-preservation prompt | General language reliability, genuine-silence versus missing-tail classification, dependable caller completion |
| Nemotron | Plan-only branch `codex/nemotron-diarization-poc`, commit `5d0e18a00da73a9fb1838500183a2c3ef7c24d7c` | Runtime/quality acceptance; the plan is not an executed result |
| Fun-ASR-Nano | Installed 0.6.1 includes `stt/models/fun_asr_nano`; upstream generate accepts language and hotwords | Local loading, actual acceleration, speed, accuracy, adapter compatibility |

The [0.6.1 upgrade report](../2026-10-22-mlx-audio-061-upgrade.md)
records 38 isolated model calls and 329 passing final-tree tests, including real
E2Es. These are historical author evidence, not tests rerun for this update.
Its pre-merge status and the changelog's pending language are stale relative to
local Git history.

PR #38 integration at `03f56b44b63cc985d7944d21bce34d26331f819c` passed the
complete **371-test suite**, including Paraformer silence and Qwen silent-first-
chunk real-model E2Es. Ruff, Tach, targeted execution/session/transport typing and
wheel/source builds passed. [CI](https://github.com/austery/local-asr-service/actions/runs/40108105380)
and [CodeQL](https://github.com/austery/local-asr-service/actions/runs/40108103966)
passed on that exact head before merging. The merge tree was identical to the
tested tree. Both parent changelog entry sets were retained. These are completed
isolated integration checks, not production acceptance or a new test run for this
documentation PR.

Local main was fast-forwarded from `e3c61f7` to `4a031c6`. Dependencies were not
synchronized and the running server was not restarted. Source availability, runtime activation and
live acceptance remain separate states.

## Ordered next-phase work

| Priority | Work | Exit evidence | Dependency / disposition |
| --- | --- | --- | --- |
| P0 | Activate and accept the merged gateway/Qwen baseline | Owner-controlled restart onto the merged source; exact source/lock/process identity; omitted/auto and explicit-language HTTP cases, silent-first-chunk and natural bilingual speech; switching/offload/clean shutdown | First operational task. PR #38 is merged and local main is synchronized; activation is still pending |
| P1 | Resolve the MOSS reliability decision | Review bounded English prompt mitigation, independent English recordings, tail evidence and truthful language metadata; record retain-limited / defer / repair decision | Before expanding MOSS caller adoption; do not weaken coverage checks to obtain 200 |
| P1 candidate A | Nemotron standalone diarization PoC | Follow existing plan's raw-turn, reference, runtime and long-form gates; Go/No-Go/Defer report | Recommended first model experiment for meeting speaker needs; not an ASR replacement |
| P1 candidate B | Fun-ASR-Nano Chinese text PoC | Matched Qwen comparison showing useful text/term improvement within a frozen resource budget | Alternative to A for immediate Chinese text needs; not automatic parallel work |
| P2 | FunASR 1.4.14 -> 1.4.16 regression study | Same checkpoints; short/long, repeated calls, VAD, punctuation, CAM++ and SenseVoice non-regression | Separate lock change; promote earlier only for a reproduced device-placement defect |
| P2 | PureSubs MOSS caller work | Both byte/duration limits, chunk-local identities, deterministic error handling and complete real caller acceptance | Conditional on MOSS reliability decision; separate repository/PR |
| P3 | SPEC-016 Phase 4 contract hardening | A concrete remaining contract defect or maintenance burden justifies scoped typed options/results/errors and narrower fitness exemptions | Deferred maintenance; not a prerequisite for isolated model PoCs |
| Watch | FireRedASR2 Swift/MLX | Reproducible target-hardware evidence and a thin upstream integration path | No sidecar or adapter work now |

P0 success permits one isolated PoC; MOSS uncertainty blocks MOSS expansion,
not unrelated standalone ASR/diarization research. Do not start both candidate
tracks merely because their runtimes are available.

## P0: activate and accept the merged baseline

PR #38 is already reviewed and merged; do not reopen that implementation task.
The running process predates the merge and was deliberately left unchanged. At
an owner-approved activation window, record the current service identity and
rollback source, then restart through the existing single-worker workflow onto
the merged source. Check installed versions against the lock before deciding
whether environment synchronization is necessary; the observed environment
already contains mlx-audio 0.6.1 and MLX 0.32.4. Do not infer a dependency install
is needed merely because application code changed.

Record the resulting source/lock, package versions and startup identity. Submit
Qwen requests through the actual HTTP endpoint using omitted
language, `auto`, explicit `en` and `zh`. Reuse the silent-first-chunk regression
and include a natural bilingual recording; constructed concatenations alone do
not establish natural language-switching quality. Check retained early/middle/tail
speech, response metadata and no silent empty-success regression.

Exercise model switching back to Paraformer, selected-model response identity,
idle worker release and clean shutdown within the single-worker contract.
Recheck health/model discovery after activation; the earlier read-only success
does not certify the new process. Preserve the prior source and lock together
for rollback. Live inference acceptance remains open. Never infer activation
from package metadata or synchronized source alone.

## P1: bound MOSS before expanding its use

Keep explicit English, the 1,800-second limit, pinned weights and conservative
output validation. The [0.5.9 report](../2026-10-19-mlx-audio-059-upgrade.md)
explicitly retains known language/coverage failures. The current main call does
not pass a language-preserving prompt; `language=en` is an admission condition,
not a demonstrated output-language guarantee.

A separate mitigation exists at `codex/moss-preserve-english`, commit
`8783f2059a7e5e6d454c07b9942fcdd58d2fbc4f` ([PR #43](https://github.com/austery/local-asr-service/pull/43), confirmed open on Oct 24). Earlier investigation
recorded English output on one full recording but continued rejection of an
approximately 11.69-second uncovered tail. That is historical evidence, not
proof of full-file success. The branch is still outside main. Refresh its head and
read its validation report before integration. Review tail audio independently;
do not conflate language mitigation with coverage repair or detected-language
metadata correctness. Freeze additional independent English inputs before tuning.

PureSubs read-only inspection at `5606c618ac8a15ff44305bfa3522fc5ecbfab535`
still finds a size-only no-split fast path in `AudioChunkingService.processAudio`.
The [caller handoff](2026-09-07-puresubs-moss-integration.md) remains relevant,
but its original inspected commit is historical. After resolving MOSS scope,
refresh caller state and implement duration-aware splitting in that repository.
Keep speaker IDs chunk-local and never store a failed partial run as complete.

## Candidate A: reuse the Nemotron plan

The plan is in the separate worktree, not merged into this source baseline:
`docs/plans/2026-10-23-nemotron-diarization-poc.md` at commit `5d0e18a` on
`codex/nemotron-diarization-poc`. Inspect it with
`git show 5d0e18a:docs/plans/2026-10-23-nemotron-diarization-poc.md`.
The existing plan is still a draft for owner review; linking it does not approve
weights or execution. Do not overwrite its independent worktree.

Start with raw speaker turns, silence/flush/reset, known returning speakers,
overlap and Chinese/English/bilingual inputs. Quantitative DER requires reviewed
reference RTTM, a fixed collar/overlap policy and recording-wide identity mapping.
No references means no numerical speaker-accuracy claim. Follow that plan's
proposed resource and quality gates rather than inventing a second benchmark.
Only after a standalone Go should an offline transcript-tagging demo be considered.
Fine timestamp alignment is a separate requirement; coarse Qwen segments plus a
maximum-overlap label cannot establish speaker-tagged subtitle accuracy.

## Candidate B: Fun-ASR-Nano, text only

The [MLX checkpoint](https://huggingface.co/mlx-community/Fun-ASR-Nano-2512)
metadata identifies Apache-2.0, 0.8B and BF16. Its card body is empty; metadata
is not runtime acceptance. Installed mlx-audio 0.6.1 source confirms the exact
implementation and `generate(audio, language="zh", hotwords=[...])` interface.
Its conversion documentation explicitly says the public checkpoint has no
segment/word timestamp head. It does not establish native speaker diarization.

Use a task-owned uv environment with an exact lock and pinned model revision.
Do not use the pasted raw-pip recipe or install a second overlapping runtime.
First compare upstream direct generation with the existing `generate_transcription`
wrapper. Verify language handling and normalized text before considering registry
metadata. The current gateway forwards language but not hotwords in its generic
MLX call and exposes no hotwords form field. A model's hotword API is not an
already-supported HTTP feature. Keep any public hotword interface as a separate
contract decision after the direct-model comparison.

Start with 60 s Chinese, then 10 min and mixed-language/term clips; run 30-60 min
only after short gates pass. Compare against frozen Qwen3 output on the same audio.
A text-only challenger cannot replace Paraformer/CAM++ or timestamped subtitle
workflows. Reject or defer if value requires forced alignment, custom recovery or
model conversion ownership in this gateway.

## Shared experiment admission and exit gates

Reuse the real corpus and existing runners. Before inference freeze audio/model
hashes, exact runtime/lock, hardware, reviewed reference windows, holdout clips and
budgets. One owned model process at a time; preserve the running service and stop
only the experiment if production begins inference or resources become unsafe.

For the new Nano track, proposed starting gates are warm RTF <=1, owned-process
peak RSS <=8 GiB and at least 20 GiB available before starting. These are review
proposals, not measured capacity or accepted SLAs. Record MLX allocation telemetry
separately; RSS alone is not total unified/Metal memory. Measure cold load and
end-to-end time separately. An initial task budget is one working day through
short/10-minute probes; unresolved runtime/API incompatibility ends in Defer,
not an unbounded adapter project.

Freeze Qwen baseline windows first. Proposed Nano value gate: >=10% relative CER
reduction on Chinese held-out windows, or >=20 percentage-point improvement in a
predeclared term-recall set, without a meaning-changing regression on mixed speech.
Report both accuracy dimensions and denominators; neither a small-window score nor
a final timestamp proves full-recording completeness. Without reviewed references,
report qualitative observations and resource cost, not CER or a quantitative Go.
Stop on CPU fallback without acceptable acceleration evidence, OOM/crash, missing
late speech, persistent repetition or a missed budget. Preserve raw output before
normalization. Timestamp/diarization scoring applies only to models claiming it.

## Maintenance and watch list

[FunASR 1.4.16](https://github.com/modelscope/FunASR/releases/tag/v1.4.16)
fixes mixed-device configuration propagation. Its published regression boundary
uses CPU stand-ins/mocked accelerators, not MPS certification. It does not establish
that our CAM++ None-timestamp failure is fixed. Retain both patch sites and fixed
ASR/VAD/punctuation/speaker checkpoints while evaluating repeated requests.

[Swift MLX v0.1.3](https://github.com/Blaizzy/mlx-audio-swift/releases/tag/v0.1.3)
lists FireRed ASR 2 support, and the
[upstream FireRed README](https://github.com/Blaizzy/mlx-audio-swift/blob/main/Sources/MLXAudioSTT/Models/FireRedASR2/README.md)
provides a call path. Thus “no Apple GPU implementation exists” is stale;
“deployable on our machine” remains unproven. Track implementation changes only.
Reopen when reproducible relevant-hardware long-form accuracy/RTF/memory evidence
and a maintainable upstream integration path exist. Do not develop a Swift sidecar
on the strength of a converted checkpoint or leaderboard score.

The April SPEC-011 retrospective supplies the runtime-first lesson, not authority
to restore its later-retired Qwen/Sortformer roadmap. Preserve Parakeet's short-only
restriction pending an actual chunking fix. No new default, automatic routing,
custom speaker reconciliation or three-stage production pipeline is proposed.
The attachment's claimed daily-monitor update was not verified here; this review
did not create or alter automations.

## Historical delivery and evidence

Retain the accepted September runtime/MOSS decisions and these records:

- Original decision map (checkout-local: `.scratch/runtime-upgrades-moss/map.md`)
- [Archived initial roadmap](2026-09-07-runtime-upgrades-and-moss-evaluation-initial.md)
- MLX 0.5.1 baseline (checkout-local: `.scratch/runtime-upgrades-moss/evidence/2026-09-07-runtime-validation.md`), PR #28
- FunASR 1.4.14 validation (checkout-local: `.scratch/runtime-upgrades-moss/evidence/2026-09-07-funasr-preflight.md`), PR #29
- Bounded MOSS admission (checkout-local: `.scratch/runtime-upgrades-moss/evidence/2026-09-07-moss-adapter-admission.md`), PR #30
- [MOSS adapter contract](2026-09-07-moss-adapter-contract.md) and duration evidence (checkout-local: `.scratch/runtime-upgrades-moss/evidence/2026-09-07-moss-duration-validation.md`)
- [ADR-002](../ADR-002-Lightweight-Local-Speech-Gateway-Boundary.md), [SPEC-016](../SPEC-016-runtime-module-deepening.md), [model roles](../../MODELS.md), [changelog](../../CHANGELOG.md)

Historical `.scratch` evidence is checkout-local and may not exist in another
clone. Record availability before relying on it. Do not infer fresh acceptance
from historical test counts, pre-merge report wording or an approved roadmap.
