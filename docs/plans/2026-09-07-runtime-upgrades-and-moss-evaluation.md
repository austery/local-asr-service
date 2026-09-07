# Runtime Upgrades and MOSS Evaluation Roadmap

Status: Accepted working roadmap; baseline measurement and runtime/model decisions remain open.
Date: 2026-09-07
Canonical decision map: [Runtime Upgrade and MOSS Adoption Decision Map](../../.scratch/runtime-upgrades-moss/map.md)

## Destination

Produce an approved implementation handoff that selects reproducible runtime versions, decides whether MOSS solves English long-form multi-speaker transcription within the existing gateway boundary, and identifies safe retirement scope. Success includes an evidence-backed No-Go; adding MOSS is not a predetermined outcome.

The decision map finishes when these decisions and their validation/rollback requirements are recorded. Production integration, release, and deletion are subsequent work. Small isolated experiments may supply decision evidence; they are not a production launch.

## Source and confidence

The supplied attachment was read in full: 527 lines containing a concluding roadmap and self-verification questions. Earlier conversation turns were not included. Lei accepted this roadmap as the working plan on 2026-09-07. Earlier conversation decisions were not independently recovered.

The main direction agrees with the accepted [Lightweight Local Speech Gateway Boundary](../ADR-002-Lightweight-Local-Speech-Gateway-Boundary.md): upstream runtimes own inference; this repository owns API compatibility, capability gating, normalization, queueing, and resource isolation.

This session performed source and metadata inspection only. No dependency resolution, installation, model download, inference, quality comparison, or test suite was run. Historical benchmark results are context, not a newly verified baseline.

## Why use Wayfinder

This effort is expected to require more than two sessions: corpus and acceptance agreement, runtime compatibility evidence, a human-reviewed MOSS verdict, and an independent FunASR/retirement decision. Unresolved runtime compatibility and long-form output semantics prevent a trustworthy implementation plan today.

Use five decision tickets, each with an independently trackable approval or evidence boundary. Keep implementation details and hypotheses in this linked planning artifact. Do not create separate tickets for every metric, model, file, or speculative failure.

No tracker configuration was found in the repository. The invoked skill's local Markdown fallback is used under `.scratch/runtime-upgrades-moss/`. `/setup-matt-pocock-skills` can configure a different tracker later. The map and its five child tickets are versioned Markdown files on the documentation branch. No GitHub issues were published.

## Worktree operating boundary

Lei approved the roadmap and local documentation commit on 2026-09-07, requiring that work happen outside the running `main` checkout. The documentation branch `codex/runtime-upgrades-moss-roadmap` is based on remote main `2ff1a34db7d6999714b6df64a313c56e484bc171` in the independent worktree `/Users/leipeng/Documents/Projects/local-asr-service-worktrees/runtime-roadmap`. The original checkout stays at `97028ba0a3b8c5774903ace18cb7fb29456d9bb5`; its running service, environment, and user files are preserved.

Continue edits and experiments in this or another isolated worktree. Use a separate environment and output directory; do not install into the running checkout's `.venv` or take over its service port. Coordinate heavyweight measurements with the active service without interrupting it. The original untracked planning files remain historical drafts; this branch holds the maintained plan.

The overall roadmap does not need another approval round. Ask only for information or a substantive judgment that cannot be obtained from available evidence, such as unavailable corpus files or human transcript-quality review. Existing push and destructive-action approval rules still apply.

## Verified starting state

| Item | Observation on 2026-09-07 | Planning consequence |
| --- | --- | --- |
| Local checkout | `main` at `97028ba0a3b8c5774903ace18cb7fb29456d9bb5` | Initial inspected checkout; deliberately preserved for the running service. |
| Remote main | `2ff1a34db7d6999714b6df64a313c56e484bc171`, four commits ahead, no divergence | Included in the documentation worktree base. The changes include Apple Speech passthrough, queue-lock scope, and resident-model restoration fixes; a measured runtime baseline remains pending. |
| Working tree | No tracked changes; an existing untracked `transcript.json` | Preserve the file and exclude it from all planning/implementation commits. |
| Lock hash | `uv.lock` SHA-256 `2fb29a6c4391a5931266cacea563663ad01ab60416ca8677a64f06d6458b7298` | Record a fresh hash after reconciliation and before each experiment. |
| Installed Python | 3.11.14 | Keep Python fixed during runtime comparisons. |
| Installed and locked packages | `mlx-audio 0.4.3`, `funasr 1.2.7`, `mlx 0.31.2`, `transformers 5.8.1`, `huggingface-hub 1.15.0`, `numpy 2.3.5`, `torch 2.12.0`, `torchaudio 2.11.0` | Selected installed metadata matches the lock. Import/inference compatibility and clean installation have not been tested. |
| Active model aliases | Paraformer, Qwen3-ASR, SenseVoice Small, Apple Speech | Preserve their current roles and validate the service paths affected by shared dependencies. |
| Experimental profile | `qwen3-sortformer` is requestable through the separate pipeline registry | Freeze development; retirement remains a distinct decision. |

Sources: [model registry](../../src/core/model_registry.py), [pipeline registry](../../src/core/pipeline_registry.py), [model evidence](../../MODELS.md), and [remote comparison](https://github.com/austery/local-asr-service/compare/97028ba0a3b8c5774903ace18cb7fb29456d9bb5...2ff1a34db7d6999714b6df64a313c56e484bc171).

## Corrections to the supplied roadmap

1. Add a source/contract preflight before changing the environment. Candidate selection must establish that a released runtime includes the intended MOSS implementation and usable checkpoints.
2. Upgrade one direct runtime at a time, but inspect its full dependency diff. A change to a shared package can affect the other runtime even when its own version is unchanged.
3. Start MOSS with an isolated probe. The current `ModelSpec` has no discovery-only/requestable switch, unlike `PipelineProfile`; adding a normal registry entry prematurely advertises the model.
4. Treat real audio as the evaluation corpus. Only reviewed human references are ground truth; Paraformer output and YouTube captions are comparison material until checked.
5. Test absolute English usability as well as improvement over Paraformer. A win against a Mandarin-focused baseline alone is insufficient.
6. Keep cleanup last. MOSS success does not establish that all alignment, diarization, worker, and chunking code is unreferenced, or that any caller can tolerate removal.
7. Preserve the FunASR patch through the upgrade. Removing it requires a reproducer against the selected upstream version and an explicit revision of the existing project instruction that prohibits removal.

## Upstream preflight findings

### Candidate runtimes

The current published MLX Audio release is **0.5.1** (2026-08-31). Its package metadata requires `transformers>=5.14.0`, above the repository's locked 5.8.1. The MOSS addition appears in the **0.4.5** release history. Evaluate a bounded candidate set: latest stable first, then an earlier MOSS-capable stable release only if there is a specific compatibility reason. Neither is accepted yet. Sources: [MLX Audio releases](https://github.com/Blaizzy/mlx-audio/releases), [0.5.1 package metadata](https://pypi.org/pypi/mlx-audio/0.5.1/json).

The current published FunASR release is **1.4.14** (2026-09-03); its metadata requires `numpy<2`, while the current lock contains 2.3.5. This creates a second shared-dependency change to examine. Re-run MLX checks after the later FunASR upgrade as well. Latest publication is a candidate, not evidence of suitability or of a fixed CAM++ bug. Source: [FunASR 1.4.14 metadata](https://pypi.org/pypi/funasr/1.4.14/json).

### MOSS integration seams

The target is **MOSS-Transcribe-Diarize 0.9B**, not another MOSS family model. The publisher advertises multilingual, timestamped speaker transcription and recordings up to 90 minutes. This is an upstream claim; the MLX implementation on this Mac and this service's preprocessing must pass independently. Source: [official model card](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize).

The inspected MLX Audio 0.5.1 tag resolves to `6b54ec6ecd99d0ad77dfa33dd129707e31bf051c`. Its MOSS implementation returns `speaker_id` and tagged text, defaults to 2,048 output tokens, and can fall back to a full-duration segment without a speaker when parsing finds no valid segments. The shared wrapper filters generation arguments against the model signature; MOSS has no explicit `language` parameter. Sources: [MOSS implementation](https://github.com/Blaizzy/mlx-audio/blob/6b54ec6ecd99d0ad77dfa33dd129707e31bf051c/mlx_audio/stt/models/moss_transcribe_diarize/moss_transcribe_diarize.py#L555), [generation wrapper](https://github.com/Blaizzy/mlx-audio/blob/6b54ec6ecd99d0ad77dfa33dd129707e31bf051c/mlx_audio/stt/generate.py#L246).

These observations define probes, not a concluded defect report:

| Seam | Existing service behavior | Evidence needed before promotion |
| --- | --- | --- |
| Speaker identity | `src/api/routes.py:368` reads `speaker`; MLX normalization currently preserves dictionary keys | Prove `speaker_id` survives normalization as the public `speaker` field; keep anonymous identity stable within the entire recording. |
| Text | The model emits timestamp/speaker markup in raw text | Preserve raw output for diagnosis; define clean public text and segment text without duplicating or losing words. |
| Output budget | `src/core/mlx_engine.py:213` does not pass a token budget | Select a bounded duration-appropriate budget and detect truncation; an HTTP 200 is insufficient. |
| Language | The existing engine forwards a normalized language argument | Verify whether auto and explicit language are honored or unsupported. Do not advertise detection or forcing based only on multilingual recognition. |
| Long audio | `src/adapters/audio_chunking.py:102` may split after 50 minutes; `src/core/mlx_engine.py:247` offsets results using the previous final timestamp | Test missing tails, trailing silence, overlap duplication, timestamp origin, and speaker reuse across independent calls. Prefer upstream handling of the whole recording when feasible. |
| Parse failure | Upstream may produce a broad fallback segment | Treat missing speaker/timeline evidence as a failed diarized-ASR probe; never fabricate labels to make the output appear valid. |
| File output | The upstream wrapper defaults to writing `transcript.*` | Run probes in an isolated output directory, preserve full results, and verify temp-file ownership/cleanup for any service integration. |

A small typed adapter change may be justified. A new `MossEngine`, custom embeddings, alignment rescue, or speaker reconciliation framework is outside this effort. If the target workload needs those, record No-Go or Defer.

## Proposed route and exit gates

| Stage | Work required to support the decision | Exit condition | Failure route |
| --- | --- | --- | --- |
| Agree and freeze | Reconcile main; identify hardware/toolchain/checkpoints; choose and review corpus; capture baseline output and costs | Lei accepts the corpus, quality rubric, practical runtime budget, and immutable evidence manifest | Fill the missing evidence; do not upgrade first. |
| MLX runtime compatibility | In isolation, resolve the chosen MLX Audio version, inspect dependency drift, and run unchanged Qwen3 plus shared-runtime/service regressions | An exact runtime/lock candidate passes the agreed non-regression gate | Keep the original environment; investigate a narrow compatibility fix or defer MOSS. |
| MOSS adoption | Probe raw upstream output, then a bounded adapter experiment; compare English multi-speaker recordings and a Chinese control | Lei records Go, No-Go, or Defer, with supported duration/language and contract evidence | Preserve useful results; do not build a replacement diarization pipeline. |
| FunASR compatibility | Separately upgrade runtime with ASR, VAD, punctuation, and CAM++ checkpoints fixed; test Paraformer and SenseVoice plus shared MLX paths | An exact candidate is accepted, or retaining 1.2.7 is explicitly justified; patch disposition is recorded | Retain the previous lock and patch. MOSS No-Go does not block this stage indefinitely. |
| Retirement and handoff | Review model roles, caller dependencies, evidence retention, rollback, and separate implementation changes | Lei approves the final handoff and a bounded retirement list | Retain uncertain shared code; record a follow-up decision only if needed for this destination. |

If the MLX gate rejects all bounded candidates, resolve the MOSS decision as Defer or No-Go with Lei, then proceed to the independent FunASR decision. Do not leave downstream work waiting forever for a model that cannot be evaluated.

## Proposed evaluation contract

Use the accepted roadmap and the numeric proposals below as the initial working rubric. Record the actual corpus, reference windows, and practical machine budget before inspecting challenger results. Any necessary change to a threshold must be explained before comparison; do not manufacture another general planning interview.

### Small real corpus

Use six recordings where available; combine stress characteristics rather than create a large benchmark platform.

| Slot | Recording | Purpose |
| --- | --- | --- |
| Chinese single speaker | 10–30 minutes | Preserve Qwen3 Chinese transcription. |
| English single speaker | 10–30 minutes | Preserve Qwen3 English transcription. |
| English conversation | 30–60 minutes, two speakers | Primary MOSS quality and recurring-speaker test. |
| English meeting | 30–60 minutes, three or more speakers | Independent primary test; include turn-taking, noise, or overlap. |
| Chinese conversation | 30–60 minutes, multiple speakers | Paraformer control and MOSS secondary comparison. |
| Mixed-language conversation | Multiple speakers with Chinese/English switches | Language and difficult-segment stress case. |

At least one multi-speaker recording must exceed the configured 50-minute service split boundary. Preserve an English recording or reference window as a holdout from any prompt/budget tuning. The existing 60-second English fixture is a smoke test, not the promotion corpus. Existing long English fixtures are candidates until duration, content, and suitability are verified.

For each file, record a stable ID, content hash, actual duration, local location, language, known speaker count, relevant acoustic characteristics, and reference windows. Keep private audio and full transcripts local; publish only deliberately selected aggregate findings.

Manually correct short windows at the start, middle, and end, plus a speaker-return/overlap/chunk-boundary window when applicable. Fix the windows before comparison. Preserve recording-wide completeness checks separately: good window scores do not prove that the entire file was transcribed.

### Acceptance rubric

| Dimension | Proposed check |
| --- | --- |
| Runtime non-regression | On identical reviewed windows, WER/CER must not worsen by more than one absolute percentage point, with no new material omission or meaning-changing error. Investigate violations before accepting a runtime. |
| MOSS English value | Proposed starting target: at least 20% relative WER reduction against frozen Paraformer on each English multi-speaker sample's reviewed windows, plus Lei's judgment that the output is usable. Use Qwen3 as a text-quality reference without treating it as a diarization competitor. |
| Speaker attribution | Proposed target: at least 95% correct speaker attribution over reviewed non-overlapping speech duration after one recording-wide label mapping. Review overlap separately. Renaming S01/S02 between models or runs is not itself an error; changing a speaker's identity within a recording is. |
| Completeness | Inspect the beginning and end, speech coverage, repeated/skipped passages, text/segment agreement, parse fallbacks, and token-limit termination. A plausible full-duration final timestamp does not prove coverage. |
| Timestamps | Require finite, in-bounds, positive-duration segments and truthful time origins. Review sampled alignment against audio. Distinguish legitimate simultaneous speech from invalid ordering; do not require all speakers' intervals to be disjoint. |
| API behavior | JSON/verbose JSON, text, and the subtitle formats claimed for the model must preserve the established public contract. Capability rejection must occur before queuing. Unsupported capabilities must be reported truthfully. |
| Stability | No OOM, process crash, silent truncation, corrupted output, or lingering owned worker/temp files on the approved corpus. Check timeout, failure recovery, model switching, idle offload, and clean shutdown. |
| Cost | Measure end-to-end cold and warm wall time, RTF = elapsed/audio duration, process-tree peak RSS, and available MLX memory telemetry. RSS alone is not total Metal/unified-memory usage. |
| Regression investigation thresholds | Start with a 20% wall-time or 25% RSS increase as a review trigger, not an automatic quality-first rejection. Re-run a suspicious comparison under matched load; agree the actual maximum runtime/memory budget with Lei. |

Do not average away a failed English recording, missing speaker coverage, or failed long-form run. Mark unmeasured criteria as unmeasured. If the results are too close to judge, Defer is preferable to a vague promotion.

### Reuse existing tools

Start from [phase3_evaluation.py](../../benchmarks/phase3_evaluation.py), which already probes the HTTP API, measures duration/RTF, summarizes segments/SRT, and samples process-tree RSS. Retain full raw HTTP and upstream outputs as well: a summary or text preview is insufficient for quality review.

Its current segment `monotonic` check treats any overlap as non-monotonic (`benchmarks/phase3_evaluation.py:185`). Interpret that flag against actual overlapping speech instead of using it as an unconditional diarization failure. Add only the missing evidence capture needed for this evaluation; no benchmark framework rewrite.

The implementation handoff should include these existing checks, using a frozen selected lock:

```bash
uv sync --frozen --dev --prerelease=allow
uv run --frozen ruff check .
uv run --frozen tach check
uv run --frozen python -m pytest tests/unit tests/integration tests/reliability
uv run --frozen mypy src/
```

The current CI runs Ruff, Tach, and unit tests. Establish their baseline result before upgrades. Record existing failures separately; do not silently fix unrelated code. Run the applicable real-model E2E tests and corpus probes on Apple Silicon, with one inference worker and no competing benchmark server. Validate a clean isolated frozen installation separately from the existing `.venv`.

## Implementation boundaries after the decisions

The final handoff will select exact versions and scope; the following is the proposed delivery order, not work performed in this session:

1. **Baseline evidence:** reconcile with remote main, preserve user files, and record the accepted corpus/checkpoint/environment manifest. Avoid mixing this with a runtime upgrade.
2. **MLX Audio runtime:** a dedicated branch/PR, for example `codex/mlx-audio-runtime-upgrade`. Include necessary shared-dependency changes and bounded compatibility fixes with before/after evidence. Do not change the Qwen3 checkpoint or add MOSS here.
3. **MOSS promotion, only on Go:** a separate `codex/moss-transcribe-diarize` branch/PR. Use the existing MLX runtime contract, minimal typed normalization, verified capability metadata, public-behavior tests, and real-service evidence. Pin the exact checkpoint revision/precision. Keep the existing defaults and model roles.
4. **FunASR runtime:** a separate `codex/funasr-runtime-upgrade` branch/PR after the MOSS verdict, with Paraformer/CAM++/VAD/punctuation weights fixed. Verify SenseVoice and recheck MLX when shared packages change.
5. **Retirement:** a separate reviewable change after runtime/adoption decisions. Retire `qwen3-sortformer` only after checking caller use, shared code references, replacement coverage, and retained historical evidence. No-Go removes only the experiment additions that were actually made; it does not justify broad deletion.

For each stage, capture the accepted base SHA and lock hash. Run experiments in a separate environment/output directory so the known-good service remains usable. Rollback uses the previous accepted checkout, lock, and checkpoint revisions. A source-only revert with upgraded packages is not a rollback.

Commits, pushes, merges, destructive cleanup, and changing the FunASR patch instruction remain subject to Lei's existing approval rules. Show each concrete diff before requesting approval. Do not delete caches or model downloads as routine closeout.

## Decisions intentionally deferred

- Exact MOSS checkpoint ID, immutable revision, and precision; whether the released runtime loads it cleanly without custom conversion.
- The feasible supported duration and generation budget on the current Mac; any narrow upstream-supported way to preserve speaker identity across service chunking.
- Exact runtime versions and shared-dependency pins after compatibility evidence.
- Final caller migration and retirement file list after the adoption verdict and a reference audit.

These questions belong to the existing decision tickets. More detailed implementation slices should be written only when their dependencies are resolved.

## Out of scope

No new Voxtral Realtime, FireRed/FireRedASR2, Qwen3-ASR 0.6B, Fun-ASR-Nano, or Parakeet effort. No further Qwen3-Sortformer optimization, custom MOSS diarization rescue, broad refactor, upstream fork/model conversion project, automatic model routing, or attempt to replace all specialist models with MOSS. Apple Speech and SenseVoice keep their current roles; this map does not redesign their product paths.

## Next session

Open the canonical map, query the frontier, and claim **Agree the evaluation corpus and acceptance gates**. Use the accepted working rubric, identify the actual recordings and missing reference evidence, then capture the baseline required to resolve that decision. Request only genuinely missing input; the overall roadmap is already accepted. Do not treat the current planning/source inspection as a passed baseline or as an accepted upgrade.
