# Runtime and MOSS Delivery Handoff — 2026-09-07

Status: Both runtime upgrades and bounded MOSS listening quality accepted by Lei. The MOSS HTTP adapter is implemented and locally validated. Three PR diffs are ready locally; no new feature push, PR publication, remote merge, or production rollout has occurred.

## Delivery boundaries

| Order | Branch / base | Scope | Lock SHA-256 |
| --- | --- | --- | --- |
| 1 | `codex/mlx-audio-runtime-upgrade` / remote `main` | Baseline/decision records and mlx-audio 0.5.1; runtime source unchanged | `e04a6f4e59d950b58b2dac8e239162b04ec5eb0c833f0a90b3b38d5369cf7bb5` |
| 2 | `codex/funasr-runtime-upgrade` / MLX branch | FunASR 1.4.14, NumPy 1.26.4 and necessary transitives; only pyproject/lock | `2c3ced2b2d2b9ffba961e0c3b0d723442077d96bcd8c1f457c4112bd030f6b4b` |
| 3 | `codex/moss-transcribe-diarize` / FunASR branch | Bounded opt-in MOSS alias, normalization, worker deadline, tests and admission docs | Same as FunASR |

Worktrees are under `/Users/leipeng/Documents/Projects/local-asr-service-worktrees/`: `runtime-roadmap`, `funasr-runtime-upgrade`, and `moss-transcribe-diarize`. Private evidence/media remain in each worktree's ignored `.runtime-validation/`; the original corpus and duration listening pages remain in `runtime-roadmap`.

MLX head is `e0f602b6606ad88c59d75ede7511a38d5f68c5a9`. FunASR head is `0ff0a22bbc032a481e38ee4e6f70c3eae95577a6`, a non-destructive merge that reconciles the accepted MLX documentation ancestry while preserving the dependency-only incremental diff. MOSS's publication head should be read from its branch at delivery time.

## Acceptance and verification

- [MLX runtime comparison](2026-09-07-runtime-validation.md): eight identical top-level texts, preserved tested contracts, user listening acceptance, fixed checkpoints, and baseline rollback.
- [FunASR comparison](2026-09-07-funasr-preflight.md): user accepted recognition changes, all 18 model/config hashes unchanged, shared Qwen behavior retained, documented Chinese inference and SenseVoice memory costs. Keep the existing CAM++ patch at both call sites.
- [MOSS admission](2026-09-07-moss-adapter-admission.md): Go for explicit English continuous speech up to 30 minutes. Public 30-minute text/segments match the accepted normalized result exactly. Inputs over 30 minutes are rejected; malformed/incomplete output is rejected conservatively. Deadline termination, failure recovery, request artifact cleanup, idle offload, switching, and shutdown were exercised.
- Final test categories all passed: MLX 308 + 1 real E2E; FunASR 308 + 1 real E2E; MOSS 353 + 1 real E2E. Ruff, Tach, sdist, and wheel passed. Existing type diagnostics are 12 on the runtime baseline and 10 on MOSS; numerical WER/CER and speaker accuracy remain unmeasured.
- Separate Standards/Spec reviews found no actionable implementation defect after checkpoint/cleanup fixes. The requested independent review of the published MOSS PR remains a delivery gate.

## Retirement and rollback

Retirement list: **empty**. Keep all current aliases and the requestable `qwen3-sortformer` experiment. Current references include `src/core/pipeline_registry.py:28`, `src/api/routes.py:322`, `src/services/transcription.py:257`, and `src/workers/model_worker.py:73`. Local reachability does not establish the absence of external callers, so no deletion or caller migration is justified here.

For baseline rollback, retain source `bb501c2`, lock SHA-256 `2fb29a6c4391a5931266cacea563663ad01ab60416ca8677a64f06d6458b7298`, the checkpoint manifest, and `.venv-baseline`. For a FunASR rollback, use the complete MLX-only source/lock/environment. MOSS is opt-in, so callers can select an existing alias; code rollback should use the full FunASR candidate source/lock. No rollback action is needed against production because it was never changed.

## Remaining delivery steps

1. Show the three final diffs and obtain explicit push approval under the repository rule; PR text is prepared locally.
2. Push feature branches and open the three PRs with the bases above. Run/read remote CI. Keep ancestry coherent when merging the stack; if using squash, reconcile subsequent branches and PR bases explicitly.
3. The first two have user quality/merge acceptance. Keep MOSS open for the user's separate agent review; address any validated findings before merge.
4. Do not update, switch, merge into, or reinstall the running local main checkout. It remains at `97028ba0a3b8c5774903ace18cb7fb29456d9bb5`, with the same pre-existing untracked files, and its port 50700 health check is healthy. Preserve all worktrees and private artifacts.
