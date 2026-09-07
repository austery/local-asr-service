# Runtime Candidate Delivery Handoff — 2026-09-07

Status: Both runtime upgrades accepted by Lei after listening. MOSS remains an experimental candidate under duration/early-stop investigation. No production promotion was performed.

## Delivery boundaries

| Candidate | Local branch | Worktree | Exact lock SHA-256 |
| --- | --- | --- | --- |
| MLX Audio 0.5.1; FunASR stays 1.2.7 | `codex/mlx-audio-runtime-upgrade` | `/Users/leipeng/Documents/Projects/local-asr-service-worktrees/runtime-roadmap` | `e04a6f4e59d950b58b2dac8e239162b04ec5eb0c833f0a90b3b38d5369cf7bb5` |
| FunASR 1.4.14 on the MLX candidate; NumPy 1.26.4 | `codex/funasr-runtime-upgrade` | `/Users/leipeng/Documents/Projects/local-asr-service-worktrees/funasr-runtime-upgrade` | `2c3ced2b2d2b9ffba961e0c3b0d723442077d96bcd8c1f457c4112bd030f6b4b` |

The FunASR candidate is stacked on the MLX candidate so its incremental dependency diff can be reviewed separately. Local commits freeze the tested declarations and locks; they do not grant acceptance, push, merge, or rollout approval. The original main checkout remains at `97028ba0a3b8c5774903ace18cb7fb29456d9bb5`. Runtime source is unchanged from the reconciled `bb501c24f86b1530254e7e08cc7ed2a478abfd12` baseline.

Private logs, audio, complete responses, checkpoint manifests, probe scripts, and listening pages referenced by these reports are under `/Users/leipeng/Documents/Projects/local-asr-service-worktrees/runtime-roadmap/.runtime-validation/`. They remain local and gitignored. The reports below contain aggregate evidence only.

## Requirement-by-requirement status

| Requirement | Evidence | Current verdict |
| --- | --- | --- |
| Reconcile remote state without changing the running main checkout | Worktree base includes `2ff1a34`; original main remains `97028ba` | Established |
| Freeze environment, checkpoints, corpus, and old runtime behavior | Original lock hash, Python/macOS/hardware identity, Qwen revision, 18 unchanged FunASR checkpoint/config hashes, four hashed corpus files, baseline HTTP outputs and rollback runs | Measured baseline established; reviewed human references and mixed/noisy coverage remain incomplete |
| Upgrade MLX Audio with unchanged Qwen checkpoint and shared-runtime verification | [MLX report](2026-09-07-runtime-validation.md); 308 tests, static checks, eight identical texts, real contracts, rollback and lifecycle | Technical comparison completed and upgrade quality accepted by Lei; no interval-level accuracy score |
| Evaluate MOSS on English long-form multi-speaker audio | [MOSS report](2026-09-07-moss-probe.md); short, medium and one-hour upstream runs using one pinned BF16 checkpoint | Current pairing fails completeness at 36:19 of 60:00; Lei requested continued investigation and retained the candidate |
| Prove a thin MOSS HTTP adapter before promotion | Raw upstream long-form gate failed; no MOSS alias or adapter was introduced | Not reached; do not advertise capabilities or build a repair pipeline to claim success |
| Independently upgrade FunASR with ASR/VAD/punctuation/CAM++ checkpoints fixed | [FunASR report](2026-09-07-funasr-preflight.md); clean install, 308 tests, eight real requests, unchanged hashes, preserved Qwen/pipeline contracts | Runtime accepted by Lei after listening; changed text and repeated cost increases are disclosed |
| Determine CAM++ patch disposition | Both unpatched upstream call sites fail the None-timestamp reproducer; both repository-patched sites pass | Retain the existing patch unchanged |
| Preserve single-worker execution, failure recovery, idle offload and shutdown | Isolated port 50710, serial model processes, corrupt-audio recovery, recorded worker exits and server exit code 0 | Established for the exercised paths; no claim about every possible model/format/failure |
| Bound retirement using actual references | Reference audit below | Proposed production retirement list is empty; final handoff approval remains open |
| Preserve honest verification claims | Existing 12 mypy errors remain explicitly reported; output differences are not WER/CER; no human-reference scores invented | Maintained |

## Reference-audited retirement proposal

Retain the existing model roles and experimental pipeline for this delivery. No production MOSS integration was added, so there is no new production MOSS code to remove after a No-Go. Preserve the private probe evidence for diagnosis and later decisions.

The existing pipeline is active and reachable, rather than demonstrably unused:

- `src/core/pipeline_registry.py:18` declares the `qwen3-sortformer` profile as requestable.
- `src/api/routes.py:322` dispatches explicit pipeline requests through `submit_pipeline`.
- `src/services/transcription.py:257` owns that submission path and its resource scheduling.
- `src/workers/model_worker.py:50` and `:61` create diarization and alignment stages; `:179` and `:191` execute them.
- `tests/integration/test_model_api.py:352` verifies explicit requestability. Real short pipeline output was identical across the tested runtime environments.
- `README.md:153` and `MODELS.md:123` document its explicit experimental availability.

This audit establishes local reachability. It does not establish the absence of external consumers. Any later deletion proposal must identify the actual caller transition and separately review shared worker, alignment, diarization, and chunking ownership. There is no evidence-backed production deletion in the current proposal.

## Reproduction and rollback

Use Python 3.11 and the selected worktree's checked-in lock. A fresh checkout can install its candidate with `uv sync --frozen --dev --prerelease=if-necessary`. Do not run this command against the original main checkout. Runtime inference must use one worker and a separate service port/output directory.

For the original baseline, the accepted source is `bb501c2`, lock SHA-256 is `2fb29a6c4391a5931266cacea563663ad01ab60416ca8677a64f06d6458b7298`, and the original checkpoint identities remain in the private manifest. `.venv-baseline` is retained and was exercised after the MLX candidate. A real rollback selects that complete source/lock/checkpoint combination; reverting only a declaration while using newer installed packages is insufficient.

For a future FunASR rollout rollback, select the MLX-only candidate's source and `e04a6f4e…` lock in its own environment; `.venv-mlx` remains available. The actual production environment was never upgraded during this work.

## Decisions needed before rollout

1. Continue MOSS duration/early-stop investigation; official 90-minute support is not a locally verified reliability guarantee. See [upstream research](2026-09-07-moss-upstream-limits.md).
2. Both runtime selections now have explicit qualitative user acceptance; see [acceptance and timing](2026-09-07-listening-acceptance-and-moss-followup.md). Numeric quality scores remain unmeasured.
3. Finalize MOSS admission/adapter scope and the proposed empty retirement list. Show the selected diff and obtain explicit approval before any new push; main merge or deployment remains a separate action.
