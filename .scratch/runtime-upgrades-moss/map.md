# Runtime Upgrade and MOSS Adoption Decision Map

Label: wayfinder:map
Status: open
Assignee: unassigned

## Destination

An approved implementation handoff selecting reproducible runtime versions, deciding whether MOSS fits English long-form multi-speaker transcription, and bounding safe retirement. The map is complete when the remaining implementation has no unresolved admission, compatibility, or scope decision.

## Notes

- Lei accepted the working roadmap on 2026-09-07. The [roadmap and source preflight](../../docs/plans/2026-09-07-runtime-upgrades-and-moss-evaluation.md) contain the evaluation defaults and delivery sequence; measured outcomes remain open. No second general plan-approval round is needed.
- Apply [Wayfinder](/Users/leipeng/Documents/global-skills/wayfinder/SKILL.md) and the accepted [Lightweight Local Speech Gateway Boundary](../../docs/ADR-002-Lightweight-Local-Speech-Gateway-Boundary.md). Consult additional skills only when they materially help the selected decision.
- Prioritize English long-form multi-speaker usability. Preserve Qwen3 single-speaker, Paraformer Chinese multi-speaker, SenseVoice, and Apple Speech roles unless a later explicit decision changes them.
- Execution override (2026-09-07): Lei authorized upgrades and verification in isolated worktrees. Continue technical work autonomously across decision boundaries where evidence collection is independent; final quality acceptance still needs the stated evidence. Promotion judgments, pushes of new diffs, merges, and destructive cleanup retain their applicable approval gates.
- Local tracker: child files under `issues/`; `Status: open` is unclaimed. Claim by setting `Assignee` to the driving developer and `Status: claimed` before work. Append the answer under `## Answer`, set `Status: resolved`, and add a named context link below. Keep answers to five bullets, linking detailed evidence.
- Blocking is the local tracker's `Blocked by` convention. Scan children in numeric order; the frontier contains open, unassigned children whose blockers are all resolved. Use the title when referring to any ticket.
- Track each substantive decision independently; the execution authorization permits continuing through multiple decisions when evidence is sufficient. A rejected runtime candidate can lead to MOSS No-Go/Defer and does not permanently block the independent FunASR decision.
- The documentation branch `codex/runtime-upgrades-moss-roadmap` includes remote main through `2ff1a34db7d6999714b6df64a313c56e484bc171`. Work only in `/Users/leipeng/Documents/Projects/local-asr-service-worktrees/runtime-roadmap` or another isolated worktree; leave the running original main checkout and its environment untouched. The measured baseline and runtime comparisons are captured; Lei has accepted both runtime upgrades after listening; MOSS long-form admission remains open.
- These map/ticket files are the maintained, versioned local tracker. Original untracked copies in the running checkout are historical drafts. Preserve its existing untracked `transcript.json`.

## Decisions so far

- [Agree the evaluation corpus and acceptance gates](issues/01-evaluation-contract.md): the measured four-file basis and qualitative user review support the runtime decisions; numeric scores remain unmeasured.
- [Select a MOSS-capable MLX Audio runtime without regressions](issues/02-mlx-runtime-compatibility.md): select MLX Audio 0.5.1 with the tested frozen lock.
- [Select a FunASR runtime and determine the CAM++ patch disposition](issues/04-funasr-runtime-compatibility.md): select FunASR 1.4.14 / NumPy 1.26.4 and retain the CAM++ patch.

## Not yet specified

Any additional compatibility or caller-transition decision exposed by the runtime/adoption evidence. Create a new ticket only when its question becomes precise and an independent approval or tracking boundary is needed; keep detailed implementation slicing for the handoff.

## Out of scope

custom model conversion or diarization recovery frameworks; new model families beyond MOSS; further Qwen3-Sortformer optimization; automatic routing; replacing all existing specialists; broad refactors; routine cache deletion.


## Current execution evidence

- [MLX runtime comparison](evidence/2026-09-07-runtime-validation.md): eight identical transcription texts, preserved tested contracts, upgrade quality accepted by Lei; speaker intervals remain unscored individually.
- [MOSS upstream probe](evidence/2026-09-07-moss-probe.md): the current 60-minute candidate stops at 36:19; Lei requested retaining MOSS and researching long-form limits; the earlier No-Go recommendation was not adopted.
- [FunASR candidate](evidence/2026-09-07-funasr-preflight.md): clean install and 308 existing tests pass; all real requests succeed; Lei accepted the changed text after listening; repeated cost increases remain disclosed.
- All validation jobs are finished. Original main remains at `97028ba0a3b8c5774903ace18cb7fb29456d9bb5`. Candidate dependencies and evidence are frozen in local branches; no new push or production integration has occurred. See the [delivery handoff](evidence/2026-09-07-delivery-handoff.md) for the branch boundaries and remaining decisions.
