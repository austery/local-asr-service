# Runtime Upgrade and MOSS Adoption Decision Map

Label: wayfinder:map
Status: resolved
Assignee: LeiP (Codex execution)

## Destination

An approved implementation handoff selecting reproducible runtime versions, deciding whether MOSS fits English long-form multi-speaker transcription, and bounding safe retirement. The map is complete when the remaining implementation has no unresolved admission, compatibility, or scope decision.

## Notes

- Lei accepted the working roadmap on 2026-09-07. The [roadmap and source preflight](../../docs/plans/2026-09-07-runtime-upgrades-and-moss-evaluation.md) contain the evaluation defaults and delivery sequence; outcomes are recorded below. No second general plan-approval round is needed.
- Apply [Wayfinder](/Users/leipeng/Documents/global-skills/wayfinder/SKILL.md) and the accepted [Lightweight Local Speech Gateway Boundary](../../docs/ADR-002-Lightweight-Local-Speech-Gateway-Boundary.md). Consult additional skills only when they materially help the selected decision.
- Prioritize English long-form multi-speaker usability. Preserve Qwen3 single-speaker, Paraformer Chinese multi-speaker, SenseVoice, and Apple Speech roles unless a later explicit decision changes them.
- Execution override (2026-09-07): Lei authorized upgrades and verification in isolated worktrees. Continue technical work autonomously across decision boundaries where evidence collection is independent; final quality acceptance still needs the stated evidence. Promotion judgments, pushes of new diffs, merges, and destructive cleanup retain their applicable approval gates.
- Local tracker: child files under `issues/`; `Status: open` is unclaimed. Claim by setting `Assignee` to the driving developer and `Status: claimed` before work. Append the answer under `## Answer`, set `Status: resolved`, and add a named context link below. Keep answers to five bullets, linking detailed evidence.
- Blocking is the local tracker's `Blocked by` convention. Scan children in numeric order; the frontier contains open, unassigned children whose blockers are all resolved. Use the title when referring to any ticket.
- Track each substantive decision independently; the execution authorization permits continuing through multiple decisions when evidence is sufficient. A rejected runtime candidate can lead to MOSS No-Go/Defer and does not permanently block the independent FunASR decision.
- PRs #28, #29, and #30 are merged. Local main and its frozen environment were synchronized at `55909f1` on Lei's request; 353 post-sync tests passed. Future edits stay in isolated worktrees.
- These map/ticket files are the versioned decision tracker. The [changelog](../../CHANGELOG.md) and [current roadmap](../../docs/plans/2026-09-07-runtime-upgrades-and-moss-evaluation.md) distinguish merged delivery from follow-up work.


## Decisions so far

- [Agree the evaluation corpus and acceptance gates](issues/01-evaluation-contract.md): the measured four-file basis and qualitative user review support the runtime decisions; numeric scores remain unmeasured.
- [Select a MOSS-capable MLX Audio runtime without regressions](issues/02-mlx-runtime-compatibility.md): select MLX Audio 0.5.1 with the tested frozen lock.
- [Select a FunASR runtime and determine the CAM++ patch disposition](issues/04-funasr-runtime-compatibility.md): select FunASR 1.4.14 / NumPy 1.26.4 and retain the CAM++ patch.

- [Decide whether MOSS solves English long-form multi-speaker transcription](issues/03-moss-adoption.md): Go for bounded English recordings up to 30 minutes; broader duration/language claims deferred.
- [Approve the retirement scope and implementation handoff](issues/05-retirement-and-handoff.md): the three PRs merged after review. Lei subsequently requested retirement of the unused `qwen3-sortformer` public entry in a separate follow-up; preserve standalone Qwen3 and historical evidence.

## Implementation follow-up

The original admission decisions are resolved. The [PureSubs MOSS integration roadmap](../../docs/plans/2026-09-07-puresubs-moss-integration.md) is the next caller task: enforce actual chunk duration as well as size, preserve chunk-local speaker identity, and verify a real long recording. It is planned, not implemented. Swagger/documentation refresh and public profile retirement are being prepared in a separate ASR worktree and remain unmerged.

## Out of scope

custom model conversion or diarization recovery frameworks; new model families beyond MOSS; further Qwen3-Sortformer optimization; automatic routing; replacing all existing specialists; broad refactors; routine cache deletion.


## Current execution evidence

- [MLX runtime comparison](evidence/2026-09-07-runtime-validation.md): eight identical transcription texts, preserved tested contracts, upgrade quality accepted by Lei; speaker intervals remain unscored individually.
- [MOSS upstream probe](evidence/2026-09-07-moss-probe.md): two disjoint 30-minute samples pass bounded coverage checks; 40/60-minute inputs stop early at 16,376 output tokens. See [duration validation](evidence/2026-09-07-moss-duration-validation.md). The implemented scope is a provisional 30-minute English opt-in adapter with completeness rejection; see [adapter admission](evidence/2026-09-07-moss-adapter-admission.md).
- [FunASR candidate](evidence/2026-09-07-funasr-preflight.md): clean install and 308 existing tests pass; all real requests succeed; Lei accepted the changed text after listening; repeated cost increases remain disclosed.
- All original evaluation jobs finished and the three reviewed PRs merged. The [delivery handoff](evidence/2026-09-07-delivery-handoff.md) is historical pre-publication evidence; see the [changelog](../../CHANGELOG.md) for the merge/synchronization state.

The decision map is resolved. Caller migration and the current unmerged follow-up remain separate delivery work.
