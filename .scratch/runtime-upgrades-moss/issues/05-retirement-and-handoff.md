# Approve the retirement scope and implementation handoff

Parent: [Runtime Upgrade and MOSS Adoption Decision Map](../map.md)
Label: wayfinder:grilling
Type: grilling
Mode: HITL
Status: resolved
Assignee: LeiP (Codex execution)
Blocked by: 04

## Question

Given the runtime and MOSS outcomes, which model roles, caller transitions, and experiment-only code removals should the final implementation plan authorize?
Why now: MOSS success alone does not prove that shared pipeline code or caller behavior can be deleted safely.
Complete when: Lei approves a reference-audited retirement list, retained evidence, separately reviewable delivery changes, acceptance checks, and rollback; unresolved implementation-blocking questions become explicit decisions before handoff.
Evidence: [Roadmap and proposed implementation boundaries](../../../docs/plans/2026-09-07-runtime-upgrades-and-moss-evaluation.md).

## Comments

- 2026-09-07 — Independent repository-reference audit and local candidate packaging started under the execution override. The proposed retirement list is empty because the existing pipeline remains requestable and no MOSS replacement has passed admission. This does not resolve the quality or final handoff approval gates. See [delivery handoff](../evidence/2026-09-07-delivery-handoff.md).


## Answer

- The delivery has no production retirement: keep all current model aliases, the CAM++ patch, and the requestable experimental Qwen/Sortformer pipeline. No caller transition is required for the opt-in MOSS alias.
- Lei requested three independently reviewable PRs: baseline/MLX, incremental FunASR dependencies, then MOSS for another agent's review. Runtime quality acceptance is already recorded.
- Feature branches are reconciled non-destructively. FunASR's diff against MLX is only the declaration and lock; MOSS follows FunASR without dependency changes.
- Preserve baseline environments, private corpus/evidence, worktrees, and the running original main checkout. Use a complete source/lock/checkpoint combination for rollback; no production environment was upgraded.
- [Final handoff](../evidence/2026-09-07-delivery-handoff.md) records tests, implementation scope, and the remaining publication/CI/review steps. Explicit push approval is still required after showing the selected diffs.
