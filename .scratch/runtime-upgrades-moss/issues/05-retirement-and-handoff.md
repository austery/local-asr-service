# Approve the retirement scope and implementation handoff

Parent: [Runtime Upgrade and MOSS Adoption Decision Map](../map.md)
Label: wayfinder:grilling
Type: grilling
Mode: HITL
Status: claimed
Assignee: LeiP (Codex execution)
Blocked by: 04

## Question

Given the runtime and MOSS outcomes, which model roles, caller transitions, and experiment-only code removals should the final implementation plan authorize?
Why now: MOSS success alone does not prove that shared pipeline code or caller behavior can be deleted safely.
Complete when: Lei approves a reference-audited retirement list, retained evidence, separately reviewable delivery changes, acceptance checks, and rollback; unresolved implementation-blocking questions become explicit decisions before handoff.
Evidence: [Roadmap and proposed implementation boundaries](../../../docs/plans/2026-09-07-runtime-upgrades-and-moss-evaluation.md).

## Comments

- 2026-09-07 — Independent repository-reference audit and local candidate packaging started under the execution override. The proposed retirement list is empty because the existing pipeline remains requestable and no MOSS replacement has passed admission. This does not resolve the quality or final handoff approval gates. See [delivery handoff](../evidence/2026-09-07-delivery-handoff.md).
