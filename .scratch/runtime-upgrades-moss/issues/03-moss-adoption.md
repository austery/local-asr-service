# Decide whether MOSS solves English long-form multi-speaker transcription

Parent: [Runtime Upgrade and MOSS Adoption Decision Map](../map.md)
Label: wayfinder:prototype
Type: prototype
Mode: HITL
Status: claimed
Assignee: LeiP (Codex execution)
Blocked by: 02

## Question

Does MOSS meet the agreed English transcription, speaker, timestamp, completeness, lifecycle, and cost gates through a thin MLX adapter?
Why now: this verdict determines promotion scope and whether the existing pipeline has a useful replacement.
Complete when: Lei reviews an isolated real-corpus probe and records Go, No-Go, or Defer with checkpoint/runtime identity, supported duration/languages, API evidence, and limitations; Chinese comparison is secondary.
Evidence: [Roadmap, integration seams, and acceptance rubric](../../../docs/plans/2026-09-07-runtime-upgrades-and-moss-evaluation.md).

## Comments

- 2026-09-07 — Isolated upstream probe started under the execution override. Runtime source is unchanged, all eight real-corpus transcription texts match, and remaining attribution/contract evidence continues separately. This claim does not resolve the runtime dependency or authorize promotion. Review windows were fixed before MOSS inference.

- 2026-09-07 — Lei judged reviewed MOSS recognition and speaker distinctions positively and requested duration/token-limit research. The previous No-Go recommendation was not adopted; keep this decision open and retain the candidate. See [listening acceptance and follow-up](../evidence/2026-09-07-listening-acceptance-and-moss-followup.md).
