# Decide whether MOSS solves English long-form multi-speaker transcription

Parent: [Runtime Upgrade and MOSS Adoption Decision Map](../map.md)
Label: wayfinder:prototype
Type: prototype
Mode: HITL
Status: resolved
Assignee: LeiP (Codex execution)
Blocked by: 02

## Question

Does MOSS meet the agreed English transcription, speaker, timestamp, completeness, lifecycle, and cost gates through a thin MLX adapter?
Why now: this verdict determines promotion scope and whether the existing pipeline has a useful replacement.
Complete when: Lei reviews an isolated real-corpus probe and records Go, No-Go, or Defer with checkpoint/runtime identity, supported duration/languages, API evidence, and limitations; Chinese comparison is secondary.
Evidence: [Roadmap, integration seams, and acceptance rubric](../../../docs/plans/2026-09-07-runtime-upgrades-and-moss-evaluation.md).

## Delivery update — 2026-09-07

PR #30 merged after review; local main and its frozen environment were synchronized at `55909f1`. Earlier comments and answers retain their evaluation-time scope. See the [changelog](../../../CHANGELOG.md) for current delivery status.

## Comments

- 2026-09-07 — Isolated upstream probe started under the execution override. Runtime source is unchanged, all eight real-corpus transcription texts match, and remaining attribution/contract evidence continues separately. This claim does not resolve the runtime dependency or authorize promotion. Review windows were fixed before MOSS inference.

- 2026-09-07 — Lei judged reviewed MOSS recognition and speaker distinctions positively and requested duration/token-limit research. The previous No-Go recommendation was not adopted; keep this decision open and retain the candidate. See [listening acceptance and follow-up](../evidence/2026-09-07-listening-acceptance-and-moss-followup.md).

- 2026-09-07 — Duration follow-up completed after Lei requested continued validation: the first 20 minutes and two disjoint 30-minute All-In inputs pass bounded coverage checks. The 40-minute input selects EOS token 151645 at 16,376 output tokens and omits 4:47; the original 60-minute input has the same output count and omits 23:41. Propose a 30-minute English experimental single-input scope, while retaining this ticket for thin-adapter and failure/lifecycle verification. See [duration evidence](../evidence/2026-09-07-moss-duration-validation.md).


## Answer

- Lei accepted the duration listening results and requested MOSS introduction as a separate reviewed PR. **Go** for explicit English continuous-speech recordings up to 30 minutes; **Defer** unrestricted 30–60-minute and other-language claims.
- The thin existing-MLX adapter pins the evaluated checkpoint, makes one upstream call, normalizes speaker/text/timestamps, and rejects oversized or detectably incomplete output. No model internals or speaker reconciliation are added.
- The public 30-minute response exactly matches the accepted normalized text and segments; short/medium, formats, 400/422/500 handling, recovery, offload, switching, deadline termination, and cleanup were exercised.
- 353 mocked/runtime-contract tests plus the real repository E2E test pass; Ruff, Tach, and build pass. Ten pre-existing type diagnostics and unmeasured numerical quality remain disclosed.
- See [adapter admission](../evidence/2026-09-07-moss-adapter-admission.md). This resolves the bounded adoption decision; publishing/merging the MOSS PR still awaits the delivery approval and external review.
