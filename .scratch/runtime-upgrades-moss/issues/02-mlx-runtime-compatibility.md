# Select a MOSS-capable MLX Audio runtime without regressions

Parent: [Runtime Upgrade and MOSS Adoption Decision Map](../map.md)
Label: wayfinder:research
Type: research
Mode: AFK
Status: resolved
Assignee: LeiP (Codex execution)
Blocked by: 01

## Question

Can a released MOSS-capable MLX Audio runtime preserve the accepted Qwen3 and shared service/runtime behavior, and which exact version and dependency set should the handoff use?
Why now: MOSS evaluation depends on a compatible runtime; changing one direct package may still move shared dependencies.
Complete when: source/checkpoint compatibility, isolated frozen installation, lock diff, unchanged-model regression evidence, and rollback identify an accepted candidate or justify retaining the baseline.
Evidence: [Roadmap and upstream preflight](../../../docs/plans/2026-09-07-runtime-upgrades-and-moss-evaluation.md).

## Comments

- 2026-09-07 — Technical upgrade/verification started under Lei's execution override while the baseline ticket continues gathering quality evidence. This does not resolve the baseline dependency or grant final runtime acceptance. See [current validation evidence](../evidence/2026-09-07-runtime-validation.md).

## Answer

- Select MLX Audio 0.5.1 with candidate commit `2161f29d844f636e571c72983786e3307362557f` and the frozen lock recorded in the validation report.
- Isolated installation, 308 tests, unchanged eight-case transcription text, public contracts, lifecycle, and baseline rollback support compatibility on the exercised paths.
- Lei reviewed and accepted the upgrade quality. Three speaker differences remain unscored at interval level; no numeric accuracy result is inferred. Existing mypy/timeline/subtitle defects remain documented.
- MOSS admission is independent. This resolution grants no new push, main merge, or rollout action. See [listening acceptance and follow-up](../evidence/2026-09-07-listening-acceptance-and-moss-followup.md).
