# Select a MOSS-capable MLX Audio runtime without regressions

Parent: [Runtime Upgrade and MOSS Adoption Decision Map](../map.md)
Label: wayfinder:research
Type: research
Mode: AFK
Status: open
Assignee: unassigned
Blocked by: 01

## Question

Can a released MOSS-capable MLX Audio runtime preserve the accepted Qwen3 and shared service/runtime behavior, and which exact version and dependency set should the handoff use?
Why now: MOSS evaluation depends on a compatible runtime; changing one direct package may still move shared dependencies.
Complete when: source/checkpoint compatibility, isolated frozen installation, lock diff, unchanged-model regression evidence, and rollback identify an accepted candidate or justify retaining the baseline.
Evidence: [Roadmap and upstream preflight](../../../docs/plans/2026-09-07-runtime-upgrades-and-moss-evaluation.md).
