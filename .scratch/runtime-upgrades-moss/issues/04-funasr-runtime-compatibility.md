# Select a FunASR runtime and determine the CAM++ patch disposition

Parent: [Runtime Upgrade and MOSS Adoption Decision Map](../map.md)
Label: wayfinder:research
Type: research
Mode: AFK
Status: claimed
Assignee: LeiP (Codex execution)
Blocked by: 03

## Question

Which FunASR runtime preserves Paraformer/CAM++ and SenseVoice behavior with existing checkpoints, and is there evidence sufficient to propose retiring the None-timestamp patch?
Why now: evaluate this separately after the MOSS verdict so runtime changes remain attributable.
Complete when: isolated install, shared-dependency regression, Chinese multi-speaker/lifecycle evidence, and a source/call-site reproducer support an exact version or retaining 1.2.7; keep the patch unless evidence and an explicit instruction revision permit removal.
Evidence: [Roadmap and FunASR boundaries](../../../docs/plans/2026-09-07-runtime-upgrades-and-moss-evaluation.md).

## Comments

- 2026-09-07 — Independent dependency/install preflight started in a private project copy and a third environment under the execution override, while MOSS evidence is being collected. FunASR 1.4.14 resolves with NumPy 1.26.4 and unchanged MLX Audio 0.5.1. No model has been loaded in this environment, no main or MLX environment has changed, and the MOSS decision remains unresolved.
