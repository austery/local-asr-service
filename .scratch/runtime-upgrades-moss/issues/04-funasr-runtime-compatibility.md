# Select a FunASR runtime and determine the CAM++ patch disposition

Parent: [Runtime Upgrade and MOSS Adoption Decision Map](../map.md)
Label: wayfinder:research
Type: research
Mode: AFK
Status: open
Assignee: unassigned
Blocked by: 03

## Question

Which FunASR runtime preserves Paraformer/CAM++ and SenseVoice behavior with existing checkpoints, and is there evidence sufficient to propose retiring the None-timestamp patch?
Why now: evaluate this separately after the MOSS verdict so runtime changes remain attributable.
Complete when: isolated install, shared-dependency regression, Chinese multi-speaker/lifecycle evidence, and a source/call-site reproducer support an exact version or retaining 1.2.7; keep the patch unless evidence and an explicit instruction revision permit removal.
Evidence: [Roadmap and FunASR boundaries](../../../docs/plans/2026-09-07-runtime-upgrades-and-moss-evaluation.md).
