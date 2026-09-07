# Select a FunASR runtime and determine the CAM++ patch disposition

Parent: [Runtime Upgrade and MOSS Adoption Decision Map](../map.md)
Label: wayfinder:research
Type: research
Mode: AFK
Status: resolved
Assignee: LeiP (Codex execution)
Blocked by: 03

## Question

Which FunASR runtime preserves Paraformer/CAM++ and SenseVoice behavior with existing checkpoints, and is there evidence sufficient to propose retiring the None-timestamp patch?
Why now: evaluate this separately after the MOSS verdict so runtime changes remain attributable.
Complete when: isolated install, shared-dependency regression, Chinese multi-speaker/lifecycle evidence, and a source/call-site reproducer support an exact version or retaining 1.2.7; keep the patch unless evidence and an explicit instruction revision permit removal.
Evidence: [Roadmap and FunASR boundaries](../../../docs/plans/2026-09-07-runtime-upgrades-and-moss-evaluation.md).

## Delivery update — 2026-09-07

PR #29 merged after review; local main and its frozen environment were synchronized at `55909f1`. Earlier comments and answers retain their evaluation-time scope. See the [changelog](../../../CHANGELOG.md) for current delivery status.

## Comments

- 2026-09-07 — Independent dependency/install preflight started in a private project copy and a third environment under the execution override, while MOSS evidence is being collected. FunASR 1.4.14 resolves with NumPy 1.26.4 and unchanged MLX Audio 0.5.1. No model has been loaded in this environment, no main or MLX environment has changed, and the MOSS decision remains unresolved.

## Answer

- Select FunASR 1.4.14 with NumPy 1.26.4 on the MLX candidate, dependency commit `b1a25c1195056a27b32fcdbcb3e09633f9090efe`.
- Clean installation, 308 tests, eight successful real requests, unchanged checkpoint hashes, preserved shared Qwen/pipeline outputs, and lifecycle evidence support the tested runtime behavior.
- Lei listened to changed recognition text, judged it improved, and explicitly accepted this upgrade. Recorded time/RSS increases remain disclosed; no WER/CER improvement is claimed.
- Retain the CAM++ patch: both unpatched upstream call sites still fail the reproducer. This decision is resolved under the execution override while MOSS remains under investigation.
- See [listening acceptance and follow-up](../evidence/2026-09-07-listening-acceptance-and-moss-followup.md); no new push, main merge, or production installation is authorized by this record.
