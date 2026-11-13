# Archive index

This directory holds documents whose task has closed: specs, plans,
investigations and handoffs. They are history, not current truth.

Rules (from `~/Documents/global-skills/agents/DOC-LIFECYCLE.md`):

- Move a document here unchanged with `git mv`. Do not edit archived files.
- Add one row below for each archived document: original path, archive date,
  and a pointer to what replaced it.
- Repair incoming links in the same PR. The link check
  (`uv run python scripts/check_doc_links.py`) skips archived files but checks
  this index.
- Agents leave this directory out of default reading. Search it when you need to
  know why something is the way it is.

| Original path | Archived | Successor |
|---|---|---|
| `docs/plans/2026-03-08-subprocess-model-isolation-design.md` | 2026-11-11 | [SPEC-009](SPEC-009-Idle-Model-Offload.md), [`src/workers/model_worker.py`](../../src/workers/model_worker.py), [`src/services/worker_session.py`](../../src/services/worker_session.py) |
| `docs/plans/2026-03-08-subprocess-model-isolation.md` | 2026-11-11 | [SPEC-009](SPEC-009-Idle-Model-Offload.md), [`src/workers/model_worker.py`](../../src/workers/model_worker.py), [`src/services/worker_session.py`](../../src/services/worker_session.py) |
| `docs/superpowers/2026-05-19-qwen3-sortformer-longform-defects.md` | 2026-11-11 | [ADR-002](../ADR-002-Lightweight-Local-Speech-Gateway-Boundary.md), [SPEC-016](SPEC-016-runtime-module-deepening.md) |
| `docs/superpowers/plans/2026-04-30-runtime-aware-asr-pipeline.md` | 2026-11-11 | Delivered in commit `8d6ec66`; pipeline later retired by [SPEC-016](SPEC-016-runtime-module-deepening.md) |
| `docs/superpowers/plans/2026-05-17-production-decoupled-diarization-pipeline.md` | 2026-11-11 | Delivered in commit `72f9ff4`; pipeline later retired by [SPEC-016](SPEC-016-runtime-module-deepening.md) |
| `docs/superpowers/plans/2026-05-18-spec-013-long-form-productionization.md` | 2026-11-11 | Retired by [ADR-002](../ADR-002-Lightweight-Local-Speech-Gateway-Boundary.md) & [SPEC-016](SPEC-016-runtime-module-deepening.md) |
| `docs/superpowers/plans/2026-05-21-qwen3-sortformer-doc-repositioning.md` | 2026-11-11 | Delivered in [ADR-002](../ADR-002-Lightweight-Local-Speech-Gateway-Boundary.md) & [SPEC-013](SPEC-013-Production-Decoupled-Diarization-Pipeline.md) |
| `docs/superpowers/plans/2026-07-04-apple-speech-service-integration.md` | 2026-11-11 | Delivered in [SPEC-014](SPEC-014-Apple-SpeechAnalyzer-Integration.md), [`src/core/apple_speech_engine.py`](../../src/core/apple_speech_engine.py) |
| `docs/superpowers/plans/2026-07-05-apple-speech-phase3-evaluation.md` | 2026-11-11 | Delivered in [`benchmarks/phase3_evaluation.py`](../../benchmarks/phase3_evaluation.py) |
| `docs/superpowers/specs/2026-05-21-qwen3-sortformer-experimental-repositioning-design.md` | 2026-11-11 | [ADR-002](../ADR-002-Lightweight-Local-Speech-Gateway-Boundary.md), [SPEC-016](SPEC-016-runtime-module-deepening.md) |
| `docs/SPEC-001-Interface-And-Schemas.md` | 2026-11-11 | [`src/api/routes.py`](../../src/api/routes.py), [`README.md`](../../README.md) |
| `docs/SPEC-002-Service-And-Engine.md` | 2026-11-11 | [`src/services/transcription.py`](../../src/services/transcription.py), [`src/core/base_engine.py`](../../src/core/base_engine.py) |
| `docs/SPEC-003-Audio-And-Text-Processing.md` | 2026-11-11 | [`src/adapters/text.py`](../../src/adapters/text.py), [`src/services/transcription.py`](../../src/services/transcription.py) |
| `docs/SPEC-004-Testing-Strategy.md` | 2026-11-11 | [`tests/`](../../tests/), [`AGENTS.md`](../../AGENTS.md) |
| `docs/SPEC-005-Speaker-Diarization.md` | 2026-11-11 | [`src/core/funasr_engine.py`](../../src/core/funasr_engine.py), [`MODELS.md`](../../MODELS.md) |
| `docs/SPEC-006-Security-And-Quality-Hardening.md` | 2026-11-11 | [`tests/unit/test_security.py`](../../tests/unit/test_security.py), [`src/api/routes.py`](../../src/api/routes.py) |
| `docs/SPEC-007-Speaker-Diarization-FunASR.md` | 2026-11-11 | [`src/core/funasr_engine.py`](../../src/core/funasr_engine.py) |
| `docs/SPEC-008-Dynamic-Model-Switching.md` | 2026-11-11 | [`src/core/model_registry.py`](../../src/core/model_registry.py), [`src/services/transcription.py`](../../src/services/transcription.py), [`AGENTS.md`](../../AGENTS.md) |
| `docs/SPEC-009-Idle-Model-Offload.md` | 2026-11-11 | [`src/workers/model_worker.py`](../../src/workers/model_worker.py), [`src/services/worker_session.py`](../../src/services/worker_session.py), [`AGENTS.md`](../../AGENTS.md) |
| `docs/SPEC-011-Decoupled-ASR-Diarization.md` | 2026-11-11 | Retired; superseded by [ADR-002](../ADR-002-Lightweight-Local-Speech-Gateway-Boundary.md) & [SPEC-016](SPEC-016-runtime-module-deepening.md) |
| `docs/SPEC-012-Apple-Native-Diarization-Capability-Probe.md` | 2026-11-11 | Completed probe; findings in [ADR-002](../ADR-002-Lightweight-Local-Speech-Gateway-Boundary.md) |
| `docs/SPEC-013-Production-Decoupled-Diarization-Pipeline.md` | 2026-11-11 | Retired from production; superseded by [ADR-002](../ADR-002-Lightweight-Local-Speech-Gateway-Boundary.md) & [SPEC-016](SPEC-016-runtime-module-deepening.md) |
| `docs/SPEC-014-Apple-SpeechAnalyzer-Integration.md` | 2026-11-11 | [`src/core/apple_speech_engine.py`](../../src/core/apple_speech_engine.py), [`MODELS.md`](../../MODELS.md), [`AGENTS.md`](../../AGENTS.md) |
| `docs/SPEC-015-Architecture-Fitness-Harness.md` | 2026-11-11 | [`tests/unit/test_architecture_fitness.py`](../../tests/unit/test_architecture_fitness.py), [`tach.toml`](../../tach.toml) |
| `docs/SPEC-016-runtime-module-deepening.md` | 2026-11-11 | Delivered in PR #38 / #39; contracts documented in [`AGENTS.md`](../../AGENTS.md) |
