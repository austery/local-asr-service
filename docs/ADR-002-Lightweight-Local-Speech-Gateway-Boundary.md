---
specId: ADR-002
title: Lightweight Local Speech Gateway Boundary
status: ✅ 已完成 (Completed)
creationDate: 2026-05-19
lastUpdateDate: 2026-05-19
deciders:
  - User (AI-Assisted)
relatedSpecs:
  - ADR-001
  - SPEC-008
  - SPEC-009
  - SPEC-011
  - SPEC-014
  - SPEC-015
tags:
  - project-boundary
  - local-asr
  - model-runtime
  - apple-silicon
  - dependency-strategy
---

# ADR-002: Lightweight Local Speech Gateway Boundary

**Date**: 2026-05-19  
**Status**: ✅ Accepted  
**Deciders**: User (AI-Assisted)

## Context

This project started as a local Apple Silicon ASR service with two practical jobs:

1. Provide a local transcription endpoint for personal data pipelines.
2. Provide a local voice-input fallback when remote APIs are unavailable, too slow, or unsuitable.

Recent work added dynamic model switching, idle subprocess offload, MLX Qwen3-ASR support, and the `qwen3-sortformer` pipeline profile. A 57-minute English multi-speaker probe initially made the Qwen3 + Sortformer direction look promising for long-form data-pipeline use: runtime was materially slower than Paraformer, but top-level English transcript quality was better for that sample. A later real English manager 1:1 probe changed the product judgment: the pipeline kept the stronger Qwen3 transcript text, but its speaker-labeled segments were unreliable and its memory footprint was too high for the value delivered by this local gateway.

The project is at risk of becoming a general speech model platform: adding more model adapters, custom diarization logic, embedding fallbacks, chunk reconciliation, Swift generation, and runtime-specific optimizations. That direction is technically interesting, but it expands maintenance cost beyond the project's original personal utility.

## Decision

**We choose to define `local-asr-service` as a lightweight OpenAI-compatible local speech gateway, not a general speech model framework.**

The project should depend on upstream speech runtimes wherever possible and own only the thin layer needed to make them useful for local services:

- HTTP API compatibility with OpenAI-style transcription endpoints.
- Queueing, single-worker execution, resource isolation, and idle subprocess offload.
- Model alias registry and capability gating.
- Minimal runtime adapters around stable upstream APIs.
- Response normalization so downstream services receive predictable JSON.
- Probe and benchmark scripts for deciding whether a model is worth enabling.

The project should not become the place where we deeply reimplement or own:

- model architecture ports;
- inference kernels;
- model conversion pipelines;
- complex diarization postprocessing;
- speaker embedding reconciliation;
- desktop or mobile app UX;
- large multi-model orchestration logic;
- Swift-native runtime code that belongs in a Swift package.

## Relationship To Adjacent Projects

| Project | Role | Relationship |
|---------|------|--------------|
| `mlx-audio` | Python/MLX upstream runtime and model catalog for Apple Silicon speech models | Prefer direct dependency or thin adapter. Do not fork or reimplement model internals unless a tiny compatibility patch is needed. |
| `speech-swift` | Swift-native Apple Silicon speech stack for macOS/iOS apps, CoreML/MLX integration, streaming dictation, diarization, and UI-adjacent use cases | Prefer it for native app and voice-input integration. If Swift-native local speech becomes the main product path, build on `speech-swift` rather than generating Swift from this Python gateway. |
| FunASR | Upstream Python runtime for Paraformer, CAM++, SenseVoice, and related models | Keep as a dependency-backed adapter. Own only monkey patches that protect this service from known upstream runtime failures. |
| `local-asr-service` | Local HTTP gateway for personal automation and fallback transcription | Own service boundaries, resource safety, API contract, model gating, and operational probes. |

## Product Boundary

### In Scope

- Maintain a reliable local `/v1/audio/transcriptions` compatible endpoint.
- Keep the service safe on M-series machines through strict single-worker execution.
- Support a small set of proven aliases: production-ready defaults plus a few
  explicitly gated experiments or discovery-only profiles.
- Use subprocess offload to make heavyweight models tolerable for intermittent personal use.
- Benchmark real files from the target workflows before promoting any model.
- Preserve response contracts across upstream model differences.

### Out Of Scope

- Becoming a model zoo.
- Adding every new upstream model immediately.
- Building a custom diarization research pipeline when upstream projects already expose one.
- Maintaining a separate Swift runtime in this repo.
- Optimizing for every possible ASR/TTS/STS capability.
- Treating experimental models as production just because a short probe succeeds.

## Model Enablement Rule

A model or pipeline can become requestable only when all conditions are true:

1. The upstream runtime API is stable enough to call through a thin adapter.
2. The model serves one of the two real use cases: data pipeline or voice-input fallback.
3. Real-media probes pass for the intended duration, language, and speaker pattern.
4. Runtime cost is acceptable for that use case, not merely impressive in isolation.
5. The adapter does not require this repo to own complex model-specific recovery logic.
6. Failure modes are bounded: timeout, memory release, response metadata, and cleanup are verified.

Discovery-only aliases are allowed, but they must remain non-requestable until the above gate passes.

## Scope Audit Findings

The 2026-05-19 boundary review found that the project has started to approach the edge of the lightweight-gateway boundary, mainly through the `qwen3-sortformer` experiment:

| Area | Finding | Decision |
|------|---------|----------|
| `TranscriptionService._run_decoupled_pipeline()` | The service now orchestrates ASR, chunking, forced alignment, diarization, speaker assignment, and resident-model restore. This is more than a simple HTTP gateway. | Keep only as a narrow opt-in batch path; do not generalize into a multi-model orchestration framework. |
| `pipeline_chunking.reconcile_chunk_speaker_labels()` | The repo contains a small speaker-label reconciliation heuristic. This is the first step toward owning diarization postprocessing. | Keep the current minimal overlap-based logic, but do not add embedding fallback or complex speaker recovery unless upstream exposes it. |
| Alignment quality gates | Local heuristics reject structurally invalid alignment output. | Keep as service safety checks, not as a growing quality-scoring system. |
| Worker job kinds | The worker now hosts ASR, alignment, and diarization runtime objects. | Accept while it protects Apple Silicon memory boundaries; avoid adding VAD, embeddings, clustering, or TTS into this worker without a new ADR. |
| Sortformer tuning constants | Local thresholds exist for the upstream runtime call. | Keep fixed constants; do not turn this repo into a Sortformer tuning harness. |

The key conclusion: the current implementation may remain as an explicit
experiment because it is gated by model selection and uses upstream runtimes,
but further custom recovery logic would cross the project boundary.

## Experimental Reachability

`qwen3-sortformer` may remain requestable as an explicit opt-in experiment:

- Callers must pass `model=qwen3-sortformer`.
- It must not become the default model.
- It is not the recommended English multi-speaker meeting path.
- Its current three-stage service pipeline is a deletion candidate if stronger
  upstream or open-source local diarized-ASR capability replaces it.
- The service should fail closed on contract errors and quality-gate failures
  rather than silently inventing more recovery behavior.

This reachability is a way to preserve the evaluation result, not a product
commitment to build a general diarization platform.

## Upstream Watch Policy

The project should periodically check upstream model/runtime progress, especially:

- `mlx-audio` STT, Qwen3-ASR, Qwen3-ForcedAligner, and Sortformer runtime changes;
- `speech-swift` native dictation, streaming ASR, CoreML/ANE, and diarization support;
- FunASR / Paraformer / SenseVoice updates for speed, language coverage, and diarization stability.

The watch process should answer one question: "Can upstream now do something that lets this repo delete code or keep a thinner adapter?" New upstream capabilities should first enter as probe scripts or dependency upgrades, not as immediate production aliases.

## Fitness Functions

To keep the boundary from drifting again, we enforce architectural constraints automatically under [SPEC-015](file:///Users/leipeng/Documents/Projects/local-asr-service/docs/SPEC-015-Architecture-Fitness-Harness.md):

- **Module Dependencies**: Enforced via **Tach** (`tach.toml`). Dependencies flow strictly inward from API to Core/Adapters.
- **Code Complexity**: Enforced via **Ruff** McCabe (`C901`) and Pylint complexity checks (`PLR0915`, `PLR0912`), limiting method length and cyclomatic complexity.
- **Semantic Project Rules**: Verified via **pytest** assertions in `tests/unit/test_architecture_fitness.py`:
  - Pipeline profiles must declare `requestable` explicitly and must have documentation in `MODELS.md`.
  - No new model runtime adapter or diarization reconciliation module may be added without updating the allowed list.
  - `TranscriptionService` job domains are restricted to `transcribe`, `align`, and `diarize`.
  - The default model in `config.py` must not point to an experimental pipeline.

## Options Considered

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| Build a full local speech platform in this repo | Maximum control; can integrate ASR, diarization, alignment, TTS, and app logic in one place | High maintenance cost; duplicates upstream work; expands a personal utility into a research platform | ❌ Rejected |
| Use only upstream tools directly and delete this service | Lowest maintenance; no adapter layer | Downstream personal services lose a stable OpenAI-compatible endpoint, queueing, idle offload, and response normalization | ❌ Rejected |
| Keep this repo as a lightweight local speech gateway | Preserves personal pipeline utility while limiting scope; lets upstream projects own model runtime complexity | Requires discipline: new models must be gated and rejected when they do not fit | ✅ Chosen |
| Move future voice-input work into Swift-native dependencies | Better fit for macOS/iOS latency, microphone, streaming, and UI integration | Separate codebase and dependency strategy; not a replacement for HTTP batch pipelines | ✅ Chosen for native voice-input path |
| Keep `qwen3-sortformer` requestable as an explicit experiment | Preserves the evaluation path and the stronger Qwen3 transcript text for follow-up probes | Speaker-segment quality and resource cost do not justify recommending it now | ✅ Chosen as a deletion candidate, not a production recommendation |

## Consequences

**Positive**

- The repo stays small enough to maintain as a personal tool.
- New model work becomes evaluation-first, not integration-first.
- `mlx-audio`, `speech-swift`, and FunASR can move quickly without forcing this repo to chase every implementation detail.
- The data pipeline keeps a stable local API even when model choices change underneath.
- Voice-input fallback can eventually move toward a native Swift stack without dragging this Python service into app development.

**Tradeoffs**

- Some promising models will remain discovery-only or be left to upstream CLIs/packages.
- The service may intentionally lag behind the latest model announcements.
- Highly customized diarization cleanup may be deferred even when it could improve one benchmark.
- The project will prefer boring dependency integration over owning clever model-specific code.

## Practical Guidelines

- Prefer adding a probe script before adding a requestable model alias.
- Prefer upstream feature branches or small upstream PRs over local forks when a model runtime needs improvement.
- Prefer a new dependency-backed adapter only when the runtime contract is clearly different from existing engines.
- Prefer deleting or deferring model-specific workaround code when upstream support becomes stable.
- Keep production defaults boring; keep experiments explicitly labeled and
  gated, and prefer discovery-only until reachable evaluation is useful.
- For Swift-native voice workflows, evaluate `speech-swift` first before adding Swift code generation or native app logic here.

## Review History

| Date | Reviewer | Note | Status |
|------|----------|------|--------|
| 2026-05-19 | User (AI-Assisted) | Initial boundary draft after qwen3-sortformer long-form probe and Paraformer comparison | 📝 Draft |
| 2026-05-19 | User (AI-Assisted) | Added scope audit findings, opt-in production reachability for `qwen3-sortformer`, upstream watch policy, and fitness-function follow-up | ✅ Accepted |
| 2026-05-21 | User (AI-Assisted) | Real English manager 1:1 probe downgraded `qwen3-sortformer` from production-adjacent batch path to experimental reachable deletion candidate | ✅ Accepted |
