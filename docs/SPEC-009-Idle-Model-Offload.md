---
specId: SPEC-009
title: Idle Model Offload (Memory Pressure Relief)
status: 📝 草案 (Draft)
priority: P1 - Core Feature
creationDate: 2026-03-06
lastUpdateDate: 2026-03-06
owner: User (AI-Assisted)
relatedSpecs:
  - SPEC-002
  - SPEC-108
tags:
  - memory-management
  - idle-offload
  - model-lifecycle
  - resource-optimization
---

# SPEC-009: Idle Model Offload (Memory Pressure Relief)

## 1. Goal

> Automatically release ASR model memory when the service is idle, so a long-running service does not permanently consume 16–23 GB of RAM while waiting for infrequent requests.

## 2. Background

### The Problem

The service loads its ASR model pipeline at startup and keeps it resident in memory for the entire process lifetime. There is **no mechanism to release the model when idle**.

Activity Monitor observation (2026-03-06):

| Process | Memory | Notes |
|---------|--------|-------|
| python3.11 (this service) | **23.38 GB** | FunASR Paraformer full pipeline |
| ollama | 7.31 GB | Separate process |
| **Total ASR + LLM** | **~30 GB** | On a 64 GB machine |

### Why 23 GB?

The FunASR Paraformer pipeline loads **four sub-models** into memory simultaneously:

| Component | Est. Memory | Purpose |
|-----------|-------------|---------|
| SEACO-Paraformer Large (ASR) | ~8–10 GB | Core speech recognition |
| fsmn-vad (VAD) | ~1 GB | Voice activity detection for long audio |
| ct-punc (Punctuation) | ~1 GB | Punctuation restoration |
| cam++ (Speaker) | ~2–3 GB | Speaker diarization (SPEC-007) |
| PyTorch MPS runtime | ~3–5 GB | Framework overhead |
| Python + deps | ~1–2 GB | Process baseline |
| **Total** | **~16–23 GB** | Matches Activity Monitor |

### Current Lifecycle (Before This SPEC)

```
Startup: create engine → engine.load() → model IN MEMORY
Running: queue.get() → transcribe → queue.get() → transcribe → ...
                                                    ↑ model stays in memory FOREVER
Shutdown: engine.release() → model freed
```

The `release()` method exists and works correctly (per SPEC-002), but is only called:
1. On service shutdown (`main.py` lifespan)
2. During dynamic model switch (`_switch_model`, per SPEC-108)

**Missing**: Automatic release after an idle period.

### Design Constraint from SPEC-108

SPEC-108 §5 established the `release()` → `load()` ordering contract: always release old model before loading new one to avoid double memory peaks on M-series. This SPEC extends that principle: if no one is using the model, release it proactively.

## 3. Design Decision

**Chosen approach**: Idle timeout in `_consume_loop` — use `asyncio.wait_for()` on queue with configurable timeout; on timeout, release model; on next request, reload.

**Rationale**: Minimal code change (localized to `_consume_loop`), reuses existing `release()`/`load()` methods, no new threads or schedulers needed.

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| `asyncio.wait_for()` timeout in consume loop | Simple, no new deps, reuses existing lifecycle | First request after offload has 10–30s reload delay | ✅ Chosen |
| Background timer task (`asyncio.create_task`) | Decoupled from consume loop | Race conditions with `_consume_loop`; needs locking; more moving parts | ❌ Rejected |
| OS-level memory pressure listener (`resource` module) | Reacts to actual pressure | macOS-specific, complex, unreliable MPS memory reporting | ❌ Rejected |
| Lazy loading only (no preload at startup) | Zero startup memory | First request always slow; poor DX | ❌ Rejected |

### Trade-off: Reload Latency

The first request after an idle offload will experience model reload delay:

| Engine | Reload Time (warm cache) | Reload Time (cold) |
|--------|--------------------------|---------------------|
| FunASR Paraformer | ~10–15s | ~30–60s (first download) |
| MLX Qwen3-ASR | ~3–5s | ~10–20s |

This is acceptable because:
1. The idle timeout (default 60s) means the model was genuinely unused
2. Reload uses warm cache (model weights on disk), so it's the faster case
3. The service is designed for batch/async use (PureSubs), not real-time voice input

## 4. Implementation Phases

### Phase 1: Core Idle Offload — Target: 2026-03-06

- [ ] Add `MODEL_IDLE_TIMEOUT_SEC` to `src/config.py` (default: `60`, `0` = disabled)
- [ ] Add env var documentation to `.env` and `.env.example`
- [ ] Modify `TranscriptionService._consume_loop()`:
  - Replace `await self.queue.get()` with `asyncio.wait_for(timeout=IDLE_TIMEOUT)`
  - On `asyncio.TimeoutError`: call `engine.release()`, set `_model_loaded = False`
  - Before inference: if `not _model_loaded`, call `engine.load()`, set `_model_loaded = True`
- [ ] Track `_model_loaded: bool` state in `TranscriptionService`
- [ ] Log idle offload / reload events at INFO level
- [ ] Log configured timeout at startup in `main.py`
- [ ] Update `AGENTS.md` with idle offload documentation

**Acceptance**: After 60s of inactivity, `python3.11` memory in Activity Monitor drops below 5 GB. Next request succeeds (with reload delay) and memory rises back.

### Phase 2: Health & Observability — Target: TBD

- [ ] Add `model_loaded: bool` to `GET /v1/models/current` response
- [ ] Add `last_inference_at: float | null` timestamp
- [ ] Add `idle_timeout_sec: int` to health endpoint

**Acceptance**: Clients can query whether model is currently loaded and plan accordingly.

## 5. Acceptance Criteria

- [ ] AC-1: Model memory is released after `MODEL_IDLE_TIMEOUT_SEC` seconds of no requests
- [ ] AC-2: Next request after offload succeeds (model auto-reloads)
- [ ] AC-3: Setting `MODEL_IDLE_TIMEOUT_SEC=0` disables idle offload (model stays resident)
- [ ] AC-4: Idle timer resets after each successful transcription
- [ ] AC-5: Dynamic model switch (SPEC-108) and idle offload do not conflict (no double-release)
- [ ] AC-6: All 112 existing tests pass without modification

## 6. Affected Files

| File | Change Type | Description |
|------|-------------|-------------|
| `src/config.py` | Modify | Add `MODEL_IDLE_TIMEOUT_SEC` |
| `src/services/transcription.py` | Modify | Idle timeout logic in `_consume_loop`, `_model_loaded` state |
| `src/main.py` | Modify | Log idle timeout config at startup |
| `.env` | Modify | Add env var documentation |
| `.env.example` | Modify | Add env var documentation |
| `AGENTS.md` | Modify | Document idle offload behavior |
| `tests/unit/test_idle_offload.py` | **New** | Unit tests IO-1 through IO-5 |

## 7. Test Strategy

### Unit Tests (`tests/unit/test_idle_offload.py`)

| ID | Test Name | Observable Behavior |
|----|-----------|---------------------|
| IO-1 | `test_should_release_model_after_idle_timeout` | After timeout expires with no jobs, `engine.release()` is called |
| IO-2 | `test_should_reload_model_on_next_request_after_offload` | Job submitted after offload triggers `engine.load()` and succeeds |
| IO-3 | `test_should_not_offload_when_timeout_is_zero` | With `timeout=0`, model stays loaded indefinitely |
| IO-4 | `test_should_reset_idle_timer_after_transcription` | Timer resets; no offload if requests keep arriving within window |
| IO-5 | `test_should_not_double_release_during_model_switch` | Offload + concurrent switch request handles gracefully |

### Manual Verification

1. Start service with `MODEL_IDLE_TIMEOUT_SEC=60`
2. Send one transcription request → observe ~20 GB memory
3. Wait 70 seconds → observe memory drop in Activity Monitor
4. Send another request → observe reload + successful transcription
5. Start service with `MODEL_IDLE_TIMEOUT_SEC=0` → verify model stays loaded

## 8. Interaction with SPEC-108 (Dynamic Model Switching)

The `_model_loaded` flag works orthogonally with `_switch_model()`:

```
Case 1: Idle offload → same model request
  _model_loaded=False → engine.load() → transcribe → reset timer

Case 2: Idle offload → different model request
  _model_loaded=False → _switch_model() handles its own release/load → transcribe

Case 3: Model loaded → different model request
  _model_loaded=True → _switch_model() releases old, loads new → transcribe

Case 4: Model loaded → same model request (normal path)
  _model_loaded=True → transcribe directly → reset timer
```

**Important**: `_switch_model()` already calls `release()` on the old engine. If the model was already offloaded (`_model_loaded=False`), `release()` on an already-released engine is a no-op (guarded by `if self.model:` in both engines). No special handling needed.

## 9. Status History

| Date | Status | Note |
|------|--------|------|
| 2026-03-06 | 📝 草案 (Draft) | Initial draft based on memory audit findings |

## 10. Related

- **Code**: `src/services/transcription.py` (primary change site)
- **Code**: `src/core/funasr_engine.py` (`release()` implementation)
- **Code**: `src/core/mlx_engine.py` (`release()` implementation)
- **Specs**: [SPEC-002](./SPEC-002-Service-And-Engine.md) (engine lifecycle)
- **Specs**: [SPEC-108](./SPEC-008-Dynamic-Model-Switching.md) (dynamic model switching, release-before-load contract)
