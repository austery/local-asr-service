---
title: Subprocess Model Isolation (SPEC-009 v2)
date: 2026-03-08
status: approved
author: Lei
relatedSpecs:
  - SPEC-009
  - SPEC-002
  - SPEC-108
---

# Design: Subprocess Model Isolation for Memory Relief

## Problem Statement

SPEC-009 implemented idle model offload via `release()` + `torch.mps.empty_cache()` + `gc.collect()`.
After implementation, Activity Monitor shows memory drops from ~23 GB to only ~15–18 GB when idle —
far above the target of < 1 GB.

**Root cause**: PyTorch's MPS (Metal) allocator maintains its own memory pool. Even after deleting
all model references and calling `empty_cache()`, macOS does not reclaim the MPS heap from the
process. The only guaranteed way to release MPS memory is to terminate the process.

## Chosen Approach: Subprocess Model Isolation (Ollama-style)

The ASR model is moved into a **child subprocess**. The main FastAPI process acts as a lightweight
proxy (< 200 MB) that manages the worker's lifecycle. When the worker exits (idle timeout or error),
the OS reclaims 100% of MPS memory automatically.

### Why not alternatives

| Alternative | Reason rejected |
|-------------|-----------------|
| More aggressive in-process gc | MPS pool is fundamentally OS-held; cannot force-release from within the process |
| pm2 / supervisord | Kills the entire FastAPI process; HTTP port goes down during 30–60s model reload |
| puresubs manages the process | Tight coupling; puresubs should not own ASR lifecycle |

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  local-asr-service main process (FastAPI, always running) │
│  Memory: < 200 MB                                         │
│                                                           │
│  HTTP :50070 ──▶ TranscriptionService (refactored)        │
│                          │                                │
│                          │  multiprocessing IPC           │
│                          ▼                                │
│          ┌──────────────────────────────┐                 │
│          │  ModelWorker subprocess      │                 │
│          │  · loads FunASR/MLX models   │                 │
│          │  · runs inference loop       │                 │
│          │  · idle timeout → sys.exit() │                 │
│          └──────────────────────────────┘                 │
│             ↑ process death = OS reclaims ALL MPS memory  │
└──────────────────────────────────────────────────────────┘
```

## Data Flow

```
[HTTP request arrives]
    │
    ├─ Worker running? ──No──▶ spawn ModelWorker, wait for READY signal
    │
    ▼
job_queue.put(TranscriptionJob)
    │
    ▼
[ModelWorker subprocess]
    job = job_queue.get(timeout=idle_timeout_sec)
    result = engine.transcribe(job.audio_path)   ← reads tmp file from disk
    result_queue.put((job.uid, result))
    │
    ├─ TimeoutError ──▶ sys.exit(0)  [OS reclaims all memory]
    │
    ▼
[Main process receives result] ──▶ HTTP response
```

**IPC mechanism**: `multiprocessing.Queue` (native Python, no extra dependencies).  
**File passing**: temp file path only (no audio binary serialization in queue).

## Error Handling

| Scenario | Handling |
|----------|----------|
| Worker startup failure (model load crash) | Main process catches `ProcessError`, returns HTTP 503 |
| Worker crashes mid-inference | `result_queue` timeout → main process `terminate()` + cleanup → HTTP 500 |
| Concurrent requests during worker spawn | `asyncio.Lock` in main process; requests queue while worker starts |
| `MODEL_IDLE_TIMEOUT_SEC=0` | Worker's queue.get() has no timeout; worker stays alive indefinitely |
| Dynamic model switch (SPEC-108) | Kill old worker subprocess, spawn new one with new model spec — naturally prevents double-load |

## Files Changed

| File | Change |
|------|--------|
| `src/services/transcription.py` | Core refactor: from "holds engine" to "manages worker subprocess" |
| `src/workers/model_worker.py` | **New**: subprocess entry point — engine load + inference loop |
| `tests/unit/test_idle_offload.py` | Update: mock subprocess instead of mock engine |
| `docs/specs/SPEC-009-Idle-Model-Offload.md` | Status → v2, record design change rationale |

`puresubs` and all other callers require **zero changes**.

## Acceptance Criteria

- AC-1: After `MODEL_IDLE_TIMEOUT_SEC` seconds of inactivity, `python3.11` memory in Activity Monitor drops **below 500 MB**
- AC-2: Next request after idle triggers worker spawn; transcription succeeds (with 10–60s reload delay)
- AC-3: `MODEL_IDLE_TIMEOUT_SEC=0` keeps worker alive indefinitely (no idle exit)
- AC-4: Dynamic model switch (SPEC-108) continues to work correctly
- AC-5: All existing tests pass (behavior unchanged from caller's perspective)
- AC-6: puresubs requires zero code changes
