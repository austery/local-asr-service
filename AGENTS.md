# Local ASR Service (Mac Silicon Optimized)

## Project Overview
A high-performance, local voice transcription service optimized for Apple Silicon (M-series) chips.
Supports **dual engine architecture**:
- **FunASR Engine**: Alibaba FunASR (Paraformer) model via PyTorch MPS, supports Speaker Diarization.
- **MLX Audio Engine**: Apple MLX native models (Qwen3-ASR, Whisper, etc.)

Provides an OpenAI Whisper-compatible HTTP API.
The project follows Clean Architecture principles to separate API, scheduling, and inference logic.

## Setup
This project uses `uv` for dependency management and requires Python 3.11.

```bash
# Install dependencies (with prerelease for mlx-audio)
uv sync --prerelease=allow

# Activate virtual environment
source .venv/bin/activate
```

## Build and Run
**Important:** strict single-process execution is required to prevent memory exhaustion on M-series chips. Do not increase worker count.

```bash
# Start with FunASR engine (default)
uv run python -m src.main

# Start with MLX Audio engine (Qwen3-ASR, Whisper, etc.)
ENGINE_TYPE=mlx uv run python -m src.main

# Use a custom model
ENGINE_TYPE=mlx MODEL_ID=mlx-community/whisper-large-v3-turbo uv run python -m src.main

# Alternative using uvicorn directly
uvicorn src.main:app --host 0.0.0.0 --port 50070 --workers 1
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGINE_TYPE` | `funasr` | Engine type: `funasr` or `mlx` |
| `MODEL_ID` | (engine default) | Override model ID for any engine |
| `FUNASR_MODEL_ID` | `iic/speech_seaco_paraformer...` | Default model for FunASR (supports diarization) |
| `MLX_MODEL_ID` | `mlx-community/Qwen3-ASR-1.7B-4bit` | Default model for MLX engine |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `50070` | Server port |
| `MAX_QUEUE_SIZE` | `50` | Max concurrent requests in queue |
| `LOG_LEVEL` | `INFO` | Logging level |

## Supported MLX Models

When using `ENGINE_TYPE=mlx`, you can switch models via `MODEL_ID`:
- `mlx-community/Qwen3-ASR-1.7B-4bit` - Alibaba Qwen3-ASR (default, recommended, fast & stable)
- `mlx-community/whisper-large-v3-turbo-asr-fp16` - OpenAI Whisper Turbo
- `mlx-community/Qwen3-ASR-1.7B-8bit` - Alibaba Qwen3-ASR
- `mlx-community/parakeet-tdt-0.6b-v2` - NVIDIA Parakeet (English only)

## Testing
The project uses `pytest` for all levels of testing.

```bash
# Run all tests
uv run python -m pytest

# Run specific test categories
uv run python -m pytest tests/unit          # Unit tests (Mocked)
uv run python -m pytest tests/integration   # API integration tests
uv run python -m pytest tests/e2e           # End-to-end tests (Real model)
uv run python -m pytest tests/reliability   # Concurrency tests
```

## Architecture
The codebase is organized into layers:
- **`src/api`**: HTTP routes and Pydantic models (contract definition).
- **`src/services`**: Async task queue management and scheduling.
- **`src/core`**: ASR engine abstraction and implementations:
  - `base_engine.py`: Engine Protocol (interface) + EngineCapabilities
  - `funasr_engine.py`: FunASR/Paraformer implementation (supports speaker diarization)
  - `mlx_engine.py`: MLX Audio implementation
  - `model_registry.py`: Model alias registry (alias → ModelSpec mapping, SPEC-108)
  - `factory.py`: Engine factory (creates engine from config or ModelSpec)
- **`src/adapters`**: Pure functions for text cleaning and audio processing.
- **`src/config.py`**: Centralized environment variable configuration.

## Code Style & Conventions
- **Python Version**: 3.11+
- **Type Hints**: Extensive use of type hints is expected.
- **Architecture**: Adhere to Clean Architecture; strictly separate concerns.
- **Concurrency**: Use `asyncio` for I/O bound tasks, but rely on the serial worker queue for model inference.
- **Testing**: maintain high test coverage, especially for the core logic.

## Key Constraints
- **Single Worker**: Always run with `workers=1` to avoid OOM on Mac Silicon.
- **Queue Limit**: The service implements a max queue depth of 50 to prevent overload.
- **No raw pip**: This project uses `uv`. Always use `uv add` / `uv sync`, never `pip install`.
- **No `any` type**: All code must be fully typed. `any` is forbidden; use proper generics or `Unknown`.

## Known Issues & Workarounds

### FunASR distribute_spk NoneType Bug (Fixed 2026-02-21)
FunASR's `campplus/utils.py:distribute_spk` crashes with `TypeError: '>' not supported between instances of 'float' and 'NoneType'` when `sv_output` contains entries with `None` timestamps (happens on short or ambiguous audio segments).

**Fix**: A module-level monkey-patch is applied in `src/core/funasr_engine.py` at import time. It replaces `funasr.models.campplus.utils.distribute_spk` with a version that filters out `None` timing entries before processing.

**Do not remove this patch** — the underlying FunASR library bug has not been fixed upstream.

## Architecture Decisions
- **Engine capabilities** are declared at startup via `EngineCapabilities` frozen dataclass (`src/core/base_engine.py`). API layer validates compatibility before queuing — do not bypass this.
- **Monkey-patching third-party libraries** is acceptable in `funasr_engine.py` only, at module level, with a clear comment. Do not patch elsewhere.
- **Temporary files** for uploads are written to disk (not held in memory) — see `src/services/transcription.py`. Always cleaned in `finally` blocks.
- **Dynamic model switching** (SPEC-108): Per-request `model` field triggers hot-swap inside `_consume_loop`. `release()` always precedes `load()` (memory safety). Passthrough values (`None`, `""`, `"whisper-1"`) skip switching. See `src/core/model_registry.py` for the alias table.
- **Model registry** (`src/core/model_registry.py`) is the single source of truth for supported model aliases. Add new models there first before referencing them anywhere else.

## Testing Notes
- Run `uv run python -m pytest` for all tests. E2E tests (`tests/e2e/`) require the real model to be downloaded and are slow.
- Unit tests mock the engine entirely — do not add real model calls to unit tests.
- Test count baseline: 112 tests (as of 2026-02-25). Do not reduce this.
