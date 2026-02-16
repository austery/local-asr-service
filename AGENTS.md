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
  - `base_engine.py`: Engine Protocol (interface)
  - `funasr_engine.py`: FunASR/Paraformer implementation (supports speaker diarization)
  - `mlx_engine.py`: MLX Audio implementation
  - `factory.py`: Engine factory (creates engine based on config)
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
