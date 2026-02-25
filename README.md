# Local ASR Service (Mac Silicon)

High-performance local speech transcription service optimized for Apple Silicon (M-series).
OpenAI Whisper-compatible HTTP API on port **50070**.

**Dual-engine architecture:**
- **FunASR** — Paraformer (Chinese SOTA) + CAM++ speaker diarization
- **MLX Audio** — Apple-native models (Qwen3-ASR, Whisper, etc.)

→ See [MODELS.md](./MODELS.md) for model list, benchmark results, and selection guide.

---

## Quick Start

```bash
# Install dependencies
uv sync --prerelease=allow

# Start with FunASR engine (default — Paraformer, supports diarization)
uv run python -m src.main

# Start with MLX engine
ENGINE_TYPE=mlx uv run python -m src.main
```

First launch downloads the model automatically (~1-2GB, may take a few minutes).

---

## Use Cases

| Scenario | Recommended | Command |
|----------|-------------|---------|
| Multi-speaker podcast / meeting | `paraformer` (default) | `uv run python -m src.main` |
| Fast single-speaker transcription | `qwen3-asr` | `ENGINE_TYPE=mlx uv run python -m src.main` |
| Bulk speed-first (no diarization) | `sensevoice-small` | `FUNASR_MODEL_ID=iic/SenseVoiceSmall uv run python -m src.main` |

---

## API

### Health check

```bash
curl http://localhost:50070/health
```

### Transcription

```bash
# Default: JSON with speaker diarization
curl http://localhost:50070/v1/audio/transcriptions \
  -F "file=@audio.mp3;type=audio/mpeg"

# Plain text (for RAG / LLM input)
curl http://localhost:50070/v1/audio/transcriptions \
  -F "file=@audio.mp3;type=audio/mpeg" \
  -F "output_format=txt"

# SRT subtitles
curl http://localhost:50070/v1/audio/transcriptions \
  -F "file=@audio.mp3;type=audio/mpeg" \
  -F "output_format=srt"

# Per-request model switch (hot-swap, no restart needed)
curl http://localhost:50070/v1/audio/transcriptions \
  -F "file=@audio.mp3;type=audio/mpeg" \
  -F "model=qwen3-asr"
```

**Request parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `file` | required | Audio file (wav, mp3, m4a, flac, ogg, webm) |
| `output_format` | `json` | Output: `json`, `txt`, `srt` |
| `response_format` | — | OpenAI alias: `verbose_json`, `text`, `vtt` |
| `with_timestamp` | `false` | Prepend `[MM:SS]` to each line in txt mode |
| `language` | `auto` | `zh`, `en`, `auto` |
| `model` | — | Alias or full model path. Omit to keep current model. |

### Query models

```bash
curl http://localhost:50070/v1/models | jq          # all registered models
curl http://localhost:50070/v1/models/current | jq  # currently loaded model + capabilities
```

### Interactive docs

```
http://localhost:50070/docs
```

---

## Configuration

```bash
# Engine and model
ENGINE_TYPE=funasr            # funasr | mlx
MODEL_ID=                     # Override model for any engine (highest priority)
FUNASR_MODEL_ID=iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
MLX_MODEL_ID=mlx-community/Qwen3-ASR-1.7B-8bit

# Service
HOST=0.0.0.0
PORT=50070
MAX_QUEUE_SIZE=50
MAX_UPLOAD_SIZE_MB=200
ALLOWED_ORIGINS=http://localhost,http://127.0.0.1
LOG_LEVEL=INFO

# Audio processing (MLX engine only)
MAX_AUDIO_DURATION_MINUTES=50   # Auto-chunk audio longer than this
SILENCE_THRESHOLD_SEC=0.5
SILENCE_NOISE_DB=-30dB
CHUNK_OVERLAP_SECONDS=15
```

Copy `.env.example` to `.env` to persist settings.

---

## Testing

```bash
uv run python -m pytest                  # all tests
uv run python -m pytest tests/unit       # unit (mocked)
uv run python -m pytest tests/integration
uv run python -m pytest tests/e2e        # real model, slow

# Code quality
uv run mypy src/
uv run ruff check src/
```

### Benchmark

```bash
uv run python benchmarks/run.py                      # default fixture
uv run python benchmarks/run.py --file path/to.wav   # specific file
uv run python benchmarks/run.py --all --save --compare  # compare all models, save JSON
```

---

## Architecture

```
src/
├── api/          # HTTP routes + Pydantic schemas (contract layer)
├── services/     # Async queue + serial worker (scheduling layer)
├── core/         # Engine abstraction + FunASR/MLX implementations
│   ├── base_engine.py      # ASREngine Protocol + EngineCapabilities
│   ├── funasr_engine.py    # FunASR/Paraformer (diarization support)
│   ├── mlx_engine.py       # MLX Audio (Qwen3-ASR, Whisper, etc.)
│   ├── model_registry.py   # Alias → ModelSpec table (SPEC-108)
│   └── factory.py          # Engine factory
├── adapters/     # Pure functions: text cleaning, audio chunking
└── config.py     # Centralized env var configuration
```

**Key constraints:**
- `workers=1` always — prevents OOM on unified memory
- Queue depth 50 — returns 503 when full
- Model switch order: `release()` → `load()` (memory-safe, no double-peak)

→ See `docs/` for full specs and ADRs.
