# Local ASR Service (Mac Silicon)

High-performance local speech transcription service optimized for Apple Silicon (M-series).
OpenAI Whisper-compatible HTTP API on port **50700**.

**Multi-engine architecture:**
- **FunASR** — Paraformer (Chinese SOTA) + CAM++ speaker diarization
- **MLX Audio** — Apple-native models (Qwen3-ASR, Whisper, etc.)
- **FireRed + Sortformer** — Decoupled bilingual ASR + diarization pipeline (SPEC-011; production-hardened, 501-gated)

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

### FireRed runtime setup

`firered-asr` now uses the **official FireRed AED runtime**, not `transformers.pipeline(...)`.
The adapter also applies **conservative pre-inference chunking** before AED transcription so long
recordings do not get sent to the bare runtime as one giant sequence.
Because the upstream `fireredasr2s` package pins `torch==2.10.0`, it is **not** locked in this repo's
`pyproject.toml`. To try FireRed locally:

```bash
# Project deps already provide: cn2an, kaldi-native-fbank, textgrid
uv pip install --no-deps git+https://github.com/FireRedTeam/FireRedASR2S.git

# Start the service with FireRed as the resident engine
ENGINE_TYPE=firered uv run python -m src.main
```

On first FireRed startup, the official checkpoint `FireRedTeam/FireRedASR2-AED` (~4.7 GB) is
downloaded via `huggingface_hub.snapshot_download(...)`.

---

## Use Cases

| Scenario | Recommended | Command |
|----------|-------------|---------|
| Multi-speaker podcast / meeting | `paraformer` (default) | `uv run python -m src.main` |
| Fast single-speaker transcription | `mlx-community/Qwen3-ASR-1.7B-4bit` (MLX default) | `ENGINE_TYPE=mlx uv run python -m src.main` |
| Bulk speed-first (no diarization) | `sensevoice-small` | `FUNASR_MODEL_ID=iic/SenseVoiceSmall uv run python -m src.main` |

## API & Web UI

### Web UI (Interactive Docs) — Recommended for Quick Testing
The easiest way to test the service without using the command line:
1. Open **[http://localhost:50700/docs](http://localhost:50700/docs)** in your browser.
2. Find the `POST /v1/audio/transcriptions` endpoint.
3. Click **"Try it out"**, upload your audio file, and click **"Execute"**.
4. You can view the result on screen or click the **"Download"** button to save it.

### CLI (curl)

#### Health check
```bash
curl http://localhost:50700/health
```

#### Transcription
```bash
# Default: JSON with speaker diarization
curl http://localhost:50700/v1/audio/transcriptions \
  -F "file=@audio.mp3;type=audio/mpeg"
```
# Plain text (for RAG / LLM input)
curl http://localhost:50700/v1/audio/transcriptions \
  -F "file=@audio.mp3;type=audio/mpeg" \
  -F "output_format=txt"

# SRT subtitles
curl http://localhost:50700/v1/audio/transcriptions \
  -F "file=@audio.mp3;type=audio/mpeg" \
  -F "output_format=srt"

# Per-request model switch (hot-swap, no restart needed)
curl http://localhost:50700/v1/audio/transcriptions \
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

**Supported Models (`model` parameter):**

| Alias | Engine | Requestable | Description |
|-------|--------|-------------|-------------|
| `paraformer` | FunASR | ✅ | Mandarin ASR + speaker diarization (best for multi-speaker, 20-60min audio) |
| `qwen3-asr` | MLX | ✅ | Qwen3-ASR-1.7B-8bit (fast, low memory, English/Chinese single-speaker) |
| `sensevoice-small` | FunASR | ✅ | SenseVoice (fastest; emotion/language detection, no timestamps) |
| `firered-asr` | FireRed | ❌ | FireRedASR2-AED via official FireRed AED runtime with conservative pre-inference chunking (startup-eligible via `ENGINE_TYPE=firered`, not POST-requestable) |
| `sortformer-diar` | Sortformer | ❌ | Diarization adapter (internal component of `firered-sortformer` pipeline, not requestable) |
| `firered-sortformer` | Pipeline | ❌ | Decoupled ASR+diarization pipeline (sequential FireRed → Sortformer; discoverable, POST returns `501` until public gate lifted) |

> **Discovery vs requestable**: `GET /v1/models` lists all registered models and pipeline profiles (including discovery-only entries). Only aliases marked ✅ can be used in `POST /v1/audio/transcriptions`. Sending `model=firered-sortformer` returns `501 Not Implemented` until the decoupled runtime is publicly enabled.

> **Startup defaults vs aliases**: `ENGINE_TYPE=mlx` currently boots `MLX_MODEL_ID=mlx-community/Qwen3-ASR-1.7B-4bit`. The requestable alias `qwen3-asr` points to the registered 8-bit variant for per-request switching and `/v1/models` discovery.

### Query models

```bash
curl http://localhost:50700/v1/models | jq          # all registered models
curl http://localhost:50700/v1/models/current | jq  # currently loaded model + capabilities
```

### Interactive docs

```
http://localhost:50700/docs
```

---

## Configuration

```bash
# Engine and model
ENGINE_TYPE=funasr            # funasr | mlx | firered
MODEL_ID=                     # Override model for any engine (highest priority)
FUNASR_MODEL_ID=iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
MLX_MODEL_ID=mlx-community/Qwen3-ASR-1.7B-4bit
FIRERED_MODEL_ID=FireRedTeam/FireRedASR2-AED  # FireRed engine default (startup-eligible; not publicly requestable yet)

# Service
HOST=0.0.0.0
PORT=50700
MAX_QUEUE_SIZE=50
MAX_UPLOAD_SIZE_MB=200
ALLOWED_ORIGINS=http://localhost,http://127.0.0.1
LOG_LEVEL=INFO
MODEL_IDLE_TIMEOUT_SEC=60     # Worker auto-terminates after idle period (0 = disabled, worker stays resident)


# Audio processing (MLX engine only)
MAX_AUDIO_DURATION_MINUTES=50   # Auto-chunk audio longer than this
SILENCE_THRESHOLD_SEC=0.5
SILENCE_NOISE_DB=-30dB
CHUNK_OVERLAP_SECONDS=15
```

Copy `.env.example` to `.env` to persist settings.

> **FireRed note**: `ENGINE_TYPE=firered` requires the upstream runtime source package:
> `uv pip install --no-deps git+https://github.com/FireRedTeam/FireRedASR2S.git`

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

## Memory Management

This service is designed for Apple Silicon M-series chips with unified memory. Model memory consumption is **17–23 GB** during transcription.

### Idle Model Offloading (SPEC-009 v2)

By default (`MODEL_IDLE_TIMEOUT_SEC=60`), the ASR model runs in an isolated child subprocess. When idle for 60+ seconds with no pending requests, the subprocess self-terminates, allowing the OS to reclaim all memory (including the PyTorch MPS Metal heap).

**Memory profile:**
- **Startup** (before first request): ~150 MB
- **Transcribing** (model active): ~20–23 GB
- **Idle** (after 60s timeout): < 500 MB

**When the next request arrives**, a new subprocess is spawned and the model reloaded:
- FunASR: ~10–30s reload (warm cache)
- MLX: ~3–5s reload

To disable idle termination (keep model resident):
```bash
MODEL_IDLE_TIMEOUT_SEC=0 uv run python -m src.main
```

**Design rationale**: In-process `release()` calls (SPEC-009 v1) could only drop memory from 23 GB to 15–18 GB because PyTorch's MPS allocator retains its Metal heap. Process termination ensures complete memory reclamation.

---

## Decoupled Pipeline (SPEC-011)

The `firered-sortformer` profile implements a **production-hardened decoupled pipeline** for ASR + diarization:

- **Sequential execution**: ASR (FireRed) → Diarization (Sortformer) → Speaker alignment
- **Independent model switching**: Models load/release in strict order (no double-peak memory)
- **Result alignment**: Diarization speaker turns aligned to ASR segment timestamps
- **Full lifecycle hardening**: Cancellation-safe cleanup, ownership gating, half-init prevention, comprehensive logging
- **501-gated public endpoint**: Pipeline is discoverable (`GET /v1/models`) but POST returns `501 Not Implemented` until explicitly enabled
  
See [SPEC-011-Decoupled-ASR-Diarization.md](./docs/SPEC-011-Decoupled-ASR-Diarization.md) for full design + API details.

---

```
src/
├── api/          # HTTP routes + Pydantic schemas (contract layer)
├── services/     # Async queue + serial worker (scheduling layer)
├── core/         # Engine abstraction + implementations
│   ├── base_engine.py        # ASREngine Protocol + EngineCapabilities
│   ├── funasr_engine.py      # FunASR/Paraformer (diarization support)
│   ├── mlx_engine.py         # MLX Audio (Qwen3-ASR, Whisper, etc.)
│   ├── firered_engine.py     # FireRed ASR adapter (SPEC-011 Phase 1 plumbing)
│   ├── sortformer_engine.py  # Sortformer diarization adapter (SPEC-011 Phase 1 plumbing)
│   ├── diarization_port.py   # Diarization port interface
│   ├── pipeline_registry.py  # Decoupled pipeline profiles (firered-sortformer)
│   ├── model_registry.py     # Alias → ModelSpec table (SPEC-108)
│   └── factory.py            # Engine factory
├── adapters/     # Pure functions: text cleaning, audio chunking, segment alignment
└── config.py     # Centralized env var configuration
```

**Key constraints:**
- `workers=1` always — prevents OOM on unified memory
- Queue depth 50 — returns 503 when full
- Model switch order: `release()` → `load()` (memory-safe, no double-peak)

→ See `docs/` for full specs and ADRs.
