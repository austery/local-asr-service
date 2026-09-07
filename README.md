# Local ASR Service (Mac Silicon)

Local speech runtime gateway optimized for Apple Silicon (M-series).
It exposes an OpenAI Whisper-compatible HTTP API on port **50700** for local
dictation, transcription, and batch ASR workflows.

The project exists because cloud speech APIs are not always reliable enough for
interactive work. A 10-20 second stall from a remote provider can break flow for
dictation and block downstream batch pipelines. This service keeps a local,
high-quality fallback available for tools that already know how to call an
OpenAI-compatible transcription endpoint.

## Project Role

This repo is the **API, queue, runtime-isolation, and response-normalization
layer**. It should stay thin around model runtimes instead of becoming a new
speech-model framework.

Primary consumers:

- **Spokenly dictation**: low-latency, single-speaker voice input through the
  OpenAI-compatible custom API provider.
- **puresubs batch transcription**: longer-form transcription and future speaker
  separation integration for offline processing.

Non-goals:

- Reimplement `mlx-audio`, WhisperKit, FunASR, Sortformer, forced alignment, or
  speaker-clustering internals.
- Turn this service into a generic audio ML framework.
- Mix low-latency dictation behavior with slower batch speaker-diarization
  behavior in one default path.

The preferred direction is to reuse strong upstream runtimes such as
`mlx-audio` for Qwen3-ASR and MOSS speaker transcription, while this
service owns the stable local HTTP contract and Apple Silicon memory boundary.

## Background

This project grew out of earlier local transcription workflows for YouTube and
puresubs. The initial reference point was Whisper-style local API servers such
as WhisperKit: useful server shape, but limited by Whisper-family model quality
and behavior. The service then explored FunASR/SenseVoice and Paraformer for
local transcription and speaker diarization, added MLX/Qwen3-ASR for stronger
Chinese/English transcription quality, and now includes bounded English MOSS speaker transcription through the upstream MLX runtime.

**Triple-engine architecture:**
- **FunASR** — Paraformer (Chinese SOTA) + CAM++ speaker diarization
- **MLX Audio** — Apple-native models (Qwen3-ASR, Whisper, etc.)
- **Apple Speech** — macOS 26+ SpeechAnalyzer/SpeechTranscriber native APIs via Swift sidecar worker (ASR-only)

**Runtime-aware model registration:** models are added through `ModelSpec` when
they fit an existing runtime contract. New engines are only needed for new
runtime APIs, not for every new model release.

→ See [MODELS.md](./MODELS.md) for model selection, [CHANGELOG.md](./CHANGELOG.md) for delivered changes, and the [runtime roadmap](docs/plans/2026-09-07-runtime-upgrades-and-moss-evaluation.md) for follow-up work.

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
| Spokenly local dictation fallback | `qwen3-asr` | `ENGINE_TYPE=mlx uv run python -m src.main` |
| Chinese/English quality-first single-speaker transcription | `qwen3-asr` | `ENGINE_TYPE=mlx uv run python -m src.main` |
| Mandarin multi-speaker podcast / meeting today | `paraformer` (default) | `uv run python -m src.main` |
| Bulk speed-first tags / language detection | `sensevoice-small` | `FUNASR_MODEL_ID=iic/SenseVoiceSmall uv run python -m src.main` |
| English multi-speaker continuous speech, at most 30 minutes | `moss-transcribe-diarize` | Per request: `model=moss-transcribe-diarize`, `language=en` |

## API & Web UI

### Web UI (Interactive Docs) — Recommended for Quick Testing
The easiest way to test the service without using the command line:
1. Open **[http://localhost:50700/docs](http://localhost:50700/docs)** in your browser.
2. Find the `POST /v1/audio/transcriptions` endpoint.
3. Click **"Try it out"** and upload your audio file.
4. Enter the exact alias in the free-text `model` field. For MOSS, use
   `moss-transcribe-diarize`, set `language` to `en`, and use an English
   continuous-speech recording no longer than 30 minutes.
5. Set `response_format` to `verbose_json` to inspect `segments[].speaker`,
   timestamps, and text, then click **"Execute"**.
6. Inspect the response or download it. A MOSS input over 1,800 seconds returns
   400; detectable incomplete output or a gap over 10 seconds returns 422.

MOSS receives one complete recording per call and does not automatically split
long uploads. A small compressed file can still exceed 30 minutes. Omitting
`model` keeps the current model; `whisper-1` is also passthrough and does not
select Whisper. Use `GET /v1/models` for the active aliases. A running server
must load the updated application before `/docs` reflects source changes.

### CLI (curl)

#### Health check
```bash
curl http://localhost:50700/health
```

#### Transcription
```bash
# Default: JSON response
curl http://localhost:50700/v1/audio/transcriptions \
  -F "file=@audio.mp3;type=audio/mpeg"

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

# English MOSS speaker transcription (one recording, at most 30 minutes)
curl http://localhost:50700/v1/audio/transcriptions \
  -F "file=@conversation.mp3;type=audio/mpeg" \
  -F "model=moss-transcribe-diarize" \
  -F "language=en" \
  -F "response_format=verbose_json"

# Apple Speech ASR-only path (requires explicit language; short codes are preferred)
curl http://localhost:50700/v1/audio/transcriptions \
  -F "file=@audio.mp3;type=audio/mpeg" \
  -F "model=apple-speech" \
  -F "language=zh"
```

**Request parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `file` | required | Audio file (wav, mp3, m4a, flac, ogg, webm) |
| `output_format` | `json` | Output: `json`, `txt`, `srt` |
| `response_format` | — | OpenAI alias: `verbose_json`, `text`, `vtt` |
| `with_timestamp` | `false` | Prepend `[MM:SS]` to each line in txt mode |
| `language` | `auto` | `zh`, `zh-CN`, `en`, `en-US`, `auto`; MOSS requires `en`; Apple Speech requires an explicit language, accepts `zh`/`en` as API-level short codes, and rejects `auto` |
| `model` | — | Alias or full model path. Omit to keep current model. |

**Supported Models (`model` parameter):**

| Alias | Engine | Description |
|-------|--------|-------------|
| `paraformer` | FunASR | FunASR/PyTorch MPS path; Mandarin-focused with CAM++ diarization |
| `qwen3-asr` | MLX | mlx-audio/MLX Metal path; Chinese/English quality-first ASR |
| `sensevoice-small` | FunASR | FunASR/PyTorch MPS path; speed-first language/emotion tags |
| `moss-transcribe-diarize` | MLX | Opt-in English speaker transcription, at most 30 minutes; requires `language=en` |
| `apple-speech` | Apple Speech | macOS 26+ SpeechAnalyzer sidecar; ASR-only path; requires explicit `language=zh/en` or `zh-CN/en-US`; short codes are mapped internally |

`qwen3-sortformer` has been retired from the public API. It is absent from
`GET /v1/models`, and explicit requests return 400. The standalone `qwen3-asr`
model remains supported. Historical evaluation records and internal pipeline
infrastructure remain for traceability and a separate reference-audited cleanup.

PureSubs currently normalizes audio at a default 64 kbps and splits by a 24 MiB
size threshold. This does not enforce MOSS's 30-minute bound: a 40-minute file
is approximately 18.3 MiB. Long-video MOSS integration is **planned, not
implemented**; see the [PureSubs handoff](docs/plans/2026-09-07-puresubs-moss-integration.md).

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
ENGINE_TYPE=funasr            # funasr | mlx
MODEL_ID=                     # Override model for any engine (highest priority)
FUNASR_MODEL_ID=iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
MLX_MODEL_ID=mlx-community/Qwen3-ASR-1.7B-8bit

# Service
HOST=0.0.0.0
PORT=50700
MAX_QUEUE_SIZE=50
MAX_UPLOAD_SIZE_MB=200
ALLOWED_ORIGINS=http://localhost,http://127.0.0.1
LOG_LEVEL=INFO
MODEL_IDLE_TIMEOUT_SEC=60     # Worker auto-terminates after idle period (0 = disabled, worker stays resident)

# Apple Speech sidecar (macOS 26+)
APPLE_SPEECH_WORKER_PATH=apple-speech-worker/.build/debug/apple-speech-worker
APPLE_SPEECH_WORKER_TIMEOUT_SEC=120
APPLE_SPEECH_MAX_CONCURRENCY=1

# Audio processing (generic MLX path; MOSS has a separate fixed 30-minute bound)
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

## Memory Management

This service is designed for Apple Silicon M-series chips with unified memory.
Memory use depends on the model. The established FunASR path can consume
**17–23 GB** while active; see [MODELS.md](MODELS.md) for the separately measured
MOSS allocator peaks and their measurement scope.

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
