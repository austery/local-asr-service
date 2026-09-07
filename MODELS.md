# Model Reference

> Single source of truth for all supported models. Update this file when adding, removing, or re-evaluating models.
> The authoritative alias table lives in `src/core/model_registry.py`.

---

## Active Models

| Alias | Engine Contract | Model ID | Diarization | Notes |
|-------|-----------------|----------|:-----------:|-------|
| `paraformer` | FunASR (`funasr.AutoModel` on PyTorch MPS/CPU) | `iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` | ✅ | Mandarin-focused default with CAM++ diarization |
| `sensevoice-small` | FunASR (`funasr.AutoModel` on PyTorch MPS/CPU) | `iic/SenseVoiceSmall` | ❌ | Fast language/emotion tags, no timestamps |
| `qwen3-asr` | mlx-audio (`load_model` + `generate_transcription` on MLX Metal) | `mlx-community/Qwen3-ASR-1.7B-8bit` | ❌ | Chinese/English quality-first ASR; language prompts are normalized before inference |
| `apple-speech` | Apple SpeechAnalyzer `SpeechTranscriber` via Swift sidecar | `apple-speech:speechTranscriber` | ❌ | macOS 26+ local ASR-only path; requires explicit `language=zh/en` or `zh-CN/en-US`; short codes are mapped internally; recommended low-resource ASR-only option after Phase 3 long-audio review (verified strong low-resource candidate); no speaker labels without a separate diarization stage |
| `moss-transcribe-diarize` | mlx-audio MLX | `OpenMOSS-Team/MOSS-Transcribe-Diarize` | ✅ | Opt-in English continuous speech, at most 30 minutes; requires `language=en` |

## Bounded MOSS (Opt-in)

`moss-transcribe-diarize` uses `OpenMOSS-Team/MOSS-Transcribe-Diarize` through
mlx-audio 0.5.1. It is an explicit opt-in English speaker transcription path;
it keeps the default Paraformer model and the standalone ASR model roles unchanged.

```bash
curl http://127.0.0.1:50700/v1/audio/transcriptions \
  -F file=@conversation.wav -F model=moss-transcribe-diarize \
  -F language=en -F response_format=verbose_json
```

- **Input:** one recording, at most 30 minutes, explicit English. The model
  receives the whole file in one call. Speaker IDs are local to that recording.
- **Output:** clean `text`, timestamped `segments` with `speaker`, input
  `duration`, and requested `language=en`. JSON, text (including optional
  timestamps), and SRT use the existing API formats.
- **Rejection:** input outside the operating scope returns 400. Detectable
  truncation, malformed speaker/timestamp output, or any uncovered interval over
  10 seconds returns 422. Continuous-speech recordings are the intended scope;
  long legitimate silence can also trigger this conservative check. Endpoint
  overruns up to 0.25 seconds are clamped to the recording duration.
- **Budget:** 32,768 output tokens; a 900-second worker inference deadline.
  Deadline termination follows the existing worker failure response (500),
  fails queued jobs on that worker, and permits the next request to reload.
- **Evidence:** two disjoint 30-minute All-In samples completed in 316 and 357
  seconds, with MLX allocator peaks of 6.16 and 6.37 GiB on the evaluation Mac.
  These are raw-model single runs; loading and API overhead are additional.
  The user accepted transcription and speaker quality by listening. Numerical
  WER/CER and speaker accuracy are unmeasured.
- **Limit:** 30 minutes is a provisional service policy, not a universal model
  maximum or a guarantee of complete output. Local 40/60-minute probes ended
  early near 16K output tokens despite an unused 32K budget. Validation catches
  structural failures, not every omission. There is no automatic splitting or
  cross-recording speaker reconciliation.

The MOSS checkpoint is pinned to `704aa4a9c304e8520be88901e0d1960158ef5b15`, the evaluated revision. Upstream output files live under the request temporary directory so parent cleanup removes them even after worker termination.

See the [adapter contract](docs/plans/2026-09-07-moss-adapter-contract.md).

## Retired Profiles

`qwen3-sortformer` is no longer registered or requestable. It is omitted from
`GET /v1/models`, and POST requests using this alias return 400. Standalone
`qwen3-asr` remains supported. Lei requested retirement after accepting the
bounded MOSS path; MOSS is not a replacement for unrestricted long recordings
or cross-recording speaker identity.

Historical pipeline evidence and generic alignment/diarization/worker code are
retained. Internal pipeline infrastructure is not a supported public model.
No model cache or downloaded weights are deleted by this change.

---

## Runtime Contract Rule

The project registers models by runtime contract, not by vendor name.

- Same runtime API means registry-only: for example, a future Qwen3-ASR model that still works with `mlx_audio.stt.utils.load_model()` and `generate_transcription()` should only need a new `ModelSpec`.
- Different runtime API means a new engine adapter: for example, the independent `parakeet-mlx` package uses `from_pretrained(...).transcribe(...)`, so it should not be hidden inside `MlxAudioEngine` unless an adapter normalizes that contract.
- Same Apple Silicon backend does not imply the same engine: MLX Metal, PyTorch MPS, CoreML/ANE, and CPU have different lifecycle and output contracts.
- The service should wrap proven upstream runtime capabilities rather than
  reimplementing model internals. MOSS uses one upstream transcription call
  with a thin output validator; this service does not reconcile speakers
  across recordings.

---

## Performance Benchmark (M1 Max, 2026-02-25)

These benchmark rows predate Apple Speech integration. Use
`benchmarks/phase3_evaluation.py` for SPEC-014 Phase 3 Apple Speech comparisons
against Paraformer and Qwen3-ASR.

### SPEC-014 Phase 3 Long Audio Probe (53m Chinese Session, 2026-07-05)

Command shape:

```bash
uv run python benchmarks/phase3_evaluation.py \
  --file /Users/leipeng/Downloads/750BF500-09E2-4821-B2B9-15383C915051.wav \
  --language zh-CN \
  --models apple-speech paraformer \
  --base-url http://127.0.0.1:50700 \
  --server-pid 76957 \
  --timeout 7200 \
  --srt-probe \
  --save
```

`qwen3-asr` was rerun with `language=zh` because the live service still had the
pre-fix Qwen3 locale alias bug where `zh-CN` was not normalized to `Chinese`.

| Model | Language | Status | Elapsed | RTF | Realtime | Peak process-tree RSS | Segment/SRT notes |
|-------|----------|--------|---------|-----|----------|------------------------|-------------------|
| `apple-speech` | `zh-CN` | ✅ | 20.77s | 0.0065 | 154.4x | 88.9 MB | 213 JSON segments; SRT valid; JSON segment monotonicity flagged false |
| `paraformer` | `zh-CN` | ✅ | 100.38s | 0.0313 | 31.9x | 6444.6 MB | 1129 JSON segments; SRT valid; monotonic timing |
| `qwen3-asr` | `zh` | ✅ | 237.92s | 0.0742 | 13.5x | 4500.5 MB | 5 JSON segments; SRT probe produced no valid cues |

Early interpretation:

- `apple-speech` is the strongest low-resource local ASR candidate for long
  Chinese dictation/transcription. Its speed and memory profile are materially
  better than both local neural-model paths on this sample. User review of the
  full transcript found the output better than expected for a familiar long
  Mandarin therapy conversation; mixed English terms were imperfect but
  recognizable.
- `paraformer` remains the structurally safest long-form meeting path when
  timestamp density, SRT correctness, and diarization matter.
- `qwen3-asr` can produce usable long-form text when given the runtime's expected
  language value, but its long-audio segment granularity is too coarse for SRT or
  downstream speaker/timeline workflows in this probe.

### Short Audio (60s, two-speaker English conversation)

| Model | RTF | Realtime | Notes |
|-------|-----|----------|-------|
| `paraformer` | 0.13 | 7.6x | Slower on short clips |
| `sensevoice-small` | ~0.067 | ~15x | No timestamps |
| `qwen3-asr` | 0.028 | 36.3x | Good on short clips |

### Long Audio (23min, bilingual conversation)

| Model | RTF | Realtime | Notes |
|-------|-----|----------|-------|
| `paraformer` | **0.015** | **65.3x** | 🏆 Best — FunASR batch processing scales with length |
| `qwen3-asr` | 0.107 | 9.3x | Autoregressive degradation on long sequences |

**Key insight**: Short-audio benchmarks are misleading. Always test with ≥10min samples for production decisions.

---

## Model Selection Guide

| Use case | Recommended model | Reason |
|----------|------------------|--------|
| Mandarin long-form podcast (20-60min) | `paraformer` | Best verified long-audio RTF, CAM++ diarization |
| Chinese/English quality-first single-speaker audio | `qwen3-asr` | MLX-native Qwen3-ASR with explicit language prompt forwarding |
| Spokenly local dictation fallback | `qwen3-asr` | Best current local path for low-latency single-speaker voice input through an OpenAI-compatible endpoint |
| Apple-native low-resource local dictation/transcription on macOS 26+ | `apple-speech` | Recommended ASR-only low-resource path after Phase 3 long-audio evidence and user quality review; no speaker labels |
| English/European-language throughput path | Re-evaluate Parakeet | Candidate after per-engine chunking and runtime validation |
| Mandarin multi-speaker meeting | `paraformer` | Best-verified long-form diarization path with CAM++ |
| English multi-speaker continuous speech up to 30 minutes | `moss-transcribe-diarize` | User-accepted bounded text/speaker quality; explicit `language=en` |
| Emotion / event tagging | `sensevoice-small` | Unique emotion/BGM tags |

The retired Qwen3/forced-alignment/Sortformer experiment preserved stronger
Qwen3 text on earlier samples but had costly, fragmented, incomplete speaker
segments on a real meeting. Historical evidence remains in the evaluation
records; it does not justify advertising that pipeline as a supported model.

For PureSubs long audio, follow the [planned caller integration](docs/plans/2026-09-07-puresubs-moss-integration.md).
Size-only splitting is insufficient, and speaker IDs from separate chunks
must not be treated as global identities.

---

## FunASR Model Details: Paraformer vs SenseVoice

| Dimension | SEACO-Paraformer (default) | SenseVoiceSmall |
|-----------|--------------------------|-----------------|
| Architecture | Non-autoregressive encoder-decoder + CIF | Non-autoregressive encoder-only |
| Mandarin CER | **1.95%** | 2.96% |
| Mixed CER | 9.65% | **6.71%** |
| Timestamps | ✅ | ❌ |
| Speaker diarization | ✅ (with CAM++) | ❌ |
| Emotion tags | ❌ | ✅ (`<\|HAPPY\|>` etc.) |
| Audio event detection | ❌ | ✅ (`<\|BGM\|>` etc.) |

> SenseVoice output contains special tags like `<|zh|><|NEUTRAL|><|Speech|>`. The service auto-cleans them via `clean_sensevoice_tags()`.

---

## Deregistered Models

Models that were evaluated and removed. Kept here as a performance review record.

### qwen3-asr-mini (Qwen3-ASR-1.7B-4bit) — Removed 2026-02-25

- **Short audio**: 36.3x RTF (60s) — looked promising
- **Long audio**: 9.3x RTF (23min) — 4× degradation due to autoregressive token dependencies
- **Verdict**: Inferior to `paraformer` for the primary use case. `qwen3-asr` (8-bit) retained for English short clips with lower memory than paraformer.

### parakeet (parakeet-tdt-0.6b-v2) — Removed 2026-02-25

- **Short audio**: 121.7x RTF (60s) — fastest model tested
- **Long audio**: ❌ Metal OOM crash on >5min audio
- **Root cause**: MLX Metal memory budget exceeded on full-length sequences; chunking threshold (50min) too high for this model
- **Verdict**: Cannot be used in production until OOM is fixed. Re-evaluate if chunking is implemented per-engine.

---

## Model Storage Paths

### FunASR (ModelScope cache)

```
~/.cache/modelscope/hub/models/iic/

Paraformer pipeline (all required):
├─ speech_seaco_paraformer_large_...  (ASR main model, ~950MB)
├─ speech_fsmn_vad_zh-cn-16k-...     (VAD, ~4MB)
├─ punc_ct-transformer_cn-en-...     (Punctuation, ~1.1GB)
└─ speech_campplus_sv_zh-cn-...      (CAM++ speaker diarization, ~28MB)

Optional:
└─ SenseVoiceSmall                   (~900MB)
```

```bash
du -sh ~/.cache/modelscope/hub/models/iic/*
```

### MLX (HuggingFace cache)

```
~/.cache/huggingface/hub/

Active:
└─ models--mlx-community--Qwen3-ASR-1.7B-8bit   (~2.3GB)
```

```bash
du -sh ~/.cache/huggingface/hub/models--mlx-community*
```

### Cleanup

```bash
# Remove a specific FunASR model
rm -rf ~/.cache/modelscope/hub/models/iic/SenseVoiceSmall

# Remove a specific MLX model
rm -rf ~/.cache/huggingface/hub/models--mlx-community--Qwen3-ASR-1.7B-8bit

# Check total cache size
du -sh ~/.cache/modelscope ~/.cache/huggingface
```
