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

## Pipeline Profiles

| Alias | Components | Requestable | Notes |
|-------|------------|:-----------:|-------|
| `qwen3-sortformer` | `qwen3-asr` + `qwen3-forced-aligner` + `sortformer-diar` | ✅ experimental opt-in | Reachable for evaluation only. The current local pipeline is a deletion candidate, not a recommended meeting-transcript path. |

---

## Runtime Contract Rule

The project registers models by runtime contract, not by vendor name.

- Same runtime API means registry-only: for example, a future Qwen3-ASR model that still works with `mlx_audio.stt.utils.load_model()` and `generate_transcription()` should only need a new `ModelSpec`.
- Different runtime API means a new engine adapter: for example, the independent `parakeet-mlx` package uses `from_pretrained(...).transcribe(...)`, so it should not be hidden inside `MlxAudioEngine` unless an adapter normalizes that contract.
- Same Apple Silicon backend does not imply the same engine: MLX Metal, PyTorch MPS, CoreML/ANE, and CPU have different lifecycle and output contracts.
- The service should wrap proven upstream runtime capabilities rather than
  reimplementing model internals. For Qwen3 speaker separation, that means
  reusing `mlx-audio` Qwen3-ASR, Qwen3-ForcedAligner, and Sortformer contracts
  instead of inventing local timestamp or diarization model logic.

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
| Multi-speaker meeting today | `paraformer` | Best-verified long-form diarization path with CAM++ |
| Experimental Apple-native English speaker-separation evaluation | `qwen3-sortformer` | Keeps the experiment callable, but current real-meeting evidence does not justify recommending it |
| Emotion / event tagging | `sensevoice-small` | Unique emotion/BGM tags |

`qwen3-sortformer` is not just "Qwen3-ASR segments plus Sortformer." Local E2E
testing showed Qwen3-ASR emits chunk-level segments for the tested English
samples, which is too coarse for truthful speaker-labeled transcript output.
The requestable experiment is therefore a three-stage pipeline: Qwen3-ASR text,
Qwen3-ForcedAligner word timestamps, and Sortformer speaker turns.

Current stance is intentionally conservative: callers must explicitly request
`model=qwen3-sortformer`, and requestable status only keeps the experiment
reachable. The 57-minute English probe showed stronger transcript text than
Paraformer with materially slower runtime, but a later real English 1:1 meeting
probe showed a worse product tradeoff: top-level Qwen3 text stayed more
readable, while speaker-labeled segments lost coverage, fragmented heavily, and
cost more unified memory than this lightweight gateway should normalize. This
repo should not grow complex speaker embedding, alignment recovery, or
diarization cleanup logic to rescue the path. Prefer a stronger upstream or
open-source local diarized-ASR capability; remove this profile later if that
replacement makes the experiment unnecessary.

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
