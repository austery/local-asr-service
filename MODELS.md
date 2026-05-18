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

## Pipeline Profiles

| Alias | Components | Requestable | Notes |
|-------|------------|:-----------:|-------|
| `qwen3-sortformer` | `qwen3-asr` + `sortformer-diar` | ✅ | Requestable worker-backed pipeline that combines Qwen3-ASR transcription with Sortformer diarization |

---

## Runtime Contract Rule

The project registers models by runtime contract, not by vendor name.

- Same runtime API means registry-only: for example, a future Qwen3-ASR model that still works with `mlx_audio.stt.utils.load_model()` and `generate_transcription()` should only need a new `ModelSpec`.
- Different runtime API means a new engine adapter: for example, the independent `parakeet-mlx` package uses `from_pretrained(...).transcribe(...)`, so it should not be hidden inside `MlxAudioEngine` unless an adapter normalizes that contract.
- Same Apple Silicon backend does not imply the same engine: MLX Metal, PyTorch MPS, CoreML/ANE, and CPU have different lifecycle and output contracts.

---

## Performance Benchmark (M1 Max, 2026-02-25)

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
| English/European-language throughput path | Re-evaluate Parakeet | Candidate after per-engine chunking and runtime validation |
| Multi-speaker meeting today | `paraformer` | Best-verified long-form diarization path with CAM++ |
| Apple-native multi-speaker pipeline | `qwen3-sortformer` | Requestable Qwen3-ASR + Sortformer pipeline when you want MLX-backed ASR plus diarization |
| Emotion / event tagging | `sensevoice-small` | Unique emotion/BGM tags |

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
