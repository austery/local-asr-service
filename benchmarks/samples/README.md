# Benchmark Samples

Place audio files here for benchmarking. Audio files are gitignored.

## Recommended Test Set

| File | Duration | Language | Speakers | Purpose |
|------|----------|----------|----------|---------|
| `short_zh.wav` | ~10s | Chinese | 1 | Latency baseline |
| `medium_en.wav` | ~60s | English | 2 | Diarization + mid-length |
| `long_mixed.mp3` | ~5min | Mixed | 2+ | Throughput + chunking |

## Quick Start

Copy the fixture file to get started:

```bash
cp tests/fixtures/two_speakers_60s.wav benchmarks/samples/medium_en.wav
```
