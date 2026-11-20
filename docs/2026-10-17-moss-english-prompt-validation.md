# MOSS English-preservation prompt validation

Date: 2026-10-17. Related: [issue #42](https://github.com/austery/local-asr-service/issues/42).

## Input and fixed conditions

- [Publisher page](https://www.pastorrick.com/current-teaching/series/finding-and-keeping-the-love-of-your-life-2026?view=how-to-get-a-marriage-makeover---part-1): *How to Get a Marriage Makeover - Part 1*.
- Original MP3: 1,193.51925 seconds, 14,323,255 bytes; SHA-256 `758cf89ca2e5c9b5a2a56718f649694a580af9547d592bc6a84a228ccf1bf6dd`.
- Model: `OpenMOSS-Team/MOSS-Transcribe-Diarize`, revision `704aa4a9c304e8520be88901e0d1960158ef5b15`.
- Installed runtime: mlx-audio 0.5.1, MLX 0.31.2, Transformers 5.16.1; Python 3.11.14 on macOS 26.6.2 arm64.
- Greedy decoding: temperature 0, 32,768 output tokens, 4,096 prefill step.

## Short input did not predict full-recording behavior

On the same first 30 seconds decoded to mono 16 kHz WAV, the MLX default,
official English example, official Chinese default, and default plus a
`language=en` keyword each produced Spanish in three of three runs. Each
condition's repeated raw output was identical. Appending an English-preservation
instruction to the MLX default produced English in three of three runs.

That appended instruction was insufficient on the original full MP3. A second
candidate leads with explicit English-only transcription, prohibits translation
and paraphrasing, retains timestamp/speaker instructions, and repeats that all
segment text must remain English. This is the prompt now fixed in
`MOSS_ENGLISH_TRANSCRIPTION_PROMPT` in `src/core/mlx_engine.py`.

| Full MP3 condition | Inference | Output tokens | Segments | Observed language | Last end |
| --- | --- | --- | --- | --- | --- |
| Default with appended English preservation | 207.681 s | 9,822 | 278 | Spanish | 1,182.04 s |
| Stronger English-preservation candidate | 202.885 s | 9,712 | 307 | English | 1,181.83 s |

Both runs exercised the MOSS adapter in an isolated checkout with the original
full MP3. The stronger candidate was selected through a diagnostic-only override
before being written into source; its complete prompt was then checked for exact
equality with the final source constant. Raw output was captured before normal
validation and cleanup. English transcript samples were inspected across the
recording, including its beginning and closing sponsorship statement. These are
single full-recording runs, not repeated reliability or accuracy benchmarks.
Several instruction changes were made together; their individual effects are
not isolated.

## Remaining failure and operational boundary

The stronger candidate still fails the unchanged coverage validator: the final
segment leaves 11.68925 seconds uncovered, exceeding the 10-second limit. The
adapter raises `TranscriptionOutputError`, which the API maps to 422. This test
does not establish a successful full-file HTTP response. Coverage rules, fixed
English response metadata, and general output-language detection are not changed.

No numerical WER/CER, speaker accuracy, native PyTorch parity, or universal
English-preservation guarantee is established. The server stayed running with
the same PID; its checkout and environment were not changed. Diagnostic model
processes ran serially after its resident model had naturally exited, then
released their models. No service restart or deployment was performed.
