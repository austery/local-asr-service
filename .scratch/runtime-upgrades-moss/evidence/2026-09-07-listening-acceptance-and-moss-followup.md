# Listening Acceptance and MOSS Follow-up — 2026-09-07

## User decision

Lei listened to the review material and explicitly accepted the recognition quality of both runtime upgrades. Lei judged the changed FunASR text to be improved. Select MLX Audio 0.5.1 and FunASR 1.4.14 as the accepted runtime candidates, with their previously recorded frozen dependency sets and unchanged checkpoints. The existing CAM++ patch remains required.

This records qualitative user acceptance. No WER/CER or speaker-accuracy score was measured, and the user did not provide corrected reference transcripts or interval-by-interval speaker annotations. Existing mypy and timestamp/subtitle defects remain documented; they are not newly passing checks. Acceptance does not itself perform a push, merge, or production-environment installation.

Lei also judged the reviewed MOSS words and speaker distinctions positively and requested investigation of its duration/token limits instead of abandoning it. The prior No-Go recommendation was not adopted. Retain MOSS as an experimental candidate; unrestricted long-form admission is unresolved because the one-hour output omitted the final 23:41.

## Recorded time changes

| Same input / runtime path | Before | After | Change |
| --- | --- | --- | --- |
| Qwen, 60-minute English, MLX upgrade only | 333.17 s | 305.80 s | 27.37 s faster; 8.2% |
| Qwen, 53-minute Chinese, MLX upgrade only | 233.45 s | 208.32 s | 25.13 s faster; 10.8% |
| Paraformer, 60-minute English, original baseline to FunASR candidate | 109.36 s | 116.23 s | 6.87 s slower; 6.3% |
| Paraformer, 53-minute Chinese, original baseline to FunASR candidate | 49.11 / 51.97 s | 77.27 / 84.62 s | Matched first/repeat observations: 28.16 / 32.65 s slower; 57.3% / 62.8% |
| MOSS, 60-minute English, direct upstream call | No previous MOSS baseline | 715.80 s inference + 2.10 s load | About 11 min 56 s inference, with output only through 36:19 |

The MLX-only Paraformer English/Chinese observations were 110.39 / 48.86 seconds; FunASR's incremental first-run differences from that environment are +5.84 / +28.41 seconds. Repeat observations are not benchmark medians. MOSS used a direct upstream call while Qwen and Paraformer used HTTP service paths, and MOSS output is incomplete; the times are practical observed costs, not a controlled equal-output speed ranking. The accepted runtime candidates are still substantially faster than the duration of these recordings.

## MOSS duration and token interpretation

See [official limits and known early-stop evidence](2026-09-07-moss-upstream-limits.md). The publisher advertises up to 90 minutes in one pass. Our current MLX/checkpoint pairing has only short/medium completed probes; the 60-minute probe fails coverage. Neither 36 minutes nor a proposed 30-minute operational cap is established as a reliable maximum.

The output budget was 32,768 tokens and actual output was 16,376 tokens. The resolved checkpoint configuration is 131,072 context positions, and input plus generated output was 64,114 tokens. Local MLX code passes the output budget to a count-bounded generator and additionally stops on EOS. Early EOS is the source-path explanation consistent with this run; the exact terminal token/logits were not captured. Similar approximately 16K early-stop reports exist on the official tracker, including maintainer acknowledgement. A learned early-stop mechanism is a plausible upstream explanation, not a locally proven root cause.

## Next verification boundary

Preserve the one-pass speaker context and do not add an unverified cross-chunk speaker reconciliation system. A bounded follow-up should record termination token/reason and test complete shorter inputs of different speech densities before proposing a local duration limit. Compare the same long recording using the official default prompt as a separate variable if investigating prompt sensitivity. Merely raising an unused budget or suppressing EOS is not a demonstrated fix. A completed shorter sample establishes sample-specific evidence, not a guarantee for every recording of that duration.

Current runtime source and dependencies remain unchanged during this research follow-up. The public HTTP MOSS adapter and admission limits remain a separate implementation decision.
