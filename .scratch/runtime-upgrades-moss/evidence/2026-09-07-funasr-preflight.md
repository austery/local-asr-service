# FunASR Runtime Validation — 2026-09-07

Status: FunASR 1.4.14 accepted by Lei after listening review. Keep the unchanged CAM++ patch and record measured cost increases; see [acceptance record](2026-09-07-listening-acceptance-and-moss-followup.md).

## Candidate and isolation

Candidate lock SHA-256: `2c3ced2b2d2b9ffba961e0c3b0d723442077d96bcd8c1f457c4112bd030f6b4b`.

The private project copy `.runtime-validation/funasr-lock-probe` uses FunASR 1.4.14 on top of the MLX Audio 0.5.1 candidate. uv resolved NumPy 2.3.5 → 1.26.4, removed editdistance and pytorch-wpe, and added rapidfuzz 3.14.6 and websockets 17.1. Other locked package versions, including MLX Audio, Transformers, Torch, and Torchaudio, stayed unchanged. The repository has no direct imports of the removed packages.

A clean `uv sync --frozen --dev --prerelease=if-necessary` installed `.venv-funasr` successfully using this isolated project's lock. The project copy contains the unchanged runtime source. Test commands run against the worktree source through explicit PYTHONPATH. Neither the original main environment nor `.venv-mlx` is changed; the original runtime-roadmap worktree's dependency declaration remains the MLX-only candidate.

## Existing checks

- Unit, integration, reliability: 308 passed in 51.39 seconds.
- Ruff: passed.
- Tach module boundaries: passed.
- mypy: the same 12 existing errors in five files, with identical diagnostic output to baseline. This is not a clean type check.

## CAM++ patch evidence

The unpatched 1.4.14 source and its imported `auto_model` call-site function both raise `TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'` on a speaker interval with a None timestamp. Importing the repository engine installs the existing patch at both locations; both patched calls filter the invalid interval and assign the valid interval's speaker. The patch remains necessary and is unchanged. No removal is proposed.

## Real-corpus setup

The same eight corpus/model combinations used for the MLX comparison run serially through the HTTP service on port 50710. The private run directory contains an `iic` link to the existing fixed checkpoint directories: upstream's local-path branch loads those files directly and bypasses download/version lookup. Model IDs and declared capabilities remain unchanged. All 18 recorded checkpoint/config hashes were rechecked after the final run and remain unchanged.

No acceptance or performance claim is made until complete outputs, shared MLX behavior after the NumPy downgrade, lifecycle, and checkpoint hashes have been compared with the baseline.

## Initial real results

The three completed Qwen cases (60 s English, 277.333 s English, and 3,206.752 s Chinese) have identical text and segments to the original frozen baseline despite the NumPy downgrade. Paraformer and SenseVoice return HTTP 200 on the 60 s fixture but produce different text: normalized alphanumeric output-change fractions are approximately 11.0% and 3.9%, respectively. These figures are model-to-model differences, not WER/CER and not proof of regression or improvement. Paraformer segment count changes from 21 to 20.

All 18 recorded checkpoint/config hashes remain unchanged after these cases. Logs confirm loading the existing local Paraformer, VAD, punctuation, CAM++, and SenseVoice checkpoint files. The private listening comparison for the changed text is `.runtime-validation/funasr-text-review/index.html` with the shared 60-second source audio and unchanged Qwen text as comparison material. Long-form results are recorded below; quality acceptance remains pending.

## Complete corpus comparison

| Case | Baseline / candidate elapsed | Baseline / candidate RSS (MiB) | Text comparison |
| --- | --- | --- | --- |
| paraformer-allin-long | 109.36 / 116.23 s | 6376.17 / 5266.91 | 10.35% normalized character change (not WER/CER) |
| paraformer-chinese-long | 49.11 / 77.27 s | 7606.17 / 6661.95 | 6.41% normalized character change (not WER/CER) |
| qwen-allin-long | 333.17 / 298.05 s | 4880.25 / 4670.06 | Identical text and segments |
| paraformer-english-short | 31.67 / 18.08 s | 3564.69 / 3306.19 | 11.00% normalized character change (not WER/CER) |
| qwen-chinese-long | 233.45 / 203.24 s | 4824.05 / 4608.28 | Identical text and segments |
| qwen-english-medium | 17.01 / 16.37 s | 3305.36 / 3265.05 | Identical text and segments |
| qwen-english-short | 9.85 / 5.84 s | 2976.75 / 2962.08 | Identical text and segments |
| sensevoice-english-short | 27.57 / 15.48 s | 3242.11 / 4462.53 | 3.92% normalized character change (not WER/CER) |

All four Qwen outputs are identical to the original baseline. Paraformer long English has 1,426 segments versus 1,419, and long Chinese has 1,127 versus 1,129. These are successful executions with changed recognition output, not a demonstrated quality improvement or regression. Human references are still required to judge the accepted WER/CER criterion.

The complete corpus server exited with code 0 in 0.241 seconds after idle offload. Supplemental checks preserve early unknown-model and unsupported-SRT rejections. Qwen SRT, the Qwen/aligner/Sortformer pipeline, and post-failure recovery responses are byte-for-byte identical to baseline. Paraformer SRT contains valid cues with changed transcript content. Corrupt audio still produces the existing generic 500 response, and a following valid request succeeds.

The private supplemental recorder initially mishandled SenseVoice's valid `segments: null` response and raised TypeError after saving the HTTP response. This was a harness failure, not an ASR-service failure; its server still exited with code 0 in 0.787 seconds. The recorder has been corrected, and a separate SenseVoice request returned HTTP 200 through the established corpus harness.

SenseVoice peak RSS repeated at 4,508.41 MiB, versus 4,462.53 MiB initially and 3,242.11 MiB in the original baseline. The roughly 39% increase triggers cost review but is not an automatic quality-first rejection. The Chinese long runtime increased from 49.11 / 51.97 seconds in two old-runtime runs to 77.27 / 84.62 seconds in two candidate runs. The increase persists on repeat and exceeds the 20% review trigger. RSS is not complete Metal memory measurement. These are isolated observations rather than benchmark medians.


## Recommendation before the user review

Keep FunASR 1.4.14 as an isolated candidate until transcript quality is reviewed; do not replace the accepted 1.2.7 environment on the strength of passing mocked tests. The unchanged-checkpoint upgrade is executable and preserves the tested service contracts and all four Qwen outputs, but changed FunASR recognition text and repeated cost increases prevent a complete non-regression acceptance claim. The CAM++ patch remains required and intact.

The final repeat returned HTTP 200 on SenseVoice, Paraformer short English, and Paraformer long Chinese, then exited normally with code 0 after idle offload. The service source is unchanged. The same tested dependency set is packaged separately on `codex/funasr-runtime-upgrade`; see the [delivery handoff](2026-09-07-delivery-handoff.md). The original private project copy and `.venv-funasr` remain available. No push, merge, or main-environment installation was performed.

## Independent dependency review

Standards review found no hard repository-rule violation in the FunASR-only dependency diff and independently passed `uv lock --check`. Spec review found the dependency, unchanged-checkpoint, source, and retained-patch scope consistent with the accepted roadmap. Both reviews keep recognition-quality and cost acceptance open; neither grants rollout or push approval.

## User acceptance

Lei subsequently accepted the upgrade and judged recognition text improved after listening. The recommendation above is historical. See the [acceptance and timing record](2026-09-07-listening-acceptance-and-moss-followup.md); no numeric quality score or production rollout is implied.
