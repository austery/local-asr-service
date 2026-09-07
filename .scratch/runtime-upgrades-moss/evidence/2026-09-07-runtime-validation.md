# Runtime Upgrade Validation — 2026-09-07

Status: MLX Audio 0.5.1 accepted by Lei after listening review. Technical evidence and unmeasured quality limits are preserved below; see [acceptance record](2026-09-07-listening-acceptance-and-moss-followup.md).

## Scope and isolation

Lei authorized upgrades and verification in the isolated worktree. Work is on `codex/mlx-audio-runtime-upgrade`, based on `bb501c24f86b1530254e7e08cc7ed2a478abfd12` (which includes remote main `2ff1a34db7d6999714b6df64a313c56e484bc171`). The original main checkout and its environment remain unchanged. Baseline and candidate dependencies are installed separately in `.venv-baseline` and `.venv-mlx`. Raw logs, complete transcripts, checkpoint records, and media are gitignored under `.runtime-validation/`.

## Reproducible environment

- macOS 26.6.2, arm64, hardware identifier Mac16,11, 64 GiB unified memory, Python 3.11.14.
- Original lock SHA-256: `2fb29a6c4391a5931266cacea563663ad01ab60416ca8677a64f06d6458b7298`.
- Qwen3-ASR checkpoint: `mlx-community/Qwen3-ASR-1.7B-8bit`, revision `a8379a2e2f9e313c9292cdf1af4055ab56d50d55`; Hugging Face access is offline during inference.
- Both clean project environments installed successfully with `uv sync --frozen --dev --prerelease=allow`, using the appropriate lock and separate `UV_PROJECT_ENVIRONMENT`.
- Candidate changes: MLX Audio 0.4.3 → 0.5.1; Transformers 5.8.1 → 5.16.1; Tokenizers 0.22.1 → 0.23.2; Safetensors 0.7.0 → 0.8.0. The upstream dependency on `mlx-lm` is removed; this repository has no direct `mlx_lm` imports.
- FunASR remains 1.2.7; MLX remains 0.31.2; NumPy remains 2.3.5; Torch/Torchaudio remain 2.12.0/2.11.0. Real imports of both engine runtimes pass in both environments.

## Automated checks

| Check | Baseline | MLX Audio 0.5.1 candidate |
| --- | --- | --- |
| Unit, integration, reliability tests | 308 passed in 41.77 s | 308 passed in 47.48 s |
| Ruff | Passed | Passed |
| Tach module boundaries | Passed | Passed |
| mypy | 12 existing errors in five files | Identical 12 errors; no new diagnostics |

Commands: `uv run --frozen --no-sync python -m pytest tests/unit tests/integration tests/reliability`, `ruff check .`, `tach check`, and `mypy src/`, with the respective environment selected. The existing mypy errors are recorded, not waived as a clean type-check result. These tests do not establish real-model recognition quality.

## Corpus and evidence limits

The current corpus contains the existing 60 s English dialogue, a 277.333 s English conversation, the previously used 3,206.751938 s Chinese recording, and a new exact 3,600 s All-In excerpt approved as a source by Lei. File hashes and local paths are recorded in the private baseline manifest.

The All-In excerpt is from the official channel's [Dario Defends Himself, Datacenter Panic, AI Doomer Trap, Senate Toss-Up](https://www.youtube.com/watch?v=Sij_v-mcZXQ), published 2026-08-21; source interval 00:05:00–01:05:00. Mono 16 kHz PCM WAV SHA-256: `94b028dbff99599edd6987c2e3c3986f63e4186099e5d1f45b74404ed71d30d7`. The official [episode index](https://allin.com/episodes) supplies the source link. Native audio download followed by local extraction succeeded; a slower superseded remote-section download was stopped.

Human-corrected reference windows are not yet available. Do not report WER/CER, speaker attribution accuracy, or broad quality acceptance from raw output alone. Mixed-language and noisy/overlap coverage still need explicit assessment.

## Real-model runtime comparison

The isolated HTTP service uses port 50710 with one inference worker, full response capture, and process-tree RSS sampling. RSS is not a complete measurement of Metal/unified-memory use. The original service port is not used.

| Case | Baseline time | Candidate time | Baseline / candidate peak RSS (MiB) | Complete output comparison |
| --- | --- | --- | --- | --- |
| paraformer-allin-long | 109.36 s | 110.39 s | 6376.17 / 6073.70 | Identical text and timing; speaker differences below |
| paraformer-chinese-long | 49.11 s | 48.86 s | 7606.17 / 7748.33 | Identical text and timing; speaker differences below |
| qwen-allin-long | 333.17 s | 305.80 s | 4880.25 / 4647.97 | Identical text and segments |
| paraformer-english-short | 31.67 s | 29.90 s | 3564.69 / 4212.77 | Identical text and segments |
| qwen-chinese-long | 233.45 s | 208.32 s | 4824.05 / 4613.41 | Identical text and segments |
| qwen-english-medium | 17.01 s | 15.02 s | 3305.36 / 3265.84 | Identical text and segments |
| qwen-english-short | 9.85 s | 6.64 s | 2976.75 / 2962.92 | Identical text and segments |
| sensevoice-english-short | 27.57 s | 24.01 s | 3242.11 / 3129.77 | Identical text and segments |

These are recorded runs, not stable performance medians. First-use cache preparation and lightweight background work affect wall time. None exceeds the initial 20% time or 25% RSS investigation trigger. Exact text equivalence demonstrates no transcription-text regression on these files; it does not establish absolute WER/CER or untested-language quality.

Paraformer has identical text, segment counts, and every timestamp. The English long file changes three speaker assignments among 1,419 segments (2.16 seconds of speech); the Chinese long file changes one among 1,129 segments (4.49 seconds). These are not explained by a global label rename. Baseline repeatability must be checked before attributing them to the upgrade or calling them harmless.

All Qwen outputs have identical segments. Long Chinese ends at 3,236.768 seconds for 3,206.752 seconds of audio; All-In ends at 3,629.952 seconds for 3,600 seconds of audio. These approximately 30-second timeline overruns exist before the upgrade. Preserve them as existing chunk/timeline defects. Paraformer's three predicted labels on the short fixture also do not establish a ground-truth speaker count.

The baseline servers exited normally with code 0 in 0.233 / 0.239 seconds. The candidate exited normally with code 0 in 0.233 seconds. Logs record model release on switches and worker exit after the idle timeout, followed by normal application shutdown. Process-tree RSS is not total Metal/unified-memory usage.

## Lockfile scope and reproducibility

Candidate lock SHA-256: `e04a6f4e59d950b58b2dac8e239162b04ec5eb0c833f0a90b3b38d5369cf7bb5`. The final lock retains the baseline `if-necessary` prerelease policy. Runtime source under `src/` is unchanged from the accepted `bb501c2` baseline.

The CUDA optional-dependency marker normalization was reproduced in a separate copy of the original project and lock using uv 0.12.10: `uv lock --upgrade-package 'mlx-audio==0.4.3' --prerelease=if-necessary`. Every package version stayed unchanged, and the CUDA entries became identical to the candidate lock. The only incoming CUDA-toolkit edge remains Torch's `sys_platform == 'linux'` edge. This is resolver normalization, not a new macOS CUDA dependency. An initial offline attempt lacked cached mlx-lm metadata; the subsequent metadata-enabled control succeeded. No hand-edited lock entries were used.

## Independent review

- Standards: no hard repository-standard violations in the dependency-only diff. The reviewer independently checked a resolvable dependency tree and `uv lock --check`.
- Spec: acceptance remains partial while full evidence, real contract checks, baseline rollback, and quality review are incomplete. The CUDA drift concern is now explained by the controlled baseline re-resolution above. Do not turn this review into a completed acceptance claim.

## Acceptance work before the user review

- Supplemental contract and baseline-environment rollback checks completed; results below.
- Baseline speaker repeat completed. One of four differences also occurs between baseline runs; the remaining three are not explained by that repeat and remain an attribution-review limitation.
- Keep unchanged timestamp/subtitle defects and existing mypy failures separate from upgrade regressions.
- Record the technical runtime verdict with explicit quality/evidence limits. MOSS admission and the later FunASR upgrade remain separate decisions.

## Supplemental contracts and rollback

Both environments were restarted after the candidate's eight-case run, first `.venv-baseline` and then `.venv-mlx`, on the same unchanged `bb501c2` runtime source and frozen checkpoint caches. This demonstrated reverting the executing runtime to the previous locked environment, not merely reverting a source declaration. The original main service was not involved. All 18 recorded FunASR checkpoint/config file hashes remain unchanged after the tests.

| Check | Baseline | Candidate | Interpretation |
| --- | --- | --- | --- |
| Unknown model | 400 | 400 | Identical early rejection |
| SenseVoice SRT | 400 | 400 | Identical unsupported-capability rejection |
| Qwen SRT | 200, plain transcript | Identical response | Existing defect: no SRT cue formatting |
| Paraformer SRT | 200, SRT cues | Identical response | Subtitle behavior preserved |
| Qwen + forced aligner + Sortformer | 200, four segments / two labels | Identical response | Existing opt-in pipeline still works on 60 s fixture |
| Corrupt WAV | 500 | 500 | Existing generic error behavior; different request IDs expected |
| Normal Qwen request after failure | 200 | Identical response | Worker recovers without service restart |

These checks cover the exercised routes and format behavior. They do not certify every model/format combination or the experimental pipeline's long-form quality. Both servers subsequently exited with code 0. Raw requests/responses and shutdown timing remain in the private evidence directory.

## Baseline speaker repeat

A second run in the unchanged `.venv-baseline` returned identical text and timestamps on both long files. English changes one label relative to the first baseline (segment 1227, 3,160.19–3,160.53 s), matching the candidate there. Chinese is identical across the two baseline runs. All repeated requests returned HTTP 200 and the server exited with code 0.

The remaining baseline/candidate disagreements are English 698.14–699.72 s and 973.99–974.23 s, and Chinese 2,628.95–2,633.44 s. This single repeat establishes some baseline variability, not that every difference is random or benign. Text compatibility and public contract equivalence are demonstrated on the tested corpus; full speaker-quality acceptance is still limited by these unreviewed intervals. No new timestamp, text, lifecycle, or formatting regression was observed.

The private listening comparison for all four originally differing intervals is `.runtime-validation/mlx-speaker-review/index.html`. The first three clips cover the unresolved differences; the fourth illustrates observed baseline variability. No speaker-accuracy score has been inferred from model-to-model agreement.
