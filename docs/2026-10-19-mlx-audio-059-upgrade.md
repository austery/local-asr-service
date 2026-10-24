# MLX Audio 0.5.9 upgrade preparation

Status: prepared for PR review; activation remains owner-controlled.

## Change and activation boundary

Raise the mlx-audio dependency floor from 0.5.1 to 0.5.9 and lock 0.5.9.
All other locked package versions stay unchanged relative to the PR base.
Keep existing model aliases, cached weights, adapter parameters, and output
validation. No model download, source migration, or Nemotron integration is
required by this change.

Creating or merging this PR does not upgrade the already-running service.
The owner will choose a maintenance window to update its checkout/environment
and restart it through the existing operator workflow. Do not synchronize the
running environment or restart its worker as part of preparing this PR.

## Isolated comparison completed before this PR

Source snapshot: `d0f29bf4fe4adff7216918facc1c58e9e4cf5681`.
The PR starts from newer main `9185a592fc470e71561b88d92a2c5685ee051570`;
the intervening change updates GitPython only. Distinguish the earlier model
comparison from validation of the final PR tree.

The A/B environments differed only in mlx-audio 0.5.1 versus 0.5.9, with MLX
0.31.2 and Transformers 5.16.1 fixed. Both used existing cached Qwen3-ASR 1.7B
8-bit and pinned MOSS weights, offline. Three English recordings (60 seconds,
277.344 seconds, and 1,193.51925 seconds) were tested twice per version/model,
with warmup and reversed version order: 24 timed adapter calls in total.

| Model | Recording | 0.5.1 median | 0.5.9 median | Result |
| --- | --- | ---: | ---: | --- |
| Qwen | 60 s | 2.416 s | 2.434 s | Identical text and segments |
| Qwen | 277.344 s | 15.402 s | 15.742 s | Identical text and segments |
| Qwen | 1,193.51925 s | 101.222 s | 101.898 s | Identical text and segments |
| MOSS | 60 s | 3.171 s | 3.227 s | Identical text and segments |
| MOSS | 277.344 s | 21.364 s | 21.337 s | Identical text and segments |
| MOSS | 1,193.51925 s | 188.862 s | 188.175 s | Same rejected output |

MLX peak allocation was effectively identical. These are exploratory desktop
wall-clock observations, not a controlled speedup claim. Both earlier A/B
non-E2E suites passed 324 tests. Full private evidence, hashes, scripts, and raw
outputs are retained locally at
`.runtime-validation/mlx-audio-ab-20261020T002125Z/REPORT.md` in the primary
checkout; these private artifacts are not included in Git.

## Compatibility findings and remaining limits

- The upstream 0.5.9 1,024-token fix applies to `mlx_audio.server`. Our adapter
  directly invokes the Python transcription function and does not use that
  server. The Qwen main inference file and transcription entry point were
  byte-identical between the two installed releases.
- The MOSS loading fix preserves already-converted convolution layouts. Our
  pinned PyTorch checkpoint takes the same conversion branch in both versions.
- MOSS still generated Spanish for the long English recording under the
  unchanged default prompt. Its 275 segments ended at 1,182.11 seconds, leaving
  an 11.40925-second tail; the existing validator rejected every run with
  `TranscriptionOutputError`. This upgrade does not fix language drift or
  change the conservative coverage policy. The separate English-prompt PR
  was not included in this comparison or this upgrade.
- No human-reference WER/CER or speaker accuracy was measured. Chinese,
  streaming, forced alignment, 60-minute inputs, and new checkpoint formats
  were not evaluated. Compatibility is bounded to the tested paths.

## Owner-controlled activation checks

At the owner's chosen maintenance window, install the reviewed lockfile in the
service environment and use the existing single-worker startup workflow.
Verify health, model discovery, and short Qwen/MOSS requests before accepting
activation. Keep the known MOSS long-recording rejection visible. Rollback is
also an owner-controlled environment synchronization and restart using the
previous dependency manifest and lock; this PR makes no data/schema migration.

## Final PR-tree verification

The upgrade tree passed Ruff, Tach boundaries, sdist/wheel builds, and the full
325-test suite, including the real FunASR E2E, with one existing unregistered
`e2e` marker warning. Network access was blocked for the test process. An initial
attempt failed during offline ModelScope discovery; task-local directory aliases
to the existing ASR/VAD/punctuation/speaker cache resolved that test-environment
issue. The aliases changed neither runtime source nor shared cache files and
were removed after validation. No additional model weights were downloaded.
