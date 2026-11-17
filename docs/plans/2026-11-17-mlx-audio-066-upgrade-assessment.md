# MLX-Audio 0.6.6 branch trial

Date: 2026-11-17. Base: c06b686. Status: validation in progress.

## Owner-approved scope

The owner asked to upgrade after inspecting available local audio. The owner
explicitly waived a full baseline/candidate A/B matrix for this task, authorized
stopping the current service, and authorized serving real traffic from a feature
branch. Use existing audio when available; do not block on a new corpus.
This is a task-specific exception to SPEC-010 section 5a, not a permanent policy
change. Merge is not authorized. Hardware verified: Apple M4 Pro, 64 GiB.

Upgrade mlx-audio from 0.6.1 to 0.6.6. The published package requires
requests >=2.33.0, so update requests from 2.32.5 to 2.33.0. These are the only
package-version changes in the resolved lock. Preserve all other runtimes,
weights, aliases, API contracts, and the FunASR speaker patch.

## Source assessment

[Release 0.6.6](https://github.com/Neywa-Labs/mlx-audio/releases/tag/v0.6.6)
includes [Qwen fix 1155](https://github.com/Neywa-Labs/mlx-audio/pull/1155).
Installed wheel source comparison confirms that Qwen passes 8192 tokens to each
internal chunk instead of spending a shared file budget. The project calls
`generate_transcription` without overriding this default. The cached 1.7B 8-bit
model selects `qwen3_asr`. This establishes reachability, not measured benefit.

The wheel comparison also found:
- Qwen batched sampling normalizes logits. Our default uses batch size one.
- MOSS removes a duplicate hotword merge. The gateway supplies no hotwords.
- Subtitle export carries rounded milliseconds into the next minute correctly.
- Shared model loading adds optional checkpoint hooks and ModelOpt handling.
  The affine Qwen checkpoint has `quantization.bits`, which bypasses ModelOpt
  weight conversion. Real model loads still need validation.
- Audio writing replaces list conversion with bulk PCM conversion. Default
  audio reading remains unchanged. Optional resampling is not selected here.
- The shared sampler accepts top_k >= vocabulary size as an unfiltered case.
  Default Qwen greedy decoding does not select this path.

Evidence: local retained `.runtime-validation/mlx-audio-066-20261117/` under
`/Users/leipeng/Documents/Projects/local-asr-service`. No new model is introduced.
No new timestamp or diarization capability is claimed.

## Available audio and bounded validation

Existing project audio includes Mandarin 30/60/300/600-second samples, mixed
60/600/1320-second samples, and English short interviews and meeting recordings.
Three previously referenced local Mandarin book-club recordings are about two
hours each. These are private inputs; do not commit audio or full transcripts.
The 22-minute mixed fixture is a prior test construction, not an independent
natural long meeting. Use it as a language/chunk control only.

Run existing short Mandarin, English and mixed cases; the existing 22-minute
mixed case; and a 45-minute excerpt from an existing Mandarin source. Check
nonempty output, end coverage, errors, elapsed time and the full saved output.
Run smoke requests for all five local aliases and finish with paraformer.
Do not run a second inference process beside the live service. Use cached weights
only. Stop on a failed model load, crash, OOM or invalid response and restore the
previous service. No CER/WER or improvement claim without human reference data.
A successful candidate-only run does not establish a measured gain over 0.6.1.

Run the repository CI commands: Ruff, Tach, document links, unit tests and worker
process lifetime tests. Required independent review uses the exact committed
snapshot. Record unavailable or unrun checks explicitly.

## Trial and recovery

Candidate checkout: `/private/tmp/local-asr-066-upgrade`, branch
`codex/mlx-audio-066-upgrade`, with its own uv-managed environment.
Previous service: main checkout at c06b686, `.venv/bin/python -m src.main`.
Keep that checkout, its environment and private configuration intact. Reuse the
existing Apple Speech binary by explicit path. On failure, stop the candidate,
verify the port is free, and restart the unchanged main checkout. Verify health
and a paraformer request. Keep recovery evidence outside the disposable checkout.

Risk: reversible code/environment selection, but already consumed transcripts
cannot be undone by reverting the runtime. Radius is not low: shared dependencies
and the serial model worker can affect other aliases. Structure is touched by
the dependency lock under the owner's task-specific approval. This is a branch
trial with residual quality risk, not automatic-merge eligibility.
