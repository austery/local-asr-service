# MLX-Audio 0.6.6 branch trial

Date: 2026-11-17. Base: c06b686. Status: candidate checks passed; live trial acceptance remains open.

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
only. Stop on a failed model load, crash, OOM, invalid response, or observed material
repetition or omitted speech and restore the
previous service. No CER/WER or improvement claim without human reference data.
A successful candidate-only run does not establish a measured gain over 0.6.1.

Run the repository CI commands: Ruff, Tach, document links, unit tests and worker
process lifetime tests. Required independent review uses the exact committed
snapshot. Record unavailable or unrun checks explicitly.

## Trial and recovery

Candidate checkout: `/Users/leipeng/Documents/Projects/local-asr-service/.worktrees/mlx-audio-066-upgrade`, branch
`codex/mlx-audio-066-upgrade`, with its own uv-managed environment.
Previous service: main checkout at c06b686, `.venv/bin/python -m src.main`.
Keep that checkout, its environment and private configuration intact. Reuse the
existing Apple Speech binary by explicit path. On failure, stop the candidate,
verify the port is free, and restart the unchanged main checkout. Verify health
and a paraformer request. Keep recovery evidence outside the disposable checkout.

Risk: One-way live effects, with reversible code/environment selection. Already consumed transcripts
cannot be undone by reverting the runtime. Radius is not low: shared dependencies
and the serial model worker can affect other aliases. Structure is touched by
the dependency lock under the owner's task-specific approval. This is a branch
trial with residual quality risk, not automatic-merge eligibility.

There is no automatic semantic-quality monitor for live traffic. Reverting the
runtime does not rewrite PureSubs results. A tail-only transcript is a cross-check,
not a human reference or proof that all speech was preserved.

## Candidate validation observed

All nine serial HTTP requests returned 200 and nonempty text.

| Case | Elapsed seconds | Output characters | Segments |
| --- | ---: | ---: | ---: |
| qwen-zh30 | 4.724 | 135 | 1 |
| qwen-en30 | 0.952 | 312 | 1 |
| qwen-mixed60 | 1.997 | 446 | 1 |
| qwen-mixed22min | 100.729 | 13433 | 2 |
| qwen-zh45min | 199.024 | 14747 | 3 |
| moss-en60 | 7.079 | 760 | 12 |
| sensevoice-zh30 | 25.527 | 129 | 0 |
| apple-en30 | 0.568 | 318 | 7 |
| paraformer-zh60 | 34.705 | 269 | 18 |

The 22-minute mixed output has two segments ending at 1320 seconds. The 45-minute
Mandarin output has three segments ending at 2700 seconds. A separate final
30-second transcription shares an 88-character contiguous normalized match with
the long output. This is a same-model consistency check, not independent accuracy
measurement. Qwen HTTP duration remains 0.0 on this existing adapter path; segment
end times and source duration were used instead. No production source was changed.
No full human audit, CER/WER, baseline A/B, peak-memory benchmark or long-running
traffic acceptance was performed. Live quality is still an owner-observed trial.

Ruff, Tach, document links and 548 unit/worker-process tests passed in the original
candidate environment. The old service and environment are retained. Independent
Claude Opus high reviews covered d3fe58a: Spec passed; Standards found documentation
follow-ups; Risk stopped automatic merge. The final document corrections accept
One-way live effects and retain the explicit owner-authorized trial. Evidence is
ignored by the existing `.runtime-validation/.gitignore`; the review's accidental
`git add -A` concern does not apply to that directory. Destructive `git clean -fdx`
would still delete ignored evidence and must not be used.

Open acceptance: observe real workloads on the branch. Merge needs an owner
decision. Keep this plan active until that acceptance closes, then archive it
under the repository document lifecycle. The prior SPEC-010 history remains as
historical context; a dated row records this task-specific owner exception.

## Recovery and live activation observed

On 2026-11-17, the validation instance exited before the old main-checkout service
restarted on port 50700. Health and a 60-second Paraformer request passed with
269 output characters. That service then stopped normally. This verifies the
retained old environment restart path; it is not a clean rebuild of the old lock.

The candidate worktree moved to the durable path above. A fresh `uv sync --frozen
--dev` installed the same lock there. Ruff, Tach, links and all 548 tests passed
again (58.86 seconds for pytest). The final service started on port 50700 with
one worker. Qwen's 30-second Mandarin request and Paraformer's 60-second request
passed. `/v1/models/current` then reported paraformer and queue_size 0.

Live process record: `trial-process.json`; live log: `trial.log`; recovery run:
`rollback-process.json`, `rollback-smoke.json`, `rollback.log`, all under the
retained evidence directory. The launcher is `launch_service.py` in that directory.
It uses the selected checkout's `.venv/bin/python -m src.main`, the existing
private configuration via symlink, cached Hugging Face weights, and the existing
Apple Speech binary by absolute path. This is a detached process, not a newly
installed login/reboot service. Do not infer automatic restart after a reboot.

For recovery, first identify and gracefully stop the current listener after
in-flight work drains. Confirm its model process and listening socket have
exited. From the main checkout, run `.venv/bin/python -m src.main`, then verify
health and paraformer. Do not run `uv sync` in the retained main checkout during
recovery. Do not alter or remove `.venv.rollback-b439f4f`, which belongs to an
older task. No stored PureSubs transcripts are changed by this recovery.
