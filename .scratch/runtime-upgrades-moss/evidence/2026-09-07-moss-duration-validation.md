# MOSS Duration Validation — 2026-09-07

Status: Completed raw upstream duration probes. User-recognized text and speaker quality is retained; the duration evidence below defines the next bounded integration scope. No MOSS HTTP adapter, production alias, library patch, or main-environment change is included in this experiment.

## Controlled setup

- Same MLX Audio 0.5.1 environment and BF16 checkpoint revision `704aa4a9c304e8520be88901e0d1960158ef5b15`; unchanged default English prompt, greedy decoding, 32,768 output tokens, and 4,096 prefill step size.
- Same original All-In excerpt, source 05:00–65:00 in the episode; original WAV SHA-256 `94b028dbff99599edd6987c2e3c3986f63e4186099e5d1f45b74404ed71d30d7` was rechecked. Prefix inputs cover its first 20, 30, and 40 minutes. The disjoint 30-minute input covers its final half, including the original failed run's missing tail.
- Input boundaries, fixed first/middle/last review windows, and the internal-gap criterion were recorded before these runs. Only one owned model process ran at a time. Each process exited before the next model process started.
- The private recorder calls the same upstream preparation, token generator, sampler, EOS rule, and parser without changing model/library code. It adds termination-token observation. Its 60-second control produced exactly the same full response and 380 generated tokens as the original probe.
- Local artifacts: `.runtime-validation/moss-duration-plan.json`, `moss_duration_probe.py`, `moss-duration/*/{metrics,response,token_ids}.json`, `moss-duration-coverage-analysis.json`, and `moss-duration-source-verification.json`. Private media and full responses remain ignored by Git.

## Recorded results

| Input | Duration | Inference | Output tokens | Last parsed end | MLX peak | Coverage observation |
| --- | --- | --- | --- | --- | --- | --- |
| allin-first-20m | 20 min | 200.71 s | 10,229 | 19:59.94 | 4.91 GiB | No large coverage gap detected |
| allin-first-30m | 30 min | 316.10 s | 13,551 | 29:59.91 | 6.16 GiB | No large coverage gap detected |
| allin-first-40m | 40 min | 465.94 s | 16,376 | 35:12.52 | 7.48 GiB | Incomplete |
| allin-last-30m | 30 min | 357.33 s | 15,337 | 29:59.84 | 6.37 GiB | No large coverage gap detected |
| Original whole excerpt | 60 min | 715.80 s | 16,376 | 36:19.08 | 9.26 GiB | Incomplete; missing final 23:40.92 |

Inference includes input preparation and generation; model loading is additional. These are individual runs on this Mac, not stable benchmark medians. MLX peak is an allocator measurement, not whole-machine memory or RSS.

## What the stop trace establishes

The 40-minute probe emitted token **151645 (`<|im_end|>`) after 16,376 output tokens**, with its 32,768 budget still available. The selected EOS was observed directly in greedy generation. The earlier 60-minute run also produced 16,376 tokens but ended at a different audio position. The 40-minute raw text stops in an unfinished speaker marker; the 60-minute raw text stops mid-utterance. These observations establish premature decoder termination rather than a fixed audio-minute boundary or parser-only loss.

The 40-minute final segment ends at 2,112.52 seconds, leaving 287.48 seconds; the baseline marks 274.91 seconds of speech in that missing interval. Our source-path inspection already found no secondary 16K cap in the count-bounded MLX generator. The exact learning/training reason for EOS remains unproven. Raw exponentiated BF16 log-probability values in private metrics are approximate and are not calibrated confidence scores; the conclusion relies on the selected token, count, and missing speech.

The successful short/20/30-minute runs also use the normal EOS token. EOS or exit code 0 alone is therefore not a completeness certificate. These probes do not establish raising an unused output budget as a remedy. The [official limits research](2026-09-07-moss-upstream-limits.md) documents related upstream reports and the publisher's distinct 90-minute claim.

## Coverage and speaker checks

The analysis checks finite/in-bounds ordered segment times, raw final timestamp closure, trailing gaps, and uncovered intervals containing at least 10 seconds of speech according to the frozen Paraformer baseline. Fixed first/middle/last windows provide content comparison and local audio for listening. Another model is corroborating evidence, not ground truth: these checks do not establish word-perfect recall, WER/CER, or speaker accuracy.

Across the shared first 20 minutes of the completed 20- and 30-minute probes, labels agree on 99.65% of their pairwise overlapping segment duration after at most one global label rename. All four labels map to themselves. This is cross-run consistency, not ground-truth attribution accuracy or a production cross-request identity solution. Independent inputs retain independent speaker namespaces.

See the private [duration listening page](/Users/leipeng/Documents/Projects/local-asr-service-worktrees/runtime-roadmap/.runtime-validation/moss-duration-review/index.html), including the fixed tail clips. The user's positive listening assessment remains recorded in the [acceptance record](2026-09-07-listening-acceptance-and-moss-followup.md).

## Bounded next step

Use 30 minutes as a provisional upper bound for the next explicitly experimental English single-recording adapter. This is an operating policy supported by the two disjoint 30-minute samples, not the model's universal maximum or proof that every 30-minute file will succeed.

The next thin adapter should preserve a whole recording in one model call, declare the tested scope explicitly, bound output tokens, normalize speaker/text fields, and reject obvious incomplete or malformed output. It must not silently split an oversized recording and imply that speaker IDs remain consistent across calls. Existing small raw endpoint overruns, such as the earlier 277-second sample, need explicit bounded normalization or rejection before public timestamp claims.

The original goal of unrestricted 30–60-minute English multi-speaker use is not fully met: the 40- and 60-minute inputs fail. A 30-minute opt-in operating range and completeness rejection are a bounded adoption proposal, not completion of that wider claim. Other recordings, high speech density, languages, overlap, silence, public API behavior, and failure/lifecycle paths still require checks appropriate to the eventual adapter scope. No existing model role or experimental pipeline is retired by these results.

## Artifact and process verification

All four duration-probe processes exited with code 0; the separate control also exited with code 0. The 12 listening clips were checked as exactly 60 seconds of mono 16 kHz PCM audio, and local report/review links resolve. No owned duration-probe process remained after completion. Main stays at `97028ba0a3b8c5774903ace18cb7fb29456d9bb5`; the MLX dependency lock remains unchanged.
