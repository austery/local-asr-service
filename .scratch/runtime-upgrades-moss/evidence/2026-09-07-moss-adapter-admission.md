# MOSS adapter admission and delivery verification — 2026-09-07

Status: **Go for bounded, explicit English continuous-speech recordings up to 30 minutes.** Lei reviewed the duration listening results, accepted text and speaker quality, and requested introduction in a separate PR for another agent to review. This is local implementation/admission evidence; publication, CI, merge, and production rollout are separate delivery states.

## Frozen candidate

- Based on FunASR feature merge `0ff0a22bbc032a481e38ee4e6f70c3eae95577a6`, which includes the accepted MLX branch. Lock SHA-256 remains `2c3ced2b2d2b9ffba961e0c3b0d723442077d96bcd8c1f457c4112bd030f6b4b`.
- Python 3.11.14, mlx-audio 0.5.1, FunASR 1.4.14, NumPy 1.26.4; fresh frozen worktree environment. MOSS uses the official BF16 checkpoint at pinned revision `704aa4a9c304e8520be88901e0d1960158ef5b15`.
- [Contract](../../../docs/plans/2026-09-07-moss-adapter-contract.md): one upstream call, 32,768 output tokens, 4,096 prefill step, greedy generation, explicit English, 1,800-second input bound, 900-second isolated-worker OS deadline. No automatic splitting, speaker reconciliation, model conversion, or new engine.
- The first offline HTTP attempt exposed an absent mutable `main` cache reference. Pinning the already evaluated checkpoint through upstream's supported `revision` parameter fixed loading. No global cache reference or model weight was modified.

## Automated checks and independent review

| Delivery tree | Unit / integration / reliability | Real E2E | Ruff / Tach | Package build | mypy |
| --- | --- | --- | --- | --- | --- |
| MLX `e0f602b` | 308 passed | 1 passed | Passed | sdist + wheel passed | 12 existing diagnostics in 5 files |
| FunASR `0ff0a22` | 308 passed | 1 passed | Passed | sdist + wheel passed | Existing baseline diagnostics; no source changes |
| MOSS candidate | 353 passed | 1 passed | Passed | sdist + wheel passed | 10 remaining existing diagnostics in 4 files |

The E2E test uses real Paraformer and existing cached ASR/VAD/punctuation/speaker model files, sequentially across the three trees. Its existing unregistered `e2e` marker emits one warning. The MOSS tests stub inference in unit tests and cover alias/capability identity, the pinned load, duration/language rejection, output normalization, raw truncation, token exhaustion, malformed timestamps/speakers, conservative gaps, endpoint clamping, overlap preservation, text/SRT, temporary-file cleanup, OS deadline termination/cancellation, HTTP error mapping, and subsequent-call recovery. Explicit result types on the touched worker remove two prior type errors; this is not a clean repository-wide mypy result.

Separate Standards and Spec reviews found no actionable implementation defect. Both rechecked checkpoint pinning and parent-owned artifact cleanup. Their reviewed implementation snapshot SHA-256 is `63be1803b48edc87657452dbcfec14a84a6c19a2c400397dfa8aa143931b6f07`; later changes only finish test annotations/import spacing and delivery documentation. The user's requested external PR review remains pending.

## Real HTTP results

A separate single-worker service ran on `127.0.0.1:50712`. It exited before any next owned real-model test began. Private responses and logs are in the MOSS worktree's `.runtime-validation/api-probe-v2/`.

| Request | Result | Observation |
| --- | --- | --- |
| 60-second English conversation | 200, 6.77 s including initial load | 12 segments, 2 speakers; end 59.94 s |
| 277.333-second English MP3 | 200, 21.28 s | 48 segments, 2 speakers; small raw endpoint overrun clamped to 277.333 s |
| First 30 minutes of the approved All-In excerpt | 200, 365.60 s | 352 segments, 4 speakers; end 1,799.91 s; duration 1,800 s |
| 40-minute input | 400, 0.20 s | Rejected before inference; no automatic split |
| `language=auto` | 400, 0.05 s | Explicit English required |
| Corrupt WAV, then valid short recording | 500, then 200 | Existing generic decode failure does not prevent the next transcription |
| Short SRT / timestamped text | 200 / 200 | Speaker labels and timestamp formatting preserved |
| Idle offload, then short request | No worker after idle; next request 200, 5.84 s | Fresh worker reload |
| OS termination during an active request, then short request | 500, then 200 in 5.78 s | Existing parent liveness handling fails the active job and reloads on the next request |
| MOSS → Qwen → MOSS | Both 200 | Release precedes replacement worker load |
| Graceful service shutdown | Exit 0 in 0.67 s | No resource-tracker warning in the probe log |

The normalized public 30-minute **text and every segment exactly match** the previously accepted raw 30-minute result after applying the same normalization; the API adds segment IDs. This is reproducibility, not independent accuracy measurement. HTTP timings are single observations on the evaluation Mac, with CPU-only delivery checks allowed concurrently; they are not controlled benchmark medians.

## Failure and artifact checks

A separate cleanup probe set a private `TMPDIR`, then verified: short request 200; the short recording with an added 15-second silent tail rejected as 422; real worker `SIGALRM` termination returned 500; next request returned 200. Before termination the request directory contained the original audio and nested MOSS output directory; afterward **zero request directories remained**. Shutdown exited 0. This exercises the same OS termination mechanism used by the deadline. The actual 900-second interval was not waited out; a subprocess test shortened the deadline and verified OS termination and cancellation after success.

The new output validator also accepts the recorded successful 20-minute and both disjoint 30-minute results and rejects the recorded truncated 40-minute output. Raw final-marker checks and conservative coverage checks reject detectable failures; they cannot certify all words are present. A 10-second gap threshold deliberately rejects some legitimate silence, as the HTTP silent-tail probe demonstrates.

## Limits and delivery

Thirty minutes is a provisional service policy, not MOSS's universal maximum. The earlier 40/60-minute failures near 16K output tokens remain valid; unrestricted 30–60-minute use is deferred. Numeric WER/CER and speaker accuracy, other languages, heavy overlap/noise, and cross-recording speaker identities remain unmeasured or unsupported by this admission.

Retirement list: **empty**. Keep all current aliases, the FunASR CAM++ patch, and the opt-in Qwen/Sortformer pipeline. No caller migration or production deployment is implied. Original main remains at `97028ba0a3b8c5774903ace18cb7fb29456d9bb5`; its existing files/environment are unchanged and its port 50700 health endpoint remains healthy.

Delivery order: MLX baseline PR → incremental FunASR PR → MOSS PR. Preserve branch ancestry when landing the stack or reconcile subsequent PR bases explicitly after squash; do not update the running local main checkout. Show the final diffs and obtain explicit push approval. The first two have user quality/merge acceptance; the MOSS PR waits for the user's separate reviewing agent.
