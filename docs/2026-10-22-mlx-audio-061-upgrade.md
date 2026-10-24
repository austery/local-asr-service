# MLX Audio 0.6.1 and Qwen automatic-language upgrade

Status: prepared for PR review; activation remains owner-controlled.

## Change and activation boundary

Raise the mlx-audio floor to 0.6.1 and lock that release. Its supported stack
also updates MLX/MLX-Metal from 0.31.2 to 0.32.4 and miniaudio from 1.61 to
1.71. Other resolved package versions remain fixed relative to the PR base.
No model weights, aliases, worker counts, token budgets, or chunk durations
change. Phonon and Nemotron are outside scope.

For Qwen only, translate the public `auto` language to Python `None` at the
upstream inference boundary. The literal `auto` string forces an invalid
`language auto<asr_text>` prefix; it does not request automatic detection.
Explicit aliases such as `en` and `zh` still map to English and Chinese.
Other MLX models keep their existing language arguments. The public API
continues to accept strings and defaults to `auto`; it does not gain a new
nullable form field or a different response schema.

Version 0.5.9 reuses the first detected chunk language for later chunks. If
that first chunk is silence, the detected no-speech language can suppress
all later speech. Version 0.6.1 detects each chunk independently when its
language argument is None. Upgrade the runtime and mapping together: changing
only the mapping on 0.5.9 exposes the reproduced empty-output failure.

PR creation or merge does not activate this change. Leave the running checkout,
environment and service process untouched. The owner will update dependencies
and restart through the existing single-worker workflow at a chosen time.
Rollback must restore both the previous adapter and dependency lock together.

## Completed isolated investigation

Source: `57a1fe35a2f505afa69eeac924295f68f26a448e`. Both runtimes used the
same existing Qwen3-ASR-1.7B-8bit checkpoint (revision
`a8379a2e2f9e313c9292cdf1af4055ab56d50d55`), identical audio, offline
loading, default 1,200-second chunking, batch size 1 and 8,192-token budget.

The constructed bilingual sample joins 11 minutes of existing Chinese speech
to 11 minutes of existing English speech; another adds 30 seconds of silence.
A boundary diagnostic puts 20 minutes of digital silence before two minutes
of bilingual speech. These are controlled concatenations, not natural
bilingual-podcast acceptance recordings.

Across 38 model calls:

- With None, the 22-minute sample changes from Chinese/Chinese chunk labels
  in 0.5.9 to Chinese/English in 0.6.1. Both versions retain later text on
  the ordinary and 30-second-leading-silence long samples.
- With an entirely silent first chunk and None, 0.5.9 returns no speech text;
  0.6.1 returns 1,008 characters of subsequent speech text. The model call
  returns successfully in both releases; this was not an HTTP endpoint test.
- A shorter diagnostic with explicit 30-second chunks reproduces the same
  problem: zero speech characters in 0.5.9, 415 in 0.6.1. This override is
  separate from the current adapter's default 20-minute chunk duration.
- Literal auto and explicit language calls do not demonstrate that automatic
  detection repair. Both releases can still hallucinate text on silence under
  forced English. No WER/CER, general speedup, sentence alignment accuracy,
  or MOSS quality improvement is claimed.

The pre-change adapter and 324 non-E2E tests passed in each runtime. The
private corpus, hashes, raw outputs, source snapshots, and exact locks remain
in `.runtime-validation/mlx-audio-061-20261022T120454Z/REPORT.md` in the
primary checkout. Private recordings and transcripts are not committed.

## Regression and final-tree validation

Unit tests check Qwen automatic detection, whitespace/case normalization,
explicit aliases, unchanged non-Qwen arguments, and preservation of speech
following an empty leading segment. These use mocked inference.

`tests/e2e/test_qwen_auto_language.py` uses the real unchanged adapter and
existing Qwen weights. It prepends 1,200 seconds of silence to the documented
local 60-second English fixture, then checks recovered early/late speech
anchors and the independently detected English language after the silent
chunk. It uses default internal chunking and does not patch upstream code.
Run the real test only on Apple Silicon with the cached model and
`tests/fixtures/two_speakers_60s.wav` available. CI's unit suite remains mocked.
The full local suite also exercises the existing real FunASR flow.

The final tree passed all 329 tests, including both real-model E2Es, with
external networking denied. Ruff, Tach boundaries and sdist/wheel builds also
passed. The new E2E failed before the mapping fix because the detected language
remained the literal auto string. The exact published head is recorded in the PR. Task-local logs
remain in `.runtime-validation/mlx-audio-061-pr/` in the primary checkout.
The running service is not an acceptance target for this PR preparation.

## Owner-controlled activation checks

At activation, synchronize the reviewed manifest/lock and adapter together.
Check health and model discovery, then omitted/auto and explicit-language
Qwen requests, including silence-leading bilingual audio. Retain conservative
MOSS validation and its known language/coverage limits. Do not change the
single-worker configuration. Full-duration chunk boundaries alone do not
prove semantic transcript completeness.
