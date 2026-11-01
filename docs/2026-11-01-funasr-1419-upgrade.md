# FunASR 1.4.19 upgrade evidence

Date: 2026-11-01. Baseline: `8d44175cb06fe0e09bab4d40999b92e74eaa84cb`.

## Scope

Raise `funasr[llm]` from `>=1.4.14` to `>=1.4.19` and lock 1.4.19.
FunASR is the only package version changed. Its new direct `threadpoolctl`
dependency reuses the already locked version. Keep model weights, adapters,
request contracts, and both `distribute_spk` patch bindings unchanged.
This prepares an upgrade PR. It does not activate the running service.

The owner accepted the bounded two-clip comparison as the basis for this PR.
The unrun checks below remain explicit limits; this is not full long-recording
or production acceptance.

## Isolated real-model comparison

Two Python 3.11.14 environments used the same cached SEACO Paraformer,
FSMN-VAD, CT punctuation, and CAM++ weight files. Recorded SHA-256 inventories
match across all runs. Installed package versions differ only in FunASR.
Torch 2.12.0 and Torchaudio 2.11.0 stay unchanged.

Each version processed the same private Mandarin 60-second and 300-second
WAV files twice in fresh, sequential processes with seed 0. ASR, VAD,
punctuation, and speaker parameters were observed on `mps:0` in both repeat
runs. Every run required MPS and asserted both project speaker-patch bindings.

The harness constructs AutoModel with explicit local cache paths, then calls
the unchanged project `FunASREngine.transcribe_file` and JSON formatter.
It uses the existing 30-second VAD limit, 60-second batch setting, VAD merge
settings, and ITN default. This is engine-level acceptance, not live HTTP E2E.

| Observation | 60-second sample | 300-second sample |
|---|---:|---:|
| Full-text characters, both versions | 269 | 1,908 |
| Segments, both versions | 18 | 151 |
| Detected speakers, both versions | 1 | 3 |
| Segments with changed text | 10 | 136 |
| Timestamp pairs changed | 0 | 0 |
| Speaker labels changed | 0 | 0 |
| Punctuation-only segments, old to new | 0 to 0 | 1 to 0 |
| Invalid timestamp bounds or decreasing starts | 0 | 0 |
| 1.4.14 inference time, two runs | 2.83–2.92 s | 10.54–10.64 s |
| 1.4.19 inference time, two runs | 3.34–3.61 s | 14.04–14.47 s |

Full text is identical between versions. Each version's complete output JSON
is identical across its two runs. The old result splits Chinese words between
adjacent segments. The new result keeps these words together and replaces the
five-minute sample's final punctuation-only segment with its closing speech.
Concatenated segment text matches full text after case and non-alphanumeric
normalization in both versions. This is a segment-allocation change, not
observed recognition of extra words.

The five-minute sample is about 35% slower by the two-run mean. These are
bounded observations, not a general performance benchmark. Import and model
loading are excluded from inference time. Process peak RSS was approximately
5.02–5.10 GB for 1.4.14 and 3.76–3.85 GB for 1.4.19. RSS is not peak unified
CPU/GPU memory. Residual Metal allocations remained after `release()`; each
isolated process then exited.

## Speaker patch limitation

Upstream 1.4.19 `distribute_spk` sums overlaps per speaker. The existing project
patch chooses the largest individual overlap and replaces both upstream call
sites. This PR therefore does not enable that upstream attribution change.
The upstream function still lacks the project's None-timestamp filtering.
Keep the patch; changing it needs a separate behavior change and tests.

Identical speaker labels do not prove identical speaker-attributed accuracy:
changed text allocation can move words between speakers. No human gold
transcript or speaker timing reference was used, so no CER or DER gain is claimed.

## Validation

The isolated comparison environments each passed Ruff, Tach, and 523 tests:

```sh
python -m pytest tests/unit tests/reliability/test_worker_session_process.py tests/integration -q
```

The final PR worktree also passed Ruff, Tach, 466 CI-path tests and 57 API
integration tests. A fresh MPS run of both clips produced complete JSON outputs
identical to the earlier 1.4.19 comparison, with matching weight hashes.
Remote check results are recorded in the PR.
Private source recordings, outputs, exact timing logs, package inventories,
weight hashes, source diffs, and reproduction scripts remain in local evidence
under `pr-reviews/funasr-ab-20261101/`. They are not committed or published.

## Unverified scope and activation

- Full-length Mandarin recordings and long-input resource behavior.
- Human-reviewed recognition accuracy, speaker accuracy, and audible boundaries.
- English and mixed-language quality, SenseVoice, explicit speaker caps, and
  exhaustive empty-token failure paths.
- Real-model HTTP submission, worker startup/offload, and live service cutover.

The running checkout and its environment are outside this change. Merge alone
does not synchronize that environment or authorize a restart. For a later
cutover, preserve the prior lock and compare the installed environment before
switching. Reverting the dependency change restores the prior package selection;
it does not undo transcripts already produced by the newer version. Preserve
source audio and prior outputs so affected outputs can be compared or regenerated.
