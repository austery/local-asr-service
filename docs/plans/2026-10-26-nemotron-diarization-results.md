# Nemotron 3 Diarization: isolated evaluation

Decision: **hold integration and continue observing upstream quality**. The
candidate is fast and inexpensive on this machine, but does not consistently
meet the proposed speaker-quality gates. Keep Qwen3 as the transcription
baseline. No production alias, adapter integration, service restart or primary
environment change is part of this experiment.

## Reproducibility and scope

Evaluation date: 2026-10-26, Apple Silicon M4 Pro. Task branch:
`codex/nemotron-diarization-poc`. The approved proposal is
[the staged PoC plan](2026-10-23-nemotron-diarization-poc.md).
The worktree incorporates main `97f2450e0fa632505c36167df04212a847b2df30`.
An isolated uv environment uses the project lock, mlx-audio 0.6.1, MLX 0.32.4
and NumPy 1.26.4. Production remains a separate checkout and environment.

| Component | Immutable revision | Parameters |
| --- | --- | --- |
| `mlx-community/Nemotron-3-Diarization` | `59ed2dbfc1346dcea9d423c71306a3a2499c568f` | threshold 0.5, min duration 0, merge gap 0 |
| cached `mlx-community/diar_sortformer_4spk-v1-fp16` | `5233f9c657bf5e5535b7b79de040dab3ff9b905f` | existing experimental adapter's 0.35 / 0.2 / 0.3 |
| `pyannote/AMI-diarization-setup` | `67c2d539286e89f68952d5dcf83912bd9f01dfae` | test split, `only_words` RTTM and UEM |

This is an **as-used comparison**, including different postprocessing. It does
not isolate model architecture or reproduce vendor benchmarks. No thresholds
were tuned on these evaluation recordings. The related AMI clips overlap with
full meetings and are not independent samples for an aggregate significance test.

AMI references derive from manual AMI annotations v1.6.2, rather than another
model's hypotheses. Audio is from the University of Edinburgh AMI mirror
(CC BY 4.0); retain AMI attribution. The annotation setup is Apache-2.0.
Nemotron weights are OpenMDW-1.1. See the [AMI download terms](https://groups.inf.ed.ac.uk/ami/download/),
[pinned annotation setup](https://github.com/pyannote/AMI-diarization-setup/tree/67c2d539286e89f68952d5dcf83912bd9f01dfae),
and [MLX conversion card](https://huggingface.co/mlx-community/Nemotron-3-Diarization).

All evaluation input is mono 16 kHz PCM16; the runner refuses other formats.
Raw turns, probabilities, audio/reference hashes, errors and per-run JSON stay
in the private retained evidence directory, outside the disposable environment:
`.runtime-validation/nemotron-diarization-poc/` in the primary repository.
Private recordings and their derived outputs are not committed.

## Reference-labeled results

DER includes overlap, with one Hungarian speaker mapping per complete recording.
Primary collar is **0.25 s total centered width** (0.125 s on either side), matching
`pyannote.metrics`; also report zero collar. UEM is applied before mapping.
The denominator is reference speaker-time, including simultaneous speakers.
Window inspections retain the whole-recording mapping.

| Recording / evaluated range | Reference speakers | Nemotron speakers | Nemotron DER, collar 0.25 / 0 | Sortformer speakers | Sortformer DER, collar 0.25 / 0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ES2004a, first 180 s | 3 | 3 | 29.90% / 30.98% | 4 | 21.31% / 24.06% |
| ES2004b, 300–600 s | 4 | 4 | 21.48% / 22.60% | 3 | 19.43% / 21.92% |
| 30 s silence + ES2004a first 120 s | 3 | 3 | 34.95% / 36.36% | 4 | 26.05% / 29.32% |
| ES2004a, full 17m29s | 4 | 4 | 25.33% / 27.04% | 4 | 59.79% / 61.83% |
| ES2004b, full 39m05s | 4 | 5 | 19.89% / 21.60% | — | Resource-stopped; no score |
| Constructed two four-person groups + returning first group, 11m30s | 8 | 7 | 31.96% / 33.88% | — | Four-channel control not used as an eight-person quality baseline |

The capacity case concatenates ES2004a 0–360 s, IS1009a 0–240 s, then
ES2004a 0–90 s. All eight reference identities actually speak. It is a
constructed stress test, not a natural eight-person conversation. The first
240-second ES2004a proposal contained only three identities; that provisional
seven-person construction was caught by reference inspection and excluded.

Nemotron's full ES2004b primary DER decomposes into **19.13% missed speech,
0.56% false alarm and 0.19% confusion**. Low confusion does not imply a complete
transcript: missing speech dominates the total. Five predicted identities for
four real speakers also fails the exact-count check.

On this same long meeting, auxiliary diagnostics show:

- Short-turn recall: **77.48%** across 222 reference turns of at most one second,
  requiring at least 50% same-speaker union coverage.
- Reference-overlap wall-time: 203.16 s; at least two correctly mapped speakers
  are simultaneously recovered for **50.35%** of it.
- Fixed-mapping zero-collar DER in the final 60 s: **25.64%**. Speech reaches the
  final reference region; this does not establish perfect tail completeness.
- Nearest same-speaker boundary distance: median 0.03 s, p90 0.56 s. This is an
  optimistic many-to-one diagnostic; missed boundaries and short-turn failures
  prevent treating it as a one-to-one timestamp-accuracy score.

A separate reference-based returning-speaker audit inspects up to the three
longest >=30 s reference gaps per person in the natural meeting, using up to
five seconds on each side. The constructed-return check instead uses each
person's longest actual turn in the first 90 s and its exact repeat at +600 s
(the longest such window is 9.51 s).
Only target-exclusive reference speech with exactly one predicted active speaker
is used; each side requires >=0.5 s exclusive reference and >=50% dominant-ID
coverage. Of ten natural ES2004b checks, two retain the same raw ID, one changes
local assignment and seven lack sufficient unambiguous coverage. The constructed
return at +600 s retains IDs for two people; the third lacks coverage. These are
bounded identity checks, not proof of permanent replacement or a pass for all
long-form identities. Exact windows and counts are retained in
`returning-speaker-audit.json` with its reproduction script
`execution-scripts/audit_returns.py`.

The other auxiliary diagnostics use raw references without collar/UEM; their
protocol and per-recording values are in `summary.json`. The separate return
audit protocol and windows are in `returning-speaker-audit.json`. They are not vendor
benchmark metrics. The 10 ms output grid alone establishes no boundary accuracy.

## Behavior and operational fit

- Nemotron emits zero speakers on empty input and 60 s digital silence.
- AMI ES2004a 120–140 s has one reference speaker and yields one identity, but
  DER is 29.52% (primary collar); correct count alone misses substantial speech.
- Existing English two-speaker fixture: two identities; adding 30 s leading
  silence still yields two. There is no human RTTM for this fixture, so no DER.
- On the same loaded Nemotron model, A / one-second silence / A produces
  identical turns and probabilities (maximum absolute difference 0), with no
  invented silence speaker. This checks independent-file state reset.
- Whole-file and persistent-state one-second `feed()` partitions have identical
  probabilities for both 60 s and 37 s + 123 samples. Final `final=True` flush
  is included. The fractional tail is below one output-grid frame; this is not
  sub-frame boundary coverage. No independent per-chunk identity remapping is used.
- Existing Chinese and constructed Chinese/English ten-minute inputs finish and
  yield two and three identities respectively. Independent Chinese/bilingual
  speaker labels are unavailable: **DER and speaker correctness are unmeasured**.
  No authenticated corpus access or full dataset archive was fetched to fill this gap.
- Sortformer empty input raises a reduction error. Digital silence produces a
  persistent speaker and ends 0.16 s beyond the file. Some other native outputs
  exceed duration by 0.09–0.16 s. Raw output is retained; diagnostic cropping
  does not convert these failed output gates into passes.

Representative Nemotron runs with explicit `mx.eval` and `mx.synchronize`:

| Input | Model load | Inference | RTF | Process peak RSS | MLX peak allocation |
| --- | ---: | ---: | ---: | ---: | ---: |
| English 60 s | 1.49 s | 0.109 s | 0.00182 | 599 MiB | 579 MiB |
| Full ES2004b, 2345.49 s | 1.55 s | 3.605 s | 0.00154 | 974 MiB | 790 MiB |
| Constructed eight identities, 690 s | 1.57 s | 1.063 s | 0.00154 | 702 MiB | 683 MiB |

Inference excludes WAV decoding, weight loading, scoring and artifact writing.
These are single-process local warm-cache observations, not end-to-end ASR or
network latency. MLX allocation and RSS overlap conceptually; do not add them.
The repeated full-meeting run has identical DER. Early exploratory timings
precede the explicit synchronization barrier and are not the table's basis.

Sortformer on ES2004a reported **21.61 GB decimal MLX peak allocation**, despite
only 0.84 GB peak RSS. This exceeds the proposed 8 GiB operational gate. The
39-minute run was terminated after system available memory fell to roughly
2.5 GiB; its output is unscored. Available memory recovered after the owned
worker exited. RSS alone was insufficient protection, so the runner now also
samples actual MLX active/cache/peak allocation every 50 ms and terminates the
owned child when the observed value exceeds 8 GiB. The allocator limit itself is
only a guideline. The parent also checks the 20 GiB available-memory reserve
every 250 ms and writes supervision diagnostics. Sampling can briefly overshoot;
these safeguards are not an atomic memory reservation. This is an experimental
harness safeguard, not a production change. The large control is not rerun
merely to reproduce resource exhaustion. A final guarded Nemotron repeat was
also correctly stopped when available memory crossed 20 GiB during loading;
it has no score and does not replace the successful long-run measurements.
A real subprocess test with injected MLX allocation readings verifies that the
MLX monitor terminates its own low-RSS child on a 9 GiB reading.

## Gate status

| Proposed gate | Initial evaluation status |
| --- | --- |
| Clean independently labeled two-speaker DER <=10% | **Unmeasured**: existing fixture has no human RTTM |
| Core 3–4-speaker DER <=20%, no worse than usable control | **Not met consistently** |
| Silence false-speech bound | **Met by Nemotron** on the digital controls |
| Reference-reviewed returning-speaker ID drift across silence/cache boundaries | **Not established as passing**: bounded reference-exclusive checks have mixed identity consistency and insufficient coverage; no permanent-replacement conclusion |
| Long final speech window | Output reaches the final annotated region; complete recovery **not established** |
| RTF and owned memory | Nemotron observations fit; large Sortformer fails resource gate |
| Natural 5–8-person quality | **Unmeasured**; separate synthetic eight-identity test fails count |
| Chinese/bilingual DER and speaker-attributed WER | **Unmeasured** |

Completing this initial evaluation does not mean all stages 1–3 acceptance gates
passed. Missing labels and identity audits remain explicit gaps; they do not
change the hold decision.

## Outcome and next decision

Standalone Nemotron loading, controls, reset, incremental flush and long input
are viable. Quality gates are not met consistently: the core clips exceed 20%
DER, the full long meeting over-splits identities, short/overlap recovery remains
incomplete and the eight-person construction misses an identity. A usable
Chinese/bilingual speaker-attributed result has not been demonstrated.

Do not begin stage 4 transcript tagging or reopen the retired pipeline from
these results. CER/WER and cpWER remain **not measured**, because no transcription
model or attribution integration was evaluated. Any follow-up should first bring
independent Chinese/bilingual and natural multi-person labels, rather than add
another production orchestration layer.
