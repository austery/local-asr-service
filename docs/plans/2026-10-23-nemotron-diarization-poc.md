# Nemotron 3 Diarization: isolated PoC proposal

Status: draft for owner review. This branch contains the plan only; do not
interpret it as approval to download weights, run models, or integrate a service.

## Question and scope

Can Nemotron assign useful, stable speaker identities to Lei's Chinese,
English and bilingual recordings on Apple Silicon, at an acceptable resource
cost? Evaluate who spoke when before evaluating speaker-tagged transcripts.

Keep main and the running service unchanged. Use the existing locked
mlx-audio 0.6.1 / MLX 0.32.4 stack in a separate uv environment. Do not revive
the retired production pipeline, add public model aliases or endpoints, change
MOSS contracts, or work on Phonon. Any production integration is a later decision.

## Verified starting points

- Primary/main and origin/main: `e3c61f78f5024e97e17412e5b62962bb3742ecfb`.
- Installed 0.6.1 contains `mlx_audio.vad.models.nemotron_diarization`, including
  whole-file, generator and incremental PCM interfaces. No framework upgrade
  is needed merely to begin this experiment.
- Candidate: `mlx-community/Nemotron-3-Diarization`, revision
  `59ed2dbfc1346dcea9d423c71306a3a2499c568f`. Its safetensors file is
  198,632,220 bytes (about 199 MB decimal), not a runtime-memory estimate.
- The published configuration has eight speaker channels, 16 kHz audio,
  160-sample hop and output_subsampling_factor=1: a 10 ms output grid.
  Frame spacing is not boundary accuracy or end-to-end latency.
- The model emits anonymous speaker IDs, not real-person identification. Preserve
  simultaneous active speakers instead of imposing a single-label timeline.
- Model license: OpenMDW-1.1; do not describe it as Apache-2.0. Record the license
  and pinned conversion provenance with the experiment artifacts.
- `src/core/diarization_port.py:25` supplies an existing independent boundary;
  `src/core/mlx_sortformer_diarizer.py:45` and `src/core/diarization_registry.py:15`
  describe the current experimental four-speaker control. They do not establish
  Nemotron compatibility or require registry changes in this first PoC.
- `src/adapters/segment_alignment.py:31` chooses one best speaker for an entire
  transcript segment. It is not an overlap-preserving evaluator, and long coarse
  Qwen chunks are not sufficient for accurate speaker-tagged subtitles.

Sources checked for planning:

- [NVIDIA model card](https://huggingface.co/nvidia/Nemotron-3-Diarization)
- [MLX conversion card](https://huggingface.co/mlx-community/Nemotron-3-Diarization)
- [Pinned MLX configuration](https://huggingface.co/mlx-community/Nemotron-3-Diarization/blob/59ed2dbfc1346dcea9d423c71306a3a2499c568f/config.json)

The NVIDIA card includes Mandarin material and evaluations. Those upstream
results are motivation, not local MLX parity or acceptance evidence.

## Stage 1: reproducible loading and interface probe

After plan approval:

1. Create a task-owned uv environment from the current lock. Preserve the
   production environment, shared caches and existing recordings.
2. Download only the pinned candidate's required model/config files, retain
   hashes and provenance, then run offline. Reuse cached control weights;
   document an unavailable control rather than silently downloading alternatives.
3. Start with the existing 60-second English two-speaker fixture. Save raw
   segments, probabilities, configuration, elapsed time, load time and memory.
   Validate finite, positive, in-range intervals and the observed speaker count.
4. Check empty/silent input, late speech onset, final partial chunk/flush and
   independent-file state reset. Labels must not leak from a previous recording.
5. Keep a standalone experimental runner first. If a typed adapter becomes useful,
   target the existing DiarizationPort; do not register it in production yet.

Do not copy Sortformer's threshold=0.35, min_duration=0.2 and merge_gap=0.3
without checking Nemotron. First record its defaults (0.5, 0.0, 0.0). Any tuning
uses a separate development clip and is frozen before held-out scoring.

## Stage 2: reference-labeled speaker-quality evaluation

Freeze audio hashes, channel/resampling recipes and recording-wide speaker
identities. Existing audio alone is not a DER reference. Select a small labeled
AMI/AliMeeting subset if available under its terms, or prepare independently
reviewed RTTM annotations for local clips. Do not use another model's output as
human ground truth. Missing labels block quantitative quality claims, not a
load/speed smoke test. Do not fetch whole datasets for the initial PoC.

| Tier | Proposed inputs | Main question |
| --- | --- | --- |
| Control | Silence, one speaker, existing 60 s English two-speaker clip | False speech, invented speakers, basic separation |
| Core | 5-10 min Chinese and English conversations, 2-4 known speakers | Merged/split speakers and short backchannels |
| Difficult | Bilingual clip, similar voices, overlap and long pauses | Confusion, missed overlaps and identity after return |
| Capacity | Labeled 5-8-speaker clip; constructed mixtures separately marked | Whether extra speaker channels are actually useful |
| Long form | One natural 20-60 min recording; label a short prefix, middle and tail | State stability, memory growth, skipped tails |

A few labeled windows cannot establish whole-file DER. Score them with their
explicit evaluation mask; do not concatenate windows or remap identities per
window. Preserve absolute offsets and a single speaker mapping per recording.
Synthetic mixtures stress capacity/overlap but do not prove natural-meeting quality.

Primary control: the existing MLX four-speaker Sortformer on identical 2-4-speaker
inputs in the same environment. Report each model's documented parameters. If
postprocessing differs, distinguish the as-used product comparison from a shared
postprocessing sensitivity check. Do not frame the four-speaker control's
capacity limit on 5-8 speakers as an accuracy improvement by itself.

## Stage 3: metrics and proposed review gates

Report per recording, not just a global average:

- DER with overlap included, decomposed into missed speech, false alarm and
  speaker confusion; use one optimal reference/hypothesis mapping per recording.
- Primary collar 0.25 s plus a stricter zero-collar view. Keep both protocols
  explicit and do not compare their values to differently scored upstream tables.
- Exact speaker count, excess/missing identities, and reference-reviewed ID drift
  when a person returns after silence or across native chunk/cache boundaries.
- Transition-boundary error, short-turn recall and overlap behavior. Do not infer
  these from the nominal 10 ms grid or the number of nonempty segments.
- RTF = elapsed processing time / audio duration, load time separately, plus
  process peak RSS and MLX peak allocation. Define preprocessing inclusion.

Proposed acceptance targets, for discussion rather than observed performance:

| Area | Candidate gate |
| --- | --- |
| Clean held-out 2-speaker audio | Correct count and DER <=10% with the primary collar |
| Held-out 3-4-speaker meetings | DER <=20%; no worse than the usable existing control on the core set |
| Silence | No invented persistent speaker; predicted speech <=0.5 s per silent minute |
| Long-form identity | No unrecovered identity replacement on annotated returning-speaker checks; no missing final speech window |
| Operational fit | Warm long-form RTF <=1 and owned-process memory <=8 GiB |

Scores failing these targets produce a hold/reject decision, not hidden threshold
changes on the evaluation set. Revisit targets with the owner if the corpus has a
materially different difficulty. Evaluate 5-8-speaker capacity separately; do not
claim it passed when that labeled tier is unavailable. Repeat representative
short/long runs to check output stability and memory growth.

Whole-file generate collects results even though encoder context is bounded;
measure actual long-input memory. For incremental checks keep one state across
chunks, flush the final partial input, and compare with whole-file output.
Do not independently diarize every audio chunk and concatenate speaker IDs.

## Stage 4: conditional transcript-tagging demonstration

Only if the standalone quality/operational gates pass and the owner agrees:
reuse a frozen Qwen transcript with sufficiently fine timestamps and attach the
speaker turns in an offline artifact. Audit the existing aligner before reuse;
it currently collapses overlap and coarse segments can span multiple speakers.
If required fine alignment weights are not already cached, report that separate
cost before adding another download. Evaluate tagging errors separately from
ASR word errors. A useful demo does not enable the retired pipeline or imply
permission to expose an API profile.

## Isolation, evidence and stopping conditions

One owned model process at a time. Check production identity/children and
available memory before inference. Require at least 20 GiB available; stop the
owned experiment if the production service begins inference, memory exceeds the
proposed cap, or output is malformed. Never stop/restart production to finish a
benchmark. Use deadlines and release the owned process/Metal allocations between
models. Retain raw output before conversion or scoring.

Keep run manifests, exact uv lock, model/audio hashes, annotation provenance,
RTTM/raw JSON, per-file scores, resource logs, failure examples and a concise
recommendation outside disposable environments. Private audio/annotations stay
untracked. Preserve shared caches and user data; remove verified inactive run
resources after exporting evidence. Keep this worktree while the plan/PoC is active.

## Deliverable and decision

Deliver a compact side-by-side report plus inspectable speaker timelines for
selected reference-labeled windows. Classify the result as: reject, continue
observing, extend independent PoC, or propose a separately reviewed integration.

Current state: branch and plan created; no weights downloaded, no environment
installed, no inference run, and no service/source/registry change made in the
primary checkout. The next action is owner review of this proposal.
