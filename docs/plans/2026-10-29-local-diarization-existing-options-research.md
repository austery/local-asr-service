# Existing local speaker-attributed transcription routes

Date: 2026-10-29. Scope: primary-source research only. No model downloads,
installation, inference, private audio access, service changes, or production code
changes were performed. Upstream `main` was inspected; it is not an immutable
release pin and can be ahead of installed packages. Recheck and pin an exact
revision before any isolated trial.

## Decision implication

Do not turn the temporary experiment into a general orchestration product yet.
Existing implementations already cover much of ASR, diarization, attribution and
output. One bounded comparison against one existing implementation is more useful
than adding engines or a UI. None of the inspected public materials establishes
that an alternative beats Qwen3-ASR, MOSS, or the current private experiment on the
owner's recordings. That requires matched inputs and human review.

Three particularly relevant reusable routes follow. They answer different
questions: upstream composition closest to the current Python stack; an official
complete native runtime; and a ready Qwen-preserving pipeline using a different
diarizer.

## 1. Reuse mlx-audio's existing Nemotron integration

The upstream [integration example](https://github.com/Blaizzy/mlx-audio/blob/main/examples/nemotron_diarization_asr.py)
contains real orchestration, not just a model card. `transcribe_with_speakers`
has masked and timestamp-based paths. Default masked mode requires Nemotron ASR's
`generate_speakers`; the timestamp path expects `AlignedResult.sentences` and
explicitly names Nemotron ASR or Parakeet. It is therefore not a ready substitute
for Qwen3's text plus a separate forced aligner.

Its `assign_speakers` implementation uses duration-weighted activity, handles a
zero-duration ASR emission with one frame, and leaves simultaneous speakers
ambiguous. That treatment is for emission timestamps, so copying it directly to
collapsed forced-alignment words would require validation. `merge_tokens` is a
readable-turn helper, not evidence that sentence continuity is solved for all
inputs. The [Nemotron adapter documentation](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/models/nemotron_diarization/README.md)
provides whole-file and continuous-state streaming inference on MLX/Metal.

Important distinction: feature masking is not physical source separation. The
example itself warns of repeated or missing words during simultaneous speech.

Usefulness: reuse upstream APIs and attribution ideas instead of maintaining a
parallel runtime. A Qwen-specific composition seam is still needed if preserving
Qwen transcription is a hard requirement. Presence on upstream main does not prove
presence or identical behavior in the installed mlx-audio 0.6.1 environment.

## 2. NVIDIA NeMo-Speech.cpp: official composed CLI, including Metal

The [official runtime README](https://github.com/NVIDIA/NeMo-Speech.cpp)
and [CLI guide](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/docs/cli.md)
document a complete `nemo-speech transcribe meeting.wav --diarize --json` path.
Default ASR is Nemotron 3.5, not Qwen. The generic `--diarize` flag currently
selects the indexed default Sortformer; choose an explicit local Nemotron 3
Diarization GGUF through `--diar-model` to test the intended eight-speaker V3
model. This route can reduce custom glue, but changes the ASR baseline.

[Build documentation](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/docs/build.md)
provides Apple Silicon Metal builds. That establishes a supported hardware route,
not observed speed on this machine. In the [recognizer source](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/src/asr/recognizer.cpp),
`Recognizer` loads a `DiarModel` when configured; this is composed native runtime
code. The [diarization implementation](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/src/asr/diar/diar_pipeline.cpp)
maintains state and an activity timeline. Long files use the streaming-state path;
full-attention `--offline` is a separate bounded mode and is not the long-meeting
recipe.

There is directly relevant long-form defect history. [PR #70](https://github.com/NVIDIA/NeMo-Speech.cpp/pull/70),
merged 2026-10-16, fixes activity-frame compaction before delayed word tagging.
The [actual diff](https://github.com/NVIDIA/NeMo-Speech.cpp/pull/70/files) changes
`DiarStream` ownership of retained frames and the recognizer's hold boundary.
The author's reported AMI improvement is not this project's acceptance evidence.
Choose a build containing that fix; still check beginning, middle and end of a
long recording. Standalone diarization success alone does not establish correct
composed word attribution.

Usefulness: the strongest existing complete native alternative to try once before
building another production pipeline. No local build or acceptance was performed.

## 3. mlx-qwen3-asr: Qwen + alignment + pyannote already composed

The canonical [moona3k repository](https://github.com/moona3k/mlx-qwen3-asr)
advertises `mlx-qwen3-asr meeting.wav --diarize --num-speakers 2 -f json` as an
experimental offline option. Its [transcription source](https://github.com/moona3k/mlx-qwen3-asr/blob/main/mlx_qwen3_asr/transcribe.py)
actually resolves a forced aligner when diarization is enabled and then performs
speaker attribution. This is a real composed Qwen route, but its default ASR size
is 0.6B; explicitly selecting 1.7B is necessary for a meaningful model-size match.

The [diarization source](https://github.com/moona3k/mlx-qwen3-asr/blob/main/mlx_qwen3_asr/diarization.py)
uses `pyannote/speaker-diarization-community-1`, not Nemotron. It accepts
CPU/MPS/CUDA/auto selection and can retry CPU after accelerator operator failure.
Thus native MLX ASR does not imply the complete pipeline is MLX-only or uniformly
accelerated. Gated model access may require accepting terms and obtaining a token
before caching models for offline use.

Its fallback behavior also differs from the private experiment: an empty native
turn list can yield one default speaker covering the file, and intervals are
assigned a dominant label. Therefore it is an alternative to evaluate, not a
semantics-preserving replacement for explicit overlap/uncertainty handling.
No local quality, speed or completeness was measured here.

## Why not broaden the shortlist further

[speech-swift](https://github.com/soniqo/speech-swift) already supplies Qwen3-ASR,
a forced aligner, and [Nemotron 3 Core ML/MLX INT8 diarization](https://github.com/soniqo/speech-swift/blob/main/docs/inference/speaker-diarization.md).
However, the inspected [TranscribeCommand.swift](https://github.com/soniqo/speech-swift/blob/main/Sources/AudioCLILib/TranscribeCommand.swift)
has engine selection and a separate MOSS branch; it does not expose a generic
Qwen-plus-Nemotron diarization flag. The existence of all three components is not
proof of a finished composition. It is useful for a Swift-native product, but
porting this private offline experiment to Swift solely to connect those pieces
would increase scope.

[FluidAudio's CLI](https://github.com/FluidInference/FluidAudio/blob/main/Documentation/CLI.md)
exposes transcription, standalone processing/diarization, and offline VBx tooling.
Its [API](https://github.com/FluidInference/FluidAudio/blob/main/Documentation/API.md)
has disk-backed long-file diarization and CoreML components. These are worth
knowing about, but selecting this SDK would introduce another backend/model
comparison rather than directly productize the existing Qwen+Nemotron result.

[WhisperX](https://github.com/m-bain/whisperX) is an established complete
Whisper/alignment/pyannote CLI. Its documented Mac route is CPU int8; CUDA speed
claims are not Mac performance evidence. It also changes both ASR and diarizer,
so it is a secondary baseline rather than the first incremental experiment.

## Bounded next action

Keep the existing private experiment frozen. Compare its original outputs and
readable rendering separately from acoustic/model quality. First human-review
matched existing outputs on short monologue, normal turn exchange, and overlap
windows. Count meaningful word errors, negation/number/name errors, speaker
attribution errors, omitted interjections and the correction work needed. Do not
compute DER without independent reference turns or claim relative WER without a
reference transcript.

If an additional runtime comparison is useful, choose only one: NeMo-Speech.cpp
for an upstream-composed native baseline, or mlx-qwen3-asr if Qwen text preservation
is non-negotiable. Freeze parameters before scoring. Evaluate long-file identity
continuity separately from short-window correctness. Promote only a small reusable
file-to-transcript command if the owner observes a meaningful reduction in review
work. Defer HTTP integration, background jobs, real-time features, cross-recording
identity, automatic engine selection and source-separation research.
