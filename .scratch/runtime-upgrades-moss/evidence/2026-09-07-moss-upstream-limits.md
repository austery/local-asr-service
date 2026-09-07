# MOSS Upstream Limits — 2026-09-07

Status: Retain MOSS for evaluation. The publisher claims support for 90-minute inputs, but this does not establish reliable 90-minute operation in the local MLX deployment. A known upstream premature-termination report closely matches the local observation. No new model inference, production integration, dependency change, or checkpoint replacement was performed for this research.

## Findings

| Question | Evidence-backed answer |
| --- | --- |
| Published maximum input duration | Up to 90 minutes in a single generation. This is the publisher's supported-use claim, not a guarantee for every recording. |
| Configured context | 131,072 positions in the exact checkpoint used locally. Audio input and generated transcript share that context. |
| Audio token rate | 12.5 audio tokens/second, plus timestamp-marker and prompt tokens. A 60-minute recording contributes 45,000 audio tokens before that overhead. |
| Output budget | A separate limit. The pinned generation config defaults to 5,120; the official Python example uses 2,048 and the long-form serving example uses 65,536. MLX's model method defaults to 2,048, overridden by our explicit 32,768. |
| Local stopping event | 16,376 generated tokens, below the requested 32,768; total reported prompt plus output is 64,114, below 131,072. Source inspection implies an EOS stop in the successful local execution path. Its exact terminal token was not recorded. |
| Reliable local duration ceiling | Not established. Successful 60-second and 277-second samples and one failed 60-minute sample do not prove a universal cutoff at 36:19, 30 minutes, or another duration. |

The [immutable model card](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize/blob/704aa4a9c304e8520be88901e0d1960158ef5b15/README.md) documents the duration and output-budget examples. The [technical report, v7](https://arxiv.org/pdf/2601.01554v7), pages 1–4, describes 128K context and single-pass operation up to 90 minutes; its listed podcast evaluation recordings span 1,528.7–3,636.5 seconds. A claimed 90-minute design envelope and reported roughly hour-long benchmarks are distinct from this project's measured completeness.

## Exact configuration and implementation

The pinned [config.json](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize/blob/704aa4a9c304e8520be88901e0d1960158ef5b15/config.json) sets `text_config.max_position_embeddings=131072`. MLX's `TextConfig` fallback of 40,960 is overridden when the checkpoint dictionary is loaded; it is not evidence of a 40,960-token active limit. See the [MLX configuration loader](https://github.com/Blaizzy/mlx-audio/blob/6b54ec6ecd99d0ad77dfa33dd129707e31bf051c/mlx_audio/stt/models/moss_transcribe_diarize/config.py#L65).

The checkpoint's [processor_config.json](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize/blob/704aa4a9c304e8520be88901e0d1960158ef5b15/processor_config.json) specifies 12.5 audio tokens/second, a merge factor of four, and a marker every five seconds. Its [preprocessor_config.json](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize/blob/704aa4a9c304e8520be88901e0d1960158ef5b15/preprocessor_config.json) specifies 16 kHz audio, a 160-sample hop, and 30-second encoder chunks. The [generation_config.json](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize/blob/704aa4a9c304e8520be88901e0d1960158ef5b15/generation_config.json) specifies EOS 151645 and a default 5,120 generated tokens. These defaults are separate from explicitly supplied limits.

The [MLX generation loop](https://github.com/Blaizzy/mlx-audio/blob/6b54ec6ecd99d0ad77dfa33dd129707e31bf051c/mlx_audio/lm/generate.py#L118) counts generated tokens until `max_tokens`; it does not contain an additional 16K cap. The [MOSS wrapper](https://github.com/Blaizzy/mlx-audio/blob/6b54ec6ecd99d0ad77dfa33dd129707e31bf051c/mlx_audio/stt/models/moss_transcribe_diarize/moss_transcribe_diarize.py#L698) passes that budget through and stops on tokenizer EOS, 151643, or 151645 before appending that token. Therefore an EOS stop is the source-path explanation consistent with the successful 16,376-token local run. It is not yet a token-trace proof of which EOS was selected or why its probability became highest. Increasing an unused budget alone is not a demonstrated remedy.

## Upstream premature-termination evidence

[OpenMOSS issue #26](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/issues/26) remains open as checked on 2026-09-07. It reports 40-minute inputs ending early despite 131K context and a 65K output request. A [project collaborator acknowledged repeated reports](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/issues/26#issuecomment-5065836718) and said the problem was being investigated for a fix.

An [independent Transformers-path report](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/issues/26#issuecomment-5142583626) records approximately 16,336–16,339 generated tokens on several longer recordings despite a 32,768 budget. Another [reporter describes EOS becoming the argmax near token 16,384](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/issues/26#issuecomment-5284106026). These are firsthand external reports, not our reproductions or a maintainer-confirmed training diagnosis. The local 16,376-token result resembles them. Learned early EOS is a plausible failure class; a hard 16K configuration limit or a specific training cutoff is not established.

[Issue #34](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/issues/34) independently reports incomplete output on a 1,748-second input through vLLM with a large requested budget; it remains open without maintainer comments. This also argues against treating any fixed duration inferred from a single recording as universally safe.

The [live model metadata](https://huggingface.co/api/models/OpenMOSS-Team/MOSS-Transcribe-Diarize) still points to our exact revision `704aa4a9c304e8520be88901e0d1960158ef5b15` as of this check. No replacement checkpoint or confirmed fix for issue #26 was identified in the reviewed official sources.

## Prompt parity and chunking semantics

The official [inference helper](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/blob/61bc29cd4120be7b5d3b761b64cd5dff57263642/moss_transcribe_diarize/inference_utils.py#L13) uses a Chinese instruction by default; MLX uses an English equivalent. This is a concrete input difference worth isolating with the existing `prompt` parameter if another controlled experiment is run. It is not a demonstrated cause or fix.

The pinned [official processor](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize/blob/704aa4a9c304e8520be88901e0d1960158ef5b15/processing_moss_transcribe_diarize.py#L173) and [MLX marker method](https://github.com/Blaizzy/mlx-audio/blob/6b54ec6ecd99d0ad77dfa33dd129707e31bf051c/mlx_audio/stt/models/moss_transcribe_diarize/moss_transcribe_diarize.py#L428) both calculate `int(12.5 * 5) == 62` and multiply by the marker index. The rounding is shared upstream behavior, not evidence of an MLX-specific defect. Both retain all remaining audio tokens.

Thirty-second encoder chunks are concatenated into one recording-level audio embedding sequence before one decoder generation. They are not separate transcription calls. Splitting the recording into independent requests changes that guarantee: speaker labels are relative to each input according to the model card, so `S01` in two requests need not denote the same person. The [official issue about independent segment labels](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/issues/20) contains no supplied cross-request identity solution. Any restricted-duration product mode must state that scope explicitly.

## Timing and practical next step

The [local probe record](2026-09-07-moss-probe.md) reports inference of 3.305 seconds for 60 seconds of audio, 21.291 seconds for 277.333 seconds, and 715.803 seconds (11 minutes 55.8 seconds) for the incomplete 60-minute result. Model loading adds approximately 2.1–2.3 seconds. The long run's elapsed time is not a complete-transcription benchmark and cannot establish the time required to finish the full recording.

The [MLX maintainer's performance PR #826](https://github.com/Blaizzy/mlx-audio/pull/826) reports a 66-minute test on M5 Max; this confirms long inputs were exercised in that implementation, not output completeness on our corpus. Its optimization is already an ancestor of the tested 0.5.1 source, as verified by the [immutable commit comparison](https://api.github.com/repos/Blaizzy/mlx-audio/compare/64e8416c303fb3b3463dab8eb4ebd78c55a87c1a...6b54ec6ecd99d0ad77dfa33dd129707e31bf051c). It is not an untried upgrade fix.

Retain the user's positive qualitative assessment of text and speaker attribution. Continue bounded investigation of completeness with an explicit stop-token trace and one controlled variable, then document only the duration and corpus envelope actually validated. A duration limit can reduce exposure to an output-length-related failure, but cannot guarantee success across speech densities and languages without additional evidence. Do not suppress EOS or force a minimum transcript length as an accepted repair without checking hallucinated tails and speaker continuity. Do not advertise 36:19 as the model maximum or promote the current incomplete long-form result.

## Later local duration follow-up

After this source-research snapshot, [local controlled duration probes](2026-09-07-moss-duration-validation.md) completed. Two disjoint 30-minute inputs pass bounded coverage checks; a 40-minute input directly emits EOS after 16,376 tokens and leaves 4:47 untranscribed. The updated operating proposal is 30 minutes for the next English experimental adapter, with explicit completeness rejection. This is a provisional tested scope, not a universal model maximum.
