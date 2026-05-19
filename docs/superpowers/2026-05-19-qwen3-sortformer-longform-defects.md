# Qwen3 Sortformer Long-Form Probe Defects

Date: 2026-05-19

## Context

The `qwen3-sortformer` profile is intentionally discovery-only after PR #15. A
57m17s four-speaker English interview was run through a temporarily requestable
test instance to validate long-form behavior before considering public enablement.

Input:

```text
/path/to/Why Does 2 + 2 = 4？ What Math Teaches Us About Deep Reality.mp3
duration: 3437.075750s
size: 79 MB
```

## Benchmark Evidence

| Model | Wall time | Approx RTF | Output shape | Quality note |
| --- | ---: | ---: | --- | --- |
| `qwen3-sortformer` | 551.09s | 0.16 | 646 segments, `Speaker 0..3` + `Unknown` | Cleaner English transcript; Unknown duration about 63s, about 1.84% of output timeline. |
| `paraformer` | 100.85s | 0.029 | 1245 segments, `Speaker 0..5` | About 5.5x faster, but English transcript quality is poor: many name/word errors, filler hallucinations, and Chinese punctuation/fillers. |

## Defects Found

1. `qwen3-sortformer` response returned `duration: 0.0` even though ffprobe
   resolved the real audio duration for chunk planning.
2. FunASR/Paraformer JSON segments exposed raw millisecond timestamps while the
   API schema and MLX path expose seconds.
3. FunASR/Paraformer JSON responses omitted `duration` even when segment end
   timestamps were present.
4. Qwen3 forced-alignment output can contain zero-duration words; when those
   words get their own speaker assignment, the service emits zero-duration
   speaker segments.
5. Long-form chunk boundaries still show occasional repeated text. The 57m17s
   probe had visible repetition near the closing credit and a few large
   boundary gaps with duplicated phrases. This remains a follow-up quality issue.

## Fix Scope In This Branch

- Treat non-positive transcript duration as missing and probe the source audio.
- Propagate resolved pipeline duration into the final pipeline response.
- Normalize FunASR JSON segment timestamps to seconds and include duration.
- Skip zero-duration aligned words when building speaker-labeled segments.

## Deferred Follow-Up

- Add lexical overlap cleanup around chunk boundaries after collecting more
  examples. This should be handled carefully because duplicate-looking phrases
  may be legitimate repeated speech in interviews.
- Add a reusable benchmark summary script for long-form probe JSON artifacts.
- Keep `qwen3-sortformer` `requestable=False` until a longer 3-5 hour probe and
  chunk-boundary cleanup are validated.
