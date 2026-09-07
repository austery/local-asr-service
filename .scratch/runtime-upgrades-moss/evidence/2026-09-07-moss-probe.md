# MOSS Upstream Probe — 2026-09-07

Status: Current long-form candidate fails the completeness gate. No-Go is recommended for this runtime/checkpoint pairing; Lei’s roadmap decision is pending. No production integration was performed.

## Identity and isolation

- Runtime: MLX Audio 0.5.1 in `.venv-mlx`; candidate lock SHA-256 `e04a6f4e59d950b58b2dac8e239162b04ec5eb0c833f0a90b3b38d5369cf7bb5`.
- Checkpoint: `OpenMOSS-Team/MOSS-Transcribe-Diarize`, immutable revision `704aa4a9c304e8520be88901e0d1960158ef5b15`, publisher BF16 weights loaded directly by the upstream MLX implementation. No custom conversion or quantization.
- The checkpoint's tokenizer configuration has no `auto_map`; no downloaded tokenizer Python code is used.
- One owned model process at a time, Hugging Face offline during inference, a 48 GiB MLX allocation limit on the 64 GiB Mac, and isolated output files. The original main checkout/environment is unchanged.
- Raw responses and local audio remain gitignored under `.runtime-validation/`. The probe calls the upstream model directly; public HTTP capabilities and normalization are not yet certified.

## Recorded runs

| Audio | Load | Inference | MLX peak | Generated / allowed tokens | Segments / predicted speakers | Final end |
| --- | --- | --- | --- | --- | --- | --- |
| English 60 s | 2.277 s | 3.305 s | 2,659,938,178 bytes | 380 / 2,048 | 12 / 2 | 59.94 s |
| English 277.333 s holdout | 2.083 s | 21.291 s | 3,058,798,702 bytes | 1,913 / 4,096 | 48 / 2 | 277.43 s |
| All-In 3,600 s | 2.104 s | 715.803 s | 9,943,850,650 bytes | 16,376 / 32,768 | 407 / 4 | 2,179.08 s |

All three recorded runs exit with code 0, have finite positive-duration segments, retain speaker IDs on every segment, and do not hit the token budget. Predicted speaker counts are not measured attribution accuracy.

The medium file's last segment exceeds actual audio duration by 0.097 seconds. This violates a literal in-bounds timestamp contract in raw upstream output. Record and inspect it before deciding whether bounded endpoint normalization is appropriate; do not silently relax the accepted criterion. No production adapter has been changed.

## Fixed review windows and limits

The All-In windows were recorded before MOSS inference: 00:00–01:00, 30:00–31:00, 49:30–50:30, and 59:00–60:00 within the one-hour excerpt. The original episode's source interval is 00:05:00–01:05:00. The medium English conversation is held out from tuning. The 49:30 window targets the configured 50-minute service split threshold, not an asserted actual speaker-return event.

Only human-reviewed references can establish WER/CER or speaker-attribution accuracy. Compare full recording coverage separately from the four windows. Anonymous speaker labels can be renamed once across a recording, but a label must remain attached to the same person when that speaker returns.

## Integration boundary still to prove

Upstream segments use `speaker_id`, while the API reads `speaker`. Raw text contains timestamp/speaker markup. The default upstream budget of 2,048 tokens is insufficient as a general long-form contract. MOSS has no explicit language-forcing argument. The existing service may split recordings after 50 minutes, which can reset anonymous speaker identity across calls.

A successful raw probe can justify a bounded experiment in the existing MLX adapter: truthful capability declarations, speaker/text normalization, bounded output budget, truncation rejection, and valid duration handling. It does not justify a new MossEngine, alignment rescue, speaker embeddings, or cross-call reconciliation framework. Preserve the other model roles and the existing pipeline until a separate retirement decision.

## Long-form completeness failure

The one-hour run exits with code 0 and finishes after 715.803 seconds, with 47,738 prompt tokens and 16,376 generated tokens. It is below the configured 32,768 output-token budget. There are 407 valid in-bounds segments with four anonymous labels, but the final segment ends at 2,179.08 seconds (36:19.08): the last 1,420.92 seconds (23:40.92) have no output.

The cleaned MOSS transcript contains 38,251 characters, versus 64,518 for Qwen and 62,299 for Paraformer. Its final passage matches the middle portion of the Qwen reference, while both reference outputs continue with later material through the recording’s end. These comparisons support a coverage failure; they are not WER ground truth. Normal exit, well-formed segments, and four speaker labels do not pass the completeness gate.

The probe was one upstream call for the whole recording, without this service's 50-minute chunking, so the result cannot be explained by the service's existing chunk-offset bug. The reason for early model termination is not established. There is no evidence yet that increasing the already-unreached budget, changing the wrapper, or splitting the recording would preserve recording-wide speaker identity and solve it.

The private listening review is `.runtime-validation/moss-review/index.html`, with four local PCM WAV files, each verified to be exactly 60 seconds. The final two review windows intentionally show no MOSS segments. A user decision was requested: record this candidate as No-Go and proceed with FunASR, or prioritize investigating early termination first. Do not describe the entire MOSS model family as incapable based on one implementation/checkpoint/corpus result.
