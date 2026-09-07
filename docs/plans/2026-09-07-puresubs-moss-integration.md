# PureSubs MOSS Integration Roadmap

Status: Planned; caller implementation has not started.
Date: 2026-09-07
Owner: PureSubs, `packages/automation-engine-ytdlp`.
Parent: [Runtime roadmap](2026-09-07-runtime-upgrades-and-moss-evaluation.md).

## Problem and verified baseline

The ASR service now accepts `model=moss-transcribe-diarize` with explicit
`language=en` and an input duration of at most 1,800 seconds. Longer input
returns 400 before inference; the MOSS path does not call the generic chunker.
The bound is a provisional service admission policy, not a universal MOSS maximum.

PureSubs was inspected read-only at commit
`b3a254ce066250a29691636343cbde0b911a6d4e`:

- `src/transcription/factory.ts` creates the local provider's chunker with a
  24 MiB size threshold, retaining a buffer below the caller's 25 MB limit.
- `src/transcription/AudioChunkingService.ts:processAudio` normalizes first
  and skips splitting whenever the output is below the byte threshold.
  The default encoding in `config/default.yml` is mono, 16 kHz, 64 kbps MP3.
  Duration is measured but is not an admission/splitting condition.
- At 64 kbps, 40 minutes is approximately 18.3 MiB before container overhead:
  it passes the size condition and fails MOSS's duration condition. A 24 MiB
  budget corresponds to about 52.4 minutes, not 30 minutes.
- `src/transcription/providers/LocalWhisperService.ts` already sends
  `response_format=verbose_json`, forwards language when supplied, and renders
  `segments[].speaker`. The ASR API accepts that format. The provider retries
  all errors and concatenates chunk text without global speaker matching.
- `src/transcription/LocalAsrModelSelector.ts` chooses `paraformer` or
  `qwen3-asr` from speaker signals unless the user/channel explicitly selects
  a model. This task does not change automatic routing.

Paths above are relative to PureSubs's `packages/automation-engine-ytdlp/`.
The ASR implementation is in [mlx_engine.py](../../src/core/mlx_engine.py);
its public scope is documented in [MODELS.md](../../MODELS.md).

## Proposed separate PureSubs PR

1. Add a model-specific duration bound alongside the existing byte bound.
   For MOSS, split when either bound is exceeded. Retain current behavior for
   other models. Resolve/validate explicit English before uploads.
2. Prefer existing silence cuts, but enforce the duration bound even when no
   suitable silence is available. Include overlap and encoded MP3 padding in
   the budget. Probe each generated chunk: duration must be positive, finite,
   and at most 1,800 seconds, with bytes within the existing size limit.
3. Keep speaker identity scoped to a chunk. For example, chunk 1 / S01 and
   chunk 2 / S01 are distinct labels unless a future independently validated
   identity-matching capability establishes equivalence. Preserve the chunk
   boundary in the saved transcript; do not silently merge same-named speakers.
   Retain meaningful chunk time offsets for review.
4. Stop retrying deterministic MOSS input/output rejections unchanged. Surface
   400/422 with the failing chunk context. A partial multi-chunk transcript
   must not be stored as a complete success. Keep current network retry policy.
5. Exercise one real English long recording through the caller and document
   its final chunk durations, text, speaker labels, failure behavior, and cost.

This is an implementation handoff, not a claim that cross-chunk identity is
solved. Cross-recording speaker matching, custom embeddings, model internals,
ASR-server splitting, and a new default model are outside this task.

## Acceptance checks

- A 40-minute input below 24 MiB is split for MOSS; no request exceeds 1,800 s.
- Boundary tests cover an exact 1,800-second encoded input and the first value
  above the limit, an oversized short file, no usable silence, overlap/padding,
  and invalid/non-finite duration. Verify actual generated files, not estimates.
- Every upload respects both bounds; non-MOSS models keep their current policy.
- `model=moss-transcribe-diarize`, `language=en`, and `verbose_json` reach the
  server. Unknown/non-English language fails clearly before upload.
- Same-named speakers from different chunks remain distinguishable in the
  stored transcript. Chunk ordering and offsets survive normalization.
- A failed middle chunk cannot produce a successful incomplete transcript;
  deterministic 400/422 responses are not retried unchanged.
- Complete a real greater-than-30-minute English caller run with user listening
  review. Record any boundary text duplication or omissions instead of claiming
  complete word coverage or globally consistent speaker identities.

## Delivery and rollback

Implement in a PureSubs worktree and submit a separate reviewed PR. Keep the
ASR 1,800-second guard as defense in depth. Rollback disables MOSS selection in
that caller and returns to its prior model policy; do not remove the ASR guard
or enlarge its limit to accommodate size-only chunks.
