# MOSS opt-in adapter contract

The user accepted the two runtime upgrades and MOSS listening results on 2026-09-07 and requested separate pull requests. This implements the bounded next step in the duration report and ADR-002, for another agent to review before merging.

- Register `moss-transcribe-diarize` using the official `OpenMOSS-Team/MOSS-Transcribe-Diarize` checkpoint through mlx-audio 0.5.1 and the existing MLX engine. Defaults remain unchanged.
- Require explicit English (`en`, `en-US`, `en_US`, `eng`, or `English`). This is the validated input scope; upstream language prompting/detection is not claimed.
- Accept one continuous-speech recording up to 1,800 seconds. Reject longer inputs before inference, without automatic splitting or cross-request speaker matching.
- Use the tested upstream defaults with greedy generation, 32,768 output tokens, and a 4,096 prefill step. Limit worker inference wall time to 900 seconds with a process-local OS alarm. A deadline terminates the isolated worker; existing liveness handling fails pending jobs and permits the next request to reload. Model startup retains the service's existing startup timeout.
- Preserve upstream segments, normalize `speaker_id` to `speaker`, strip speaker markup from segment text, and construct top-level text from validated segments. Support JSON, text, and SRT through existing API formats. IDs identify speakers only within this recording.
- Reject exhausted budgets, unparsed/trailing raw output, absent speaker labels, invalid or unordered timestamps, and uncovered spans over 10 seconds. The gap check is conservative: legitimate long silence may also be rejected. It does not detect every omission or certify transcription accuracy.
- Clamp endpoint overruns of at most 0.25 seconds to the input duration (the accepted medium probe overshot by 0.097 seconds); reject larger errors.
- Invalid MOSS input returns HTTP 400; structurally incomplete output returns 422. An inference deadline or other worker/runtime failure follows existing HTTP 500 handling. Temp artifacts and models retain existing subprocess cleanup/idle behavior.
- Keep output validation as a pure helper, with no new inference engine, model internals, speaker reconciliation, or retirement of existing aliases. Test the public contract, real API behavior, and failure/recovery path.

Thirty minutes is a provisional operating policy supported by two disjoint English samples, not a universal model maximum. The official longer-duration claim does not supersede local 40/60-minute truncation evidence. Numerical WER/CER and speaker accuracy remain unmeasured.

The MOSS checkpoint is pinned to `704aa4a9c304e8520be88901e0d1960158ef5b15`, the evaluated revision. Upstream output files live under the request temporary directory so parent cleanup removes them even after worker termination.
