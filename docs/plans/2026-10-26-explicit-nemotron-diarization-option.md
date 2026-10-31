# Explicit experimental Nemotron diarization option

Status: implementation authorized by the owner on 2026-10-26. Current production
service/environment activation remains excluded. Empty optional transcription
selection means disabled; only standalone execution requires an explicit alias.

## Intended behavior

Make Nemotron discoverable and manually selectable for standalone **who spoke
when** experiments. Every request must explicitly select it. Ordinary
transcription requests must never infer or inherit this diarizer. Preserve the
configured ASR default and the service's logical current transcription model. Existing native Paraformer/CAM++ and MOSS speaker
output remains unchanged; the opt-in guarantee specifically concerns Nemotron.

PR #58 retained the pinned model reference and offline evaluation tools. The
recorded quality decision remains hold for automatic speaker-attributed
transcription. The owner now proposes an explicitly selected experimental HTTP
entry point; this does not mean the model passed quality acceptance.

## Current code constraints

- `src/api/routes.py:76-91,342-368`: `models` contains ASR selections; documented
  aliases are accepted by `/v1/audio/transcriptions`. Do not insert a pure
  diarizer into that array and imply it transcribes text.
- `src/core/model_registry.py:129-131`: omitted ASR model values mean current
  model, not necessarily startup default. Nemotron must never become this state.
- `src/core/diarization_registry.py:7-29` and `src/core/diarization_port.py:25-30`
  already provide independent diarizer metadata and a typed adapter boundary.
- `src/workers/model_worker.py:73-81`: the existing diarizer factory chooses
  Sortformer for every MLX diarizer. Adding a registry row alone is incorrect.
- `src/workers/model_worker.py:174-181`: the existing resident worker eagerly
  loads an ASR engine before accepting any diarization job. Do not reuse that
  startup path and accidentally load ASR plus Nemotron.
- `src/services/transcription.py:108-177,201-226`: resident and explicit Apple
  sidecar admission take different paths. Both must honor experiment exclusion.
- SPEC-016 retired production transcription/alignment/diarization orchestration
  and borrow/restore logic. This proposal does not reinstate them.

## Discovery contract

Keep existing `/v1/models.models` entries and `current` semantics unchanged. Add
one separate, additive `diarizers` array to the same response, visible in Swagger.
The example shows only the new field; existing `models` and `current` remain:

```json
{
  "diarizers": [{
    "alias": "nemotron-diar",
    "model_id": "mlx-community/Nemotron-3-Diarization",
    "task": "diarization",
    "experimental": true,
    "requires_explicit_selection": true,
    "endpoint": "/v1/audio/diarizations",
    "max_speakers": 8,
    "weights_cached": true
  }]
}
```

The values above illustrate a response, not the currently running API.
`weights_cached` reflects local presence of the pinned snapshot; listing never
loads ML frameworks, downloads weights or changes model selection. It is not a
quality or runtime-readiness guarantee. Only Nemotron is newly exposed; retained
Sortformer internals do not automatically become public choices.

This service change exposes API/Swagger discovery. A separate caller's existing
ASR dropdown will not automatically display a new diarizer group; any such UI
adaptation needs its own concrete caller change. Validate known in-repo/client
schema consumers before adding the response field rather than assume strict
clients accept it.

## Explicit request and response

Add `POST /v1/audio/diarizations`, multipart upload:

| Field | Contract |
| --- | --- |
| `file` | Required; reuse supported upload validation and request-owned disk files |
| `diarizer` | Required exact alias `nemotron-diar`; no default, empty value, `auto`, current-model passthrough or implicit selection |

Example after implementation and manual activation:

```bash
curl http://127.0.0.1:50700/v1/audio/diarizations \
  -F 'file=@episode.m4a' \
  -F 'diarizer=nemotron-diar'
```

JSON response:

```json
{
  "task": "diarization",
  "diarizer": "nemotron-diar",
  "model_id": "mlx-community/Nemotron-3-Diarization",
  "revision": "59ed2dbfc1346dcea9d423c71306a3a2499c568f",
  "experimental": true,
  "duration": 120.0,
  "num_speakers": 2,
  "turns": [
    {"speaker": "Speaker 0", "start": 1.2, "end": 4.6},
    {"speaker": "Speaker 1", "start": 3.8, "end": 5.1}
  ]
}
```

Preserve overlapping turns and anonymous, recording-local identities. Do not
return fabricated text, force one speaker per time interval, concatenate fresh
per-chunk IDs, identify real people, or compute DER without a human reference.
Output identity comes from this request's immutable selection, never ASR state.

This initial API returns JSON speaker turns, not a speaker-tagged transcript,
SRT, combined Qwen result, background job system or streaming HTTP response.
The existing offline tools remain available for RTTM/probability inspection.

Invalid selection fails before worker startup. Missing standalone selection returns 422; empty/auto/unknown aliases return 400. Known pure diarizer aliases/model IDs passed
as ASR `model` return 400 with the correct endpoint, before generic path fallback.
A nonempty `diarizer` form field on `/v1/audio/transcriptions` returns 400;
missing, None and empty/whitespace values mean disabled and preserve ordinary ASR.
it must not be silently ignored or launch an implicit two-stage workflow.

## Resource admission: recommended first version

Use fail-fast experimental admission, not a new general scheduling framework:

1. Acquire a short shared admission lock and check no active/queued ASR or Apple
   sidecar request, no live resident ASR process, no other experiment and no shutdown.
2. If busy, return 503 with `Retry-After` and a stable `service_busy` reason.
   Never interrupt a running job, force ASR offload, borrow the resident worker,
   or silently change its selected model. Wait for normal idle offload instead.
3. Atomically claim an exclusive experimental lease before releasing the lock.
   All resident, passthrough and explicit Apple admission paths must check this
   lease under the same admission boundary. A preflight-only flag is racy.
4. While the experiment holds the lease, new ASR/Apple/diarization requests return
   the same retryable 503. Already admitted ASR work is never cancelled to make room.
5. One request-owned isolated diarization child loads only Nemotron. A dedicated
   session owns spawn, completion, timeout, cancellation, signals and reap. HTTP
   code never owns raw subprocess handles; the gateway imports no MLX inference.
6. Keep the lease through cancellation and cleanup. Only release it after the
   child is reaped; disposal failure retains ownership and blocks a replacement.
7. Leave the logical ASR selection untouched. The next ASR request loads that
   same selected ASR model normally; no proactive restore step runs.

The admission rule must track native work, not just HTTP waiter count: a cancelled
Apple `to_thread` waiter does not prove its native sidecar exited. Do not declare
resources idle until the owning runtime confirms disposal.

This deliberately affects availability while a manually selected experiment
runs. It avoids a second model competing for memory and avoids reintroducing a
pipeline scheduler. With `MODEL_IDLE_TIMEOUT_SEC=0`, an already resident ASR
process will not naturally offload; the experimental request stays busy rather
than implicitly unloading it. If concurrent use or automatic handoff is required,
that is a separate design decision, not hidden scope in this change.

Proposed guards: one model process, sampled available-memory reserve of 20 GiB,
owned RSS/observed MLX allocation termination threshold of 8 GiB, and 900-second
execution deadline. Reuse the PR #58 semantics and tests where appropriate;
`set_memory_limit` remains advisory and samples can overshoot. Do not import a
benchmark CLI wholesale into the gateway; extract only the needed owned-process
safeguard implementation and retain the standalone entry point.

Decode supported formats into request-owned mono 16 kHz PCM16 without changing
originals. Reuse the configured upload limit (`MAX_UPLOAD_SIZE_MB`, currently
200 MiB). Proposed decoded-audio disk budget: 512 MiB PCM16, approximately
4 h 40 min at this format; return an explicit input-limit error rather than fill
unbounded disk or truncate. Do not introduce a separate arbitrary 60-minute
limit. The ASR 50-minute splitting threshold is not a diarization recipe: preserve
one continuous identity state across native chunks. The real long-form evidence
currently covers 39 minutes, so test a representative 2–3 hour input/resource
case in isolation before claiming full-episode support. These are proposed
resource limits, not established accuracy guarantees.

## Model adapter and failure behavior

- Register typed experimental metadata separately from ASR ModelSpec. Add an
  explicit backend discriminator to the existing diarizer spec/factory; runtime
  `mlx` alone cannot choose between Sortformer and Nemotron.
- Implement `MlxNemotronDiarizer` against `DiarizationPort` using the pinned local
  snapshot above and native continuous-state whole-file inference. Harden turn
  validation for finite, nonnegative, positive-duration, in-range intervals.
- Use the cached snapshot only. Request handling does not install packages or
  fetch weights. Missing snapshot returns 503 `model_unavailable` with setup
  guidance; metadata remains discoverable. Model license remains OpenMDW-1.1.
- Default postprocessing stays threshold 0.5, min duration 0 and merge gap 0.
  No public tuning parameters in the first API; retain tuning in offline tools.
- Deadline returns 504 after cleanup; malformed output or worker failure returns
  502; insufficient resources returns 503. Do not return partial success/HTTP 200
  for crashed or invalid output. Empty/silent valid input may return zero turns.
- Uploaded/decoded temporary files are request-owned and removed after the worker
  finishes disposal, including failure, disconnect and shutdown paths.

## Implementation slices after owner confirmation

1. Typed metadata/discovery and Nemotron adapter/factory discrimination; unit tests
   for cached revision, no import/load during discovery and independent backends.
2. Request-owned diarization lifetime plus shared nonpreemptive admission; real
   lightweight subprocess tests for cancellation, SIGTERM, shutdown and no overlap
   with resident or Apple work. No ML inference in unit tests.
3. HTTP endpoint, schema/OpenAPI descriptions, errors and request-owned upload
   cleanup; focused integration tests and compatibility checks for existing models.
4. Isolated real-model smoke/long-boundary verification, documentation and a PR.
   Show exact diff/test/CI evidence. Activation remains a later manual owner step.

Keep these changes one bounded standalone-diarization capability. Do not alter
ASR switching semantics, revive `qwen3-sortformer`, add forced alignment, or build
speaker-attributed transcription in these slices.

## Acceptance checklist

- Discovery exposes Nemotron as experimental diarization without changing ASR
  entries/current/default; discovery causes zero model loads/downloads.
- Only an explicit valid `diarizer` on the new endpoint can start Nemotron.
  Requests before and after it do not inherit the choice.
- Missing/blank/auto/wrong-field/wrong-endpoint selections fail before inference.
- Ordinary Qwen/Paraformer/native diarization contracts remain compatible.
- Concurrent ASR, Apple sidecar, experiment, cancellation and shutdown schedules
  cannot run a second native model while the experimental lease is active.
- ASR busy/offload-disabled cases return retryable errors, never preemption or a
  hidden resident-model switch. Selection stays unchanged on success and failure.
- Single file, silence, overlap and invalid native output produce the documented
  response behavior; returned model/revision match actual immutable execution.
- No orphan worker, leaked upload, early lease release or reuse before disposal.
- Frozen-lock isolated validation passes; production environment/service untouched.
- Real-program listening and quality acceptance remain owner follow-up, not a
  claim made by green software tests.
