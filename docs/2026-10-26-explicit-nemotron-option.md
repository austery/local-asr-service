# Explicit experimental Nemotron option: delivery evidence

Scope: owner-approved standalone speaker activity, discoverable separately from
ASR, explicitly requested on every inference. Ordinary transcription selection
may be absent/None/empty/whitespace and stays disabled. No production restart,
primary checkout synchronization or primary environment update was performed.

See the [approved plan](plans/2026-10-26-explicit-nemotron-diarization-option.md)
and [prior quality report](plans/2026-10-26-nemotron-diarization-results.md).
Quality acceptance remains separate from this bounded experimental entry point.

## Implemented behavior

- `/v1/models` keeps ASR `models` and `current`, adds `diarizers` with experimental
  metadata, explicit-selection requirement and pinned-cache presence. Listing does
  not import MLX/PyTorch models or download weights. Only Nemotron becomes public;
  retained Sortformer adapters remain internal controls.
- `/v1/audio/diarizations` requires `diarizer=nemotron-diar`. Return validated
  overlapping speaker turns and request-owned model/revision identity, no text.
  Missing/empty behavior was tested against the actual multipart API: missing
  returns 422, explicit empty returns 400. Auto/unknown/hidden aliases return 400.
- ASR rejects nonempty diarizer choices and known pure diarizer model IDs early.
  Empty choices preserve ordinary transcription and native speaker functionality.
- Shared short admission protects resident and Apple paths. A standalone request
  only claims an exclusive lease after all native ASR work is idle/offloaded.
  Native Apple work retains its ownership despite HTTP cancellation. Cleanup must
  finish before another model is admitted, including repeated cancellation.
  A body-consumed ASGI disconnect listener cancels and joins the owned operation;
  it does not rely on nonblocking disconnect polling through BaseHTTPMiddleware.
  Regression coverage includes a real worker that ignores SIGTERM until escalation.
  Already exited resident workers are settled through WorkerSession before
  experimental startup; disposal failure keeps the experiment busy.
- The dedicated child decodes request-owned audio and loads only the cached
  pinned Nemotron snapshot. Process-group cleanup also covers decode descendants.
  Timeout, malformed output, missing cache, resource pressure and decoded-budget
  errors fail explicitly instead of returning partial success.

CI runs unit, mocked HTTP integration and lightweight reliability suites,
including the actual ASGI disconnect/process regressions. Real-model E2E
remains isolated local acceptance rather than requiring weights in GitHub CI.

The existing resident ASR worker cannot simply host this request: it eagerly loads
ASR at startup, and the old MLX diarizer factory selected Sortformer for every
backend. The implementation adds explicit backend discrimination and an isolated
request lifetime; it does not restore the retired pipeline or a model-restoration
state machine. The allocator monitor is shared with the offline probe instead of
importing the benchmark CLI into the gateway.

## Execution evidence

Validation uses a separate frozen-lock uv environment, mlx-audio 0.6.1/MLX 0.32.4,
existing cached weights and copies of existing inputs. No dependency change.

| Check | Observed result |
| --- | --- |
| Complete pytest suite, including existing real-model E2E | 472 passed in 61.37 s |
| Focused API/admission scenarios | 30 passed |
| Owned-process tests (real lightweight children) | 10 cases, covered in the complete run |
| Ruff, Tach, mypy for new boundaries | Passed |
| Real 60 s English input through isolated ASGI HTTP | 200, two speakers, ~2.48 s total |
| Repeated AMI M4A, 9,381.9735 s (~156 min) | 200, five identities, ~16.55 s total |

Both real requests preserve the full pre/post model-discovery response, keep
`current=paraformer`, and never load the replica's resident ASR worker. The probe
continuously checks the actual running server's identity/children and cancels
its owned operation if production inference starts. It does not send inference
to the running server or open a second listening port.

The long input repeats the same public AMI ES2004b recording four times and encodes
it to AAC/M4A. It tests decode, input duration, continuous inference and resource
limits. Five identities on that construction is **not** a quality pass. It is
neither a natural 156-minute meeting nor an evaluation of the owner's podcasts.
The unmodified public AMI source and prior manual references remain retained.

Private raw results and logs: primary checkout
`.runtime-validation/nemotron-option/`. Relevant files include
`real-http-short-disconnect-fixed.json`, `real-http-long-disconnect-fixed.json`, their logs and
`pytest-full-1793047303557786000.log`. Private inputs are not committed.

## Compatibility and limits

The known PureSubs selector uses ASR aliases (`paraformer`/`qwen3-asr`); its
frontend DTO uses a separately assembled local/cloud list. This PR does not
adapt that UI or claim Nemotron appears in its existing ASR dropdown. The new
service capability group is available to explicit clients and Swagger. Existing
ASR discovery and request contracts retain their integration coverage.

This remains single-process gateway admission, not a cross-process GPU lock.
Keep one server worker. Host memory and MLX thresholds are sampled; external
programs can consume resources between samples. With idle offload disabled, a
resident ASR process keeps standalone requests busy. Cancellation/shutdown waits
for actual native disposal; disposal failure retains ownership and prevents reuse.

Current model quality gaps, missing independent Chinese/bilingual references,
clean two-speaker quality acceptance and final speaker-attributed WER remain
unresolved. No Qwen alignment/tagging workflow is added, and no real-program
listening acceptance is claimed.
