# Explicit cloud diarized transcription

Status: owner-approved contract; Deepgram implemented with offline validation; independent review and live acceptance pending.
Date: 2026-10-29.
Inspected base: `d968b82fef1cc370df2742354f22dc5aa2ac9c5e`.

The owner confirmed explicit cloud model selection, combined transcription and
speaker labels, retention of local models, and no automatic provider fallback.
No production code, credentials, dependencies, or running service were changed
while preparing this proposal. Provider behavior below is documentation evidence,
not a successful API test or a comparative quality result.

## Scope and rationale

Extend the existing lightweight gateway with three thin file-transcription
adapters, in priority order: Deepgram, OpenAI, Sonix. Callers keep uploading to
`POST /v1/audio/transcriptions`; a cloud alias selects the remote provider.
Each provider performs transcription and diarization together. Do not combine
local ASR text with cloud speaker turns or revive the retired local pipeline.

This extends ADR-002's local-only product scope while preserving its thin-adapter
boundary. Record the reviewed scope amendment in ADR-002 during implementation.
Preserve SPEC-016's immutable request selection and completion identity.

## Existing seams and constraints

- `src/core/model_registry.py:23`: ModelSpec is the alias/capability source of
  truth. Its engine discriminator currently admits only local runtimes.
- `src/api/routes.py:53`: Segment already carries speaker, start, end, and text;
  TranscriptionResponse also carries duration, language, and executed model.
- `src/services/execution.py:20`: ExecutionResult owns response identity. Keep
  selecting and validating one immutable plan per request.
- `src/services/transcription.py:108`: submit owns request-local upload files and
  dispatch. Add the remote branch before resident-worker dispatch.
- `src/services/transcription.py:295`: resident switching closes the previous
  worker. Cloud requests must bypass this switch and WorkerSession entirely.
- `src/api/routes.py:342`: discovery currently marks every entry requestable.
  Cloud readiness needs a real credential/acceptance gate.
- `pyproject.toml`: httpx is currently a development dependency. Promote it through
  uv and update the lockfile if chosen for production async HTTP.
- `tests/unit/test_architecture_fitness.py` and `tach.toml` gate adapter files and
  dependencies. Update them with the reviewed architecture, without relaxing
  existing complexity or local lifetime checks.

The existing untracked Nemotron proposal and transcript.json belong to other
work and must remain untouched.

## Public contract

Proposed aliases, registered centrally:

| Priority | Alias | Upstream selection | Initial gate |
| --- | --- | --- | --- |
| 1 | `deepgram-nova-3-diarize` | Nova-3 + fixed batch diarizer v1 + utterances | Enable after real-media acceptance |
| 2 | `openai-transcribe-diarize` | gpt-4o-transcribe-diarize + diarized_json + auto chunking | Enable after acceptance; disclose scheduled retirement |
| 3 | `sonix-transcribe-diarize` | Sonix file transcription and JSON export | Hold until automatic single-track speaker clustering is verified |

Usage after implementation and activation:

```bash
curl http://127.0.0.1:50700/v1/audio/transcriptions \
  -F 'file=@meeting.m4a' \
  -F 'model=deepgram-nova-3-diarize' \
  -F 'language=en' \
  -F 'response_format=verbose_json'
```

Reuse existing JSON, text, and SRT behavior. Normalize provider output into typed
segments before rendering; return the executed alias through ExecutionResult.
Speaker labels are anonymous, recording-local identifiers such as `Speaker 0`.
Preserve overlap; never assign identity across recordings or independent uploads.
Do not synthesize labels when upstream omits them. Validate finite timestamps,
positive segment durations, text types, and speaker presence for speech output.
Valid silence may return an empty transcript without invented speakers.

Keep current local aliases and their switching semantics. Explicit cloud calls
do not change `/v1/models/current`, spawn/unload local models, or become the model
used by omitted/empty/whisper-1 passthrough requests. In version one, cloud aliases
are per-request only; reject them as startup MODEL_ID with actionable guidance.
No new public streaming API or default-provider setting is included.

Extend discovery additively with execution location/provider, configured status,
and a bounded unavailable reason. Compute requestable from configuration and
acceptance policy; key presence alone is not proof of availability or quality.
List cloud aliases without network requests, model loads, or secret disclosure.

## Provider boundaries

Deepgram: use `/v1/listen`, `model=nova-3`, `diarize_model=v1`, and utterance
output. Pin v1 following the dated fixture comparison below; the original v2
proposal is superseded for this slice. Do not follow latest.
Map speaker numbers to strings and retain word timestamps within their spoken
segments. Do not use deprecated diarize=true. Resolve explicit language and auto
language detection using the provider's actual language matrix; reject unsupported
combinations before upload. Do not assume Chinese-English code-switching support
merely because both languages are individually supported.

OpenAI: use only the documented diarization model for this alias. Neither
gpt-transcribe nor gpt-live-transcribe is a demonstrated equivalent. Current
official docs schedule the old model's shutdown for 2027-02-26; do not silently
replace it with an ASR-only model. Revalidate availability at implementation and
acceptance, and disable selection when retired. The documented 25 MB upload limit
is stricter than this gateway's 200 MiB default. Version one rejects oversized
inputs before submission rather than splitting and concatenating speaker IDs.

Sonix: submit media, retain its returned job ID, poll status with bounded backoff,
then fetch transcript.json. A response containing speaker fields does not itself
prove automatic single-track clustering: older official help says manual labeling
is needed, while the current API page advertises diarization. Verify an unedited
single-track recording before promotion. Multitrack channel attribution is a
separate input mode, not proof that mixed voices can be separated automatically.
Use direct multipart uploads within its documented 100 MB limit; do not publish
temporary public URLs to work around larger uploads. Map language codes explicitly
(for example Mandarin cmn), using the provider's documented support.

For all providers, enforce both the gateway and upstream size/format limits.
Reject unsupported inputs before upload. Do not truncate or independently split
long recordings in version one; cross-chunk speaker reconciliation is out of scope.
Document measured duration coverage separately from upload size limits.

## Remote execution and ownership

Add a typed async remote transcription port with normalized results and stable
error categories. Provider adapters own authentication, HTTP schemas, polling,
language mapping, and normalization. A service-owned remote session owns bounded
admission, active tasks, deadlines, cancellation, and client closure. Reuse the
existing upload/ExecutionPlan/ExecutionResult boundary, not synchronous ASREngine
load/release methods or multiprocessing jobs.

Proposed initial cloud concurrency is one active upload/job across providers;
waiting and active cloud requests count toward the existing total queue bound.
Acquire capacity atomically and recheck shutdown at admission. A semaphore alone
must not allow an unbounded waiting queue. Waiting requests have a deadline too.
Keep the local single-worker memory rule; cloud network work needs no ML worker.

Set explicit connect, upload, read, and total deadlines, plus a response-size bound.
Proposed total deadline: 1,800 seconds, configurable and bounded; check caller
timeouts before activation. Sonix polling stays within the same total deadline.
No automatic retries of paid submissions or cross-provider fallback in version
one. Polling an existing Sonix job is not resubmitting it. Ambiguous submission
timeouts must report uncertainty rather than cause duplicate billing.

On disconnect/shutdown, stop local upload/poll work, await owned task disposal,
and close clients before deleting local files. Cancellation cannot guarantee
remote work or billing stopped. Persist only minimal recoverable job evidence
(provider, local request ID, remote job ID when known, status), outside disposable
upload folders; never log credentials, transcript text, or full upstream bodies.
Define the evidence retention location before the first real API probe. Do not
automatically delete remote media or claim that local cleanup removed it.

Proposed gateway errors: invalid input 400, oversized input 413, unavailable local
configuration/admission or upstream rate limiting 503, upstream authentication or
service failure 502, invalid upstream output 502, and execution deadline 504.
Return sanitized messages and retry guidance without provider response bodies.

Credentials stay server-side (DEEPGRAM_API_KEY, OPENAI_API_KEY, SONIX_API_KEY),
never in form fields, discovery, logs, or commits. Explicit alias selection is the
cloud upload boundary; local/default requests remain local. Do not accept arbitrary
provider endpoints or follow cross-origin redirects carrying authorization.

## Delivery and acceptance

1. Review this contract and record the narrow ADR-002 scope amendment.
2. Implement the common typed cloud boundary, discovery/readiness, and Deepgram
   adapter first. Add OpenAI and Sonix independently against that same port.
3. Cover public HTTP contracts, provider normalization, no cloud side effects on
   local current/default, unsupported-language/oversize preflight rejection, queue
   overflow, cancellation, shutdown, invalid output, and ambiguous timeout paths
   with deterministic local HTTP fakes. No paid model calls in unit tests.
4. Run CI commands: `uv run ruff check .`, `uv run tach check`, and
   `uv run pytest tests/unit tests/reliability/test_worker_session_process.py`.
   Also run affected integration tests and appropriate local-model regression
   acceptance. This production-code change is not the offline test-only lane.
5. Probe each real provider using a designated non-sensitive sample and configured
   credentials. Include at least two speakers, target language(s), short overlap,
   and a representative long recording. Record latency, billed usage where
   available, transcript completeness, and speaker stability. Human-reference
   scoring is required for DER claims; listening acceptance remains explicit.
6. Promote only providers that pass their real-media and failure/lifetime gates;
   otherwise retain discovery-only status with the exact missing evidence.
7. Deliver reviewed code/PR and evidence. Activation of the running service and
   service restart are distinct later actions. No restart is implied by this plan.

## Official sources inspected

- [Deepgram diarization](https://developers.deepgram.com/docs/diarization)
- [Deepgram file API](https://developers.deepgram.com/reference/speech-to-text/listen-pre-recorded)
- [Deepgram file limits](https://developers.deepgram.com/docs/pre-recorded-audio)
- [OpenAI file transcription](https://developers.openai.com/api/docs/guides/speech-to-text)
- [OpenAI retirement notice](https://developers.openai.com/api/docs/deprecations#2026-08-26-transcription-models)
- [Sonix API overview](https://sonix.ai/api)
- [Sonix API reference](https://sonix.ai/docs/api)
- [Sonix older single-track help](https://help.sonix.ai/en/articles/1980122-how-does-sonix-handle-multiple-speakers)

## Deepgram delivery status (2026-10-29)

The initial slice implements only Deepgram behind `DEEPGRAM_ENABLED=false`.
OpenAI and Sonix remain future scope; no adapters or credentials for them are
introduced. The four-module boundary and narrow Tach/fitness additions are
recorded in ADR-002. Local defaults and passthrough remain local.

Official language documentation was checked during implementation:
[Models and languages](https://developers.deepgram.com/docs/models-languages-overview)
includes explicit English `en` and Mandarin `zh` for Nova-3. This slice accepts
only those two exact language values. Broader language modes, automatic language
detection, and Chinese-English code-switching are not enabled or quality-verified.

The HTTP adapter has connect=10s, upload inactivity=60s, read inactivity=660s,
pool=10s, and a queue-inclusive total deadline (default/max 1800s, minimum 1s).
Responses are bounded to 16 MiB and compressed responses are rejected. Uploads
remain bounded to the lower of the configured gateway limit and 200 MiB; no
splitting or truncation is performed. Supported input extensions are WAV, MP3,
M4A, MP4, FLAC, OGG, and WebM. Corrupt audio is left to provider validation;
preflight checks extension, nonempty size, language, and output mode.

The initial implementor made no paid call, activation, or service restart.
Subsequent coordinator probes are recorded below; gateway v1 and broader live
acceptance remain pending. A valid key alone does not demonstrate acceptance.
For live probes, retain
service logs containing UUID-only provider/local request correlations under
`pr-reviews/deepgram-20261029/` outside disposable uploads. Record the sample,
status, request IDs and billed usage there without credentials or transcript
content; absence of a provider request ID is not evidence no charge occurred.
The independent reviewer must assess the fixed commit, and remaining real-media
acceptance must pass before operator enablement. This is a production change,
not the offline test-only acceptance lane.

### Initial implementation validation

- Required CI lane: Ruff and Tach passed; unit plus real worker-process lifetime
  tests: 427 passed. Existing assertions and thresholds were preserved.
- All integration tests plus concurrency regressions: 59 passed.
- Targeted mypy: the four new production modules, changed service/routes, and the
  new behavioral tests passed (seven source files).
- Existing Paraformer silence HTTP/process E2E passed using task-private clones
  of cached weights, an external-network guard, and a task-only ModelScope
  `local_files_only` wrapper. No model download or production restart occurred.
  The first attempt failed because ModelScope tried to create a lock in the
  sandbox-protected shared cache. This smoke test establishes local execution,
  not diarization quality or long-audio acceptance.
- `httpx` was promoted to production at the existing locked version 0.28.1.
  `uv lock --check --offline` passed with 149 packages; no package versions changed.
  Fresh `uv sync --frozen --dev` failed sandbox DNS while fetching llvmlite.
  Tests used a task-private uv environment with read-only access to existing
  installed packages. Checked HTTP/framework/test/runtime package versions match
  the lock; this is not a successful fresh-install claim.

These checks describe the initial implementation snapshot. Independent review
subsequently found no blocking static/diff defect and requested a CHANGELOG entry.
The parameter amendment below requires its own fixed-head review and gateway v1
probe. Remote CI, broader quality/duration scoring, billed-usage assessment, and
operator activation remain pending.


### Fixed v1 repair evidence (2026-10-29)

The owner authorized up to five coordinator calls and confirmed the fixture has
two voices by listening. The implementor makes no API calls in this repair.
All three existing probes used `tests/fixtures/two_speakers_60s.wav` (60.0s),
SHA-256 `a44ad63de6d62d4aad1e562c765ff8c1e82bba3743054bf973ca1e06c7c49a19`.
Call 1 through the v2 gateway returned HTTP 200 with 14 utterances, all speaker 0.
Direct provider calls 2 and 3 compared the same bytes:

| Evidence | Requested/actual diarizer | Word labels | Speaker transition |
| --- | --- | --- | --- |
| `diagnostic-02.summary.json` / `.raw.json` | v1 / arch v1, UUID `a9f85c2b-5afb-4b9d-b49b-492c43a01cfb` | 0: 28, 1: 125 | 13.599999s |
| `diagnostic-03.summary.json` / `.raw.json` | v2 / arch v2, UUID `6946b038-8264-4505-acf7-9822c009fecb` | 0: 153 | None |

Evidence lives in `pr-reviews/deepgram-20261029/` outside the disposable worktree.
Both raw responses report ASR `general-nova-3` version `2025-07-31.0`. The
implementor checked summary/raw agreement, fixture hash, and both full responses
through the unchanged normalizer: v1 retains 14 utterances with two labels and
its original float offsets; v2 retains 14 speaker-0 utterances without invention.
A minimal sanitized v1 shape now guards the two labels and transition offset.

The coordinator's bounded disposition is fixed `diarize_model=v1` for this slice.
This supersedes the original v2 proposal because v2 missed the actual required
two-speaker case; it is not a general policy change or a global quality ranking.
No alias, architecture, fallback, tuning interface, or local behavior changes.
Default enablement stays false. Gateway v1 validation is still pending and is
reserved for the coordinator's next authorized call (call 4) after this commit.
English/Mandarin quality, overlap and long-audio coverage remain limited or
unverified; no DER or broad speaker-accuracy claim follows from this fixture.

The prior cached one-second local E2E remains applicable: this repair changes
only the remote diarizer request parameter, its description/test expectation,
and documentation. It does not change native inference, admission, lifetime, or
rendering. Repeating native inference would not validate the changed parameter;
request-contract and normalization regressions plus gateway v1 acceptance cover
that behavior. Existing local source/tests and protected policy are unchanged.
