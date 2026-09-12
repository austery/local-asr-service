---
specId: SPEC-016
title: Runtime Module Deepening
status: Ready for Implementation
priority: P3 - Quality
creationDate: 2026-09-11
lastUpdateDate: 2026-09-12
relatedSpecs:
  - ADR-002
  - SPEC-009
  - SPEC-015
---

# SPEC-016: Runtime Module Deepening

## Goal

Reduce the knowledge callers and tests need about model selection, execution,
and cleanup while preserving the lightweight speech gateway scope.

## Baseline and authorization

The user authorized a plan followed by ordered implementation in a separate
worktree. The base is `main` at `d4104f0`; the first delivery uses
`refactor/speech-runtime-seams`. The main checkout and its untracked
`transcript.json` are outside this change. Merge and deployment require separate
approval.

The public pipeline registry is empty following the approved retirement of
`qwen3-sortformer`. Nevertheless, production request handling still supports
synthetic pipeline profiles, alignment/diarization orchestration, a pipeline
reservation lock, and resident-model restoration.

## Ordered delivery plan

Each phase is independently reviewable. The first PR covers Phase 1 only;
subsequent phases start from the resulting simpler runtime and require their
own concrete design and acceptance evidence before implementation.

### Phase 1: Retire production pipeline orchestration

- [x] Remove `submit_pipeline` and its orchestration-only helpers from the live
  transcription Module, together with unreachable HTTP pipeline dispatch.
- [x] Preserve active aliases, native FunASR/MOSS diarization, the empty historic
  registry, independent alignment/diarization adapters, worker job support, and
  pure algorithm tests. Retain historical specifications and probe reports.
- [x] Replace the already-broken long-form pipeline probe with an explicit
  retirement message and a pinned historical source reference. It must exit
  before reading media, loading models, or writing output.
- [x] Replace pipeline reservation with the existing spawn lock for atomic
  passthrough resolution and enqueue. Release that lock before awaiting inference;
  retain the separate Apple sidecar concurrency limit.
- [x] Remove tests exclusively asserting deleted orchestration behavior; preserve
  and extend active dispatch, queue capacity, cancellation, switch failure, and
  temp-file cleanup coverage. Record the test-count change explicitly.
- [x] Add an architectural guard against importing retired orchestration concerns
  into production request handling. Update current guidance to distinguish the
  live gateway from retained experimental building blocks.

**Acceptance:** active model discovery and HTTP errors remain compatible;
`qwen3-sortformer` returns 400 before submission; worker and sidecar dispatch,
queue rejection, cleanup, and model switching regressions pass. Ruff, Tach, and
the complete unit/integration/reliability suites pass on this worktree. Real-model
E2E remains a separate resource-intensive acceptance gate, not inferred from mocks.

**Rationale:** Moving the entire experiment behind another production Interface
would retain a capability with no requestable profile. Retiring the orchestration
directly removes its lock/restore obligations from every active caller. Independent
runtime adapters remain available for isolated evaluations; no generic model
orchestration framework is introduced.

### Phase 2: Resolve each request's execution model once

- [x] Reproduce whether admission-time capability checks and enqueue-time model
  selection can diverge under concurrent switching.
- [x] Specify passthrough timing explicitly, including switch failure and requests
  already queued. Decide between an admission snapshot and enqueue-time resolution
  before introducing an immutable execution plan.
- [x] Make capability validation, execution, and response model identity share that
  plan. Preserve early validation and custom-model behavior.

**Acceptance:** a deterministic concurrent test proves the selected model,
validated capabilities, and returned identity agree; invalid requests never reach
inference. This phase must not silently change the model-switch policy.

### Phase 3: Deepen worker lifecycle ownership

- [x] Introduce an internal worker-session Module that owns startup, IPC, exit,
  pending completions, and resource release; inject the process transport for tests.
- [x] Consolidate cleanup for normal exit, startup failure, timeout, and crash.
- [x] Keep Apple sidecar lifetime distinct; share an execution Interface only where
  both production adapters have a real common contract.

**Acceptance:** tests through the session Interface cover startup failure,
cancellation, switching, crash, idle exit/restart, and complete process/queue
reclamation. Real subprocess tests use a lightweight test worker, not ML models.

### Phase 4: Tighten transcription contracts and fitness gates

- [ ] Introduce typed options, normalized results, and stable error categories at
  the runtime Seam; normalize upstream differences inside each Adapter.
- [ ] Replace tests coupled to removed private state with behavioral tests through
  that Seam, retaining every active regression scenario.
- [ ] Narrow per-file complexity suppressions after the preceding changes reduce
  the actual obligations. Do not add pass-through Modules merely to satisfy counts.

**Acceptance:** typed errors retain HTTP status compatibility, model-specific
formatting and limits remain unchanged, and refactoring an Implementation no longer
requires updating caller tests that inspect its locks or dictionaries.

## Delivery evidence

Phase 1 merged in PR #37. The following table records its validation.
Phase 2 is implemented in a separate worktree; its validation is recorded below.
Phases 3–4 remain planned, not implemented.

| Check | Result |
|---|---|
| Baseline unit/integration/reliability at `d4104f0` | 353 passed |
| Worktree unit/integration/reliability | 324 passed |
| Full worktree suite, including real Paraformer silence E2E | 325 passed in 30.77 seconds |
| Ruff | Passed |
| Tach | Passed |
| `git diff --check` | Passed |
| `uv build --no-build-isolation` | Wheel and source distribution built |

Test accounting: 35 tests exclusively covering removed pipeline orchestration
were retired; six cases were added for cancellation, admission/overflow, enqueue
failure, and the retired probe. The architecture scope test was replaced one for
one. Existing active model tests and the independent algorithm/adapter tests are
retained. The E2E run emitted the existing unregistered `e2e` marker warning.

The worktree reused the existing dependency environment via an ignored `.venv`
symlink, with `uv run --no-sync`; no dependency or lockfile update was made. The
E2E used its own TestClient and model subprocess, not the running HTTP server.
This one-second silence test establishes the real process/HTTP path only; it does
not establish long-audio quality, Apple sidecar runtime acceptance, or production
load behavior.

A separate second review pass checked the final diff against the Phase 1
requirements: native diarization and model-specific deadlines are unchanged;
worker selection stays inside the spawn lock; inference waits outside it;
registration stays after worker readiness; explicit Apple requests retain their
independent sidecar path. No blocking issue was found in that pass. Existing
model-identity timing and lifecycle failure-path concerns remain Phase 2/3 work,
not claims fixed by this retirement.

## Phase 2 execution contract (2026-09-12)

Phase 1 merged as PR #37 at `d0f29bf`. Phase 2 uses the separate
`refactor/request-execution` worktree and does not alter the running server.

A controlled switch from Paraformer to SenseVoice reproduced a successful HTTP
response labelled `paraformer` although the queued job used `sensevoice-small`.
The route captured identity before waiting for the spawn lock. Capability checks
read the same stale snapshot and can accept unsupported timestamp requests.

Chosen contract:

- Explicit selection is fixed by the request. Validate it before waiting for a
  worker or copying the upload; explicit Apple requests retain independent dispatch.
- Passthrough selection occurs after acquiring the spawn lock, as before. Resolve
  an immutable execution plan there, validate its capabilities before enqueue or
  worker startup, and carry its identity through completion. Do not turn a
  passthrough into an explicit switch to an earlier snapshot.
- `submit` returns a typed completion containing the normalized payload and the
  actual model identity. HTTP rendering reads that completion, not mutable runtime
  state. There is one submission Interface, not a second compatibility method.
- Move request capability checks into this execution Module. HTTP still validates
  uploads, resolves aliases, maps response formats, and maps input errors to 400.
  Passthrough capability errors may wait for an in-progress switch, but never run
  inference. Explicit invalid requests still fail immediately.
- Preserve custom-model conservative capabilities, remote error status mapping,
  queue capacity, and the existing behavior that a model switch terminates old
  worker requests. Changes to that scheduling policy belong to a separate design.

An admission-time snapshot was rejected: passing that snapshot as an explicit
model could switch back to an earlier model and cancel other queued requests.
Fixing only response metadata was rejected because validation could still target
a different model from inference.

Acceptance uses event-controlled concurrent tests through HTTP plus the real
submission path with a fake worker transport; no timing sleeps or real model are
needed to reproduce the interleaving. Check both supported-to-unsupported and
unsupported-to-supported timestamp transitions, actual response identity, explicit
selection stability, language gating, and custom-model behavior.

### Phase 2 validation and review

- Full worktree suite: **338 passed in 38.33 seconds**, including real Paraformer
  one-second silence E2E; the existing unregistered `e2e` marker warning remains.
- Thirteen new event-controlled execution cases cover concurrent selection,
  capability transitions in both directions, switch failure, explicit pinning,
  pre-worker rejection, Apple language validation, custom paths, and identity
  after a later runtime change. No active tests were removed.
- Ruff, Tach, and `git diff --check` passed. Targeted mypy for
  `src/services/execution.py` passed with imported modules followed silently;
  this is not a whole-repository type-check claim.
- Wheel and source distribution built with `uv build --no-build-isolation`.
- A separate second review pass checked that passthrough never becomes an explicit
  historical-model switch, validation precedes enqueue, registration still follows
  worker readiness, and completion metadata is independent of runtime mutations.
  Existing worker shutdown, crash recovery, and queue policy remain Phase 3 scope.

The internal Python return contract changes from a bare payload to
`ExecutionResult(payload, model)`. The single production caller and all test
stand-ins were migrated; the external HTTP response shape remains unchanged.
The worktree shares the existing dependency environment via `.venv` and uses
`--no-sync`. Neither dependency versions nor the running server were changed.

## Phase 3 worker-session contract (2026-09-12)

This phase is stacked on PR #38 (`11353c2`) in the isolated
`refactor/worker-session` worktree. PR #38 remains a separate review/merge step.

The previous shutdown implementation was reproduced with a process whose timed
`join` returns while still alive: `kill()` was never called and the service
cleared its process handle. Startup error branches also discarded the process
without joining it or closing its queues. Ownership now stays in one Module:

- `WorkerSession` owns startup/readiness, pending completions, its result reader,
  and disposal. Its caller selects a `WorkerConfig`, awaits `start`, synchronously
  enqueues under the admission lock, then awaits the returned future outside that
  lock. Successful reuse of the same live configuration does not restart it.
- The internal transport Seam has a real multiprocessing Adapter and a controlled
  test Adapter. The service no longer holds process/queue/reader fields or pending
  completion dictionaries. Upload directories remain owned by `submit`'s `finally`.
- Startup failure, deadline, cancellation, idle exit, process death, and explicit
  shutdown all dispose the owned transport. Readers are joined before replacement;
  an old reader cannot consume a new startup handshake. No blocking executor
  `Queue.get` is left behind on startup timeout.
- Disposal waits for graceful exit, then checks liveness after both timed
  terminate/join and kill/join. If the process survives, keep ownership and reject
  replacement. Cancellation waits for cleanup before releasing lifetime ownership.
- After the child is reaped, close the parent job-reader endpoint so a blocked
  feeder exits on EPIPE, then close/join queues and close both endpoints. A killed
  child can retain the Queue read lock or leave a partially consumed frame, so
  draining through `Queue.get` is unsafe. An independent SIGSTOP-then-kill probe
  reproduced this defect in the first implementation and is retained as a test.
- This disposal uses CPython Queue's `_reader`, `_writer`, and `_ignore_epipe`
  internals, isolated and typed in the transport Adapter. The flag is set before
  any parent feeder starts; endpoint closure happens only after child reaping.
  This avoids discarding the feeder with `cancel_join_thread`, which would hide a
  leak. The trade-off is a CPython dependency: the real spawn probes are in CI and
  must pass when upgrading Python. Local validation uses CPython 3.11.
- Terminal/malformed messages fail pending jobs. Per-job legacy and typed errors
  preserve their existing HTTP mapping. Apple sidecar lifetime remains separate.
- Shutdown blocks later admission and waits for an already-starting resident
  session. Explicit ModelSpec switches still close the old session even if two
  specs reference identical weights. Cancelling a request releases its waiter/upload, not native
  inference. Model switching still terminates outstanding jobs on the old model;
  this phase does not introduce a drain-before-switch scheduling policy.

Keeping lifecycle methods on the service with shared mutable queues was rejected:
that would move code without transferring ownership. Replacing the multiprocessing
wire protocol was also excluded; typed job payloads remain Phase 4 work.

### Regression migration and acceptance

The old suites installed private process handles, dictionaries, and reader tasks.
They now exercise real submission/session behavior with the controlled transport.
Repeated assertions across the old suites were consolidated; active scenarios
remain represented as follows:

| Previous obligation | Current behavioral evidence |
| --- | --- |
| Result/text delivery, remote error types, upload lifetime | `test_service.py` result/error parameter sets |
| Queue full, internal admission, overflow before completion | `test_service.py` and real concurrent admission in `test_concurrency.py` |
| Cancel waiting for lock, cancel explicit/passthrough, enqueue failure | `test_service.py` cancellation and cleanup cases |
| Old reader must not consume new READY | `test_worker_session.py` replacement/late-result case |
| Same-model reuse, selected model after switch, old-job termination | `test_dynamic_switching.py` |
| Failed switch cleanup and recovery, Apple exclusion | `test_dynamic_switching.py` and unchanged Apple tests |
| Lazy state, capabilities, idle/crash restart | `test_idle_offload.py` |
| Selection/validation/response agreement | All 13 Phase 2 HTTP cases retained with the real session |

Lightweight real-subprocess probes run in watchdog-isolated Python processes.
Each performs three lifetimes and checks child reaping, absence of feeder threads,
clean interpreter exit, and absence of resource-tracker warnings. Modes cover
normal shutdown, idle, crash, load error, invalid startup, startup timeout,
SIGTERM-resistant worker, a two-megabyte queued job with no consumer, and a
SIGSTOP-paused consumer killed while waiting inside Queue.get. These
probes do not load ML models. They establish lifecycle behavior for these cases,
not long-audio inference or live-server acceptance.

### Phase 3 validation and second review

- Final full suite: **358 passed in 38.68 seconds**, including real Paraformer
  one-second silence E2E. The pre-existing unregistered `e2e` marker warning remains.
- Nine watchdog-isolated process modes, each repeated three times, passed. The
  SIGSTOP case first failed during queue draining and passed after endpoint-based
  disposal replaced it. These probes now run in CI alongside unit tests.
- Ruff, Tach, `git diff --check`, and targeted mypy for `worker_session.py` and
  `transport.py` passed. No new complexity exemptions or `Any` annotations.
- `uv build --no-build-isolation` produced the wheel and source distribution.
- The separate second review checked ownership transfer, cleanup cancellation,
  same-weight/different-spec switching, failed replacement, sidecar exclusion, and
  every active regression in the migration table. The killed-consumer read-lock
  defect found during that pass was fixed and re-tested with a real process.

The worktree reuses the existing ignored `.venv` with `--no-sync`. Dependencies,
main checkout, and the running HTTP server were not changed. PR #38 is the base;
this phase does not merge it or deploy either phase. Typed job/options contracts
and narrowing the service's legacy lint exemptions remain Phase 4.
