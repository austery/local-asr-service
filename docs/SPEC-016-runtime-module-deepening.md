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

- [ ] Introduce an internal worker-session Module that owns startup, IPC, exit,
  pending completions, and resource release; inject the process transport for tests.
- [ ] Consolidate cleanup for normal exit, startup failure, timeout, and crash.
- [ ] Keep Apple sidecar lifetime distinct; share an execution Interface only where
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
