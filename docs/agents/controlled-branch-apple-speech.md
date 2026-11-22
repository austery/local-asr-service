# Controlled Branch evidence: Apple Speech sidecar (draft)

Status: **draft, not accepted.** The author's record has not been verified by an
independent reviewer, and no real-run isolation test has been done (see Gaps).
Nothing in [impact-radius.md](./impact-radius.md) changes until the owner accepts
this record. The Radius table keeps Apple Speech as Trunk.

Snapshot: `origin/main` at `29ab5dc`, read on 2026-11-22. The evidence format
follows "Controlled Branch evidence" in the owner's global
`PERSONAL-REPO-AUTOMATION.md`.

## Claim

A change confined to the Apple Speech sidecar path has an effect that stops at
that path, and a failure there fails one request, not the service. If accepted,
this scope would be a controlled Branch:

- `src/core/apple_speech_engine.py`
- `src/adapters/apple_speech_worker_client.py`
- `apple-speech-worker/Sources/` (the Swift sidecar)

Out of scope, and still Trunk: `src/services/transcription.py`,
`src/api/routes.py`, `src/config.py`, `src/core/model_registry.py`,
`src/core/apple_speech_port.py` and the dispatch code that decides when a request
goes to the sidecar. A change there is Trunk even when its purpose is Apple
Speech. The Structure rules and the verifier paths in `.github/CODEOWNERS` also
still apply.

## Consumers and boundary

Direct callers of the in-scope code (from `rg` on the snapshot):

| In-scope code | Called by |
| --- | --- |
| `AppleSpeechEngine` | `TranscriptionService._get_apple_speech_engine` (`src/services/transcription.py`) |
| `AppleSpeechWorkerClient` | `AppleSpeechEngine.from_config` only |
| Swift sidecar | `AppleSpeechWorkerClient._run_json`, one `subprocess.run` per call |

Callers of the service path, outside the claimed scope: the HTTP route for
`model=apple-speech`, and passthrough requests when Apple Speech is the startup
model (`MODEL_ID=apple-speech`).

Containment mechanisms in the code:

1. **Separate process per call.** `subprocess.run` starts the sidecar for each
   call, so its memory returns to the OS when it exits. It shares no Python
   state with the resident worker.
2. **Per-job deadline.** `subprocess.run(..., timeout=APPLE_SPEECH_WORKER_TIMEOUT_SEC)`
   (default 120 s). Python kills the child on timeout. The client turns that into
   `AppleSpeechWorkerUnavailableError`. The resident worker has no such deadline
   except for MOSS, which is why no MLX or FunASR engine qualifies.
3. **Own admission and concurrency.** Explicit Apple requests count against
   `APPLE_SPEECH_MAX_QUEUE_SIZE` (`_sidecar_admitted`) and run under
   `_sidecar_semaphore` (`APPLE_SPEECH_MAX_CONCURRENCY`, default 1). They do not
   take `_spawn_lock` and do not use the resident worker queue.
4. **Per-request failure.** A non-zero exit, bad JSON or timeout raises an error
   that `routes.py` maps to HTTP 500 for that request. `submit()` releases the
   temp directory and the admission counter in `finally`.

## Failure propagation

| Failure | Reaches | Stops where | Evidence |
| --- | --- | --- | --- |
| Sidecar hangs | The one request, plus one thread of the default executor for up to the timeout | Timeout kills the child; the semaphore is released when the `async with` exits | Timeout mapping is unit-tested with a mocked `TimeoutExpired` |
| Sidecar crashes or prints bad JSON | The one request (HTTP 500) | `_run_json` raises `AppleSpeechWorkerResponseError` | `tests/unit/test_apple_speech_worker_client.py` |
| Apple queue full | Apple callers (HTTP 503) | Separate counter; resident `MAX_QUEUE_SIZE` unaffected | `test_submit_apple_speech_rejects_when_apple_queue_is_full`, `test_pending_apple_speech_requests_do_not_fill_the_local_model_queue` |
| First `load()` blocks | Apple callers waiting on `_apple_speech_load_lock` | Runs in a thread, so the event loop stays free | `test_first_apple_speech_load_does_not_block_the_event_loop` |
| Temp file leak on error | Disk | `finally` removes the temp dir | `test_submit_apple_speech_cleans_temp_dir_on_error` |
| Memory use | The machine, shared with the resident worker | Not established; see Gaps | `MODELS.md` benchmark: 88.9 MB peak process-tree RSS |

## Gaps

These stop an acceptance until they are closed or the owner accepts them.

1. **No real isolation run.** No test runs an Apple request while a FunASR job is
   active. The claim that Apple work cannot slow or starve the resident worker
   rests on reading the code. Passing unit tests do not establish it (policy:
   tests alone do not establish an unexamined boundary).
2. **Memory is measured as process-tree RSS only.** The 88.9 MB figure in
   `MODELS.md` counts the server's process tree. Apple's speech framework may
   use memory in system services outside that tree. I have not verified this.
   Explicit Apple requests run while a FunASR job can hold about 20 GB, so
   this matters on a 64 GiB machine.
3. **The timeout is tested with a mock only.** No test hangs a real sidecar and
   shows the child is gone afterwards.
4. **Passthrough is not contained.** When Apple Speech is the startup model, a
   passthrough request takes `_spawn_lock` and counts toward `MAX_QUEUE_SIZE`
   (`_submit_resident_job`). This is Trunk code and outside the claimed scope.
5. **Default thread pool is shared.** `asyncio.to_thread` uses the event loop's
   default executor, which other code can also use. A hung sidecar holds one
   thread for up to the timeout. I did not check what else uses the executor.
6. **Door and Structure not assessed here.** The scope must stay Two-way and
   leave the response fields, aliases and IPC contract unchanged. The
   `apple-speech` alias is a stored PureSubs value, so renaming it is One-way.

## Proposed verification run

Not run. Run it in a planned window, with the owner at the machine, because the
live service holds the model and competing model processes are not allowed.

1. Start an isolated service instance with `ENGINE_TYPE=funasr`.
2. Start a long `paraformer` request. While it runs, send an explicit
   `model=apple-speech` request with `language=zh`.
3. Record: both results, the FunASR elapsed time against its solo baseline,
   peak memory of the whole machine (not only the process tree), and the
   resident worker's queue state.
4. Hang the sidecar on purpose (for example a stub worker that sleeps) with a
   short `APPLE_SPEECH_WORKER_TIMEOUT_SEC`. Confirm the request returns 500, the
   child process is gone, and the next Apple request succeeds.

If both pass, an independent reviewer verifies this record against the exact
snapshot, and the owner decides whether to reclassify the scope above.
