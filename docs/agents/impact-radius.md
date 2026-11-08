# Impact radius map

The owner's global `PERSONAL-REPO-AUTOMATION.md` defines Door (Two-way /
One-way), Radius (Leaf / Branch / Trunk), Structure, the controlled-Branch
evidence, and the execution gates. This file maps those terms onto
local-asr-service and sets the review depth for each class.

## North Star

The product intent lives outside this repository, because the service has
already been rebuilt on another base once and the intent outlived it:

- `/Users/leipeng/Documents/research-notes/05-archive/2026-05-19_local-asr-service_boundary-creep-and-fitness-functions.md`
  (boundary decision and fitness functions)
- `/Users/leipeng/Documents/research-notes/05-archive/2026-07-06_local-asr-service_project-origin-and-early-evolution.md`
  (origin and early evolution)

Read only each document's `## Current core` section. In-repository
implementation of that intent:
[ADR-002](../ADR-002-Lightweight-Local-Speech-Gateway-Boundary.md). It does not
replace the North Star.

As of 2026-11-08 neither North Star document has a `## Current core` section.
Until the owner writes it, the Trunk split and the conflict examples below are
derived from ADR-002 ("Decision": the thin layer it owns and the list it does
not own) and must be rechecked against `## Current core` once it exists.

**Trunk follows from the thin layer.** The mission depends on four things, and
each one is Trunk below:

| Thin-layer responsibility | Trunk paths |
| --- | --- |
| HTTP compatibility with OpenAI-style transcription | `src/api/routes.py`, `src/main.py` |
| Queueing, single-worker execution, resource isolation, idle offload | `src/services/transcription.py`, `execution.py`, `worker_session.py`; `src/workers/` |
| Model aliases and capability gating | `src/core/model_registry.py`, `base_engine.py`, `factory.py`, `src/config.py` |
| Response normalization (predictable JSON for callers) | the transcript producers listed under Structure paths, led by `src/core/funasr_engine.py` |

**`north_star_conflict` examples.** A task that does any of these is a product
decision for the owner, not an implementation detail, whatever its radius:

- port or reimplement a model architecture, or vendor a fork of upstream model
  code (beyond the existing narrow FunASR `distribute_spk` patch);
- add inference kernels, quantization, or model conversion pipelines;
- add speaker embedding fallback, clustering, cross-chunk speaker
  reconciliation, or other diarization postprocessing beyond the existing
  minimal logic;
- add a new job domain or a general multi-model orchestration path (for example,
  restoring the retired `qwen3-sortformer` pipeline or its borrow/restore logic);
- make an experimental or cloud alias the default model, or add automatic
  routing between models;
- add app UI, microphone capture, streaming voice input, or Swift-native runtime
  code beyond the thin Apple Speech sidecar.

Each change is classified on three separate axes:

1. **Door:** can the change and its effects be undone?
2. **Radius:** if the change is wrong, how far does the failure spread?
3. **Structure:** does the change touch a structure path (below)? The changed
   file paths decide this, not judgment.

Keep Radius and Structure apart. This service has no database, job store or
result cache: it keeps request state in memory and writes uploads to temporary
directories that it deletes (`src/services/transcription.py`). Its long-lived
structure is what other systems store or depend on: the HTTP contract and the
transcript text formats that PureSubs stores, the model aliases that PureSubs
stores, the model identities that decide multi-gigabyte cache downloads, the
worker IPC messages, the Apple Speech sidecar protocol and its OS asset
reservation, and the dependency lock.

## Structure paths

A change that adds, modifies or deletes any of these files has
**Structure = touched**. The whole file counts; the reasons name the parts that
make it structure.

- `src/api/routes.py`: request form fields, `ALLOWED_AUDIO_TYPES` (PureSubs sends
  `audio/mpeg`), `_RESPONSE_FORMAT_MAP`, the response models (`Segment`,
  `TranscriptionResponse`, `ModelInfo`, `ModelsResponse`), the segment mapping,
  the error-status mapping, and the `/v1/models/current` keys.
- Transcript text producers, whose output PureSubs stores as
  `[<speaker>]: <text>` (`LocalWhisperService.ts`, lines ~238-244):
  `src/core/funasr_engine.py` (`Speaker N` labels, txt and srt rendering),
  `src/core/apple_speech_engine.py` (json and srt rendering returned directly
  by `routes.py`), `src/adapters/moss_output.py`, `src/adapters/deepgram_output.py`,
  `src/services/remote_session.py` (`render_remote`).
- `src/core/model_registry.py`: aliases, model IDs, and `MOSS_MODEL_REVISION`.
  PureSubs stores alias strings in its database (`schema.prisma`, line ~49), in
  its dashboard list (`channel-dialog.component.ts`, `knownAsr`), in job metadata
  (`userSelectedModel`), and hard-codes `paraformer`
  (`src/infrastructure/podcast/audio-material.ts`).
- Model identity and cache layout: `src/config.py` (`FUNASR_MODEL_ID`,
  `MLX_MODEL_ID`, `APPLE_SPEECH_WORKER_PATH`), the FunASR auxiliary models
  `fsmn-vad`, `ct-punc` and `cam++` in `src/core/funasr_engine.py`, and the load
  options in `src/core/mlx_engine.py` (including the MOSS `revision`). These
  decide what is downloaded into `~/.cache/modelscope/hub/` and
  `~/.cache/huggingface/hub/`.
- Worker IPC: `src/workers/model_worker.py` (the `WorkerJob` dataclass and the
  result tuples) and `src/workers/transport.py` (framing). Parent and child must
  agree on them.
- Apple Speech sidecar protocol and binary location:
  `apple-speech-worker/Package.swift` (product name),
  `apple-speech-worker/Sources/apple-speech-worker/main.swift` (stdout line,
  exit codes), `apple-speech-worker/Sources/AppleSpeechWorkerCore/WorkerModels.swift`,
  `WorkerCommands.swift`, the Python client `src/adapters/apple_speech_worker_client.py`,
  the port `src/core/apple_speech_port.py`, and the fixture
  `docs/fixtures/apple-speech-transcribe-en-US.json`. Python and Swift are built
  separately, so a one-sided change fails at run time.
- Apple Speech OS state:
  `apple-speech-worker/Sources/AppleSpeechWorkerCore/LiveAppleSpeechRuntime.swift`,
  which calls `AssetInventory.reserve(locale:)` on each request.
- Dependencies: `pyproject.toml` and `uv.lock`. A runtime upgrade can change
  transcript output for the same weights. On 2026-11-07 a local isolated A/B
  (kept outside the repository) found that FunASR 1.4.23 changed segmentation
  and the speaker count on the same clip compared with 1.4.19.
- Policy fallback globs: `**/migrations/**`, `**/*.prisma`, `**/schema.*`,
  `**/*.sql`. None exists today; adding one is a new data model and needs a
  design.

Structure = touched means:

- The change must match a design the owner approved before implementation that
  names each request field, response field, text format, alias, model ID or
  revision, IPC field, sidecar message, OS reservation or dependency it adds or
  alters. Anything not named there stops and waits for the owner.
- A contract, text-format or alias change also states its PureSubs impact and
  whether PureSubs needs a coordinated change.
- A change to an inference runtime in `uv.lock` (`funasr`, `torch`,
  `torchaudio`, `mlx`, `mlx-audio`, `mlx-metal`) attaches a real-model A/B on the
  owner-approved clips. A dev-tool change (`ruff`, `tach`, `pytest`, `mypy`) is
  also a verifier change (see Always stop).

Add a path here when the project gains another kind of persistent structure, such
as a job store, a result cache, or a new sidecar protocol.

## Radius in this repository

Radius answers only how far a failure spreads. Consumers below come from the
`src/` import graph on `90a350b`. `tach.toml` enforces layer direction, but it
treats `src.core` as one module, so it does not isolate engines from each other.

All local models (FunASR and MLX) run in one worker subprocess with one serial
queue. `WorkerSession.enqueue` (`src/services/worker_session.py`) waits without a
timeout; the only per-job inference deadline is for MOSS
(`src/workers/model_worker.py`). A hung or memory-exhausting job therefore blocks
every later request for every alias, including `paraformer`, the PureSubs
default. This is why no engine is a controlled Branch today.

| Radius | In this repository |
| --- | --- |
| **Trunk** | **Request execution and admission:** `src/services/transcription.py`, `src/services/execution.py`, `src/services/worker_session.py`. Every HTTP request passes through them; they own `_spawn_lock`, model switching, upload temp files and worker lifetime (`AGENTS.md`, SPEC-016). **Worker process and IPC:** `src/workers/model_worker.py`, `src/workers/transport.py`; a hang or leak blocks every request and shutdown (history: the 2026-04-03 shutdown hang). **HTTP layer and startup:** `src/api/routes.py`, the only caller-facing surface; `src/main.py`, the composition root that starts and stops the service. **Deepgram remote session:** `src/services/remote_session.py`. `TranscriptionService` builds `RemoteSession` for every instance and calls its `start()` on startup and `close()` on shutdown; `GET /v1/models` calls `deepgram_unavailable_reason()` and `deepgram_configured()` for every caller; `deepgram_unavailable_reason()` is the only check that blocks paid uploads. **Shared contracts:** `src/core/base_engine.py` (`ASREngine`, `EngineCapabilities`, `TranscriptionInputError`), imported by `api`, `services`, every engine and the registries. `src/core/model_registry.py`, imported by `api`, `services`, `workers`, `main`, `mlx_engine` and `factory`. `src/core/factory.py`, the only constructor of local engines, called by the worker (`model_worker.create_engine`). **Configuration:** `src/config.py`, which loads `.env` and is imported by every layer (`MODEL_IDLE_TIMEOUT_SEC`, `MAX_QUEUE_SIZE`, `DEEPGRAM_ENABLED`). **The FunASR engine** `src/core/funasr_engine.py`: the production default, and the home of the required `distribute_spk` patch |
| **Branch** | `src/adapters/text.py`: a pure text-cleanup function, imported lazily by `funasr_engine` only for SenseVoice (around lines 223-226). A failure is limited to SenseVoice responses. No other path is mapped as Branch until it has the containment evidence the policy requires |
| **Leaf** | Test additions under `tests/` that leave existing assertions, fixtures, helpers, configuration and baselines unchanged (see the offline test-only lane in `AGENTS.md`). `examples/demo_diarization.py` and `scripts/probe_qwen3_sortformer_longform.py`, which nothing in `src/` or `tests/` imports. A script that posts to the running service on port 50700 is not Leaf (see Unknown). `.env.example`, which no code reads. Historical records not referenced by `AGENTS.md` or `MODELS.md`: `docs/superpowers/`, `docs/2026-*-upgrade.md`, `.scratch/`, and the `docs/plans/` files not listed under Always stop. Documentation that states no rule (typo or link fixes in `README.md`, benchmark prose in `MODELS.md`) |

Not yet mapped, so treat as **Unknown**:

- `src/core/mlx_engine.py` and `src/adapters/audio_chunking.py` (Qwen3-ASR,
  SenseVoice-MLX, MOSS). Missing evidence: a per-job timeout or proof that a hang
  or memory spike in one MLX model cannot block other aliases. `qwen3-asr` is
  also a stored PureSubs channel preference.
- `src/core/apple_speech_engine.py` and the sidecar internal
  `apple-speech-worker/Sources/AppleSpeechWorkerCore/WorkerRuntime.swift`.
  Each sidecar call has a 120-second timeout
  (`src/adapters/apple_speech_worker_client.py`). Missing evidence: whether a
  request that waits for that timeout holds `_spawn_lock` and blocks admission
  for other aliases.
- `src/adapters/deepgram_http.py`. It puts the API key in the `Authorization`
  header and makes the paid upload. Missing evidence: none needed to stop; see
  Always stop and One-way effects.
- The retired decoupled-diarization modules: `src/core/diarization_registry.py`,
  `src/core/alignment_registry.py`, `src/core/pipeline_registry.py`,
  `src/core/mlx_sortformer_diarizer.py`, `src/core/mlx_qwen_forced_aligner.py`,
  the ports `src/core/diarization_port.py` and `src/core/alignment_port.py`,
  and `src/adapters/segment_alignment.py`, `src/adapters/pipeline_chunking.py`.
  `model_worker.py` can still build a diarizer or aligner, but no producer of
  `requested_diarizer_alias` or `requested_aligner_alias` exists in `src/`.
  `audio_chunking.py` (the MLX path) imports `pipeline_chunking.py`. Missing
  evidence: whether a live path reaches them, and whether SPEC-016 intends to
  delete or keep them.
- `src/core/remote_port.py`, imported by `api`, `services` and `adapters`.
  Missing evidence: whether its types change anything outside the Deepgram path.
- `examples/demo_clean_tags.py` and any script that posts to the running service
  on port 50700. Running it with another model alias makes the live service
  switch models, which shuts down the resident worker and can fail requests in
  flight. Missing evidence: that it targets an isolated service instance.
- `.github/dependabot.yml` and any path this table does not name.

## One-way effects in this repository

- **Paid provider calls.** A request routed to Deepgram (`DEEPGRAM_ENABLED=true`
  with a key) spends money and sends audio to a third party. Its live acceptance
  is open (Issue #62).
- **Contract, text-format or alias changes after data exists.** PureSubs stores
  alias strings and formatted transcripts. A removed alias, renamed field or
  changed speaker label cannot be undone for rows already written.
- **Model downloads and cache deletion.** A changed model ID or revision
  downloads gigabytes into the user caches. Deleting a cache forces a
  re-download and can lose a revision that upstream has since replaced.
- **Apple Speech asset reservations.** `AssetInventory.reserve` changes OS-level
  asset state. Whether reservations are system-wide or limited in number is
  Unknown; treat a change in reserved locales or modules as One-way until known.
- **Live service actions.** Restarting or switching the running service on port
  50700, or changing its virtual environment. PureSubs depends on it, and the
  recorded recovery is a retained environment kept outside the repository (local
  `pr-reviews/`), not a code revert.

## Always stop, whatever the class

- **Verifier changes.**
  - `.github/workflows/ci.yml` and its test selection (`tests/unit`,
    `tests/reliability/test_worker_session_process.py`).
  - `tach.toml` boundary rules and `tests/unit/test_architecture_fitness.py`
    (SPEC-015).
  - Test configuration in `pyproject.toml`: `[tool.ruff*]`,
    `[tool.pytest.ini_options]`, `[tool.mypy]`. Version changes of `ruff`,
    `tach`, `pytest` or `mypy` in `uv.lock`.
  - Shared test helpers `tests/helpers/` and any `conftest.py`; existing test
    assertions, fixtures and thresholds; the test-count baseline in `AGENTS.md`.
  - The Swift contract tests:
    `apple-speech-worker/Sources/AppleSpeechWorkerContractTests/` and their target
    in `apple-speech-worker/Package.swift`.
  - `benchmarks/`: the model-comparison harness named in `MODELS.md`
    (`benchmarks/phase3_evaluation.py`), imported by CI unit tests
    (`test_diarization_metrics.py`, `test_diarization_poc.py`,
    `test_phase3_evaluation.py`).

  Adding coverage for an approved behavior change is not a verifier change;
  weakening or removing a check is.
- **Policy and acceptance text.** `AGENTS.md` (and its `CLAUDE.md` symlink),
  this file and anything under `docs/agents/`, ADRs (`docs/ADR-*.md`), accepted
  specs (`docs/SPEC-*.md`), and the plans that `AGENTS.md` or `MODELS.md` treat
  as contracts or acceptance criteria:
  `docs/plans/2026-09-07-puresubs-moss-integration.md`,
  `docs/plans/2026-09-07-moss-adapter-contract.md`,
  `docs/plans/2026-10-29-cloud-diarized-transcription.md`.
- **Paid-call gate and secrets.** `deepgram_unavailable_reason()` and
  `deepgram_configured()` in `src/services/remote_session.py`, `DEEPGRAM_*`
  settings in `src/config.py`, and key handling in `src/adapters/deepgram_http.py`.
- **The FunASR monkey patch.** Any change to the `distribute_spk` patch or its
  two bindings in `src/core/funasr_engine.py`. A broken binding only logs a
  warning, so tests can stay green while speaker attribution changes.

The real-model A/B harness and the approved audio clips live outside the
repository (local, Git-ignored `pr-reviews/`). A path rule cannot protect them;
see Prerequisites.

## Review depth

| Classification | Delivery and review | Owner involvement |
| --- | --- | --- |
| Leaf, Two-way, Structure untouched | One implementation; Standards, Spec and cross-vendor Risk review; then merge | None |
| Controlled Branch, Two-way, Structure untouched | The author records consumers and failure propagation; the reviews verify that evidence | Only when author and reviewer disagree |
| Structure touched, within the approved design | Independent reviews, plus the PureSubs impact statement or the real-model A/B named above | Design approval before work; at merge, only when the review finds a difference from the design |
| Wrong-but-green domains (policy L2): engine output, inference runtime dependencies, worker lifetime and crash recovery, paid calls | L2 reviews (cross-vendor Standards and Spec, plus Risk), and real-run evidence: a real-model engine run on the approved clips; an HTTP request against an isolated service instance when the request path changes | Decision before merge |
| Trunk, One-way, Unknown, Structure outside the design, or an "Always stop" item | Owner design approval before implementation; independent reviews; real-run evidence for runtime changes | Design approval before work, decision before merge |

Unit and integration tests mock the engine (`AGENTS.md`), so a green CI run says
nothing about real-model output.

## Slicing example: report a truncated transcript

Callers want to know when a transcript may be incomplete, for example when a
model stops at its output-token limit. Today most runtime slices stop, because
no engine is a controlled Branch yet. The work splits like this:

1. **Design gate (owner approves).** Which engines can detect truncation, the
   exact new response field (for example an optional `finish_reason` on
   `TranscriptionResponse`), its values, and how PureSubs should react.
2. **Structure and Trunk ticket, within the design.** Add the optional field to
   `src/api/routes.py` and the engine result in `src/core/base_engine.py`. Owner
   decision before merge, with the PureSubs impact statement.
3. **Unknown and wrong-but-green ticket.** Fill the field in
   `src/core/mlx_engine.py` for one MLX model, with a real-model run on a long
   clip. Owner decision before merge.
4. **Leaf ticket.** Add unit tests for the new field mapping without changing
   existing assertions. Merge after review.
5. **Stops.** Filling the field in the FunASR engine touches the Trunk production
   engine and its patch file. Making `finish_reason` required, or renaming
   `segments`, is One-way for stored PureSubs data.

## Prerequisites before automatic merge

As of 2026-11-08, none of these holds, so automatic merge stays off:

- **`main` requires the CI check.** The only ruleset on `main` ("protest my main
  branch") blocks deletion and force-push. It does not require the
  `Lint, Boundary, and Unit Tests` job, there is no classic branch protection,
  and GitHub auto-merge is disabled.
- **Agents and the owner have different GitHub identities.** PRs are authored and
  merged as the owner's account, and `ci.yml` runs from the PR itself, so a PR can
  change the check that judges it. Until agents act through a separate account or
  GitHub App, the verifier and Structure rules above are followed by agents, not
  enforced by GitHub.
- **Structure and verifier paths are checked by a rule the PR cannot change.** For
  example, a CODEOWNERS entry plus a required code-owner review for the paths
  above.
- **Engine containment exists.** A per-job inference timeout, or equivalent
  evidence, so that an engine change can be a controlled Branch.
- **The acceptance harness is reproducible from the repository.** The real-model
  A/B scripts and the identity (hashes) of the approved clips are recorded in a
  tracked file, so a path rule can protect them.
- **The Unknown paths above are resolved.**
