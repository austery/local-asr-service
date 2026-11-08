# Impact radius map

The owner's global `PERSONAL-REPO-AUTOMATION.md` defines Door (Two-way /
One-way), Radius (Leaf / Branch / Trunk), Structure, the controlled-Branch
evidence, and the execution gates. This file maps those terms onto
local-asr-service and sets the review depth for each class.

Each change is classified on three separate axes:

1. **Door:** can the change and its effects be undone?
2. **Radius:** if the change is wrong, how far does the failure spread?
3. **Structure:** does the change touch a structure path (below)? The changed
   file paths decide this, not judgment.

Keep Radius and Structure apart. This service has no database. Its long-lived
structure is the HTTP response shape, the model aliases that callers store, the
model cache layout, the worker IPC messages, the Apple Speech sidecar protocol,
and the dependency lock. A one-line change to any of them can still be Leaf by
radius, but it stops at the Structure gate.

## Structure paths

A change that adds, modifies or deletes anything under these paths, or the named
symbols inside them, has **Structure = touched**:

- **HTTP response and request contract.** `Segment`, `TranscriptionResponse`,
  `ModelInfo`, `ModelsResponse`, `_RESPONSE_FORMAT_MAP`, and the form fields of
  `create_transcription` in `src/api/routes.py`. PureSubs posts `file`, `model`,
  `response_format=verbose_json` and `language`, then reads `text`,
  `segments[].speaker` and `segments[].text`
  (`puresubs/packages/automation-engine-ytdlp/src/transcription/providers/LocalWhisperService.ts`)
  and stores the formatted transcript.
- **Model aliases.** The alias table in `src/core/model_registry.py`. PureSubs
  stores these alias strings in its database
  (`puresubs/.../prisma/schema.prisma`, comment "Matches local-asr-service model
  aliases"), so renaming or removing an alias breaks stored rows and caller
  configuration.
- **Model identity and cache layout.** Default model IDs in `src/config.py`
  (`FUNASR_MODEL_ID`, `MLX_MODEL_ID`) and `model_id` values in
  `src/core/model_registry.py`. They decide the directories used under
  `~/.cache/modelscope/hub/` and `~/.cache/huggingface/hub/`, which hold
  multi-gigabyte downloads.
- **Worker IPC messages.** The job and result dataclasses in
  `src/workers/model_worker.py` and the framing in `src/workers/transport.py`.
  The parent and the child process must agree on them.
- **Apple Speech sidecar protocol.** `apple-speech-worker/Sources/AppleSpeechWorkerCore/WorkerModels.swift`,
  `WorkerCommands.swift`, the Python client `src/adapters/apple_speech_worker_client.py`,
  the port `src/core/apple_speech_port.py`, and the contract fixture
  `docs/fixtures/apple-speech-transcribe-en-US.json`. Python and Swift are built
  separately, so a one-sided change breaks at run time.
- **Dependency lock.** `pyproject.toml` and `uv.lock`. A runtime upgrade can
  change transcription output for the same weights (FunASR 1.4.19 and 1.4.23
  changed segmentation and speaker labels on identical input).
- **Policy fallback globs.** `**/migrations/**`, `**/*.prisma`, `**/schema.*`,
  `**/*.sql`. None exists today; adding one is a new data model and needs a
  design.

Structure = touched means:

- The change must match a design the owner approved before implementation that
  names each response field, alias, model ID, message field or dependency it
  adds or alters. Anything not named there stops and waits for the owner.
- An alias or response-field change also states the PureSubs impact and whether
  PureSubs needs a coordinated change. Removing or renaming a field or alias is
  One-way for already stored PureSubs data.
- A dependency change in `uv.lock` that touches an inference runtime (`funasr`,
  `torch`, `torchaudio`, `mlx`, `mlx-audio`, `mlx-metal`) attaches a real-model
  A/B on the approved clips. A lock change limited to dev tools (`ruff`, `tach`,
  `pytest`) needs only CI.

Add a path here when the project gains another kind of persistent structure, such
as a job store, a result cache, or a new sidecar protocol.

## Radius in this repository

Radius answers only how far a failure spreads. Module dependencies are enforced
by `tach.toml`; consumers below come from the import graph on `90a350b`.

| Radius | In this repository |
| --- | --- |
| **Trunk** | **Request execution and admission:** `src/services/transcription.py`, `src/services/execution.py`, `src/services/worker_session.py`. Every HTTP request passes through them; they own `_spawn_lock`, model switching, upload temp files and worker lifetime (CLAUDE.md, SPEC-016). **Worker process and IPC:** `src/workers/model_worker.py`, `src/workers/transport.py`; a hang or leak here blocks every request and shutdown (history: the 2026-04-03 shutdown hang). **HTTP layer and startup:** `src/api/routes.py` (the only caller-facing surface), `src/main.py` (composition root, worker start/stop). **Shared contracts:** `src/core/base_engine.py` (`ASREngine`, `EngineCapabilities`, `TranscriptionInputError`), imported by `api`, `services`, every engine and both registries. `src/core/model_registry.py`, imported by `api`, `services`, `workers`, `main` and `mlx_engine`. `src/core/factory.py`. **Configuration:** `src/config.py` and `.env.example` (environment defaults read by every layer, including `MODEL_IDLE_TIMEOUT_SEC`, `MAX_QUEUE_SIZE`, `DEEPGRAM_ENABLED`). **The FunASR engine** `src/core/funasr_engine.py`, because it is the production default and holds the required module-level `distribute_spk` patch on two bindings. **Boundary rules:** `tach.toml` |
| **Branch** | The internals of one engine behind an unchanged `ASREngine` contract and unchanged registry entries: `src/core/mlx_engine.py` (Qwen3-ASR, SenseVoice, MOSS), `src/core/apple_speech_engine.py`. Its output adapters: `src/adapters/moss_output.py`, `src/adapters/audio_chunking.py` (used by `mlx_engine`). The Deepgram remote path behind an unchanged `src/core/remote_port.py`: `src/services/remote_session.py`, `src/adapters/deepgram_http.py`, `src/adapters/deepgram_output.py`. Swift sidecar internals that leave the protocol files unchanged: `apple-speech-worker/Sources/AppleSpeechWorkerCore/LiveAppleSpeechRuntime.swift`, `WorkerRuntime.swift`. `src/adapters/text.py` (imported lazily by `funasr_engine` for tag cleanup) |
| **Leaf** | Test additions under `tests/` that leave existing assertions, fixtures, helpers and baselines unchanged (see the offline test-only lane in `AGENTS.md`). `benchmarks/`, `examples/`, and `scripts/probe_qwen3_sortformer_longform.py`, which nothing in `src/` imports. Historical records: `docs/plans/`, `docs/superpowers/`, `docs/2026-*-upgrade.md`, `.scratch/`. Documentation that states no rule (typo or link fixes in `README.md`, `MODELS.md` benchmark text) |

Not yet mapped, so treat as **Unknown**:

- The retired decoupled-diarization modules: `src/core/diarization_registry.py`,
  `src/core/alignment_registry.py`, `src/core/pipeline_registry.py`,
  `src/core/mlx_sortformer_diarizer.py`, `src/core/mlx_qwen_forced_aligner.py`,
  the ports `src/core/diarization_port.py` and `src/core/alignment_port.py`,
  and `src/adapters/pipeline_chunking.py`, `src/adapters/segment_alignment.py`.
  `model_worker.py` can still build a diarizer or aligner, but no producer of
  `requested_diarizer_alias` or `requested_aligner_alias` exists in `src/`.
  `audio_chunking.py` (production MLX path) imports `pipeline_chunking.py`.
  Missing evidence: whether any live path still reaches them, and whether
  SPEC-016 intends to delete or keep them.
- `src/core/remote_port.py`. It is imported by `api`, `services` and `adapters`,
  which suggests Trunk, but Deepgram is disabled by default. Missing evidence:
  whether a disabled remote path counts as a shared contract.
- `.github/dependabot.yml` and any path this table does not name.

## One-way effects in this repository

- **Paid provider calls.** Any request routed to Deepgram (`DEEPGRAM_ENABLED=true`
  with a key) spends money and sends audio to a third party. Its live acceptance
  is still open (Issue #62).
- **Alias or response-field removal.** PureSubs stores alias strings and formatted
  transcripts. A removed alias or field cannot be undone for rows already written.
- **Model downloads and cache deletion.** A changed model ID downloads gigabytes
  into the user caches. Deleting a cache forces a re-download and can lose a
  revision that upstream has since replaced.
- **Live service actions.** Restarting or switching the running service at port
  50700, or changing its virtual environment. PureSubs depends on it, and the
  recorded recovery is a retained environment (`pr-reviews/funasr-cutover-20261101/CUTOVER.md`),
  not a code revert.

## Always stop, whatever the class

- **Verifier changes.** `.github/workflows/ci.yml`; the CI test selection
  (`tests/unit`, `tests/reliability/test_worker_session_process.py`); `tach.toml`
  rules; `tests/unit/test_architecture_fitness.py` (SPEC-015); shared test
  helpers `tests/helpers/` and any `conftest.py`; existing test assertions,
  fixtures and thresholds; the test-count baseline in `AGENTS.md`; Ruff
  configuration in `pyproject.toml`; and the real-model A/B scripts used for
  acceptance. Adding coverage for an approved behavior change is not a verifier
  change; weakening or removing a check is.
- **Policy text.** `AGENTS.md` (and its `CLAUDE.md` symlink), this file and
  anything under `docs/agents/`, ADRs (`docs/ADR-*.md`) and accepted specs
  (`docs/SPEC-*.md`), including the offline test-only acceptance rule.
- **The FunASR monkey patch.** Any change to the `distribute_spk` patch or its
  two bindings in `src/core/funasr_engine.py`. A broken binding only logs a
  warning, so tests can stay green while speaker attribution changes.

## Review depth

| Classification | Delivery and review | Owner involvement |
| --- | --- | --- |
| Leaf, Two-way, Structure untouched | One implementation; Standards, Spec and cross-vendor Risk review; then merge | None |
| Controlled Branch, Two-way, Structure untouched | The author records consumers and failure propagation for the one engine or remote path; the reviews verify that evidence. A runtime behavior change in an engine also attaches a real-model run on the approved clips | Only when author and reviewer disagree |
| Structure touched, within the approved design | Independent reviews, plus the PureSubs impact statement or the real-model A/B named above | Design approval before work; at merge, only when the review finds a difference from the design |
| Trunk, One-way, Unknown, Structure outside the design, or an "Always stop" item | Owner design approval before implementation; independent reviews; for runtime changes, evidence from a real run (real-model engine run, and an HTTP request against an isolated service instance when the request path changes) | Design approval before work, decision before merge |

Unit and integration tests mock the engine (`AGENTS.md`), so a green CI run says
nothing about real-model output. Engine, dependency and worker-lifetime changes
are wrong-but-green candidates and need real-run evidence.

## Slicing example: report a truncated transcript

Callers want to know when a transcript may be incomplete, for example when a
model stops at its output-token limit. The work splits like this:

1. **Design gate (owner approves).** Which engines can detect truncation, the
   exact new response field (for example an optional `finish_reason` on
   `TranscriptionResponse`), its values, and how PureSubs should react.
2. **Structure ticket, within the design.** Add the optional field to
   `src/api/routes.py` and the `ASREngine` result. This touches a structure path
   and the Trunk HTTP layer: owner decision before merge, with the PureSubs
   impact statement.
3. **Branch ticket.** Fill the field inside `src/core/mlx_engine.py` for one MLX
   model, with a real-model run on a long clip. The contract is already merged, so
   the change stays inside one engine.
4. **Leaf ticket.** Add unit tests for the new field mapping without changing
   existing assertions.
5. **Stops.** Changing the FunASR engine to fill the field touches the Trunk
   production engine and its patch file. Making `finish_reason` required, or
   renaming `segments`, is One-way for stored PureSubs data.

## Prerequisites before automatic merge

As of 2026-11-08, none of these holds, so automatic merge stays off:

- **`main` requires the CI check.** `main` has no ruleset requiring the
  `Lint, Boundary, and Unit Tests` job, and GitHub auto-merge is not enabled.
- **Agents and the owner have different GitHub identities.** PRs are authored and
  merged as the owner's account, and `ci.yml` runs from the PR itself, so a PR can
  change the check that judges it. Until agents act through a separate account or
  GitHub App, the verifier and Structure rules above are followed by agents, not
  enforced by GitHub.
- **Structure and verifier paths are checked by a rule the PR cannot change.** For
  example, a CODEOWNERS entry plus a required code-owner review for the paths
  above.
- **The Unknown paths above are resolved.**
