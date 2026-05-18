# Production Decoupled Diarization Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `SPEC-013` so the service can execute a real `qwen3-sortformer` decoupled ASR + diarization pipeline through the subprocess worker and return truthful speaker-labeled results.

**Architecture:** Keep the existing `ASREngine` path for transcription jobs and add a separate diarization runtime path rather than overloading `ASREngine` with a second responsibility. Extend the worker IPC contract with an explicit `job_kind`, add an MLX Sortformer diarizer wrapper behind a small registry, and make pipeline execution hold a worker reservation so model switching and resident restore become deterministic.

**Tech Stack:** Python 3.11, FastAPI, asyncio, multiprocessing.Queue IPC, `mlx-audio>=0.4.3`, pytest, MLX Metal on Apple Silicon.

---

## File Structure

- Create: `src/core/diarization_registry.py`
  - Define `DiarizationSpec`, `lookup_diarizer()`, and the built-in `sortformer-diar` entry.
- Create: `src/core/mlx_sortformer_diarizer.py`
  - Wrap `mlx_audio.vad` Sortformer and adapt `DiarizationSegment` to local `SpeakerTurn`.
- Modify: `src/core/diarization_port.py`
  - Keep `SpeakerTurn`, but add any small typed helpers needed by the diarizer wrapper.
- Modify: `src/workers/model_worker.py`
  - Add `job_kind`, diarization runtime loading, and explicit diarization result messages.
- Modify: `src/services/transcription.py`
  - Submit diarization jobs through the worker, reserve the worker across the full pipeline, and restore deterministically.
- Modify: `src/core/pipeline_registry.py`
  - Flip `qwen3-sortformer` to requestable only in the final enablement task.
- Modify: `tests/unit/test_worker.py`
  - Cover diarization job execution and worker-side errors.
- Modify: `tests/unit/test_service.py`
  - Cover worker-backed `_diarize_with_alias()`, deterministic reservation/restore behavior, and typed coercion.
- Create: `tests/unit/test_diarization_registry.py`
  - Lock the `sortformer-diar` registry contract.
- Create: `tests/unit/test_mlx_sortformer_diarizer.py`
  - Lock the MLX adapter shape without requiring a real model.
- Modify: `tests/integration/test_model_api.py`
  - Update requestable pipeline expectations once the profile is enabled.
- Modify: `README.md`
  - Document the now-requestable pipeline and runtime dependency expectations.
- Modify: `MODELS.md`
  - Mark `qwen3-sortformer` as enabled and describe the production caveats.

### Task 1: Add a Diarization Runtime Contract

**Files:**
- Create: `src/core/diarization_registry.py`
- Create: `src/core/mlx_sortformer_diarizer.py`
- Create: `tests/unit/test_diarization_registry.py`
- Create: `tests/unit/test_mlx_sortformer_diarizer.py`
- Modify: `src/core/diarization_port.py`

- [ ] **Step 1: Write the failing registry and adapter tests**

Create `tests/unit/test_diarization_registry.py`:

```python
import pytest

from src.core.diarization_registry import lookup_diarizer


def test_lookup_should_resolve_sortformer_diar() -> None:
    spec = lookup_diarizer("sortformer-diar")

    assert spec.alias == "sortformer-diar"
    assert spec.runtime == "mlx"
    assert spec.model_id == "mlx-community/diar_sortformer_4spk-v1-fp16"


def test_lookup_should_raise_for_unknown_alias() -> None:
    with pytest.raises(KeyError, match="Unknown diarizer"):
        lookup_diarizer("missing-diarizer")
```

Create `tests/unit/test_mlx_sortformer_diarizer.py`:

```python
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.core.mlx_sortformer_diarizer import MlxSortformerDiarizer


def test_diarize_file_should_map_segments_to_speaker_turns() -> None:
    model = MagicMock()
    model.generate.return_value = SimpleNamespace(
        segments=[
            SimpleNamespace(start=0.0, end=1.5, speaker=0),
            SimpleNamespace(start=2.0, end=3.0, speaker=2),
        ]
    )

    diarizer = MlxSortformerDiarizer("mlx-community/diar_sortformer_4spk-v1-fp16")

    with patch("src.core.mlx_sortformer_diarizer.vad_load", return_value=model):
        diarizer.load()
        turns = diarizer.diarize_file("audio.wav")

    assert [(t.speaker, t.start, t.end) for t in turns] == [
        ("Speaker 0", 0.0, 1.5),
        ("Speaker 2", 2.0, 3.0),
    ]


def test_diarize_file_should_require_load() -> None:
    diarizer = MlxSortformerDiarizer("mlx-community/diar_sortformer_4spk-v1-fp16")

    try:
        diarizer.diarize_file("audio.wav")
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "not loaded" in str(exc)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run python -m pytest tests/unit/test_diarization_registry.py tests/unit/test_mlx_sortformer_diarizer.py -q
```

Expected: FAIL because `src.core.diarization_registry` and `src.core.mlx_sortformer_diarizer` do not exist.

- [ ] **Step 3: Write the minimal registry and MLX adapter**

Create `src/core/diarization_registry.py`:

```python
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DiarizationSpec:
    alias: str
    runtime: Literal["mlx"]
    model_id: str
    description: str


_REGISTRY: dict[str, DiarizationSpec] = {
    "sortformer-diar": DiarizationSpec(
        alias="sortformer-diar",
        runtime="mlx",
        model_id="mlx-community/diar_sortformer_4spk-v1-fp16",
        description="MLX Sortformer diarization runtime.",
    )
}


def lookup_diarizer(alias: str) -> DiarizationSpec:
    try:
        return _REGISTRY[alias]
    except KeyError as exc:
        raise KeyError(f"Unknown diarizer: '{alias}'") from exc
```

Create `src/core/mlx_sortformer_diarizer.py`:

```python
from typing import Any

from mlx_audio.vad import load as vad_load

from src.core.diarization_port import SpeakerTurn


class MlxSortformerDiarizer:
    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self._model: Any | None = None

    def load(self) -> None:
        self._model = vad_load(self._model_id)

    def diarize_file(self, file_path: str) -> list[SpeakerTurn]:
        if self._model is None:
            raise RuntimeError("Sortformer diarizer not loaded")
        result = self._model.generate(file_path, threshold=0.5, verbose=False)
        return [
            SpeakerTurn(
                speaker=f"Speaker {segment.speaker}",
                start=float(segment.start),
                end=float(segment.end),
            )
            for segment in result.segments
        ]

    def release(self) -> None:
        self._model = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run python -m pytest tests/unit/test_diarization_registry.py tests/unit/test_mlx_sortformer_diarizer.py -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/diarization_registry.py src/core/mlx_sortformer_diarizer.py src/core/diarization_port.py tests/unit/test_diarization_registry.py tests/unit/test_mlx_sortformer_diarizer.py
git commit -m "feat(core): add sortformer diarization runtime contract"
```

### Task 2: Extend Worker IPC with a Diarization Job Kind

**Files:**
- Modify: `src/workers/model_worker.py`
- Modify: `tests/unit/test_worker.py`

- [ ] **Step 1: Write the failing worker diarization tests**

Append to `tests/unit/test_worker.py`:

```python
def test_processes_diarization_job_and_puts_result() -> None:
    job_q = multiprocessing.Queue()
    result_q = multiprocessing.Queue()
    job_q.put(
        WorkerJob(
            uid="diar-1",
            temp_file_path="/tmp/test.wav",
            params={},
            job_kind="diarize",
            requested_diarizer_alias="sortformer-diar",
        )
    )
    job_q.put(None)

    mock_engine = MagicMock()
    mock_diarizer = MagicMock()
    mock_diarizer.diarize_file.return_value = ["speaker-turns"]

    with (
        patch("src.workers.model_worker.create_engine", return_value=mock_engine),
        patch("src.workers.model_worker.create_diarizer", return_value=mock_diarizer),
        pytest.raises(SystemExit),
    ):
        run_worker(job_q, result_q, engine_type="mlx", model_id="qwen", idle_timeout=0)

    result_q.get_nowait()
    result_msg = result_q.get_nowait()
    assert result_msg == ("RESULT", "diar-1", ["speaker-turns"])


def test_puts_error_when_diarization_job_fails() -> None:
    job_q = multiprocessing.Queue()
    result_q = multiprocessing.Queue()
    job_q.put(
        WorkerJob(
            uid="diar-fail",
            temp_file_path="/tmp/test.wav",
            params={},
            job_kind="diarize",
            requested_diarizer_alias="sortformer-diar",
        )
    )
    job_q.put(None)

    mock_engine = MagicMock()
    mock_diarizer = MagicMock()
    mock_diarizer.diarize_file.side_effect = RuntimeError("diarizer failed")

    with (
        patch("src.workers.model_worker.create_engine", return_value=mock_engine),
        patch("src.workers.model_worker.create_diarizer", return_value=mock_diarizer),
        pytest.raises(SystemExit),
    ):
        run_worker(job_q, result_q, engine_type="mlx", model_id="qwen", idle_timeout=0)

    result_q.get_nowait()
    err_msg = result_q.get_nowait()
    assert err_msg[0] == "ERROR"
    assert err_msg[1] == "diar-fail"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run python -m pytest tests/unit/test_worker.py -q
```

Expected: FAIL because `WorkerJob` has no `job_kind`, and `create_diarizer()` does not exist.

- [ ] **Step 3: Implement worker-side diarization dispatch**

Update `src/workers/model_worker.py` with these exact structural changes:

```python
@dataclass
class WorkerJob:
    uid: str
    temp_file_path: str
    params: dict[str, Any]
    requested_model_spec_alias: str | None = field(default=None)
    job_kind: Literal["transcribe", "diarize"] = "transcribe"
    requested_diarizer_alias: str | None = field(default=None)


def create_diarizer(alias: str):
    from src.core.diarization_registry import lookup_diarizer
    from src.core.mlx_sortformer_diarizer import MlxSortformerDiarizer

    spec = lookup_diarizer(alias)
    if spec.runtime != "mlx":
        raise ValueError(f"Unsupported diarization runtime: {spec.runtime}")
    return MlxSortformerDiarizer(spec.model_id)
```

Then branch inside `run_worker()`:

```python
diarizers: dict[str, Any] = {}

if job.job_kind == "transcribe":
    result = engine.transcribe_file(...)
elif job.job_kind == "diarize":
    if job.requested_diarizer_alias is None:
        raise RuntimeError("Diarization job missing requested_diarizer_alias")
    diarizer = diarizers.get(job.requested_diarizer_alias)
    if diarizer is None:
        diarizer = create_diarizer(job.requested_diarizer_alias)
        diarizer.load()
        diarizers[job.requested_diarizer_alias] = diarizer
    result = diarizer.diarize_file(job.temp_file_path)
else:
    raise RuntimeError(f"Unsupported worker job kind: {job.job_kind}")
```

Before every worker exit path, release cached diarizers:

```python
for diarizer in diarizers.values():
    diarizer.release()
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run python -m pytest tests/unit/test_worker.py -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/workers/model_worker.py tests/unit/test_worker.py
git commit -m "feat(worker): add diarization job kind"
```

### Task 3: Submit Diarization Jobs Through the Service and Reserve the Worker

**Files:**
- Modify: `src/services/transcription.py`
- Modify: `tests/unit/test_service.py`

- [ ] **Step 1: Write the failing service tests for worker-backed diarization and deterministic restore**

Add to `tests/unit/test_service.py`:

```python
@pytest.mark.asyncio
async def test_diarize_with_alias_should_submit_diarization_job(funasr_spec):
    svc = _setup_service(funasr_spec)
    svc._result_reader_task = asyncio.create_task(svc._result_reader_loop())

    async def deliver() -> None:
        await asyncio.sleep(0.05)
        svc._result_queue.put(
            ("RESULT", "req-pipeline:diarize", [SpeakerTurn(speaker="Speaker 0", start=0.0, end=1.0)])
        )

    asyncio.create_task(deliver())
    try:
        result = await svc._diarize_with_alias("audio.wav", "req-pipeline", "sortformer-diar")
    finally:
        await _stop_service(svc)

    assert result[0].speaker == "Speaker 0"


@pytest.mark.asyncio
async def test_pipeline_should_hold_worker_reservation_until_restore(funasr_spec):
    from src.core.pipeline_registry import lookup_profile

    svc = _setup_service(funasr_spec)
    profile = replace(lookup_profile("qwen3-sortformer"), requestable=True)
    observed_current_specs = []

    async def fake_submit_worker_job(**kwargs):
        observed_current_specs.append(svc.current_model_spec.alias if svc.current_model_spec else None)
        if kwargs["request_id"].endswith(":transcribe"):
            return {"text": "hello", "segments": [{"text": "hello", "start": 0.0, "end": 1.0}]}
        return [SpeakerTurn(speaker="Speaker 0", start=0.0, end=1.0)]

    svc._submit_worker_job = AsyncMock(side_effect=fake_submit_worker_job)
    svc._switch_worker = AsyncMock(side_effect=lambda spec: setattr(svc, "_current_model_spec", spec))

    result = await svc._run_decoupled_pipeline("audio.wav", {"output_format": "json"}, "req", profile)

    assert result["segments"][0]["speaker"] == "Speaker 0"
    assert observed_current_specs == ["qwen3-asr", "qwen3-asr"]
    assert svc.current_model_spec == funasr_spec
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run python -m pytest tests/unit/test_service.py -q
```

Expected: FAIL because `_diarize_with_alias()` still raises `NotImplementedError`.

- [ ] **Step 3: Implement service-side diarization submission and worker reservation**

Update `src/services/transcription.py` with these exact structural changes:

```python
self._pipeline_lock: asyncio.Lock = asyncio.Lock()
```

Replace `_diarize_with_alias()` with:

```python
async def _diarize_with_alias(
    self,
    temp_file_path: str,
    request_id: str,
    alias: str,
) -> list[SpeakerTurn]:
    result = await self._submit_worker_job(
        temp_file_path=temp_file_path,
        params={},
        request_id=f"{request_id}:diarize",
        model_spec=None,
    )
    if not isinstance(result, list) or not all(isinstance(item, SpeakerTurn) for item in result):
        raise TypeError("Expected diarization result as list[SpeakerTurn]")
    return result
```

Add a new helper that can enqueue non-transcription jobs:

```python
async def _submit_worker_job(
    self,
    temp_file_path: str,
    params: dict[str, object],
    request_id: str,
    model_spec: ModelSpec | None,
    temp_dir: str | None = None,
    job_kind: Literal["transcribe", "diarize"] = "transcribe",
    diarizer_alias: str | None = None,
) -> object:
    ...
    self._job_queue.put_nowait(
        WorkerJob(
            uid=request_id,
            temp_file_path=temp_file_path,
            params=params,
            job_kind=job_kind,
            requested_diarizer_alias=diarizer_alias,
        )
    )
```

Call it from `_diarize_with_alias()` with:

```python
result = await self._submit_worker_job(
    temp_file_path=temp_file_path,
    params={},
    request_id=f"{request_id}:diarize",
    model_spec=None,
    job_kind="diarize",
    diarizer_alias=alias,
)
```

Then reserve the worker across the full pipeline:

```python
async with self._pipeline_lock:
    previous_spec = self._current_model_spec
    try:
        ...
    finally:
        await self._restore_resident_model(previous_spec)
```

Inside `_run_decoupled_pipeline()`, ensure the ASR model is switched before the first worker job:

```python
target_spec = self._lookup_model_spec(profile.transcription_alias)
if self._current_model_spec != target_spec:
    await self._switch_worker(target_spec)
```

Finally, remove the best-effort early return from `_restore_resident_model()` and make it deterministic under the pipeline reservation:

```python
if self._pending:
    raise RuntimeError("Cannot restore resident model while worker jobs are still pending")
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run python -m pytest tests/unit/test_service.py tests/unit/test_worker.py -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/services/transcription.py tests/unit/test_service.py
git commit -m "feat(service): run diarization jobs through worker"
```

### Task 4: Enable the Public Pipeline and Lock Integration Behavior

**Files:**
- Modify: `src/core/pipeline_registry.py`
- Modify: `tests/integration/test_model_api.py`
- Modify: `README.md`
- Modify: `MODELS.md`

- [ ] **Step 1: Write the failing integration expectations**

Update `tests/integration/test_model_api.py` to reflect enabled pipeline behavior:

```python
def test_models_endpoint_should_mark_qwen3_sortformer_requestable(client) -> None:
    response = client.get("/v1/models")

    assert response.status_code == 200
    body = response.json()
    models = {item["alias"]: item for item in body["models"]}
    assert models["qwen3-sortformer"]["requestable"] is True


def test_should_submit_enabled_pipeline_profile_without_patching_requestable_flag() -> None:
    qwen_spec = real_lookup("qwen3-asr")
    mock_service = _make_mock_service(
        qwen_spec.capabilities,
        {
            "text": "pipeline result",
            "segments": [{"text": "hello", "start": 0.0, "end": 1.0, "speaker": "Speaker 0"}],
            "duration": 1.0,
        },
        current_model_spec=qwen_spec,
    )

    with (
        patch("src.main.TranscriptionService", return_value=mock_service),
        patch("src.main.lookup", return_value=qwen_spec),
        TestClient(app) as c,
    ):
        response = c.post(
            "/v1/audio/transcriptions",
            data={"model": "qwen3-sortformer", "output_format": "json"},
            files={"file": _audio_file()},
        )

    assert response.status_code == 200
    mock_service.submit_pipeline.assert_awaited_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run python -m pytest tests/integration/test_model_api.py -q
```

Expected: FAIL because `qwen3-sortformer` is still marked non-requestable.

- [ ] **Step 3: Enable the pipeline profile and update docs**

Change `src/core/pipeline_registry.py`:

```python
    "qwen3-sortformer": PipelineProfile(
        alias="qwen3-sortformer",
        transcription_alias="qwen3-asr",
        diarization_alias="sortformer-diar",
        description=(
            "Decoupled Qwen3-ASR transcription plus Sortformer diarization profile. "
            "Returns speaker-labeled JSON segments via worker-backed Sortformer diarization."
        ),
        capabilities=EngineCapabilities(timestamp=True, diarization=True, language_detect=True),
        requestable=True,
    )
```

Update `README.md` and `MODELS.md` to remove “discovery-only” wording and replace it with:

```markdown
`qwen3-sortformer` is a requestable decoupled pipeline profile backed by Qwen3-ASR transcription plus MLX Sortformer diarization. It remains subject to Apple Silicon memory limits and the current Sortformer threshold defaults.
```

- [ ] **Step 4: Run the focused verification suite**

Run:

```bash
uv run python -m pytest tests/unit/test_diarization_registry.py tests/unit/test_mlx_sortformer_diarizer.py tests/unit/test_worker.py tests/unit/test_service.py tests/integration/test_model_api.py -q
```

Expected: PASS

- [ ] **Step 5: Run the repo baseline verification**

Run:

```bash
uv run python -m pytest tests/unit tests/integration -q
```

Expected: PASS with no regressions.

- [ ] **Step 6: Commit**

```bash
git add src/core/pipeline_registry.py tests/integration/test_model_api.py README.md MODELS.md
git commit -m "feat(api): enable qwen3 sortformer pipeline"
```

## Self-Review

- `SPEC-013` coverage:
  - dedicated diarization worker path: Task 2
  - orchestration and deterministic restore: Task 3
  - requestable public enablement: Task 4
  - validated runtime intake from `SPEC-012`: Task 1
- Placeholder scan:
  - no `TBD`, `TODO`, or “similar to previous task” shortcuts remain
- Type consistency:
  - worker uses `job_kind: Literal["transcribe", "diarize"]`
  - service expects diarization results as `list[SpeakerTurn]`
  - public pipeline remains `qwen3-sortformer` with diarizer alias `sortformer-diar`

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-17-production-decoupled-diarization-pipeline.md`. Two execution options:

1. Subagent-Driven (recommended) - I dispatch a fresh subagent per task, review between tasks, fast iteration
2. Inline Execution - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
