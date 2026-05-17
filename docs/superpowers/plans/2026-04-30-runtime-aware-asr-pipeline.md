# Runtime-Aware ASR Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement SPEC-011 by making model registration runtime-aware, fixing Qwen3-ASR language forwarding, and adding gated decoupled ASR + diarization pipeline primitives.

**Architecture:** Keep the public OpenAI-compatible API stable. Treat engine adapters as runtime-contract wrappers, model registry entries as user-facing capabilities, and pipeline profiles as compositions of ASR and diarization components. Reuse the archived FireRed branch only for architecture patterns, not for FireRed runtime code.

**Tech Stack:** Python 3.11, FastAPI, pytest, `funasr`, `mlx-audio`, multiprocessing worker isolation, MLX Metal, PyTorch MPS.

---

## File Structure

- Modify: `src/core/mlx_engine.py`
  - Add Qwen3-ASR language normalization and forward `language` into `generate_transcription()`.
  - Keep unknown MLX models conservative.

- Modify: `src/core/model_registry.py`
  - Clarify descriptions.
  - Keep Parakeet as validation-gated unless the selected runtime contract is confirmed.

- Create: `src/core/diarization_port.py`
  - Define `SpeakerTurn` and `DiarizationPort`.

- Create: `src/adapters/segment_alignment.py`
  - Pure function for assigning transcript segments to speaker turns by maximum overlap.

- Create: `src/core/pipeline_registry.py`
  - Define non-requestable `qwen3-sortformer` profile.

- Modify: `src/services/transcription.py`
  - Add gated decoupled pipeline submission path after primitives are tested.

- Modify: `src/workers/model_worker.py`
  - Add `job_kind` only if orchestration sends diarization jobs through the existing worker.

- Modify: `src/api/routes.py`
  - Resolve pipeline aliases separately from normal model aliases only after the profile is registered.

- Test: `tests/unit/test_mlx_engine.py`
- Test: `tests/unit/test_model_registry.py`
- Create: `tests/unit/test_diarization_port.py`
- Create: `tests/unit/test_segment_alignment.py`
- Create: `tests/unit/test_pipeline_registry.py`
- Modify: `tests/unit/test_service.py`
- Modify: `tests/unit/test_worker.py` if `job_kind` is introduced.
- Modify: `tests/integration/test_model_api.py` for `/v1/models` visibility once pipeline aliases are exposed.

---

## Task 1: Qwen3-ASR Language Forwarding

**Files:**
- Modify: `src/core/mlx_engine.py`
- Test: `tests/unit/test_mlx_engine.py`

- [ ] **Step 1: Write failing tests**

Add these tests to `tests/unit/test_mlx_engine.py`:

```python
class TestMlxLanguageNormalization:
    def test_should_map_english_code_for_qwen3_asr(self) -> None:
        from src.core.mlx_engine import _normalize_mlx_language

        assert _normalize_mlx_language("mlx-community/Qwen3-ASR-1.7B-8bit", "en") == "English"

    def test_should_map_chinese_code_for_qwen3_asr(self) -> None:
        from src.core.mlx_engine import _normalize_mlx_language

        assert _normalize_mlx_language("mlx-community/Qwen3-ASR-1.7B-8bit", "zh") == "Chinese"

    def test_should_map_cantonese_code_for_qwen3_asr(self) -> None:
        from src.core.mlx_engine import _normalize_mlx_language

        assert _normalize_mlx_language("mlx-community/Qwen3-ASR-1.7B-8bit", "yue") == "Cantonese"

    def test_should_preserve_language_for_non_qwen_mlx_model(self) -> None:
        from src.core.mlx_engine import _normalize_mlx_language

        assert _normalize_mlx_language("mlx-community/whisper-large-v3", "en") == "en"
```

Update the existing `test_transcribe_success_single_chunk` and `test_transcribe_with_verbose` assertions so `generate_transcription()` is expected to receive `language="English"` for Qwen3-ASR.

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
uv run python -m pytest tests/unit/test_mlx_engine.py -q
```

Expected: tests fail because `_normalize_mlx_language` does not exist and `generate_transcription()` does not receive `language`.

- [ ] **Step 3: Implement minimal language normalization**

In `src/core/mlx_engine.py`, add:

```python
_QWEN3_LANGUAGE_ALIASES: dict[str, str] = {
    "auto": "English",
    "en": "English",
    "eng": "English",
    "english": "English",
    "zh": "Chinese",
    "cn": "Chinese",
    "chinese": "Chinese",
    "yue": "Cantonese",
    "cantonese": "Cantonese",
}


def _is_qwen3_asr_model(model_id: str) -> bool:
    return "qwen3-asr" in model_id.lower()


def _normalize_mlx_language(model_id: str, language: str) -> str:
    if not _is_qwen3_asr_model(model_id):
        return language
    normalized_key = language.strip().lower()
    return _QWEN3_LANGUAGE_ALIASES.get(normalized_key, language)
```

Then update the `generate_transcription()` call:

```python
normalized_language = _normalize_mlx_language(self.model_id, language)
result = generate_transcription(
    model=self.model,
    audio=chunk_path,
    format=output_format,
    verbose=verbose,
    language=normalized_language,
)
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
uv run python -m pytest tests/unit/test_mlx_engine.py -q
```

Expected: all `test_mlx_engine.py` tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/core/mlx_engine.py tests/unit/test_mlx_engine.py
git commit -m "fix(core): forward qwen3 asr language prompts"
```

---

## Task 2: Runtime-Aware Registry Clarity

**Files:**
- Modify: `src/core/model_registry.py`
- Test: `tests/unit/test_model_registry.py`

- [ ] **Step 1: Write failing tests**

Add these tests to `tests/unit/test_model_registry.py`:

```python
class TestRuntimeContracts:
    def test_qwen3_asr_should_use_mlx_runtime_contract(self) -> None:
        spec = lookup("qwen3-asr")

        assert spec.engine_type == "mlx"
        assert "mlx-audio" in spec.description.lower()
        assert "MLX Metal" in spec.description

    def test_sensevoice_should_use_funasr_runtime_contract(self) -> None:
        spec = lookup("sensevoice-small")

        assert spec.engine_type == "funasr"
        assert "FunASR" in spec.description
        assert "PyTorch" in spec.description
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
uv run python -m pytest tests/unit/test_model_registry.py -q
```

Expected: description assertions fail.

- [ ] **Step 3: Update registry descriptions only**

In `src/core/model_registry.py`, update descriptions:

```python
description=(
    "Qwen3-ASR 1.7B 8-bit via mlx-audio runtime on MLX Metal. "
    "Quality-first English/Chinese/mixed-language transcription; no diarization."
),
```

```python
description=(
    "SenseVoice Small via FunASR runtime on PyTorch MPS/CPU. "
    "Fast language/emotion-tag model; no timestamps or diarization."
),
```

For `paraformer`:

```python
description=(
    "SEACO Paraformer via FunASR runtime on PyTorch MPS/CPU. "
    "Mandarin-focused transcription with CAM++ diarization."
),
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
uv run python -m pytest tests/unit/test_model_registry.py -q
```

Expected: all registry tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/core/model_registry.py tests/unit/test_model_registry.py
git commit -m "docs(core): clarify model runtime contracts"
```

---

## Task 3: Diarization Port and Segment Alignment

**Files:**
- Create: `src/core/diarization_port.py`
- Create: `src/adapters/segment_alignment.py`
- Create: `tests/unit/test_diarization_port.py`
- Create: `tests/unit/test_segment_alignment.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_diarization_port.py`:

```python
from src.core.diarization_port import SpeakerTurn


def test_speaker_turn_is_immutable_value_object() -> None:
    turn = SpeakerTurn(speaker="Speaker 1", start=1.0, end=2.5)

    assert turn.speaker == "Speaker 1"
    assert turn.start == 1.0
    assert turn.end == 2.5
```

Create `tests/unit/test_segment_alignment.py`:

```python
import pytest

from src.adapters.segment_alignment import align_speakers
from src.core.diarization_port import SpeakerTurn


def test_should_assign_speaker_with_largest_overlap() -> None:
    transcript_segments = [
        {"text": "hello", "start": 0.0, "end": 2.0},
        {"text": "world", "start": 2.0, "end": 4.0},
    ]
    speaker_turns = [
        SpeakerTurn(speaker="Speaker A", start=0.0, end=1.0),
        SpeakerTurn(speaker="Speaker B", start=1.0, end=4.0),
    ]

    aligned = align_speakers(transcript_segments, speaker_turns)

    assert aligned[0]["speaker"] == "Speaker A"
    assert aligned[1]["speaker"] == "Speaker B"


def test_should_assign_unknown_when_no_turn_overlaps() -> None:
    aligned = align_speakers(
        [{"text": "gap", "start": 10.0, "end": 11.0}],
        [SpeakerTurn(speaker="Speaker A", start=0.0, end=1.0)],
    )

    assert aligned[0]["speaker"] == "Unknown"


def test_should_reject_missing_timestamp() -> None:
    with pytest.raises(ValueError, match="segment missing required timestamp"):
        align_speakers([{"text": "bad", "start": 0.0}], [])
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
uv run python -m pytest tests/unit/test_diarization_port.py tests/unit/test_segment_alignment.py -q
```

Expected: import failures because files do not exist.

- [ ] **Step 3: Implement the port and alignment helper**

Create `src/core/diarization_port.py`:

```python
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SpeakerTurn:
    speaker: str
    start: float
    end: float


class DiarizationPort(Protocol):
    def load(self) -> None:
        raise NotImplementedError

    def diarize_file(self, file_path: str) -> list[SpeakerTurn]:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError
```

Create `src/adapters/segment_alignment.py`:

```python
from src.core.diarization_port import SpeakerTurn


def _read_timestamp(segment: dict[str, object], field: str) -> float:
    if field not in segment:
        raise ValueError(f"segment missing required timestamp: {field}")
    value = segment[field]
    if isinstance(value, bool):
        raise ValueError(f"segment timestamp must be numeric: {field}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"segment timestamp must be numeric: {field}") from exc


def align_speakers(
    transcript_segments: list[dict[str, object]],
    speaker_turns: list[SpeakerTurn],
) -> list[dict[str, object]]:
    aligned: list[dict[str, object]] = []
    for segment in transcript_segments:
        start = _read_timestamp(segment, "start")
        end = _read_timestamp(segment, "end")
        best_speaker = "Unknown"
        best_overlap = 0.0

        for turn in speaker_turns:
            overlap = max(0.0, min(end, turn.end) - max(start, turn.start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn.speaker

        merged = dict(segment)
        merged["speaker"] = best_speaker
        aligned.append(merged)

    return aligned
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
uv run python -m pytest tests/unit/test_diarization_port.py tests/unit/test_segment_alignment.py -q
```

Expected: all new tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/core/diarization_port.py src/adapters/segment_alignment.py tests/unit/test_diarization_port.py tests/unit/test_segment_alignment.py
git commit -m "feat(core): add diarization port and segment alignment"
```

---

## Task 4: Pipeline Registry with Gated qwen3-sortformer

**Files:**
- Create: `src/core/pipeline_registry.py`
- Create: `tests/unit/test_pipeline_registry.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_pipeline_registry.py`:

```python
import pytest

from src.core.pipeline_registry import list_all_profiles, lookup_profile


def test_should_resolve_qwen3_sortformer_profile() -> None:
    profile = lookup_profile("qwen3-sortformer")

    assert profile.alias == "qwen3-sortformer"
    assert profile.transcription_alias == "qwen3-asr"
    assert profile.diarization_alias == "sortformer-diar"
    assert profile.capabilities.timestamp is True
    assert profile.capabilities.diarization is True
    assert profile.requestable is False


def test_should_list_profiles_sorted_by_alias() -> None:
    aliases = [profile.alias for profile in list_all_profiles()]

    assert aliases == sorted(aliases)
    assert "qwen3-sortformer" in aliases


def test_should_raise_for_unknown_profile() -> None:
    with pytest.raises(KeyError, match="Unknown pipeline profile"):
        lookup_profile("missing-profile")
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
uv run python -m pytest tests/unit/test_pipeline_registry.py -q
```

Expected: import failure because `pipeline_registry.py` does not exist.

- [ ] **Step 3: Implement profile registry**

Create `src/core/pipeline_registry.py`:

```python
from dataclasses import dataclass

from src.core.base_engine import EngineCapabilities


@dataclass(frozen=True)
class PipelineProfile:
    alias: str
    transcription_alias: str
    diarization_alias: str
    description: str
    capabilities: EngineCapabilities
    requestable: bool = False


_REGISTRY: dict[str, PipelineProfile] = {
    "qwen3-sortformer": PipelineProfile(
        alias="qwen3-sortformer",
        transcription_alias="qwen3-asr",
        diarization_alias="sortformer-diar",
        description=(
            "Decoupled Qwen3-ASR transcription plus Sortformer diarization profile. "
            "Discovery-only until Sortformer runtime validation passes."
        ),
        capabilities=EngineCapabilities(timestamp=True, diarization=True, language_detect=True),
        requestable=False,
    )
}


def lookup_profile(alias: str) -> PipelineProfile:
    try:
        return _REGISTRY[alias]
    except KeyError as exc:
        raise KeyError(f"Unknown pipeline profile: '{alias}'") from exc


def list_all_profiles() -> list[PipelineProfile]:
    return sorted(_REGISTRY.values(), key=lambda profile: profile.alias)
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
uv run python -m pytest tests/unit/test_pipeline_registry.py -q
```

Expected: all profile tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/core/pipeline_registry.py tests/unit/test_pipeline_registry.py
git commit -m "feat(core): add gated qwen3 sortformer profile"
```

---

## Task 5: API Discovery for Pipeline Profiles

**Files:**
- Modify: `src/api/routes.py`
- Test: `tests/integration/test_model_api.py`

- [ ] **Step 1: Write failing integration test**

Add to `tests/integration/test_model_api.py`:

```python
def test_models_endpoint_should_include_non_requestable_pipeline_profiles(client):
    response = client.get("/v1/models")

    assert response.status_code == 200
    body = response.json()
    models = {item["alias"]: item for item in body["models"]}
    assert "qwen3-sortformer" in models
    assert models["qwen3-sortformer"]["capabilities"]["diarization"] is True
    assert models["qwen3-sortformer"]["requestable"] is False
```

If `ModelInfo` does not yet expose `requestable`, this test should fail.

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
uv run python -m pytest tests/integration/test_model_api.py -q
```

Expected: response does not include pipeline profiles or `requestable`.

- [ ] **Step 3: Add requestable metadata and merge profiles into discovery**

In `src/api/routes.py`, update `ModelInfo`:

```python
class ModelInfo(BaseModel):
    alias: str
    model_id: str
    engine_type: str
    description: str
    capabilities: dict[str, bool]
    requestable: bool = True
```

Import profiles:

```python
from src.core.pipeline_registry import list_all_profiles
```

Update `list_models()` to append profile entries:

```python
model_entries = [
    ModelInfo(
        alias=spec.alias,
        model_id=spec.model_id,
        engine_type=spec.engine_type,
        description=spec.description,
        capabilities={k: v for k, v in asdict(spec.capabilities).items()},
        requestable=True,
    )
    for spec in list_all()
]
profile_entries = [
    ModelInfo(
        alias=profile.alias,
        model_id=f"{profile.transcription_alias}+{profile.diarization_alias}",
        engine_type="pipeline",
        description=profile.description,
        capabilities={k: v for k, v in asdict(profile.capabilities).items()},
        requestable=profile.requestable,
    )
    for profile in list_all_profiles()
]
```

Return `models=sorted(model_entries + profile_entries, key=lambda item: item.alias)`.

- [ ] **Step 4: Run test to verify GREEN**

Run:

```bash
uv run python -m pytest tests/integration/test_model_api.py -q
```

Expected: model API integration tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/api/routes.py tests/integration/test_model_api.py
git commit -m "feat(api): expose gated pipeline profiles"
```

---

## Task 6: Keep Pipeline POST Gated

**Files:**
- Modify: `src/api/routes.py`
- Test: `tests/integration/test_model_api.py`

- [ ] **Step 1: Write failing test**

Add to `tests/integration/test_model_api.py`:

```python
def test_should_return_501_for_non_requestable_pipeline_profile(client, tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake wav")

    with audio_path.open("rb") as audio:
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", audio, "audio/wav")},
            data={"model": "qwen3-sortformer"},
        )

    assert response.status_code == 501
    assert "not enabled" in response.json()["detail"]
```

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
uv run python -m pytest tests/integration/test_model_api.py -q
```

Expected: request fails with 400 unknown model, not 501.

- [ ] **Step 3: Resolve pipeline aliases separately from models**

In `src/api/routes.py`, add:

```python
from src.core.pipeline_registry import PipelineProfile, lookup_profile
```

Add helper:

```python
def _resolve_pipeline_profile(model: str | None) -> PipelineProfile | None:
    if is_passthrough(model):
        return None
    try:
        return lookup_profile(model)  # type: ignore[arg-type]
    except KeyError:
        return None
```

Before `_resolve_model(model)`, resolve profile:

```python
resolved_profile = _resolve_pipeline_profile(model)
if resolved_profile is not None and not resolved_profile.requestable:
    raise HTTPException(
        status_code=501,
        detail=f"Pipeline profile '{resolved_profile.alias}' is discoverable but not enabled for POST yet.",
    )
```

Only call `_resolve_model(model)` if no profile was found.

- [ ] **Step 4: Run test to verify GREEN**

Run:

```bash
uv run python -m pytest tests/integration/test_model_api.py -q
```

Expected: pipeline POST gate returns 501 and existing model requests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/api/routes.py tests/integration/test_model_api.py
git commit -m "feat(api): gate pipeline profile requests"
```

---

## Task 7: Serial Pipeline Orchestration Internals

**Files:**
- Modify: `src/services/transcription.py`
- Modify: `src/workers/model_worker.py`
- Test: `tests/unit/test_service.py`
- Test: `tests/unit/test_worker.py`

- [ ] **Step 1: Write unit tests for service orchestration**

Add focused tests to `tests/unit/test_service.py` after existing `TestTranscriptionService` tests:

```python
@pytest.mark.asyncio
async def test_decoupled_pipeline_should_align_speakers_and_restore_previous_model(funasr_spec):
    from src.core.pipeline_registry import lookup_profile

    svc = _setup_service(funasr_spec)
    profile = lookup_profile("qwen3-sortformer")

    async def fake_transcribe(temp_file_path, params, request_id, alias):
        assert alias == "qwen3-asr"
        return {"text": "hello world", "segments": [{"text": "hello", "start": 0.0, "end": 1.0}]}

    async def fake_diarize(temp_file_path, request_id, alias):
        assert alias == "sortformer-diar"
        return [SpeakerTurn(speaker="Speaker A", start=0.0, end=1.0)]

    svc._transcribe_with_alias = fake_transcribe
    svc._diarize_with_alias = fake_diarize
    svc._restore_resident_model = AsyncMock()

    result = await svc._run_decoupled_pipeline(
        "audio.wav",
        {"output_format": "json"},
        "req-pipeline",
        profile,
    )

    assert result["segments"][0]["speaker"] == "Speaker A"
    svc._restore_resident_model.assert_awaited_once_with(funasr_spec)
```

Required imports:

```python
from unittest.mock import AsyncMock
from src.core.diarization_port import SpeakerTurn
```

- [ ] **Step 2: Run service test to verify RED**

Run:

```bash
uv run python -m pytest tests/unit/test_service.py -q
```

Expected: failure because `_run_decoupled_pipeline`, `_transcribe_with_alias`, and `_diarize_with_alias` do not exist.

- [ ] **Step 3: Implement minimal internal orchestration**

In `src/services/transcription.py`, add imports:

```python
from src.adapters.segment_alignment import align_speakers
from src.core.diarization_port import SpeakerTurn
from src.core.pipeline_registry import PipelineProfile
```

Add a result alias after the imports:

```python
TranscriptionResult = str | dict[str, Any]
```

Add these methods to `TranscriptionService`:

```python
async def _transcribe_with_alias(
    self,
    temp_file_path: str,
    params: dict[str, Any],
    request_id: str,
    alias: str,
) -> TranscriptionResult:
    result = await self._submit_worker_job(
        temp_file_path=temp_file_path,
        params=params,
        request_id=f"{request_id}:transcribe",
        model_spec=self._lookup_model_spec(alias),
    )
    if isinstance(result, str):
        return result
    return cast(dict[str, Any], result)


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
        model_spec=self._lookup_model_spec(alias),
    )
    return cast(list[SpeakerTurn], result)


def _lookup_model_spec(self, alias: str) -> ModelSpec:
    from src.core.model_registry import lookup

    return lookup(alias)


async def _run_decoupled_pipeline(
    self,
    temp_file_path: str,
    params: dict[str, Any],
    request_id: str,
    profile: PipelineProfile,
) -> TranscriptionResult:
    previous_spec = self._current_model_spec
    try:
        transcript_result = await self._transcribe_with_alias(
            temp_file_path,
            params,
            request_id,
            profile.transcription_alias,
        )
        if not isinstance(transcript_result, dict):
            return transcript_result

        try:
            speaker_turns = await self._diarize_with_alias(
                temp_file_path,
                request_id,
                profile.diarization_alias,
            )
        except Exception as exc:
            if self._is_worker_lifecycle_error(exc):
                raise
            self.logger.warning(
                "[%s] Decoupled pipeline diarization failed for profile %s; returning transcript-only result: %s",
                request_id,
                profile.alias,
                exc,
            )
            return transcript_result

        segments = transcript_result.get("segments")
        if not isinstance(segments, list):
            return transcript_result

        try:
            aligned_segments = align_speakers(
                cast(list[dict[str, object]], segments),
                speaker_turns,
            )
        except (TypeError, ValueError) as exc:
            self.logger.warning(
                "[%s] Decoupled pipeline alignment failed for profile %s; returning transcript-only result: %s",
                request_id,
                profile.alias,
                exc,
            )
            return transcript_result

        return {**transcript_result, "segments": aligned_segments}
    finally:
        await self._restore_resident_model(previous_spec)


async def _restore_resident_model(self, previous_spec: ModelSpec | None) -> None:
    async with self._spawn_lock:
        current_spec = self._current_model_spec
        if current_spec == previous_spec:
            return
        if previous_spec is None:
            await self._shutdown_worker()
            self._current_model_spec = None
            return
        await self._switch_worker(previous_spec)


@staticmethod
def _is_worker_lifecycle_error(exc: Exception) -> bool:
    return isinstance(exc, RuntimeError) and str(exc).startswith(
        (
            "Worker terminated",
            "Worker process died unexpectedly",
            "Worker failed to start",
            "Worker failed to load model",
            "Worker sent unexpected startup message",
        )
    )
```

Refactor the existing `submit()` queue-registration block into a reusable helper used by `_transcribe_with_alias()` and `_diarize_with_alias()`:

```python
async def _submit_worker_job(
    self,
    temp_file_path: str,
    params: dict[str, Any],
    request_id: str,
    model_spec: ModelSpec | None,
    temp_dir: str | None = None,
) -> object:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[object] = loop.create_future()

    try:
        async with self._spawn_lock:
            if model_spec is not None and model_spec != self._current_model_spec:
                await self._switch_worker(model_spec)
            elif not self.model_loaded:
                await self._spawn_worker()

            if self._job_queue is None:
                raise RuntimeError("Job queue is None after successful spawn — this is a bug")

            self._pending[request_id] = future
            if temp_dir is not None:
                self._temp_dirs[request_id] = temp_dir

            self._job_queue.put_nowait(
                WorkerJob(
                    uid=request_id,
                    temp_file_path=temp_file_path,
                    params=params,
                )
            )
    except BaseException:
        self._pending.pop(request_id, None)
        self._temp_dirs.pop(request_id, None)
        raise

    return await future
```

Then simplify `submit()` so it writes the upload to disk and calls `_submit_worker_job(temp_file_path=temp_path, params=params, request_id=request_id, model_spec=model_spec, temp_dir=temp_dir)`. Keep this task internal-only; do not wire public POST dispatch here unless the profile becomes requestable.

- [ ] **Step 4: Run service tests to verify GREEN**

Run:

```bash
uv run python -m pytest tests/unit/test_service.py -q
```

Expected: service tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/services/transcription.py src/workers/model_worker.py tests/unit/test_service.py tests/unit/test_worker.py
git commit -m "feat(service): add internal decoupled pipeline orchestration"
```

---

## Task 8: Documentation Updates

**Files:**
- Modify: `README.md`
- Modify: `MODELS.md`
- Modify: `AGENTS.md` only if new constraints or known pitfalls are discovered during implementation.

- [ ] **Step 1: Update docs after code behavior is verified**

Document these user-facing rules:

```text
Qwen3-ASR: quality-first Chinese/English/mixed-language MLX path.
Parakeet: validation-gated throughput-first English/European-language MLX path.
Paraformer: Mandarin-focused FunASR path with CAM++ diarization.
SenseVoice: fast FunASR path for language/emotion tags, not diarization.
qwen3-sortformer: discovery-only profile until Sortformer runtime validation passes.
```

- [ ] **Step 2: Run docs-adjacent tests**

Run:

```bash
uv run python -m pytest tests/integration/test_model_api.py tests/unit/test_model_registry.py -q
```

Expected: docs-visible model descriptions and `/v1/models` behavior remain consistent.

- [ ] **Step 3: Commit**

```bash
git add README.md MODELS.md AGENTS.md
git commit -m "docs: document runtime-aware asr model selection"
```

---

## Final Verification

Run:

```bash
uv run python -m pytest tests/unit tests/integration -q
```

Expected: all unit and integration tests pass.

Run:

```bash
git status --short
```

Expected: only intentionally uncommitted files remain. If `uv.lock` is still dirty from earlier dependency inspection and was not part of this plan, leave it unstaged and call it out.

---

## Self-Review

- SPEC coverage: Phase 1 maps to Qwen3 language forwarding; Phase 2 maps to runtime-aware registry semantics; Phases 3-7 map to gated decoupled pipeline primitives and internal orchestration; Phase 8 maps to public docs.
- Scope: this plan deliberately keeps `qwen3-sortformer` non-requestable until Sortformer runtime validation passes.
- Risk: Task 7 is the largest change. If it grows beyond focused internal orchestration, split it into gate/lifecycle cleanup and alignment/degradation subtasks before implementation.
