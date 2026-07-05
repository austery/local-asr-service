# Apple Speech Service Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the SPEC-014 Phase 1 closeout and implement the smallest Phase 2 service integration so `model=apple-speech` can run through `/v1/audio/transcriptions` without changing the existing FunASR, MLX, or pipeline behavior.

**Architecture:** Keep the Swift `apple-speech-worker` as the Apple Speech process boundary. Python adds a typed core port, a narrow `AppleSpeechEngine`, registry entries, and a service-layer sidecar path that bypasses the existing multiprocessing model worker because the Swift worker already owns process isolation and macOS Speech framework permissions.

**Tech Stack:** Python 3.11, FastAPI, pytest, SwiftPM executable contract tests, Apple Speech framework sidecar CLI, `uv`.

---

## Current Baseline

- Worktree: `/Users/leipeng/Documents/Projects/local-asr-service/.worktrees/spec-014-apple-speech-phase1-plan`
- Branch: `codex/spec-014-apple-speech-phase1-plan`
- Python baseline: `uv run python -m pytest` -> `252 passed, 1 warning`
- Swift contract baseline: `swift run --package-path apple-speech-worker apple-speech-worker-contract-tests` -> `contract-tests: passed`
- SwiftPM caveat: run Swift commands outside the Codex filesystem sandbox. Inside the sandbox, SwiftPM fails at manifest evaluation with `sandbox-exec: sandbox_apply: Operation not permitted`.

## Scope Boundary

In scope:

- Sync SPEC-014 Phase 1 status with code already present on `main`.
- Add the `apple-speech` model registry entry.
- Add a typed core Apple Speech port.
- Implement `AppleSpeechEngine` as an `ASREngine` adapter around `AppleSpeechWorkerClient`.
- Route explicit `model=apple-speech` requests through the Swift sidecar path.
- Keep `apple-dictation` hidden until the Swift runtime supports `DictationTranscriber` transcription.
- Preserve existing `/v1/audio/transcriptions` response shape: rich JSON by default, segments only for JSON/verbose JSON, no speaker labels for Apple-only mode.
- Add API, service, registry, engine, and documentation tests.

Out of scope:

- No live microphone or daemon IPC.
- No Apple diarization promotion.
- No forced alignment or Sortformer integration for Apple output.
- No default model change.
- No rewrite of the existing `model_worker.py` multiprocessing path.

## File Map

- Modify `docs/SPEC-014-Apple-SpeechAnalyzer-Integration.md`: mark Phase 1 implemented where evidence already exists, and clarify Phase 2 starts from service integration.
- Create `src/core/apple_speech_port.py`: shared typed request, segment, metadata, result, and error types.
- Modify `src/adapters/apple_speech_worker_client.py`: import the shared port types instead of declaring duplicate dataclasses.
- Create `src/core/apple_speech_engine.py`: `ASREngine` implementation that maps service params to worker CLI calls.
- Modify `src/core/model_registry.py`: add `EngineType = Literal["funasr", "mlx", "apple-speech"]` and built-in Apple aliases.
- Modify `src/services/transcription.py`: add a sidecar path for Apple model specs, queue accounting for sidecar requests, and error cleanup.
- Modify `src/api/routes.py`: document Apple aliases in OpenAPI text and map Apple sidecar failures to clear HTTP errors.
- Modify `MODELS.md`: document Apple Speech as ASR-only, requestable, and macOS 26-gated.
- Add `tests/unit/test_apple_speech_engine.py`: mapping, locale resolution, no fake speakers, and worker error propagation.
- Modify `tests/unit/test_apple_speech_worker_client.py`: ensure client still parses shared port types.
- Modify `tests/unit/test_model_registry.py`: Apple aliases and capabilities.
- Add `tests/unit/test_service_apple_speech.py`: service sidecar path bypasses multiprocessing worker and cleans temp files.
- Modify `tests/integration/test_model_api.py`: model list and POST routing behavior for Apple aliases.
- Modify `tests/unit/test_architecture_fitness.py`: allow the new core/adapter files if the static allowlist requires it.

---

### Task 1: Sync SPEC-014 Phase 1 Status

**Files:**
- Modify: `docs/SPEC-014-Apple-SpeechAnalyzer-Integration.md`

- [ ] **Step 1: Update Phase 1 checkboxes and evidence**

Edit the Phase 1 section so it records what is already implemented on `main`:

```markdown
### Phase 1: `apple-speech-worker` CLI

- [x] Implement `capabilities` command.
- [x] Implement `prepare` command.
- [x] Implement `transcribe` command.
- [x] Convert input audio to the analyzer's required format when needed.
- [x] Return deterministic JSON.
- [x] Separate stdout JSON from stderr logs.
- [x] Add timeout and structured error codes.
- [x] Add CLI contract tests that assert stdout is valid JSON with no human log lines.

Acceptance: Python can call the CLI and parse stable JSON.

Phase 1 evidence from 2026-07-04:

- Swift package builds in a non-sandboxed shell.
- `swift run --package-path apple-speech-worker apple-speech-worker-contract-tests` prints `contract-tests: passed`.
- `tests/unit/test_apple_speech_worker_client.py` verifies Python subprocess JSON parsing, stderr failure handling, invalid stdout rejection, timeout handling, and missing binary errors.
- `tests/unit/test_apple_speech_worker_source_contracts.py` verifies the live runtime cancels its result collection task on early exit.

Phase 1 decision: **GO for Phase 2 Python service integration**. Runtime verification for Apple Speech framework capability discovery and real transcription must continue outside the Codex filesystem sandbox.
```

- [ ] **Step 2: Verify doc diff**

Run:

```bash
git diff -- docs/SPEC-014-Apple-SpeechAnalyzer-Integration.md
```

Expected: only Phase 1 status/evidence changes.

- [ ] **Step 3: Commit**

```bash
git add docs/SPEC-014-Apple-SpeechAnalyzer-Integration.md
git commit -m "docs: sync apple speech phase 1 status"
```

---

### Task 2: Add Shared Apple Speech Port Types

**Files:**
- Create: `src/core/apple_speech_port.py`
- Modify: `src/adapters/apple_speech_worker_client.py`
- Modify: `tests/unit/test_apple_speech_worker_client.py`

- [ ] **Step 1: Write the failing import test**

Append this test to `tests/unit/test_apple_speech_worker_client.py`:

```python
from src.core.apple_speech_port import TranscriptionResult


def test_transcribe_returns_shared_port_type() -> None:
    completed = subprocess.CompletedProcess(
        args=["worker"],
        returncode=0,
        stdout=(
            '{"jobId":null,"engine":"apple-speech","module":"speechTranscriber",'
            '"locale":"en-US","text":"hello world",'
            '"segments":[{"id":0,"start":0.0,"end":1.0,"text":"hello world",'
            '"isFinal":true,"confidence":null,"speaker":null}],'
            '"metadata":{"local":true,"appleApi":true,"volatileIncluded":false,'
            '"timingGranularity":"segment","assetManagedBySystem":true,"durationMs":42}}'
        ),
        stderr="",
    )

    with patch("src.adapters.apple_speech_worker_client.subprocess.run", return_value=completed):
        result = AppleSpeechWorkerClient(Path("/tmp/apple-speech-worker")).transcribe(
            input_path=Path("/tmp/audio.wav"),
            locale="en-US",
            module="speechTranscriber",
        )

    assert isinstance(result, TranscriptionResult)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run python -m pytest tests/unit/test_apple_speech_worker_client.py::test_transcribe_returns_shared_port_type -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.core.apple_speech_port'`.

- [ ] **Step 3: Create shared port module**

Create `src/core/apple_speech_port.py`:

```python
"""Typed port contract for the Apple Speech sidecar worker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

AppleSpeechModule = Literal["speechTranscriber", "dictationTranscriber"]
TimingGranularity = Literal["none", "segment", "word", "unknown"]


class AppleSpeechError(RuntimeError):
    """Base error for Apple Speech sidecar integration failures."""


class AppleSpeechWorkerUnavailableError(AppleSpeechError):
    """Raised when the worker binary cannot be executed or times out."""


class AppleSpeechWorkerResponseError(AppleSpeechError):
    """Raised when the worker returns invalid JSON or an unsuccessful result."""


@dataclass(frozen=True)
class WorkerModules:
    speech_transcriber: bool
    dictation_transcriber: bool
    speech_detector: bool


@dataclass(frozen=True)
class WorkerCapabilities:
    runtime: str
    platform: str
    os_version: str
    supported: bool
    supported_locales: list[str]
    modules: WorkerModules
    notes: list[str]


@dataclass(frozen=True)
class AssetPreparationResult:
    locale: str
    module: AppleSpeechModule
    supported: bool
    allocated: bool
    downloaded: bool
    duration_ms: int


@dataclass(frozen=True)
class AppleSpeechRequest:
    input_path: Path
    locale: str
    module: AppleSpeechModule
    include_audio_ranges: bool = True
    include_volatile: bool = False


@dataclass(frozen=True)
class TranscriptionSegment:
    id: int
    start: float
    end: float
    text: str
    is_final: bool
    confidence: float | None
    speaker: str | None


@dataclass(frozen=True)
class TranscriptionMetadata:
    local: bool
    apple_api: bool
    volatile_included: bool
    timing_granularity: TimingGranularity
    asset_managed_by_system: bool
    duration_ms: int


@dataclass(frozen=True)
class TranscriptionResult:
    job_id: str | None
    engine: str
    module: AppleSpeechModule
    locale: str
    text: str
    segments: list[TranscriptionSegment]
    metadata: TranscriptionMetadata
```

- [ ] **Step 4: Refactor client imports**

In `src/adapters/apple_speech_worker_client.py`, remove the local dataclasses and import the shared types:

```python
from src.core.apple_speech_port import (
    AppleSpeechModule,
    AppleSpeechWorkerResponseError,
    AppleSpeechWorkerUnavailableError,
    AssetPreparationResult,
    TranscriptionMetadata,
    TranscriptionResult,
    TranscriptionSegment,
    WorkerCapabilities,
    WorkerModules,
)

AppleSpeechWorkerError = AppleSpeechWorkerResponseError
```

Update method signatures:

```python
def prepare(self, locale: str, module: AppleSpeechModule) -> AssetPreparationResult:
    ...

def transcribe(
    self,
    input_path: Path,
    locale: str,
    module: AppleSpeechModule,
    audio_time_ranges: bool = True,
    include_volatile: bool = False,
) -> TranscriptionResult:
    ...
```

Change timeout and missing-binary branches to raise `AppleSpeechWorkerUnavailableError`:

```python
except subprocess.TimeoutExpired as exc:
    raise AppleSpeechWorkerUnavailableError(
        f"apple-speech-worker timed out after {self.timeout_seconds:.1f}s"
    ) from exc
except OSError as exc:
    raise AppleSpeechWorkerUnavailableError(
        f"failed to run apple-speech-worker at {self.worker_path}: {exc}"
    ) from exc
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
uv run python -m pytest tests/unit/test_apple_speech_worker_client.py -q
```

Expected: all worker client tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/core/apple_speech_port.py src/adapters/apple_speech_worker_client.py tests/unit/test_apple_speech_worker_client.py
git commit -m "feat: add apple speech port types"
```

---

### Task 3: Implement AppleSpeechEngine

**Files:**
- Create: `src/core/apple_speech_engine.py`
- Create: `tests/unit/test_apple_speech_engine.py`

- [ ] **Step 1: Write failing engine tests**

Create `tests/unit/test_apple_speech_engine.py`:

```python
from pathlib import Path

import pytest

from src.core.apple_speech_engine import AppleSpeechEngine
from src.core.apple_speech_port import (
    AppleSpeechWorkerResponseError,
    TranscriptionMetadata,
    TranscriptionResult,
    TranscriptionSegment,
)


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, str, str, bool, bool]] = []

    def transcribe(
        self,
        input_path: Path,
        locale: str,
        module: str,
        audio_time_ranges: bool = True,
        include_volatile: bool = False,
    ) -> TranscriptionResult:
        self.calls.append((input_path, locale, module, audio_time_ranges, include_volatile))
        return TranscriptionResult(
            job_id=None,
            engine="apple-speech",
            module="speechTranscriber",
            locale=locale,
            text="hello world",
            segments=[
                TranscriptionSegment(
                    id=0,
                    start=0.0,
                    end=1.25,
                    text="hello world",
                    is_final=True,
                    confidence=None,
                    speaker=None,
                )
            ],
            metadata=TranscriptionMetadata(
                local=True,
                apple_api=True,
                volatile_included=False,
                timing_granularity="segment",
                asset_managed_by_system=True,
                duration_ms=1250,
            ),
        )

    def capabilities(self) -> object:
        return object()


def test_transcribe_file_returns_service_response_shape() -> None:
    client = FakeClient()
    engine = AppleSpeechEngine(client=client, module="speechTranscriber")

    result = engine.transcribe_file("/tmp/audio.wav", language="zh-CN", output_format="json")

    assert result == {
        "text": "hello world",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 1.25,
                "text": "hello world",
                "speaker": None,
            }
        ],
        "duration": 1.25,
        "language": "zh-CN",
    }
    assert client.calls == [(Path("/tmp/audio.wav"), "zh-CN", "speechTranscriber", True, False)]


def test_transcribe_file_maps_short_language_codes_to_apple_locales() -> None:
    client = FakeClient()
    engine = AppleSpeechEngine(client=client, module="speechTranscriber")

    engine.transcribe_file("/tmp/audio.wav", language="zh", output_format="json")
    engine.transcribe_file("/tmp/audio.wav", language="en", output_format="json")
    engine.transcribe_file("/tmp/audio.wav", language="auto", output_format="json")

    assert [call[1] for call in client.calls] == ["zh-CN", "en-US", "en-US"]


def test_transcribe_file_never_synthesizes_speaker_labels() -> None:
    client = FakeClient()
    engine = AppleSpeechEngine(client=client, module="speechTranscriber")

    result = engine.transcribe_file("/tmp/audio.wav", language="en-US", output_format="json")

    assert isinstance(result, dict)
    segments = result["segments"]
    assert isinstance(segments, list)
    assert segments[0]["speaker"] is None


def test_transcribe_file_returns_plain_text_for_txt_output() -> None:
    client = FakeClient()
    engine = AppleSpeechEngine(client=client, module="speechTranscriber")

    result = engine.transcribe_file("/tmp/audio.wav", language="en-US", output_format="txt")

    assert result == "hello world"


def test_worker_errors_are_not_hidden() -> None:
    class FailingClient(FakeClient):
        def transcribe(
            self,
            input_path: Path,
            locale: str,
            module: str,
            audio_time_ranges: bool = True,
            include_volatile: bool = False,
        ) -> TranscriptionResult:
            raise AppleSpeechWorkerResponseError("unsupported locale: fr-FR")

    engine = AppleSpeechEngine(client=FailingClient(), module="speechTranscriber")

    with pytest.raises(AppleSpeechWorkerResponseError, match="unsupported locale"):
        engine.transcribe_file("/tmp/audio.wav", language="fr-FR", output_format="json")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run python -m pytest tests/unit/test_apple_speech_engine.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.core.apple_speech_engine'`.

- [ ] **Step 3: Implement engine**

Create `src/core/apple_speech_engine.py`:

```python
"""ASREngine adapter for the Apple Speech sidecar worker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.adapters.apple_speech_worker_client import AppleSpeechWorkerClient
from src.core.base_engine import EngineCapabilities
from src.core.apple_speech_port import AppleSpeechModule, TranscriptionResult


@dataclass(frozen=True)
class AppleSpeechEngineConfig:
    worker_path: Path
    timeout_seconds: float = 120.0
    default_locale: str = "en-US"


class AppleSpeechEngine:
    """Blocking ASR engine wrapper around the Swift Apple Speech worker CLI."""

    def __init__(
        self,
        *,
        client: AppleSpeechWorkerClient,
        module: AppleSpeechModule,
        default_locale: str = "en-US",
    ) -> None:
        self._client = client
        self._module = module
        self._default_locale = default_locale

    @classmethod
    def from_config(
        cls,
        config: AppleSpeechEngineConfig,
        module: AppleSpeechModule,
    ) -> AppleSpeechEngine:
        return cls(
            client=AppleSpeechWorkerClient(
                config.worker_path,
                timeout_seconds=config.timeout_seconds,
            ),
            module=module,
            default_locale=config.default_locale,
        )

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            timestamp=True,
            diarization=False,
            emotion_tags=False,
            language_detect=False,
        )

    def load(self) -> None:
        self._client.capabilities()

    def transcribe_file(
        self,
        file_path: str,
        language: str = "auto",
        output_format: str = "json",
        with_timestamp: bool = False,
        include_volatile: bool = False,
        **_kwargs: object,
    ) -> str | dict[str, object]:
        locale = self._resolve_locale(language)
        result = self._client.transcribe(
            input_path=Path(file_path),
            locale=locale,
            module=self._module,
            audio_time_ranges=True,
            include_volatile=include_volatile,
        )
        if output_format == "txt":
            return result.text
        return self._to_service_dict(result)

    def release(self) -> None:
        return None

    def _resolve_locale(self, language: str) -> str:
        normalized = language.strip()
        if not normalized or normalized.lower() == "auto":
            return self._default_locale
        lowered = normalized.lower()
        if lowered == "zh":
            return "zh-CN"
        if lowered == "en":
            return "en-US"
        return normalized

    @staticmethod
    def _to_service_dict(result: TranscriptionResult) -> dict[str, object]:
        duration = 0.0
        if result.segments:
            duration = max(segment.end for segment in result.segments)
        return {
            "text": result.text,
            "segments": [
                {
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "speaker": None,
                }
                for segment in result.segments
            ],
            "duration": duration,
            "language": result.locale,
        }
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run python -m pytest tests/unit/test_apple_speech_engine.py -q
```

Expected: all AppleSpeechEngine tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/core/apple_speech_engine.py tests/unit/test_apple_speech_engine.py
git commit -m "feat: add apple speech engine adapter"
```

---

### Task 4: Register Apple Model Aliases

**Files:**
- Modify: `src/core/model_registry.py`
- Modify: `src/config.py`
- Modify: `tests/unit/test_model_registry.py`
- Modify: `MODELS.md`

- [ ] **Step 1: Write failing registry tests**

Append to `tests/unit/test_model_registry.py`:

```python
def test_apple_speech_alias_should_use_apple_speech_runtime_contract() -> None:
    spec = lookup("apple-speech")

    assert spec.engine_type == "apple-speech"
    assert spec.model_id == "apple-speech:speechTranscriber"
    assert spec.capabilities.timestamp is True
    assert spec.capabilities.diarization is False
    assert "ASR-only" in spec.description


def test_apple_dictation_alias_should_stay_hidden_until_runtime_support_exists() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        lookup("apple-dictation")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run python -m pytest tests/unit/test_model_registry.py::test_apple_speech_alias_should_use_apple_speech_runtime_contract tests/unit/test_model_registry.py::test_apple_dictation_alias_should_stay_hidden_until_runtime_support_exists -q
```

Expected: FAIL with `Unknown model: 'apple-speech'` for the new Apple Speech alias.

- [ ] **Step 3: Add Apple engine type and registry entries**

In `src/core/model_registry.py`, change the engine type literal:

```python
EngineType = Literal["funasr", "mlx", "apple-speech"]
```

Add one `ModelSpec` entry to `_REGISTRY`; keep `apple-dictation` hidden until the Swift runtime supports `DictationTranscriber` transcription:

```python
ModelSpec(
    alias="apple-speech",
    model_id="apple-speech:speechTranscriber",
    engine_type="apple-speech",
    description=(
        "Apple SpeechAnalyzer SpeechTranscriber sidecar. Local macOS 26+ ASR-only "
        "runtime for dictation and short/medium transcription; speaker labels require "
        "a separate diarization stage."
    ),
    capabilities=EngineCapabilities(timestamp=True, diarization=False, emotion_tags=False, language_detect=False),
),
```

Keep `src/config.py` startup engines limited to directly loadable Python worker runtimes:

```python
EngineType = Literal["funasr", "mlx"]
```

- [ ] **Step 4: Document Apple models**

In `MODELS.md`, add rows under Active Models:

```markdown
| `apple-speech` | Apple SpeechAnalyzer `SpeechTranscriber` via Swift sidecar | `apple-speech:speechTranscriber` | ❌ | macOS 26+ local ASR-only path; no speaker labels without a separate diarization stage |
```

Add to Model Selection Guide:

```markdown
| Apple-native local dictation on macOS 26+ | `apple-speech` | Uses the system SpeechAnalyzer model through the Swift sidecar; ASR-only until diarization gates pass |
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
uv run python -m pytest tests/unit/test_model_registry.py tests/integration/test_model_api.py::test_should_return_model_list_on_get_models -q
```

Expected: all selected tests pass and `/v1/models` includes Apple aliases.

- [ ] **Step 6: Commit**

```bash
git add src/core/model_registry.py src/config.py tests/unit/test_model_registry.py MODELS.md
git commit -m "feat: register apple speech model aliases"
```

---

### Task 5: Add Service-Layer Apple Sidecar Path

**Files:**
- Modify: `src/config.py`
- Modify: `src/services/transcription.py`
- Create: `tests/unit/test_service_apple_speech.py`

- [ ] **Step 1: Write failing service tests**

Create `tests/unit/test_service_apple_speech.py`:

```python
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest
from starlette.datastructures import UploadFile

from src.core.model_registry import lookup
from src.services.transcription import TranscriptionService


class FakeAppleSpeechEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, bool]] = []

    def transcribe_file(
        self,
        file_path: str,
        language: str = "auto",
        output_format: str = "json",
        with_timestamp: bool = False,
        **kwargs: object,
    ) -> dict[str, object]:
        self.calls.append((file_path, language, output_format, with_timestamp))
        assert Path(file_path).exists()
        return {
            "text": "apple result",
            "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "apple result", "speaker": None}],
            "duration": 1.0,
            "language": "en-US",
        }


@pytest.mark.asyncio
async def test_submit_apple_speech_bypasses_multiprocessing_worker() -> None:
    service = TranscriptionService(engine_type="funasr", model_id="iic/default")
    fake_engine = FakeAppleSpeechEngine()
    upload = UploadFile(filename="audio.wav", file=BytesIO(b"RIFF\x00\x00\x00\x00WAVEfmt "))

    with (
        patch.object(service, "_get_apple_speech_engine", return_value=fake_engine),
        patch.object(service, "_submit_worker_job") as worker_submit,
    ):
        result = await service.submit(
            upload,
            {"language": "en", "output_format": "json", "with_timestamp": False},
            request_id="req-1",
            model_spec=lookup("apple-speech"),
        )

    assert result["text"] == "apple result"
    assert result["language"] == "en-US"
    assert fake_engine.calls[0][1:] == ("en", "json", False)
    worker_submit.assert_not_called()
    assert service.queue_size == 0


@pytest.mark.asyncio
async def test_submit_apple_speech_cleans_temp_dir_on_error() -> None:
    service = TranscriptionService(engine_type="funasr", model_id="iic/default")
    upload = UploadFile(filename="audio.wav", file=BytesIO(b"RIFF\x00\x00\x00\x00WAVEfmt "))

    class FailingEngine(FakeAppleSpeechEngine):
        def transcribe_file(
            self,
            file_path: str,
            language: str = "auto",
            output_format: str = "json",
            with_timestamp: bool = False,
            **kwargs: object,
        ) -> dict[str, object]:
            assert Path(file_path).exists()
            raise RuntimeError("apple worker failed")

    with patch.object(service, "_get_apple_speech_engine", return_value=FailingEngine()):
        with pytest.raises(RuntimeError, match="apple worker failed"):
            await service.submit(
                upload,
                {"language": "en", "output_format": "json", "with_timestamp": False},
                request_id="req-2",
                model_spec=lookup("apple-speech"),
            )

    assert service.queue_size == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run python -m pytest tests/unit/test_service_apple_speech.py -q
```

Expected: FAIL because `model_spec.engine_type == "apple-speech"` still flows into `_submit_worker_job`.

- [ ] **Step 3: Add config values**

In `src/config.py`, add:

```python
APPLE_SPEECH_WORKER_PATH = os.getenv(
    "APPLE_SPEECH_WORKER_PATH",
    str(Path(__file__).parent.parent / "apple-speech-worker" / ".build" / "debug" / "apple-speech-worker"),
)
APPLE_SPEECH_WORKER_TIMEOUT_SEC = float(os.getenv("APPLE_SPEECH_WORKER_TIMEOUT_SEC", "120"))
APPLE_SPEECH_DEFAULT_LOCALE = os.getenv("APPLE_SPEECH_DEFAULT_LOCALE", "en-US")
```

- [ ] **Step 4: Add service sidecar state and helpers**

In `src/services/transcription.py`, import:

```python
from pathlib import Path

from src.config import (
    APPLE_SPEECH_DEFAULT_LOCALE,
    APPLE_SPEECH_WORKER_PATH,
    APPLE_SPEECH_WORKER_TIMEOUT_SEC,
)
from src.core.apple_speech_engine import AppleSpeechEngine, AppleSpeechEngineConfig
from src.core.apple_speech_port import AppleSpeechModule
```

Add fields in `__init__`:

```python
self._sidecar_pending: set[str] = set()
self._apple_speech_engines: dict[str, AppleSpeechEngine] = {}
```

Change `queue_size`:

```python
@property
def queue_size(self) -> int:
    """Number of jobs currently in-flight across model worker and sidecar paths."""
    return len(self._pending) + len(self._sidecar_pending)
```

Add helper methods:

```python
def _active_job_count(self) -> int:
    return len(self._pending) + len(self._sidecar_pending)


def _is_apple_speech_spec(self, model_spec: ModelSpec | None) -> bool:
    return model_spec is not None and model_spec.engine_type == "apple-speech"


def _apple_speech_module_for_spec(self, model_spec: ModelSpec) -> AppleSpeechModule:
    if model_spec.model_id == "apple-speech:dictationTranscriber":
        return "dictationTranscriber"
    return "speechTranscriber"


def _get_apple_speech_engine(self, model_spec: ModelSpec) -> AppleSpeechEngine:
    module = self._apple_speech_module_for_spec(model_spec)
    engine = self._apple_speech_engines.get(module)
    if engine is None:
        engine = AppleSpeechEngine.from_config(
            AppleSpeechEngineConfig(
                worker_path=Path(APPLE_SPEECH_WORKER_PATH),
                timeout_seconds=APPLE_SPEECH_WORKER_TIMEOUT_SEC,
                default_locale=APPLE_SPEECH_DEFAULT_LOCALE,
            ),
            module=module,
        )
        self._apple_speech_engines[module] = engine
    return engine
```

- [ ] **Step 5: Route Apple submit before multiprocessing worker**

In `submit`, change the capacity check:

```python
if self._active_job_count() >= self._max_queue_size:
    self.logger.warning(f"[{request_id}] Queue full, rejecting request")
    raise RuntimeError("Service busy: Queue is full.")
```

After writing `temp_path`, branch before `_submit_worker_job`:

```python
if self._is_apple_speech_spec(model_spec):
    if model_spec is None:
        raise RuntimeError("Apple Speech request requires a resolved model spec")
    result = await self._submit_apple_speech_job(
        temp_file_path=temp_path,
        params=params,
        request_id=request_id,
        model_spec=model_spec,
    )
else:
    result = await self._submit_worker_job(
        temp_file_path=temp_path,
        params=params,
        request_id=request_id,
        model_spec=model_spec,
        temp_dir=temp_dir,
    )
return self._coerce_transcription_result(result)
```

Add:

```python
async def _submit_apple_speech_job(
    self,
    temp_file_path: str,
    params: dict[str, object],
    request_id: str,
    model_spec: ModelSpec,
) -> object:
    if self._active_job_count() >= self._max_queue_size:
        self.logger.warning(f"[{request_id}] Queue full, rejecting Apple Speech request")
        raise RuntimeError("Service busy: Queue is full.")

    self._sidecar_pending.add(request_id)
    try:
        engine = self._get_apple_speech_engine(model_spec)
        language = params.get("language", "auto")
        output_format = params.get("output_format", "json")
        with_timestamp = params.get("with_timestamp", False)
        return await asyncio.to_thread(
            engine.transcribe_file,
            temp_file_path,
            language=language if isinstance(language, str) else "auto",
            output_format=output_format if isinstance(output_format, str) else "json",
            with_timestamp=with_timestamp if isinstance(with_timestamp, bool) else False,
        )
    finally:
        self._sidecar_pending.discard(request_id)
```

Update other capacity checks in `_enqueue_worker_job` and `submit_pipeline` to use `_active_job_count()` so sidecar requests count against `MAX_QUEUE_SIZE`.

- [ ] **Step 6: Run service tests**

Run:

```bash
uv run python -m pytest tests/unit/test_service_apple_speech.py tests/unit/test_service.py -q
```

Expected: selected service tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/config.py src/services/transcription.py tests/unit/test_service_apple_speech.py
git commit -m "feat: route apple speech through sidecar path"
```

---

### Task 6: Add API Integration Coverage

**Files:**
- Modify: `src/api/routes.py`
- Modify: `tests/integration/test_model_api.py`

- [ ] **Step 1: Write failing API tests**

Append to `tests/integration/test_model_api.py`:

```python
def test_models_endpoint_should_include_apple_speech_aliases(client) -> None:
    response = client.get("/v1/models")

    assert response.status_code == 200
    models = {item["alias"]: item for item in response.json()["models"]}
    assert models["apple-speech"]["engine_type"] == "apple-speech"
    assert models["apple-speech"]["capabilities"]["timestamp"] is True
    assert models["apple-speech"]["capabilities"]["diarization"] is False
    assert "apple-dictation" not in models


def test_should_submit_apple_speech_model_spec_to_service() -> None:
    qwen_spec = real_lookup("qwen3-asr")
    mock_service = _make_mock_service(
        qwen_spec.capabilities,
        {
            "text": "apple result",
            "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "apple result", "speaker": None}],
            "duration": 1.0,
            "language": "en-US",
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
            data={"model": "apple-speech", "language": "en", "output_format": "json"},
            files={"file": _audio_file()},
        )

    assert response.status_code == 200
    assert response.json()["model"] == "apple-speech"
    assert response.json()["segments"][0]["speaker"] is None
    submitted_spec = mock_service.submit.await_args.kwargs["model_spec"]
    assert submitted_spec.alias == "apple-speech"
    mock_service.submit_pipeline.assert_not_awaited()
```

- [ ] **Step 2: Run test to verify it fails if Task 4 is not complete**

Run:

```bash
uv run python -m pytest tests/integration/test_model_api.py::test_models_endpoint_should_include_apple_speech_aliases tests/integration/test_model_api.py::test_should_submit_apple_speech_model_spec_to_service -q
```

Expected before Task 4: FAIL because Apple aliases are absent. Expected after Task 4: PASS.

- [ ] **Step 3: Update API docs text**

In `src/api/routes.py`, update the model field description examples:

```python
"Examples: 'paraformer', 'qwen3-asr', 'qwen3-sortformer', 'apple-speech'."
```

Update the endpoint docstring model switching list:

```python
- Pass `model=apple-speech` for macOS 26+ Apple SpeechAnalyzer ASR-only sidecar transcription.
- `apple-dictation` remains hidden until the Swift runtime supports DictationTranscriber.
```

- [ ] **Step 4: Update OpenAPI docs test**

In `test_openapi_transcription_docs_should_use_current_model_aliases`, add:

```python
assert "model=apple-speech" in description
assert "'apple-speech'" in model_description
```

- [ ] **Step 5: Run integration tests**

Run:

```bash
uv run python -m pytest tests/integration/test_model_api.py tests/integration/test_api.py -q
```

Expected: integration tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/api/routes.py tests/integration/test_model_api.py
git commit -m "test: cover apple speech API routing"
```

---

### Task 7: Architecture Fitness and Documentation

**Files:**
- Modify: `tests/unit/test_architecture_fitness.py`
- Modify: `README.md`
- Modify: `docs/SPEC-014-Apple-SpeechAnalyzer-Integration.md`

- [ ] **Step 1: Run architecture fitness to expose allowlist drift**

Run:

```bash
uv run python -m pytest tests/unit/test_architecture_fitness.py -q
```

Expected: either PASS or FAIL listing the new Apple Speech files that need to be added to the approved module gate.

- [ ] **Step 2: Update fitness allowlist only if the test fails**

If the failure names `apple_speech_engine.py` or `apple_speech_port.py`, add those filenames next to the existing Apple worker client allowance in `tests/unit/test_architecture_fitness.py`.

Use this exact allowed-entry set for the Apple Speech additions:

```python
"apple_speech_engine.py",
"apple_speech_port.py",
```

- [ ] **Step 3: Update README environment table**

Add these rows to `README.md`:

```markdown
| `APPLE_SPEECH_WORKER_PATH` | `apple-speech-worker/.build/debug/apple-speech-worker` | Swift sidecar binary path for Apple Speech models |
| `APPLE_SPEECH_WORKER_TIMEOUT_SEC` | `120` | Timeout for one Apple Speech worker CLI invocation |
| `APPLE_SPEECH_DEFAULT_LOCALE` | `en-US` | Locale used when API callers pass `language=auto` to Apple Speech models |
| `APPLE_SPEECH_MAX_CONCURRENCY` | `1` | Maximum concurrent Apple Speech sidecar transcriptions |
```

- [ ] **Step 4: Update SPEC-014 Phase 2 evidence**

Append to Phase 2 after the checklist:

```markdown
Phase 2 implementation note:

- The Python service routes Apple Speech requests through a direct sidecar path rather than through `src/workers/model_worker.py`; the Swift CLI is already the process boundary for Apple Speech framework access.
- Apple Speech sidecar transcription is capped by `APPLE_SPEECH_MAX_CONCURRENCY` (default `1`) to preserve the Mac Silicon single-inference memory discipline while still counting waiting sidecar jobs against `MAX_QUEUE_SIZE`.
- `apple-speech` preserves the existing local JSON response shape and does not emit non-null speaker labels.
- `GET /v1/models` advertises the `apple-speech` alias as requestable; `apple-dictation` stays hidden until the Swift runtime supports `DictationTranscriber` transcription.
```

- [ ] **Step 5: Run docs and architecture focused tests**

Run:

```bash
uv run python -m pytest tests/unit/test_architecture_fitness.py tests/integration/test_model_api.py::test_openapi_transcription_docs_should_use_current_model_aliases -q
```

Expected: selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add tests/unit/test_architecture_fitness.py README.md docs/SPEC-014-Apple-SpeechAnalyzer-Integration.md
git commit -m "docs: document apple speech service integration"
```

---

### Task 8: Verification Bundle

**Files:**
- No source edits expected.

- [ ] **Step 1: Run Python full test suite**

Run:

```bash
uv run python -m pytest
```

Expected: all tests pass. The existing `PytestUnknownMarkWarning` for `e2e` is acceptable only if it remains the sole warning.

- [ ] **Step 2: Run Swift contract executable outside sandbox**

Run outside Codex filesystem sandbox:

```bash
swift run --package-path apple-speech-worker apple-speech-worker-contract-tests
```

Expected:

```text
contract-tests: passed
```

- [ ] **Step 3: Build worker binary outside sandbox**

Run:

```bash
swift build --package-path apple-speech-worker
```

Expected: build succeeds and produces `apple-speech-worker/.build/debug/apple-speech-worker`.

- [ ] **Step 4: Run local mocked API smoke**

Run:

```bash
uv run python -m pytest tests/integration/test_model_api.py::test_should_submit_apple_speech_model_spec_to_service -q
```

Expected: test passes.

- [ ] **Step 5: Run real sidecar smoke only if macOS 26 Speech permissions are available**

Run from a non-sandboxed terminal:

```bash
APPLE_SPEECH_WORKER_PATH="$PWD/apple-speech-worker/.build/debug/apple-speech-worker" \
uv run python -m src.main
```

In another terminal:

```bash
curl -X POST http://localhost:50700/v1/audio/transcriptions \
  -F "file=@/absolute/path/to/project-owned-sample.wav" \
  -F "model=apple-speech" \
  -F "language=en-US" \
  -F "response_format=verbose_json"
```

Expected JSON shape:

```json
{
  "text": "recognized text",
  "duration": 1.0,
  "language": "en-US",
  "model": "apple-speech",
  "segments": [
    {
      "id": 0,
      "speaker": null,
      "start": 0.0,
      "end": 1.0,
      "text": "recognized text"
    }
  ]
}
```

- [ ] **Step 6: Inspect final diff**

Run:

```bash
git diff main...HEAD --stat
git diff main...HEAD -- docs/SPEC-014-Apple-SpeechAnalyzer-Integration.md src/core src/adapters src/services src/api tests README.md MODELS.md
```

Expected: only SPEC-014 Apple Speech service integration files changed.

---

## Self-Review

Spec coverage:

- SPEC-014 Phase 1 CLI and Python worker-client boundary is covered by Task 1 and existing worker client/Swift contract tests.
- Phase 2 registry, adapter, API, response shape, and existing route preservation are covered by Tasks 2 through 7.
- AC-7 no fake speaker labels is covered by `test_transcribe_file_never_synthesizes_speaker_labels` and API assertion that `speaker` remains `None`.
- AC-9 clear failures is partially covered through worker timeout/missing binary/invalid JSON tests and API error propagation. Real unsupported OS and unsupported locale behavior still requires non-sandboxed Apple Speech runtime smoke.

Intentional gaps:

- Project-owned Chinese/English mixed fixture is still a gate for broader bilingual quality claims. This plan enables the service path but does not make quality claims.
- Phase 3 benchmarking, Phase 4 dictation vocabulary, Phase 5 diarization integration, and Phase 6 production decision remain separate specs or follow-up plans.

Type consistency:

- Registry uses `engine_type="apple-speech"` only for routing and API discovery.
- `AppleSpeechEngine` is not created through `src/core/factory.py` or `src/workers/model_worker.py`; it is created directly by `TranscriptionService` for explicit Apple model specs.
- `AppleSpeechModule` values match the Swift CLI values: `"speechTranscriber"` and `"dictationTranscriber"`.

Placeholder scan:

- No implementation step relies on unspecified validation or generic error handling.
- Every new behavior has a named test and an exact verification command.
