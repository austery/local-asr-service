# SPEC-013 Long-Form Productionization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Productionize the Qwen3-ASR + Qwen3-ForcedAligner + Sortformer pipeline for truthful offline batch speaker separation up to 5-hour audio inputs.

**Architecture:** Add a shared chunk plan with extraction windows and non-overlapping emit windows. Run Qwen3-ASR, Qwen3-ForcedAligner, and Sortformer per chunk, convert chunk-local timestamps to one global timeline, drop duplicate overlap output by midpoint, and fail loud when alignment quality collapses. Keep `qwen3-sortformer` discovery-only until a real-runtime 5-hour probe passes.

**Tech Stack:** Python 3.11, FastAPI service layer, multiprocessing worker IPC, ffmpeg/ffprobe through existing `AudioChunkingService`, mlx-audio Qwen3-ASR / Qwen3-ForcedAligner / Sortformer, pytest, ruff.

---

## Core Design Rules

- A chunk has two ranges:
  - `start` / `end`: audio extracted and sent to model, includes overlap context.
  - `emit_start` / `emit_end`: absolute timeline range allowed to emit final words/turns.
- Words and speaker turns are converted to absolute timestamps with `chunk.start + local_time`.
- Any word or turn whose midpoint is outside the chunk emit window is dropped to avoid overlap duplication.
- Forced alignment must never run on a full long-form file. The current upstream Qwen3-ForcedAligner has no long-audio chunking and collapses timestamps on 10-minute input.
- `qwen3-sortformer` remains `requestable=False` until Task 7 passes real-runtime validation.

## File Structure

- Create `src/adapters/pipeline_chunking.py`: pure dataclasses and functions for chunk plans, offset stitching, emit-window filtering, monotonic validation, and tail-collapse detection.
- Modify `src/adapters/audio_chunking.py`: expose duration probing and chunk extraction methods usable by pipeline orchestration without changing existing ASR behavior.
- Modify `src/services/transcription.py`: choose short-form existing path vs long-form chunked path; orchestrate chunked ASR, forced alignment, diarization, global stitching, speaker assignment, and guardrails.
- Modify `docs/SPEC-013-Production-Decoupled-Diarization-Pipeline.md`: update checklist as tasks land.
- Add `tests/unit/test_pipeline_chunking.py`: pure tests for chunk plan and stitching.
- Modify `tests/unit/test_service.py`: orchestration tests using mocked worker calls.
- Modify `tests/unit/test_audio_chunking.py`: tests for reusable duration/extraction wrapper.
- Modify `tests/integration/test_model_api.py`: keep public API gate until real-runtime validation passes.
- Create `scripts/probe_qwen3_sortformer_longform.py`: manual real-runtime validation script; output artifacts go to `/tmp` and are not committed.

---

### Task 1: Introduce Shared Pipeline Chunk Planning

**Files:**
- Create: `src/adapters/pipeline_chunking.py`
- Test: `tests/unit/test_pipeline_chunking.py`

- [x] **Step 1: Write the failing chunk-plan tests**

Create `tests/unit/test_pipeline_chunking.py`:

```python
import pytest

from src.adapters.pipeline_chunking import ChunkWindow, build_chunk_plan


def test_build_chunk_plan_should_cover_full_duration_without_emit_gaps() -> None:
    plan = build_chunk_plan(
        duration_seconds=18_000.0,
        chunk_seconds=900.0,
        overlap_seconds=15.0,
    )

    assert plan[0] == ChunkWindow(index=0, start=0.0, end=900.0, emit_start=0.0, emit_end=885.0)
    assert plan[-1].end == 18_000.0
    assert plan[-1].emit_end == 18_000.0
    assert [window.index for window in plan] == list(range(len(plan)))
    assert all(plan[i].emit_end == plan[i + 1].emit_start for i in range(len(plan) - 1))
    assert all(window.start <= window.emit_start < window.emit_end <= window.end for window in plan)


def test_build_chunk_plan_should_return_one_chunk_for_short_audio() -> None:
    plan = build_chunk_plan(
        duration_seconds=60.0,
        chunk_seconds=900.0,
        overlap_seconds=15.0,
    )

    assert plan == [ChunkWindow(index=0, start=0.0, end=60.0, emit_start=0.0, emit_end=60.0)]


def test_build_chunk_plan_should_reject_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="overlap_seconds must be less than chunk_seconds"):
        build_chunk_plan(duration_seconds=120.0, chunk_seconds=60.0, overlap_seconds=60.0)
```

- [x] **Step 2: Run the test to verify RED**

Run:

```bash
uv run python -m pytest tests/unit/test_pipeline_chunking.py -q
```

Expected: fails because `src.adapters.pipeline_chunking` does not exist.

- [x] **Step 3: Implement minimal chunk-plan module**

Create `src/adapters/pipeline_chunking.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkWindow:
    index: int
    start: float
    end: float
    emit_start: float
    emit_end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def build_chunk_plan(
    *,
    duration_seconds: float,
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[ChunkWindow]:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds must be non-negative")
    if overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be less than chunk_seconds")

    if duration_seconds <= chunk_seconds:
        return [ChunkWindow(index=0, start=0.0, end=duration_seconds, emit_start=0.0, emit_end=duration_seconds)]

    windows: list[ChunkWindow] = []
    emit_start = 0.0
    index = 0
    while emit_start < duration_seconds:
        start = max(0.0, emit_start - overlap_seconds)
        emit_end = min(duration_seconds, emit_start + chunk_seconds - overlap_seconds)
        end = min(duration_seconds, emit_end + overlap_seconds)
        windows.append(
            ChunkWindow(
                index=index,
                start=round(start, 3),
                end=round(end, 3),
                emit_start=round(emit_start, 3),
                emit_end=round(emit_end, 3),
            )
        )
        emit_start = emit_end
        index += 1

    return windows
```

- [x] **Step 4: Verify GREEN**

Run:

```bash
uv run python -m pytest tests/unit/test_pipeline_chunking.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/adapters/pipeline_chunking.py tests/unit/test_pipeline_chunking.py
git commit -m "feat: add shared pipeline chunk planning"
```

---

### Task 2: Add Global Timestamp Stitching Helpers

**Files:**
- Modify: `src/adapters/pipeline_chunking.py`
- Test: `tests/unit/test_pipeline_chunking.py`

- [x] **Step 1: Write failing tests for word and turn stitching**

Append to `tests/unit/test_pipeline_chunking.py`:

```python
from src.adapters.pipeline_chunking import (
    offset_words_to_global_timeline,
    offset_turns_to_global_timeline,
)
from src.core.alignment_port import AlignedWord
from src.core.diarization_port import SpeakerTurn


def test_offset_words_should_apply_chunk_start_and_drop_overlap_duplicates() -> None:
    window = ChunkWindow(index=1, start=885.0, end=1800.0, emit_start=900.0, emit_end=1785.0)
    words = [
        AlignedWord(text="context", start=1.0, end=2.0),
        AlignedWord(text="kept", start=20.0, end=21.0),
        AlignedWord(text="tail", start=910.0, end=911.0),
    ]

    result = offset_words_to_global_timeline(words, window)

    assert result == [AlignedWord(text="kept", start=905.0, end=906.0)]


def test_offset_turns_should_apply_chunk_start_and_clip_to_emit_window() -> None:
    window = ChunkWindow(index=1, start=885.0, end=1800.0, emit_start=900.0, emit_end=1785.0)
    turns = [
        SpeakerTurn(speaker="Speaker 0", start=0.0, end=20.0),
        SpeakerTurn(speaker="Speaker 1", start=20.0, end=40.0),
        SpeakerTurn(speaker="Speaker 2", start=900.0, end=915.0),
    ]

    result = offset_turns_to_global_timeline(turns, window)

    assert result == [
        SpeakerTurn(speaker="Speaker 0", start=900.0, end=905.0),
        SpeakerTurn(speaker="Speaker 1", start=905.0, end=925.0),
    ]
```

- [x] **Step 2: Run the test to verify RED**

Run:

```bash
uv run python -m pytest tests/unit/test_pipeline_chunking.py -q
```

Expected: fails because stitching helpers do not exist.

- [x] **Step 3: Implement stitching helpers**

Add to `src/adapters/pipeline_chunking.py`:

```python
from src.core.alignment_port import AlignedWord
from src.core.diarization_port import SpeakerTurn


def _midpoint(start: float, end: float) -> float:
    return (start + end) / 2


def _is_midpoint_in_emit_window(start: float, end: float, window: ChunkWindow) -> bool:
    midpoint = _midpoint(start, end)
    return window.emit_start <= midpoint < window.emit_end


def offset_words_to_global_timeline(
    words: list[AlignedWord],
    window: ChunkWindow,
) -> list[AlignedWord]:
    result: list[AlignedWord] = []
    for word in words:
        global_start = round(window.start + word.start, 3)
        global_end = round(window.start + word.end, 3)
        if _is_midpoint_in_emit_window(global_start, global_end, window):
            result.append(AlignedWord(text=word.text, start=global_start, end=global_end))
    return result


def offset_turns_to_global_timeline(
    turns: list[SpeakerTurn],
    window: ChunkWindow,
) -> list[SpeakerTurn]:
    result: list[SpeakerTurn] = []
    for turn in turns:
        global_start = max(window.emit_start, round(window.start + turn.start, 3))
        global_end = min(window.emit_end, round(window.start + turn.end, 3))
        if global_end <= global_start:
            continue
        if _is_midpoint_in_emit_window(global_start, global_end, window):
            result.append(SpeakerTurn(speaker=turn.speaker, start=global_start, end=global_end))
    return result
```

- [x] **Step 4: Verify GREEN**

Run:

```bash
uv run python -m pytest tests/unit/test_pipeline_chunking.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/adapters/pipeline_chunking.py tests/unit/test_pipeline_chunking.py
git commit -m "feat: stitch chunk outputs onto global timeline"
```

---

### Task 3: Add Alignment Quality Gates

**Files:**
- Modify: `src/adapters/pipeline_chunking.py`
- Test: `tests/unit/test_pipeline_chunking.py`

- [x] **Step 1: Write failing tests for monotonic and tail-collapse guardrails**

Append to `tests/unit/test_pipeline_chunking.py`:

```python
from src.adapters.pipeline_chunking import validate_aligned_word_quality


def test_validate_aligned_word_quality_should_reject_non_monotonic_words() -> None:
    words = [
        AlignedWord(text="first", start=10.0, end=11.0),
        AlignedWord(text="second", start=9.0, end=9.5),
    ]

    with pytest.raises(ValueError, match="non-monotonic"):
        validate_aligned_word_quality(words, expected_duration_seconds=20.0)


def test_validate_aligned_word_quality_should_reject_tail_timestamp_collapse() -> None:
    words = [
        AlignedWord(text=f"w{i}", start=float(i), end=float(i) + 0.5)
        for i in range(20)
    ]
    words.extend(
        AlignedWord(text=f"tail{i}", start=245.04, end=245.04)
        for i in range(12)
    )

    with pytest.raises(ValueError, match="tail timestamp collapse"):
        validate_aligned_word_quality(words, expected_duration_seconds=600.0)


def test_validate_aligned_word_quality_should_accept_short_valid_alignment() -> None:
    words = [
        AlignedWord(text="hello", start=0.1, end=0.5),
        AlignedWord(text="world", start=0.6, end=1.0),
    ]

    validate_aligned_word_quality(words, expected_duration_seconds=1.2)
```

- [x] **Step 2: Run the test to verify RED**

Run:

```bash
uv run python -m pytest tests/unit/test_pipeline_chunking.py -q
```

Expected: fails because `validate_aligned_word_quality` does not exist.

- [x] **Step 3: Implement quality validation**

Add to `src/adapters/pipeline_chunking.py`:

```python
def validate_aligned_word_quality(
    words: list[AlignedWord],
    *,
    expected_duration_seconds: float,
    tail_word_count: int = 10,
) -> None:
    if not words:
        raise ValueError("alignment quality gate failed: no aligned words")

    previous_start = -1.0
    for word in words:
        if word.start < previous_start:
            raise ValueError("alignment quality gate failed: non-monotonic aligned words")
        previous_start = word.start

    if len(words) >= tail_word_count:
        tail = words[-tail_word_count:]
        tail_positions = {(word.start, word.end) for word in tail}
        if len(tail_positions) <= 2 and expected_duration_seconds - tail[-1].end > 60.0:
            raise ValueError("alignment quality gate failed: tail timestamp collapse")
```

- [x] **Step 4: Verify GREEN**

Run:

```bash
uv run python -m pytest tests/unit/test_pipeline_chunking.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/adapters/pipeline_chunking.py tests/unit/test_pipeline_chunking.py
git commit -m "feat: add long-form alignment quality gates"
```

---

### Task 4: Expose Reusable Audio Duration and Chunk Extraction

**Files:**
- Modify: `src/adapters/audio_chunking.py`
- Test: `tests/unit/test_audio_chunking.py`

- [x] **Step 1: Write failing tests for pipeline chunk extraction**

Append to `tests/unit/test_audio_chunking.py`:

```python
from src.adapters.pipeline_chunking import ChunkWindow


def test_extract_pipeline_chunk_should_use_ffmpeg_with_window(service, mock_ffmpeg, tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"fake wav")
    output = tmp_path / "chunk_000.wav"
    window = ChunkWindow(index=0, start=10.0, end=20.0, emit_start=10.0, emit_end=20.0)

    service.extract_pipeline_chunk(str(source), str(output), window)

    command = mock_ffmpeg.call_args_list[-1][0][0]
    assert command[:4] == ["ffmpeg", "-i", str(source), "-ss"]
    assert command[4] == "10.0"
    assert "-to" in command
    assert "20.0" in command
    assert str(output) in command
```

- [x] **Step 2: Run the test to verify RED**

Run:

```bash
uv run python -m pytest tests/unit/test_audio_chunking.py -q -k extract_pipeline_chunk
```

Expected: fails because `extract_pipeline_chunk` does not exist.

- [x] **Step 3: Implement extraction wrapper**

Add to `AudioChunkingService` in `src/adapters/audio_chunking.py`:

```python
    def get_audio_duration(self, audio_path: str) -> float:
        return self._get_audio_duration(audio_path)

    def extract_pipeline_chunk(
        self,
        audio_path: str,
        output_path: str,
        window: "ChunkWindow",
    ) -> str:
        from src.adapters.pipeline_chunking import ChunkWindow

        if not isinstance(window, ChunkWindow):
            raise TypeError("window must be a ChunkWindow")

        cmd = [
            "ffmpeg",
            "-i",
            audio_path,
            "-ss",
            str(window.start),
            "-to",
            str(window.end),
            "-ac",
            "1",
            "-ar",
            str(self.sample_rate),
            "-c:a",
            "pcm_s16le",
            "-y",
            output_path,
        ]
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to extract pipeline chunk {window.index}: {e.stderr}") from e
        return output_path
```

- [x] **Step 4: Verify GREEN**

Run:

```bash
uv run python -m pytest tests/unit/test_audio_chunking.py -q -k extract_pipeline_chunk
```

Expected: test passes.

- [ ] **Step 5: Commit**

```bash
git add src/adapters/audio_chunking.py tests/unit/test_audio_chunking.py
git commit -m "feat: expose audio chunk extraction for pipeline orchestration"
```

---

### Task 5: Add Chunked Forced-Alignment Orchestration

**Files:**
- Modify: `src/services/transcription.py`
- Test: `tests/unit/test_service.py`

- [x] **Step 1: Write failing tests for chunked alignment offsets**

Append to `tests/unit/test_service.py`:

```python
from src.adapters.pipeline_chunking import ChunkWindow


@pytest.mark.asyncio
async def test_chunked_alignment_should_apply_offsets_and_drop_overlap(funasr_spec):
    svc = _setup_service(funasr_spec)
    windows = [
        ChunkWindow(index=0, start=0.0, end=900.0, emit_start=0.0, emit_end=885.0),
        ChunkWindow(index=1, start=870.0, end=1800.0, emit_start=885.0, emit_end=1800.0),
    ]

    async def fake_align(temp_file_path, text, language, request_id, alias, pipeline_reserved=False):
        if temp_file_path.endswith("chunk_000.wav"):
            return [AlignedWord(text="first", start=10.0, end=11.0)]
        return [
            AlignedWord(text="overlap", start=1.0, end=2.0),
            AlignedWord(text="second", start=20.0, end=21.0),
        ]

    svc._align_with_alias = fake_align

    result = await svc._align_chunks_with_alias(
        chunk_paths=["/tmp/chunk_000.wav", "/tmp/chunk_001.wav"],
        chunk_texts=["first", "overlap second"],
        windows=windows,
        language="English",
        request_id="req",
        alias="qwen3-forced-aligner",
        pipeline_reserved=True,
    )

    assert result == [
        AlignedWord(text="first", start=10.0, end=11.0),
        AlignedWord(text="second", start=890.0, end=891.0),
    ]
```

- [x] **Step 2: Run the test to verify RED**

Run:

```bash
uv run python -m pytest tests/unit/test_service.py -q -k chunked_alignment
```

Expected: fails because `_align_chunks_with_alias` does not exist.

- [x] **Step 3: Implement chunked alignment helper**

Add imports in `src/services/transcription.py`:

```python
from src.adapters.pipeline_chunking import ChunkWindow, offset_words_to_global_timeline
```

Add method to `TranscriptionService`:

```python
    async def _align_chunks_with_alias(
        self,
        *,
        chunk_paths: list[str],
        chunk_texts: list[str],
        windows: list[ChunkWindow],
        language: str,
        request_id: str,
        alias: str,
        pipeline_reserved: bool,
    ) -> list[AlignedWord]:
        if not (len(chunk_paths) == len(chunk_texts) == len(windows)):
            raise ValueError("chunk_paths, chunk_texts, and windows must have the same length")

        merged: list[AlignedWord] = []
        for chunk_path, chunk_text, window in zip(chunk_paths, chunk_texts, windows):
            if not chunk_text.strip():
                continue
            words = await self._align_with_alias(
                chunk_path,
                chunk_text,
                language,
                f"{request_id}:chunk-{window.index}",
                alias,
                pipeline_reserved=pipeline_reserved,
            )
            merged.extend(offset_words_to_global_timeline(words, window))
        return merged
```

- [x] **Step 4: Verify GREEN**

Run:

```bash
uv run python -m pytest tests/unit/test_service.py -q -k chunked_alignment
```

Expected: test passes.

- [ ] **Step 5: Commit**

```bash
git add src/services/transcription.py tests/unit/test_service.py
git commit -m "feat: add chunked forced-alignment orchestration"
```

---

### Task 6: Add Real Audio Chunk Extraction in Service

**Files:**
- Modify: `src/services/transcription.py`
- Test: `tests/unit/test_service.py`
- Test: `tests/unit/test_audio_chunking.py`

- [x] **Step 1: Write failing test for extracted chunk paths**

Append to `tests/unit/test_service.py`:

```python
@pytest.mark.asyncio
async def test_long_form_pipeline_should_extract_chunk_files_before_alignment(funasr_spec, tmp_path):
    svc = _setup_service(funasr_spec)
    windows = [
        ChunkWindow(index=0, start=0.0, end=300.0, emit_start=0.0, emit_end=285.0),
        ChunkWindow(index=1, start=270.0, end=600.0, emit_start=285.0, emit_end=600.0),
    ]
    extracted: list[str] = []

    def fake_extract(source_path, output_path, window):
        extracted.append(output_path)
        return output_path

    svc._audio_chunker.extract_pipeline_chunk = fake_extract

    paths = svc._extract_pipeline_chunks("audio.wav", str(tmp_path), windows)

    assert paths == [str(tmp_path / "pipeline_chunk_000.wav"), str(tmp_path / "pipeline_chunk_001.wav")]
    assert extracted == paths
```

- [x] **Step 2: Run the test to verify RED**

Run:

```bash
uv run python -m pytest tests/unit/test_service.py -q -k extract_chunk_files
```

Expected: fails because `_extract_pipeline_chunks` does not exist.

- [x] **Step 3: Implement extraction helper**

Add method to `TranscriptionService`:

```python
    def _extract_pipeline_chunks(
        self,
        temp_file_path: str,
        temp_dir: str,
        windows: list[ChunkWindow],
    ) -> list[str]:
        paths: list[str] = []
        for window in windows:
            output_path = os.path.join(temp_dir, f"pipeline_chunk_{window.index:03d}.wav")
            paths.append(self._audio_chunker.extract_pipeline_chunk(temp_file_path, output_path, window))
        return paths
```

Ensure `__init__` has an audio chunker instance:

```python
        self._audio_chunker = AudioChunkingService()
```

If constructing `AudioChunkingService()` in tests triggers ffmpeg checks, patch `_check_ffmpeg_availability` in the affected tests or lazily instantiate the chunker in `_get_audio_chunker()`.

- [x] **Step 4: Verify GREEN**

Run:

```bash
uv run python -m pytest tests/unit/test_service.py -q -k extract_chunk_files
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/services/transcription.py tests/unit/test_service.py
git commit -m "feat: extract pipeline audio chunks for long-form alignment"
```

---

### Task 7: Add Per-Chunk ASR Text

**Files:**
- Modify: `src/services/transcription.py`
- Test: `tests/unit/test_service.py`

- [x] **Step 1: Write failing test for per-chunk ASR text**

Append to `tests/unit/test_service.py`:

```python
@pytest.mark.asyncio
async def test_long_form_pipeline_should_align_each_chunk_with_its_own_transcript(funasr_spec):
    svc = _setup_service(funasr_spec)
    windows = [
        ChunkWindow(index=0, start=0.0, end=300.0, emit_start=0.0, emit_end=285.0),
        ChunkWindow(index=1, start=270.0, end=600.0, emit_start=285.0, emit_end=600.0),
    ]

    async def fake_transcribe(temp_file_path, params, request_id, alias, pipeline_reserved=False):
        if temp_file_path.endswith("chunk_000.wav"):
            return {"text": "chunk zero", "segments": None, "duration": 300.0, "language": "en"}
        return {"text": "chunk one", "segments": None, "duration": 330.0, "language": "en"}

    svc._transcribe_with_alias = fake_transcribe

    result = await svc._transcribe_chunks_with_alias(
        chunk_paths=["/tmp/chunk_000.wav", "/tmp/chunk_001.wav"],
        windows=windows,
        params={"output_format": "json", "language": "en"},
        request_id="req",
        alias="qwen3-asr",
        pipeline_reserved=True,
    )

    assert result == ["chunk zero", "chunk one"]
```

- [x] **Step 2: Run the test to verify RED**

Run:

```bash
uv run python -m pytest tests/unit/test_service.py -q -k transcribe_each_chunk
```

Expected: fails because `_transcribe_chunks_with_alias` does not exist.

- [x] **Step 3: Implement chunk ASR helper**

Add method to `TranscriptionService`:

```python
    async def _transcribe_chunks_with_alias(
        self,
        *,
        chunk_paths: list[str],
        windows: list[ChunkWindow],
        params: dict[str, object],
        request_id: str,
        alias: str,
        pipeline_reserved: bool,
    ) -> list[str]:
        if len(chunk_paths) != len(windows):
            raise ValueError("chunk_paths and windows must have the same length")

        texts: list[str] = []
        chunk_params = {**params, "output_format": "json"}
        for chunk_path, window in zip(chunk_paths, windows):
            result = await self._transcribe_with_alias(
                chunk_path,
                chunk_params,
                f"{request_id}:chunk-{window.index}",
                alias,
                pipeline_reserved=pipeline_reserved,
            )
            if not isinstance(result, dict):
                raise TypeError("Chunk transcription must return a JSON object")
            text = result.get("text")
            texts.append(text if isinstance(text, str) else "")
        return texts
```

- [x] **Step 4: Verify GREEN**

Run:

```bash
uv run python -m pytest tests/unit/test_service.py -q -k transcribe_each_chunk
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/services/transcription.py tests/unit/test_service.py
git commit -m "feat: align long-form chunks with per-chunk qwen transcripts"
```

---

### Task 8: Route Long-Form Pipeline Through Real Chunked ASR and Alignment

**Files:**
- Modify: `src/services/transcription.py`
- Test: `tests/unit/test_service.py`

- [x] **Step 1: Write failing test for long-form routing**

Append to `tests/unit/test_service.py`:

```python
@pytest.mark.asyncio
async def test_long_form_pipeline_should_extract_transcribe_and_align_real_chunks(funasr_spec, tmp_path):
    from src.core.pipeline_registry import lookup_profile

    svc = _setup_service(funasr_spec)
    profile = replace(lookup_profile("qwen3-sortformer"), requestable=True)
    transcript = {"text": "full text", "segments": None, "duration": 600.0, "language": "en"}

    extracted_windows: list[ChunkWindow] = []

    def fake_extract_chunks(temp_file_path, temp_dir, windows):
        extracted_windows.extend(windows)
        return [str(tmp_path / f"chunk_{window.index:03d}.wav") for window in windows]

    async def fake_transcribe(temp_file_path, params, request_id, alias, pipeline_reserved=False):
        return transcript

    async def fake_transcribe_chunks(**kwargs):
        return [f"text {index}" for index, _path in enumerate(kwargs["chunk_paths"])]

    async def fake_align_chunks(**kwargs):
        return [AlignedWord(text="hello", start=10.0, end=10.5)]

    async def fake_diarize(temp_file_path, request_id, alias, pipeline_reserved=False):
        return [SpeakerTurn(speaker="Speaker 0", start=0.0, end=600.0)]

    svc._extract_pipeline_chunks = fake_extract_chunks
    svc._transcribe_with_alias = fake_transcribe
    svc._transcribe_chunks_with_alias = AsyncMock(side_effect=fake_transcribe_chunks)
    svc._align_chunks_with_alias = AsyncMock(side_effect=fake_align_chunks)
    svc._diarize_with_alias = fake_diarize
    svc._switch_worker = AsyncMock(side_effect=lambda spec: setattr(svc, "_current_model_spec", spec))
    svc._restore_resident_model = AsyncMock()

    result = await svc._run_decoupled_pipeline("audio.wav", {"output_format": "json"}, "req", profile)

    assert extracted_windows
    assert svc._transcribe_chunks_with_alias.await_count == 1
    assert svc._align_chunks_with_alias.await_count == 1
    assert result["segments"] == [
        {"id": 0, "speaker": "Speaker 0", "start": 10.0, "end": 10.5, "text": "hello"}
    ]
```

- [x] **Step 2: Run the test to verify RED**

Run:

```bash
uv run python -m pytest tests/unit/test_service.py -q -k long_form_pipeline
```

Expected: fails because `_run_decoupled_pipeline` still uses single-file alignment for long audio.

- [x] **Step 3: Implement long-form routing with real chunks**

Add constants near the top of `src/services/transcription.py`:

```python
PIPELINE_ALIGN_CHUNK_SECONDS = 300.0
PIPELINE_ALIGN_OVERLAP_SECONDS = 15.0
```

Add import:

```python
from src.adapters.pipeline_chunking import build_chunk_plan
```

Inside `_run_decoupled_pipeline`, use this branch when `duration > PIPELINE_ALIGN_CHUNK_SECONDS`:

```python
                        windows = build_chunk_plan(
                            duration_seconds=float(duration),
                            chunk_seconds=PIPELINE_ALIGN_CHUNK_SECONDS,
                            overlap_seconds=PIPELINE_ALIGN_OVERLAP_SECONDS,
                        )
                        temp_dir = tempfile.mkdtemp(prefix="asr_pipeline_chunks_")
                        try:
                            chunk_paths = self._extract_pipeline_chunks(temp_file_path, temp_dir, windows)
                            chunk_texts = await self._transcribe_chunks_with_alias(
                                chunk_paths=chunk_paths,
                                windows=windows,
                                params=params,
                                request_id=request_id,
                                alias=profile.transcription_alias,
                                pipeline_reserved=True,
                            )
                            aligned_words = await self._align_chunks_with_alias(
                                chunk_paths=chunk_paths,
                                chunk_texts=chunk_texts,
                                windows=windows,
                                language=self._resolve_alignment_language(params, transcript_result),
                                request_id=request_id,
                                alias=profile.alignment_alias,
                                pipeline_reserved=True,
                            )
                        finally:
                            shutil.rmtree(temp_dir, ignore_errors=True)
```

Do not pass the original full audio path into `_align_chunks_with_alias` for long-form audio.

- [x] **Step 4: Verify GREEN**

Run:

```bash
uv run python -m pytest tests/unit/test_service.py -q -k long_form_pipeline
```

Expected: test passes.

- [ ] **Step 5: Commit**

```bash
git add src/services/transcription.py tests/unit/test_service.py
git commit -m "feat: route long-form pipeline through real chunks"
```

---

### Task 9: Add Chunked Diarization and Deterministic Speaker Reconciliation

**Files:**
- Modify: `src/services/transcription.py`
- Test: `tests/unit/test_service.py`

- [x] **Step 1: Write failing test for chunked diarization offsets**

Append to `tests/unit/test_service.py`:

```python
@pytest.mark.asyncio
async def test_chunked_diarization_should_offset_turns_and_drop_overlap(funasr_spec):
    svc = _setup_service(funasr_spec)
    windows = [
        ChunkWindow(index=0, start=0.0, end=300.0, emit_start=0.0, emit_end=285.0),
        ChunkWindow(index=1, start=270.0, end=600.0, emit_start=285.0, emit_end=600.0),
    ]

    async def fake_diarize(temp_file_path, request_id, alias, pipeline_reserved=False):
        if temp_file_path.endswith("chunk_000.wav"):
            return [SpeakerTurn(speaker="Speaker 0", start=10.0, end=20.0)]
        return [
            SpeakerTurn(speaker="Speaker 0", start=1.0, end=2.0),
            SpeakerTurn(speaker="Speaker 1", start=20.0, end=30.0),
        ]

    svc._diarize_with_alias = fake_diarize

    result = await svc._diarize_chunks_with_alias(
        chunk_paths=["/tmp/chunk_000.wav", "/tmp/chunk_001.wav"],
        windows=windows,
        request_id="req",
        alias="sortformer-diar",
        pipeline_reserved=True,
    )

    assert result == [
        SpeakerTurn(speaker="Speaker 0", start=10.0, end=20.0),
        SpeakerTurn(speaker="Speaker 1", start=290.0, end=300.0),
    ]
```

- [x] **Step 2: Run the test to verify RED**

Run:

```bash
uv run python -m pytest tests/unit/test_service.py -q -k chunked_diarization
```

Expected: fails because `_diarize_chunks_with_alias` does not exist.

- [x] **Step 3: Implement chunk diarization helper**

Add import:

```python
from src.adapters.pipeline_chunking import offset_turns_to_global_timeline
```

Add method:

```python
    async def _diarize_chunks_with_alias(
        self,
        *,
        chunk_paths: list[str],
        windows: list[ChunkWindow],
        request_id: str,
        alias: str,
        pipeline_reserved: bool,
    ) -> list[SpeakerTurn]:
        if len(chunk_paths) != len(windows):
            raise ValueError("chunk_paths and windows must have the same length")

        merged: list[SpeakerTurn] = []
        for chunk_path, window in zip(chunk_paths, windows):
            turns = await self._diarize_with_alias(
                chunk_path,
                f"{request_id}:chunk-{window.index}",
                alias,
                pipeline_reserved=pipeline_reserved,
            )
            merged.extend(offset_turns_to_global_timeline(turns, window))
        return merged
```

- [x] **Step 4: Add conservative speaker reconciliation note in code**

For this task, keep local speaker labels as returned per chunk after offset filtering. Do not claim stable cross-chunk identity yet. Add this explicit implementation note above the helper:

```python
        # SPEC-013: local Sortformer labels are chunk-local. Public enablement
        # requires validated cross-chunk speaker reconciliation.
```

- [x] **Step 5: Verify GREEN**

Run:

```bash
uv run python -m pytest tests/unit/test_service.py -q -k chunked_diarization
```

Expected: test passes.

- [ ] **Step 6: Commit**

```bash
git add src/services/transcription.py tests/unit/test_service.py
git commit -m "feat: add chunked diarization timeline stitching"
```

---

### Task 10: Wire Quality Gates Into Pipeline Result

**Files:**
- Modify: `src/services/transcription.py`
- Test: `tests/unit/test_service.py`
- Test: `tests/integration/test_model_api.py`

- [x] **Step 1: Write failing service test for tail collapse**

Append to `tests/unit/test_service.py`:

```python
@pytest.mark.asyncio
async def test_pipeline_should_fail_loudly_when_alignment_quality_gate_fails(funasr_spec):
    from src.core.pipeline_registry import lookup_profile

    svc = _setup_service(funasr_spec)
    profile = replace(lookup_profile("qwen3-sortformer"), requestable=True)
    collapsed = [AlignedWord(text=f"w{i}", start=245.04, end=245.04) for i in range(12)]

    async def fake_transcribe(temp_file_path, params, request_id, alias, pipeline_reserved=False):
        return {"text": " ".join(word.text for word in collapsed), "segments": None, "duration": 600.0, "language": "en"}

    async def fake_align(temp_file_path, text, language, request_id, alias, pipeline_reserved=False):
        return collapsed

    svc._transcribe_with_alias = fake_transcribe
    svc._align_with_alias = fake_align
    svc._switch_worker = AsyncMock(side_effect=lambda spec: setattr(svc, "_current_model_spec", spec))
    svc._restore_resident_model = AsyncMock()

    with pytest.raises(ValueError, match="alignment quality gate failed"):
        await svc._run_decoupled_pipeline("audio.wav", {"output_format": "json"}, "req", profile)
```

- [x] **Step 2: Run the test to verify RED**

Run:

```bash
uv run python -m pytest tests/unit/test_service.py -q -k quality_gate
```

Expected: fails because pipeline does not call `validate_aligned_word_quality`.

- [x] **Step 3: Implement gate call**

Add import:

```python
from src.adapters.pipeline_chunking import validate_aligned_word_quality
```

Before diarization in `_run_decoupled_pipeline`, after aligned words are produced:

```python
                if aligned_words is not None:
                    duration_for_quality = transcript_result.get("duration")
                    if isinstance(duration_for_quality, int | float) and not isinstance(duration_for_quality, bool):
                        validate_aligned_word_quality(
                            aligned_words,
                            expected_duration_seconds=float(duration_for_quality),
                        )
```

- [x] **Step 4: Verify GREEN**

Run:

```bash
uv run python -m pytest tests/unit/test_service.py tests/integration/test_model_api.py -q -k "quality_gate or qwen3-sortformer"
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/services/transcription.py tests/unit/test_service.py tests/integration/test_model_api.py
git commit -m "feat: fail loud on collapsed long-form alignment"
```

---

### Task 11: Real-Runtime Probe Script and Public Enablement Gate

**Files:**
- Create: `scripts/probe_qwen3_sortformer_longform.py`
- Modify: `docs/SPEC-013-Production-Decoupled-Diarization-Pipeline.md`
- Modify after probe success only: `src/core/pipeline_registry.py`, `MODELS.md`, `tests/integration/test_model_api.py`

- [x] **Step 1: Add probe script**

Create `scripts/probe_qwen3_sortformer_longform.py`:

```python
import argparse
import json
from pathlib import Path

from fastapi import UploadFile

from src.core.pipeline_registry import lookup_profile
from src.services.transcription import TranscriptionService


async def main_async(audio_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = lookup_profile("qwen3-sortformer")
    service = TranscriptionService(engine_type="mlx", model_id="mlx-community/Qwen3-ASR-1.7B-8bit")
    service.is_running = True
    await service.start_worker()
    try:
        with audio_path.open("rb") as handle:
            result = await service.submit_pipeline(
                UploadFile(file=handle, filename=audio_path.name),
                {"output_format": "json", "language": "en"},
                request_id="probe-qwen3-sortformer-longform",
                profile=profile,
            )
        (output_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        await service.stop_worker()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import asyncio

    asyncio.run(main_async(Path(args.audio), Path(args.output)))


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Keep profile gated**

Verify `src/core/pipeline_registry.py` still contains:

```python
requestable=False
```

If the probe uses the internal `submit_pipeline(...)`, it may temporarily pass `replace(profile, requestable=True)` inside the script. Do not enable public API yet.

- [ ] **Step 3: Run short and long real-runtime probes**

Run short sample first:

```bash
uv run python scripts/probe_qwen3_sortformer_longform.py \
  --audio tests/fixtures/two_speakers_60s.wav \
  --output /tmp/local_asr_qwen3_sortformer_probe_60s
```

Then run the 10-minute local sample:

```bash
uv run python scripts/probe_qwen3_sortformer_longform.py \
  --audio tests/fixtures/Blair_FEB_09_last10min.wav \
  --output /tmp/local_asr_qwen3_sortformer_probe_10m
```

Expected: 60s passes; 10m has full-duration aligned timeline without max timestamp collapse.

- [ ] **Step 4: Only after 60s and 10m pass, run 5-hour validation**

Run:

```bash
uv run python scripts/probe_qwen3_sortformer_longform.py \
  --audio /path/to/5h.wav \
  --output /tmp/local_asr_qwen3_sortformer_probe_5h
```

Expected: no tail collapse, monotonic aligned words, full-duration coverage, readable speaker-labeled segments.

- [ ] **Step 5: Enable public requestability only after 5-hour validation**

Modify `src/core/pipeline_registry.py`:

```python
requestable=True
```

Update `MODELS.md` and `docs/SPEC-013-Production-Decoupled-Diarization-Pipeline.md` to say `qwen3-sortformer` is requestable with 5-hour validation evidence.

- [ ] **Step 6: Commit**

```bash
git add scripts/probe_qwen3_sortformer_longform.py docs/SPEC-013-Production-Decoupled-Diarization-Pipeline.md
git commit -m "test: add qwen3 sortformer long-form probe script"
```

If requestability was enabled after probe success:

```bash
git add src/core/pipeline_registry.py MODELS.md tests/integration/test_model_api.py docs/SPEC-013-Production-Decoupled-Diarization-Pipeline.md
git commit -m "feat: enable qwen3 sortformer after long-form probe"
```

---

### Task 12: Lightweight Cross-Chunk Speaker Reconciliation (Priority)

**Files:**
- Modify: `src/adapters/pipeline_chunking.py`
- Modify: `src/services/transcription.py`
- Test: `tests/unit/test_pipeline_chunking.py`
- Test: `tests/unit/test_service.py`

- [x] **Step 1: Write a failing unit test for overlap-based label remap**

Append to `tests/unit/test_pipeline_chunking.py`:

```python
def test_reconcile_chunk_speaker_labels_should_remap_swapped_labels_by_overlap() -> None:
    existing = [
        SpeakerTurn(speaker="Speaker 1", start=270.0, end=277.5),
        SpeakerTurn(speaker="Speaker 0", start=277.5, end=285.0),
    ]
    chunk_turns = [
        SpeakerTurn(speaker="Speaker 0", start=270.0, end=277.5),
        SpeakerTurn(speaker="Speaker 1", start=277.5, end=285.0),
        SpeakerTurn(speaker="Speaker 1", start=285.0, end=305.0),
    ]

    result = reconcile_chunk_speaker_labels(
        existing_turns=existing,
        chunk_turns=chunk_turns,
        overlap_start=270.0,
        overlap_end=285.0,
    )

    assert result == [
        SpeakerTurn(speaker="Speaker 1", start=270.0, end=277.5),
        SpeakerTurn(speaker="Speaker 0", start=277.5, end=285.0),
        SpeakerTurn(speaker="Speaker 0", start=285.0, end=305.0),
    ]
```

- [x] **Step 2: Verify RED**

Run:

```bash
uv run python -m pytest tests/unit/test_pipeline_chunking.py -q -k reconcile_chunk_speaker_labels
```

Expected: fails because helper does not exist.

- [x] **Step 3: Implement minimal reconciliation helper**

Add to `src/adapters/pipeline_chunking.py`:

```python
def reconcile_chunk_speaker_labels(
    *,
    existing_turns: list[SpeakerTurn],
    chunk_turns: list[SpeakerTurn],
    overlap_start: float,
    overlap_end: float,
) -> list[SpeakerTurn]:
    ...
```

Implementation rule:
- compute overlap scores between `(chunk_speaker, existing_speaker)` only inside overlap window,
- greedily assign one chunk speaker to one existing speaker by highest overlap,
- remap the full chunk turn list using that mapping,
- keep unmatched speakers unchanged.

- [x] **Step 4: Add failing service-level orchestration test**

Append to `tests/unit/test_service.py` a test where:
- chunk 0 emits overlap turns as `Speaker 1` then `Speaker 0`,
- chunk 1 swaps labels in the overlap but should emit post-overlap turn as `Speaker 0` after remap.

- [x] **Step 5: Verify RED (service)**

Run:

```bash
uv run python -m pytest tests/unit/test_service.py -q -k chunked_diarization_reconcile
```

Expected: fails because `_diarize_chunks_with_alias` only offsets turns without remap.

- [x] **Step 6: Implement minimal service wiring**

In `src/services/transcription.py`:
- build chunk-global turns for current chunk,
- call `reconcile_chunk_speaker_labels(...)` against already-merged turns using current overlap window,
- then apply emit-window clipping and append to merged result.

- [x] **Step 7: Verify GREEN**

Run:

```bash
uv run python -m pytest tests/unit/test_pipeline_chunking.py tests/unit/test_service.py -q -k "reconcile_chunk_speaker_labels or chunked_diarization_reconcile"
```

Expected: both new tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/adapters/pipeline_chunking.py src/services/transcription.py tests/unit/test_pipeline_chunking.py tests/unit/test_service.py
git commit -m "feat: add lightweight cross-chunk speaker reconciliation"
```

---

### Task 13: Sortformer Runtime Parameter Tuning

**Files:**
- Modify: `src/core/mlx_sortformer_diarizer.py`
- Test: `tests/unit/test_mlx_sortformer_diarizer.py`

- [x] **Step 1: Add a unit test that locks runtime generation parameters**

The diarizer test now verifies that Sortformer is called with explicit
`threshold`, `min_duration`, and `merge_gap` values.

- [x] **Step 2: Tune Sortformer for lower `Unknown` coverage**

Runtime parameters:

```python
SORTFORMER_THRESHOLD = 0.35
SORTFORMER_MIN_DURATION = 0.2
SORTFORMER_MERGE_GAP = 0.3
```

10-minute Blair probe result:

| Probe | Segments | Max end | Zero-duration segments | `Unknown` duration |
|-------|----------|---------|------------------------|--------------------|
| Baseline | 15 | ~596.76s | 2 | ~275.44s |
| Overlap reconciliation only | 15 | ~596.76s | 2 | ~275.44s |
| Word-level fallback experiment | 2 | ~596.76s | 0 | ~513.96s |
| Sortformer tuned | 73 | ~596.76s | 5 | ~17.41s |

60-second two-speaker regression check:

- unchanged after tuning,
- 4 speaker-labeled segments,
- 2 speakers detected,
- max end remains 60.0s.

Decision:

- Keep overlap reconciliation plus Sortformer parameter tuning.
- Do not keep the word-level fallback experiment; it regressed the real
  10-minute probe by increasing `Unknown` coverage.
- Keep `qwen3-sortformer` non-requestable until restore semantics and 5-hour
  validation pass.

---

## Final Verification Sweep

- [ ] Run lint on touched files:

```bash
uv run ruff check src/adapters/pipeline_chunking.py src/adapters/audio_chunking.py src/services/transcription.py tests/unit/test_pipeline_chunking.py tests/unit/test_audio_chunking.py tests/unit/test_service.py tests/integration/test_model_api.py scripts/probe_qwen3_sortformer_longform.py
```

- [ ] Run unit + integration tests:

```bash
uv run python -m pytest tests/unit tests/integration -q
```

- [ ] Confirm local fixture remains untracked unless explicitly requested:

```bash
git status --short
```

Expected: `tests/fixtures/Blair_FEB_09_last10min.wav` is not staged.

---

## Known Production Risk

Cross-chunk speaker identity is not fully solved by timestamp overlap alone. Overlap-based remapping can reconcile adjacent chunks when the same speakers are active in the overlap, but it can split the same person into multiple global speakers if a speaker disappears and returns later. Do not mark `qwen3-sortformer` as production-requestable until real 10-minute and 5-hour probes show acceptable speaker stability for the target puresubs use case.
