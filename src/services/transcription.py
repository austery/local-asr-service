import asyncio
import logging
import multiprocessing
import os
import queue as _stdlib_queue
import shutil
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Literal, TypedDict

from fastapi import UploadFile

from src.adapters.audio_chunking import AudioChunkingService
from src.adapters.pipeline_chunking import (
    ChunkWindow,
    build_chunk_plan,
    clip_turns_to_emit_window,
    offset_words_to_global_timeline,
    reconcile_chunk_speaker_labels,
    validate_aligned_word_quality,
)
from src.adapters.segment_alignment import align_speakers
from src.config import (
    APPLE_SPEECH_MAX_CONCURRENCY,
    APPLE_SPEECH_WORKER_PATH,
    APPLE_SPEECH_WORKER_TIMEOUT_SEC,
)
from src.core.alignment_port import AlignedWord, normalize_alignment_language
from src.core.apple_speech_engine import AppleSpeechEngine, AppleSpeechEngineConfig
from src.core.apple_speech_port import AppleSpeechModule
from src.core.base_engine import EngineCapabilities
from src.core.diarization_port import SpeakerTurn
from src.core.model_registry import ModelSpec
from src.core.pipeline_registry import PipelineProfile
from src.workers.model_worker import WorkerJob, run_worker


class TranscriptionResultDict(TypedDict, total=False):
    text: str
    segments: list[dict[str, object]] | None
    duration: float
    language: str


TranscriptionResult = str | TranscriptionResultDict
WorkerStartupMessage = tuple[Literal["READY"], None] | tuple[Literal["LOAD_ERROR"], str]
WorkerResultMessage = (
    tuple[Literal["RESULT"], str, object]
    | tuple[Literal["ERROR"], str, str]
    | tuple[Literal["ERROR"], str, str, str]
    | tuple[Literal["IDLE_EXIT"], None]
)
WorkerMessage = WorkerStartupMessage | WorkerResultMessage
PIPELINE_ALIGN_CHUNK_SECONDS = 300.0
PIPELINE_ALIGN_OVERLAP_SECONDS = 15.0
PIPELINE_PENDING_DRAIN_TIMEOUT_SECONDS = 30.0
PIPELINE_PENDING_DRAIN_POLL_SECONDS = 0.01


class PipelineQualityError(ValueError):
    """Pipeline produced structurally invalid alignment output."""


class WorkerRemoteError(RuntimeError):
    """Exception raised by the worker process and reconstructed in the parent."""

    def __init__(self, exc_type_name: str, message: str) -> None:
        super().__init__(message)
        self.exc_type_name = exc_type_name


class TranscriptionService:
    """
    Manages a ModelWorker child process via multiprocessing.Queue IPC.

    The worker subprocess self-terminates on idle timeout, allowing the OS to
    reclaim ML framework memory (MPS/CUDA) that cannot be freed in-process.
    """

    def __init__(
        self,
        engine_type: str,
        model_id: str,
        max_queue_size: int = 50,
        initial_model_spec: ModelSpec | None = None,
        idle_timeout: int = 60,
    ) -> None:
        self._engine_type = engine_type
        self._model_id = model_id
        self._current_model_spec = initial_model_spec
        self._idle_timeout = idle_timeout
        self._max_queue_size = max_queue_size

        self._worker: multiprocessing.Process | None = None
        self._job_queue: multiprocessing.Queue[WorkerJob | None] | None = None
        self._result_queue: multiprocessing.Queue[WorkerMessage] | None = None
        self._pending: dict[str, asyncio.Future[object]] = {}
        self._temp_dirs: dict[str, str] = {}
        self._pipeline_lock: asyncio.Lock = asyncio.Lock()
        self._spawn_lock: asyncio.Lock = asyncio.Lock()
        self._result_reader_task: asyncio.Task[None] | None = None
        self._audio_chunker: AudioChunkingService | None = None
        self._sidecar_pending: set[str] = set()
        self._sidecar_semaphore = asyncio.Semaphore(APPLE_SPEECH_MAX_CONCURRENCY)
        self._apple_speech_engines: dict[str, AppleSpeechEngine] = {}
        self.is_running = False

        self.logger = logging.getLogger(__name__)

    @property
    def current_model_spec(self) -> ModelSpec | None:
        return self._current_model_spec

    @property
    def model_loaded(self) -> bool:
        """True if worker subprocess is alive."""
        return self._worker is not None and self._worker.is_alive()

    def _get_audio_chunker(self) -> AudioChunkingService:
        if self._audio_chunker is None:
            self._audio_chunker = AudioChunkingService()
        return self._audio_chunker

    @property
    def capabilities(self) -> EngineCapabilities:
        """Return capabilities from current model spec — no live engine needed."""
        if self._current_model_spec is not None:
            return self._current_model_spec.capabilities
        return EngineCapabilities()

    @property
    def queue_size(self) -> int:
        """Number of jobs currently in-flight across worker and sidecar paths."""
        return self._active_job_count()

    @property
    def max_queue_size(self) -> int:
        """Maximum allowed concurrent in-flight jobs."""
        return self._max_queue_size

    async def start_worker(self) -> None:
        """Mark service as running. Worker spawns lazily on first request."""
        self.is_running = True
        self.logger.info("🚦 Service initialized (worker spawns on first request).")

    async def stop_worker(self) -> None:
        """Gracefully stop worker subprocess and result reader."""
        self.is_running = False
        await self._shutdown_worker()
        if self._result_reader_task and not self._result_reader_task.done():
            self._result_reader_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._result_reader_task

    async def submit(
        self,
        file: UploadFile,
        params: dict[str, object],
        request_id: str = "unknown",
        model_spec: ModelSpec | None = None,
    ) -> TranscriptionResult:
        if self._active_job_count() >= self._max_queue_size:
            self.logger.warning(f"[{request_id}] Queue full, rejecting request")
            raise RuntimeError("Service busy: Queue is full.")

        temp_dir = tempfile.mkdtemp(prefix="asr_task_")
        try:
            file_ext = os.path.splitext(file.filename or "upload.wav")[1] or ".wav"
            temp_path = os.path.join(temp_dir, f"original{file_ext}")
            with open(temp_path, "wb") as buf:
                shutil.copyfileobj(file.file, buf)

            if self._is_apple_speech_spec(model_spec):
                if model_spec is None:
                    raise RuntimeError("Apple Speech request requires a resolved model spec")
                try:
                    result = await self._submit_apple_speech_job(
                        temp_file_path=temp_path,
                        params=params,
                        request_id=request_id,
                    )
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)
            else:
                result = await self._submit_worker_job(
                    temp_file_path=temp_path,
                    params=params,
                    request_id=request_id,
                    model_spec=model_spec,
                    temp_dir=temp_dir,
                )
            return self._coerce_transcription_result(result)

        except BaseException:
            self._discard_request_state(request_id)
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    async def submit_pipeline(
        self,
        file: UploadFile,
        params: dict[str, object],
        request_id: str,
        profile: PipelineProfile,
    ) -> TranscriptionResult:
        if not profile.requestable:
            raise RuntimeError(f"Pipeline profile '{profile.alias}' is not enabled for requests.")
        if self._active_job_count() >= self._max_queue_size:
            self.logger.warning(f"[{request_id}] Queue full, rejecting pipeline request")
            raise RuntimeError("Service busy: Queue is full.")

        temp_dir = tempfile.mkdtemp(prefix="asr_pipeline_")
        try:
            file_ext = os.path.splitext(file.filename or "upload.wav")[1] or ".wav"
            temp_path = os.path.join(temp_dir, f"original{file_ext}")
            with open(temp_path, "wb") as buf:
                shutil.copyfileobj(file.file, buf)

            return await self._run_decoupled_pipeline(
                temp_file_path=temp_path,
                params=params,
                request_id=request_id,
                profile=profile,
            )
        except BaseException:
            self._discard_pipeline_request_state(request_id)
            raise
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def _submit_worker_job(
        self,
        temp_file_path: str,
        params: dict[str, object],
        request_id: str,
        model_spec: ModelSpec | None,
        temp_dir: str | None = None,
        job_kind: Literal["transcribe", "align", "diarize"] = "transcribe",
        aligner_alias: str | None = None,
        diarizer_alias: str | None = None,
        pipeline_reserved: bool = False,
    ) -> object:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[object] = loop.create_future()

        try:
            if pipeline_reserved:
                await self._enqueue_worker_job(
                    future=future,
                    temp_file_path=temp_file_path,
                    params=params,
                    request_id=request_id,
                    model_spec=model_spec,
                    temp_dir=temp_dir,
                    job_kind=job_kind,
                    aligner_alias=aligner_alias,
                    diarizer_alias=diarizer_alias,
                )
            else:
                async with self._pipeline_lock:
                    await self._enqueue_worker_job(
                        future=future,
                        temp_file_path=temp_file_path,
                        params=params,
                        request_id=request_id,
                        model_spec=model_spec,
                        temp_dir=temp_dir,
                        job_kind=job_kind,
                        aligner_alias=aligner_alias,
                        diarizer_alias=diarizer_alias,
                    )
        except BaseException:
            self._discard_request_state(request_id)
            raise

        try:
            return await future
        except BaseException:
            self._discard_request_state(request_id)
            raise

    def _active_job_count(self) -> int:
        return len(self._pending) + len(self._sidecar_pending)

    @staticmethod
    def _is_apple_speech_spec(model_spec: ModelSpec | None) -> bool:
        return model_spec is not None and model_spec.engine_type == "apple-speech"

    def _get_apple_speech_engine(self) -> AppleSpeechEngine:
        module: AppleSpeechModule = "speechTranscriber"
        engine = self._apple_speech_engines.get(module)
        if engine is None:
            engine = AppleSpeechEngine.from_config(
                AppleSpeechEngineConfig(
                    worker_path=Path(APPLE_SPEECH_WORKER_PATH),
                    timeout_seconds=APPLE_SPEECH_WORKER_TIMEOUT_SEC,
                ),
                module=module,
            )
            engine.load()
            self._apple_speech_engines[module] = engine
        return engine

    async def _submit_apple_speech_job(
        self,
        temp_file_path: str,
        params: dict[str, object],
        request_id: str,
    ) -> object:
        if self._active_job_count() >= self._max_queue_size:
            self.logger.warning(f"[{request_id}] Queue full, rejecting Apple Speech request")
            raise RuntimeError("Service busy: Queue is full.")

        self._sidecar_pending.add(request_id)
        try:
            async with self._sidecar_semaphore:
                engine = self._get_apple_speech_engine()
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

    async def _enqueue_worker_job(
        self,
        future: asyncio.Future[object],
        temp_file_path: str,
        params: dict[str, object],
        request_id: str,
        model_spec: ModelSpec | None,
        temp_dir: str | None,
        job_kind: Literal["transcribe", "align", "diarize"],
        aligner_alias: str | None,
        diarizer_alias: str | None,
    ) -> None:
        async with self._spawn_lock:
            if self._active_job_count() >= self._max_queue_size:
                self.logger.warning(f"[{request_id}] Queue full, rejecting worker job")
                raise RuntimeError("Service busy: Queue is full.")

            if model_spec is not None and model_spec != self._current_model_spec:
                await self._switch_worker(model_spec)
            elif not self.model_loaded:
                await self._spawn_worker()

            if self._job_queue is None:
                raise RuntimeError("Job queue is None after successful spawn — this is a bug")

            # Register AFTER the worker is ready so that _shutdown_worker (called
            # during a concurrent model switch) does not cancel this request's
            # future or delete its temp dir before the job is even queued.
            self._pending[request_id] = future
            if temp_dir is not None:
                self._temp_dirs[request_id] = temp_dir

            self._job_queue.put_nowait(WorkerJob(
                uid=request_id,
                temp_file_path=temp_file_path,
                params=params,
                job_kind=job_kind,
                requested_model_spec_alias=model_spec.alias if model_spec is not None else None,
                requested_aligner_alias=aligner_alias,
                requested_diarizer_alias=diarizer_alias,
            ))

    async def _transcribe_with_alias(
        self,
        temp_file_path: str,
        params: dict[str, object],
        request_id: str,
        alias: str,
        pipeline_reserved: bool = False,
    ) -> TranscriptionResult:
        model_spec = self._lookup_model_spec(alias)
        result = await self._submit_worker_job(
            temp_file_path=temp_file_path,
            params=params,
            request_id=f"{request_id}:transcribe",
            model_spec=model_spec,
            pipeline_reserved=pipeline_reserved,
        )
        return self._coerce_transcription_result(result)

    async def _align_with_alias(
        self,
        temp_file_path: str,
        text: str,
        language: str,
        request_id: str,
        alias: str,
        pipeline_reserved: bool = False,
    ) -> list[AlignedWord]:
        result = await self._submit_worker_job(
            temp_file_path=temp_file_path,
            params={"text": text, "language": language},
            request_id=f"{request_id}:align",
            model_spec=None,
            job_kind="align",
            aligner_alias=alias,
            pipeline_reserved=pipeline_reserved,
        )
        if not isinstance(result, list) or not all(isinstance(item, AlignedWord) for item in result):
            raise TypeError("Expected alignment result as list[AlignedWord]")
        return result

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
        for chunk_path, chunk_text, window in zip(chunk_paths, chunk_texts, windows, strict=True):
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
        for chunk_path, window in zip(chunk_paths, windows, strict=True):
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

    async def _extract_pipeline_chunks(
        self,
        temp_file_path: str,
        temp_dir: str,
        windows: list[ChunkWindow],
    ) -> list[str]:
        chunker = self._get_audio_chunker()
        paths: list[str] = []
        for window in windows:
            output_path = os.path.join(temp_dir, f"pipeline_chunk_{window.index:03d}.wav")
            paths.append(
                await asyncio.to_thread(
                    chunker.extract_pipeline_chunk,
                    temp_file_path,
                    output_path,
                    window,
                )
            )
        return paths

    async def _resolve_pipeline_duration(
        self,
        temp_file_path: str,
        transcript_result: TranscriptionResultDict,
        *,
        request_id: str,
    ) -> float | None:
        duration = transcript_result.get("duration")
        if (
            isinstance(duration, int | float)
            and not isinstance(duration, bool)
            and duration > 0.0
        ):
            return float(duration)

        try:
            return await asyncio.to_thread(
                self._get_audio_chunker().get_audio_duration,
                temp_file_path,
            )
        except Exception as exc:
            self.logger.warning(
                "[%s] Could not resolve audio duration for pipeline chunking from %s; falling back to short-form path: %s",
                request_id,
                temp_file_path,
                exc,
                exc_info=True,
            )
            return None

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
        for chunk_path, window in zip(chunk_paths, windows, strict=True):
            turns = await self._diarize_with_alias(
                chunk_path,
                f"{request_id}:chunk-{window.index}",
                alias,
                pipeline_reserved=pipeline_reserved,
            )

            chunk_turns_global: list[SpeakerTurn] = []
            for turn in turns:
                global_start = round(window.start + turn.start, 3)
                global_end = min(window.end, round(window.start + turn.end, 3))
                if global_end <= global_start:
                    continue
                chunk_turns_global.append(
                    SpeakerTurn(
                        speaker=turn.speaker,
                        start=global_start,
                        end=global_end,
                    )
                )

            if merged and chunk_turns_global and window.emit_start > window.start:
                chunk_turns_global = reconcile_chunk_speaker_labels(
                    existing_turns=merged,
                    chunk_turns=chunk_turns_global,
                    overlap_start=window.start,
                    overlap_end=window.emit_start,
                )

            merged.extend(clip_turns_to_emit_window(chunk_turns_global, window))
        return merged

    async def _diarize_with_alias(
        self,
        temp_file_path: str,
        request_id: str,
        alias: str,
        pipeline_reserved: bool = False,
    ) -> list[SpeakerTurn]:
        result = await self._submit_worker_job(
            temp_file_path=temp_file_path,
            params={},
            request_id=f"{request_id}:diarize",
            model_spec=None,
            job_kind="diarize",
            diarizer_alias=alias,
            pipeline_reserved=pipeline_reserved,
        )
        if not isinstance(result, list) or not all(isinstance(item, SpeakerTurn) for item in result):
            raise TypeError("Expected diarization result as list[SpeakerTurn]")
        return result

    def _lookup_model_spec(self, alias: str) -> ModelSpec:
        from src.core.model_registry import lookup

        return lookup(alias)

    @staticmethod
    def _coerce_transcription_result(result: object) -> TranscriptionResult:
        if isinstance(result, str):
            return result
        if not isinstance(result, dict):
            raise TypeError(f"Expected transcription result as str or dict, got {type(result).__name__}")

        coerced: TranscriptionResultDict = {}
        text = result.get("text", "")
        coerced["text"] = text if isinstance(text, str) else ""

        segments = result.get("segments")
        if segments is None or isinstance(segments, list):
            coerced["segments"] = segments
        else:
            raise TypeError("Expected transcription result 'segments' to be a list or None")

        duration = result.get("duration")
        if isinstance(duration, int | float) and not isinstance(duration, bool):
            coerced["duration"] = float(duration)

        language = result.get("language")
        if isinstance(language, str):
            coerced["language"] = language

        return coerced

    async def _run_decoupled_pipeline(
        self,
        temp_file_path: str,
        params: dict[str, object],
        request_id: str,
        profile: PipelineProfile,
    ) -> TranscriptionResult:
        async with self._pipeline_lock:
            await self._wait_for_pending_work_to_drain()
            previous_spec = self._current_model_spec
            pipeline_temp_dir: str | None = None
            try:
                target_spec = self._lookup_model_spec(profile.transcription_alias)
                if self._current_model_spec != target_spec:
                    await self._switch_worker(target_spec)

                transcript_result = await self._transcribe_with_alias(
                    temp_file_path,
                    params,
                    request_id,
                    profile.transcription_alias,
                    pipeline_reserved=True,
                )
                if not isinstance(transcript_result, dict):
                    return transcript_result

                transcript_text = transcript_result.get("text")
                if not isinstance(transcript_text, str) or not transcript_text.strip():
                    self.logger.warning(
                        "[%s] Decoupled pipeline has no transcript text to align for profile %s; returning transcript-only result",
                        request_id,
                        profile.alias,
                    )
                    return transcript_result

                aligned_words: list[AlignedWord] | None = None
                pipeline_chunk_paths: list[str] | None = None
                pipeline_windows: list[ChunkWindow] | None = None
                pipeline_duration = await self._resolve_pipeline_duration(
                    temp_file_path,
                    transcript_result,
                    request_id=request_id,
                )
                if pipeline_duration is not None:
                    transcript_result["duration"] = pipeline_duration
                if profile.alignment_alias is not None:
                    alignment_language = self._resolve_alignment_language(params, transcript_result)
                    if (
                        pipeline_duration is not None
                        and pipeline_duration > PIPELINE_ALIGN_CHUNK_SECONDS
                    ):
                        windows = build_chunk_plan(
                            duration_seconds=pipeline_duration,
                            chunk_seconds=PIPELINE_ALIGN_CHUNK_SECONDS,
                            overlap_seconds=PIPELINE_ALIGN_OVERLAP_SECONDS,
                        )
                        pipeline_temp_dir = tempfile.mkdtemp(prefix="asr_pipeline_chunks_")
                        chunk_paths = await self._extract_pipeline_chunks(
                            temp_file_path,
                            pipeline_temp_dir,
                            windows,
                        )
                        pipeline_chunk_paths = chunk_paths
                        pipeline_windows = windows
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
                            language=alignment_language,
                            request_id=request_id,
                            alias=profile.alignment_alias,
                            pipeline_reserved=True,
                        )
                    else:
                        aligned_words = await self._align_with_alias(
                            temp_file_path,
                            transcript_text,
                            alignment_language,
                            request_id,
                            profile.alignment_alias,
                            pipeline_reserved=True,
                        )

                if aligned_words is not None:
                    try:
                        validate_aligned_word_quality(
                            aligned_words,
                            expected_duration_seconds=pipeline_duration,
                        )
                    except ValueError as exc:
                        raise PipelineQualityError(str(exc)) from exc

                try:
                    if pipeline_chunk_paths is not None and pipeline_windows is not None:
                        speaker_turns = await self._diarize_chunks_with_alias(
                            chunk_paths=pipeline_chunk_paths,
                            windows=pipeline_windows,
                            request_id=request_id,
                            alias=profile.diarization_alias,
                            pipeline_reserved=True,
                        )
                    else:
                        speaker_turns = await self._diarize_with_alias(
                            temp_file_path,
                            request_id,
                            profile.diarization_alias,
                            pipeline_reserved=True,
                        )
                except Exception as exc:
                    if self._is_worker_lifecycle_error(exc):
                        raise
                    if self._is_diarization_contract_error(exc):
                        raise
                    if isinstance(exc, MemoryError | OSError | TimeoutError):
                        raise
                    self.logger.warning(
                        "[%s] Decoupled pipeline diarization failed for profile %s; returning transcript-only result: %s",
                        request_id,
                        profile.alias,
                        exc,
                        exc_info=True,
                    )
                    return transcript_result

                if aligned_words is not None:
                    return {
                        **transcript_result,
                        "segments": self._align_words_to_speaker_segments(aligned_words, speaker_turns),
                    }

                segments = transcript_result.get("segments")
                if not isinstance(segments, list) or not segments:
                    self.logger.warning(
                        "[%s] Decoupled pipeline has no transcript segments to align for profile %s; returning transcript-only result",
                        request_id,
                        profile.alias,
                    )
                    return transcript_result

                try:
                    aligned_segments = align_speakers(
                        self._coerce_segment_list(segments),
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
                if pipeline_temp_dir is not None:
                    await self._remove_pipeline_temp_dir(pipeline_temp_dir)
                await self._restore_resident_model(previous_spec)

    @staticmethod
    def _coerce_segment_list(segments: list[object]) -> list[dict[str, object]]:
        if not all(isinstance(segment, dict) for segment in segments):
            raise TypeError("Expected transcription segments to be dictionaries")
        return segments

    @staticmethod
    def _resolve_alignment_language(
        params: dict[str, object],
        transcript_result: TranscriptionResultDict,
    ) -> str:
        language = params.get("language")
        if isinstance(language, str) and language.strip().lower() != "auto":
            return normalize_alignment_language(language)

        detected_language = transcript_result.get("language")
        if isinstance(detected_language, str) and detected_language.strip().lower() != "auto":
            return normalize_alignment_language(detected_language)

        return "English"

    @staticmethod
    def _align_words_to_speaker_segments(
        aligned_words: list[AlignedWord],
        speaker_turns: list[SpeakerTurn],
    ) -> list[dict[str, object]]:
        segments: list[dict[str, object]] = []
        current_speaker: str | None = None
        current_words: list[str] = []
        current_start = 0.0
        current_end = 0.0

        for word in aligned_words:
            if word.end <= word.start:
                continue
            speaker = TranscriptionService._speaker_for_word(word, speaker_turns)
            if current_speaker is None:
                current_speaker = speaker
                current_start = word.start
            elif speaker != current_speaker and current_words:
                segments.append({
                    "id": len(segments),
                    "speaker": current_speaker,
                    "start": current_start,
                    "end": current_end,
                    "text": " ".join(current_words),
                })
                current_speaker = speaker
                current_start = word.start
                current_words = []

            current_words.append(word.text)
            current_end = word.end

        if current_speaker is not None and current_words:
            segments.append({
                "id": len(segments),
                "speaker": current_speaker,
                "start": current_start,
                "end": current_end,
                "text": " ".join(current_words),
            })

        return segments

    @staticmethod
    def _speaker_for_word(word: AlignedWord, speaker_turns: list[SpeakerTurn]) -> str:
        best_speaker = "Unknown"
        best_overlap = 0.0
        for turn in speaker_turns:
            overlap = max(0.0, min(word.end, turn.end) - max(word.start, turn.start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn.speaker

        if best_overlap > 0:
            return best_speaker

        midpoint = (word.start + word.end) / 2
        for turn in speaker_turns:
            if turn.start <= midpoint <= turn.end:
                return turn.speaker
        return best_speaker

    async def _wait_for_pending_work_to_drain(
        self,
        *,
        timeout_seconds: float = PIPELINE_PENDING_DRAIN_TIMEOUT_SECONDS,
    ) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        while self._pending:
            remaining = deadline - loop.time()
            if remaining <= 0:
                pending_ids = ", ".join(sorted(self._pending))
                raise RuntimeError(
                    f"Timed out waiting for pending worker jobs to drain: {pending_ids}"
                )
            await asyncio.sleep(min(PIPELINE_PENDING_DRAIN_POLL_SECONDS, remaining))

    async def _remove_pipeline_temp_dir(self, temp_dir: str) -> None:
        await asyncio.to_thread(shutil.rmtree, temp_dir, ignore_errors=True)

    def _discard_request_state(self, request_id: str) -> None:
        future = self._pending.pop(request_id, None)
        if future is not None and not future.done():
            future.cancel()
        temp_dir = self._temp_dirs.pop(request_id, None)
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _discard_pipeline_request_state(self, request_id: str) -> None:
        self._discard_request_state(request_id)
        self._discard_request_state(f"{request_id}:transcribe")
        self._discard_request_state(f"{request_id}:align")
        self._discard_request_state(f"{request_id}:diarize")
        for uid in list(self._pending):
            if uid.startswith(f"{request_id}:chunk-"):
                self._discard_request_state(uid)

    async def _restore_resident_model(self, previous_spec: ModelSpec | None) -> None:
        async with self._spawn_lock:
            current_spec = self._current_model_spec
            if current_spec == previous_spec:
                return
            if self._pending:
                raise RuntimeError("Cannot restore resident model while worker jobs are still pending")
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

    @staticmethod
    def _is_diarization_contract_error(exc: Exception) -> bool:
        if isinstance(exc, (NotImplementedError, TypeError, KeyError)):
            return True
        contract_type_names = {"KeyError", "NotImplementedError", "TypeError", "ValueError"}
        if isinstance(exc, WorkerRemoteError):
            return exc.exc_type_name in contract_type_names
        if not isinstance(exc, RuntimeError):
            return False
        message = str(exc)
        return any(marker in message for marker in (
            "Unsupported job_kind",
            "Alignment job requires",
            "Unsupported alignment runtime",
            "Unknown alignment alias",
            "Diarization job requires",
            "Unsupported diarization runtime",
            "Unknown diarization alias",
        ))

    async def _stop_result_reader_task(self) -> None:
        task = self._result_reader_task
        if task is None:
            return
        if task.done():
            self._result_reader_task = None
            return
        if task is asyncio.current_task():
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        self._result_reader_task = None

    async def _spawn_worker(self, model_spec: ModelSpec | None = None) -> None:
        await self._stop_result_reader_task()

        for old_q in (self._job_queue, self._result_queue):
            if old_q is not None:
                try:
                    old_q.close()
                    old_q.join_thread()
                except Exception:
                    self.logger.warning("Failed to close stale worker queue during respawn", exc_info=True)

        self._job_queue = multiprocessing.Queue()
        self._result_queue = multiprocessing.Queue()

        effective_spec = model_spec or self._current_model_spec
        engine_type = effective_spec.engine_type if effective_spec else self._engine_type
        model_id = effective_spec.model_id if effective_spec else self._model_id

        self._worker = multiprocessing.Process(
            target=run_worker,
            args=(
                self._job_queue,
                self._result_queue,
                engine_type,
                model_id,
                self._idle_timeout,
            ),
            daemon=True,
        )
        self._worker.start()

        loop = asyncio.get_running_loop()
        try:
            msg = await asyncio.wait_for(
                loop.run_in_executor(None, self._result_queue.get),
                timeout=120.0,
            )
        except TimeoutError as exc:
            self._worker.terminate()
            self._worker = None
            raise RuntimeError("Worker failed to start within 120s timeout.") from exc

        if not isinstance(msg, tuple) or len(msg) != 2 or not isinstance(msg[0], str):
            self._worker.terminate()
            self._worker = None
            raise RuntimeError(f"Worker sent unexpected startup message: {msg!r}")

        if msg[0] == "LOAD_ERROR":
            self._worker.terminate()
            self._worker = None
            raise RuntimeError(f"Worker failed to load model: {msg[1]}")

        if msg[0] != "READY":
            self._worker.terminate()
            self._worker = None
            raise RuntimeError(f"Worker sent unexpected startup message: {msg!r}")

        if model_spec is not None:
            self._current_model_spec = model_spec

        if self._result_reader_task is None or self._result_reader_task.done():
            self._result_reader_task = asyncio.create_task(self._result_reader_loop())

        self.logger.info("✅ Worker subprocess ready.")

    async def _switch_worker(self, new_spec: ModelSpec) -> None:
        old_alias = self._current_model_spec.alias if self._current_model_spec else "unknown"
        self.logger.info(f"🔄 Switching worker model: {old_alias} → {new_spec.alias}")
        await self._shutdown_worker()
        await self._spawn_worker(new_spec)

    async def _shutdown_worker(self) -> None:
        """Gracefully shutdown worker subprocess and clean up IPC resources.

        Critical: This method MUST properly reap the child process and close
        all multiprocessing.Queue instances to avoid zombie processes and
        resource_tracker hangs that prevent the main process from exiting.
        """
        for _uid, fut in list(self._pending.items()):
            if not fut.done():
                fut.set_exception(RuntimeError("Worker terminated (model switch or shutdown)"))
        self._pending.clear()
        for temp_dir in self._temp_dirs.values():
            shutil.rmtree(temp_dir, ignore_errors=True)
        self._temp_dirs.clear()

        if self._worker is None:
            return

        # 1. Send shutdown sentinel (allows graceful engine.release())
        if self._job_queue is not None:
            try:
                self._job_queue.put(None, timeout=1.0)
            except Exception as exc:
                self.logger.warning("Failed to send shutdown sentinel to worker: %s", exc)

        # 2. Wait for graceful exit
        loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self._worker.join(timeout=5)),  # type: ignore[union-attr]
                timeout=6.0,
            )
        except TimeoutError:
            self.logger.warning("Worker did not exit gracefully within 5s timeout")

        # 3. Force-kill if still alive
        if self._worker.is_alive():
            self.logger.warning("Sending SIGTERM to worker subprocess")
            self._worker.terminate()

            # CRITICAL: join() after terminate() to reap the zombie process.
            # Without this, the process becomes a zombie and resource_tracker
            # cannot exit, causing the main process to hang indefinitely.
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: self._worker.join(timeout=3)),  # type: ignore[union-attr]
                    timeout=4.0,
                )
            except TimeoutError:
                self.logger.error("Worker did not respond to SIGTERM within 3s, using SIGKILL")
                self._worker.kill()  # SIGKILL (last resort)
                # Final join (no timeout wrap — must wait for kill to complete)
                await loop.run_in_executor(None, lambda: self._worker.join(timeout=2))  # type: ignore[union-attr]

        self._worker = None

        # 4. Clean up IPC queues to stop feeder threads
        # multiprocessing.Queue has a background "feeder thread" that serializes
        # and writes data to the underlying pipe. If not explicitly cleaned up,
        # this thread may block waiting for the pipe to flush, and resource_tracker
        # will not exit until all Queue resources are properly closed.
        for q in (self._job_queue, self._result_queue):
            if q is not None:
                try:
                    q.close()
                    q.join_thread()  # Wait for feeder thread to finish
                except Exception as exc:
                    self.logger.warning("Failed to clean up queue: %s", exc)

    async def _result_reader_loop(self) -> None:
        """Polls result_queue (non-blocking) every 50ms, resolves pending Futures."""
        _liveness_ticks = 0
        while self.is_running:
            if self._result_queue is None:
                await asyncio.sleep(0.05)
                continue
            try:
                msg = self._result_queue.get_nowait()
            except _stdlib_queue.Empty:
                _liveness_ticks += 1
                if _liveness_ticks >= 20:  # check liveness ~every 1s
                    _liveness_ticks = 0
                    if self._worker is not None and not self._worker.is_alive() and self._pending:
                        exit_code = self._worker.exitcode
                        self.logger.error(
                            "Worker process died unexpectedly (exit code %s) with %d pending job(s) — failing all",
                            exit_code,
                            len(self._pending),
                        )
                        self._fail_all_pending(
                            RuntimeError(f"Worker process died unexpectedly (exit code {exit_code})")
                        )
                        self._worker = None
                await asyncio.sleep(0.05)
                continue

            try:
                msg_type: str = msg[0]
                if msg_type == "RESULT":
                    self._resolve_future(msg[1], result=msg[2])
                elif msg_type == "ERROR":
                    if len(msg) == 4:
                        self._resolve_future(msg[1], error=WorkerRemoteError(msg[2], msg[3]))
                    else:
                        self._resolve_future(msg[1], error=RuntimeError(msg[2]))
                elif msg_type == "IDLE_EXIT":
                    self.logger.info("💤 Worker exited due to idle timeout — memory reclaimed by OS")
                    if self._worker:
                        worker = self._worker
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(
                            None, lambda current_worker=worker: current_worker.join(timeout=1)
                        )
                    self._worker = None
            except Exception:
                self.logger.exception("Unexpected error processing IPC message: %r", msg)

    def _fail_all_pending(self, error: Exception) -> None:
        """Fail all in-flight futures with the given error (e.g., after worker crash)."""
        for _uid, fut in list(self._pending.items()):
            if not fut.done():
                fut.set_exception(error)
        self._pending.clear()
        for temp_dir in self._temp_dirs.values():
            shutil.rmtree(temp_dir, ignore_errors=True)
        self._temp_dirs.clear()

    def _resolve_future(
        self,
        uid: str,
        result: object | None = None,
        error: Exception | None = None,
    ) -> None:
        future = self._pending.pop(uid, None)
        self._cleanup_temp(uid)
        if future is None or future.done():
            return
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(result)

    def _cleanup_temp(self, uid: str) -> None:
        temp_dir = self._temp_dirs.pop(uid, None)
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
