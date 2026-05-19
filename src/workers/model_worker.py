"""ModelWorker subprocess entry point for memory-isolated audio inference jobs.

Runs inside a child process. Communicates with the parent via multiprocessing
Queues using a simple IPC protocol:

  ("READY", None)          — engine loaded, ready for jobs
  ("LOAD_ERROR", str)      — engine.load() failed; process exits with code 1
  ("RESULT", uid, result)  — job succeeded
  ("ERROR", uid, exc_type, str) — job raised an exception
  ("IDLE_EXIT", None)      — idle timeout reached; process exits with code 0
"""
from __future__ import annotations

import logging
import queue
import sys
from dataclasses import dataclass, field
from multiprocessing import Queue
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from src.core.alignment_port import AlignmentPort
    from src.core.base_engine import ASREngine
    from src.core.diarization_port import DiarizationPort

logger = logging.getLogger(__name__)


@dataclass
class WorkerJob:
    """Picklable job descriptor passed from parent → worker via Queue."""

    uid: str
    temp_file_path: str
    params: dict[str, Any]
    job_kind: Literal["transcribe", "align", "diarize"] = field(default="transcribe")
    requested_model_spec_alias: str | None = field(default=None)
    requested_aligner_alias: str | None = field(default=None)
    requested_diarizer_alias: str | None = field(default=None)


def create_engine(engine_type: str, model_id: str) -> ASREngine:
    """Thin wrapper around the factory; imported lazily to keep this module
    importable in the main process without triggering heavy ML framework loads."""
    from src.core.factory import _create_by_type  # noqa: PLC0415

    return _create_by_type(engine_type, model_id)


def create_diarizer(alias: str) -> DiarizationPort:
    """Construct a diarizer from the registry alias."""
    from src.core.diarization_registry import lookup_diarizer  # noqa: PLC0415
    from src.core.mlx_sortformer_diarizer import MlxSortformerDiarizer  # noqa: PLC0415

    spec = lookup_diarizer(alias)
    if spec.runtime == "mlx":
        return MlxSortformerDiarizer(model_id=spec.model_id)
    raise ValueError(f"Unsupported diarization runtime: {spec.runtime}")


def create_aligner(alias: str) -> AlignmentPort:
    """Construct an aligner from the registry alias."""
    from src.core.alignment_registry import lookup_aligner  # noqa: PLC0415
    from src.core.mlx_qwen_forced_aligner import MlxQwenForcedAligner  # noqa: PLC0415

    spec = lookup_aligner(alias)
    if spec.runtime == "mlx":
        return MlxQwenForcedAligner(model_id=spec.model_id)
    raise ValueError(f"Unsupported alignment runtime: {spec.runtime}")


def _put_result(q: Queue, item: Any) -> None:
    """Send a worker IPC message using the public Queue API."""
    q.put(item)


def _release_diarizers(diarizers: dict[str, DiarizationPort]) -> None:
    for alias, diarizer in diarizers.items():
        try:
            diarizer.release()
        except Exception:
            logger.warning("diarizer.release() failed during worker cleanup for %s", alias, exc_info=True)


def _release_aligners(aligners: dict[str, AlignmentPort]) -> None:
    for alias, aligner in aligners.items():
        try:
            aligner.release()
        except Exception:
            logger.warning("aligner.release() failed during worker cleanup for %s", alias, exc_info=True)


def _get_or_create_diarizer(
    diarizers: dict[str, DiarizationPort],
    alias: str,
) -> DiarizationPort:
    diarizer = diarizers.get(alias)
    if diarizer is None:
        diarizer = create_diarizer(alias)
        diarizer.load()
        diarizers[alias] = diarizer
    return diarizer


def _get_or_create_aligner(
    aligners: dict[str, AlignmentPort],
    alias: str,
) -> AlignmentPort:
    aligner = aligners.get(alias)
    if aligner is None:
        aligner = create_aligner(alias)
        aligner.load()
        aligners[alias] = aligner
    return aligner


def run_worker(
    job_queue: Queue[WorkerJob | None],
    result_queue: Queue[tuple[str, Any]],
    engine_type: str,
    model_id: str,
    idle_timeout: float,
) -> None:
    """Main loop executed inside the worker subprocess.

    Args:
        job_queue:    Receives WorkerJob instances or None (shutdown sentinel).
        result_queue: Sends IPC tuples back to the parent process.
        engine_type:  Engine backend identifier ("mlx" | "funasr").
        model_id:     Model identifier string passed to the engine.
        idle_timeout: Seconds to wait for a job before triggering IDLE_EXIT.
                      0 means block indefinitely (no idle timeout).
    """
    logging.basicConfig(
        level="INFO",
        format="[worker] %(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        engine = create_engine(engine_type, model_id)
        engine.load()
    except Exception as exc:
        _put_result(result_queue, ("LOAD_ERROR", str(exc)))
        sys.exit(1)

    _put_result(result_queue, ("READY", None))

    get_timeout: float | None = idle_timeout if idle_timeout > 0 else None
    aligners: dict[str, AlignmentPort] = {}
    diarizers: dict[str, DiarizationPort] = {}

    try:
        try:
            while True:
                try:
                    job: WorkerJob | None = job_queue.get(timeout=get_timeout)
                except queue.Empty:
                    try:
                        engine.release()
                    except Exception:
                        logger.warning("engine.release() failed during idle timeout — proceeding with IDLE_EXIT", exc_info=True)
                    _put_result(result_queue, ("IDLE_EXIT", None))
                    sys.exit(0)

                if job is None:
                    logger.info("Received shutdown sentinel — releasing model and exiting")
                    try:
                        engine.release()
                    except Exception:
                        logger.warning("engine.release() failed during shutdown", exc_info=True)
                    sys.exit(0)

                try:
                    if job.job_kind == "diarize":
                        alias = job.requested_diarizer_alias
                        if not alias:
                            raise ValueError("Diarization job requires requested_diarizer_alias")
                        diarizer = _get_or_create_diarizer(diarizers, alias)
                        result = diarizer.diarize_file(job.temp_file_path)
                    elif job.job_kind == "align":
                        alias = job.requested_aligner_alias
                        if not alias:
                            raise ValueError("Alignment job requires requested_aligner_alias")
                        text = job.params.get("text")
                        if not isinstance(text, str) or not text.strip():
                            raise ValueError("Alignment job requires non-empty text")
                        language = job.params.get("language", "English")
                        if not isinstance(language, str):
                            raise ValueError("Alignment job language must be a string")
                        aligner = _get_or_create_aligner(aligners, alias)
                        result = aligner.align_file(
                            job.temp_file_path,
                            text=text,
                            language=language,
                        )
                    elif job.job_kind == "transcribe":
                        result = engine.transcribe_file(
                            job.temp_file_path,
                            language=job.params.get("language", "auto"),
                            output_format=job.params.get("output_format", "txt"),
                            with_timestamp=job.params.get("with_timestamp", False),
                            use_itn=job.params.get("use_itn", True),
                        )
                    else:
                        raise ValueError(f"Unsupported job_kind: {job.job_kind}")
                    _put_result(result_queue, ("RESULT", job.uid, result))
                except Exception as exc:
                    logger.exception("%s failed for job %s", job.job_kind.title(), job.uid)
                    _put_result(result_queue, ("ERROR", job.uid, type(exc).__name__, str(exc)))
        finally:
            _release_aligners(aligners)
            _release_diarizers(diarizers)
    finally:
        aligners.clear()
        diarizers.clear()
