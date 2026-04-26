"""ModelWorker subprocess entry point for memory-isolated ASR inference.

Runs inside a child process. Communicates with the parent via multiprocessing
Queues using a simple IPC protocol:

  ("READY", None)          — engine loaded, ready for jobs
  ("LOAD_ERROR", str)      — engine.load() failed; process exits with code 1
  ("RESULT", uid, result)  — transcription or diarization succeeded
  ("ERROR", uid, str)      — job execution raised an exception
  ("IDLE_EXIT", None)      — idle timeout reached; process exits with code 0
"""

import io
import logging
import pickle
import queue
import sys
from dataclasses import dataclass, field
from multiprocessing import Queue
from multiprocessing.reduction import ForkingPickler
from typing import Any, Literal, cast

from src.core.base_engine import ASREngine
from src.core.diarization_port import DiarizationPort
from src.core.factory import EngineInstance

logger = logging.getLogger(__name__)

JobKind = Literal["transcribe", "diarize"]


@dataclass
class WorkerJob:
    """Picklable job descriptor passed from parent → worker via Queue."""

    uid: str
    temp_file_path: str
    params: dict[str, Any]
    requested_model_spec_alias: str | None = field(default=None)
    job_kind: JobKind = field(default="transcribe")


def create_engine(engine_type: str, model_id: str) -> EngineInstance:
    """Thin wrapper around the factory; imported lazily to keep this module
    importable in the main process without triggering heavy ML framework loads."""
    from src.core.factory import _create_by_type  # noqa: PLC0415

    return _create_by_type(engine_type, model_id)


def _sync_put(q: Queue, item: Any) -> None:
    """Write directly to the queue's OS pipe, bypassing the async feeder thread.

    multiprocessing.Queue.put() buffers items in a deque and drains them via a
    background thread. In same-process tests, get_nowait() may be called before
    the feeder thread has had a chance to flush. Writing straight to the
    underlying Connection ensures data is available immediately.
    """
    buf = io.BytesIO()
    ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(item)
    # NOTE: Uses CPython internals (_writer) to bypass the feeder thread.
    # This is only necessary in same-process unit tests where get_nowait()
    # is called immediately after put(). In production (cross-process), the
    # parent's blocking queue.get() has ample time for the feeder thread to flush.
    q._writer.send_bytes(buf.getvalue())  # type: ignore[attr-defined]


def _run_job(engine: EngineInstance, job: WorkerJob) -> object:
    if job.job_kind == "diarize":
        diarization_engine = cast(DiarizationPort, engine)
        return diarization_engine.diarize_file(job.temp_file_path)

    transcription_engine = cast(ASREngine, engine)
    return transcription_engine.transcribe_file(
        job.temp_file_path,
        language=job.params.get("language", "auto"),
        output_format=job.params.get("output_format", "txt"),
        with_timestamp=job.params.get("with_timestamp", False),
        use_itn=job.params.get("use_itn", True),
    )


def run_worker(
    job_queue: "Queue[WorkerJob | None]",
    result_queue: "Queue[tuple[str, Any]]",
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
        _sync_put(result_queue, ("LOAD_ERROR", str(exc)))
        sys.exit(1)

    _sync_put(result_queue, ("READY", None))

    get_timeout: float | None = idle_timeout if idle_timeout > 0 else None

    while True:
        try:
            job: WorkerJob | None = job_queue.get(timeout=get_timeout)
        except queue.Empty:
            try:
                engine.release()
            except Exception:
                logger.warning("engine.release() failed during idle timeout — proceeding with IDLE_EXIT", exc_info=True)
            _sync_put(result_queue, ("IDLE_EXIT", None))
            sys.exit(0)

        if job is None:
            logger.info("Received shutdown sentinel — releasing model and exiting")
            try:
                engine.release()
            except Exception:
                logger.warning("engine.release() failed during shutdown", exc_info=True)
            sys.exit(0)

        try:
            result = _run_job(engine, job)
            _sync_put(result_queue, ("RESULT", job.uid, result))
        except Exception as exc:
            logger.exception("Job failed for %s (%s)", job.uid, job.job_kind)
            _sync_put(result_queue, ("ERROR", job.uid, str(exc)))
