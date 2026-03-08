"""ModelWorker subprocess entry point for memory-isolated ASR inference.

Runs inside a child process. Communicates with the parent via multiprocessing
Queues using a simple IPC protocol:

  ("READY", None)          — engine loaded, ready for jobs
  ("LOAD_ERROR", str)      — engine.load() failed; process exits with code 1
  ("RESULT", uid, result)  — transcription succeeded
  ("ERROR", uid, str)      — transcription raised an exception
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
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WorkerJob:
    """Picklable job descriptor passed from parent → worker via Queue."""

    uid: str
    temp_file_path: str
    params: dict[str, Any]
    requested_model_spec_alias: str | None = field(default=None)


def create_engine(engine_type: str, model_id: str) -> Any:
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


def run_worker(
    job_queue: Any,
    result_queue: Any,
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

    engine = create_engine(engine_type, model_id)

    try:
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
            engine.release()
            _sync_put(result_queue, ("IDLE_EXIT", None))
            sys.exit(0)

        if job is None:
            logger.info("Received shutdown sentinel — releasing model and exiting")
            try:
                engine.release()
            except Exception:
                pass
            sys.exit(0)

        try:
            result = engine.transcribe_file(
                job.temp_file_path,
                language=job.params.get("language", "auto"),
                output_format=job.params.get("output_format", "txt"),
                with_timestamp=job.params.get("with_timestamp", False),
                use_itn=True,
            )
            _sync_put(result_queue, ("RESULT", job.uid, result))
        except Exception as exc:
            logger.exception("Transcription failed for job %s", job.uid)
            _sync_put(result_queue, ("ERROR", job.uid, str(exc)))

