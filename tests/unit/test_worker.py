"""Unit tests for model_worker subprocess entry point."""

import multiprocessing
import queue
from unittest.mock import MagicMock, patch

import pytest

from src.core.diarization_port import SpeakerTurn
from src.workers.model_worker import WorkerJob, run_worker


def _make_job(
    uid: str = "job-1",
    path: str = "/tmp/test.wav",
    params: dict[str, object] | None = None,
    job_kind: str = "transcribe",
) -> WorkerJob:
    return WorkerJob(
        uid=uid,
        temp_file_path=path,
        params=params or {"language": "auto", "output_format": "txt", "with_timestamp": False},
        job_kind=job_kind,
    )


class TestRunWorker:
    def test_sends_ready_after_load(self):
        """Worker must put READY on result_queue after engine.load() succeeds."""
        job_q = multiprocessing.Queue()
        result_q = multiprocessing.Queue()
        job_q.put(None)  # immediate shutdown after READY

        mock_engine = MagicMock()

        with patch("src.workers.model_worker.create_engine", return_value=mock_engine):
            with pytest.raises(SystemExit):
                run_worker(job_q, result_q, engine_type="mlx", model_id="test-model", idle_timeout=0)

        msg = result_q.get_nowait()
        assert msg == ("READY", None)
        mock_engine.load.assert_called_once()

    def test_sends_load_error_if_load_fails(self):
        """Worker must put LOAD_ERROR and exit if model load raises."""
        job_q = multiprocessing.Queue()
        result_q = multiprocessing.Queue()

        mock_engine = MagicMock()
        mock_engine.load.side_effect = RuntimeError("out of memory")

        with patch("src.workers.model_worker.create_engine", return_value=mock_engine):
            with pytest.raises(SystemExit):
                run_worker(job_q, result_q, engine_type="mlx", model_id="test-model", idle_timeout=0)

        msg = result_q.get_nowait()
        assert msg[0] == "LOAD_ERROR"
        assert "out of memory" in msg[1]

    def test_transcribes_job_and_puts_result(self):
        """Worker processes transcription jobs and puts (RESULT, uid, output) on result_queue."""
        job_q = multiprocessing.Queue()
        result_q = multiprocessing.Queue()

        job = _make_job(uid="abc-123")
        job_q.put(job)
        job_q.put(None)  # shutdown

        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = {"text": "hello world", "segments": None}

        with patch("src.workers.model_worker.create_engine", return_value=mock_engine):
            with pytest.raises(SystemExit):
                run_worker(job_q, result_q, engine_type="funasr", model_id="para", idle_timeout=0)

        ready = result_q.get_nowait()
        assert ready == ("READY", None)
        result_msg = result_q.get_nowait()
        assert result_msg[0] == "RESULT"
        assert result_msg[1] == "abc-123"
        mock_engine.transcribe_file.assert_called_once_with(
            "/tmp/test.wav",
            language="auto",
            output_format="txt",
            with_timestamp=False,
            use_itn=True,
        )
        mock_engine.diarize_file.assert_not_called()

    def test_diarizes_job_and_puts_result(self):
        """Worker routes diarization jobs to diarize_file and still emits RESULT tuples."""
        job_q = multiprocessing.Queue()
        result_q = multiprocessing.Queue()

        job = _make_job(uid="diar-1", job_kind="diarize")
        job_q.put(job)
        job_q.put(None)

        mock_engine = MagicMock()
        diarization_result = [SpeakerTurn(speaker="speaker_0", start=0.0, end=1.25)]
        mock_engine.diarize_file.return_value = diarization_result

        with patch("src.workers.model_worker.create_engine", return_value=mock_engine):
            with pytest.raises(SystemExit):
                run_worker(job_q, result_q, engine_type="mlx", model_id="sortformer", idle_timeout=0)

        ready = result_q.get_nowait()
        assert ready == ("READY", None)
        result_msg = result_q.get_nowait()
        assert result_msg == ("RESULT", "diar-1", diarization_result)
        mock_engine.diarize_file.assert_called_once_with("/tmp/test.wav")
        mock_engine.transcribe_file.assert_not_called()

    def test_puts_error_on_transcription_failure(self):
        """Worker puts (ERROR, uid, msg) when transcribe_file raises."""
        job_q = multiprocessing.Queue()
        result_q = multiprocessing.Queue()

        job = _make_job(uid="fail-1")
        job_q.put(job)
        job_q.put(None)

        mock_engine = MagicMock()
        mock_engine.transcribe_file.side_effect = RuntimeError("GPU crash")

        with patch("src.workers.model_worker.create_engine", return_value=mock_engine):
            with pytest.raises(SystemExit):
                run_worker(job_q, result_q, engine_type="funasr", model_id="para", idle_timeout=0)

        result_q.get_nowait()  # READY
        err_msg = result_q.get_nowait()
        assert err_msg[0] == "ERROR"
        assert err_msg[1] == "fail-1"
        assert "GPU crash" in err_msg[2]

    def test_idle_exit_when_timeout_reached(self):
        """Worker puts IDLE_EXIT and releases engine when queue.get() times out."""
        job_q = multiprocessing.Queue()
        result_q = multiprocessing.Queue()
        # No jobs — let idle timeout fire immediately (timeout=0.01s)

        mock_engine = MagicMock()

        with patch("src.workers.model_worker.create_engine", return_value=mock_engine):
            with pytest.raises(SystemExit):
                run_worker(job_q, result_q, engine_type="mlx", model_id="test", idle_timeout=0.01)

        ready = result_q.get_nowait()
        assert ready == ("READY", None)
        idle_exit = result_q.get_nowait()
        assert idle_exit == ("IDLE_EXIT", None)
        mock_engine.release.assert_called_once()
