"""Unit tests for model_worker subprocess entry point."""
import multiprocessing
from unittest.mock import MagicMock, patch

import pytest

from src.workers.model_worker import WorkerJob, run_worker


def _get_result(result_q):
    return result_q.get(timeout=1.0)


def _make_job(uid="job-1", path="/tmp/test.wav", params=None):
    return WorkerJob(
        uid=uid,
        temp_file_path=path,
        params=params or {"language": "auto", "output_format": "txt", "with_timestamp": False},
    )


def _make_diarize_job(
    uid="diarize-1",
    path="/tmp/test.wav",
    requested_diarizer_alias="sortformer-diar",
):
    return WorkerJob(
        uid=uid,
        temp_file_path=path,
        params={},
        job_kind="diarize",
        requested_diarizer_alias=requested_diarizer_alias,
    )


def _make_align_job(
    uid="align-1",
    path="/tmp/test.wav",
    requested_aligner_alias="qwen3-forced-aligner",
):
    return WorkerJob(
        uid=uid,
        temp_file_path=path,
        params={"text": "hello world", "language": "English"},
        job_kind="align",
        requested_aligner_alias=requested_aligner_alias,
    )


def _make_custom_kind_job(
    uid="custom-kind-1",
    path="/tmp/test.wav",
    job_kind="summarize",
):
    return WorkerJob(
        uid=uid,
        temp_file_path=path,
        params={},
        job_kind=job_kind,  # type: ignore[arg-type]
    )


class TestRunWorker:
    def test_sends_ready_after_load(self):
        """Worker must put READY on result_queue after engine.load() succeeds."""
        job_q = multiprocessing.Queue()
        result_q = multiprocessing.Queue()
        job_q.put(None)  # immediate shutdown after READY

        mock_engine = MagicMock()

        with (
            patch("src.workers.model_worker.create_engine", return_value=mock_engine),
            pytest.raises(SystemExit),
        ):
            run_worker(job_q, result_q, engine_type="mlx", model_id="test-model", idle_timeout=0)

        msg = _get_result(result_q)
        assert msg == ("READY", None)
        mock_engine.load.assert_called_once()

    def test_sends_load_error_if_load_fails(self):
        """Worker must put LOAD_ERROR and exit if model load raises."""
        job_q = multiprocessing.Queue()
        result_q = multiprocessing.Queue()

        mock_engine = MagicMock()
        mock_engine.load.side_effect = RuntimeError("out of memory")

        with (
            patch("src.workers.model_worker.create_engine", return_value=mock_engine),
            pytest.raises(SystemExit),
        ):
            run_worker(job_q, result_q, engine_type="mlx", model_id="test-model", idle_timeout=0)

        msg = _get_result(result_q)
        assert msg[0] == "LOAD_ERROR"
        assert "out of memory" in msg[1]

    def test_transcribes_job_and_puts_result(self):
        """Worker processes a job and puts (RESULT, uid, output) on result_queue."""
        job_q = multiprocessing.Queue()
        result_q = multiprocessing.Queue()

        job = _make_job(uid="abc-123")
        job_q.put(job)
        job_q.put(None)  # shutdown

        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = {"text": "hello world", "segments": None}

        with (
            patch("src.workers.model_worker.create_engine", return_value=mock_engine),
            pytest.raises(SystemExit),
        ):
            run_worker(job_q, result_q, engine_type="funasr", model_id="para", idle_timeout=0)

        ready = _get_result(result_q)
        assert ready == ("READY", None)
        result_msg = _get_result(result_q)
        assert result_msg[0] == "RESULT"
        assert result_msg[1] == "abc-123"

    def test_puts_error_on_transcription_failure(self):
        """Worker puts (ERROR, uid, msg) when transcribe_file raises."""
        job_q = multiprocessing.Queue()
        result_q = multiprocessing.Queue()

        job = _make_job(uid="fail-1")
        job_q.put(job)
        job_q.put(None)

        mock_engine = MagicMock()
        mock_engine.transcribe_file.side_effect = RuntimeError("GPU crash")

        with (
            patch("src.workers.model_worker.create_engine", return_value=mock_engine),
            pytest.raises(SystemExit),
        ):
            run_worker(job_q, result_q, engine_type="funasr", model_id="para", idle_timeout=0)

        _get_result(result_q)  # READY
        err_msg = _get_result(result_q)
        assert err_msg[0] == "ERROR"
        assert err_msg[1] == "fail-1"
        assert err_msg[2] == "RuntimeError"
        assert "GPU crash" in err_msg[3]

    def test_idle_exit_when_timeout_reached(self):
        """Worker puts IDLE_EXIT and releases engine when queue.get() times out."""
        job_q = multiprocessing.Queue()
        result_q = multiprocessing.Queue()
        # No jobs — let idle timeout fire immediately (timeout=0.01s)

        mock_engine = MagicMock()

        with (
            patch("src.workers.model_worker.create_engine", return_value=mock_engine),
            pytest.raises(SystemExit),
        ):
            run_worker(job_q, result_q, engine_type="mlx", model_id="test", idle_timeout=0.01)

        ready = _get_result(result_q)
        assert ready == ("READY", None)
        idle_exit = _get_result(result_q)
        assert idle_exit == ("IDLE_EXIT", None)
        mock_engine.release.assert_called_once()

    def test_diarizes_job_and_puts_result(self):
        """Worker dispatches diarize jobs through the dedicated diarizer runtime."""
        job_q = multiprocessing.Queue()
        result_q = multiprocessing.Queue()

        job = _make_diarize_job(uid="diarize-ok")
        job_q.put(job)
        job_q.put(None)

        mock_engine = MagicMock()
        mock_diarizer = MagicMock()
        mock_diarizer.diarize_file.return_value = [
            {"speaker": "Speaker 1", "start": 0.0, "end": 1.0},
        ]

        with (
            patch("src.workers.model_worker.create_engine", return_value=mock_engine),
            patch("src.workers.model_worker.create_diarizer", return_value=mock_diarizer),
            pytest.raises(SystemExit),
        ):
            run_worker(job_q, result_q, engine_type="mlx", model_id="test-model", idle_timeout=0)

        ready = _get_result(result_q)
        assert ready == ("READY", None)
        result_msg = _get_result(result_q)
        assert result_msg == (
            "RESULT",
            "diarize-ok",
            [{"speaker": "Speaker 1", "start": 0.0, "end": 1.0}],
        )
        mock_diarizer.diarize_file.assert_called_once_with("/tmp/test.wav")
        mock_diarizer.release.assert_called_once()

    def test_aligns_job_and_puts_result(self):
        """Worker dispatches align jobs through the dedicated forced aligner runtime."""
        job_q = multiprocessing.Queue()
        result_q = multiprocessing.Queue()

        job = _make_align_job(uid="align-ok")
        job_q.put(job)
        job_q.put(None)

        mock_engine = MagicMock()
        mock_aligner = MagicMock()
        mock_aligner.align_file.return_value = [
            {"text": "hello", "start": 0.0, "end": 0.5},
            {"text": "world", "start": 0.5, "end": 1.0},
        ]

        with (
            patch("src.workers.model_worker.create_engine", return_value=mock_engine),
            patch("src.workers.model_worker.create_aligner", return_value=mock_aligner),
            pytest.raises(SystemExit),
        ):
            run_worker(job_q, result_q, engine_type="mlx", model_id="test-model", idle_timeout=0)

        ready = _get_result(result_q)
        assert ready == ("READY", None)
        result_msg = _get_result(result_q)
        assert result_msg == (
            "RESULT",
            "align-ok",
            [
                {"text": "hello", "start": 0.0, "end": 0.5},
                {"text": "world", "start": 0.5, "end": 1.0},
            ],
        )
        mock_aligner.align_file.assert_called_once_with(
            "/tmp/test.wav",
            text="hello world",
            language="English",
        )
        mock_aligner.release.assert_called_once()

    def test_puts_error_on_diarization_failure(self):
        """Worker puts ERROR when the dedicated diarizer raises."""
        job_q = multiprocessing.Queue()
        result_q = multiprocessing.Queue()

        job = _make_diarize_job(uid="diarize-fail")
        job_q.put(job)
        job_q.put(None)

        mock_engine = MagicMock()
        mock_diarizer = MagicMock()
        mock_diarizer.diarize_file.side_effect = RuntimeError("diarizer crashed")

        with (
            patch("src.workers.model_worker.create_engine", return_value=mock_engine),
            patch("src.workers.model_worker.create_diarizer", return_value=mock_diarizer),
            pytest.raises(SystemExit),
        ):
            run_worker(job_q, result_q, engine_type="mlx", model_id="test-model", idle_timeout=0)

        _get_result(result_q)  # READY
        err_msg = _get_result(result_q)
        assert err_msg[0] == "ERROR"
        assert err_msg[1] == "diarize-fail"
        assert err_msg[2] == "RuntimeError"
        assert "diarizer crashed" in err_msg[3]

    def test_puts_error_on_unsupported_job_kind(self):
        """Worker rejects unknown job kinds instead of falling through to transcription."""
        job_q = multiprocessing.Queue()
        result_q = multiprocessing.Queue()

        job = _make_custom_kind_job(uid="bad-kind")
        job_q.put(job)
        job_q.put(None)

        mock_engine = MagicMock()

        with (
            patch("src.workers.model_worker.create_engine", return_value=mock_engine),
            pytest.raises(SystemExit),
        ):
            run_worker(job_q, result_q, engine_type="mlx", model_id="test-model", idle_timeout=0)

        _get_result(result_q)  # READY
        err_msg = _get_result(result_q)
        assert err_msg[0] == "ERROR"
        assert err_msg[1] == "bad-kind"
        assert err_msg[2] == "ValueError"
        assert "Unsupported job_kind" in err_msg[3]
        mock_engine.transcribe_file.assert_not_called()

    def test_puts_error_when_diarize_job_is_missing_alias(self):
        """Worker rejects diarize jobs without a requested diarizer alias."""
        job_q = multiprocessing.Queue()
        result_q = multiprocessing.Queue()

        job = _make_diarize_job(uid="missing-alias", requested_diarizer_alias=None)
        job_q.put(job)
        job_q.put(None)

        mock_engine = MagicMock()

        with (
            patch("src.workers.model_worker.create_engine", return_value=mock_engine),
            pytest.raises(SystemExit),
        ):
            run_worker(job_q, result_q, engine_type="mlx", model_id="test-model", idle_timeout=0)

        _get_result(result_q)  # READY
        err_msg = _get_result(result_q)
        assert err_msg[0] == "ERROR"
        assert err_msg[1] == "missing-alias"
        assert err_msg[2] == "ValueError"
        assert "requested_diarizer_alias" in err_msg[3]
