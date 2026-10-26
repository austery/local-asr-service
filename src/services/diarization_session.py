"""Own one standalone model process, including cancellation-safe spawn and reap."""

import asyncio
import json
import os
import signal
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Protocol

import psutil

from src.core.diarization_result import DiarizationError, DiarizationPayload, parse_payload
from src.services.worker_session import finish_cleanup

MAX_DIAGNOSTIC_BYTES = 4096


class DiarizationSessionPort(Protocol):
    async def run(self, audio: Path, output: Path, checkpoint: Path) -> DiarizationPayload: ...
    async def close(self) -> None: ...


class DiarizationSession:
    def __init__(
        self,
        command: tuple[str, ...] | None = None,
        *,
        timeout: float = 900,
        reserve_bytes: int = 20 * 1024**3,
    ) -> None:
        self._command = command or (sys.executable, "-m", "src.workers.diarization_worker")
        self._timeout = timeout
        self._reserve = reserve_bytes
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    async def run(self, audio: Path, output: Path, checkpoint: Path) -> DiarizationPayload:
        started = time.monotonic()
        try:
            with output.with_suffix(".log").open("wb") as log:
                async with self._lock:
                    if self._closed:
                        raise DiarizationError("worker_closed", "Diarization session is closed")
                    self._check_memory()
                    env = os.environ | {
                        "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                        "HF_DATASETS_OFFLINE": "1",
                    }

                    async def spawn() -> None:
                        self._process = await asyncio.create_subprocess_exec(
                            *self._command,
                            str(audio),
                            str(output),
                            str(checkpoint),
                            stdout=log,
                            stderr=log,
                            env=env,
                            start_new_session=True,
                        )

                    # Cancellation during spawn retains the child before disposal.
                    try:
                        await finish_cleanup(spawn())
                    except OSError as exc:
                        raise DiarizationError(
                            "worker_failed", "Unable to start isolated diarization worker"
                        ) from exc
                process = self._process
                if process is None:
                    raise DiarizationError("worker_failed", "Worker did not start")
                await self._wait_for_worker(process, started)
                return self._read_result(output, process.returncode)
        finally:
            await finish_cleanup(self.close())

    async def _wait_for_worker(self, process: asyncio.subprocess.Process, started: float) -> None:
        while process.returncode is None:
            self._check_memory()
            if time.monotonic() - started >= self._timeout:
                raise DiarizationError("timeout", "Diarization deadline exceeded", 504)
            try:
                owned = psutil.Process(process.pid)
                rss = owned.memory_info().rss + sum(
                    p.memory_info().rss for p in owned.children(recursive=True)
                )
                if rss > 8 * 1024**3:
                    raise DiarizationError(
                        "resource_limit", "Owned worker memory budget exceeded", 503
                    )
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                pass
            await asyncio.sleep(0.05)
        await process.wait()

    @staticmethod
    def _read_result(output: Path, returncode: int | None) -> DiarizationPayload:
        if not output.is_file() or output.stat().st_size > 16 * 1024**2:
            memory = output.with_suffix(".memory.json")
            try:
                diagnostic = (
                    json.loads(memory.read_text())
                    if memory.is_file() and memory.stat().st_size <= MAX_DIAGNOSTIC_BYTES
                    else {}
                )
            except (OSError, ValueError):
                diagnostic = {}
            if isinstance(diagnostic, dict) and diagnostic.get("stop_reason"):
                raise DiarizationError(
                    "resource_limit", "MLX allocation monitor stopped the worker", 503
                )
            raise DiarizationError("worker_failed", "Worker exited without a bounded result")
        try:
            data = json.loads(output.read_text())
        except (ValueError, OSError) as exc:
            raise DiarizationError("invalid_output", "Worker returned invalid JSON") from exc
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            error = data["error"]
            code, message, status = (
                error.get("code"),
                error.get("message"),
                error.get("status"),
            )
            if (
                isinstance(code, str)
                and isinstance(message, str)
                and status in (400, 502, 503, 504)
            ):
                raise DiarizationError(code, message, int(status))
        if returncode:
            raise DiarizationError("worker_failed", "Diarization worker failed")
        return parse_payload(data)

    def _check_memory(self) -> None:
        if psutil.virtual_memory().available < self._reserve:
            raise DiarizationError("resource_limit", "Less than 20 GiB available; retry later", 503)

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            process = self._process
            if process is None:
                return
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), timeout=3)
            except TimeoutError:
                os.killpg(process.pid, signal.SIGKILL)
                await asyncio.wait_for(process.wait(), timeout=3)
            # Dispose decode descendants even if the worker exited first.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            deadline = time.monotonic() + 3
            while True:
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    self._process = None
                    return
                if time.monotonic() >= deadline:
                    raise DiarizationError("cleanup_failed", "Owned process group still exists")
                await asyncio.sleep(0.05)
