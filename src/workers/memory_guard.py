"""Observed MLX allocation guard shared by isolated model workers."""

import json
import os
import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol


class MemoryRuntime(Protocol):
    def get_active_memory(self) -> int: ...
    def get_cache_memory(self) -> int: ...
    def get_peak_memory(self) -> int: ...


@contextmanager
def allocation_guard(runtime: MemoryRuntime, path: Path, budget: int) -> Iterator[None]:
    """Stop this owned child on observed MLX pressure; limits alone are advisory.

    Sampling can overshoot between checks. The parent independently preserves
    system memory and deadlines; this thread never targets another process.
    """
    stopped = threading.Event()

    def monitor() -> None:
        peak_seen = 0
        while not stopped.is_set():
            try:
                active = runtime.get_active_memory()
                cache = runtime.get_cache_memory()
                peak_seen = max(peak_seen, runtime.get_peak_memory(), active + cache)
                record = {
                    "active_bytes": active,
                    "cache_bytes": cache,
                    "peak_observed_bytes": peak_seen,
                    "budget_bytes": budget,
                    "stop_reason": None
                    if peak_seen <= budget
                    else "MLX allocation exceeded owned budget",
                }
                path.write_text(json.dumps(record, indent=2))
                if peak_seen > budget:
                    os.kill(os.getpid(), signal.SIGTERM)
                    return
            except Exception as exc:
                # A broken monitor must fail closed rather than run unprotected.
                try:
                    path.write_text(json.dumps({"stop_reason": f"MLX monitor failed: {exc}"}))
                finally:
                    os.kill(os.getpid(), signal.SIGTERM)
                return
            stopped.wait(0.05)

    thread = threading.Thread(target=monitor, name="poc-mlx-memory", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=2)
