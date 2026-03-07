import asyncio
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool

from src.config import MODEL_IDLE_TIMEOUT_SEC
from src.core.base_engine import ASREngine
from src.core.factory import create_engine_for_spec
from src.core.model_registry import ModelSpec


@dataclass
class TranscriptionJob:
    uid: str
    temp_dir: str
    temp_file_path: str
    params: dict[str, Any]
    future: asyncio.Future  # type: ignore[type-arg]
    received_at: float
    # None = use whatever model is currently loaded (no switch needed)
    requested_model_spec: ModelSpec | None = field(default=None)


class TranscriptionService:
    """
    转录服务调度器。
    职责：
    1. 管理异步队列 (Async Queue)
    2. 协调 Engine 进行串行推理
    3. 在两次推理之间按需热换模型
    4. 管理临时文件的生命周期
    5. 空闲超时自动卸载模型 (SPEC-009)
    """

    def __init__(
        self,
        engine: ASREngine,
        max_queue_size: int = 50,
        initial_model_spec: ModelSpec | None = None,
        idle_timeout: int = MODEL_IDLE_TIMEOUT_SEC,
    ):
        self.engine = engine
        self._current_model_spec = initial_model_spec
        self._engine_degraded = False
        self._model_loaded = True  # Engine is loaded at startup by main.py
        self._idle_timeout = idle_timeout
        self.logger = logging.getLogger(__name__)
        self.queue: asyncio.Queue[TranscriptionJob | None] = asyncio.Queue(maxsize=max_queue_size)
        self.is_running = False
        self.logger.info(f"🚦 Service initialized. Queue size: {max_queue_size}")
        if idle_timeout > 0:
            self.logger.info(f"💤 Idle offload enabled: model will be released after {idle_timeout}s of inactivity")
        else:
            self.logger.info("💤 Idle offload disabled (MODEL_IDLE_TIMEOUT_SEC=0)")

    @property
    def current_model_spec(self) -> ModelSpec | None:
        """The ModelSpec of the currently loaded engine (None if not tracked)."""
        return self._current_model_spec

    @property
    def model_loaded(self) -> bool:
        """Whether the engine model is currently loaded in memory."""
        return self._model_loaded

    async def start_worker(self) -> None:
        """启动后台消费者循环 (在 main.py 的 lifespan 中调用)"""
        self.is_running = True
        asyncio.create_task(self._consume_loop())
        self.logger.info("👷 Background worker started.")

    async def submit(
        self,
        file: UploadFile,
        params: dict[str, Any],
        request_id: str = "unknown",
        model_spec: ModelSpec | None = None,
    ) -> str | dict[str, Any]:
        """
        提交任务接口 (供 API 层调用)。
        model_spec=None 表示使用当前已加载的模型（不换模）。
        """
        if self.queue.full():
            self.logger.warning(f"[{request_id}] Queue full, rejecting request")
            raise RuntimeError("Service busy: Queue is full.")

        temp_dir = tempfile.mkdtemp(prefix="asr_task_")

        try:
            file_ext = os.path.splitext(file.filename or "upload.wav")[1] or ".wav"
            temp_path = os.path.join(temp_dir, f"original{file_ext}")

            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            loop = asyncio.get_running_loop()
            future: asyncio.Future[str | dict[str, Any]] = loop.create_future()

            job = TranscriptionJob(
                uid=request_id,
                temp_dir=temp_dir,
                temp_file_path=temp_path,
                params=params,
                future=future,
                received_at=time.time(),
                requested_model_spec=model_spec,
            )

            await self.queue.put(job)
            result: str | dict[str, Any] = await future
            return result

        except Exception as e:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise e

    async def stop_worker(self) -> None:
        """优雅停止消费者循环"""
        self.is_running = False
        await self.queue.put(None)  # type: ignore[arg-type]

    async def _switch_model(self, new_spec: ModelSpec, job_uid: str) -> None:
        """
        热换模型：release 旧引擎 → load 新引擎 → 原子替换。

        Per SPEC-108 §5: release 发生在 load 之前，确保 M-series 上不会双倍内存峰值。
        如果 load 失败，尝试恢复原引擎；恢复失败时记录降级状态并继续。
        """
        old_engine = self.engine
        old_spec = self._current_model_spec
        old_alias = old_spec.alias if old_spec else "unknown"
        new_alias = new_spec.alias

        self.logger.info(f"[{job_uid}] 🔄 Switching model: {old_alias} → {new_alias}")
        switch_start = time.time()

        # Step 1: Release old engine (blocking → thread pool).
        # Per SPEC-108 §5: release MUST succeed before load to avoid double memory peak on M-series.
        # If release fails, abort the switch immediately — do NOT proceed to load.
        try:
            await run_in_threadpool(old_engine.release)
        except Exception as e:
            self.logger.error(
                f"[{job_uid}] ❌ Old engine release failed: {e}. "
                f"Aborting switch to protect memory budget — engine remains {old_alias}.",
                exc_info=True,
            )
            raise RuntimeError(
                f"Model switch aborted: failed to release '{old_alias}' ({e}). "
                f"The current engine is still loaded and usable."
            ) from e

        # Step 2: Create & load new engine (blocking → thread pool)
        new_engine = create_engine_for_spec(new_spec)
        try:
            await run_in_threadpool(new_engine.load)
        except Exception as load_err:
            self.logger.error(
                f"[{job_uid}] ❌ New engine load failed: {load_err}. "
                f"Attempting to restore previous engine ({old_alias}).",
                exc_info=True,
            )
            try:
                await run_in_threadpool(old_engine.load)
                self.engine = old_engine
                self._current_model_spec = old_spec
                self.logger.info(f"[{job_uid}] ✅ Restored previous engine: {old_alias}")
            except Exception as restore_err:
                # Both load and restore failed: engine is now in an undefined state.
                # Mark degraded so subsequent jobs fail fast with a clear message
                # instead of silently crashing against an unloaded engine.
                self._engine_degraded = True
                self.logger.error(
                    f"[{job_uid}] ❌ FATAL: Both model switch and recovery failed. "
                    f"load_err={load_err!r}, restore_err={restore_err!r}. "
                    f"Service is degraded — manual restart required.",
                    exc_info=True,
                )
                raise RuntimeError(
                    f"Engine unrecoverable: switch to '{new_alias}' failed ({load_err}), "
                    f"restore of '{old_alias}' also failed ({restore_err}). "
                    f"Service must be restarted."
                ) from restore_err
            raise load_err  # Fail this job; engine restored, next job can proceed

        # Step 3: Atomic swap (single-threaded consumer — no lock needed)
        self.engine = new_engine
        self._current_model_spec = new_spec
        self._model_loaded = True  # New engine is freshly loaded

        elapsed = time.time() - switch_start
        self.logger.info(f"[{job_uid}] ✅ Model switch complete: {new_alias} ({elapsed:.2f}s)")

    async def _consume_loop(self) -> None:
        """
        消费者循环 (Strict Serial Execution).
        This is the single thread of control for all inference — no concurrency here.

        SPEC-009: When idle_timeout > 0, the loop uses asyncio.wait_for() to
        detect idle periods. After idle_timeout seconds with no incoming jobs,
        the model is released from memory. The next incoming job triggers a
        reload before inference.
        """
        while self.is_running:
            # === Wait for next job (with optional idle timeout) ===
            try:
                if self._idle_timeout > 0:
                    job = await asyncio.wait_for(
                        self.queue.get(), timeout=self._idle_timeout
                    )
                else:
                    job = await self.queue.get()
            except asyncio.TimeoutError:
                # Idle timeout reached — offload model to free memory (SPEC-009)
                if self._model_loaded:
                    model_alias = (
                        self._current_model_spec.alias
                        if self._current_model_spec
                        else "unknown"
                    )
                    self.logger.info(
                        f"💤 Idle timeout ({self._idle_timeout}s) reached. "
                        f"Offloading model '{model_alias}' to free memory..."
                    )
                    try:
                        await run_in_threadpool(self.engine.release)
                        self._model_loaded = False
                        self.logger.info(
                            f"✅ Model '{model_alias}' offloaded. "
                            f"Will reload on next request."
                        )
                    except Exception as e:
                        self.logger.error(
                            f"❌ Failed to offload model: {e}", exc_info=True
                        )
                continue  # Back to waiting for next job

            if job is None:
                break

            queue_time = time.time() - job.received_at
            self.logger.info(f"[{job.uid}] Starting transcription (queue_time={queue_time:.2f}s)")

            inference_start = time.time()
            try:
                # === Degraded engine guard ===
                if self._engine_degraded:
                    raise RuntimeError(
                        "Service is in a degraded state (engine unrecoverable). "
                        "Manual restart required."
                    )

                # === Model switch check (before inference) ===
                requested = job.requested_model_spec
                if requested is not None and requested != self._current_model_spec:
                    # _switch_model handles its own release/load cycle and sets
                    # _model_loaded = True. If the model was already offloaded,
                    # release() is a no-op (guarded by `if self.model:` in engines).
                    await self._switch_model(requested, job.uid)
                elif not self._model_loaded:
                    # Model was offloaded due to idle timeout — reload it (SPEC-009)
                    model_alias = (
                        self._current_model_spec.alias
                        if self._current_model_spec
                        else "unknown"
                    )
                    self.logger.info(
                        f"[{job.uid}] 🔄 Reloading model '{model_alias}' "
                        f"(was offloaded due to idle timeout)..."
                    )
                    reload_start = time.time()
                    await run_in_threadpool(self.engine.load)
                    self._model_loaded = True
                    reload_time = time.time() - reload_start
                    self.logger.info(
                        f"[{job.uid}] ✅ Model '{model_alias}' reloaded in {reload_time:.2f}s"
                    )

                # === Core inference ===
                output_format = job.params.get("output_format", "txt")
                with_timestamp = job.params.get("with_timestamp", False)

                result_data = await run_in_threadpool(
                    self.engine.transcribe_file,
                    file_path=job.temp_file_path,
                    language=job.params.get("language", "auto"),
                    output_format=output_format,
                    with_timestamp=with_timestamp,
                    use_itn=True,
                )

                inference_time = time.time() - inference_start
                total_time = time.time() - job.received_at

                if isinstance(result_data, dict):
                    result: str | dict[str, Any] = {
                        "text": result_data.get("text", ""),
                        "duration": total_time,
                        "segments": result_data.get("segments"),
                    }
                else:
                    result = result_data

                self.logger.info(
                    f"[{job.uid}] Transcription completed: "
                    f"format={output_format}, queue_time={queue_time:.2f}s, "
                    f"inference_time={inference_time:.2f}s"
                )

                if not job.future.done():
                    job.future.set_result(result)

            except Exception as e:
                self.logger.exception(f"❌ [{job.uid}] Job failed: {e}")
                if not job.future.done():
                    job.future.set_exception(e)

            finally:
                if os.path.exists(job.temp_dir):
                    shutil.rmtree(job.temp_dir, ignore_errors=True)
                self.queue.task_done()
