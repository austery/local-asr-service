"""Service-owned remote tasks, serial admission, queue-inclusive deadlines."""

import asyncio
from pathlib import Path

from src.adapters.deepgram_http import DeepgramAdapter
from src.config import (
    DEEPGRAM_API_KEY,
    DEEPGRAM_ENABLED,
    DEEPGRAM_TOTAL_TIMEOUT_SEC,
    MAX_UPLOAD_SIZE_MB,
)
from src.core.remote_port import RemoteTranscriber, RemoteTranscript, RemoteTranscriptionError
from src.services.execution import TranscriptionResult
from src.services.worker_session import finish_cleanup

_AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".m4a", ".mp4", ".flac", ".ogg", ".webm"})
_GATEWAY_CEILING = 200 * 1024 * 1024


def deepgram_configured() -> bool:
    """Credential presence only, never a claim of provider acceptance."""
    return bool(DEEPGRAM_API_KEY.strip())


def deepgram_unavailable_reason() -> str | None:
    if not DEEPGRAM_ENABLED:
        return "Deepgram disabled: operator must complete live acceptance and set DEEPGRAM_ENABLED=true."
    if not deepgram_configured():
        return "Deepgram enabled but DEEPGRAM_API_KEY is missing; configure it server-side."
    return None


class RemoteSession:
    def __init__(self, provider: RemoteTranscriber | None = None, *, timeout: float = DEEPGRAM_TOTAL_TIMEOUT_SEC) -> None:
        self._provider = provider or DeepgramAdapter(DEEPGRAM_API_KEY)
        self._timeout = timeout
        self._semaphore = asyncio.Semaphore(1)
        self._tasks: set[asyncio.Task[RemoteTranscript]] = set()
        self._stopping = False

    def start(self) -> None:
        if self._stopping and self._tasks:
            raise RuntimeError("Remote session is still disposing requests")
        self._stopping = False

    def preflight(self, filename: str, size: int, params: dict[str, object]) -> None:
        reason = deepgram_unavailable_reason()
        if reason:
            raise RemoteTranscriptionError(503, reason)
        if params.get("language") not in {"en", "zh"}:
            raise RemoteTranscriptionError(400, "Deepgram requires explicit language=en or language=zh; auto/multi modes are not enabled.")
        if params.get("output_format", "json") not in {"json", "txt", "srt"}:
            raise RemoteTranscriptionError(400, "Deepgram supports json, txt/text, and srt output only.")
        if size > min(_GATEWAY_CEILING, MAX_UPLOAD_SIZE_MB * 1024 * 1024):
            raise RemoteTranscriptionError(413, "Deepgram upload exceeds gateway size limit (at most 200 MiB).")
        if size <= 0 or Path(filename).suffix.lower() not in _AUDIO_EXTENSIONS:
            raise RemoteTranscriptionError(400, "Deepgram requires a nonempty supported audio file: wav/mp3/m4a/mp4/flac/ogg/webm.")

    def deadline(self) -> float:
        return asyncio.get_running_loop().time() + self._timeout

    async def submit(self, path: Path, language: str, request_id: str, deadline: float) -> RemoteTranscript:
        if self._stopping:
            raise RemoteTranscriptionError(503, "Deepgram session is stopping.")
        task = asyncio.create_task(self._run(path, language, request_id, deadline))
        self._tasks.add(task)
        try:
            return await asyncio.shield(task)
        finally:
            await finish_cleanup(self._dispose(task))

    async def _run(self, path: Path, language: str, request_id: str, deadline: float) -> RemoteTranscript:
        started = False
        try:
            async with asyncio.timeout_at(deadline), self._semaphore:
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError
                started = True
                result = await self._provider.transcribe(path, language, request_id)
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError
                return result
        except TimeoutError:
            detail = "result unknown; do not automatically resend" if started else "expired in queue; no upload was sent"
            raise RemoteTranscriptionError(504, f"Deepgram total deadline exceeded: {detail}.") from None

    async def _dispose(self, task: asyncio.Task[RemoteTranscript]) -> None:
        if not task.done() and not task.cancelling():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self._tasks.discard(task)

    async def close(self) -> None:
        self._stopping = True
        tasks = list(self._tasks)
        for task in tasks:
            if not task.done() and not task.cancelling():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.difference_update(tasks)


def render_remote(result: RemoteTranscript, output_format: object, with_timestamp: bool = False) -> TranscriptionResult:
    if output_format == "srt":
        return "\n\n".join(
            f"{i}\n{_srt_time(s.start)} --> {_srt_time(s.end)}\n[{s.speaker}]: {s.text}"
            for i, s in enumerate(result.segments, 1)
        )
    if output_format == "txt":
        return "\n".join(
            (f"[{int(s.start // 60):02}:{int(s.start % 60):02}] " if with_timestamp else "")
            + f"[{s.speaker}]: {s.text}" for s in result.segments
        )
    return {
        "text": result.text, "duration": result.duration, "language": result.language,
        "segments": [{"speaker": s.speaker, "start": s.start, "end": s.end, "text": s.text} for s in result.segments],
    }


def _srt_time(seconds: float) -> str:
    millis = round(seconds * 1000)
    hours, millis = divmod(millis, 3600000)
    minutes, millis = divmod(millis, 60000)
    seconds_int, millis = divmod(millis, 1000)
    return f"{hours:02}:{minutes:02}:{seconds_int:02},{millis:03}"
