"""Single-submit Deepgram batch adapter. Never follow redirects or retry uploads."""

import json
import logging
from collections.abc import AsyncIterator
from http import HTTPStatus
from pathlib import Path
from uuid import UUID

import httpx

from src.adapters.deepgram_output import normalize_deepgram
from src.core.remote_port import RemoteTranscript, RemoteTranscriptionError

_ENDPOINT = "https://api.deepgram.com/v1/listen"
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_CHUNK_BYTES = 64 * 1024
_HTTP_TIMEOUT = httpx.Timeout(connect=10, write=60, read=660, pool=10)
logger = logging.getLogger(__name__)


def _safe_request_id(value: object) -> str | None:
    try:
        return str(UUID(value)) if isinstance(value, str) else None
    except ValueError:
        return None


async def _audio(path: Path) -> AsyncIterator[bytes]:
    # Bounded disk reads; no executor may retain the file after cancellation.
    with path.open("rb") as source:
        while chunk := source.read(_CHUNK_BYTES):
            yield chunk


class DeepgramAdapter:
    def __init__(self, api_key: str, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._api_key = api_key
        self._transport = transport

    async def transcribe(self, path: Path, language: str, request_id: str) -> RemoteTranscript:
        try:
            async with httpx.AsyncClient(
                timeout=_HTTP_TIMEOUT, follow_redirects=False, trust_env=False,
                transport=self._transport,
            ) as client:
                return await self._send(client, path, language, request_id)
        except httpx.TimeoutException:
            raise RemoteTranscriptionError(504, "Deepgram transport deadline exceeded; result unknown. Do not automatically resend.") from None
        except httpx.HTTPError:
            raise RemoteTranscriptionError(502, "Deepgram connection failed; result unknown. Do not automatically resend.") from None

    async def _send(self, client: httpx.AsyncClient, path: Path, language: str, request_id: str) -> RemoteTranscript:
        async with client.stream(
            "POST", _ENDPOINT,
            params={"model": "nova-3", "diarize_model": "v1", "utterances": "true", "language": language},
            headers={"Authorization": f"Token {self._api_key}", "Content-Type": "application/octet-stream", "Accept-Encoding": "identity", "Content-Length": str(path.stat().st_size)},
            content=_audio(path),
        ) as response:
            self._record_id(request_id, response.headers.get("dg-request-id"))
            self._check_status(response.status_code)
            if response.headers.get("content-encoding", "identity").lower() != "identity":
                raise RemoteTranscriptionError(502, "Deepgram returned an unsupported response encoding; do not resend.")
            body = bytearray()
            async for chunk in response.aiter_bytes(_CHUNK_BYTES):
                body.extend(chunk)
                if len(body) > _MAX_RESPONSE_BYTES:
                    raise RemoteTranscriptionError(502, "Deepgram response exceeded safety limit; do not automatically resend.")
            try:
                data: object = json.loads(body)
            except (ValueError, UnicodeError, RecursionError):
                raise RemoteTranscriptionError(502, "Deepgram returned invalid JSON; do not automatically resend.") from None
            if isinstance(data, dict) and isinstance(data.get("metadata"), dict):
                self._record_id(request_id, data["metadata"].get("request_id"))
            return normalize_deepgram(data, language)

    @staticmethod
    def _record_id(request_id: str, provider_id: object) -> None:
        safe_id = _safe_request_id(provider_id)
        local_id = _safe_request_id(request_id)
        if safe_id and local_id:
            logger.info("Deepgram evidence local_request_id=%s provider_request_id=%s", local_id, safe_id)

    @staticmethod
    def _check_status(status: int) -> None:
        if status == HTTPStatus.TOO_MANY_REQUESTS:
            raise RemoteTranscriptionError(503, "Deepgram rate limited the request; no automatic retry was made.")
        if status in {401, 403}:
            raise RemoteTranscriptionError(502, "Deepgram authentication failed; operator must check server credentials.")
        if not HTTPStatus.OK <= status < HTTPStatus.MULTIPLE_CHOICES:
            raise RemoteTranscriptionError(502, "Deepgram rejected or failed the request; do not automatically resend. Check provider usage first.")
