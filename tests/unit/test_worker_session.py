"""Worker-session Interface under controlled transport failure and cancellation."""

import asyncio
import threading
from unittest.mock import MagicMock, patch

import pytest

from src.services.worker_session import WorkerRemoteError, WorkerSession
from src.workers.model_worker import WorkerJob
from src.workers.transport import ProcessTransport, WorkerConfig
from tests.helpers.worker_transport import FakeTransport, WorkerHarness

CONFIG = WorkerConfig("funasr", "test", 0)
OTHER = WorkerConfig("mlx", "other", 0)


@pytest.mark.parametrize("message", [("LOAD_ERROR", "bad model"), ("WRONG", None), "garbage"])
async def test_startup_message_failure_disposes_and_allows_retry(message: object) -> None:
    harness = WorkerHarness()

    def configure(transport: FakeTransport) -> None:
        transport.ready = False
        transport.messages.put(message)

    harness.configure = configure
    session = WorkerSession(harness.factory)
    with pytest.raises(RuntimeError, match="model|startup"):
        await session.start(CONFIG)
    assert harness.transports[0].closed
    assert not session.alive
    harness.configure = None
    await session.start(CONFIG)
    assert session.alive
    await session.close()


@pytest.mark.parametrize("failure", ["timeout", "death", "cancel", "start"])
async def test_incomplete_startup_reclaims_transport(failure: str) -> None:
    harness = WorkerHarness()
    harness.configure = lambda t: setattr(t, "ready", False)
    session = WorkerSession(harness.factory, startup_timeout=0.03)
    start = asyncio.create_task(session.start(CONFIG))
    transport = await harness.next_transport()
    if failure == "death":
        transport.running = False
    elif failure == "cancel":
        start.cancel()
    elif failure == "start":
        # start() throwing after partial initialization has the same disposal path.
        start.cancel()
        with pytest.raises(asyncio.CancelledError):
            await start
        harness.configure = lambda t: setattr(t, "start_error", RuntimeError("start failed"))
        start = asyncio.create_task(session.start(CONFIG))
    expected = asyncio.CancelledError if failure == "cancel" else RuntimeError
    with pytest.raises(expected):
        await asyncio.wait_for(start, 1)
    assert all(t.closed for t in harness.transports)
    assert not session.alive
    await session.close()


async def test_replacement_stops_old_reader_before_new_handshake() -> None:
    harness = WorkerHarness()
    session = WorkerSession(harness.factory)
    await session.start(CONFIG)
    old = session.enqueue(WorkerJob("old", "a.wav", {}))
    await session.start(OTHER)
    with pytest.raises(RuntimeError, match="terminated"):
        await old
    transport = harness.transports[-1]
    new = session.enqueue(WorkerJob("new", "b.wav", {}))
    # A late message in the old transport cannot steal/resolve new completions.
    harness.transports[0].messages.put(("RESULT", "new", "stale"))
    transport.messages.put(("RESULT", "new", "fresh"))
    assert await asyncio.wait_for(new, 1) == "fresh"
    await session.close()


@pytest.mark.parametrize("message", [
    ("ERROR", "job", "legacy"),
    ("ERROR", "job", "ValueError", "typed"),
    ("RESULT", "job", "success"),
])
async def test_completion_types_and_capacity_release(message: tuple[str, ...]) -> None:
    harness = WorkerHarness()
    session = WorkerSession(harness.factory)
    await session.start(CONFIG)
    future = session.enqueue(WorkerJob("job", "a.wav", {}))
    assert session.pending_count == 1
    harness.transports[0].messages.put(message)
    if message[0] == "RESULT":
        assert await asyncio.wait_for(future, 1) == "success"
    else:
        with pytest.raises(RuntimeError) as caught:
            await asyncio.wait_for(future, 1)
        if len(message) == 4:
            assert isinstance(caught.value, WorkerRemoteError)
            assert caught.value.exc_type_name == "ValueError"
    assert session.pending_count == 0
    await session.close()


@pytest.mark.parametrize("message", [("RESULT",), ("UNKNOWN", "job", "payload"), ("IDLE_EXIT", None)])
async def test_terminal_message_fails_all_pending(message: object) -> None:
    harness = WorkerHarness()
    session = WorkerSession(harness.factory)
    await session.start(CONFIG)
    futures = [session.enqueue(WorkerJob(str(i), "a.wav", {})) for i in range(2)]
    harness.transports[0].messages.put(message)
    for future in futures:
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(future, 1)
    assert session.pending_count == 0
    await session.close()
    assert harness.transports[0].closed


async def test_cancel_releases_slot_and_duplicate_does_not_replace_original() -> None:
    harness = WorkerHarness()
    session = WorkerSession(harness.factory)
    await session.start(CONFIG)
    first = session.enqueue(WorkerJob("same", "a.wav", {}))
    with pytest.raises(RuntimeError, match="Duplicate"):
        session.enqueue(WorkerJob("same", "b.wav", {}))
    assert not first.done()
    first.cancel()
    assert first.cancelled()
    assert session.pending_count == 0
    await session.close()


async def test_cancelled_shutdown_waits_for_disposal_before_replacement() -> None:
    entered = threading.Event()
    release = threading.Event()
    harness = WorkerHarness()
    session = WorkerSession(harness.factory)
    await session.start(CONFIG)
    transport = harness.transports[0]
    original_close = transport.close

    def slow_close() -> None:
        entered.set()
        assert release.wait(2), "test did not release cleanup"
        original_close()

    with patch.object(transport, "close", side_effect=slow_close):
        close = asyncio.create_task(session.close())
        assert await asyncio.to_thread(entered.wait, 1)
        close.cancel()
        close.cancel()  # repeated cancellation must not abandon the cleanup task
        replacement = asyncio.create_task(session.start(OTHER))
        try:
            assert len(harness.transports) == 1
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(close, 1)
            await asyncio.wait_for(replacement, 1)
        finally:
            release.set()
    assert transport.closed
    assert len(harness.transports) == 2
    await session.close()


async def test_failed_disposal_blocks_replacement_until_retry_succeeds() -> None:
    harness = WorkerHarness()
    session = WorkerSession(harness.factory)
    await session.start(CONFIG)
    transport = harness.transports[0]
    with patch.object(transport, "close", side_effect=RuntimeError("still alive")):
        with pytest.raises(RuntimeError, match="still alive"):
            await session.start(OTHER)
        assert len(harness.transports) == 1
        assert not session.alive
    await session.start(OTHER)
    assert transport.closed
    await session.close()


@pytest.mark.parametrize("survives_kill", [False, True])
def test_process_transport_checks_liveness_after_each_timed_join(survives_kill: bool) -> None:
    process = MagicMock()
    process.pid = 123
    process.is_alive.return_value = True
    if not survives_kill:
        process.kill.side_effect = lambda: setattr(process.is_alive, "return_value", False)
    with patch("src.workers.transport.multiprocessing.Process", return_value=process):
        transport = ProcessTransport(CONFIG)
    try:
        if survives_kill:
            with pytest.raises(RuntimeError, match="survived SIGKILL"):
                transport.close()
            process.close.assert_not_called()
            process.is_alive.return_value = False
        else:
            transport.close()
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        assert process.join.call_count == 3
    finally:
        transport.close()


async def test_cancelled_close_waiting_for_startup_still_disposes_started_worker() -> None:
    harness = WorkerHarness()
    harness.configure = lambda t: setattr(t, "ready", False)
    session = WorkerSession(harness.factory)
    startup = asyncio.create_task(session.start(CONFIG))
    transport = await harness.next_transport()
    close = asyncio.create_task(session.close())
    # Yield once so close has entered its lifetime wait before cancellation.
    await asyncio.sleep(0)
    close.cancel()
    transport.messages.put(("READY", None))
    await asyncio.wait_for(startup, 1)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(close, 1)
    assert transport.closed
    assert not session.alive
