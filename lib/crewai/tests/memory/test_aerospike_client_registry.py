"""Tests for the process-wide Aerospike client registry and storage lifecycle.

The module's ``_connect`` seam is replaced by a counting fake, so these tests
never open a connection and never need the Aerospike client installed.
"""

from __future__ import annotations

import copy
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

import crewai.memory.storage.aerospike_storage as aerospike_storage
from crewai.memory.storage.aerospike_storage import (
    AerospikeSpec,
    AerospikeStorage,
    _Connection,
    _get_connection,
    shutdown_connections,
)


class FakeCluster:
    def __init__(self, fail_on_close: bool = False) -> None:
        self.close_calls = 0
        self._fail_on_close = fail_on_close

    def close(self) -> None:
        self.close_calls += 1
        if self._fail_on_close:
            raise RuntimeError("close failed")


class ConnectRecorder:
    """Counting replacement for ``_connect``."""

    def __init__(self, delay: float = 0.0) -> None:
        self.calls: list[tuple[str, int]] = []
        self.clusters: list[FakeCluster] = []
        self._delay = delay

    def __call__(self, host: str, port: int) -> _Connection:
        self.calls.append((host, port))
        time.sleep(self._delay)
        cluster = FakeCluster()
        self.clusters.append(cluster)
        return _Connection(cluster=cluster, session=object())


@pytest.fixture
def connect(monkeypatch: pytest.MonkeyPatch) -> ConnectRecorder:
    recorder = ConnectRecorder()
    monkeypatch.setattr(aerospike_storage, "_connect", recorder)
    monkeypatch.setattr(aerospike_storage, "_connections", {})
    monkeypatch.setattr(aerospike_storage, "_atexit_registered", False)
    monkeypatch.setattr(aerospike_storage.atexit, "register", lambda fn: fn)
    fake_data_set = SimpleNamespace(
        of=lambda namespace, name: SimpleNamespace(id=lambda key: (namespace, name, key))
    )
    monkeypatch.setattr(
        aerospike_storage, "_sdk", lambda: (SimpleNamespace(DataSet=fake_data_set), None)
    )
    monkeypatch.setattr(AerospikeStorage, "_ensure_indexes", lambda self: None)
    return recorder


def _storage(host: str = "localhost", port: int = 3000) -> AerospikeStorage:
    return AerospikeStorage(AerospikeSpec(host, port, "test"))


def test_one_connection_per_host_port(connect: ConnectRecorder) -> None:
    storages = [_storage() for _ in range(10)]
    assert len(connect.calls) == 1
    assert len({id(s._session) for s in storages}) == 1

    _storage(port=3001)
    assert len(connect.calls) == 2

    _storage(host="LOCALHOST")
    assert len(connect.calls) == 2


def test_connection_created_once_under_thread_contention(
    connect: ConnectRecorder,
) -> None:
    connect._delay = 0.05
    barrier = threading.Barrier(16)
    results: list[_Connection] = []

    def worker() -> None:
        barrier.wait()
        results.append(_get_connection("localhost", 3000))

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(connect.calls) == 1
    assert len(results) == 16
    assert all(connection is results[0] for connection in results)


def test_storage_close_does_not_close_shared_client(connect: ConnectRecorder) -> None:
    first, second = _storage(), _storage()

    first.close()

    assert connect.clusters[0].close_calls == 0
    assert second._session is first._session
    assert _get_connection("localhost", 3000).session is second._session


def test_deepcopy_returns_same_storage(connect: ConnectRecorder) -> None:
    storage = _storage()

    assert copy.deepcopy(storage) is storage
    assert copy.copy(storage) is storage
    assert copy.deepcopy({"storage": storage})["storage"] is storage


def test_shutdown_closes_each_client_once_and_is_idempotent(
    connect: ConnectRecorder,
) -> None:
    _get_connection("localhost", 3000)
    _get_connection("localhost", 3001)
    connect.clusters[0]._fail_on_close = True

    shutdown_connections()
    shutdown_connections()

    assert [cluster.close_calls for cluster in connect.clusters] == [1, 1]
    assert aerospike_storage._connections == {}


def test_shutdown_is_registered_once_at_exit(
    connect: ConnectRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    registered: list[Any] = []
    monkeypatch.setattr(aerospike_storage.atexit, "register", registered.append)

    _get_connection("localhost", 3000)
    _get_connection("localhost", 3001)

    assert registered == [shutdown_connections]


def test_connections_are_created_on_one_thread_that_outlives_the_caller(
    connect: ConnectRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A client created on a thread that exits crashes the interpreter later."""
    creators: list[threading.Thread] = []

    def record_creator(host: str, port: int) -> _Connection:
        creators.append(threading.current_thread())
        return connect(host, port)

    monkeypatch.setattr(aerospike_storage, "_connect", record_creator)

    callers = [
        threading.Thread(target=_get_connection, args=("localhost", port))
        for port in (3000, 3001)
    ]
    for caller in callers:
        caller.start()
        caller.join()

    assert len(creators) == 2
    assert creators[0] is creators[1]
    assert creators[0] not in callers
    assert creators[0].is_alive() and creators[0].daemon


def test_a_failed_connect_reaches_the_caller_and_is_not_cached(
    connect: ConnectRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(host: str, port: int) -> _Connection:
        raise ConnectionError("no server")

    monkeypatch.setattr(aerospike_storage, "_connect", refuse)
    with pytest.raises(ConnectionError, match="no server"):
        _get_connection("localhost", 3000)

    monkeypatch.setattr(aerospike_storage, "_connect", connect)
    assert _get_connection("localhost", 3000).cluster is connect.clusters[0]
