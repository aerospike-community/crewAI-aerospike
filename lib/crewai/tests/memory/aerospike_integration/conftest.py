"""Fixtures for the Aerospike memory backend tests that need a live server.

This is the only place the Aerospike backend's tests read environment
variables. The server address comes from ``AEROSPIKE_TEST_HOST`` (default
127.0.0.1), ``AEROSPIKE_TEST_PORT`` (default 3000) and
``AEROSPIKE_TEST_NAMESPACE`` (default test). When the server is unreachable the
tests skip, unless ``AEROSPIKE_TEST_REQUIRE`` is set, in which case they fail.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import crewai.memory.storage.aerospike_storage as aerospike_module
import crewai.memory.storage.factory as storage_factory
from crewai.memory.storage.aerospike_storage import (
    AerospikeStorage,
    _get_connection,
    _sdk,
    _truncate,
    aerospike_storage_factory,
)
from crewai.memory.types import MemoryRecord


_HERE = Path(__file__).parent
_EMBEDDING_DIM = 16

# Every set these tests truncate starts with this prefix, so a misconfigured
# environment can never wipe a real set (see ``_truncate_test_set``).
_TEST_SET_PREFIX = "t_"
# Passes of each set scan merged by the ``repeat_scans`` fixture.
_SCAN_PASSES = 3


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark every test in this directory as needing a live Aerospike server."""
    for item in items:
        if _HERE in Path(str(item.fspath)).parents:
            item.add_marker(pytest.mark.aerospike)


def _probe(host: str, port: int) -> str | None:
    """Try an SDK connect; return why it failed, or None when it worked.

    The SDK has no public connect-timeout option but gives up by itself within
    about two seconds (measured against a closed port and an unroutable
    address), so an absent server skips quickly. The connect runs on the calling
    thread on purpose: a client created on a helper thread that had exited
    crashed the interpreter when other threads used it later.
    """
    try:
        _get_connection(host, port)
    except Exception as exc:
        return type(exc).__name__
    return None


@pytest.fixture(scope="session")
def aerospike_server() -> tuple[str, int, str]:
    """Return (host, port, namespace) once the server answers a client connect.

    Reachability is proven with the SDK itself, never a Python socket, because
    the repository's network block intercepts Python sockets.
    """
    host = os.environ.get("AEROSPIKE_TEST_HOST", "127.0.0.1")
    port = int(os.environ.get("AEROSPIKE_TEST_PORT", "3000"))
    namespace = os.environ.get("AEROSPIKE_TEST_NAMESPACE", "test")
    problem = _probe(host, port)
    if problem is not None:
        message = f"No Aerospike server reachable at {host}:{port} ({problem})"
        if os.environ.get("AEROSPIKE_TEST_REQUIRE"):
            pytest.fail(message)
        pytest.skip(message)
    return host, port, namespace


@pytest.fixture
def aerospike_spec(aerospike_server: tuple[str, int, str]) -> str:
    """Storage spec whose set prefix is unique per xdist worker."""
    host, port, namespace = aerospike_server
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    return f"aerospike://{host}:{port}/{namespace}?set_prefix={_TEST_SET_PREFIX}crewai_{worker}"


def _truncate_test_set(session: Any, namespace: str, set_name: str) -> None:
    """Truncate one set, refusing any set that is not a test set."""
    if not set_name.startswith(_TEST_SET_PREFIX):
        raise AssertionError(
            f"Refusing to truncate set {set_name!r}: test sets start with "
            f"{_TEST_SET_PREFIX!r}."
        )
    sdk, _ = _sdk()
    _truncate(session, sdk.DataSet.of(namespace, set_name), namespace, set_name)


@pytest.fixture
def truncate_test_set() -> Callable[[Any, str, str], None]:
    """The guarded truncate helper, for tests of the guard itself."""
    return _truncate_test_set


@pytest.fixture(autouse=True)
def clean_sets(
    aerospike_server: tuple[str, int, str], aerospike_spec: str
) -> Iterator[None]:
    """Truncate this worker's sets before each test."""
    host, port, namespace = aerospike_server
    prefix = aerospike_spec.rsplit("=", 1)[1]
    session = _get_connection(host, port).session
    for set_name in (prefix, f"{prefix}_meta"):
        _truncate_test_set(session, namespace, set_name)
    yield


@pytest.fixture(autouse=True)
def repeat_scans(monkeypatch: pytest.MonkeyPatch) -> None:
    """Union a few passes of each set scan, working around a dev client defect.

    The development client, on the server build it was developed against,
    silently returns an incomplete answer to about 1 to 2 percent of set scans
    and index queries: no error, sometimes no rows at all and sometimes only some
    of them. It is reproducible with plain SDK calls, with no other client and
    no writes in flight, and the server's own query counters show the full row
    count, so the rows are lost on the client side. Without this a test fails at
    random. Each scan here is therefore made three times and the scope records
    are merged by key, so a test fails only if a record is dropped three times
    in a row. Every row is still produced by the library's own code path.
    """
    real_candidates = AerospikeStorage._iter_candidates
    def iter_candidates(
        self: AerospikeStorage, *args: Any, **kwargs: Any
    ) -> Iterator[Any]:
        seen: set[tuple[Any, Any]] = set()
        for _pass in range(_SCAN_PASSES):
            for row in real_candidates(self, *args, **kwargs):
                ident = (row.get("scope"), row.get("chunk"))
                if ident not in seen:
                    seen.add(ident)
                    yield row
            if len(seen) >= aerospike_module._SCAN_RECORDS_LIMIT:
                return  # the library's own cap was reached: keep its answer

    monkeypatch.setattr(AerospikeStorage, "_iter_candidates", iter_candidates)


@pytest.fixture
def storage(aerospike_spec: str) -> AerospikeStorage:
    """An AerospikeStorage for this worker's sets."""
    built = aerospike_storage_factory(aerospike_spec)
    assert isinstance(built, AerospikeStorage)
    return built


@pytest.fixture
def prefix(aerospike_spec: str) -> str:
    """This worker's set prefix."""
    return aerospike_spec.rsplit("=", 1)[1]


@pytest.fixture
def dump(
    aerospike_server: tuple[str, int, str], prefix: str
) -> Callable[[], dict[str, dict[str, Any]]]:
    """Return a function that reads every scope record, keyed ``scope|chunk``."""
    host, port, namespace = aerospike_server

    def read() -> dict[str, dict[str, Any]]:
        sdk, _ = _sdk()
        session = _get_connection(host, port).session
        found: dict[str, dict[str, Any]] = {}
        for _pass in range(_SCAN_PASSES):  # see ``repeat_scans``
            with session.query(sdk.DataSet.of(namespace, prefix)).execute() as stream:
                for row in stream:
                    if row.is_ok and row.record is not None:
                        bins = row.record.bins
                        found[f"{bins['scope']}|{bins['chunk']}"] = bins
        return found

    return read


@pytest.fixture
def make() -> Callable[..., MemoryRecord]:
    """Build a MemoryRecord: ``make(content, scope, embedding, **fields)``."""

    def build(
        content: str,
        scope: str = "/a",
        embedding: list[float] | None = None,
        **fields: Any,
    ) -> MemoryRecord:
        if "record_id" in fields:
            fields["id"] = fields.pop("record_id")
        return MemoryRecord(content=content, scope=scope, embedding=embedding, **fields)

    return build


@pytest.fixture
def registered_factory() -> Iterator[None]:
    """Register the Aerospike factory, restoring the previous one afterwards."""
    original = storage_factory._factory
    storage_factory.set_memory_storage_factory(aerospike_storage_factory)
    yield
    storage_factory.set_memory_storage_factory(original)


@pytest.fixture
def embedder() -> Callable[[list[str]], list[list[float]]]:
    """Deterministic embedder: sha256-seeded, normalized, 16 dimensions."""

    def embed(texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
            vector = np.random.default_rng(seed).normal(size=_EMBEDDING_DIM)
            out.append((vector / np.linalg.norm(vector)).tolist())
        return out

    return embed
