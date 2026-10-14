"""No-network unit tests for the Aerospike memory backend.

Nothing here opens a connection: an autouse fixture makes the module's one
connect function fail, so a test that reaches for a server fails loudly. The
behavior that needs a server lives in ``aerospike_integration/``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
import inspect
import os
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import crewai.memory.storage.aerospike_storage as aerospike_storage
from crewai.memory.storage.aerospike_storage import (
    AerospikeStorage,
    _ancestors,
    _batch_dimension,
    _check_dimension,
    _child_scope,
    _cosine_scores,
    _entry_matches,
    _entry_to_record,
    _from_micros,
    _from_vector,
    _normalize_scope,
    _off_event_loop,
    _record_to_entry,
    _scope_matches,
    _to_micros,
    _to_vector,
    aerospike_storage_factory,
)
from crewai.memory.storage.backend import (
    EmbeddingDimensionMismatchError,
    StorageBackend,
)
from crewai.memory.types import MemoryRecord


@pytest.fixture(autouse=True)
def never_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that tries to open a real connection."""

    def refuse(host: str, port: int) -> Any:
        raise AssertionError("unit tests must not connect to Aerospike")

    monkeypatch.setattr(aerospike_storage, "_connect", refuse)
    monkeypatch.setattr(aerospike_storage, "_connections", {})


def _unconnected() -> AerospikeStorage:
    return object.__new__(AerospikeStorage)


def test_unit_suite_never_connects() -> None:
    with pytest.raises(AssertionError, match="must not connect"):
        aerospike_storage._get_connection("localhost", 3000)


def test_protocol_surface_without_a_connection() -> None:
    storage = _unconnected()

    assert isinstance(storage, StorageBackend)
    for name in ("asave", "asearch", "adelete"):
        assert inspect.iscoroutinefunction(getattr(storage, name))
    assert callable(storage.touch_records)
    assert callable(storage.close)


def test_native_vector_roundtrip() -> None:
    sdk = pytest.importorskip("aerospike_sdk")
    values = [0.1, -0.25, 3.5, 0.0]

    vector = _to_vector(values)

    assert isinstance(vector, sdk.Vector)
    assert not isinstance(vector, bytes | bytearray)
    back = _from_vector(vector)
    assert back.dtype == np.float32
    assert len(back) == len(values)
    np.testing.assert_allclose(back, values, rtol=1e-6)


def test_dimension_mismatch_raises_embedding_dimension_mismatch_error() -> None:
    with pytest.raises(EmbeddingDimensionMismatchError) as exc_info:
        _check_dimension(4, 8)
    assert exc_info.value.stored_dim == 4
    assert exc_info.value.new_dim == 8

    _check_dimension(4, 4)
    _check_dimension(None, 8)


def test_batch_dimension() -> None:
    def record(embedding: list[float] | None) -> MemoryRecord:
        return MemoryRecord(content="x", embedding=embedding)

    assert _batch_dimension([record([0.1] * 4), record(None), record([0.2] * 4)]) == 4
    assert _batch_dimension([record(None)]) == 0
    assert _batch_dimension([]) == 0
    with pytest.raises(EmbeddingDimensionMismatchError) as exc_info:
        _batch_dimension([record([0.1] * 4), record([0.1] * 8)])
    assert (exc_info.value.stored_dim, exc_info.value.new_dim) == (4, 8)


def test_cosine_scoring_helper() -> None:
    query = np.array([1.0, 0.0])
    matrix = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-3.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ]
    )

    scores = _cosine_scores(matrix, query)

    np.testing.assert_allclose(scores, [1.0, 0.0, -1.0, 2**-0.5, 0.0], atol=1e-12)
    assert scores.dtype == np.float64
    assert all(-1.0 <= float(score) <= 1.0 for score in scores)
    assert list(_cosine_scores(matrix, np.zeros(2))) == [0.0] * 5


def test_time_conversion() -> None:
    naive = datetime(2026, 3, 4, 5, 6, 7, 123456)
    plus_five = timezone(timedelta(hours=5))
    aware = datetime(2026, 3, 4, 10, 6, 7, 123456, tzinfo=plus_five)

    assert _to_micros(naive) == _to_micros(naive.replace(tzinfo=timezone.utc))
    assert _to_micros(aware) == _to_micros(datetime(2026, 3, 4, 5, 6, 7, 123456))
    assert _from_micros(_to_micros(naive)) == naive
    assert _from_micros(_to_micros(aware)).tzinfo is None
    assert _from_micros(0) == datetime(1970, 1, 1)
    assert _to_micros(datetime(1969, 12, 31, 23, 59, 59, 999999)) == -1


def test_scope_normalization_and_ancestors() -> None:
    assert _normalize_scope("a/b") == "/a/b"
    assert _normalize_scope("/a/b/") == "/a/b"
    assert _normalize_scope("//a//b//") == "/a/b"
    assert _normalize_scope("/") == "/"
    assert _normalize_scope("") == "/"
    assert _normalize_scope(None) == "/"

    assert _ancestors("/a/b") == ["/", "/a", "/a/b"]
    assert _ancestors("/") == ["/"]


def test_chunk_keys_are_scope_and_number() -> None:
    storage = _unconnected()
    storage._scope_set = SimpleNamespace(id=lambda key: key)  # type: ignore[attr-defined]

    assert storage._key("/a/b", 0) == "/a/b|0"
    assert storage._key("/", 12) == "/|12"


def test_scope_prefix_matching_is_path_segment_aware() -> None:
    assert _scope_matches("/a/b", "/a")
    assert _scope_matches("/a", "/a")
    assert _scope_matches("/a", "/a/")
    assert not _scope_matches("/ab", "/a")
    assert not _scope_matches("/a", "/a/b")
    assert _scope_matches("/anything", None)
    assert _scope_matches("/anything", "/")

    assert _child_scope("/", "/a/b") == "/a"
    assert _child_scope("/a", "/a/b/c") == "/a/b"
    assert _child_scope("/a", "/a") is None
    assert _child_scope("/a", "/ab") is None


def test_entry_filters() -> None:
    entry = {
        "cat": ["x", "y"],
        "md": '{"team": "red", "nested": {"k": [1, 2]}, "n": 3}',
    }

    assert _entry_matches(entry, None, None)
    assert _entry_matches(entry, ["y", "zzz"], None)
    assert not _entry_matches(entry, ["zzz"], None)
    assert _entry_matches(entry, None, {"team": "red", "n": 3})
    assert not _entry_matches(entry, None, {"team": "red", "n": 4})
    assert _entry_matches(entry, None, {"nested": {"k": [1, 2]}})
    assert not _entry_matches(entry, None, {"nested": {"k": [1]}})
    assert not _entry_matches(entry, None, {"missing": "anything"})
    assert not _entry_matches(entry, ["x"], {"team": "blue"})
    assert not _entry_matches({"cat": [], "md": ""}, None, {"team": "red"})


def test_entry_record_conversion_roundtrip() -> None:
    record = MemoryRecord(
        id="rec-1",
        content="a memory with unicode é中",
        scope="/a/b",
        categories=["c1", "c2"],
        metadata={"k": "v", "nested": {"n": [1, 2.5, None]}},
        importance=0.8,
        created_at=datetime(2026, 1, 2, 3, 4, 5, 678901),
        last_accessed=datetime(2026, 2, 3, 4, 5, 6, 789012),
        source="user-1",
        private=True,
    )

    entry = _record_to_entry(record)
    back = _entry_to_record("/a/b", "rec-1", entry, [0.5, 0.25])

    assert back.model_dump() == record.model_dump()
    assert back.embedding == [0.5, 0.25]
    assert "src" in entry


def test_locate_retries_an_empty_index_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _unconnected()
    row = SimpleNamespace(
        is_ok=True,
        record=SimpleNamespace(bins={"scope": "/a", "chunk": 0}),
    )

    class Stream:
        def __init__(self, rows: list[Any]) -> None:
            self.rows = rows

        def __enter__(self) -> Stream:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def __iter__(self) -> Any:
            return iter(self.rows)

    class Builder:
        calls = 0

        def bins(self, _bins: list[str]) -> Builder:
            return self

        def filter(self, _filter: Any) -> Builder:
            return self

        def stream(self) -> Stream:
            self.calls += 1
            return Stream([] if self.calls == 1 else [row])

    builder = Builder()
    api = SimpleNamespace(
        Filter=SimpleNamespace(contains=lambda *_args: None),
        CollectionIndexType=SimpleNamespace(MAP_KEYS="map_keys"),
    )
    monkeypatch.setattr(storage, "_api", api, raising=False)
    monkeypatch.setattr(
        storage,
        "_connection",
        SimpleNamespace(session=SimpleNamespace(query=lambda *_args: builder)),
        raising=False,
    )
    monkeypatch.setattr(storage, "_scope_set", object(), raising=False)

    assert storage._locate(["rec-1"]) == {"rec-1": [("/a", 0)]}
    assert builder.calls == 2


def test_iter_candidates_retries_an_empty_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _unconnected()
    bins = {"scope": "/a", "chunk": 0, "mem": {"rec-1": {}}}
    row = SimpleNamespace(is_ok=True, record=SimpleNamespace(bins=bins))

    class Stream:
        def __init__(self, rows: list[Any]) -> None:
            self.rows = rows

        def __enter__(self) -> Stream:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def __iter__(self) -> Any:
            return iter(self.rows)

    class Builder:
        calls = 0

        def bins(self, _bins: list[str]) -> Builder:
            return self

        def stream(self) -> Stream:
            self.calls += 1
            return Stream([] if self.calls == 1 else [row])

    builder = Builder()
    monkeypatch.setattr(storage, "_api", SimpleNamespace(), raising=False)
    monkeypatch.setattr(
        storage,
        "_connection",
        SimpleNamespace(session=SimpleNamespace(query=lambda *_args: builder)),
        raising=False,
    )
    monkeypatch.setattr(storage, "_scope_set", object(), raising=False)

    assert list(storage._iter_candidates(None, None, ("mem",))) == [bins]
    assert builder.calls == 2


def test_entry_without_source_omits_it() -> None:
    record = MemoryRecord(content="plain", scope="/a")

    entry = _record_to_entry(record)

    assert "src" not in entry
    assert _entry_to_record("/a", record.id, entry).source is None
    assert _entry_to_record("/a", record.id, entry).embedding is None


@pytest.mark.asyncio
async def test_async_methods_do_not_block_event_loop() -> None:
    storage = _unconnected()

    def slow(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.3)
        return 0

    storage.save = slow  # type: ignore[method-assign]
    storage.search = slow  # type: ignore[method-assign]
    storage.delete = slow  # type: ignore[method-assign]
    calls = {
        "asave": lambda: storage.asave([]),
        "asearch": lambda: storage.asearch([0.1, 0.2]),
        "adelete": lambda: storage.adelete(scope_prefix="/a"),
    }

    async def ticks_during(call: Callable[[], Any]) -> int:
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        task = asyncio.create_task(ticker())
        try:
            await call()
        finally:
            task.cancel()
        return ticks

    for name, call in calls.items():
        ticks = await ticks_during(call)
        # A blocked loop would tick once or not at all during the 0.3s call.
        assert ticks >= 10, f"{name} blocked the event loop (ticks={ticks})"


@pytest.mark.asyncio
async def test_blocking_methods_called_inside_a_loop_run_on_another_thread() -> None:
    """The SDK refuses blocking calls on an event-loop thread (e.g. ``arecall``)."""

    @_off_event_loop
    def blocking(a: int, b: int = 0) -> tuple[int, bool]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return a + b, False
        return a + b, True

    @_off_event_loop
    def failing() -> None:
        raise ValueError("boom")

    # Called from the loop thread: runs elsewhere, keeps result and exceptions.
    assert blocking(1, b=2) == (3, False)
    with pytest.raises(ValueError, match="boom"):
        failing()
    # Called from a plain thread: runs directly.
    assert await asyncio.to_thread(blocking, 4) == (4, False)


def test_library_code_reads_no_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    class NoEnvironment:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError("library code must not read the environment")

        def __getitem__(self, key: str) -> Any:
            raise AssertionError("library code must not read the environment")

        def get(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("library code must not read the environment")

        def __contains__(self, key: object) -> bool:
            raise AssertionError("library code must not read the environment")

    class FakeStorage:
        def __init__(self, spec: Any) -> None:
            self.spec = spec

    monkeypatch.setattr(aerospike_storage, "AerospikeStorage", FakeStorage)

    # Guard only the library calls, so the test runner can still read its own.
    with monkeypatch.context() as scoped:
        scoped.setattr(os, "environ", NoEnvironment())
        built = aerospike_storage_factory(
            "aerospike://db.example.com:3000/test?set_prefix=p"
        )
        other = aerospike_storage_factory("/some/path")

    assert other is None
    assert isinstance(built, FakeStorage)
    assert (built.spec.host, built.spec.port) == ("db.example.com", 3000)
    assert (built.spec.namespace, built.spec.set_prefix) == ("test", "p")


def test_missing_sdk_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "aerospike_sdk", None)
    monkeypatch.setitem(sys.modules, "aerospike_sdk.sync", None)

    with pytest.raises(ImportError, match=r"crewai\[aerospike\]"):
        aerospike_storage._sdk()
