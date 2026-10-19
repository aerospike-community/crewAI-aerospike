"""Tests for the bounds on one Aerospike memory call.

The server is faked, so these run with no network and without the Aerospike
client installed.
"""

from __future__ import annotations

from collections.abc import Iterator
import logging
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import crewai.memory.storage.aerospike_storage as aerospike_storage
from crewai.memory.storage.aerospike_storage import (
    AerospikeSpec,
    AerospikeStorage,
    _check_ids,
    _check_limit,
    _Connection,
    _Item,
    _plan_chunks,
)
from crewai.memory.types import MemoryRecord


@pytest.fixture
def storage(monkeypatch: pytest.MonkeyPatch) -> AerospikeStorage:
    data_set = SimpleNamespace(
        of=lambda namespace, name: SimpleNamespace(id=lambda key: (namespace, name, key))
    )
    monkeypatch.setattr(
        aerospike_storage,
        "_connect",
        lambda host, port: _Connection(cluster=object(), session=object()),
    )
    monkeypatch.setattr(aerospike_storage, "_connections", {})
    monkeypatch.setattr(aerospike_storage.atexit, "register", lambda fn: fn)
    monkeypatch.setattr(
        aerospike_storage, "_sdk", lambda: (SimpleNamespace(DataSet=data_set), None)
    )
    monkeypatch.setattr(AerospikeStorage, "_ensure_indexes", lambda self: None)
    return AerospikeStorage(AerospikeSpec("localhost", 3000, "test"))


def test_check_limit_validates_and_clamps() -> None:
    assert _check_limit(1) == 1
    assert _check_limit(250) == 250
    assert _check_limit(10**9) == aerospike_storage._MAX_LIMIT
    for bad in (0, -3, True, 1.5, "10", None):
        with pytest.raises(ValueError, match="positive integer"):
            _check_limit(bad)  # type: ignore[arg-type]


def test_check_ids_deduplicates_and_caps() -> None:
    assert _check_ids(["a", "b", "a"]) == ["a", "b"]
    limit = aerospike_storage._MAX_RECORD_IDS
    assert len(_check_ids([str(i) for i in range(limit)])) == limit
    with pytest.raises(ValueError, match="record ids"):
        _check_ids([str(i) for i in range(limit + 1)])


def _items(sizes: list[int]) -> list[_Item]:
    return [_Item(f"id{i}", {}, None, size) for i, size in enumerate(sizes)]


def test_plan_chunks_never_fills_a_chunk_past_the_limit() -> None:
    items = _items([400] * 10)

    plan = _plan_chunks(items, chunk=0, used=0, limit=1_000)

    assert [chunk for chunk, _ in plan] == [0, 1, 2, 3, 4]
    assert [len(group) for _, group in plan] == [2, 2, 2, 2, 2]
    assert [i.id for _, group in plan for i in group] == [i.id for i in items]
    for _chunk, group in plan:
        assert sum(i.size for i in group) <= 1_000


def test_plan_chunks_starts_from_the_open_chunk_and_its_fill() -> None:
    plan = _plan_chunks(_items([300, 300, 300]), chunk=4, used=700, limit=1_000)

    assert [(chunk, len(group)) for chunk, group in plan] == [(4, 1), (5, 2)]


def test_plan_chunks_gives_an_oversized_memory_its_own_chunk() -> None:
    plan = _plan_chunks(_items([100, 5_000, 100]), chunk=0, used=0, limit=1_000)

    assert [(chunk, [i.size for i in group]) for chunk, group in plan] == [
        (0, [100]),
        (1, [5_000]),
        (2, [100]),
    ]


def test_plan_chunks_with_no_items_plans_nothing() -> None:
    assert _plan_chunks([], chunk=0, used=0, limit=1_000) == []


def test_chunk_limit_must_be_positive_and_below_the_cap(
    storage: AerospikeStorage,
) -> None:
    spec = AerospikeSpec("localhost", 3000, "test")
    for bad in (0, -1, aerospike_storage._MAX_CHUNK_LIMIT_BYTES + 1):
        with pytest.raises(ValueError, match="chunk_limit_bytes"):
            AerospikeStorage(spec, chunk_limit_bytes=bad)
    assert AerospikeStorage(spec, chunk_limit_bytes=1)._chunk_limit_bytes == 1


def test_save_rejects_oversized_input_before_touching_the_server(
    storage: AerospikeStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("the server must not be reached")

    monkeypatch.setattr(AerospikeStorage, "_write_items", fail)
    monkeypatch.setattr(AerospikeStorage, "_ensure_dimension", fail)
    cases = [
        MemoryRecord(content="x", scope="/" + "s" * 600),
        MemoryRecord(content="x", scope="/" + "/".join("d" * 1 for _ in range(40))),
        MemoryRecord(content="x", categories=["c" * 2_000]),
        MemoryRecord(content="x", id="i" * 2_000),
        MemoryRecord(content="x" * (aerospike_storage._MAX_ENTRY_BYTES + 1)),
    ]

    for record in cases:
        with pytest.raises(ValueError):
            storage.save([record])


def test_empty_id_list_is_treated_as_no_id_filter(
    storage: AerospikeStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    deleted_scopes: list[str | None] = []

    def delete_scopes(_storage: AerospikeStorage, scope_prefix: str | None) -> int:
        deleted_scopes.append(scope_prefix)
        return 3

    monkeypatch.setattr(AerospikeStorage, "_delete_scopes", delete_scopes)

    assert storage.delete(record_ids=[]) == 3
    assert storage.delete(scope_prefix="/a", record_ids=[]) == 3
    assert deleted_scopes == [None, "/a"]


def _vector(*values: float) -> Any:
    return SimpleNamespace(numpy_value=np.array(values, dtype=np.float32))


def test_search_stops_reading_at_the_entry_cap(
    storage: AerospikeStorage,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(aerospike_storage, "_SCAN_ENTRIES_LIMIT", 5)
    pulled: list[int] = []

    def candidates(self: AerospikeStorage, *_args: Any, **_kwargs: Any) -> Iterator[Any]:
        for chunk in range(100):  # 100 records of 3 memories: 300 candidates
            pulled.append(chunk)
            yield {
                "scope": f"/s{chunk}",
                "chunk": 0,
                "emb": {f"{chunk}-{i}": _vector(1.0, 0.0) for i in range(3)},
            }

    def fetch(
        self: AerospikeStorage, requests: list[tuple[Any, list[str]]], with_emb: bool
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        return [
            (
                {i: {"c": i, "ts": 1, "cat": [], "md": "{}"} for i in ids},
                {i: _vector(1.0, 0.0) for i in ids},
            )
            for _key, ids in requests
        ]

    monkeypatch.setattr(AerospikeStorage, "_stored_dim", lambda self, refresh=False: 2)
    monkeypatch.setattr(AerospikeStorage, "_iter_candidates", candidates)
    monkeypatch.setattr(AerospikeStorage, "_fetch_entries", fetch)

    with caplog.at_level(logging.WARNING, logger=aerospike_storage.__name__):
        results = storage.search([1.0, 0.0], limit=1_000)

    assert len(results) == 5
    assert len(pulled) <= 3  # the stream was abandoned, not drained
    assert "narrow the scope" in caplog.text


def test_search_clamps_the_limit(
    storage: AerospikeStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aerospike_storage, "_MAX_LIMIT", 2)

    def candidates(self: AerospikeStorage, *_args: Any, **_kwargs: Any) -> Iterator[Any]:
        yield {"scope": "/s", "chunk": 0, "emb": {f"m{i}": _vector(1.0, 0.0) for i in range(6)}}

    def fetch(
        self: AerospikeStorage, requests: list[tuple[Any, list[str]]], with_emb: bool
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        return [
            (
                {i: {"c": i, "ts": 1, "cat": [], "md": "{}"} for i in ids},
                {i: _vector(1.0, 0.0) for i in ids},
            )
            for _key, ids in requests
        ]

    monkeypatch.setattr(AerospikeStorage, "_stored_dim", lambda self, refresh=False: 2)
    monkeypatch.setattr(AerospikeStorage, "_iter_candidates", candidates)
    monkeypatch.setattr(AerospikeStorage, "_fetch_entries", fetch)

    assert len(storage.search([1.0, 0.0], limit=10**6)) == 2
    with pytest.raises(ValueError, match="positive integer"):
        storage.search([1.0, 0.0], limit=0)
