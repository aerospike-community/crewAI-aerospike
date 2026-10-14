"""Record layout, indexes, dimension tracking and chunking against a live server."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np
import pytest

import crewai.memory.storage.aerospike_storage as aerospike_module
from crewai.memory.storage.aerospike_storage import (
    AerospikeStorage,
    _get_connection,
    _sdk,
    aerospike_storage_factory,
)
from crewai.memory.storage.backend import EmbeddingDimensionMismatchError
from crewai.memory.types import MemoryRecord


Make = Callable[..., MemoryRecord]
Dump = Callable[[], dict[str, dict[str, Any]]]


def test_save_creates_one_record_per_scope(
    storage: AerospikeStorage, make: Make, dump: Dump
) -> None:
    sdk, _ = _sdk()
    storage.save(
        [
            make("a1", "/a", [1, 0, 0, 0], categories=["x"]),
            make("a2", "/a", [0, 1, 0, 0], categories=["y"]),
            make("a3", "/a", None),
            make("b1", "/b", [0, 0, 1, 0]),
            make("b2", "/b", [0, 0, 0, 1], categories=["x"]),
        ]
    )

    records = dump()

    assert set(records) == {"/a|0", "/b|0"}
    for bins in records.values():
        assert set(bins) == {
            "scope", "chunk", "anc", "mem", "emb", "cats", "created_at", "bytes",
        }  # fmt: skip
    assert len(records["/a|0"]["mem"]) == 3
    assert len(records["/b|0"]["mem"]) == 2
    assert len(records["/a|0"]["emb"]) == 2
    assert records["/a|0"]["anc"] == ["/", "/a"]
    assert sorted(records["/a|0"]["cats"]) == ["x", "y"]
    assert len(records["/a|0"]["created_at"]) == 3
    for bins in records.values():
        for value in bins["emb"].values():
            assert isinstance(value, sdk.Vector)
            assert len(value.numpy_value) == 4


def _index_names(
    aerospike_server: tuple[str, int, str], prefix: str
) -> list[str]:
    host, port, namespace = aerospike_server
    session = _get_connection(host, port).session
    return sorted(
        index["name"]
        for index in session.info().secondary_indexes(namespace)
        if index["set"] == prefix
    )


def test_indexes_exist_once_after_connect(
    storage: AerospikeStorage,
    aerospike_server: tuple[str, int, str],
    prefix: str,
) -> None:
    assert _index_names(aerospike_server, prefix) == sorted(
        f"{prefix}_{suffix}"
        for suffix in ("scope_idx", "cats_idx", "created_idx", "memids_idx")
    )


def test_reconnect_creates_no_index_error(
    storage: AerospikeStorage,
    aerospike_spec: str,
    aerospike_server: tuple[str, int, str],
    prefix: str,
) -> None:
    before = _index_names(aerospike_server, prefix)
    aerospike_module._indexes_ready.clear()

    again = aerospike_storage_factory(aerospike_spec)

    assert again is not None
    assert _index_names(aerospike_server, prefix) == before
    assert len(before) == 4


def test_dimension_mismatch_on_save_and_update(
    storage: AerospikeStorage, make: Make
) -> None:
    first = make("first", "/a", [1, 0, 0, 0])
    storage.save([first])

    with pytest.raises(EmbeddingDimensionMismatchError) as saved:
        storage.save([make("wide", "/a", [1.0] * 8)])
    assert (saved.value.stored_dim, saved.value.new_dim) == (4, 8)
    assert "reset-memories" in str(saved.value)

    with pytest.raises(EmbeddingDimensionMismatchError) as updated:
        storage.update(first.model_copy(update={"embedding": [1.0] * 8}))
    assert (updated.value.stored_dim, updated.value.new_dim) == (4, 8)

    with pytest.raises(EmbeddingDimensionMismatchError):
        storage.save([make("m1", "/a", [1, 0, 0, 0]), make("m2", "/a", [1.0] * 8)])

    storage.save([make("plain", "/a", None)])
    assert storage.count() == 2

    storage.reset()
    storage.save([make("wide", "/a", [1.0] * 8)])
    assert storage.count() == 1


def test_chunk_rollover_only_near_limit(
    aerospike_spec: str, make: Make, dump: Dump
) -> None:
    limit = 3_000
    storage = AerospikeStorage(
        aerospike_module.parse_aerospike_spec(aerospike_spec), chunk_limit_bytes=limit
    )
    below = [make(f"small {i}", "/s", [1, 0, 0, 0]) for i in range(3)]
    storage.save(below)
    assert set(dump()) == {"/s|0"}
    assert len(dump()["/s|0"]["mem"]) == 3

    saved = list(below)
    first_overflow_bytes = None
    for i in range(40):
        record = make(f"entry {i} " + "x" * 100, "/s", [0, 1, 0, 0])
        storage.save([record])
        saved.append(record)
        records = dump()
        if "/s|1" in records and first_overflow_bytes is None:
            first_overflow_bytes = records["/s|0"]["bytes"]
    records = dump()

    assert "/s|1" in records
    # The next chunk appears only when one more memory would not fit in the first.
    assert first_overflow_bytes is not None
    assert limit - 400 < first_overflow_bytes <= limit
    for bins in records.values():
        assert bins["bytes"] <= limit  # far below the 1 MiB server ceiling
    assert sum(len(b["mem"]) for b in records.values()) == len(saved)
    for record in saved:
        found = storage.get_record(record.id)
        assert found is not None and found.content == record.content


def test_one_save_larger_than_a_chunk_is_split(
    aerospike_spec: str, make: Make, dump: Dump
) -> None:
    limit = 2_000
    storage = AerospikeStorage(
        aerospike_module.parse_aerospike_spec(aerospike_spec), chunk_limit_bytes=limit
    )
    batch = [make(f"entry {i} " + "y" * 200, "/big", [1, 0, 0, 0]) for i in range(30)]

    storage.save(batch)

    records = dump()
    assert len(records) > 1
    assert sum(len(b["mem"]) for b in records.values()) == 30
    for bins in records.values():
        assert bins["bytes"] <= limit
    assert storage.count("/big") == 30


def test_get_record_roundtrip(storage: AerospikeStorage, make: Make) -> None:
    embedding = [0.1, -0.25, 0.5, 0.75]
    record = make(
        "Launch review on Friday",
        "/proj/a",
        embedding,
        categories=["planning", "q3"],
        metadata={"owner": "sam", "nested": {"a": [1, 2, {"b": None}]}, "flag": True},
        importance=0.8,
        source="user-1",
        private=True,
        created_at=datetime(2024, 5, 6, 7, 8, 9, 123456),
        last_accessed=datetime(2024, 5, 7, 8, 9, 10, 654321),
    )
    storage.save([record])

    found = storage.get_record(record.id)

    assert found is not None
    assert found.content == record.content
    assert found.scope == "/proj/a"
    assert found.categories == ["planning", "q3"]
    assert found.metadata == record.metadata
    assert found.importance == 0.8
    assert found.source == "user-1"
    assert found.private is True
    assert found.created_at == datetime(2024, 5, 6, 7, 8, 9, 123456)
    assert found.last_accessed == datetime(2024, 5, 7, 8, 9, 10, 654321)
    assert found.embedding is not None
    np.testing.assert_allclose(found.embedding, embedding, atol=1e-6)
    assert storage.get_record("no-such-id") is None


def test_update_replaces_moves_and_inserts(
    storage: AerospikeStorage, make: Make, dump: Dump
) -> None:
    stay = make("keep me", "/a", [1, 0, 0, 0])
    move = make("move me", "/b", [0, 1, 0, 0], metadata={"v": 1})
    storage.save([stay, move])

    storage.update(stay.model_copy(update={"content": "changed", "metadata": {"v": 2}}))
    found = storage.get_record(stay.id)
    assert found is not None
    assert (found.content, found.metadata) == ("changed", {"v": 2})
    assert storage.count() == 2

    storage.update(move.model_copy(update={"scope": "/c"}))
    moved = storage.get_record(move.id)
    assert moved is not None and moved.scope == "/c"
    assert set(dump()) == {"/a|0", "/c|0"}  # the emptied old record is gone
    assert storage.count() == 2

    fresh = make("brand new", "/d", [0, 0, 1, 0])
    storage.update(fresh)
    inserted = storage.get_record(fresh.id)
    assert inserted is not None and inserted.scope == "/d"
    assert storage.count() == 3


def test_update_without_embedding_drops_the_old_vector(
    storage: AerospikeStorage, make: Make, dump: Dump
) -> None:
    record = make("vector first", "/a", [1, 0, 0, 0])
    storage.save([record])

    storage.update(record.model_copy(update={"embedding": None}))

    assert dump()["/a|0"].get("emb", {}) == {}
    assert storage.search([1, 0, 0, 0]) == []


def test_hostile_strings_roundtrip(storage: AerospikeStorage, make: Make) -> None:
    quote_scope = "/a\"b'c/d;e\\f/ünï/ключ"
    long_text = "q" * 500
    hostile = make(
        "quotes ' \" ; \\ ☃",
        quote_scope,
        [1, 0, 0, 0],
        record_id="id'\"; DROP \\ ü " + "i" * 400,
        categories=["it's", 'say "hi"', "semi;colon", "back\\slash", long_text],
        metadata={"k'ey": "v\"al;\\", "long": long_text, "uni": "é中"},
    )
    bystander = make("bystander", "/a", [0, 1, 0, 0], categories=["it's"])
    storage.save([hostile, bystander])

    found = storage.get_record(hostile.id)
    assert found is not None
    assert (found.id, found.scope, found.categories) == (
        hostile.id,
        quote_scope,
        hostile.categories,
    )
    assert found.metadata == hostile.metadata

    only_hostile = storage.search([1, 0, 0, 0], scope_prefix=quote_scope, min_score=-1)
    assert [r.id for r, _ in only_hostile] == [hostile.id]
    by_category = storage.search([1, 0, 0, 0], categories=['say "hi"'], min_score=-1)
    assert [r.id for r, _ in by_category] == [hostile.id]
    by_long = storage.search([1, 0, 0, 0], categories=[long_text], min_score=-1)
    assert [r.id for r, _ in by_long] == [hostile.id]

    # A prefix or id shaped like an injection matches nothing and breaks nothing.
    assert storage.search([1, 0, 0, 0], scope_prefix="/a' OR '1'='1", min_score=-1) == []
    assert storage.get_record("x' OR '1'='1") is None
    assert storage.delete(record_ids=["x' OR '1'='1"]) == 0
    assert storage.count() == 2

    assert storage.delete(record_ids=[hostile.id]) == 1
    assert storage.count() == 1
    assert storage.get_record(bystander.id) is not None
