"""Per-method behavior of delete, touch, count, listings, scopes and reset."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from threading import Thread
from typing import Any

import pytest

from crewai.memory.storage.aerospike_storage import (
    AerospikeStorage,
    _get_connection,
    _sdk,
)
from crewai.memory.types import MemoryRecord


Make = Callable[..., MemoryRecord]
Dump = Callable[[], dict[str, dict[str, Any]]]

OLD = datetime(2020, 1, 1)
CUTOFF = datetime(2023, 1, 1)
NEW = datetime(2025, 1, 1)


def _seed(storage: AerospikeStorage, make: Make) -> dict[str, MemoryRecord]:
    records = {
        "r1": make("r1", "/a", [1, 0, 0, 0], categories=["x"], metadata={"k": 1}, created_at=OLD),
        "r2": make("r2", "/a", [0, 1, 0, 0], categories=["x"], metadata={"k": 2}, created_at=OLD),
        "r3": make("r3", "/a/b", [0, 0, 1, 0], categories=["y"], metadata={"k": 1}, created_at=OLD),
        "r4": make("r4", "/a", [0, 0, 0, 1], categories=["x"], metadata={"k": 1}, created_at=NEW),
        "r5": make("r5", "/c", [1, 1, 0, 0], categories=["x"], metadata={"k": 1}, created_at=OLD),
    }  # fmt: skip
    storage.save(list(records.values()))
    return records


def _remaining(storage: AerospikeStorage) -> set[str]:
    return {r.content for r in storage.list_records(limit=100)}


def test_delete_combines_filters_as_a_conjunction(
    storage: AerospikeStorage, make: Make
) -> None:
    _seed(storage, make)

    removed = storage.delete(
        scope_prefix="/a",
        categories=["x"],
        metadata_filter={"k": 1},
        older_than=CUTOFF,
    )

    assert removed == 1
    assert _remaining(storage) == {"r2", "r3", "r4", "r5"}

    cases: list[tuple[dict[str, Any], set[str]]] = [
        ({"record_ids": ["r1", "r5"]}, {"r1", "r5"}),
        ({"scope_prefix": "/a"}, {"r1", "r2", "r3", "r4"}),
        ({"categories": ["y"]}, {"r3"}),
        ({"categories": ["x", "y"]}, {"r1", "r2", "r3", "r4", "r5"}),
        ({"metadata_filter": {"k": 2}}, {"r2"}),
        ({"older_than": CUTOFF}, {"r1", "r2", "r3", "r5"}),
        ({"record_ids": ["r1", "r5"], "scope_prefix": "/a"}, {"r1"}),
        ({"record_ids": []}, set()),
    ]
    for kwargs, expected in cases:
        storage.reset()
        by_name = _seed(storage, make)
        if "record_ids" in kwargs:
            kwargs = {**kwargs, "record_ids": [by_name[n].id for n in kwargs["record_ids"]]}
        removed = storage.delete(**kwargs)
        assert removed == len(expected), kwargs
        assert _remaining(storage) == {"r1", "r2", "r3", "r4", "r5"} - expected, kwargs


def test_delete_whole_scope_and_everything(
    storage: AerospikeStorage, make: Make, dump: Dump
) -> None:
    _seed(storage, make)

    assert storage.delete(scope_prefix="/a") == 4
    assert set(dump()) == {"/c|0"}
    assert storage.count() == 1

    _seed(storage, make)
    assert storage.delete() == 6
    assert dump() == {}
    assert storage.count() == 0
    # The store metadata survives, so a different dimension is still refused.
    assert storage._stored_dim(refresh=True) == 4


def test_delete_removes_emptied_records_and_keeps_others(
    storage: AerospikeStorage, make: Make, dump: Dump
) -> None:
    base = datetime(2024, 1, 1)
    e = [make(f"e{i}", "/e", [1, 0, 0, 0], created_at=base + timedelta(seconds=i)) for i in range(2)]
    f = [make(f"f{i}", "/f", [0, 1, 0, 0], created_at=base + timedelta(seconds=10 + i)) for i in range(3)]
    shared = [make(f"s{i}", "/g", [0, 0, 1, 0], created_at=base) for i in range(2)]
    storage.save([*e, *f, *shared])

    assert storage.delete(record_ids=[r.id for r in e]) == 2
    assert storage.delete(record_ids=[f[0].id]) == 1
    assert storage.delete(record_ids=[shared[0].id]) == 1

    records = dump()
    assert set(records) == {"/f|0", "/g|0"}
    assert len(records["/f|0"]["mem"]) == 2
    assert len(records["/f|0"]["emb"]) == 2
    assert len(records["/f|0"]["created_at"]) == 2
    assert f[0].created_at not in [
        datetime(1970, 1, 1) + timedelta(microseconds=ts) for ts in records["/f|0"]["created_at"]
    ]
    # The survivor still has its timestamp even though the removed one shared it.
    assert len(records["/g|0"]["created_at"]) == 1
    assert storage.get_record(shared[1].id) is not None


def test_scope_prefix_is_path_segment_aware(
    storage: AerospikeStorage, make: Make
) -> None:
    storage.save(
        [
            make("exact", "/a", [1, 0, 0, 0]),
            make("child", "/a/b", [1, 0, 0, 0]),
            make("sibling", "/ab", [1, 0, 0, 0]),
        ]
    )

    assert storage.count("/a") == 2
    assert {r.content for r in storage.list_records("/a")} == {"exact", "child"}
    assert {r.content for r, _ in storage.search([1, 0, 0, 0], "/a")} == {"exact", "child"}
    assert storage.delete(scope_prefix="/a") == 2
    assert _remaining(storage) == {"sibling"}


def test_touch_records_updates_last_accessed(
    storage: AerospikeStorage, make: Make
) -> None:
    created = datetime(2024, 1, 1)
    accessed = datetime(2024, 2, 2)
    a = make("a", "/t", [1, 0, 0, 0], created_at=created, last_accessed=accessed)
    b = make("b", "/t", [0, 1, 0, 0], created_at=created, last_accessed=accessed)
    c = make("c", "/u", [0, 0, 1, 0], created_at=created, last_accessed=accessed)
    storage.save([a, b, c])

    storage.touch_records([a.id, c.id, "no-such-id"])
    storage.touch_records([])

    after = {r.content: r for r in storage.list_records()}
    assert after["a"].last_accessed > accessed
    assert after["c"].last_accessed > accessed
    assert after["b"].last_accessed == accessed
    assert {r.created_at for r in after.values()} == {created}


def test_count_and_list_records_ordering_and_paging(
    storage: AerospikeStorage, make: Make
) -> None:
    base = datetime(2024, 3, 1)
    records = [
        make("oldest", "/a", [1, 0, 0, 0], created_at=base),
        make("tie-b", "/a/b", [1, 0, 0, 0], created_at=base + timedelta(days=1), record_id="tie-b"),
        make("tie-a", "/a", [1, 0, 0, 0], created_at=base + timedelta(days=1), record_id="tie-a"),
        make("newest", "/z", [1, 0, 0, 0], created_at=base + timedelta(days=2)),
    ]  # fmt: skip
    storage.save(records)

    assert storage.count() == storage.count("/") == 4
    assert storage.count("/a") == 3
    assert storage.count("/missing") == 0

    listed = storage.list_records()
    assert [r.content for r in listed] == ["newest", "tie-a", "tie-b", "oldest"]
    assert all(r.embedding is None for r in listed)
    assert [r.content for r in storage.list_records(limit=2, offset=1)] == ["tie-a", "tie-b"]
    assert [r.content for r in storage.list_records("/a", limit=10)] == ["tie-a", "tie-b", "oldest"]
    assert storage.list_records(offset=10) == []


def test_get_scope_info_list_scopes_list_categories(
    storage: AerospikeStorage, make: Make
) -> None:
    base = datetime(2024, 3, 1)
    storage.save(
        [
            make("a", "/a", [1, 0, 0, 0], categories=["x", "y"], created_at=base),
            make("b", "/a/b", [1, 0, 0, 0], categories=["y"], created_at=base + timedelta(days=3)),
            make("d", "/a/c/d", [1, 0, 0, 0], categories=["z"], created_at=base + timedelta(days=1)),
            make("z", "/z", [1, 0, 0, 0], categories=["x"], created_at=base + timedelta(days=9)),
        ]  # fmt: skip
    )

    info = storage.get_scope_info("/a")
    assert info.path == "/a"
    assert info.record_count == 3
    assert info.categories == ["x", "y", "z"]
    assert info.oldest_record == base
    assert info.newest_record == base + timedelta(days=3)
    assert info.child_scopes == ["/a/b", "/a/c"]

    root = storage.get_scope_info("/")
    assert (root.path, root.record_count) == ("/", 4)
    assert root.child_scopes == ["/a", "/z"]

    unknown = storage.get_scope_info("/nope")
    assert (unknown.record_count, unknown.categories, unknown.child_scopes) == (0, [], [])
    assert unknown.oldest_record is None and unknown.newest_record is None

    assert storage.list_scopes("/") == ["/a", "/z"]
    assert storage.list_scopes("/a") == ["/a/b", "/a/c"]
    assert storage.list_scopes("/a/c/d") == []
    assert storage.list_categories("/a") == {"x": 1, "y": 2, "z": 1}
    assert storage.list_categories() == {"x": 2, "y": 2, "z": 1}


def test_reset_leaves_other_sets_untouched(
    storage: AerospikeStorage,
    make: Make,
    aerospike_server: tuple[str, int, str],
    prefix: str,
) -> None:
    host, port, namespace = aerospike_server
    sdk, _ = _sdk()
    session = _get_connection(host, port).session
    decoy_set = sdk.DataSet.of(namespace, f"{prefix}_decoy")
    other_prefix = sdk.DataSet.of(namespace, f"{prefix}x")
    session.upsert(decoy_set.id("keep")).bin("v").set_to(1).execute().close()
    session.upsert(other_prefix.id("keep")).bin("v").set_to(2).execute().close()
    try:
        _seed(storage, make)

        storage.reset("/a")
        assert _remaining(storage) == {"r5"}
        storage.reset("/")
        assert storage.count() == 0

        # The dimension was cleared too, so a new embedding size is accepted.
        storage.save([make("wide", "/a", [1.0] * 8)])
        assert storage.count() == 1
        storage.reset(None)
        assert storage.count() == 0

        for data_set, expected in ((decoy_set, 1), (other_prefix, 2)):
            with session.query(data_set.id("keep")).execute() as stream:
                assert [row.record.bins["v"] for row in stream if row.is_ok] == [expected]
    finally:
        session.truncate(decoy_set)
        session.truncate(other_prefix)


def test_concurrent_saves_to_one_scope_lose_nothing(
    storage: AerospikeStorage, make: Make
) -> None:
    threads_n, per_thread = 8, 25
    errors: list[BaseException] = []

    def work(n: int) -> None:
        try:
            for i in range(per_thread):
                storage.save([make(f"t{n}-{i}", "/same", [1, 0, 0, 0])])
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [Thread(target=work, args=(n,)) for n in range(threads_n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert storage.count("/same") == threads_n * per_thread
    assert len(storage.list_records("/same", limit=1000)) == threads_n * per_thread


def test_stale_delete_does_not_lose_a_concurrent_save(
    storage: AerospikeStorage, make: Make
) -> None:
    first = make("first", "/race", [1, 0, 0, 0])
    storage.save([first])
    stale = storage._read_bins([storage._key("/race", 0)], ["mem"])[0]
    assert stale is not None
    second = make("second", "/race", [0, 1, 0, 0])
    storage.save([second])  # bumps the record's generation after the read

    removed = storage._remove_entries(
        "/race", 0, lambda record_id, _entry: record_id == first.id, preread=stale
    )

    assert removed == 1
    assert storage.get_record(first.id) is None
    assert storage.get_record(second.id) is not None


def test_delete_many_ids_is_bounded(storage: AerospikeStorage) -> None:
    with pytest.raises(ValueError, match="record ids"):
        storage.delete(record_ids=[f"id-{i}" for i in range(10_001)])
