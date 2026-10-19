"""End-to-end tracer: Memory with an aerospike:// spec against a live server."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest

from crewai.memory.storage.aerospike_storage import _get_connection, _sdk
from crewai.memory.unified_memory import Memory


@pytest.fixture
def memory(
    registered_factory: None,
    aerospike_spec: str,
    embedder: Callable[[list[str]], list[list[float]]],
) -> Iterator[Memory]:
    mem = Memory(storage=aerospike_spec, llm=MagicMock(), embedder=embedder)
    yield mem
    mem.close()


def test_memory_remember_then_recall_roundtrip(memory: Memory) -> None:
    remembered = memory.remember(
        "The launch review is on Friday",
        scope="/proj/a",
        categories=["planning"],
        importance=0.7,
        metadata={"owner": "sam"},
    )
    assert remembered is not None

    matches = memory.recall("The launch review is on Friday", scope="/proj", depth="shallow")

    assert matches
    record = matches[0].record
    assert record.id == remembered.id
    assert record.content == "The launch review is on Friday"
    assert record.scope == "/proj/a"
    assert record.categories == ["planning"]
    assert record.metadata == {"owner": "sam"}


def test_scope_record_follows_the_schema_contract(
    memory: Memory,
    aerospike_server: tuple[str, int, str],
    aerospike_spec: str,
    dump: Callable[[], dict[str, dict[str, Any]]],
) -> None:
    remembered = memory.remember(
        "The launch review is on Friday",
        scope="/proj/a",
        categories=["planning"],
        importance=0.7,
        metadata={"owner": "sam"},
    )
    assert remembered is not None

    host, port, namespace = aerospike_server
    sdk, _ = _sdk()
    session = _get_connection(host, port).session
    scope_set = sdk.DataSet.of(namespace, aerospike_spec.rsplit("=", 1)[1])
    records = dump()

    assert list(records) == ["/proj/a|0"]
    bins = records["/proj/a|0"]
    assert set(bins) == {"scope", "chunk", "anc", "mem", "emb", "cats", "created_at", "bytes"}
    assert bins["scope"] == "/proj/a"
    assert bins["anc"] == ["/", "/proj", "/proj/a"]
    assert bins["cats"] == ["planning"]
    assert set(bins["mem"]) == {remembered.id}
    assert isinstance(bins["emb"][remembered.id], sdk.Vector)
    with session.query(scope_set.id("/proj/a|0")).execute() as stream:
        assert [row.is_ok for row in stream] == [True]


def test_two_memory_instances_share_one_session(
    registered_factory: None,
    aerospike_spec: str,
    embedder: Callable[[list[str]], list[list[float]]],
) -> None:
    first = Memory(storage=aerospike_spec, llm=MagicMock(), embedder=embedder)
    second = Memory(storage=aerospike_spec, llm=MagicMock(), embedder=embedder)
    try:
        assert first._storage._session is second._storage._session
    finally:
        first.close()
        second.close()
