"""Ranking, filters, caps, async variants and the protocol surface."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import inspect
import logging
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

import crewai.memory.storage.aerospike_storage as aerospike_module
from crewai.memory.storage.aerospike_storage import AerospikeStorage
from crewai.memory.storage.backend import EmbeddingDimensionMismatchError, StorageBackend
from crewai.memory.types import MemoryRecord
from crewai.memory.unified_memory import Memory


Make = Callable[..., MemoryRecord]

QUERY = [1.0, 0.0, 0.0, 0.0]
# Embedding -> hand-computed cosine similarity with QUERY.
KNOWN = {
    "same": ([1.0, 0.0, 0.0, 0.0], 1.0),
    "diagonal": ([1.0, 1.0, 0.0, 0.0], 1.0 / np.sqrt(2.0)),
    "tilted": ([3.0, 4.0, 0.0, 0.0], 0.6),
    "orthogonal": ([0.0, 1.0, 0.0, 0.0], 0.0),
    "opposite": ([-2.0, 0.0, 0.0, 0.0], -1.0),
}


def _seed_known(storage: AerospikeStorage, make: Make) -> None:
    storage.save([make(name, "/k", emb) for name, (emb, _score) in KNOWN.items()])


def test_search_orders_by_cosine(storage: AerospikeStorage, make: Make) -> None:
    _seed_known(storage, make)

    results = storage.search(QUERY, min_score=-1.0)

    assert [r.content for r, _ in results] == [
        "same", "diagonal", "tilted", "orthogonal", "opposite",
    ]  # fmt: skip
    for record, score in results:
        assert score == pytest.approx(KNOWN[record.content][1], abs=1e-6)
        assert record.embedding is not None
        np.testing.assert_allclose(record.embedding, KNOWN[record.content][0], atol=1e-6)


def test_search_limit_and_min_score_inclusive(
    storage: AerospikeStorage, make: Make
) -> None:
    _seed_known(storage, make)

    assert [r.content for r, _ in storage.search(QUERY, limit=2, min_score=-1.0)] == [
        "same",
        "diagonal",
    ]
    # The default threshold of 0.0 keeps the orthogonal memory and drops the opposite.
    assert {r.content for r, _ in storage.search(QUERY)} == {
        "same", "diagonal", "tilted", "orthogonal",
    }  # fmt: skip

    tilted_score = next(s for r, s in storage.search(QUERY) if r.content == "tilted")
    kept = storage.search(QUERY, min_score=tilted_score)
    dropped = storage.search(QUERY, min_score=tilted_score + 1e-9)
    assert [r.content for r, _ in kept] == ["same", "diagonal", "tilted"]
    assert [r.content for r, _ in dropped] == ["same", "diagonal"]


def test_search_filters(storage: AerospikeStorage, make: Make) -> None:
    storage.save(
        [
            make("a-x1", "/a", [1, 0, 0, 0], categories=["x"], metadata={"k": 1}),
            make("a-y1", "/a", [1, 1, 0, 0], categories=["y"], metadata={"k": 1}),
            make("ab-x1", "/a/b", [1, 2, 0, 0], categories=["x"], metadata={"k": 2}),
            make("sib-x1", "/ab", [1, 3, 0, 0], categories=["x"], metadata={"k": 1}),
        ]
    )

    def names(**kwargs: Any) -> list[str]:
        return [r.content for r, _ in storage.search(QUERY, min_score=-1, **kwargs)]

    assert names(scope_prefix="/a") == ["a-x1", "a-y1", "ab-x1"]
    assert names(scope_prefix="/a/b") == ["ab-x1"]
    assert names(categories=["x"]) == ["a-x1", "ab-x1", "sib-x1"]
    assert names(categories=["x", "y"]) == ["a-x1", "a-y1", "ab-x1", "sib-x1"]
    assert names(metadata_filter={"k": 1}) == ["a-x1", "a-y1", "sib-x1"]
    assert names(scope_prefix="/a", categories=["x"]) == ["a-x1", "ab-x1"]
    assert names(scope_prefix="/a", categories=["x"], metadata_filter={"k": 1}) == ["a-x1"]
    assert names(categories=["nope"]) == []


def test_filtered_search_reaches_below_the_top_hits(
    storage: AerospikeStorage, make: Make
) -> None:
    records = []
    for i in range(30):
        angle = i * 0.05  # the first memories are the closest to the query
        records.append(
            make(
                f"m{i:02d}",
                "/deep",
                [float(np.cos(angle)), float(np.sin(angle)), 0, 0],
                categories=["near"] if i < 20 else ["far"],
            )
        )
    storage.save(records)

    results = storage.search(QUERY, categories=["far"], limit=3)

    assert [r.content for r, _ in results] == ["m20", "m21", "m22"]


def test_search_skips_records_without_embeddings_and_scores_zero_norm_as_zero(
    storage: AerospikeStorage, make: Make
) -> None:
    storage.save(
        [
            make("embedded", "/z", [1, 0, 0, 0]),
            make("no-vector", "/z", None),
            make("zero", "/z", [0, 0, 0, 0]),
        ]
    )

    results = storage.search(QUERY, min_score=-1.0)

    assert [(r.content, s) for r, s in results] == [("embedded", 1.0), ("zero", 0.0)]
    assert storage.search([0, 0, 0, 0], min_score=-1.0)[0][1] == 0.0


def test_search_dimension_mismatch_raises(
    storage: AerospikeStorage, make: Make
) -> None:
    assert storage.search(QUERY) == []  # an empty store has nothing to rank
    storage.save([make("one", "/a", [1, 0, 0, 0])])

    with pytest.raises(EmbeddingDimensionMismatchError) as caught:
        storage.search([1.0] * 8)

    assert (caught.value.stored_dim, caught.value.new_dim) == (4, 8)
    assert storage.search([]) == []


def test_async_variants_match_sync(storage: AerospikeStorage, make: Make) -> None:
    records = [make(name, "/k", emb) for name, (emb, _s) in KNOWN.items()]

    async def scenario() -> tuple[list[tuple[MemoryRecord, float]], int]:
        await storage.asave(records)
        found = await storage.asearch(QUERY, scope_prefix="/k", min_score=-1.0)
        removed = await storage.adelete(record_ids=[records[0].id])
        return found, removed

    # The sync view right after asave must equal what asearch returned.
    async def saved_then_search() -> list[tuple[MemoryRecord, float]]:
        await storage.asave(records)
        return await storage.asearch(QUERY, scope_prefix="/k", min_score=-1.0)

    via_async = asyncio.run(saved_then_search())
    via_sync = storage.search(QUERY, scope_prefix="/k", min_score=-1.0)
    assert [(r.id, s) for r, s in via_async] == [(r.id, s) for r, s in via_sync]
    assert len(via_async) == 5

    _, removed = asyncio.run(scenario())
    assert removed == 1
    assert storage.count("/k") == 4
    assert asyncio.run(storage.adelete(scope_prefix="/k")) == 4
    assert storage.count() == 0


def test_protocol_surface_complete(storage: AerospikeStorage) -> None:
    assert isinstance(storage, StorageBackend)
    methods = [
        "save", "search", "delete", "update", "get_record", "list_records",
        "get_scope_info", "list_scopes", "list_categories", "count", "reset",
        "asave", "asearch", "adelete",
    ]  # fmt: skip
    for name in methods:
        assert callable(getattr(storage, name)), name
    for name in ("asave", "asearch", "adelete"):
        assert inspect.iscoroutinefunction(getattr(storage, name)), name
    assert callable(storage.touch_records)
    assert callable(storage.close)


def test_search_through_memory_recall_shallow(
    registered_factory: None,
    aerospike_spec: str,
    embedder: Callable[[list[str]], list[list[float]]],
) -> None:
    memory = Memory(storage=aerospike_spec, llm=MagicMock(), embedder=embedder)
    try:
        texts = [
            "The launch review is on Friday",
            "Quarterly budget is under review",
            "The office plants need watering",
        ]
        for text in texts:
            memory.remember(text, scope="/proj", categories=["note"], importance=0.5)

        matches = memory.recall("The launch review is on Friday", scope="/proj", depth="shallow")

        assert matches[0].record.content == texts[0]
        # Unrelated memories rank below it, and only those with a score >= 0 return.
        assert {m.record.content for m in matches} <= set(texts)
    finally:
        memory.close()


# -- caps -------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, True, 2.5, None])
def test_search_and_list_reject_a_non_positive_limit(
    storage: AerospikeStorage, bad: Any
) -> None:
    with pytest.raises(ValueError, match="limit"):
        storage.search(QUERY, limit=bad)
    with pytest.raises(ValueError, match="limit"):
        storage.list_records(limit=bad)
    with pytest.raises(ValueError, match="offset"):
        storage.list_records(offset=-1)


def test_limit_is_clamped_to_the_cap(
    storage: AerospikeStorage, make: Make, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage.save([make(f"m{i}", "/c", [1, 0, 0, 0]) for i in range(8)])
    monkeypatch.setattr(aerospike_module, "_MAX_LIMIT", 3)

    assert len(storage.search(QUERY, limit=10**9)) == 3
    assert len(storage.list_records(limit=10**9)) == 3


def test_search_reads_at_most_the_entry_cap(
    storage: AerospikeStorage,
    make: Make,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    storage.save([make(f"m{i}", f"/s{i % 3}", [1, 0, 0, 0]) for i in range(12)])
    monkeypatch.setattr(aerospike_module, "_SCAN_ENTRIES_LIMIT", 5)

    with caplog.at_level(logging.WARNING, logger=aerospike_module.__name__):
        results = storage.search(QUERY, limit=100)
        listed = storage.list_records(limit=100)
        info = storage.get_scope_info("/")

    assert len(results) == 5
    assert len(listed) == 5
    assert info.record_count == 5
    assert "narrow the scope" in caplog.text


def test_scans_stop_at_the_record_cap(
    storage: AerospikeStorage,
    make: Make,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    storage.save([make(f"m{i}", f"/r{i}", [1, 0, 0, 0]) for i in range(6)])
    monkeypatch.setattr(aerospike_module, "_SCAN_RECORDS_LIMIT", 2)

    with caplog.at_level(logging.WARNING, logger=aerospike_module.__name__):
        assert len(storage.list_scopes("/")) == 2
        assert len(storage.search(QUERY, limit=100)) == 2
    assert "scope records" in caplog.text


def test_oversized_inputs_are_rejected_before_any_write(
    storage: AerospikeStorage, make: Make, dump: Callable[[], dict[str, Any]]
) -> None:
    too_long_scope = "/" + "s" * 600
    too_deep_scope = "/" + "/".join("d" for _ in range(40))
    too_big = "c" * 600_000
    cases = [
        make("scope", too_long_scope, [1, 0, 0, 0]),
        make("deep", too_deep_scope, [1, 0, 0, 0]),
        make("category", "/a", [1, 0, 0, 0], categories=["k" * 2_000]),
        make("id", "/a", [1, 0, 0, 0], record_id="i" * 2_000),
        make(too_big, "/a", [1, 0, 0, 0]),
    ]

    for record in cases:
        with pytest.raises(ValueError):
            storage.save([record])

    assert dump() == {}
    # A search prefix longer than any stored scope simply matches nothing.
    assert storage.search(QUERY, scope_prefix=too_long_scope) == []


def test_chunk_limit_is_validated(aerospike_spec: str) -> None:
    spec = aerospike_module.parse_aerospike_spec(aerospike_spec)
    for bad in (0, -5, aerospike_module._MAX_CHUNK_LIMIT_BYTES + 1):
        with pytest.raises(ValueError, match="chunk_limit_bytes"):
            AerospikeStorage(spec, chunk_limit_bytes=bad)
