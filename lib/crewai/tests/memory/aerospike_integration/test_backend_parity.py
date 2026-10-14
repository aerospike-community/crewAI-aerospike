"""One contract suite over LanceDB, Qdrant Edge and Aerospike.

Every test runs against each backend and asserts only what the shared
``StorageBackend`` contract promises, by comparing to the expected answer
computed from the test data. Where a sibling backend has a documented quirk
(LanceDB matches scope prefixes by string, deletes root-scope records on a scope
delete, and scores by distance), the data avoids it rather than asserting it.
Scores are never compared across backends because each scores differently.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
import importlib.util
import math
from pathlib import Path
from typing import Any

import pytest

from crewai.memory.storage.aerospike_storage import AerospikeStorage
from crewai.memory.storage.backend import (
    EmbeddingDimensionMismatchError,
    StorageBackend,
)
from crewai.memory.types import MemoryRecord


DIM = 8
BACKENDS = ["lancedb", "qdrant_edge", "aerospike"]

pytestmark = pytest.mark.parametrize("backend", BACKENDS, indirect=True)


@pytest.fixture
def backend(request: pytest.FixtureRequest, tmp_path: Path) -> StorageBackend:
    """The storage backend named by the test parameter, empty and ready."""
    name = request.param
    if name == "lancedb":
        from crewai.memory.storage.lancedb_storage import LanceDBStorage

        return LanceDBStorage(path=str(tmp_path / "lance"), vector_dim=DIM)
    if name == "qdrant_edge":
        if importlib.util.find_spec("qdrant_edge") is None:
            pytest.skip("qdrant-edge-py not installed")
        from crewai.memory.storage.qdrant_edge_storage import QdrantEdgeStorage

        return QdrantEdgeStorage(path=str(tmp_path / "edge"), vector_dim=DIM)
    built = request.getfixturevalue("storage")
    assert isinstance(built, AerospikeStorage)
    return built


def vector(degrees: float, dim: int = DIM) -> list[float]:
    """Unit vector at an angle from the first axis, in the first two dimensions."""
    angle = math.radians(degrees)
    return [math.cos(angle), math.sin(angle)] + [0.0] * (dim - 2)


NOW = datetime(2026, 6, 1, 12, 0, 0)


def record(
    content: str,
    scope: str,
    degrees: float,
    categories: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    age_days: int = 0,
    **fields: Any,
) -> MemoryRecord:
    return MemoryRecord(
        content=content,
        scope=scope,
        embedding=vector(degrees),
        categories=categories or [],
        metadata=metadata or {},
        created_at=NOW - timedelta(days=age_days),
        last_accessed=NOW - timedelta(days=age_days),
        **fields,
    )


@pytest.fixture
def corpus() -> dict[str, MemoryRecord]:
    """Records in scopes that no backend's prefix handling confuses."""
    items = [
        record("s1", "/team/sales", 10, ["deal", "q3"], {"owner": "ann", "n": 1}, 1),
        record("s2", "/team/sales", 20, ["deal"], {"owner": "bob", "n": 2}, 2),
        record("e1", "/team/eng", 30, ["bug"], {"owner": "ann", "n": 3}, 3),
        record("e2", "/team/eng", 40, ["bug", "q3"], {"owner": "bob", "n": 4}, 4),
        record("p1", "/personal", 50, ["note"], {"owner": "ann", "n": 5}, 5),
        record("p2", "/personal", 60, ["note", "deal"], {"owner": "cy", "n": 6}, 6),
    ]
    return {r.content: r for r in items}


@pytest.fixture
def loaded(backend: StorageBackend, corpus: dict[str, MemoryRecord]) -> StorageBackend:
    backend.save(list(corpus.values()))
    return backend


def ids(corpus: dict[str, MemoryRecord], *names: str) -> set[str]:
    return {corpus[name].id for name in names}


def found(results: list[tuple[MemoryRecord, float]]) -> list[str]:
    return [r.id for r, _score in results]


def test_parity_save_get_roundtrip(backend: StorageBackend) -> None:
    original = record(
        "round trip",
        "/a/b",
        15,
        categories=["c1", "c2"],
        metadata={"k": "v", "n": 3},
        importance=0.8,
        source="user-1",
        private=True,
    )
    backend.save([original])

    for fetched in (
        backend.get_record(original.id),
        backend.list_records("/a/b")[0],
    ):
        assert fetched is not None
        assert fetched.id == original.id
        assert fetched.content == "round trip"
        assert fetched.scope == "/a/b"
        assert sorted(fetched.categories) == ["c1", "c2"]
        assert fetched.metadata == {"k": "v", "n": 3}
        assert fetched.importance == pytest.approx(0.8, abs=1e-6)
        assert fetched.source == "user-1"
        assert fetched.private is True
    assert backend.get_record("no-such-id") is None


def test_parity_search_rank_order(
    loaded: StorageBackend, corpus: dict[str, MemoryRecord]
) -> None:
    # The query points along the first axis, so a smaller angle ranks higher.
    expected = [corpus[name].id for name in ("s1", "s2", "e1", "e2", "p1", "p2")]

    assert found(loaded.search(vector(0), limit=6)) == expected
    assert found(loaded.search(vector(0), limit=3)) == expected[:3]
    reverse = [corpus[name].id for name in ("p2", "p1", "e2", "e1", "s2", "s1")]
    assert found(loaded.search(vector(90), limit=6)) == reverse


def test_parity_filters(
    loaded: StorageBackend, corpus: dict[str, MemoryRecord]
) -> None:
    def search(**kwargs: Any) -> set[str]:
        return set(found(loaded.search(vector(0), limit=20, **kwargs)))

    assert search() == ids(corpus, "s1", "s2", "e1", "e2", "p1", "p2")
    assert search(scope_prefix="/team") == ids(corpus, "s1", "s2", "e1", "e2")
    assert search(scope_prefix="/team/sales") == ids(corpus, "s1", "s2")
    assert search(scope_prefix="/personal") == ids(corpus, "p1", "p2")
    assert search(categories=["deal"]) == ids(corpus, "s1", "s2", "p2")
    assert search(categories=["bug", "note"]) == ids(corpus, "e1", "e2", "p1", "p2")
    assert search(metadata_filter={"owner": "ann"}) == ids(corpus, "s1", "e1", "p1")
    assert search(metadata_filter={"owner": "ann", "n": 3}) == ids(corpus, "e1")
    assert search(scope_prefix="/team", categories=["q3"]) == ids(corpus, "s1", "e2")
    assert search(
        scope_prefix="/team", categories=["q3"], metadata_filter={"owner": "bob"}
    ) == ids(corpus, "e2")
    assert search(scope_prefix="/nowhere") == set()

    assert {r.id for r in loaded.list_records("/team", limit=50)} == ids(
        corpus, "s1", "s2", "e1", "e2"
    )


def test_parity_counts_and_scopes(
    loaded: StorageBackend, corpus: dict[str, MemoryRecord]
) -> None:
    assert loaded.count() == 6
    assert loaded.count("/team") == 4
    assert loaded.count("/team/eng") == 2
    assert loaded.count("/nowhere") == 0

    assert loaded.list_scopes("/") == ["/personal", "/team"]
    assert loaded.list_scopes("/team") == ["/team/eng", "/team/sales"]
    assert loaded.list_scopes("/personal") == []

    info = loaded.get_scope_info("/team")
    assert info.path == "/team"
    assert info.record_count == 4
    assert info.categories == ["bug", "deal", "q3"]
    assert info.child_scopes == ["/team/eng", "/team/sales"]
    assert info.oldest_record is not None and info.newest_record is not None
    assert info.oldest_record < info.newest_record
    assert loaded.get_scope_info("/").record_count == 6
    assert loaded.get_scope_info("/nowhere").record_count == 0

    assert loaded.list_categories() == {"deal": 3, "q3": 2, "bug": 2, "note": 2}
    assert loaded.list_categories("/team/sales") == {"deal": 2, "q3": 1}


def test_parity_delete_and_reset(
    loaded: StorageBackend, corpus: dict[str, MemoryRecord]
) -> None:
    def remaining() -> set[str]:
        return {r.id for r in loaded.list_records(limit=100)}

    assert loaded.delete(record_ids=[corpus["p2"].id]) == 1
    assert remaining() == ids(corpus, "s1", "s2", "e1", "e2", "p1")
    assert loaded.get_record(corpus["p2"].id) is None

    assert loaded.delete(scope_prefix="/personal") == 1
    assert remaining() == ids(corpus, "s1", "s2", "e1", "e2")

    assert loaded.delete(categories=["q3"]) == 2
    assert remaining() == ids(corpus, "s2", "e1")

    assert loaded.delete(metadata_filter={"owner": "bob"}) == 1
    assert remaining() == ids(corpus, "e1")
    assert loaded.delete(record_ids=["no-such-id"]) == 0

    loaded.save([corpus["s1"], corpus["s2"]])
    loaded.reset("/team/sales")
    assert remaining() == ids(corpus, "e1")
    assert loaded.list_scopes("/team") == ["/team/eng"]

    loaded.reset()
    assert loaded.count() == 0
    assert loaded.list_scopes("/") == []
    assert loaded.search(vector(0)) == []

    loaded.save([corpus["s1"]])  # usable after a full reset
    assert found(loaded.search(vector(0))) == [corpus["s1"].id]


def test_parity_empty_record_ids_are_ignored(
    loaded: StorageBackend, corpus: dict[str, MemoryRecord]
) -> None:
    assert loaded.delete(record_ids=[], scope_prefix="/personal") == 2
    assert {record.id for record in loaded.list_records(limit=100)} == ids(
        corpus, "s1", "s2", "e1", "e2"
    )
    assert loaded.delete(record_ids=[]) == 4
    assert loaded.count() == 0


def test_parity_dimension_mismatch(backend: StorageBackend) -> None:
    backend.save([record("first", "/a", 10)])

    stray = MemoryRecord(content="stray", scope="/a", embedding=[0.1] * (DIM // 2))
    with pytest.raises(EmbeddingDimensionMismatchError):
        backend.save([stray])
    with pytest.raises(EmbeddingDimensionMismatchError):
        backend.save([record("ok", "/a", 20), stray])
    with pytest.raises(EmbeddingDimensionMismatchError):
        backend.search([0.1] * (DIM // 2))

    assert backend.count() == 1


def test_parity_list_records_newest_first(backend: StorageBackend) -> None:
    oldest = record("old", "/a", 10, age_days=30)
    middle = record("mid", "/a", 20, age_days=10)
    newest = record("new", "/a", 30, age_days=1)
    backend.save([middle, newest, oldest])

    order = [oldest.id, middle.id, newest.id][::-1]
    assert [r.id for r in backend.list_records(limit=10)] == order
    assert [r.id for r in backend.list_records("/a", limit=10)] == order
    assert [r.id for r in backend.list_records(limit=2, offset=1)] == order[1:]


def test_parity_update_replaces_a_record(
    loaded: StorageBackend, corpus: dict[str, MemoryRecord]
) -> None:
    changed = corpus["s1"].model_copy(
        update={
            "content": "s1 changed",
            "scope": "/team/archive",
            "categories": ["won"],
            "embedding": vector(85),
        }
    )

    loaded.update(changed)

    fetched = loaded.get_record(changed.id)
    assert fetched is not None
    assert (fetched.content, fetched.scope, fetched.categories) == (
        "s1 changed",
        "/team/archive",
        ["won"],
    )
    assert loaded.count() == 6
    assert loaded.list_scopes("/team") == ["/team/archive", "/team/eng", "/team/sales"]
    assert found(loaded.search(vector(90), limit=1)) == [changed.id]


def test_parity_touch_records_updates_last_accessed(
    loaded: StorageBackend, corpus: dict[str, MemoryRecord]
) -> None:
    touch: Callable[[list[str]], None] | None = getattr(loaded, "touch_records", None)
    assert touch is not None
    before = corpus["s1"].last_accessed

    touch([corpus["s1"].id])

    touched = loaded.get_record(corpus["s1"].id)
    untouched = loaded.get_record(corpus["s2"].id)
    assert touched is not None and untouched is not None
    assert touched.last_accessed > before + timedelta(days=30)
    assert untouched.last_accessed == corpus["s2"].last_accessed
