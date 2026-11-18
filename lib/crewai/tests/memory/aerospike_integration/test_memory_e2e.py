"""End-to-end tests that drive ``Memory`` itself over a live Aerospike server.

Everything goes through the public ``Memory`` API with the storage factory
registered, so these prove the backend is called the way the host framework
calls it. Explicit scope, categories and importance mean no model call happens.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import hashlib
from pathlib import Path
import subprocess
import sys
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

import crewai.memory.storage.aerospike_storage as aerospike_storage
from crewai.memory.storage.aerospike_storage import AerospikeStorage
from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.memory.unified_memory import Memory


Embedder = Callable[[list[str]], list[list[float]]]

SALES = [
    "Acme renewal closes at the end of the quarter",
    "The Globex deal needs legal review first",
    "Initech asked for a volume discount",
]
ENG = [
    "The login page crashes on empty passwords",
    "Nightly build fails on the arm runner",
]
PERSONAL = [
    "Dentist appointment moved to Thursday",
    "Renew the passport before spring",
]


@pytest.fixture
def embedder() -> Embedder:
    """Deterministic embedder with non-negative components.

    Every pair of texts then has a cosine similarity of at least zero, so the
    default ``min_score`` of recall never hides a memory, and the 64 dimensions
    keep distinct texts well apart, so nothing is treated as a near duplicate.
    """

    def embed(texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            seed = int.from_bytes(
                hashlib.sha256(text.encode("utf-8")).digest()[:8], "big"
            )
            vector = np.abs(np.random.default_rng(seed).normal(size=64))
            out.append((vector / np.linalg.norm(vector)).tolist())
        return out

    return embed


@pytest.fixture
def make_memory(
    registered_factory: None, aerospike_spec: str, embedder: Embedder
) -> Iterator[Callable[[], Memory]]:
    """Build Memory objects over this worker's spec and close them afterwards."""
    built: list[Memory] = []

    def build() -> Memory:
        memory = Memory(storage=aerospike_spec, llm=MagicMock(), embedder=embedder)
        built.append(memory)
        return memory

    yield build
    for memory in built:
        memory.close()


def _remember(
    memory: Memory,
    contents: list[str],
    scope: str,
    categories: list[str],
    **fields: Any,
) -> list[str]:
    ids = []
    for content in contents:
        record = memory.remember(
            content, scope=scope, categories=categories, importance=0.5, **fields
        )
        assert record is not None
        ids.append(record.id)
    return ids


def _top(memory: Memory, query: str, **kwargs: Any) -> str:
    matches = memory.recall(query, depth="shallow", **kwargs)
    assert matches, f"no match for {query!r}"
    return matches[0].record.content


def test_memory_end_to_end_through_registered_factory(
    make_memory: Callable[[], Memory],
) -> None:
    memory = make_memory()
    assert isinstance(memory._storage, AerospikeStorage)

    sales_ids = _remember(memory, SALES[:2], "/team/sales", ["deal"])
    sales_ids += _remember(memory, SALES[2:], "/team/sales", ["deal", "discount"])
    eng_ids = _remember(memory, ENG, "/team/eng", ["bug"])
    personal_ids = _remember(memory, PERSONAL, "/personal", ["note"])

    # Counts and listings agree.
    assert memory.info("/").record_count == 7
    assert len(memory.list_records(limit=100)) == 7
    assert memory.list_scopes("/") == ["/personal", "/team"]
    assert memory.list_scopes("/team") == ["/team/eng", "/team/sales"]
    assert memory.info("/team").record_count == 5
    assert memory.info("/team/sales").record_count == 3
    assert memory.info("/team").child_scopes == ["/team/eng", "/team/sales"]
    assert memory.list_categories() == {"deal": 3, "discount": 1, "bug": 2, "note": 2}
    assert memory.list_categories("/team/sales") == {"deal": 3, "discount": 1}
    assert "/team/sales (3 records)" in memory.tree("/team")

    # Recall: ranking, scope, category and limit.
    for content in SALES + ENG + PERSONAL:
        assert _top(memory, content) == content
    assert _top(memory, SALES[0], scope="/team/sales") == SALES[0]
    scoped = memory.recall(SALES[0], scope="/team", depth="shallow", limit=10)
    assert {m.record.id for m in scoped} == set(sales_ids + eng_ids)
    assert len(memory.recall(SALES[0], depth="shallow", limit=3)) == 3
    only_bugs = memory.recall(ENG[0], categories=["bug"], depth="shallow")
    assert {m.record.id for m in only_bugs} == set(eng_ids)
    only_discount = memory.recall(
        SALES[2], scope="/team", categories=["discount"], depth="shallow"
    )
    assert [m.record.content for m in only_discount] == [SALES[2]]
    scores = [m.score for m in memory.recall(ENG[0], depth="shallow", limit=7)]
    assert scores == sorted(scores, reverse=True)

    # Update content, categories and importance, then move a memory.
    target = sales_ids[0]
    updated = memory.update(
        target,
        content="Acme signed the renewal",
        categories=["deal", "won"],
        importance=0.9,
    )
    assert updated.content == "Acme signed the renewal"
    assert _top(memory, "Acme signed the renewal") == "Acme signed the renewal"
    assert memory.info("/").record_count == 7
    assert memory.list_categories("/team/sales")["won"] == 1
    memory.update(target, scope="/team/archive")
    assert memory.list_scopes("/team") == ["/team/archive", "/team/eng", "/team/sales"]
    assert memory.info("/team/archive").record_count == 1
    assert memory.info("/team/sales").record_count == 2

    # Forget a scope, then by ids.
    assert memory.forget(scope="/team/eng") == 2
    assert memory.info("/").record_count == 5
    assert memory.list_scopes("/team") == ["/team/archive", "/team/sales"]
    assert memory.forget(record_ids=personal_ids[:1]) == 1
    assert memory.info("/personal").record_count == 1
    assert len(memory.list_records(limit=100)) == 4

    # Reset empties the store; it stays usable afterwards.
    memory.reset()
    assert memory.info("/").record_count == 0
    assert memory.list_scopes("/") == []
    assert memory.recall(SALES[1], depth="shallow") == []
    _remember(memory, SALES[:1], "/again", ["deal"])
    assert _top(memory, SALES[0]) == SALES[0]


def test_other_specs_still_fall_through_with_factory_registered(
    registered_factory: None, embedder: Embedder, tmp_path: Path
) -> None:
    memory = Memory(storage=str(tmp_path / "db"), llm=MagicMock(), embedder=embedder)
    try:
        assert isinstance(memory._storage, LanceDBStorage)
    finally:
        memory.close()


def test_second_memory_sees_data_written_by_the_first(
    make_memory: Callable[[], Memory], monkeypatch: pytest.MonkeyPatch
) -> None:
    first = make_memory()
    _remember(first, SALES, "/team/sales", ["deal"])
    first.close()

    # A fresh process: no cached client and no cached "indexes created" state,
    # so the second Memory reconnects and creates indexes that already exist.
    aerospike_storage.shutdown_connections()
    monkeypatch.setattr(aerospike_storage, "_indexes_ready", set())

    second = make_memory()
    assert second.info("/team/sales").record_count == 3
    assert _top(second, SALES[1]) == SALES[1]
    _remember(second, ENG[:1], "/team/eng", ["bug"])
    assert second.info("/").record_count == 4


@pytest.mark.asyncio
async def test_async_memory_methods(make_memory: Callable[[], Memory]) -> None:
    memory = make_memory()

    record = await memory.aremember(
        SALES[0], scope="/team/sales", categories=["deal"], importance=0.5
    )
    assert record is not None
    await memory.aremember_many(
        ENG, scope="/team/eng", categories=["bug"], importance=0.5
    )

    matches = await memory.arecall(ENG[1], depth="shallow", limit=5)
    assert matches[0].record.content == ENG[1]
    assert {m.record.content for m in matches} == {SALES[0], *ENG}
    scoped = await memory.arecall(SALES[0], scope="/team/sales", depth="shallow")
    assert [m.record.id for m in scoped] == [record.id]
    assert memory.info("/").record_count == 3


@pytest.mark.asyncio
async def test_memory_can_be_built_and_used_inside_a_running_event_loop(
    make_memory: Callable[[], Memory], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force a real connect and index creation inside the loop.
    aerospike_storage.shutdown_connections()
    monkeypatch.setattr(aerospike_storage, "_indexes_ready", set())

    memory = make_memory()
    await memory.aremember(SALES[0], scope="/s", categories=["deal"], importance=0.5)

    assert memory.info("/s").record_count == 1
    assert memory.list_scopes("/") == ["/s"]
    assert [
        m.record.content for m in await memory.arecall(SALES[0], depth="shallow")
    ] == [SALES[0]]


_BUILD_ON_A_SHORT_LIVED_THREAD = """
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from crewai.memory.storage.aerospike_storage import aerospike_storage_factory
from crewai.memory.types import MemoryRecord

built = []
builder = threading.Thread(target=lambda: built.append(aerospike_storage_factory(sys.argv[1])))
builder.start()
builder.join()
storage = built[0]
storage.save([MemoryRecord(content="x", scope="/t", embedding=[1.0, 0.0])])
with ThreadPoolExecutor(max_workers=1) as pool:
    assert pool.submit(storage.count).result() == 1
    assert pool.submit(storage.list_scopes, "/").result() == ["/t"]
"""


def test_storage_built_on_a_short_lived_thread_is_usable_from_other_threads(
    aerospike_spec: str,
) -> None:
    # A client created on a thread that then exits used to crash the whole
    # interpreter (a segmentation fault, not an exception) when another thread
    # scanned a set. That only reproduces reliably in a fresh process.
    done = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-I",
            "-W",
            "ignore",
            "-c",
            _BUILD_ON_A_SHORT_LIVED_THREAD,
            aerospike_spec,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert done.returncode == 0, done.stderr[-2000:]


def test_remember_many_then_recall_drains_background_saves(
    make_memory: Callable[[], Memory],
) -> None:
    memory = make_memory()
    contents = SALES + ENG + PERSONAL[:1]

    memory.remember_many(contents, scope="/batch", categories=["x"], importance=0.5)
    matches = memory.recall(contents[0], scope="/batch", depth="shallow", limit=50)

    assert {m.record.content for m in matches} == set(contents)
    assert memory.info("/batch").record_count == len(contents)


def test_scope_view_and_slice_over_aerospike(make_memory: Callable[[], Memory]) -> None:
    memory = make_memory()
    _remember(memory, SALES, "/team/sales", ["deal"])
    _remember(memory, ENG, "/team/eng", ["bug"])
    _remember(memory, PERSONAL, "/personal", ["note"])

    team = memory.scope("/team")
    assert {m.record.content for m in team.recall(SALES[0], depth="shallow")} == set(
        SALES + ENG
    )
    assert team.list_scopes("/") == ["/team/eng", "/team/sales"]
    assert team.info("/").record_count == 5
    record = team.remember(
        "Hiring plan for next year", scope="/hr", categories=["plan"], importance=0.5
    )
    assert record is not None and record.scope == "/team/hr"
    assert memory.info("/team/hr").record_count == 1
    assert memory.info("/personal").record_count == 2

    view = memory.slice(["/team/eng", "/personal"], read_only=True)
    seen = {m.record.content for m in view.recall(ENG[0], depth="shallow", limit=20)}
    assert seen == set(ENG + PERSONAL)


def test_hostile_scope_and_category_strings_through_memory(
    make_memory: Callable[[], Memory],
) -> None:
    memory = make_memory()
    scope = "/o'brien;drop/café/中文"
    categories = ['it\'s; "quoted"', "café", "中文", "a|b"]

    record = memory.remember(
        "Hostile strings are data, never syntax",
        scope=scope,
        categories=categories,
        metadata={'k"ey': "v'al;ue"},
        importance=0.5,
    )
    assert record is not None
    _remember(memory, ["A plain neighbour"], "/plain", ["ok"])

    assert memory.info(scope).record_count == 1
    assert memory.list_scopes("/") == ["/o'brien;drop", "/plain"]
    for category in categories:
        matches = memory.recall(
            "Hostile strings are data, never syntax",
            scope="/o'brien;drop",
            categories=[category],
            depth="shallow",
        )
        assert [m.record.id for m in matches] == [record.id]
    assert set(memory.list_categories(scope)) == set(categories)
    assert memory.forget(scope="/o'brien;drop") == 1
    assert memory.info("/").record_count == 1
