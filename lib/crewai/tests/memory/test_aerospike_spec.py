"""Tests for the Aerospike storage spec grammar and factory fall-through.

None of these tests need a server: the storage class is replaced by a fake
wherever a storage would otherwise be built.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import crewai.memory.storage.aerospike_storage as aerospike_storage
import crewai.memory.storage.factory as factory
from crewai.memory.storage.aerospike_storage import (
    AerospikeSpec,
    aerospike_storage_factory,
    parse_aerospike_spec,
)
from crewai.memory.unified_memory import Memory


@pytest.fixture(autouse=True)
def reset_factory():
    """Reset the process-wide factory around each test."""
    original = factory._factory
    factory.set_memory_storage_factory(None)
    yield
    factory.set_memory_storage_factory(original)


class RecordingStorage:
    """Stands in for AerospikeStorage so no connection is ever opened."""

    def __init__(self, spec: AerospikeSpec) -> None:
        self.spec = spec


@pytest.fixture
def fake_storage(monkeypatch: pytest.MonkeyPatch) -> type[RecordingStorage]:
    monkeypatch.setattr(aerospike_storage, "AerospikeStorage", RecordingStorage)
    return RecordingStorage


def test_parse_valid_spec() -> None:
    spec = parse_aerospike_spec("aerospike://localhost:3000/test")

    assert spec == AerospikeSpec("localhost", 3000, "test", "crewai_memory")


def test_parse_set_prefix_and_ipv6() -> None:
    assert parse_aerospike_spec("aerospike://h:3000/test?set_prefix=agent_mem") == (
        AerospikeSpec("h", 3000, "test", "agent_mem")
    )
    assert parse_aerospike_spec("aerospike://[::1]:3000/test").host == "::1"
    assert parse_aerospike_spec("AEROSPIKE://h:3000/test").host == "h"


@pytest.mark.parametrize(
    "spec",
    [
        "aerospike://h/test",
        "aerospike://h:/test",
        "aerospike://h:0/test",
        "aerospike://h:65536/test",
        "aerospike://h:abc/test",
        "aerospike://h:-1/test",
        "aerospike://[::1]/test",
    ],
)
def test_parse_rejects_bad_port(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_aerospike_spec(spec)


@pytest.mark.parametrize(
    "spec",
    [
        "aerospike://h:3000",
        "aerospike://h:3000/",
        "aerospike://h:3000/" + "n" * 32,
        "aerospike://h:3000/te;st",
        "aerospike://h:3000/te:st",
        "aerospike://h:3000/te/st",
        "aerospike://h:3000/../x",
        "aerospike://h:3000/..",
        "aerospike://h:3000/test/",
    ],
)
def test_parse_rejects_bad_namespace(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_aerospike_spec(spec)


def test_parse_accepts_longest_namespace() -> None:
    assert parse_aerospike_spec("aerospike://h:3000/" + "n" * 31).namespace == "n" * 31


@pytest.mark.parametrize("spec", ["aerospike://:3000/test", "aerospike://[::zz]:3000/test"])
def test_parse_rejects_bad_host(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_aerospike_spec(spec)


def test_parse_rejects_credentials_without_echo() -> None:
    spec = "aerospike://alice:s3cr3tpw@localhost:3000/test"

    with pytest.raises(ValueError) as excinfo:
        parse_aerospike_spec(spec)

    message = str(excinfo.value)
    assert "credentials" in message
    assert "alice" not in message
    assert "s3cr3tpw" not in message
    assert spec not in message


@pytest.mark.parametrize(
    ("key", "value"),
    [("user", "alice"), ("password", "s3cr3tpw"), ("token", "tok3n123")],
)
def test_parse_rejects_unknown_query_keys(key: str, value: str) -> None:
    with pytest.raises(ValueError) as excinfo:
        parse_aerospike_spec(f"aerospike://h:3000/test?{key}={value}")

    message = str(excinfo.value)
    assert f"'{key}'" in message
    assert value not in message


def test_parse_rejects_bad_set_prefix() -> None:
    for query in ("set_prefix=", "set_prefix=a-b", "set_prefix=" + "a" * 41):
        with pytest.raises(ValueError):
            parse_aerospike_spec(f"aerospike://h:3000/test?{query}")
    with pytest.raises(ValueError):
        parse_aerospike_spec("aerospike://h:3000/test?set_prefix=a&set_prefix=b")


@pytest.mark.parametrize(
    "spec",
    [
        "aerospike://h:3000/test#frag",
        "aerospike://h:3000/te%73t",
        "aerospike://h:3000/te st",
        "aerospike://h:3000/test\n",
        "aerospike://h:3000/te\x00st",
        "aerospike://h:3000/te\x7fst",
        "aerospike://h:3000/test?set_prefix=" + "a" * 600,
        "aerospike://" + "h" * 600 + ":3000/test",
    ],
)
def test_parse_rejects_hostile_input(spec: str) -> None:
    with pytest.raises(ValueError) as excinfo:
        parse_aerospike_spec(spec)

    assert spec not in str(excinfo.value)


def test_parse_ignores_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AEROSPIKE_HOST", "env-host")
    monkeypatch.setenv("AEROSPIKE_PORT", "9999")

    spec = parse_aerospike_spec("aerospike://spec-host:3000/test")

    assert (spec.host, spec.port) == ("spec-host", 3000)


@pytest.mark.parametrize(
    "spec",
    ["lancedb", "qdrant-edge", "/some/dir", "", "aerospike", "aerospikex://h:1/ns"],
)
def test_factory_returns_none_for_other_specs(
    spec: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(_: str) -> Any:
        raise AssertionError("the spec must not be parsed")

    monkeypatch.setattr(aerospike_storage, "parse_aerospike_spec", fail)

    assert aerospike_storage_factory(spec) is None


def test_factory_raises_for_malformed_aerospike_spec() -> None:
    with pytest.raises(ValueError):
        aerospike_storage_factory("aerospike://:3000/test")


def test_factory_builds_storage_for_valid_spec(
    fake_storage: type[RecordingStorage],
) -> None:
    storage = aerospike_storage_factory("aerospike://h:3000/test?set_prefix=mem")

    assert isinstance(storage, fake_storage)
    assert storage.spec == AerospikeSpec("h", 3000, "test", "mem")


def test_registered_factory_falls_through_to_lancedb(
    fake_storage: type[RecordingStorage], tmp_path: Path
) -> None:
    factory.set_memory_storage_factory(aerospike_storage_factory)

    builtin = Memory(storage=str(tmp_path / "db"))
    routed = Memory(storage="aerospike://localhost:3000/test")

    assert type(builtin._storage).__name__ == "LanceDBStorage"
    assert isinstance(routed._storage, fake_storage)
