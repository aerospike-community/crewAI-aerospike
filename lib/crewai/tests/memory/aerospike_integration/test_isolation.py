"""The integration fixtures never touch a set that is not a test set."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from crewai.memory.storage.aerospike_storage import _get_connection, _sdk


def test_truncation_refuses_sets_outside_the_test_prefix(
    aerospike_server: tuple[str, int, str],
    truncate_test_set: Callable[[Any, str, str], None],
) -> None:
    host, port, namespace = aerospike_server
    sdk, _ = _sdk()
    session = _get_connection(host, port).session
    decoy = sdk.DataSet.of(namespace, "crewai_decoy_guard")
    session.upsert(decoy.id("keep")).bin("v").set_to(1).execute().close()
    try:
        with pytest.raises(AssertionError, match="Refusing to truncate"):
            truncate_test_set(session, namespace, "crewai_decoy_guard")
        with session.query(decoy.id("keep")).execute() as stream:
            rows = list(stream)
        assert len(rows) == 1 and rows[0].is_ok and rows[0].record is not None
        assert rows[0].record.bins["v"] == 1
    finally:
        session.delete(decoy.id("keep")).execute().close()


def test_spec_uses_a_test_set_prefix(aerospike_spec: str) -> None:
    assert "set_prefix=t_" in aerospike_spec
