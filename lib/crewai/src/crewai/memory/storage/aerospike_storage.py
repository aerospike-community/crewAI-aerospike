"""Aerospike storage backend for the unified memory system.

Register it once at application startup, then select it with an
``aerospike://host:port/namespace`` storage spec::

    from crewai.memory import Memory
    from crewai.memory.storage.aerospike_storage import aerospike_storage_factory
    from crewai.memory.storage.factory import set_memory_storage_factory

    set_memory_storage_factory(aerospike_storage_factory)
    memory = Memory(storage="aerospike://127.0.0.1:3000/test")

Every other storage spec is left to the built-in backends. Each logical scope
is stored as one record whose map bins hold the memory entries, and embeddings
are stored as native vector values inside that record. A scope gets a further
record (a chunk) only when the first one nears its size limit.

Cost model. Search is a brute-force scan and rank: there is no
approximate-nearest-neighbor index and the server does not rank vectors nested
in a map. Each query narrows the candidate records on the server (by scope
prefix through a secondary index, and by category), then reads the embeddings of
every candidate memory and computes cosine similarity on the client. Bytes moved
and compute grow linearly with candidate count times embedding dimension, so
scoping a query with a scope prefix is the main lever.

Bounds. Every call is bounded so one request cannot exhaust the server or the
client:

* ``limit`` must be a positive integer and is clamped to ``_MAX_LIMIT`` (1000).
* A search or listing reads at most ``_SCAN_ENTRIES_LIMIT`` (50,000) memories
  and ``_SCAN_RECORDS_LIMIT`` (100,000) scope records. Past the bound it stops
  reading, logs a warning and works with what it read (truncation, never an
  error), so narrow the scope to get complete results.
* ``record_ids`` accepts at most ``_MAX_RECORD_IDS`` (10,000) ids, and a
  category filter narrows on the server only up to ``_MAX_FILTER_CATEGORIES``
  (32) categories (beyond that it is still applied exactly on the client).
* A scope is at most ``_MAX_SCOPE_LENGTH`` (512) characters and
  ``_MAX_SCOPE_DEPTH`` (32) segments, and a category or id at most
  ``_MAX_INDEXED_STRING`` (1024) bytes. Saving anything larger raises
  ``ValueError``.
* A scope record is chunked at ``chunk_limit_bytes`` (default 262,144, at most
  ``_MAX_CHUNK_LIMIT_BYTES``), well under the server's record size ceiling. A
  save splits across chunks so none is filled past the limit, one memory is at
  most ``_MAX_ENTRY_BYTES`` (524,288) and a record-too-big reply moves to the
  next chunk once.

The Aerospike client is imported lazily. Install it with
``pip install "crewai[aerospike]"``.
"""

from __future__ import annotations

import asyncio
import atexit
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
import contextvars
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import functools
import importlib
import json
import logging
import queue
import re
import threading
import time
from typing import Any, TypeVar

import numpy as np

from crewai.memory.storage.backend import EmbeddingDimensionMismatchError
from crewai.memory.types import MemoryRecord, ScopeInfo


_logger = logging.getLogger(__name__)

_F = TypeVar("_F", bound=Callable[..., Any])

DEFAULT_SET_PREFIX = "crewai_memory"

# Soft ceiling for the payload estimate of one scope record before a new chunk
# is started. The server's hard record size limit is higher.
_DEFAULT_CHUNK_LIMIT_BYTES = 262_144

# Key of the single store-metadata record in the meta set.
_META_KEY = "store"

_SCHEME = "aerospike://"
_MAX_SPEC_LENGTH = 512
_HOST_RE = re.compile(r"[A-Za-z0-9.-]{1,253}")
_PORT_RE = re.compile(r"[0-9]{1,5}")
_NAMESPACE_RE = re.compile(r"[A-Za-z0-9_-]{1,31}")
_SET_PREFIX_RE = re.compile(r"[A-Za-z0-9_]{1,40}")
_QUERY_KEY_RE = re.compile(r"[A-Za-z0-9_]{1,40}")

_EPOCH = datetime(1970, 1, 1)

# Fixed per-entry overhead added to the payload estimate of a saved memory.
_ENTRY_OVERHEAD_BYTES = 128

# Bounds on the work and the record size of one call (see the module docstring).
_MAX_CHUNK_LIMIT_BYTES = 524_288
_MAX_ENTRY_BYTES = 524_288
_MAX_LIMIT = 1_000
_SCAN_ENTRIES_LIMIT = 50_000
_SCAN_RECORDS_LIMIT = 100_000
_MAX_RECORD_IDS = 10_000
_MAX_FILTER_CATEGORIES = 32
_MAX_SCOPE_LENGTH = 512
_MAX_SCOPE_DEPTH = 32
_MAX_INDEXED_STRING = 1_024

# Keys per batch call, chunk keys probed per round trip, and attempts at a
# generation-checked write before giving up.
_BATCH_KEYS = 100
_PROBE_WINDOW = 8
_GENERATION_RETRIES = 5

_MIN_TS = -(2**62)
_STORE_FORMAT = 1
_TRUNCATE_WAIT_SECONDS = 10.0

# Secondary indexes: (name suffix, bin, element type, collection kind).
_INDEX_SPECS = (
    ("scope_idx", "anc", "string", "LIST"),
    ("cats_idx", "cats", "string", "LIST"),
    ("created_idx", "created_at", "numeric", "LIST"),
    ("memids_idx", "mem", "string", "MAP_KEYS"),
)
_indexes_ready: set[tuple[str, int, str, str]] = set()
_indexes_lock = threading.Lock()


@dataclass(frozen=True)
class AerospikeSpec:
    """Parsed ``aerospike://host:port/namespace`` storage spec."""

    host: str
    port: int
    namespace: str
    set_prefix: str = DEFAULT_SET_PREFIX


def parse_aerospike_spec(spec: str) -> AerospikeSpec:
    """Parse an ``aerospike://host:port/namespace[?set_prefix=NAME]`` spec.

    The grammar is a strict allowlist. Credentials are not supported, so an
    at-sign and any query key other than ``set_prefix`` are rejected. Error
    messages are fixed strings and never repeat a value from the spec.

    Raises:
        ValueError: If the spec is not a well-formed Aerospike spec.
    """
    if len(spec) > _MAX_SPEC_LENGTH:
        raise ValueError(
            f"Aerospike storage spec is longer than {_MAX_SPEC_LENGTH} characters."
        )
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in spec):
        raise ValueError(
            "Aerospike storage spec must not contain whitespace or control characters."
        )
    if "%" in spec or "#" in spec:
        raise ValueError(
            "Aerospike storage spec must not contain percent-encoding or a fragment."
        )
    if "@" in spec:
        raise ValueError(
            "Aerospike storage spec must not contain credentials; "
            "authentication is not supported."
        )
    if not spec.lower().startswith(_SCHEME):
        raise ValueError("Aerospike storage spec must start with aerospike://.")

    rest, _, query = spec[len(_SCHEME) :].partition("?")
    authority, slash, namespace = rest.partition("/")
    if not slash or not _NAMESPACE_RE.fullmatch(namespace):
        raise ValueError(
            "Aerospike storage spec needs one namespace of 1 to 31 letters, "
            "digits, underscores or hyphens, as in aerospike://host:port/namespace."
        )

    if authority.startswith("["):
        host, bracket, port_part = authority[1:].partition("]")
        if not bracket or not port_part.startswith(":") or not _is_ipv6(host):
            raise ValueError("Aerospike storage spec has an invalid IPv6 host.")
        port_text = port_part[1:]
    else:
        host, colon, port_text = authority.partition(":")
        if not colon or not _HOST_RE.fullmatch(host):
            raise ValueError(
                "Aerospike storage spec needs a host and a port, "
                "as in aerospike://host:port/namespace."
            )
    if not _PORT_RE.fullmatch(port_text) or not 1 <= int(port_text) <= 65535:
        raise ValueError("Aerospike storage spec port must be between 1 and 65535.")

    set_prefix = DEFAULT_SET_PREFIX
    if query:
        seen = False
        for part in query.split("&"):
            key, _, value = part.partition("=")
            if key != "set_prefix":
                shown = key if _QUERY_KEY_RE.fullmatch(key) else "(invalid)"
                raise ValueError(
                    f"Aerospike storage spec does not support the query key "
                    f"'{shown}'; only set_prefix is allowed."
                )
            if seen or not _SET_PREFIX_RE.fullmatch(value):
                raise ValueError(
                    "Aerospike storage spec set_prefix must be given once, as "
                    "1 to 40 letters, digits or underscores."
                )
            seen = True
            set_prefix = value

    return AerospikeSpec(
        host=host, port=int(port_text), namespace=namespace, set_prefix=set_prefix
    )


def _is_ipv6(host: str) -> bool:
    import ipaddress

    try:
        ipaddress.IPv6Address(host)
    except ValueError:
        return False
    return True


def aerospike_storage_factory(spec: str) -> AerospikeStorage | None:
    """Memory storage factory for ``aerospike://`` specs.

    Pass it to ``set_memory_storage_factory``. Returns ``None`` for any other
    spec so the built-in backends keep handling those.

    Raises:
        ValueError: If the spec starts with ``aerospike://`` but is malformed.
    """
    if not spec.lower().startswith(_SCHEME):
        return None
    return AerospikeStorage(parse_aerospike_spec(spec))


def _sdk() -> tuple[Any, Any]:
    """Import the Aerospike SDK lazily and return its package and sync module."""
    try:
        sdk = importlib.import_module("aerospike_sdk")
        sync = importlib.import_module("aerospike_sdk.sync")
    except ImportError as exc:
        raise ImportError(
            "The Aerospike memory backend needs the Aerospike client. "
            'Install it with: pip install "crewai[aerospike]"'
        ) from exc
    return sdk, sync


@dataclass(frozen=True)
class _Connection:
    """One Aerospike cluster connection and its session, shared per process."""

    cluster: Any
    session: Any


def _connect(host: str, port: int) -> _Connection:
    """Open a cluster connection. The only function that builds an SDK client."""
    sdk, sync = _sdk()
    _logger.debug("Connecting to Aerospike at %s:%s", host, port)
    cluster = sync.ClusterDefinition(host, port).connect()
    session = cluster.create_session(sdk.Behavior.DEFAULT)
    return _Connection(cluster=cluster, session=session)


_connect_jobs: queue.SimpleQueue[tuple[Callable[[], _Connection], Future[_Connection]]]
_connect_jobs = queue.SimpleQueue()
_connect_thread: threading.Thread | None = None


def _connect_forever() -> None:
    while True:
        job, future = _connect_jobs.get()
        try:
            future.set_result(job())
        except BaseException as exc:  # handed to the caller, who re-raises it
            future.set_exception(exc)


def _connect_on_long_lived_thread(host: str, port: int) -> _Connection:
    """Open a connection on a thread that lives as long as the process.

    A client created on a thread that has since exited crashes the interpreter
    (a segmentation fault, not an exception) when another thread later runs a
    set scan on it. The caller may be a short-lived thread, for example the one
    that builds the storage from inside a running event loop, so every
    connection is created on one daemon thread instead. Call with
    ``_connections_lock`` held.
    """
    global _connect_thread
    if _connect_thread is None:
        _connect_thread = threading.Thread(
            target=_connect_forever, name="crewai-aerospike-connect", daemon=True
        )
        _connect_thread.start()
    future: Future[_Connection] = Future()
    _connect_jobs.put((functools.partial(_connect, host, port), future))
    return future.result()


_connections: dict[tuple[str, int], _Connection] = {}
_connections_lock = threading.Lock()
_atexit_registered = False


def _get_connection(host: str, port: int) -> _Connection:
    """Return the process-wide connection for a host and port, creating it once."""
    global _atexit_registered
    key = (host.lower(), port)
    with _connections_lock:
        connection = _connections.get(key)
        if connection is None:
            connection = _connect_on_long_lived_thread(host, port)
            _connections[key] = connection
            if not _atexit_registered:
                atexit.register(shutdown_connections)
                _atexit_registered = True
        return connection


def shutdown_connections() -> None:
    """Close every cached connection once. Safe to call repeatedly.

    Registered to run at interpreter exit. Storages built before the call must
    not be used afterwards.
    """
    with _connections_lock:
        connections = list(_connections.values())
        _connections.clear()
    for connection in connections:
        _close_cluster(connection.cluster)


def _close_cluster(cluster: Any) -> None:
    try:
        cluster.close()
    except Exception:
        _logger.debug("Error closing Aerospike connection", exc_info=True)


def _to_vector(values: Any) -> Any:
    """Build the SDK's native float32 vector value from a sequence of floats."""
    sdk, _ = _sdk()
    return sdk.Vector(np.asarray(values, dtype=np.float32))


def _from_vector(value: Any) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Return a float32 numpy array from a stored native vector value."""
    return np.asarray(value.numpy_value, dtype=np.float32)


def _check_dimension(stored: int | None, new: int) -> None:
    """Raise if a store that already has a dimension is given a different one."""
    if stored is not None and stored != new:
        raise EmbeddingDimensionMismatchError(stored, new)


def _batch_dimension(records: list[MemoryRecord]) -> int:
    """Common embedding dimension of a batch, 0 when none has an embedding."""
    dimension = 0
    for record in records:
        if record.embedding:
            if dimension:
                _check_dimension(dimension, len(record.embedding))
            dimension = len(record.embedding)
    return dimension


def _cosine_scores(
    matrix: np.ndarray[Any, Any], query: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """Cosine similarity of each matrix row to the query, 0.0 for a zero norm."""
    norms = np.linalg.norm(matrix, axis=1) * float(np.linalg.norm(query))
    dots = matrix @ query
    scores: np.ndarray[Any, Any] = np.divide(
        dots, norms, out=np.zeros_like(dots), where=norms > 0
    )
    return scores


def _normalize_scope(scope: str | None) -> str:
    """Return a scope with a leading slash and no trailing slash."""
    parts = [p for p in (scope or "").split("/") if p]
    return "/" + "/".join(parts)


def _ancestors(scope: str) -> list[str]:
    """Return the root, every ancestor and the scope itself, shallowest first."""
    parts = [p for p in scope.split("/") if p]
    return ["/"] + ["/" + "/".join(parts[: i + 1]) for i in range(len(parts))]


def _scope_matches(scope: str, prefix: str | None) -> bool:
    """Path-segment aware prefix match: ``/a`` matches ``/a`` and ``/a/b``."""
    if prefix is None:
        return True
    normalized = _normalize_scope(prefix)
    return (
        normalized == "/" or scope == normalized or scope.startswith(normalized + "/")
    )


def _to_micros(value: datetime) -> int:
    """Epoch microseconds UTC. Naive datetimes are treated as UTC."""
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    delta = value - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def _from_micros(value: int) -> datetime:
    """Naive UTC datetime from epoch microseconds."""
    return _EPOCH + timedelta(microseconds=int(value))


def _prefix_of(scope_prefix: str | None) -> str | None:
    """Normalized scope prefix, or None when it covers every scope."""
    if scope_prefix is None:
        return None
    normalized = _normalize_scope(scope_prefix)
    return None if normalized == "/" else normalized


def _check_limit(value: int, name: str = "limit") -> int:
    """Validate a result count and clamp it to ``_MAX_LIMIT``."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return min(value, _MAX_LIMIT)


def _check_ids(record_ids: list[str]) -> list[str]:
    ids = list(dict.fromkeys(record_ids))
    if len(ids) > _MAX_RECORD_IDS:
        raise ValueError(f"At most {_MAX_RECORD_IDS} record ids are allowed per call.")
    return ids


def _check_indexed(value: str, what: str) -> None:
    if len(value.encode("utf-8")) > _MAX_INDEXED_STRING:
        raise ValueError(
            f"A memory {what} must be at most {_MAX_INDEXED_STRING} bytes."
        )


def _record_to_entry(record: MemoryRecord) -> dict[str, Any]:
    """Entry map stored under ``mem[id]``."""
    entry: dict[str, Any] = {
        "c": record.content,
        "cat": list(record.categories),
        "md": json.dumps(record.metadata),
        "imp": float(record.importance),
        "ts": _to_micros(record.created_at),
        "la": _to_micros(record.last_accessed),
        "prv": record.private,
    }
    if record.source is not None:
        entry["src"] = record.source
    return entry


def _estimate_bytes(record_id: str, entry: dict[str, Any], dimension: int) -> int:
    """Rough payload size of one stored memory, used for chunk rollover."""
    size = _ENTRY_OVERHEAD_BYTES + len(record_id.encode("utf-8"))
    size += len(entry["c"].encode("utf-8")) + len(entry["md"].encode("utf-8"))
    size += sum(len(c.encode("utf-8")) + 8 for c in entry["cat"])
    size += len(entry.get("src", "").encode("utf-8"))
    if dimension:
        size += 4 * dimension + 16
    return size


def _entry_to_record(
    scope: str,
    record_id: str,
    entry: dict[str, Any],
    embedding: list[float] | None = None,
) -> MemoryRecord:
    """Rebuild a ``MemoryRecord`` from a stored entry map."""
    return MemoryRecord(
        id=record_id,
        content=str(entry["c"]),
        scope=scope,
        categories=list(entry.get("cat") or []),
        metadata=json.loads(entry["md"]) if entry.get("md") else {},
        importance=float(entry.get("imp", 0.5)),
        created_at=_from_micros(entry["ts"]),
        last_accessed=_from_micros(entry.get("la", entry["ts"])),
        embedding=embedding,
        source=entry.get("src"),
        private=bool(entry.get("prv", False)),
    )


def _entry_matches(
    entry: dict[str, Any],
    categories: list[str] | None,
    metadata_filter: dict[str, Any] | None,
) -> bool:
    """Exact entry-level filters: any category, and every metadata pair equal."""
    if categories and not any(c in entry.get("cat", []) for c in categories):
        return False
    if metadata_filter:
        metadata = json.loads(entry["md"]) if entry.get("md") else {}
        if not all(metadata.get(k) == v for k, v in metadata_filter.items()):
            return False
    return True


def _child_scope(base: str, scope: str) -> str | None:
    """Immediate child of ``base`` on the way to ``scope``, if ``scope`` is below it."""
    root = "" if base == "/" else base
    if not scope.startswith(root + "/") or scope == base:
        return None
    first = scope[len(root) + 1 :].split("/", 1)[0]
    return f"{root}/{first}" if first else None


def _truncate(session: Any, dataset: Any, namespace: str, set_name: str) -> None:
    """Truncate a set and wait until the server has finished sweeping it.

    The server sweeps a truncated set in the background, and a scan that starts
    meanwhile can miss records written right after the truncate. The namespace
    and set name come from the validated spec, never from caller data.
    """
    session.truncate(dataset)
    command = f"sets/{namespace}/{set_name}"
    deadline = time.monotonic() + _TRUNCATE_WAIT_SECONDS
    while time.monotonic() < deadline:
        reply = session.info().info(command)
        if not any("truncating=true" in str(v) for v in reply.values()):
            return
        time.sleep(0.01)
    _logger.warning("Aerospike set %s was still truncating after the wait.", set_name)


def _last(value: Any) -> Any:
    """Result of the final operation on a bin (batch replies are lists)."""
    return value[-1] if isinstance(value, list) else value


@dataclass
class _Item:
    """One memory prepared for writing."""

    id: str
    entry: dict[str, Any]
    vector: Any
    size: int


def _plan_chunks(
    items: list[_Item], chunk: int, used: int, limit: int
) -> list[tuple[int, list[_Item]]]:
    """Split a scope's items over chunks so none is filled past ``limit``.

    ``chunk`` is the open chunk and ``used`` its estimated bytes. A memory
    larger than the limit gets a chunk of its own.
    """
    plan: list[tuple[int, list[_Item]]] = []
    current: list[_Item] = []
    for item in items:
        if used + item.size > limit and (current or used > 0):
            if current:
                plan.append((chunk, current))
            chunk, used, current = chunk + 1, 0, []
        current.append(item)
        used += item.size
    if current:
        plan.append((chunk, current))
    return plan


def _off_event_loop(method: _F) -> _F:
    """Run a blocking storage method on a worker thread when called from a loop.

    The SDK's blocking surface refuses to run on an event-loop thread, and
    ``Memory.arecall`` and friends call the synchronous storage methods from
    one. The calling loop stays blocked while the call runs, as it does with the
    other backends; the ``a*`` methods avoid even that by running in a thread.
    """

    @functools.wraps(method)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return method(*args, **kwargs)
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                contextvars.copy_context().run, method, *args, **kwargs
            ).result()

    return wrapper  # type: ignore[return-value]


class AerospikeStorage:
    """Aerospike-backed storage for the unified memory system.

    The connection is shared by every storage built for the same host and
    port. Settings come only from the spec; nothing is read from the
    environment.
    """

    @_off_event_loop
    def __init__(
        self,
        spec: AerospikeSpec,
        chunk_limit_bytes: int = _DEFAULT_CHUNK_LIMIT_BYTES,
    ) -> None:
        """Initialize the storage.

        Args:
            spec: Parsed storage spec (host, port, namespace, set prefix).
            chunk_limit_bytes: Payload estimate at which a scope record is
                considered full and a new chunk is started. At most
                ``_MAX_CHUNK_LIMIT_BYTES``.
        """
        if not 0 < chunk_limit_bytes <= _MAX_CHUNK_LIMIT_BYTES:
            raise ValueError(
                f"chunk_limit_bytes must be between 1 and {_MAX_CHUNK_LIMIT_BYTES}."
            )
        self._spec = spec
        self._chunk_limit_bytes = chunk_limit_bytes
        self._connection = _get_connection(spec.host, spec.port)
        sdk, _ = _sdk()
        self._api = sdk
        self._scope_set = sdk.DataSet.of(spec.namespace, spec.set_prefix)
        self._meta_set = sdk.DataSet.of(spec.namespace, f"{spec.set_prefix}_meta")
        self._meta_key = self._meta_set.id(_META_KEY)
        self._lock = threading.Lock()
        self._dim: int | None = None
        self._open_chunks: dict[str, tuple[int, int]] = {}
        self._ensure_indexes()

    @property
    def _session(self) -> Any:
        return self._connection.session

    def close(self) -> None:
        """Do nothing: the client is shared process-wide.

        Its lifecycle belongs to the connection registry, so closing one
        storage (``Memory.close`` calls this) must not break the others.
        """

    def __copy__(self) -> AerospikeStorage:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> AerospikeStorage:
        # Never clone a connection: ``Memory.__deepcopy__`` copies its storage.
        return self

    # -- schema -------------------------------------------------------------

    def _ensure_indexes(self) -> None:
        """Create the four secondary indexes once per process and set prefix.

        Creating an index that already exists is not an error, so reconnecting
        (or another process starting first) is harmless.
        """
        spec = self._spec
        token = (spec.host.lower(), spec.port, spec.namespace, spec.set_prefix)
        with _indexes_lock:
            if token in _indexes_ready:
                return
            api = self._api
            for suffix, bin_name, kind, collection in _INDEX_SPECS:
                builder = (
                    self._session.index(dataset=self._scope_set)
                    .on_bin(bin_name)
                    .named(f"{spec.set_prefix}_{suffix}")
                )
                builder = builder.string() if kind == "string" else builder.numeric()
                builder = builder.collection(
                    getattr(api.CollectionIndexType, collection)
                )
                try:
                    builder.create().wait_till_complete_blocking()
                except api.IndexAlreadyExistsError:
                    continue
            _indexes_ready.add(token)

    # -- dimension ----------------------------------------------------------

    def _stored_dim(self, refresh: bool = False) -> int | None:
        """Embedding dimension of the store, cached once known."""
        with self._lock:
            if self._dim is not None and not refresh:
                return self._dim
        row = self._read_bins([self._meta_key], ["dim"])[0]
        dim = int(row[0]["dim"]) if row and row[0].get("dim") is not None else None
        with self._lock:
            self._dim = dim
        return dim

    def _ensure_dimension(self, new_dim: int) -> None:
        """Record the dimension on the first embedding and reject any other."""
        stored = self._stored_dim()
        if stored is not None and stored != new_dim:
            # The cached value may predate a reset by another storage.
            stored = self._stored_dim(refresh=True)
        if stored is None:
            try:
                self._session.insert(self._meta_key).bin("dim").set_to(new_dim).bin(
                    "fmt"
                ).set_to(_STORE_FORMAT).execute().close()
            except self._api.RecordExistsError:
                # Lost the race to set the dimension: compare with the winner.
                stored = self._stored_dim(refresh=True)
            else:
                with self._lock:
                    self._dim = new_dim
                return
        _check_dimension(stored, new_dim)

    # -- low-level reads ----------------------------------------------------

    def _key(self, scope: str, chunk: int) -> Any:
        return self._scope_set.id(f"{scope}|{chunk}")

    def _read_bins(
        self, keys: list[Any], bins: list[str]
    ) -> list[tuple[dict[str, Any], int] | None]:
        """Batch-read bins; one (bins, generation) or None per key, in order."""
        out: list[tuple[dict[str, Any], int] | None] = []
        for start in range(0, len(keys), _BATCH_KEYS):
            window = keys[start : start + _BATCH_KEYS]
            with (
                self._session.query(*window)
                .bins(bins)
                .include_missing_keys()
                .execute() as stream
            ):
                rows = list(stream)
            if len(rows) != len(window):
                raise RuntimeError(
                    "Aerospike batch read returned an unexpected row count."
                )
            for row in rows:
                if row.is_ok and row.record is not None:
                    out.append((row.record.bins, row.record.generation))
                elif row.result_code == self._api.ResultCode.KEY_NOT_FOUND_ERROR:
                    out.append(None)
                else:
                    row.or_raise()
                    out.append(None)
        return out

    def _read_sizes(self, keys: list[Any]) -> list[int]:
        """Number of memories per record, counted on the server."""
        out: list[int] = []
        for start in range(0, len(keys), _BATCH_KEYS):
            window = keys[start : start + _BATCH_KEYS]
            with (
                self._session.query(*window)
                .bin("mem")
                .map_size()
                .include_missing_keys()
                .execute() as stream
            ):
                rows = list(stream)
            for row in rows:
                ok = row.is_ok and row.record is not None
                out.append(int(row.record.bins.get("mem") or 0) if ok else 0)
        return out

    def _fetch_entries(
        self, requests: list[tuple[Any, list[str]]], with_emb: bool
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Read exactly the given memory ids from each record, on the server.

        Returns one ``(mem entries, emb vectors)`` pair per request, in order.
        """
        out: list[tuple[dict[str, Any], dict[str, Any]]] = []
        if not requests:
            return out
        for start in range(0, len(requests), _BATCH_KEYS):
            window = requests[start : start + _BATCH_KEYS]
            builder: Any = None
            for key, ids in window:
                builder = (
                    self._session.query(key) if builder is None else builder.query(key)
                )
                builder = builder.bin("mem").on_map_key_list(ids).get_keys_and_values()
                if with_emb:
                    builder = (
                        builder.bin("emb").on_map_key_list(ids).get_keys_and_values()
                    )
            with builder.include_missing_keys().execute() as stream:
                rows = list(stream)
            if len(rows) != len(window):
                raise RuntimeError(
                    "Aerospike batch read returned an unexpected row count."
                )
            for row in rows:
                if row.is_ok and row.record is not None:
                    bins = row.record.bins
                    out.append((bins.get("mem") or {}, bins.get("emb") or {}))
                elif row.result_code == self._api.ResultCode.KEY_NOT_FOUND_ERROR:
                    out.append(({}, {}))
                else:
                    row.or_raise()
                    out.append(({}, {}))
        return out

    def _categories_exp(self, categories: list[str]) -> Any:
        """Server filter: the record's ``cats`` holds any of the categories."""
        api = self._api
        terms = [
            api.Exp.gt(
                api.Exp.list_get_by_value(
                    api.ListReturnType.COUNT,
                    api.Exp.string_val(c),
                    api.Exp.list_bin("cats"),
                    [],
                ),
                api.Exp.int_val(0),
            )
            for c in categories
        ]
        return terms[0] if len(terms) == 1 else api.Exp.or_(terms)

    def _iter_candidates(
        self,
        scope_prefix: str | None,
        categories: list[str] | None = None,
        bins: tuple[str, ...] = (),
        older_than_us: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield the bins of each scope record that can hold a match.

        Narrowing happens on the server with typed filters and bound values:
        a scope prefix uses the scope index (and, with categories, a server
        filter on the categories bin), categories alone use the categories
        index, and ``older_than_us`` alone uses the created_at index. Nothing
        is formatted into query text. Records stream lazily and at most
        ``_SCAN_RECORDS_LIMIT`` are produced. ``scope`` and ``chunk`` are always
        included so callers can rebuild the record key.
        """
        api = self._api
        prefix = _prefix_of(scope_prefix)
        if prefix is not None and len(prefix) > _MAX_SCOPE_LENGTH:
            return
        cats = list(dict.fromkeys(categories or []))
        narrow = 0 < len(cats) <= _MAX_FILTER_CATEGORIES and all(
            len(c.encode("utf-8")) <= _MAX_INDEXED_STRING for c in cats
        )
        plans: list[tuple[Any, Any]]
        if prefix is not None:
            plans = [
                (
                    api.Filter.contains("anc", prefix, api.CollectionIndexType.LIST),
                    self._categories_exp(cats) if narrow else None,
                )
            ]
        elif narrow:
            plans = [
                (api.Filter.contains("cats", c, api.CollectionIndexType.LIST), None)
                for c in cats
            ]
        elif older_than_us is not None:
            plans = [
                (
                    api.Filter.contains_range(
                        "created_at",
                        _MIN_TS,
                        older_than_us - 1,
                        api.CollectionIndexType.LIST,
                    ),
                    None,
                )
            ]
        else:
            plans = [(None, None)]

        seen: set[tuple[Any, Any]] | None = set() if len(plans) > 1 else None
        produced = 0
        for index_filter, server_filter in plans:
            # A scan that returns no rows is confirmed once: the server build
            # this was developed against occasionally answers a scan of a
            # populated set with zero rows (about 0.3 percent of scans).
            for attempt in range(2):
                builder = self._session.query(self._scope_set).bins(
                    ["scope", "chunk", *bins]
                )
                if index_filter is not None:
                    builder = builder.filter(index_filter)
                if server_filter is not None:
                    builder = builder.where(server_filter)
                rows_seen = 0
                with builder.stream() as stream:
                    for row in stream:
                        rows_seen += 1
                        if not row.is_ok or row.record is None:
                            row.or_raise()
                            continue
                        found = row.record.bins
                        if "scope" not in found or "chunk" not in found:
                            continue
                        if seen is not None:
                            ident = (found.get("scope"), found.get("chunk"))
                            if ident in seen:
                                continue
                            seen.add(ident)
                        if produced >= _SCAN_RECORDS_LIMIT:
                            _logger.warning(
                                "Aerospike memory scan stopped after %d scope "
                                "records; narrow the scope for complete results.",
                                _SCAN_RECORDS_LIMIT,
                            )
                            return
                        produced += 1
                        yield found
                if rows_seen or attempt:
                    break

    def _locate(self, record_ids: list[str]) -> dict[str, list[tuple[str, int]]]:
        """Find the records holding each memory id through the memory-id index."""
        api = self._api
        found: dict[str, list[tuple[str, int]]] = {}
        for record_id in _check_ids(record_ids):
            if len(record_id.encode("utf-8")) > _MAX_INDEXED_STRING:
                continue
            index_filter = api.Filter.contains(
                "mem", record_id, api.CollectionIndexType.MAP_KEYS
            )
            for _attempt in range(2):
                locations: list[tuple[str, int]] = []
                with (
                    self._session.query(self._scope_set)
                    .bins(["scope", "chunk"])
                    .filter(index_filter)
                    .stream() as stream
                ):
                    for row in stream:
                        if not row.is_ok or row.record is None:
                            row.or_raise()
                            continue
                        bins = row.record.bins
                        locations.append((bins["scope"], int(bins["chunk"])))
                if locations:
                    found[record_id] = locations
                    break
        return found

    # -- writes -------------------------------------------------------------

    def _check_record(self, record: MemoryRecord, scope: str) -> None:
        if len(scope) > _MAX_SCOPE_LENGTH or scope.count("/") > _MAX_SCOPE_DEPTH:
            raise ValueError(
                f"A memory scope must be at most {_MAX_SCOPE_LENGTH} characters "
                f"and {_MAX_SCOPE_DEPTH} segments deep."
            )
        _check_indexed(record.id, "id")
        for category in record.categories:
            _check_indexed(category, "category")

    def _prepare(self, record: MemoryRecord, dimension: int) -> _Item:
        entry = _record_to_entry(record)
        size = _estimate_bytes(record.id, entry, dimension)
        if size > _MAX_ENTRY_BYTES:
            raise ValueError(
                f"A memory is too large to store ({_MAX_ENTRY_BYTES} byte limit)."
            )
        vector = _to_vector(record.embedding) if record.embedding else None
        return _Item(record.id, entry, vector, size)

    def _open_chunk(self, scope: str) -> tuple[int, int]:
        """Chunk that takes the next write for a scope, and its estimated bytes."""
        with self._lock:
            cached = self._open_chunks.get(scope)
        if cached is not None:
            return cached
        start = 0
        while True:
            keys = [self._key(scope, c) for c in range(start, start + _PROBE_WINDOW)]
            for offset, row in enumerate(self._read_bins(keys, ["bytes"])):
                used = int(row[0].get("bytes") or 0) if row else 0
                if row is None or used < self._chunk_limit_bytes:
                    return start + offset, used
            start += _PROBE_WINDOW

    def _group_ops(
        self, builder: Any, scope: str, chunk: int, items: list[_Item]
    ) -> Any:
        """Add the single atomic multi-operation write for one chunk record."""
        builder = (
            builder.bin("scope")
            .set_to(scope)
            .bin("chunk")
            .set_to(chunk)
            .bin("anc")
            .list_append_items(
                _ancestors(scope), unique=True, no_fail=True, partial=True
            )
            .bin("mem")
            .map_upsert_items({i.id: i.entry for i in items})
        )
        embedded = {i.id: i.vector for i in items if i.vector is not None}
        plain = [i.id for i in items if i.vector is None]
        if embedded:
            builder = builder.bin("emb").map_upsert_items(embedded)
        if plain:
            # A replaced memory without an embedding must not keep a stale vector.
            builder = builder.bin("emb").on_map_key_list(plain).remove()
        categories = list(dict.fromkeys(c for i in items for c in i.entry["cat"]))
        if categories:
            builder = builder.bin("cats").list_append_items(
                categories, unique=True, no_fail=True, partial=True
            )
        return (
            builder.bin("created_at")
            .list_append_items([i.entry["ts"] for i in items])
            .bin("bytes")
            .add(sum(i.size for i in items))
            .bin("bytes")
            .get()
        )

    def _dispatch(self, groups: list[tuple[str, int, list[_Item]]]) -> list[int | None]:
        """Write each chunk record; one batch call when there are several.

        Returns the record's byte estimate after each write, or None for a
        group the server rejected as too big for one record.
        """
        too_big = self._api.ResultCode.RECORD_TOO_BIG
        builder: Any = None
        for scope, chunk, items in groups:
            key = self._key(scope, chunk)
            builder = (
                self._session.upsert(key) if builder is None else builder.upsert(key)
            )
            builder = self._group_ops(builder, scope, chunk, items)
        try:
            with builder.execute() as stream:
                rows = list(stream)
        except self._api.RecordTooBigError:
            return [None]  # a single-key write raises instead of replying in-stream
        outcomes: list[int | None] = []
        for row in rows:
            if row.is_ok and row.record is not None:
                outcomes.append(int(_last(row.record.bins["bytes"])))
            elif row.result_code == too_big:
                outcomes.append(None)
            else:
                row.or_raise()
        return outcomes

    def _write_items(self, by_scope: dict[str, list[_Item]]) -> None:
        """Place items into chunk records and write them, batching the writes."""
        plan: list[tuple[str, int, list[_Item]]] = []
        for scope, items in by_scope.items():
            chunk, used = self._open_chunk(scope)
            plan.extend(
                (scope, target, group)
                for target, group in _plan_chunks(
                    items, chunk, used, self._chunk_limit_bytes
                )
            )
        outcomes = self._dispatch(plan)
        for (scope, chunk, items), outcome in zip(plan, outcomes, strict=True):
            if outcome is None:
                # The estimate ran low: try the next chunk once, then give up.
                chunk += 1
                outcome = self._dispatch([(scope, chunk, items)])[0]
                if outcome is None:
                    raise ValueError("A memory is too large for one record.")
            with self._lock:
                if outcome >= self._chunk_limit_bytes:
                    self._open_chunks[scope] = (chunk + 1, 0)
                else:
                    self._open_chunks[scope] = (chunk, outcome)

    @_off_event_loop
    def save(self, records: list[MemoryRecord]) -> None:
        if not records:
            return
        scopes: list[str] = []
        for record in records:
            scope = _normalize_scope(record.scope)
            self._check_record(record, scope)
            scopes.append(scope)
        dimension = _batch_dimension(records)
        if dimension:
            self._ensure_dimension(dimension)
        by_scope: dict[str, dict[str, _Item]] = {}
        for record, scope in zip(records, scopes, strict=True):
            item = self._prepare(record, len(record.embedding or ()))
            by_scope.setdefault(scope, {})[record.id] = item
        self._write_items({s: list(items.values()) for s, items in by_scope.items()})

    def _remove_entries(
        self,
        scope: str,
        chunk: int,
        match: Callable[[str, dict[str, Any]], bool],
        preread: tuple[dict[str, Any], int] | None = None,
    ) -> int:
        """Remove the memories of one record that ``match`` selects.

        One generation-checked write per record: a concurrent save makes the
        check fail and the removal is retried against the fresh record. A record
        left with no memory is deleted. Returns the number removed.
        """
        api = self._api
        key = self._key(scope, chunk)
        for _ in range(_GENERATION_RETRIES):
            row = preread or self._read_bins([key], ["mem"])[0]
            preread = None
            if row is None:
                return 0
            bins, generation = row
            mem = bins.get("mem") or {}
            gone = [i for i, e in mem.items() if match(i, e)]
            if not gone:
                return 0
            gone_set = set(gone)
            try:
                if len(gone) == len(mem):
                    self._session.delete(key).ensure_generation_is(
                        generation
                    ).execute().close()
                else:
                    removed_ts = {mem[i]["ts"] for i in gone}
                    kept = Counter(
                        e["ts"]
                        for i, e in mem.items()
                        if i not in gone_set and e["ts"] in removed_ts
                    )
                    freed = sum(_estimate_bytes(i, mem[i], 0) for i in gone)
                    write = (
                        self._session.update(key)
                        .ensure_generation_is(generation)
                        .bin("mem")
                        .on_map_key_list(gone)
                        .remove()
                        .bin("emb")
                        .on_map_key_list(gone)
                        .remove()
                        .bin("created_at")
                        .on_list_value_list(sorted(removed_ts))
                        .remove()
                    )
                    for ts, count in kept.items():
                        # Removal by value drops every copy; restore the kept ones.
                        write = write.bin("created_at").list_append_items([ts] * count)
                    write.bin("bytes").add(-freed).execute().close()
            except api.GenerationError:
                continue
            except api.RecordNotFoundError:
                return 0
            with self._lock:
                self._open_chunks.pop(scope, None)
            return len(gone)
        raise RuntimeError("Could not remove memories because of concurrent writes.")

    @_off_event_loop
    def update(self, record: MemoryRecord) -> None:
        scope = _normalize_scope(record.scope)
        self._check_record(record, scope)
        dimension = len(record.embedding or ())
        if dimension:
            self._ensure_dimension(dimension)
        item = self._prepare(record, dimension)
        locations = self._locate([record.id]).get(record.id, [])
        in_place = next((loc for loc in locations if loc[0] == scope), None)
        if in_place is None:
            self._write_items({scope: [item]})
        else:
            self._replace_in_place(in_place, item)
        for old_scope, old_chunk in locations:
            if (old_scope, old_chunk) != in_place:
                self._remove_entries(old_scope, old_chunk, lambda i, _e: i == record.id)

    def _replace_in_place(self, location: tuple[str, int], item: _Item) -> None:
        scope, chunk = location
        key = self._key(scope, chunk)
        ((old_entries, old_vectors),) = self._fetch_entries([(key, [item.id])], True)
        old = old_entries.get(item.id)
        # update-only: a record deleted meanwhile must not come back half-built.
        builder = (
            self._session.update(key).bin("mem").map_upsert_items({item.id: item.entry})
        )
        if item.vector is not None:
            builder = builder.bin("emb").map_upsert_items({item.id: item.vector})
        else:
            builder = builder.bin("emb").on_map_key_list([item.id]).remove()
        if item.entry["cat"]:
            builder = builder.bin("cats").list_append_items(
                item.entry["cat"], unique=True, no_fail=True, partial=True
            )
        delta = item.size
        if old is not None:
            dim = self._stored_dim() or 0
            delta -= _estimate_bytes(item.id, old, dim if item.id in old_vectors else 0)
            if old["ts"] != item.entry["ts"]:
                row = self._read_bins([key], ["mem"])[0]
                shared = row is not None and any(
                    e["ts"] == old["ts"]
                    for i, e in (row[0].get("mem") or {}).items()
                    if i != item.id
                )
                if not shared:
                    builder = (
                        builder.bin("created_at").on_list_value(old["ts"]).remove()
                    )
                builder = builder.bin("created_at").list_append_items(
                    [item.entry["ts"]]
                )
        try:
            builder.bin("bytes").add(delta).execute().close()
        except self._api.RecordNotFoundError:
            self._write_items({scope: [item]})

    @_off_event_loop
    def get_record(self, record_id: str) -> MemoryRecord | None:
        for scope, chunk in self._locate([record_id]).get(record_id, []):
            ((entries, vectors),) = self._fetch_entries(
                [(self._key(scope, chunk), [record_id])], True
            )
            entry = entries.get(record_id)
            if entry is None:
                continue
            vector = vectors.get(record_id)
            embedding = _from_vector(vector).tolist() if vector is not None else None
            return _entry_to_record(scope, record_id, entry, embedding)
        return None

    @_off_event_loop
    def touch_records(self, record_ids: list[str]) -> None:
        """Set ``last_accessed`` to now for the given ids, one write per record."""
        if not record_ids:
            return
        now = _to_micros(datetime.now(timezone.utc))
        by_record: dict[tuple[str, int], list[str]] = {}
        for record_id, locations in self._locate(record_ids).items():
            for location in locations:
                by_record.setdefault(location, []).append(record_id)
        for (scope, chunk), ids in by_record.items():
            if not self._touch(scope, chunk, ids, now) and len(ids) > 1:
                # An id deleted since it was located fails the whole write.
                for record_id in ids:
                    self._touch(scope, chunk, [record_id], now)

    def _touch(self, scope: str, chunk: int, ids: list[str], now: int) -> bool:
        write = self._session.update(self._key(scope, chunk))
        for record_id in ids:
            write = write.bin("mem").on_map_key(record_id).on_map_key("la").set_to(now)
        try:
            write.execute().close()
        except self._api.AerospikeError:
            _logger.debug("touch_records skipped in %s", scope, exc_info=True)
            return False
        return True

    # -- search -------------------------------------------------------------

    @_off_event_loop
    def search(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        """Rank memories by cosine similarity to the query embedding.

        This is a brute-force scan and rank. Candidate scope records are
        narrowed on the server by scope prefix (and category), then every
        candidate memory's embedding is read and scored on the client, so cost
        grows linearly with candidate count times dimension. Scope the query
        with ``scope_prefix`` to keep it small. A search reads at most
        ``_SCAN_ENTRIES_LIMIT`` embeddings and logs a warning if it stops
        there. ``limit`` must be positive and is clamped to ``_MAX_LIMIT``.

        Raises:
            ValueError: If ``limit`` is not a positive integer.
            EmbeddingDimensionMismatchError: If the query dimension differs
                from the stored one.
        """
        limit = _check_limit(limit)
        if not query_embedding:
            return []
        query = np.asarray(query_embedding, dtype=np.float64)
        stored = self._stored_dim()
        if stored is not None and stored != query.shape[0]:
            stored = self._stored_dim(refresh=True)
        if stored is None:
            return []
        _check_dimension(stored, int(query.shape[0]))

        ranked: list[tuple[float, str, str, int]] = []
        for bins in self._iter_candidates(scope_prefix, categories, ("emb",)):
            stored_vectors = bins.get("emb") or {}
            room = _SCAN_ENTRIES_LIMIT - len(ranked)
            if len(stored_vectors) > room:
                _logger.warning(
                    "Aerospike memory search stopped after %d memories; "
                    "narrow the scope for complete results.",
                    _SCAN_ENTRIES_LIMIT,
                )
                stored_vectors = dict(list(stored_vectors.items())[:room])
            if stored_vectors:
                matrix = np.vstack(
                    [_from_vector(v) for v in stored_vectors.values()]
                ).astype(np.float64)
                _check_dimension(int(matrix.shape[1]), int(query.shape[0]))
                scores = _cosine_scores(matrix, query)
                for record_id, score in zip(stored_vectors, scores, strict=True):
                    ranked.append(
                        (float(score), record_id, bins["scope"], int(bins["chunk"]))
                    )
            if len(ranked) >= _SCAN_ENTRIES_LIMIT:
                break

        ranked.sort(key=lambda r: (-r[0], r[1]))
        results: list[tuple[MemoryRecord, float]] = []
        step = limit * 3
        for start in range(0, len(ranked), step):
            window = ranked[start : start + step]
            tranche = [r for r in window if r[0] >= min_score]
            if not tranche:
                break
            by_record: dict[tuple[str, int], list[str]] = {}
            for _score, record_id, scope, chunk in tranche:
                by_record.setdefault((scope, chunk), []).append(record_id)
            fetched = dict(
                zip(
                    by_record,
                    self._fetch_entries(
                        [(self._key(s, c), ids) for (s, c), ids in by_record.items()],
                        True,
                    ),
                    strict=True,
                )
            )
            for score, record_id, scope, chunk in tranche:
                entries, vectors = fetched[(scope, chunk)]
                entry = entries.get(record_id)
                if entry is None or record_id not in vectors:
                    continue
                if not _entry_matches(entry, categories, metadata_filter):
                    continue
                embedding = _from_vector(vectors[record_id]).tolist()
                results.append(
                    (_entry_to_record(scope, record_id, entry, embedding), score)
                )
                if len(results) >= limit:
                    return results
            if len(tranche) < len(window):
                break
        return results

    # -- delete, listing, scopes ---------------------------------------------

    def _delete_keys(self, keys: list[Any]) -> None:
        """Delete whole records, in batches, ignoring records already gone."""
        for start in range(0, len(keys), _BATCH_KEYS):
            window = keys[start : start + _BATCH_KEYS]
            self._session.delete(*window).execute(
                on_error=self._api.ErrorStrategy.IN_STREAM
            ).close()

    @_off_event_loop
    def delete(
        self,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        ids = _check_ids(record_ids) if record_ids is not None else None
        if ids is not None and not ids:
            ids = None
        if not (categories or metadata_filter or older_than is not None or ids):
            return self._delete_scopes(scope_prefix)
        cutoff = _to_micros(older_than) if older_than is not None else None
        id_set = set(ids) if ids is not None else None
        prefix = _prefix_of(scope_prefix)

        def match(record_id: str, entry: dict[str, Any]) -> bool:
            if id_set is not None and record_id not in id_set:
                return False
            if cutoff is not None and entry["ts"] >= cutoff:
                return False
            return _entry_matches(entry, categories, metadata_filter)

        if id_set is not None:
            targets = [
                loc
                for locations in self._locate(list(id_set)).values()
                for loc in locations
                if _scope_matches(loc[0], prefix)
            ]
            targets = list(dict.fromkeys(targets))
        else:
            targets = [
                (b["scope"], int(b["chunk"]))
                for b in self._iter_candidates(prefix, categories, (), cutoff)
            ]
        removed = 0
        for start in range(0, len(targets), _BATCH_KEYS // 4):
            window = targets[start : start + _BATCH_KEYS // 4]
            rows = self._read_bins([self._key(s, c) for s, c in window], ["mem"])
            for (scope, chunk), row in zip(window, rows, strict=True):
                if row is not None:
                    removed += self._remove_entries(scope, chunk, match, row)
        return removed

    def _delete_scopes(self, scope_prefix: str | None) -> int:
        """Delete whole records of a subtree (all records for no prefix)."""
        keys = [
            self._key(b["scope"], int(b["chunk"]))
            for b in self._iter_candidates(scope_prefix)
        ]
        removed = sum(self._read_sizes(keys))
        self._delete_keys(keys)
        with self._lock:
            self._open_chunks.clear()
        return removed

    @_off_event_loop
    def list_records(
        self,
        scope_prefix: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        limit = _check_limit(limit)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer.")
        rows = [
            (entry["ts"], record_id, scope, entry)
            for scope, record_id, entry in self._scan_entries(scope_prefix)
        ]
        rows.sort(key=lambda r: (-r[0], r[1]))
        return [
            _entry_to_record(scope, record_id, entry)
            for _ts, record_id, scope, entry in rows[offset : offset + limit]
        ]

    def _scan_entries(
        self, scope_prefix: str | None
    ) -> Iterator[tuple[str, str, dict[str, Any]]]:
        """Yield (scope, id, entry) for a prefix, at most ``_SCAN_ENTRIES_LIMIT``."""
        seen = 0
        for bins in self._iter_candidates(scope_prefix, None, ("mem",)):
            for record_id, entry in (bins.get("mem") or {}).items():
                if seen >= _SCAN_ENTRIES_LIMIT:
                    _logger.warning(
                        "Aerospike memory scan stopped after %d memories; "
                        "narrow the scope for complete results.",
                        _SCAN_ENTRIES_LIMIT,
                    )
                    return
                seen += 1
                yield bins["scope"], record_id, entry

    @_off_event_loop
    def get_scope_info(self, scope: str) -> ScopeInfo:
        path = _normalize_scope(scope)
        categories: set[str] = set()
        children: set[str] = set()
        oldest: int | None = None
        newest: int | None = None
        count = 0
        for entry_scope, _record_id, entry in self._scan_entries(path):
            count += 1
            categories.update(entry.get("cat") or [])
            ts = entry["ts"]
            oldest = ts if oldest is None else min(oldest, ts)
            newest = ts if newest is None else max(newest, ts)
            child = _child_scope(path, entry_scope)
            if child:
                children.add(child)
        return ScopeInfo(
            path=path,
            record_count=count,
            categories=sorted(categories),
            oldest_record=_from_micros(oldest) if oldest is not None else None,
            newest_record=_from_micros(newest) if newest is not None else None,
            child_scopes=sorted(children),
        )

    @_off_event_loop
    def list_scopes(self, parent: str = "/") -> list[str]:
        base = _normalize_scope(parent)
        children = {
            child
            for bins in self._iter_candidates(base)
            if (child := _child_scope(base, bins["scope"]))
        }
        return sorted(children)

    @_off_event_loop
    def list_categories(self, scope_prefix: str | None = None) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _scope, _record_id, entry in self._scan_entries(scope_prefix):
            for category in entry.get("cat") or []:
                counts[category] = counts.get(category, 0) + 1
        return counts

    @_off_event_loop
    def count(self, scope_prefix: str | None = None) -> int:
        keys = [
            self._key(b["scope"], int(b["chunk"]))
            for b in self._iter_candidates(scope_prefix)
        ]
        return sum(self._read_sizes(keys))

    @_off_event_loop
    def reset(self, scope_prefix: str | None = None) -> None:
        """Remove memories under a prefix, or this store's two sets for none.

        A full reset truncates only this prefix's scope and meta sets and never
        touches another set.
        """
        if _prefix_of(scope_prefix) is None:
            spec = self._spec
            _truncate(self._session, self._scope_set, spec.namespace, spec.set_prefix)
            _truncate(
                self._session,
                self._meta_set,
                spec.namespace,
                f"{spec.set_prefix}_meta",
            )
            with self._lock:
                self._dim = None
                self._open_chunks.clear()
            return
        self._delete_scopes(scope_prefix)

    async def asave(self, records: list[MemoryRecord]) -> None:
        await asyncio.to_thread(self.save, records)

    async def asearch(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        return await asyncio.to_thread(
            self.search,
            query_embedding,
            scope_prefix=scope_prefix,
            categories=categories,
            metadata_filter=metadata_filter,
            limit=limit,
            min_score=min_score,
        )

    async def adelete(
        self,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        return await asyncio.to_thread(
            self.delete,
            scope_prefix=scope_prefix,
            categories=categories,
            record_ids=record_ids,
            older_than=older_than,
            metadata_filter=metadata_filter,
        )
