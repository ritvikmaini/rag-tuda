"""Index generations.

Every build writes its own pair of tables, <collection>_g<NNNN>_chunks and <collection>_g<NNNN>_sections: schema v2
(typed chunks and sections). The assistant reads the two typed views of the live generation (`<collection>_chunks`,
`<collection>_sections`); a publish repoints them in one transaction with a lock timeout; the previous generation stays
for a rollback; older generations go with plain drops (no cascade), so a table a view still reads can never be dropped
by mistake. One build lock (a session advisory lock on a direct connection) serialises builds, rollbacks and the CLI. index_generations records every generation (status staging, live, previous,
failed, dropped); index_build_lock names the holder of the lock.

No config is read here: the caller passes the target and the connection factory, so tests
import this module without settings. CLI, in the indexer container (it connects as the container's role, rag_indexer;
indexer.cli wires it):
    indexer build | status | housekeep [--stale] | rollback | compare <a> <b> | publish <n>
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from indexer.index import cache, ddl
from indexer.index.ddl import ident

log = logging.getLogger(__name__)

# Named constants, not configuration
LOCK_NAMESPACE = 2  # first key of the two-key advisory lock; the second is hashtext(collection)
LOCK_TIMEOUT = "5s"  # SET LOCAL lock_timeout of every publish and drop
PUBLISH_ATTEMPTS = 5  # a publish is retried this often on a lock timeout
RETRY_SLEEP_S = 2
LAYOUT = "v2"  # the registry's layout column: typed chunks and sections
STATUSES = ("staging", "live", "previous", "failed", "dropped")
LOCK_NOT_AVAILABLE = "55P03"  # SQLSTATE of a lock timeout
DDL = (
    "CREATE TABLE IF NOT EXISTS index_generations ("
    " collection text NOT NULL, generation integer NOT NULL,"
    " status text NOT NULL CHECK (status IN ('staging', 'live', 'previous', 'failed', 'dropped')),"
    " layout text NOT NULL CHECK (layout IN ('legacy', 'v2')),"
    " tables jsonb NOT NULL,"
    " created_at timestamptz NOT NULL DEFAULT now(), published_at timestamptz,"
    " git_commit text, config_hash text, embedding_model text, embedding_revision text,"
    " counts jsonb, timings jsonb, detail text,"
    " PRIMARY KEY (collection, generation))",
    "CREATE TABLE IF NOT EXISTS index_build_lock ("
    " collection text PRIMARY KEY, holder text NOT NULL, started_at timestamptz NOT NULL DEFAULT now(), pid integer)",
    ddl.EMBEDDING_CACHE,
    ddl.INDEX_DOCUMENTS,
)


class GenerationError(RuntimeError):
    """A generation step refused or failed; the views are unchanged unless the message says otherwise."""


class RollbackRefused(GenerationError):
    """Nothing to roll back to; the CLI prints "refused: ..." and exits 1."""


class BuildBusy(GenerationError):
    """Another build, rollback or CLI step holds the build lock."""

    def __init__(self, holder=None, started_at=None):
        self.holder, self.started_at = holder, started_at
        isoformat = getattr(started_at, "isoformat", None)
        since = isoformat() if isoformat else (started_at or "an unknown time")
        super().__init__(f"a build holds the lock since {since}" + (f" ({holder})" if holder else ""))


class PublishedUnchecked(GenerationError):
    """The publish committed (both views read the new generation), but the check after the commit failed."""

    def __init__(self, message, previous, seconds):
        self.previous, self.seconds = previous, seconds
        super().__init__(message)


@dataclass(frozen=True)
class Target:
    """One collection: its generation tables and its two live views."""

    collection: str

    def table(self, n):
        return f"{self.collection}_g{int(n):04d}"

    def tables(self, n):
        return ddl.v2_tables(self.table(n))

    @property
    def typed_views(self):
        """The two typed views of schema v2, the assistant's reads: <collection>_chunks, _sections."""
        return (self.collection + "_chunks", self.collection + "_sections")

    @classmethod
    def for_collection(cls, name):
        """The profile's collection: its generation tables and the typed views the assistant reads."""
        return cls(name)


# --- pure helpers (unit-tested without a database) ---------------------------------------------------------------


def missing_indexes(kind, rows):
    """The required indexes of a children or parents table that no valid index provides. rows: (indexdef, valid, unique)."""
    return [
        name
        for name, needle, unique in ddl.REQUIRED_INDEXES[kind]
        if not any(
            valid and needle in definition and (is_unique or not unique) for definition, valid, is_unique in rows
        )
    ]


def viewdef_table(viewdef):
    """The table a pg_get_viewdef() text selects from, or None."""
    m = re.search(r'\bFROM\s+(?:public\.)?"?([A-Za-z0-9_]+)"?', viewdef or "")
    return m.group(1) if m else None


def is_lock_timeout(exc):
    return getattr(exc, "sqlstate", None) == LOCK_NOT_AVAILABLE


def _jsonable(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


# --- connections and small queries -------------------------------------------------------------------------------


def with_connection(fn, *args, connect_fn, **kwargs):
    """Runs fn(conn, ...) on its own direct connection from connect_fn (the endpoints call this in a worker thread)."""
    conn = connect_fn(application_name="indexer-generations")
    try:
        return fn(conn, *args, **kwargs)
    finally:
        conn.close()


def ensure_tables(conn):
    """The registry and the lock table; the assistant can read the registry, so a question during a fresh cluster's first
    build finds 'no live generation', not a missing privilege. The grant is made whenever the reader lacks SELECT, not
    only on creation: an indexer killed between CREATE and GRANT is repaired by the next call."""
    for statement in DDL:
        conn.execute(statement)
    if not conn.execute("SELECT has_table_privilege(%s, 'index_generations', 'SELECT')", (ddl.READER,)).fetchone()[0]:
        conn.execute(ddl.REGISTRY_GRANT)


def _has_registry(conn):
    return conn.execute("SELECT to_regclass('public.index_generations') IS NOT NULL").fetchone()[0]


def relkind(conn, name):
    row = conn.execute(
        "SELECT c.relkind::text FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = 'public' AND c.relname = %s",
        (name,),
    ).fetchone()
    return row[0] if row else None


def index_rows(conn, table):
    return conn.execute(
        "SELECT pg_get_indexdef(x.indexrelid), x.indisvalid, x.indisunique FROM pg_index x "
        "JOIN pg_class c ON c.oid = x.indrelid JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = 'public' AND c.relname = %s",
        (table,),
    ).fetchall()


def _size(conn, name):
    row = conn.execute(
        "SELECT pg_total_relation_size(c.oid) FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = 'public' AND c.relname = %s",
        (name,),
    ).fetchone()
    return int(row[0]) if row else 0


def typed_view_tables(conn, target):
    """{typed view: the table it selects from} for the two typed views that exist as views (schema v2)."""
    result = {}
    for view in target.typed_views:
        if relkind(conn, view) == "v":
            result[view] = viewdef_table(conn.execute("SELECT pg_get_viewdef(%s::regclass)", (view,)).fetchone()[0])
    return result


_ROW_COLUMNS = (
    "generation",
    "status",
    "layout",
    "tables",
    "created_at",
    "published_at",
    "git_commit",
    "config_hash",
    "embedding_model",
    "embedding_revision",
    "counts",
    "timings",
    "detail",
)


def generations(conn, target):
    if not _has_registry(conn):
        return []
    rows = conn.execute(
        "SELECT " + ", ".join(_ROW_COLUMNS) + " FROM index_generations WHERE collection = %s ORDER BY generation",
        (target.collection,),
    ).fetchall()
    return [dict(zip(_ROW_COLUMNS, row)) for row in rows]


def _live(rows):
    return next((r for r in rows if r["status"] == "live"), None)


def _newest_previous(rows):
    """The previous generation a rollback goes to: the one live most recently (published_at), then the highest number."""
    previous = [r for r in rows if r["status"] == "previous"]
    return max(
        previous,
        key=lambda r: (
            r["published_at"] is not None,
            r["published_at"].timestamp() if r["published_at"] is not None else 0.0,
            r["generation"],
        ),
        default=None,
    )


def next_generation(conn, target):
    return conn.execute(
        "SELECT coalesce(max(generation), 0) + 1 FROM index_generations WHERE collection = %s", (target.collection,)
    ).fetchone()[0]


def _in_transaction(conn, statements, attempts=None, sleep=None):
    """Runs statements() in one transaction with SET LOCAL lock_timeout; retried on a lock timeout. Re-raises the last
    error (nothing is changed by a failed attempt)."""
    attempts = PUBLISH_ATTEMPTS if attempts is None else attempts
    for attempt in range(1, attempts + 1):
        try:
            with conn.transaction():
                conn.execute(f"SET LOCAL lock_timeout = '{LOCK_TIMEOUT}'")
                return statements()
        except Exception as exc:
            if not is_lock_timeout(exc) or attempt == attempts:
                raise
            log.warning(
                f"[Generations] lock timeout ({attempt}/{attempts}): a reader holds the relation; "
                f"retrying in {RETRY_SLEEP_S} s"
            )
            (sleep or time.sleep)(RETRY_SLEEP_S)


def _create_views(conn, target, row):
    """Repoints the two typed views to the generation's tables, inside the caller's transaction, so a type mismatch
    leaves every view unchanged."""
    tables = row["tables"]
    statements = ddl.typed_views(target.collection, tables["children"], tables["parents"])
    for statement in statements:
        conn.execute(statement)
    conn.execute(ddl.view_grant(target.collection))


def _check_views(conn, target, row):
    tables = row["tables"]
    want = dict(zip(target.typed_views, (tables["children"], tables["parents"])))
    got = typed_view_tables(conn, target)
    if got != want:
        raise GenerationError(f"after the commit the views read {got}, expected {want}")
    for view in want:
        if conn.execute(f"SELECT 1 FROM {ident(view)} LIMIT 1").fetchone() is None:
            raise GenerationError(f"{view} returns no row after the commit")


# --- the steps -------------------------------------------------------------------------------------------------


def verify(conn, target, n, written=None, children=None, parents=None):
    """The gate before publish: every required index valid (by definition), rows present and equal to what the
    writer inserted, no row without an embedding and no orphan chunk; then ANALYZE. Returns the counts."""
    names = target.tables(n) if n is not None else {}
    children = children or names["children"]
    parents = parents or names["parents"]
    problems = []
    for kind, table in (("children", children), ("parents", parents)):
        if relkind(conn, table) != "r":
            problems.append(f"{table} is not a table")
            continue
        problems += [f"{table}: no valid {name} index" for name in missing_indexes(kind, index_rows(conn, table))]
    if problems:
        raise GenerationError("verify failed: " + "; ".join(problems))
    c_rows, no_embedding, orphans = conn.execute(
        f"SELECT count(*), count(*) FILTER (WHERE c.embedding IS NULL), "
        f"count(*) FILTER (WHERE NOT EXISTS (SELECT 1 FROM {ident(parents)} s WHERE s.section_id = c.section_id)) "
        f"FROM {ident(children)} c"
    ).fetchone()
    p_rows = conn.execute(f"SELECT count(*) FROM {ident(parents)}").fetchone()[0]
    counts = {"children": c_rows, "parents": p_rows}
    if c_rows <= 0 or p_rows <= 0:
        problems.append(f"empty tables ({counts})")
    if written is not None and (written.get("children"), written.get("parents")) != (c_rows, p_rows):
        problems.append(f"the writer inserted {written}, the tables hold {counts}")
    if no_embedding or orphans:
        problems.append(f"rows without embedding {no_embedding}, orphan chunks {orphans}")
    if problems:
        raise GenerationError("verify failed: " + "; ".join(problems))
    conn.execute(f"ANALYZE {ident(children)}")
    conn.execute(f"ANALYZE {ident(parents)}")
    log.info(f"[Generations] verified {children}: {c_rows} chunks, {p_rows} sections, indexes ok")
    return counts


def begin(conn, target, prov=None):
    """Starts a build's generation: records generation max+1 as staging (layout v2). Returns (n, None)."""
    ensure_tables(conn)
    n = next_generation(conn, target)
    tables = target.tables(n)
    for table in tables.values():
        if relkind(conn, table) is not None:
            raise GenerationError(f"{table} exists but no generation row names it; rename or drop it first")
    prov = prov or {}
    conn.execute(
        "INSERT INTO index_generations (collection, generation, status, layout, tables, git_commit, config_hash, "
        "embedding_model, embedding_revision) VALUES (%s, %s, 'staging', %s, %s::jsonb, %s, %s, %s, %s)",
        (
            target.collection,
            n,
            LAYOUT,
            json.dumps(tables),
            prov.get("git_commit"),
            prov.get("config_hash"),
            prov.get("embedding_model"),
            prov.get("embedding_revision"),
        ),
    )
    log.info(f"[Generations] build writes generation {n} ({', '.join(tables.values())})")
    return n, None


def annotate(conn, target, n, counts=None, timings=None):
    conn.execute(
        "UPDATE index_generations SET counts = coalesce(%s::jsonb, counts), "
        "timings = coalesce(timings, '{}'::jsonb) || coalesce(%s::jsonb, '{}'::jsonb) "
        "WHERE collection = %s AND generation = %s",
        (json.dumps(counts) if counts else None, json.dumps(timings) if timings else None, target.collection, n),
    )


def mark_failed(conn, target, n, reason):
    conn.execute(
        "UPDATE index_generations SET status = 'failed', detail = %s WHERE collection = %s AND generation = %s",
        (str(reason)[:2000], target.collection, n),
    )
    log.error(f"[Generations] generation {n} failed: {reason}")


def publish(conn, target, n, note=None):
    """Repoints both views to generation n in one transaction (lock_timeout, retried), n becomes live and the old live
    previous; then reads both viewdefs and one row through each view. Never drops a view.
    Returns (the generation that was live before, seconds)."""
    rows = generations(conn, target)
    row = next((r for r in rows if r["generation"] == n), None)
    if row is None or row["status"] == "dropped":
        raise GenerationError(f"generation {n} has no usable row")
    for table in row["tables"].values():
        if relkind(conn, table) != "r":
            raise GenerationError(f"{table} is missing: generation {n} cannot be published")
    before = _live(rows)
    started = time.monotonic()

    def statements():
        _create_views(conn, target, row)
        conn.execute(
            "UPDATE index_generations SET status = 'previous' WHERE collection = %s AND status = 'live' "
            "AND generation <> %s",
            (target.collection, n),
        )
        conn.execute(
            "UPDATE index_generations SET status = 'live', published_at = now(), detail = coalesce(%s, detail) "
            "WHERE collection = %s AND generation = %s",
            (note, target.collection, n),
        )

    try:
        _in_transaction(conn, statements)
    except Exception as exc:
        raise GenerationError(f"publish of generation {n} failed, the views are unchanged: {exc}") from exc
    previous = before["generation"] if before and before["generation"] != n else None
    try:
        _check_views(conn, target, row)
        record_documents(conn, target, row)  # the registry follows every publish (build, CLI, rollback)
    except Exception as exc:
        raise PublishedUnchecked(
            f"generation {n} is published, but the check after the commit failed: {exc}",
            previous,
            round(time.monotonic() - started, 3),
        ) from exc
    seconds = round(time.monotonic() - started, 3)
    log.info(f"[Generations] published generation {n} (previous {previous}) in {seconds} s")
    return previous, seconds


def rollback(conn, target):
    """Repoints the views to the newest previous generation; the generation that was live becomes previous."""
    rows = generations(conn, target)
    live, previous = _live(rows), _newest_previous(rows)
    if previous is None:
        raise RollbackRefused("no previous generation")
    moved = {"live": previous["generation"], "previous": live["generation"] if live else None}
    try:
        publish(conn, target, previous["generation"], note="rollback")
    except PublishedUnchecked as exc:  # committed: the views moved, so the caller still reloads and reports it
        log.warning(f"[Generations] {exc}")
        moved["warning"] = str(exc)
    log.info(f"[Generations] rollback: generation {moved['live']} live, generation {moved['previous']} previous")
    return moved


def _like(prefix, middle):
    return prefix.replace("_", "\\_") + "\\_" + middle + "\\_%"


def _relations(conn, kind, pattern):
    return [
        r[0]
        for r in conn.execute(
            "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = 'public' "
            "AND c.relkind = %s AND c.relname LIKE %s ORDER BY c.relname",
            (kind, pattern),
        ).fetchall()
    ]


def _drop(conn, kind, name, result):
    size = _size(conn, name)
    try:
        _in_transaction(conn, lambda: conn.execute(f"DROP {kind} {ident(name)}"), attempts=1)
    except Exception as exc:
        reason = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        result["errors"].append(f"{name}: {reason}")
        log.warning(f"[Generations] kept {name}: {reason}")
        return False
    result["dropped_tables" if kind == "TABLE" else "dropped_views"].append(name)
    result["freed_bytes"] += size
    return True


def housekeep(conn, target, stale=False):
    """Plain drops: views an older image left behind (<collection>_old_*), then every generation that is neither
    live nor the newest previous (its row becomes 'dropped'); with stale=True also the tables <collection>_staging_* and
    <collection>_old_* no row names (stale tables, only on an explicit request). Each drop is its own
    transaction with lock_timeout; a refused drop (a dependent view, a lock) is reported and leaves the row as it is.
    Last, the embedding-cache entries neither the live nor the previous generation used are pruned."""
    result = {"dropped_tables": [], "dropped_views": [], "kept": [], "errors": [], "freed_bytes": 0, "cache_pruned": 0}
    rows = generations(conn, target)
    keep = [r for r in (_live(rows), _newest_previous(rows)) if r]
    keep_numbers = {r["generation"] for r in keep}
    result["kept"] = [f"{r['generation']} ({r['status']})" for r in keep]
    read = set(typed_view_tables(conn, target).values())  # tables a typed view reads are never dropped
    for view in _relations(conn, "v", _like(target.collection, "old")):
        _drop(conn, "VIEW", view, result)
    for row in rows:
        if row["generation"] in keep_numbers or row["status"] == "dropped":
            continue
        complete = True
        for table in (row["tables"].get("children"), row["tables"].get("parents")):
            if not table or relkind(conn, table) is None:
                continue
            if table in read:
                complete = False
                continue
            dropped = _drop(conn, "TABLE", table, result)
            complete = complete and dropped
        if complete:
            conn.execute(
                "UPDATE index_generations SET status = 'dropped' WHERE collection = %s AND generation = %s",
                (target.collection, row["generation"]),
            )
    if keep_numbers:  # entries neither the live nor the previous generation used
        result["cache_pruned"] = cache.prune(conn, target.collection, min(keep_numbers))
    if stale:
        named = {t for r in rows for t in r["tables"].values()}
        for middle in ("staging", "old"):
            for table in _relations(conn, "r", _like(target.collection, middle)):
                if table not in named:
                    _drop(conn, "TABLE", table, result)
    log.info(
        f"[Generations] housekeeping: dropped {', '.join(result['dropped_views'] + result['dropped_tables']) or 'nothing'}; "
        f"kept generations {', '.join(result['kept']) or '-'}; {len(result['errors'])} errors; "
        f"freed {result['freed_bytes'] / 1e9:.1f} GB; pruned {result['cache_pruned']} cache entries"
    )
    return result


def stale_count(conn, target):
    named = {t for r in generations(conn, target) for t in r["tables"].values()}
    tables = _relations(conn, "r", _like(target.collection, "staging")) + _relations(
        conn, "r", _like(target.collection, "old")
    )
    return len([t for t in tables if t not in named]) + len(_relations(conn, "v", _like(target.collection, "old")))


# --- the document registry ------------------------------------------------------------------------------


_DOC_KEY = "coalesce(nullif(metadata ->> 'doc_key', ''), nullif(metadata ->> 'url', ''), '')"  # the writer's doc_key
_SECTION_SHA = "encode(sha256(convert_to(content, 'UTF8')), 'hex')"
_DOCUMENTS = (  # per document of a sections table: its section hashes in part order
    f"SELECT source, {_DOC_KEY} AS doc_key, string_agg({_SECTION_SHA}, '' ORDER BY part_index) AS h FROM {{}} GROUP BY 1, 2"
)


def document_changes(conn, target, n):
    """{added, changed, removed}: generation n's documents against the live generation's, matched by (source, doc_key);
    a document changed when a section text or the section order differs."""
    rows = generations(conn, target)
    new = next(r for r in rows if r["generation"] == n)
    live = _live(rows)
    a = _DOCUMENTS.format(ident(new["tables"]["parents"]))
    if live is None or live["generation"] == n:
        return {"added": conn.execute(f"SELECT count(*) FROM ({a}) a").fetchone()[0], "changed": 0, "removed": 0}
    b = _DOCUMENTS.format(ident(live["tables"]["parents"]))
    added, changed, removed = conn.execute(
        "SELECT count(*) FILTER (WHERE b.doc_key IS NULL), count(*) FILTER (WHERE a.h <> b.h), "
        f"count(*) FILTER (WHERE a.doc_key IS NULL) FROM ({a}) a FULL JOIN ({b}) b "
        "ON a.source = b.source AND a.doc_key = b.doc_key"
    ).fetchone()
    return {"added": added, "changed": changed, "removed": removed}


def record_documents(conn, target, row):
    """index_documents after a publish of row: one row per document; first_generation and last_generation never move
    backwards (a rollback publishes an older generation), the hash and url follow the newest generation."""
    return conn.execute(
        "INSERT INTO index_documents (document_id, collection, source, url, doc_key, first_generation, last_generation, "
        f"content_sha256) SELECT document_id, %(c)s::text, min(source), min(url), min({_DOC_KEY}), %(n)s::integer, %(n)s::integer, "
        f"encode(sha256(convert_to(string_agg({_SECTION_SHA}, '' ORDER BY part_index), 'UTF8')), 'hex') "
        f"FROM {ident(row['tables']['parents'])} GROUP BY document_id ON CONFLICT (collection, document_id) DO UPDATE SET "
        "first_generation = least(index_documents.first_generation, excluded.first_generation), "
        "last_generation = greatest(index_documents.last_generation, excluded.last_generation), "
        "content_sha256 = CASE WHEN excluded.last_generation >= index_documents.last_generation "
        "THEN excluded.content_sha256 ELSE index_documents.content_sha256 END, "
        "url = CASE WHEN excluded.last_generation >= index_documents.last_generation "
        "THEN excluded.url ELSE index_documents.url END",
        {"c": target.collection, "n": row["generation"]},
    ).rowcount


# --- the build lock ----------------------------------------------------------------------------------


def lock_status(conn, target):
    """pg_locks is the truth (a crashed holder's session lock is gone with its connection); the row adds who and since."""
    row = conn.execute(
        "SELECT l.pid FROM pg_locks l WHERE l.locktype = 'advisory' AND l.granted AND l.objsubid = 2 "
        "AND l.classid::bigint = %s::bigint AND l.objid::bigint = (hashtext(%s)::bigint & 4294967295)",
        (LOCK_NAMESPACE, target.collection),
    ).fetchone()
    status = {"locked": row is not None, "holder": None, "started_at": None, "pid": row[0] if row else None}
    if row is not None and conn.execute("SELECT to_regclass('public.index_build_lock') IS NOT NULL").fetchone()[0]:
        info = conn.execute(
            "SELECT holder, started_at FROM index_build_lock WHERE collection = %s", (target.collection,)
        ).fetchone()
        if info:
            status["holder"], status["started_at"] = info[0], _jsonable(info[1])
    return status


class BuildLock:
    """pg_try_advisory_lock(2, hashtext(collection)) on a direct autocommit connection kept open for the whole build;
    a second build, a rollback or a CLI step gets BuildBusy. No in-process lock: the database is the truth."""

    def __init__(self, target, holder, connect_fn):
        self.target, self.holder, self._connect, self.conn = target, holder, connect_fn, None

    def acquire(self):
        conn = self._connect(application_name="indexer-build")
        try:
            got = conn.execute(
                "SELECT pg_try_advisory_lock(%s::int, hashtext(%s))", (LOCK_NAMESPACE, self.target.collection)
            ).fetchone()[0]
            if not got:
                info = lock_status(conn, self.target)
                raise BuildBusy(info.get("holder"), info.get("started_at"))
            ensure_tables(conn)
            conn.execute(
                "INSERT INTO index_build_lock (collection, holder, started_at, pid) VALUES (%s, %s, now(), "
                "pg_backend_pid()) ON CONFLICT (collection) DO UPDATE SET holder = EXCLUDED.holder, "
                "started_at = EXCLUDED.started_at, pid = EXCLUDED.pid",
                (self.target.collection, self.holder),
            )
        except Exception:
            conn.close()
            raise
        self.conn = conn
        log.info(f"[Generations] build lock taken: {self.holder}")
        return self

    def release(self):
        if self.conn is None:
            return
        try:
            self.conn.execute(
                "DELETE FROM index_build_lock WHERE collection = %s AND pid = pg_backend_pid()",
                (self.target.collection,),
            )
            self.conn.execute(
                "SELECT pg_advisory_unlock(%s::int, hashtext(%s))", (LOCK_NAMESPACE, self.target.collection)
            )
        except Exception as exc:
            log.warning(f"[Generations] releasing the build lock: {exc} (the session lock ends with the connection)")
        finally:
            self.conn.close()
            self.conn = None
            log.info(f"[Generations] build lock released: {self.holder}")

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
        return False


def status(conn, target):
    """The status the CLI prints: the lock, the live names, the generations."""
    rows = generations(conn, target)
    return {
        "status": "ok",
        "lock": lock_status(conn, target),
        "typed_views": typed_view_tables(conn, target),
        "live": _jsonable(_live(rows)),
        "previous": _jsonable(_newest_previous(rows)),
        "generations": _jsonable(rows),
        "stale_tables": stale_count(conn, target),
    }


# --- the frozen-input comparison ----------------------------------------------------------------

_PARENT_ID = (
    "md5(coalesce(metadata->>'url', '') || '|' || coalesce(metadata->>'page_number', '') || '|' || "
    "coalesce(metadata->>'part_index', '') || '|' || md5(coalesce(content, '')) || '|' || "
    "md5((metadata - 'section_id' - 'scrapedAt')::text))"
)


def compare(conn, target, a, b):
    """Parent and child identities of generations a and b (random ids and scrapedAt left out), compared as multisets
    through temporary tables of this session."""
    known = {r["generation"]: r for r in generations(conn, target)}
    rows = {n: known.get(n) or {"layout": "v2", "tables": target.tables(n)} for n in (a, b)}
    for row in rows.values():
        for table in row["tables"].values():
            if relkind(conn, table) != "r":
                raise GenerationError(f"{table} does not exist")
    temp = ("cmp_pa", "cmp_pb", "cmp_ca", "cmp_cb")
    conn.execute("DROP TABLE IF EXISTS " + ", ".join("pg_temp." + t for t in temp))
    for side, n in (("a", a), ("b", b)):
        tables = rows[n]["tables"]
        conn.execute(
            f"CREATE TEMP TABLE cmp_p{side} AS SELECT section_id, {_PARENT_ID} AS id, "
            f"metadata->>'url' AS url, metadata->>'page_number' AS page, "
            f"coalesce(metadata->>'origin', '-') || '/' || coalesce(metadata->>'source_type', '-') AS source "
            f"FROM {ident(tables['parents'])}"
        )
        conn.execute(
            f"CREATE TEMP TABLE cmp_c{side} AS SELECT md5(coalesce(p.id, 'orphan') || '|' || md5(c.content)) AS id "
            f"FROM {ident(tables['children'])} c LEFT JOIN cmp_p{side} p "
            "ON p.section_id = c.section_id"
        )
    report: dict[str, Any] = {"generations": {}}
    for side, n in (("a", a), ("b", b)):
        report["generations"][n] = {
            "parents": conn.execute(f"SELECT count(*) FROM cmp_p{side}").fetchone()[0],
            "children": conn.execute(f"SELECT count(*) FROM cmp_c{side}").fetchone()[0],
        }
    for kind, (ta, tb) in (("parents", ("cmp_pa", "cmp_pb")), ("children", ("cmp_ca", "cmp_cb"))):
        only_a, only_b = conn.execute(
            f"SELECT (SELECT count(*) FROM (SELECT id FROM {ta} EXCEPT ALL SELECT id FROM {tb}) d), "
            f"(SELECT count(*) FROM (SELECT id FROM {tb} EXCEPT ALL SELECT id FROM {ta}) d)"
        ).fetchone()
        report[kind] = {"only_a": only_a, "only_b": only_b}
    sources = {}
    for side, n in (("a", a), ("b", b)):
        for source, count in conn.execute(f"SELECT source, count(*) FROM cmp_p{side} GROUP BY 1 ORDER BY 1").fetchall():
            sources.setdefault(source, {})[n] = count
    report["sources"] = sources
    samples = []
    for side, other, n in (("a", "b", a), ("b", "a", b)):
        samples += [
            f"only in {n}: {url} page {page}"
            for url, page in conn.execute(
                f"SELECT url, page FROM cmp_p{side} WHERE id IN (SELECT id FROM cmp_p{side} EXCEPT SELECT id FROM cmp_p{other}) "
                f"ORDER BY url, page LIMIT 5"
            ).fetchall()
        ]
    report["samples"] = samples
    conn.execute("DROP TABLE IF EXISTS " + ", ".join("pg_temp." + t for t in temp))
    report["identical"] = all(report[k]["only_a"] == 0 and report[k]["only_b"] == 0 for k in ("parents", "children"))
    return report


def compare_lines(report, a, b):
    lines = [
        f"generation {n}: {c['parents']} parents, {c['children']} children" for n, c in report["generations"].items()
    ]
    for kind in ("parents", "children"):
        lines.append(f"{kind} only in {a}: {report[kind]['only_a']}, only in {b}: {report[kind]['only_b']}")
    for source, counts in sorted(report["sources"].items()):
        lines.append(f"source {source}: {counts.get(a, 0)} | {counts.get(b, 0)}")
    lines += report["samples"]
    lines.append("RESULT " + ("identical" if report["identical"] else "DIFFERENT"))
    return lines


# --- the CLI ---------------------------------------------------------------------------------------------------


USAGE = "usage: indexer build | status | housekeep [--stale] | rollback | compare <a> <b> | publish <n>"


def main(argv, *, target, connect_fn, out=print):
    """status, housekeep, rollback and compare with their output; indexer.cli adds publish and wires the arguments."""
    args = list(argv)
    command = args[0] if args else ""
    if (
        command not in ("status", "housekeep", "rollback", "compare")
        or (command == "compare" and (len(args) != 3 or not all(a.isdigit() for a in args[1:])))
        or (command == "housekeep" and args[1:] not in ([], ["--stale"]))
        or (command in ("status", "rollback") and len(args) != 1)
    ):
        out(USAGE)
        return 2
    if command in ("status", "compare"):
        conn = connect_fn(application_name=f"indexer-cli {command}")
        try:
            if command == "status":
                out(json.dumps(status(conn, target), indent=1, default=str))
                return 0
            report = compare(conn, target, int(args[1]), int(args[2]))
            for line in compare_lines(report, int(args[1]), int(args[2])):
                out(line)
            return 0 if report["identical"] else 1
        finally:
            conn.close()
    lock = BuildLock(target, f"cli {' '.join(args)}", connect_fn)
    try:
        lock.acquire()
    except BuildBusy as exc:
        out(f"busy: {exc}")
        return 1
    try:
        conn = connect_fn(application_name=f"indexer-cli {command}")
        try:
            if command == "housekeep":
                result = housekeep(conn, target, stale=args[1:] == ["--stale"])
                for name in result["dropped_views"]:
                    out(f"dropped view {name}")
                for name in result["dropped_tables"]:
                    out(f"dropped table {name}")
                for error in result["errors"]:
                    out(f"kept {error}")
                out(
                    f"kept generations {', '.join(result['kept']) or '-'}; {len(result['errors'])} errors; "
                    f"freed {result['freed_bytes'] / 1e9:.1f} GB"
                )
                return 1 if result["errors"] else 0
            else:
                moved = rollback(conn, target)
                out(f"rollback: generation {moved['live']} live, generation {moved['previous']} previous")
        finally:
            conn.close()
    except GenerationError as exc:
        out(f"refused: {exc}")
        return 1
    finally:
        lock.release()
    return 0
