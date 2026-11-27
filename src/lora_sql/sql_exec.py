"""SQL extraction, normalisation and execution-based comparison on SQLite."""

from __future__ import annotations

import re
import sqlite3
import time
from collections import Counter

_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Pull the first SQL statement out of a model reply (handles ``` fences)."""
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    text = text.strip()
    if ";" in text:
        text = text.split(";", 1)[0]
    return text.strip()


def normalize_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";").replace('"', "'")
    return re.sub(r"\s+", " ", sql).lower()


def _connect(db_path: str) -> sqlite3.Connection:
    # Read-only so a generated DROP/DELETE can never touch the benchmark DB.
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def execute(conn: sqlite3.Connection, sql: str, timeout_s: float = 5.0):
    """Run ``sql``; returns (ok, rows_or_error). Aborts queries slower than ``timeout_s``."""
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    try:
        return True, conn.execute(sql).fetchall()
    except Exception as e:  # noqa: BLE001 - any failure means "invalid prediction"
        return False, str(e)
    finally:
        conn.set_progress_handler(None, 0)


def results_match(gold_rows: list, pred_rows: list, ordered: bool) -> bool:
    if ordered:
        return gold_rows == pred_rows
    return Counter(gold_rows) == Counter(pred_rows)


def execution_match(db_path: str, gold_sql: str, pred_sql: str, timeout_s: float = 5.0) -> dict:
    """Spider-style execution accuracy for one example."""
    conn = _connect(db_path)
    try:
        gold_ok, gold = execute(conn, gold_sql, timeout_s)
        pred_ok, pred = execute(conn, pred_sql, timeout_s)
    finally:
        conn.close()
    if not gold_ok:
        return {"valid": pred_ok, "match": None, "error": f"gold failed: {gold}"}
    if not pred_ok:
        return {"valid": False, "match": False, "error": pred}
    ordered = "order by" in gold_sql.lower()
    return {"valid": True, "match": results_match(gold, pred, ordered), "error": None}


def schema_from_sqlite(db_path: str) -> str:
    """The CREATE TABLE statements of a database, used as the prompt's schema."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL "
            "AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    finally:
        conn.close()
    return "\n".join(r[0].strip() + ";" for r in rows)


def is_valid_on_ddl(ddl: str, sql: str) -> bool:
    """Does ``sql`` parse and run against an empty DB built from ``ddl``?"""
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(ddl)
        ok, _ = execute(conn, sql)
        return ok
    except sqlite3.Error:
        return False
    finally:
        conn.close()
