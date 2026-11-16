"""Opening the database: read-only, with a deadline, a few queries at a time."""

from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from alltouchdowns_mcp import db as database
from alltouchdowns_mcp.db import Coverage, open_database
from alltouchdowns_mcp.errors import ToolRefusal
from alltouchdowns_mcp.tools.leaderboard import get_leaderboard
from tests.conftest import FIXTURE_DB

SLOW_QUERY = "SELECT COUNT(*) FROM touchdowns a, touchdowns b, touchdowns c"  # a 3-way cross join


def test_writes_are_refused() -> None:
    with open_database(FIXTURE_DB, timeout_ms=2000) as db:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            db.conn.execute("DELETE FROM touchdowns")
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            db.conn.execute("CREATE TABLE scratch (x)")


def test_a_query_that_runs_too_long_is_abandoned() -> None:
    with pytest.raises(ToolRefusal) as refusal, open_database(FIXTURE_DB, timeout_ms=30) as db:
        db.conn.execute(SLOW_QUERY).fetchall()

    assert refusal.value.code == "timeout"


def test_the_connection_is_closed_afterwards() -> None:
    with open_database(FIXTURE_DB, timeout_ms=2000) as db:
        conn = db.conn
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        conn.execute("SELECT 1")


def test_many_calls_at_once_all_finish(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calls beyond the number of slots wait their turn. Their deadline starts when they get
    a slot, so none of them times out just for having queued."""

    def one_call(_: int) -> int:
        with open_database(FIXTURE_DB, timeout_ms=250) as db:
            return len(get_leaderboard(db, limit=50).rows)

    with ThreadPoolExecutor(max_workers=40) as pool:
        results = list(pool.map(one_call, range(200)))

    assert len(set(results)) == 1 and results[0] >= 50


def test_a_call_that_cannot_get_a_slot_in_time_is_told_the_server_is_busy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(database, "_QUERY_SLOTS", threading.BoundedSemaphore(1))
    monkeypatch.setattr(database, "_MAX_WAIT_FOR_SLOT_SECONDS", 0.05)

    # The first open holds the only slot while the second one tries, and fails, to get it.
    with (
        open_database(FIXTURE_DB, timeout_ms=2000),
        pytest.raises(ToolRefusal) as refusal,
        open_database(FIXTURE_DB, timeout_ms=2000),
    ):
        pass
    assert refusal.value.code == "busy"

    with open_database(FIXTURE_DB, timeout_ms=2000):  # the slot was given back
        pass


def test_coverage_describes_what_is_loaded() -> None:
    with open_database(FIXTURE_DB, timeout_ms=2000) as db:
        coverage = db.coverage

    assert coverage.seasons == (2007, 2019) and coverage.describe() == "2007 and 2019"
    assert coverage.season_in_progress is False  # both seasons have their Championship
    assert replace(coverage, seasons=tuple(range(1999, 2027))).describe() == "1999 to 2026"
    with pytest.raises(ToolRefusal) as refusal:
        coverage.require(2012)
    assert refusal.value.code == "not_covered" and "2007 and 2019" in str(refusal.value)


def test_a_season_without_its_championship_is_in_progress(edge_db: database.Database) -> None:
    assert isinstance(edge_db.coverage, Coverage)
    assert edge_db.coverage.season_in_progress is True
