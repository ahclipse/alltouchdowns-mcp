"""Opening the database: read-only, with a limit on work, a few queries at a time."""

from __future__ import annotations

import itertools
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from alltouchdowns_mcp import db as database
from alltouchdowns_mcp.config import Settings
from alltouchdowns_mcp.db import Coverage, open_database
from alltouchdowns_mcp.errors import ToolRefusal
from alltouchdowns_mcp.tools.leaderboard import get_leaderboard
from tests.conftest import FIXTURE_DB

LIMIT = 20_000_000
HUGE_QUERY = "SELECT COUNT(*) FROM touchdowns a, touchdowns b, touchdowns c"  # 20 billion rows


def test_writes_are_refused() -> None:
    with open_database(FIXTURE_DB, step_limit=LIMIT) as db:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            db.conn.execute("DELETE FROM touchdowns")
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            db.conn.execute("CREATE TABLE scratch (x)")


def test_a_query_that_does_too_much_work_is_abandoned() -> None:
    with pytest.raises(ToolRefusal) as refusal, open_database(FIXTURE_DB, step_limit=LIMIT) as db:
        db.conn.execute(HUGE_QUERY).fetchall()

    assert refusal.value.code == "too_broad"


def test_the_limit_is_on_work_done_and_not_on_time_taken(monkeypatch: pytest.MonkeyPatch) -> None:
    """A busy host makes every query slower. That must not turn an ordinary query into a
    refused one, so the limit may not depend on the clock."""
    an_hour_passes_at_every_look = itertools.count(0, 3600)
    monkeypatch.setattr(time, "monotonic", lambda: next(an_hour_passes_at_every_look))
    monkeypatch.setattr(time, "time", lambda: next(an_hour_passes_at_every_look))

    with open_database(FIXTURE_DB, step_limit=LIMIT) as db:
        rows = get_leaderboard(db, limit=50).rows

    assert len(rows) >= 50


def test_ordinary_calls_use_a_small_part_of_the_limit() -> None:
    """If this fails, either a query got much more expensive or the limit is too tight."""
    steps = 0

    def count() -> bool:
        nonlocal steps
        steps += 1000
        return False

    with open_database(FIXTURE_DB, step_limit=LIMIT) as db:
        db.conn.set_progress_handler(count, 1000)
        get_leaderboard(db, limit=50)  # both seasons ranked together: the heaviest call here

    assert 0 < steps < Settings(db_path=FIXTURE_DB).query_step_limit / 20


def test_the_same_call_is_stopped_at_the_same_point_on_any_machine() -> None:
    """With no allowance at all, a query is stopped at its first report: a count of steps,
    which no amount of speed or slowness changes."""
    with pytest.raises(ToolRefusal) as refusal, open_database(FIXTURE_DB, step_limit=0) as db:
        db.conn.execute("SELECT COUNT(*) FROM touchdowns a, touchdowns b").fetchall()

    assert refusal.value.code == "too_broad"


def test_the_connection_is_closed_afterwards() -> None:
    with open_database(FIXTURE_DB, step_limit=LIMIT) as db:
        conn = db.conn
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        conn.execute("SELECT 1")


def test_many_calls_at_once_all_finish(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calls beyond the number of slots wait their turn, and none is refused for the wait."""

    def one_call(_: int) -> int:
        with open_database(FIXTURE_DB, step_limit=LIMIT) as db:
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
        open_database(FIXTURE_DB, step_limit=LIMIT),
        pytest.raises(ToolRefusal) as refusal,
        open_database(FIXTURE_DB, step_limit=LIMIT),
    ):
        pass
    assert refusal.value.code == "busy"

    with open_database(FIXTURE_DB, step_limit=LIMIT):  # the slot was given back
        pass


def test_coverage_describes_what_is_loaded() -> None:
    with open_database(FIXTURE_DB, step_limit=LIMIT) as db:
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
