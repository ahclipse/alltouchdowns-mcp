"""Ground truth: each case's correct answer, computed with SQL at eval time.

The answer is never stored. It is computed from the same database file the server is serving,
so a data refresh can never leave a stale answer key behind.

For the truth to be independent of the server, its SQL may only use columns the export copied
straight from the source. Columns the export *derives* (labels, stored totals, the
Championship flag) are off limits: if the export got one of those wrong, the server and a
truth query built on it would agree with each other and the eval would pass anyway.
`run_independent` is what enforces that.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import quote

from evals.schema import Case

# Derived by the export, so not readable from truth SQL.
DERIVED_COLUMNS = frozenset({
    "week_label", "stage", "touchdown_count", "touchdowns_incomplete", "played_at",  # games
    "search_name", "name_suffix", "first_season", "last_season",  # players
    "td_regular", "td_postseason", "team_abbrs",
    "label",  # teams
    "scorer_name", "clock",  # touchdowns: merged from two columns / reformatted
})  # fmt: skip
DERIVED_TABLES = frozenset({"seasons", "meta"})  # built entirely by the export

# The Championship is the one fact truth SQL may not look up, so it is written down here by
# hand: the final game of each fixture season was played in postseason week 21.
CHAMPIONSHIP_WEEK = {2007: 21, 2019: 21}

# An answer is a number, a yes/no, a set of items, or items in ranked groups (a group holds
# items that are tied, in which case any order among them is right).
Truth = int | bool | frozenset[str] | list[frozenset[str]]


def open_database(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{quote(str(path))}?mode=ro&immutable=1", uri=True)


def cases_that_hold(conn: sqlite3.Connection, cases: Sequence[Case]) -> list[Case]:
    """The cases that are valid for this database. A few are marked `requires = "fixture"`:
    they are only true while 2007 and 2019 are the only seasons loaded."""
    (coverage,) = conn.execute("SELECT value FROM meta WHERE key = 'coverage'").fetchone()
    return [case for case in cases if coverage != "full" or case.requires == "any"]


def sql_problems(sql: str) -> list[str]:
    """Reasons a query is not acceptable, before it is even run. Empty means none."""
    problems = []
    statement = sql.strip().rstrip(";")
    if not re.match(r"(?i)(select|with)\b", statement):
        problems.append("must be a single SELECT (or WITH ... SELECT)")
    if ";" in statement:
        problems.append("must be one statement")
    return problems


def run_independent(conn: sqlite3.Connection, sql: str) -> list[tuple]:
    """Run a query with everything the export derives made unreadable.

    SQLite enforces this itself: while a statement is compiled it asks the authorizer about
    every column the statement reads. So a name in upper case, a column reached through an
    alias and `SELECT *` are all refused, which a search of the SQL text would miss.
    """
    conn.set_authorizer(_refuse_derived)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.set_authorizer(None)


def _refuse_derived(action: int, table: str | None, column: str | None, *_: object) -> int:
    reads_derived = action == sqlite3.SQLITE_READ and (
        (table or "").lower() in DERIVED_TABLES or (column or "").lower() in DERIVED_COLUMNS
    )
    return sqlite3.SQLITE_DENY if reads_derived else sqlite3.SQLITE_OK


def compute(conn: sqlite3.Connection, case: Case) -> Truth:
    """Run a case's truth SQL and shape the result for its kind of answer."""
    assert case.truth_sql is not None
    rows = run_independent(conn, case.truth_sql)

    if case.answer_kind in ("number", "boolean"):
        if len(rows) != 1 or len(rows[0]) != 1 or not isinstance(rows[0][0], int):
            raise ValueError(f"{case.id}: expected one integer, got {rows[:3]}")
        return rows[0][0] if case.answer_kind == "number" else bool(rows[0][0])

    if case.answer_kind == "set":
        if not rows or any(len(row) != 1 or row[0] is None for row in rows):
            raise ValueError(f"{case.id}: expected rows of one non-null item, got {rows[:3]}")
        return frozenset(str(row[0]) for row in rows)

    # ordered_list: rows of (item, rank). Equal ranks are a tie group. Rows must already be
    # sorted so that equal ranks are adjacent.
    if not rows or any(len(row) != 2 or row[0] is None for row in rows):
        raise ValueError(f"{case.id}: expected rows of (item, rank), got {rows[:3]}")
    groups: list[tuple[object, set[str]]] = []
    for item, rank in rows:
        if groups and groups[-1][0] == rank:
            groups[-1][1].add(str(item))
        else:
            groups.append((rank, {str(item)}))
    ranks = [rank for rank, _ in groups]
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"{case.id}: rows with equal rank must be adjacent")
    return [frozenset(items) for _, items in groups]


def trap_value(conn: sqlite3.Connection, case: Case) -> int | None:
    """For a decline case, the number an agent would report if it answered anyway."""
    if not case.trap_sql:
        return None
    ((value,),) = run_independent(conn, case.trap_sql)
    return value


def fold(text: str) -> str:
    """Compare names loosely: case, extra spaces and periods ("St Louis") do not matter."""
    return " ".join(text.replace(".", "").casefold().split())


def franchises(conn: sqlite3.Connection) -> dict[str, str]:
    """Every way of writing a team, folded, mapped to its franchise.

    An agent may answer "SD", "LAC", "San Diego" or "Los Angeles (LAC)" for the same team, and
    each is right. A city is included only when a single franchise has ever used it, so a bare
    "Los Angeles" is not accepted as naming either team there.
    """
    names: dict[str, str] = {}
    city_owners: dict[str, set[str]] = {}
    for abbr, franchise, city, label in conn.execute(
        "SELECT abbr, franchise, city, label FROM teams"
    ):
        names[fold(abbr)] = names[fold(label)] = franchise
        city_owners.setdefault(fold(city), set()).add(franchise)
    for city, owners in city_owners.items():
        if len(owners) == 1:
            names.setdefault(city, next(iter(owners)))
    return names
