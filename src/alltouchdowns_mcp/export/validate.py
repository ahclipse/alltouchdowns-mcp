"""Export step 3: check the new database, then swap it into place.

Each check returns a list of problems as plain sentences. An empty list means it passed.
Nothing here changes the database; a failed check simply means the new file is never published
and the server keeps reading the previous one.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from alltouchdowns_mcp import legal
from alltouchdowns_mcp.export.source import open_immutable

# The only columns that hold people's names, and so the only ones the name allowlist applies to.
_NAME_COLUMNS = frozenset({("players", "display_name"), ("touchdowns", "scorer_name")})

_TEXT_COLUMNS = """
SELECT m.name, c.name
FROM sqlite_master m, pragma_table_info(m.name) c
WHERE m.type = 'table' AND c.type = 'TEXT'
ORDER BY m.name, c.cid
"""


@dataclass(frozen=True)
class Expectations:
    """What a healthy export looks like. The ranges come from 27 seasons of real data, where
    a completed season has 259 to 285 games and 1,172 to 1,543 touchdowns."""

    franchises: int = 32
    first_season: int = 1999
    season_games: tuple[int, int] = (255, 290)
    season_touchdowns: tuple[int, int] = (1100, 1700)
    min_games: int = 7400
    min_touchdowns: int = 36800
    min_players: int = 3700


def check_legal(path: Path) -> list[str]:
    """Scan every text value in the database for banned terms.

    The columns are read from the schema, so a column added later is scanned without anyone
    having to remember to list it here.
    """
    problems: list[str] = []
    db = open_immutable(path)
    try:
        for table, column in db.execute(_TEXT_COLUMNS).fetchall():
            is_name = (table, column) in _NAME_COLUMNS
            values = db.execute(f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL")
            for (value,) in values:
                if legal.find_banned(value, allow_names=is_name):
                    problems.append(f"banned term in {table}.{column}: {value!r}")
    finally:
        db.close()
    return problems


def check_structure(
    path: Path,
    *,
    only_seasons: Sequence[int] | None = None,
    expect: Expectations | None = None,
) -> list[str]:
    """Check internal consistency and that the contents are a plausible size.

    `only_seasons` is given when the export was limited to chosen seasons (the test fixture).
    Whole-history checks are skipped then, and the season list must match exactly.
    """
    expect = expect or Expectations()
    problems: list[str] = []
    db = open_immutable(path)
    try:
        problems += _check_consistency(db)
        problems += _check_seasons(db, only_seasons, expect)
        if only_seasons is None:
            problems += _check_totals(db, expect)
    except sqlite3.DatabaseError as error:
        return [f"the database file is damaged: {error}"]
    finally:
        db.close()
    return problems


def check_against_previous(path: Path, previous: Path) -> list[str]:
    """Compare with the database currently being served. History should only ever grow, apart
    from small corrections, so a shrinking count means something upstream went wrong."""
    new, old = _totals(path), _totals(previous)
    problems: list[str] = []

    if not old["games"] - 2 <= new["games"] <= old["games"] + 40:
        problems.append(f"games went from {old['games']} to {new['games']}")
    if not old["touchdowns"] * 0.995 <= new["touchdowns"] <= old["touchdowns"] + 250:
        problems.append(f"touchdowns went from {old['touchdowns']} to {new['touchdowns']}")
    if new["players"] < old["players"] * 0.99:
        problems.append(f"players went from {old['players']} to {new['players']}")
    return problems


def publish(candidate: Path, out: Path) -> None:
    """Replace `out` with `candidate` atomically, keeping the old file as `<out>.prev`.

    A reader that already has the old file open keeps reading it; the next open sees the new
    one. `candidate` must be in the same directory so the rename stays on one filesystem.
    """
    if out.exists():
        shutil.copy2(out, out.with_name(out.name + ".prev"))
    os.replace(candidate, out)


def _check_consistency(db: sqlite3.Connection) -> list[str]:
    problems: list[str] = []

    if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
        problems.append("integrity check failed")
    if db.execute("PRAGMA foreign_key_check").fetchone():
        problems.append("a row refers to a team, game or player that was not exported")

    checks = {
        "a game's touchdown_count disagrees with its touchdown rows": """
            SELECT COUNT(*) FROM games g
            WHERE g.touchdown_count <>
                  (SELECT COUNT(*) FROM touchdowns t WHERE t.game_id = g.game_id)""",
        "a game's touchdowns are not numbered 1..n": """
            SELECT COUNT(*) FROM (
              SELECT game_id FROM touchdowns
              GROUP BY game_id HAVING MIN(game_seq) <> 1 OR MAX(game_seq) <> COUNT(*))""",
        "a touchdown's season, week or postseason flag disagrees with its game": """
            SELECT COUNT(*) FROM touchdowns t JOIN games g ON g.game_id = t.game_id
            WHERE t.season <> g.season OR t.week <> g.week OR t.postseason <> g.postseason""",
        "a player's stored totals disagree with their touchdown rows": """
            SELECT COUNT(*) FROM players p
            WHERE p.td_regular + p.td_postseason <>
                  (SELECT COUNT(*) FROM touchdowns t WHERE t.player_slug = p.slug)""",
        "a game flagged incomplete has touchdown rows": """
            SELECT COUNT(*) FROM games WHERE touchdowns_incomplete AND touchdown_count > 0""",
    }
    for problem, sql in checks.items():
        if (count := db.execute(sql).fetchone()[0]) > 0:
            problems.append(f"{problem} ({count} rows)")
    return problems


def _check_seasons(
    db: sqlite3.Connection, only_seasons: Sequence[int] | None, expect: Expectations
) -> list[str]:
    problems: list[str] = []
    seasons = [s for (s,) in db.execute("SELECT season FROM seasons ORDER BY season")]

    if only_seasons is not None:
        if seasons != sorted(set(only_seasons)):
            problems.append(f"expected seasons {sorted(set(only_seasons))}, found {seasons}")
    elif not seasons or seasons != list(range(expect.first_season, seasons[-1] + 1)):
        problems.append(f"seasons are not continuous from {expect.first_season}: {seasons}")

    # Size ranges apply to completed seasons only; the current season is still filling up.
    per_season = db.execute("""
        SELECT s.season,
               (SELECT COUNT(*) FROM games g WHERE g.season = s.season),
               (SELECT COUNT(*) FROM touchdowns t WHERE t.season = s.season)
        FROM seasons s WHERE s.complete = 1 ORDER BY s.season""")
    for season, games, touchdowns in per_season:
        if not expect.season_games[0] <= games <= expect.season_games[1]:
            problems.append(f"season {season} has {games} games")
        if not expect.season_touchdowns[0] <= touchdowns <= expect.season_touchdowns[1]:
            problems.append(f"season {season} has {touchdowns} touchdowns")
    return problems


def _check_totals(db: sqlite3.Connection, expect: Expectations) -> list[str]:
    problems: list[str] = []
    franchises = db.execute("SELECT COUNT(DISTINCT franchise) FROM teams").fetchone()[0]
    if franchises != expect.franchises:
        problems.append(f"{franchises} franchises, expected {expect.franchises}")

    minimums = {
        "games": expect.min_games,
        "touchdowns": expect.min_touchdowns,
        "players": expect.min_players,
    }
    for table, minimum in minimums.items():
        count = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if count < minimum:
            problems.append(f"only {count} {table}, expected at least {minimum}")
    return problems


def _totals(path: Path) -> dict[str, int]:
    db = open_immutable(path)
    try:
        return {
            table: db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("games", "touchdowns", "players")
        }
    finally:
        db.close()
