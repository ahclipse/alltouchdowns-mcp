"""Shared test helpers.

`SiteDb` builds a tiny database with the same tables and columns as the site's, so export tests
can set up exactly the situation they are about (a held-back game, a banned name, ...) without
needing the private site database.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path

import pytest

from alltouchdowns_mcp.db import Database, open_database

# The day every export test pretends it is. Games dated within 7 days of it are "recent".
TODAY = date(2026, 11, 16)
RECENT = "2026-11-15"
OLD = "2026-10-04"

FIXTURE_DB = Path(__file__).parent / "fixtures" / "seasons-2007-2019.db"

_SITE_SCHEMA = """
CREATE TABLE teams (id TEXT PRIMARY KEY, city TEXT NOT NULL);
CREATE TABLE team_aliases (
  abbr TEXT PRIMARY KEY, team_id TEXT NOT NULL, city TEXT NOT NULL,
  first_season INTEGER NOT NULL, last_season INTEGER);
CREATE TABLE players (
  id INTEGER PRIMARY KEY AUTOINCREMENT, slug TEXT NOT NULL, display_name TEXT NOT NULL,
  first_name TEXT, last_name TEXT, position TEXT, source TEXT NOT NULL, source_id TEXT NOT NULL);
CREATE TABLE games (
  id TEXT PRIMARY KEY, season INTEGER NOT NULL, week INTEGER NOT NULL,
  postseason INTEGER NOT NULL, home_team_id TEXT NOT NULL, away_team_id TEXT NOT NULL,
  home_abbr TEXT NOT NULL, away_abbr TEXT NOT NULL, home_score INTEGER, away_score INTEGER,
  date TEXT NOT NULL, source TEXT NOT NULL, kickoff_time TEXT,
  status TEXT NOT NULL DEFAULT 'final');
CREATE TABLE touchdowns (
  id INTEGER PRIMARY KEY AUTOINCREMENT, game_id TEXT NOT NULL, season INTEGER NOT NULL,
  week INTEGER NOT NULL, team_id TEXT NOT NULL, team_abbr TEXT NOT NULL, player_id INTEGER,
  player_name TEXT, td_type TEXT NOT NULL, quarter INTEGER, clock TEXT,
  game_seq INTEGER NOT NULL, yards INTEGER, description TEXT,
  home_score_after INTEGER, away_score_after INTEGER,
  source TEXT NOT NULL, source_id TEXT NOT NULL);
"""

# (franchise id, city), then (abbreviation, franchise id, era city, first season, last season).
_TEAMS = [
    ("KC", "Kansas City"),
    ("DEN", "Denver"),
    ("NE", "New England"),
    ("LA", "Los Angeles"),
    ("LAC", "Los Angeles"),
    ("NYG", "New York"),
    ("NYJ", "New York"),
    ("LV", "Las Vegas"),
]
_ALIASES = [
    ("KC", "KC", "Kansas City", 1999, None),
    ("DEN", "DEN", "Denver", 1999, None),
    ("NE", "NE", "New England", 1999, None),
    ("LA", "LA", "Los Angeles", 2016, None),
    ("LAC", "LAC", "Los Angeles", 2017, None),
    ("NYG", "NYG", "New York", 1999, None),
    ("NYJ", "NYJ", "New York", 1999, None),
    ("LV", "LV", "Las Vegas", 2020, None),
    ("SD", "LAC", "San Diego", 1999, 2016),
    ("OAK", "LV", "Oakland", 1999, 2019),
    ("STL", "LA", "St. Louis", 1999, 2015),
]

DESCRIPTION = "(10:00) J.Example up the middle for 5 yards, TOUCHDOWN. SECRET-PLAY-TEXT"


def _first_sunday(season: int) -> date:
    """Roughly when a season starts: early September of the year it is named after."""
    return date(season, 9, 7)


class SiteDb:
    """A small database shaped like the site's. Call `save()` when the rows are in."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._db = sqlite3.connect(path)
        self._db.executescript(_SITE_SCHEMA)
        self._db.executemany("INSERT INTO teams VALUES (?, ?)", _TEAMS)
        self._db.executemany("INSERT INTO team_aliases VALUES (?, ?, ?, ?, ?)", _ALIASES)
        self._franchise = {abbr: team_id for abbr, team_id, *_ in _ALIASES}
        self._games: dict[str, tuple[int, int]] = {}
        self._next_seq: dict[str, int] = {}

    def player(self, name: str, slug: str | None = None, position: str = "RB") -> int:
        slug = slug or name.lower().replace(" ", "-").replace(".", "")
        cursor = self._db.execute(
            "INSERT INTO players (slug, display_name, first_name, last_name, position, source,"
            " source_id) VALUES (?, ?, ?, ?, ?, 'nflverse', ?)",
            (slug, name, "LegalFirst", "LegalLast", position, f"00-{slug}"),
        )
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    def game(
        self,
        season: int,
        week: int,
        away: str,
        home: str,
        *,
        score: tuple[int, int] | None = (17, 20),
        postseason: bool = False,
        status: str = "final",
        source: str = "nflverse",
        date: str | None = None,
    ) -> str:
        """Add a game and return its id. `score` is (away, home); None means not played.

        Without a `date`, the game is dated by its season and week, like a real schedule.
        """
        game_id = f"{season}-{week:02}-{away}-{home}"
        date = date or str(_first_sunday(season) + timedelta(weeks=week - 1))
        away_score, home_score = score if score else (None, None)
        self._db.execute(
            "INSERT INTO games (id, season, week, postseason, home_team_id, away_team_id,"
            " home_abbr, away_abbr, home_score, away_score, date, source, kickoff_time, status)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '13:00', ?)",
            (
                game_id,
                season,
                week,
                int(postseason),
                self._franchise[home],
                self._franchise[away],
                home,
                away,
                home_score,
                away_score,
                date,
                source,
                status,
            ),
        )
        self._games[game_id] = (season, week)
        return game_id

    def touchdown(
        self,
        game_id: str,
        player_id: int | None = None,
        *,
        team: str | None = None,
        name: str | None = "J.Example",
        td_type: str = "rush",
        source: str = "nflverse",
        quarter: int = 1,
        clock: str = "10:00",
        yards: int = 5,
        score_after: tuple[int | None, int | None] = (6, 0),
    ) -> None:
        """Add the next touchdown of a game. `team` defaults to the away team."""
        season, week = self._games[game_id]
        team = team or game_id.split("-")[2]
        seq = self._next_seq.get(game_id, 1)
        self._next_seq[game_id] = seq + 1
        self._db.execute(
            "INSERT INTO touchdowns (game_id, season, week, team_id, team_abbr, player_id,"
            " player_name, td_type, quarter, clock, game_seq, yards, description,"
            " away_score_after, home_score_after, source, source_id)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                game_id,
                season,
                week,
                self._franchise[team],
                team,
                player_id,
                name,
                td_type,
                quarter,
                clock,
                seq,
                yards,
                DESCRIPTION,
                score_after[0],
                score_after[1],
                source,
                f"{seq * 100}.0",
            ),
        )

    def save(self) -> Path:
        self._db.commit()
        self._db.close()
        return self.path


@pytest.fixture
def site(tmp_path: Path) -> SiteDb:
    """An empty site database in a directory of its own, as on a real machine."""
    directory = tmp_path / "site"
    directory.mkdir()
    return SiteDb(directory / "site.db")


def rows(path: Path, sql: str, *params: object) -> list[tuple]:
    """Run a query against a database file and return every row."""
    db = sqlite3.connect(path)
    try:
        return db.execute(sql, params).fetchall()
    finally:
        db.close()


# --- Tool tests: the committed fixture, opened the way the server opens it -------------------


@pytest.fixture
def db() -> Iterator[Database]:
    with open_database(FIXTURE_DB, timeout_ms=2000) as database:
        yield database


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def edge_db(tmp_path: Path) -> Iterator[Database]:
    """A tiny exported database holding what the two-season fixture lacks: a game whose
    touchdown records are missing, and two players whose names differ only in punctuation."""
    from alltouchdowns_mcp.export.build import build

    site = SiteDb(tmp_path / "edge-site.db")
    receiver = site.player("DJ Example", position="WR")
    cornerback = site.player("D.J. Example", slug="d-j-example", position="CB")
    complete = site.game(2025, 1, "KC", "DEN", score=(14, 7))
    site.touchdown(complete, receiver, team="KC", td_type="pass")
    site.touchdown(complete, receiver, team="KC", td_type="pass")
    site.touchdown(complete, cornerback, team="DEN", td_type="int_return")
    site.game(2025, 2, "KC", "NE", score=(42, 10))  # no touchdown rows: the source lost them
    out = tmp_path / "edge.db"
    build(site.save(), out, today=TODAY)
    with open_database(out, timeout_ms=2000) as database:
        yield database
