"""Export step 2: build the slim public database from the verified copy of the site database.

Everything that decides what may be published lives in the SQL below, so the rules can be read
in one place:

* only finished games, and only nflverse rows
* a recent game is held back until its nflverse touchdowns have arrived
* play descriptions, provider ids and surrogate ids are never selected
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from importlib import resources
from pathlib import Path
from urllib.parse import quote

from alltouchdowns_mcp.names import search_name

SCHEMA_VERSION = 1

# nflverse play-by-play can lag a finished game by a day or more. Until it arrives the site
# shows touchdowns from another provider, which this project must never publish. So a game
# from the last HOLD_BACK_DAYS days is exported only once it has nflverse touchdown rows and
# no rows from anyone else. After that, nflverse is taken as the final word.
HOLD_BACK_DAYS = 7

# A game with no touchdown rows is normal when both teams only kicked field goals. In 27
# seasons the highest score a team reached that way is 15. A team at 19 or more with no rows
# means the source is missing them, so the game is flagged instead of reported as zero.
INCOMPLETE_SCORE = 19

_SCHEMA = (resources.files("alltouchdowns_mcp") / "export" / "schema.sql").read_text("utf-8")

# The site's rule for the Championship week: the postseason week with exactly one game. It is
# computed over every scheduled game, before any filtering, because postseason games arrive
# round by round and a held-back game must not make another week look like the final.
_FINAL_WEEKS = """
CREATE TEMP TABLE final_weeks AS
SELECT season, MAX(week) AS week
FROM (
  SELECT season, week FROM src.games
  WHERE postseason = 1
  GROUP BY season, week
  HAVING COUNT(*) = 1
)
GROUP BY season
"""

_TOUCHDOWN_COUNTS = """
CREATE TEMP TABLE td_counts AS
SELECT game_id,
       SUM(source = 'nflverse')  AS nflverse,
       SUM(source <> 'nflverse') AS other
FROM src.touchdowns
GROUP BY game_id
"""

# A city shared by two franchises (Los Angeles, New York) is shown with its abbreviation.
_TEAMS = """
INSERT INTO teams (abbr, franchise, city, label, first_season, last_season)
SELECT a.abbr, a.team_id, a.city,
       CASE WHEN (SELECT COUNT(*) FROM src.teams t WHERE t.city = a.city) > 1
            THEN a.city || ' (' || a.abbr || ')'
            ELSE a.city END,
       a.first_season, a.last_season
FROM src.team_aliases a
"""

_PUBLISHABLE = """
    g.status = 'final'
AND g.source = 'nflverse'
AND g.season IN (SELECT season FROM wanted_seasons)
"""

_NOT_HELD_BACK = """
(g.date < :cutoff OR (COALESCE(c.nflverse, 0) > 0 AND COALESCE(c.other, 0) = 0))
"""

_GAMES = f"""
INSERT INTO games (
  game_id, season, week, postseason, week_label, stage, date, played_at,
  away_abbr, home_abbr, away_franchise, home_franchise, away_score, home_score,
  touchdown_count, touchdowns_incomplete)
SELECT g.id, g.season, g.week, g.postseason,
       CASE WHEN g.postseason = 1 AND g.week = fw.week THEN 'Championship'
            ELSE 'Week ' || g.week END,
       CASE WHEN g.postseason = 0 THEN 'Regular season'
            WHEN g.week = fw.week THEN 'Championship'
            ELSE 'Playoffs' END,
       g.date,
       -- Kick-off times are missing for 1999; those games sort at the end of their day.
       g.date || 'T' || COALESCE(g.kickoff_time, '23:59'),
       g.away_abbr, g.home_abbr, g.away_team_id, g.home_team_id,
       g.away_score, g.home_score,
       COALESCE(c.nflverse, 0),
       COALESCE(c.nflverse, 0) = 0
         AND (COALESCE(c.other, 0) > 0 OR MAX(g.away_score, g.home_score) >= :incomplete_score)
FROM src.games g
LEFT JOIN final_weeks fw ON fw.season = g.season
LEFT JOIN td_counts c ON c.game_id = g.id
WHERE {_PUBLISHABLE} AND {_NOT_HELD_BACK}
"""

# Every finished game that was not published, whatever the reason: its nflverse touchdowns have
# not arrived, or the game row itself still comes from another provider.
_HELD_BACK_COUNT = """
SELECT COUNT(*)
FROM src.games g
WHERE g.status = 'final'
  AND g.season IN (SELECT season FROM wanted_seasons)
  AND g.id NOT IN (SELECT game_id FROM games)
"""

# `pass` and `rush` become `receiving` and `rushing`: a touchdown is credited to the player who
# scored it, so on a pass that is the receiver. The passer is never credited.
_TOUCHDOWNS = """
INSERT INTO touchdowns (
  game_id, game_seq, season, week, postseason, played_at, team_abbr, franchise,
  player_slug, scorer_name, td_type, quarter, clock, yards,
  away_score_after, home_score_after)
SELECT t.game_id, t.game_seq, t.season, t.week, g.postseason, g.played_at,
       t.team_abbr, t.team_id,
       p.slug,
       COALESCE(p.display_name, t.player_name),
       CASE t.td_type WHEN 'pass' THEN 'receiving' WHEN 'rush' THEN 'rushing'
            ELSE t.td_type END,
       t.quarter,
       CASE WHEN length(t.clock) = 4 THEN '0' || t.clock ELSE t.clock END,
       t.yards, t.away_score_after, t.home_score_after
FROM src.touchdowns t
JOIN games g ON g.game_id = t.game_id
LEFT JOIN src.players p ON p.id = t.player_id
WHERE t.source = 'nflverse'
"""

_PLAYER_TOTALS = """
SELECT p.slug, p.display_name, p.position,
       MIN(t.season), MAX(t.season),
       SUM(t.postseason = 0), SUM(t.postseason = 1)
FROM touchdowns t
JOIN src.players p ON p.slug = t.player_slug
GROUP BY p.slug
"""

_PLAYER_TEAMS = """
SELECT player_slug, team_abbr
FROM touchdowns
WHERE player_slug IS NOT NULL
GROUP BY player_slug, team_abbr
ORDER BY player_slug, MIN(played_at), team_abbr
"""

# A season is complete once its Championship game is in this database. Asking the source
# instead would call a season complete while that game is still held back.
_SEASONS = """
INSERT INTO seasons (season, final_week, complete)
SELECT s.season, fw.week,
       EXISTS (SELECT 1 FROM games x WHERE x.season = s.season AND x.stage = 'Championship')
FROM (SELECT DISTINCT season FROM games) s
LEFT JOIN final_weeks fw ON fw.season = s.season
"""


@dataclass(frozen=True)
class BuildResult:
    games: int
    touchdowns: int
    players: int
    seasons: tuple[int, ...]
    held_back: int
    incomplete: int


def build(
    source: Path,
    out: Path,
    *,
    today: date,
    seasons: Sequence[int] | None = None,
    exported_at: datetime | None = None,
) -> BuildResult:
    """Write a new slim database at `out` from the site database copy at `source`.

    `today` is the current date in US Eastern time, the zone game dates are recorded in.
    `seasons` limits the export to those seasons (used to build the test fixture).
    """
    out.unlink(missing_ok=True)
    # uri=True on the main connection is what allows the ATTACH below to use a URI.
    db = sqlite3.connect(f"file:{quote(str(out))}?mode=rwc", uri=True)
    try:
        db.execute("ATTACH DATABASE ? AS src", (f"file:{quote(str(source))}?mode=ro&immutable=1",))
        db.executescript(_SCHEMA)
        with db:  # one transaction: the file is either complete or empty
            _select_seasons(db, seasons)
            db.execute(_FINAL_WEEKS)
            db.execute(_TOUCHDOWN_COUNTS)
            db.execute(_TEAMS)
            params = {
                "cutoff": (today - timedelta(days=HOLD_BACK_DAYS)).isoformat(),
                "incomplete_score": INCOMPLETE_SCORE,
            }
            db.execute(_GAMES, params)
            held_back = db.execute(_HELD_BACK_COUNT).fetchone()[0]
            db.execute(_TOUCHDOWNS)
            _insert_players(db)
            db.execute(_SEASONS)
            result = _summarise(db, held_back)
            _write_meta(db, result, exported_at or datetime.now(UTC), seasons is not None)
        db.execute("DETACH DATABASE src")
        db.execute("ANALYZE")  # statistics for the query planner
        db.commit()
        db.execute("VACUUM")  # rewrite the file compactly, with no leftover bytes in its pages
        return result
    finally:
        db.close()


def _select_seasons(db: sqlite3.Connection, seasons: Sequence[int] | None) -> None:
    db.execute("CREATE TEMP TABLE wanted_seasons (season INTEGER PRIMARY KEY)")
    if seasons is None:
        db.execute("INSERT INTO wanted_seasons SELECT DISTINCT season FROM src.games")
    else:
        db.executemany("INSERT INTO wanted_seasons VALUES (?)", [(s,) for s in set(seasons)])


def _insert_players(db: sqlite3.Connection) -> None:
    teams: dict[str, list[str]] = {}
    for slug, abbr in db.execute(_PLAYER_TEAMS):
        teams.setdefault(slug, []).append(abbr)

    rows = []
    for slug, name, position, first, last, regular, postseason in db.execute(_PLAYER_TOTALS):
        searchable, suffix = search_name(name)
        rows.append(
            (
                slug,
                name,
                searchable,
                suffix,
                position,
                first,
                last,
                regular,
                postseason,
                ",".join(teams[slug]),
            )
        )
    db.executemany("INSERT INTO players VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)


def _summarise(db: sqlite3.Connection, held_back: int) -> BuildResult:
    def count(sql: str) -> int:
        return db.execute(sql).fetchone()[0]

    return BuildResult(
        games=count("SELECT COUNT(*) FROM games"),
        touchdowns=count("SELECT COUNT(*) FROM touchdowns"),
        players=count("SELECT COUNT(*) FROM players"),
        seasons=tuple(s for (s,) in db.execute("SELECT season FROM seasons ORDER BY season")),
        held_back=held_back,
        incomplete=count("SELECT COUNT(*) FROM games WHERE touchdowns_incomplete"),
    )


def _write_meta(
    db: sqlite3.Connection, result: BuildResult, exported_at: datetime, partial: bool
) -> None:
    data_through = db.execute("SELECT MAX(date) FROM games").fetchone()[0]
    meta = {
        "schema_version": SCHEMA_VERSION,
        "exported_at": exported_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_through": data_through or "",
        # "partial" marks a database built for selected seasons only, such as the test fixture.
        "coverage": "partial" if partial else "full",
        "games": result.games,
        "touchdowns": result.touchdowns,
        "players": result.players,
        "held_back": result.held_back,
    }
    db.executemany("INSERT INTO meta VALUES (?, ?)", [(k, str(v)) for k, v in meta.items()])
