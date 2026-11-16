"""Row-to-model helpers and SQL fragments the tools share."""

from __future__ import annotations

from alltouchdowns_mcp.db import Database
from alltouchdowns_mcp.errors import ToolRefusal
from alltouchdowns_mcp.models import (
    Game,
    MatchKind,
    PlayerMatch,
    Score,
    Scorer,
    SeasonType,
    TdCount,
    TeamScore,
    Touchdown,
)
from alltouchdowns_mcp.teams import Teams

MAX_PAGE = 50

# Expects `touchdowns t` joined to `games g`.
TOUCHDOWN_COLUMNS = """
  t.game_id, t.game_seq, t.season, g.week_label, g.stage, g.date, t.team_abbr,
  CASE WHEN t.franchise = g.home_franchise THEN g.away_abbr ELSE g.home_abbr END,
  t.player_slug, t.scorer_name, t.td_type, t.quarter, t.clock, t.yards,
  t.away_score_after, t.home_score_after
"""

# Expects `games g`.
GAME_COLUMNS = """
  g.game_id, g.season, g.week, g.week_label, g.stage, g.date,
  g.away_abbr, g.away_score, g.home_abbr, g.home_score,
  g.touchdown_count, g.touchdowns_incomplete
"""

PLAYER_COLUMNS = """
  slug, display_name, position, first_season, last_season,
  td_regular, td_postseason, team_abbrs
"""

# Appended to a WHERE clause. `{t}` is the table alias that has a `postseason` column.
SEASON_TYPE_SQL: dict[SeasonType, str] = {
    "all": "",
    "regular": " AND {t}.postseason = 0",
    "postseason": " AND {t}.postseason = 1",
}


def td_count(regular: int | None, postseason: int | None) -> TdCount:
    regular, postseason = regular or 0, postseason or 0
    return TdCount(regular_season=regular, postseason=postseason, total=regular + postseason)


def touchdown(row: tuple, teams: Teams) -> Touchdown:
    (game_id, game_seq, season, week_label, stage, date, team, opponent, slug, name,
     td_type, quarter, clock, yards, away_after, home_after) = row  # fmt: skip
    score_after = None
    if away_after is not None and home_after is not None:
        score_after = Score(away=away_after, home=home_after)
    return Touchdown(
        game_id=game_id,
        game_seq=game_seq,
        season=season,
        week_label=week_label,
        stage=stage,
        date=date,
        team=teams.team(team),
        opponent=teams.team(opponent),
        scorer=Scorer(slug=slug, name=name),
        td_type=td_type,
        quarter=quarter,
        overtime=quarter is not None and quarter >= 5,
        clock=clock,
        yards=yards,
        score_after=score_after,
    )


def game(row: tuple, teams: Teams) -> Game:
    (game_id, season, week, week_label, stage, date, away, away_score, home, home_score,
     touchdown_count, incomplete) = row  # fmt: skip
    return Game(
        game_id=game_id,
        season=season,
        week=week,
        week_label=week_label,
        stage=stage,
        date=date,
        away=TeamScore(abbr=away, name=teams.label(away), score=away_score),
        home=TeamScore(abbr=home, name=teams.label(home), score=home_score),
        touchdown_count=touchdown_count,
        touchdowns_incomplete=bool(incomplete),
    )


def player(row: tuple, teams: Teams, match: MatchKind | None = None) -> PlayerMatch:
    slug, name, position, first, last, regular, postseason, team_abbrs = row
    return PlayerMatch(
        slug=slug,
        name=name,
        position=position,
        first_season=first,
        last_season=last,
        teams=[teams.team(abbr) for abbr in team_abbrs.split(",")],
        touchdowns=td_count(regular, postseason),
        match=match,
    )


def require_player(db: Database, slug: str) -> PlayerMatch:
    row = db.conn.execute(
        f"SELECT {PLAYER_COLUMNS} FROM players WHERE slug = ?", (slug.strip().lower(),)
    ).fetchone()
    if row is None:
        raise ToolRefusal("unknown_player")
    return player(row, db.teams)


def page_size(limit: int) -> int:
    if not 1 <= limit <= MAX_PAGE:
        raise ToolRefusal("invalid_argument", f"limit must be between 1 and {MAX_PAGE}.")
    return limit


def game_scope(
    *,
    season: int | None = None,
    week: int | None = None,
    franchise: str | None = None,
    season_type: SeasonType = "all",
    player: str | None = None,
) -> tuple[str, dict[str, object]]:
    """A WHERE clause over `games g` for the games a set of filters covers."""
    clauses = ["1 = 1"]
    params: dict[str, object] = {}
    if season is not None:
        clauses.append("g.season = :season")
        params["season"] = season
    if week is not None:
        clauses.append("g.week = :week")
        params["week"] = week
    if franchise is not None:
        clauses.append(":franchise IN (g.away_franchise, g.home_franchise)")
        params["franchise"] = franchise
    if player is not None:
        # The games of the teams this player scored for, in the seasons he scored for them.
        clauses.append(
            """EXISTS (SELECT 1 FROM touchdowns x
                       WHERE x.player_slug = :player AND x.season = g.season
                         AND x.franchise IN (g.away_franchise, g.home_franchise))"""
        )
        params["player"] = player
    return " AND ".join(clauses) + SEASON_TYPE_SQL[season_type].format(t="g"), params


def incomplete_notes(db: Database, scope: tuple[str, dict[str, object]]) -> list[str]:
    """Warn when the games in scope include one whose touchdown records are missing upstream.

    Without this a count that silently leaves a whole game out would look complete.
    """
    where, params = scope
    games = db.conn.execute(
        f"SELECT g.game_id FROM games g WHERE g.touchdowns_incomplete = 1 AND {where}"
        " ORDER BY g.game_id",
        params,
    ).fetchall()
    if not games:
        return []
    listed = ", ".join(game_id for (game_id,) in games[:5])
    more = f" and {len(games) - 5} more" if len(games) > 5 else ""
    return [
        f"The source has no touchdown records for {len(games)} game(s) in this range "
        f"({listed}{more}). Counts here are lower than the true totals."
    ]
