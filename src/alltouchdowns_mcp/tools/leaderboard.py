"""get_leaderboard."""

from __future__ import annotations

from alltouchdowns_mcp.db import Database
from alltouchdowns_mcp.errors import ToolRefusal
from alltouchdowns_mcp.models import (
    AppliedFilters,
    LeaderboardResult,
    LeaderboardRow,
    Scorer,
    SeasonType,
    TdType,
    TieOmitted,
)
from alltouchdowns_mcp.tools import shared

MAX_LIMIT = 50
# Ties at the cut-off are kept, which can return far more rows than asked for: a top 10 of
# interception returns is 60 players in one season. Beyond this many rows, the last tied
# group is left out and reported as a count instead.
MAX_ROWS = 100

# Which count the ranking uses, as (column, description), per season_type.
_METRICS: dict[SeasonType, tuple[str, str]] = {
    "regular": ("regular_season", "regular-season touchdowns"),
    "postseason": ("postseason", "postseason touchdowns"),
    "all": ("total", "regular-season and postseason touchdowns combined"),
}

# RANK() gives tied players the same rank, so `rank <= :limit` keeps the whole tied group at
# the cut-off. The {placeholders} are filled with fixed fragments from this file, never with
# caller text. With `per_season` the rows being ranked are player-seasons, not players.
_LEADERBOARD = """
WITH totals AS (
  SELECT player_slug, {season_column} AS season,
         SUM(postseason = 0) AS regular_season,
         SUM(postseason = 1) AS postseason,
         COUNT(*) AS total
  FROM touchdowns t
  WHERE player_slug IS NOT NULL AND {where}
  GROUP BY player_slug, {season_column}
),
ranked AS (
  SELECT *, RANK() OVER (ORDER BY {metric} DESC) AS rank
  FROM totals WHERE {metric} > 0
)
SELECT r.rank, r.season, r.player_slug, p.display_name, p.position,
       r.regular_season, r.postseason
FROM ranked r JOIN players p ON p.slug = r.player_slug
WHERE r.rank <= :limit
ORDER BY r.rank, p.display_name, r.player_slug, r.season
"""


def get_leaderboard(
    db: Database,
    *,
    season: int | None = None,
    td_type: TdType | None = None,
    team: str | None = None,
    season_type: SeasonType = "regular",
    per_season: bool = False,
    limit: int = 10,
) -> LeaderboardResult:
    if not 1 <= limit <= MAX_LIMIT:
        raise ToolRefusal("invalid_argument", f"limit must be between 1 and {MAX_LIMIT}.")

    clauses = ["1 = 1"]
    params: dict[str, object] = {"limit": limit}
    applied = AppliedFilters(season=season, td_type=td_type, season_type=season_type)
    franchise = None
    if season is not None:
        db.coverage.require(season)
        clauses.append("t.season = :season")
        params["season"] = season
    if td_type is not None:
        clauses.append("t.td_type = :td_type")
        params["td_type"] = td_type
    if team is not None:
        franchise = db.teams.resolve(team)
        applied.team = db.teams.describe(franchise)
        clauses.append("t.franchise = :franchise")
        params["franchise"] = franchise

    metric, description = _METRICS[season_type]
    sql = _LEADERBOARD.format(
        where=" AND ".join(clauses),
        metric=metric,
        season_column="t.season" if per_season else "NULL",
    )
    rows = db.conn.execute(sql, params).fetchall()

    tie_omitted = None
    if len(rows) > MAX_ROWS:
        last_rank = rows[-1][0]
        tied = [row for row in rows if row[0] == last_rank]
        regular, postseason = tied[0][5], tied[0][6]
        count = {"regular_season": regular, "postseason": postseason}.get(
            metric, regular + postseason
        )
        tie_omitted = TieOmitted(rank=last_rank, players=len(tied), touchdowns=count)
        rows = rows[: len(rows) - len(tied)]

    # Say exactly what was added up. "Postseason touchdowns" alone could be read as one
    # postseason when it is in fact every postseason since coverage began.
    if season is not None:
        span = f"in the {season} season"
    elif per_season:
        span = "in a single season"
    else:
        span = f"added up over seasons {db.coverage.describe()}"

    return LeaderboardResult(
        ranked_by=f"{description} {span}",
        rows=[
            LeaderboardRow(
                rank=rank,
                season=row_season,
                player=Scorer(slug=slug, name=name),
                position=position,
                touchdowns=shared.td_count(regular, postseason),
            )
            for rank, row_season, slug, name, position, regular, postseason in rows
        ],
        tie_omitted=tie_omitted,
        filters=applied,
        notes=shared.incomplete_notes(
            db, shared.game_scope(season=season, franchise=franchise, season_type=season_type)
        ),
        source=db.coverage.source(),
    )
