"""get_game and list_games."""

from __future__ import annotations

from alltouchdowns_mcp import cursor as cursors
from alltouchdowns_mcp.db import Database
from alltouchdowns_mcp.errors import ToolRefusal
from alltouchdowns_mcp.models import GetGameResult, ListGamesResult, SeasonType
from alltouchdowns_mcp.tools import shared

_GAME_BY_ID = f"SELECT {shared.GAME_COLUMNS} FROM games g WHERE g.game_id = ?"

# A franchise plays at most once in a week, so this finds at most one game.
_GAME_BY_PARTS = f"""
SELECT {shared.GAME_COLUMNS} FROM games g
WHERE g.season = ? AND g.week = ? AND ? IN (g.away_franchise, g.home_franchise)
"""

_GAME_TOUCHDOWNS = f"""
SELECT {shared.TOUCHDOWN_COLUMNS}
FROM touchdowns t JOIN games g ON g.game_id = t.game_id
WHERE t.game_id = ? ORDER BY t.game_seq
"""


def get_game(
    db: Database,
    *,
    game_id: str | None = None,
    season: int | None = None,
    week: int | None = None,
    team: str | None = None,
) -> GetGameResult:
    by_parts = (season, week, team)
    if game_id is not None and all(part is None for part in by_parts):
        row = db.conn.execute(_GAME_BY_ID, (game_id.strip().upper(),)).fetchone()
    elif game_id is None and all(part is not None for part in by_parts):
        assert season is not None and team is not None
        db.coverage.require(season)
        row = db.conn.execute(_GAME_BY_PARTS, (season, week, db.teams.resolve(team))).fetchone()
    else:
        raise ToolRefusal(
            "invalid_argument", "Pass game_id on its own, or season, week and team together."
        )
    if row is None:
        raise ToolRefusal(
            "not_found", "The team may not have played that week, or the game is not final yet."
        )

    game = shared.game(row, db.teams)
    notes = []
    if game.touchdowns_incomplete:
        notes.append(
            "The source has no touchdown records for this game, although the score shows "
            "touchdowns were scored. Do not report it as a game without touchdowns."
        )
    elif game.touchdown_count == 0:
        notes.append("No touchdowns were scored in this game.")

    return GetGameResult(
        game=game,
        touchdowns=[
            shared.touchdown(td, db.teams)
            for td in db.conn.execute(_GAME_TOUCHDOWNS, (game.game_id,))
        ],
        notes=notes,
        source=db.coverage.source(),
    )


def list_games(
    db: Database,
    *,
    season: int,
    week: int | None = None,
    team: str | None = None,
    season_type: SeasonType = "all",
    limit: int = shared.MAX_PAGE,
    cursor: str | None = None,
) -> ListGamesResult:
    limit = shared.page_size(limit)
    db.coverage.require(season)

    clauses = ["g.season = :season"]
    params: dict[str, object] = {"season": season, "season_type": season_type}
    if week is not None:
        clauses.append("g.week = :week")
        params["week"] = week
    if team is not None:
        clauses.append(":franchise IN (g.away_franchise, g.home_franchise)")
        params["franchise"] = db.teams.resolve(team)
    where = " AND ".join(clauses) + shared.SEASON_TYPE_SQL[season_type].format(t="g")

    (total,) = db.conn.execute(f"SELECT COUNT(*) FROM games g WHERE {where}", params).fetchone()

    # Games in the order they were played. A cursor is only good for exactly this search on
    # exactly this version of the data.
    search = {**params, "data": db.coverage.exported_at}
    page_where = where
    if cursor is not None:
        played_at, game_id = cursors.decode(cursor, search, (str, str))
        page_where += " AND (g.played_at, g.game_id) > (:after_time, :after_id)"
        params = {**params, "after_time": played_at, "after_id": game_id}

    rows = db.conn.execute(
        f"SELECT {shared.GAME_COLUMNS}, g.played_at FROM games g WHERE {page_where}"
        " ORDER BY g.played_at, g.game_id LIMIT :limit",
        {**params, "limit": limit + 1},
    ).fetchall()
    games = [shared.game(row[:-1], db.teams) for row in rows[:limit]]

    next_cursor = None
    if len(rows) > limit:
        next_cursor = cursors.encode((rows[limit - 1][-1], games[-1].game_id), search)
    return ListGamesResult(
        total_matches=total, games=games, next_cursor=next_cursor, source=db.coverage.source()
    )
