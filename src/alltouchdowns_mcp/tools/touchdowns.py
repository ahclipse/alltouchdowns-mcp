"""search_touchdowns."""

from __future__ import annotations

from typing import Literal

from alltouchdowns_mcp import cursor as cursors
from alltouchdowns_mcp.db import Database
from alltouchdowns_mcp.errors import ToolRefusal
from alltouchdowns_mcp.models import AppliedFilters, SearchTouchdownsResult, SeasonType, TdType
from alltouchdowns_mcp.tools import shared

Order = Literal["oldest", "newest", "longest"]

# For each order: ORDER BY, the "rows after this key" condition, and the key's types.
# `played_at` is the game's date and kick-off, so this is true chronological order. Games that
# kick off at the same moment are ordered by game id, and touchdowns within a game by sequence.
_ORDERS: dict[Order, tuple[str, str, tuple[type, ...]]] = {
    "oldest": (
        "t.played_at, t.game_id, t.game_seq",
        "(t.played_at, t.game_id, t.game_seq) > (:k0, :k1, :k2)",
        (str, str, int),
    ),
    "newest": (
        "t.played_at DESC, t.game_id DESC, t.game_seq DESC",
        "(t.played_at, t.game_id, t.game_seq) < (:k0, :k1, :k2)",
        (str, str, int),
    ),
    "longest": (
        "t.yards DESC, t.played_at, t.game_id, t.game_seq",
        "(t.yards < :k0 OR"
        " (t.yards = :k0 AND (t.played_at, t.game_id, t.game_seq) > (:k1, :k2, :k3)))",
        (int, str, str, int),
    ),
}

MAX_YARDS = 150  # a touchdown cannot be longer than the field; this bounds the argument


def search_touchdowns(
    db: Database,
    *,
    player: str | None = None,
    team: str | None = None,
    season: int | None = None,
    week: int | None = None,
    td_type: TdType | None = None,
    quarter: int | None = None,
    min_yards: int | None = None,
    season_type: SeasonType = "all",
    order: Order = "oldest",
    limit: int = 25,
    cursor: str | None = None,
) -> SearchTouchdownsResult:
    limit = shared.page_size(limit)
    if quarter is not None and not 1 <= quarter <= 5:
        raise ToolRefusal("invalid_argument", "quarter must be 1 to 4, or 5 for overtime.")
    if min_yards is not None and not 0 <= min_yards <= MAX_YARDS:
        raise ToolRefusal("invalid_argument", f"min_yards must be between 0 and {MAX_YARDS}.")

    # Resolve what the caller named into what the data holds, and say back what was understood.
    applied = AppliedFilters(
        season=season, week=week, td_type=td_type, quarter=quarter, min_yards=min_yards,
        season_type=season_type,
    )  # fmt: skip
    if player is not None:
        applied.player = shared.require_player(db, player)
    franchise = None
    if team is not None:
        franchise = db.teams.resolve(team)
        applied.team = db.teams.describe(franchise)
    if season is not None:
        db.coverage.require(season)

    # Every clause is a fixed string. The caller's values only ever travel as bound parameters.
    conditions: list[tuple[str, str, object]] = [
        ("t.player_slug = :player", "player", applied.player.slug if applied.player else None),
        ("t.franchise = :franchise", "franchise", franchise),
        ("t.season = :season", "season", season),
        ("t.week = :week", "week", week),
        ("t.td_type = :td_type", "td_type", td_type),
        ("t.yards >= :min_yards", "min_yards", min_yards),
        # Quarter 5 means overtime, which the data records as quarter 5 or higher.
        ("t.quarter >= 5" if quarter == 5 else "t.quarter = :quarter", "quarter", quarter),
    ]
    clauses = [clause for clause, _, value in conditions if value is not None]
    params: dict[str, object] = {name: value for _, name, value in conditions if value is not None}
    if order == "longest":
        clauses.append("t.yards IS NOT NULL")
    where = " AND ".join(clauses or ["1 = 1"]) + shared.SEASON_TYPE_SQL[season_type].format(t="t")

    regular, postseason = db.conn.execute(
        f"SELECT SUM(t.postseason = 0), SUM(t.postseason = 1) FROM touchdowns t WHERE {where}",
        params,
    ).fetchone()

    # Paging through everything needs at least a player, team or season. Without one, the
    # first page still answers "the longest ever" style questions, but the whole dataset
    # cannot be walked page by page.
    narrowed = any(value is not None for value in (player, team, season))
    order_by, after_key, key_shape = _ORDERS[order]
    # Everything that defines the result set. A cursor is only good for exactly this.
    search = {**params, "season_type": season_type, "order": order,
              "data": db.coverage.exported_at}  # fmt: skip
    page_where, page_params = where, dict(params)
    if cursor is not None:
        if not narrowed:
            raise ToolRefusal("bad_cursor")
        key = cursors.decode(cursor, search, key_shape)
        page_where += f" AND {after_key}"
        page_params |= {f"k{i}": part for i, part in enumerate(key)}

    rows = db.conn.execute(
        f"""SELECT {shared.TOUCHDOWN_COLUMNS}, t.played_at
            FROM touchdowns t JOIN games g ON g.game_id = t.game_id
            WHERE {page_where} ORDER BY {order_by} LIMIT :limit""",
        {**page_params, "limit": limit + 1},  # one extra row shows whether another page exists
    ).fetchall()
    page = [shared.touchdown(row[:-1], db.teams) for row in rows[:limit]]

    notes = shared.incomplete_notes(
        db,
        shared.game_scope(
            season=season,
            week=week,
            franchise=franchise,
            season_type=season_type,
            player=applied.player.slug if applied.player else None,
        ),
    )
    next_cursor = None
    if len(rows) > limit:
        if narrowed:
            last, played_at = page[-1], rows[limit - 1][-1]
            key = (played_at, last.game_id, last.game_seq)
            if order == "longest":
                key = (last.yards or 0, *key)
            next_cursor = cursors.encode(key, search)
        else:
            notes.append("More results exist. Add a player, team or season to page through them.")

    return SearchTouchdownsResult(
        matched=shared.td_count(regular, postseason),
        touchdowns=page,
        next_cursor=next_cursor,
        filters=applied,
        notes=notes,
        source=db.coverage.source(),
    )
