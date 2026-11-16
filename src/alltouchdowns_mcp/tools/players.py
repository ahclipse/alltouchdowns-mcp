"""find_players and get_player."""

from __future__ import annotations

from alltouchdowns_mcp.db import Database
from alltouchdowns_mcp.errors import ToolRefusal
from alltouchdowns_mcp.models import (
    FindPlayersResult,
    GetPlayerResult,
    MatchKind,
    SeasonCount,
    TeamCount,
    TypeCount,
)
from alltouchdowns_mcp.names import match_kind, name_tokens, split_suffix
from alltouchdowns_mcp.tools import shared

_MATCH_ORDER: tuple[MatchKind, ...] = ("exact", "all_words", "word_prefix")

# Matching runs in Python over every player, so the query is kept short: the work grows with
# the number of words, and no name needs more than a handful.
MAX_QUERY_CHARS = 100
MAX_QUERY_WORDS = 6

_ALL_PLAYERS = f"SELECT search_name, name_suffix, {shared.PLAYER_COLUMNS} FROM players"

# Players whose names match once punctuation and suffixes are ignored: "DJ Moore" and
# "D.J. Moore", or a father and son. These are exactly the pairs find_players cannot separate.
_NAMESAKES = f"""
SELECT {shared.PLAYER_COLUMNS} FROM players
WHERE search_name = (SELECT search_name FROM players WHERE slug = :slug) AND slug <> :slug
ORDER BY td_regular + td_postseason DESC, slug
"""

_BY_TYPE = """
SELECT td_type, SUM(postseason = 0), SUM(postseason = 1)
FROM touchdowns WHERE player_slug = ?
GROUP BY td_type ORDER BY COUNT(*) DESC, td_type
"""

_BY_SEASON = """
SELECT season, SUM(postseason = 0), SUM(postseason = 1)
FROM touchdowns WHERE player_slug = ?
GROUP BY season ORDER BY season
"""

# One row per (season, team), in the order the player first scored for each team.
_SEASON_TEAMS = """
SELECT season, team_abbr
FROM touchdowns WHERE player_slug = ?
GROUP BY season, team_abbr ORDER BY season, MIN(played_at)
"""

_BY_TEAM = """
SELECT team_abbr, MIN(season), MAX(season), SUM(postseason = 0), SUM(postseason = 1)
FROM touchdowns WHERE player_slug = ?
GROUP BY team_abbr ORDER BY MIN(played_at)
"""

_PLAYER_TOUCHDOWNS = f"""
SELECT {shared.TOUCHDOWN_COLUMNS}
FROM touchdowns t JOIN games g ON g.game_id = t.game_id
WHERE t.player_slug = ?
"""
_FIRST_TOUCHDOWN = _PLAYER_TOUCHDOWNS + " ORDER BY t.played_at, t.game_seq LIMIT 1"
_LAST_TOUCHDOWN = _PLAYER_TOUCHDOWNS + " ORDER BY t.played_at DESC, t.game_seq DESC LIMIT 1"


def find_players(db: Database, query: str, limit: int = 10) -> FindPlayersResult:
    limit = shared.page_size(limit)
    if len(query) > MAX_QUERY_CHARS:
        raise ToolRefusal("invalid_argument", "The name is too long. Give just the player's name.")
    words, suffix = split_suffix(name_tokens(query))
    if sum(len(word) for word in words) < 2:
        raise ToolRefusal("invalid_argument", "Give at least two letters of the player's name.")
    if len(words) > MAX_QUERY_WORDS:
        raise ToolRefusal("invalid_argument", "Too many words. Give just the player's name.")

    candidates = db.conn.execute(_ALL_PLAYERS).fetchall()
    ranked: list[tuple[tuple, MatchKind]] = _rank(candidates, words, suffix)
    notes = []

    if not ranked and len(words) > 1:
        # "Jon Smith" finds nothing but there are Smiths: offer them rather than nothing.
        ranked = [(row, "last_name_only") for row, _ in _rank(candidates, words[-1:], suffix)]
        if ranked:
            notes.append("No full-name match. These players share the last name.")
    if not ranked:
        notes.append(
            "No touchdown scorer matches. Only players with at least one touchdown in the "
            "covered seasons are listed."
        )

    return FindPlayersResult(
        matches=[shared.player(row[2:], db.teams, kind) for row, kind in ranked[:limit]],
        total_matches=len(ranked),
        notes=notes,
        source=db.coverage.source(),
    )


def _rank(
    candidates: list[tuple], words: list[str], suffix: str | None
) -> list[tuple[tuple, MatchKind]]:
    """Players matching `words`, best first.

    Order: kind of match, then a matching suffix ("jr") if the query gave one, then career
    touchdowns, so that of two players with the same name the prominent one comes first.
    """
    scored: list[tuple[tuple, tuple, MatchKind]] = []
    for row in candidates:
        search_name, name_suffix, slug, *_, last_season, regular, postseason, _teams = row
        kind = match_kind(words, search_name.split())
        if kind is None:
            continue
        wrong_suffix = suffix is not None and suffix != name_suffix
        order = (
            _MATCH_ORDER.index(kind),
            wrong_suffix,
            -(regular + postseason),
            -last_season,
            slug,
        )
        scored.append((order, row, kind))
    return [(row, kind) for _, row, kind in sorted(scored, key=lambda item: item[0])]


def get_player(db: Database, slug: str) -> GetPlayerResult:
    player = shared.require_player(db, slug)
    conn, teams = db.conn, db.teams
    key = (player.slug,)

    season_teams: dict[int, list[str]] = {}
    for season, abbr in conn.execute(_SEASON_TEAMS, key):
        season_teams.setdefault(season, []).append(abbr)

    notes = [
        "Counts are touchdowns this player scored. Passes thrown are not credited to the passer."
    ]
    if player.first_season == db.coverage.seasons[0]:
        notes.append(
            f"Coverage starts in {db.coverage.seasons[0]}. Earlier touchdowns are not included."
        )
    notes += shared.incomplete_notes(db, shared.game_scope(player=player.slug))

    return GetPlayerResult(
        player=player,
        by_type=[
            TypeCount(td_type=td_type, touchdowns=shared.td_count(regular, postseason))
            for td_type, regular, postseason in conn.execute(_BY_TYPE, key)
        ],
        by_season=[
            SeasonCount(
                season=season,
                teams=[teams.team(abbr) for abbr in season_teams[season]],
                touchdowns=shared.td_count(regular, postseason),
            )
            for season, regular, postseason in conn.execute(_BY_SEASON, key)
        ],
        by_team=[
            TeamCount(
                team=teams.team(abbr),
                first_season=first,
                last_season=last,
                touchdowns=shared.td_count(regular, postseason),
            )
            for abbr, first, last, regular, postseason in conn.execute(_BY_TEAM, key)
        ],
        first_touchdown=shared.touchdown(conn.execute(_FIRST_TOUCHDOWN, key).fetchone(), teams),
        last_touchdown=shared.touchdown(conn.execute(_LAST_TOUCHDOWN, key).fetchone(), teams),
        same_name_players=[
            shared.player(row, teams) for row in conn.execute(_NAMESAKES, {"slug": player.slug})
        ],
        notes=notes,
        source=db.coverage.source(),
    )
