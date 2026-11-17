"""The MCP server: six read-only tools over the slim database.

Each tool here is a thin wrapper. Its signature and docstring are what the client model sees,
so they are written for that reader. The work happens in `alltouchdowns_mcp.tools`.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from importlib.metadata import version
from typing import Annotated, Any

from mcp.server import MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field

from alltouchdowns_mcp import ATTRIBUTION
from alltouchdowns_mcp.config import Settings
from alltouchdowns_mcp.db import Coverage, open_database
from alltouchdowns_mcp.guard import ExitGate
from alltouchdowns_mcp.models import (
    FindPlayersResult,
    GetGameResult,
    GetPlayerResult,
    LeaderboardResult,
    ListGamesResult,
    SearchTouchdownsResult,
    SeasonType,
    TdType,
)
from alltouchdowns_mcp.tools import games, leaderboard, players, touchdowns
from alltouchdowns_mcp.tools.touchdowns import Order

_READ_ONLY = ToolAnnotations(
    read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
)

# Parameter descriptions, written once so every tool words them the same way.
_TEAM = "City or abbreviation, e.g. 'Kansas City' or 'KC'. Never a nickname."
_SEASON = "Year the season started, e.g. 2019."
_WEEK = "Week number. Weeks keep counting into the postseason."
_SLUG = "Player slug, exactly as find_players returned it."
_TD_TYPE = (
    "Kind of touchdown. 'receiving' is a caught pass, credited to the receiver. "
    "'fumble_return' is any recovered loose ball: a fumble run back, a ball recovered in the "
    "end zone, and a blocked kick run back or recovered. 'other' is the few that fit no "
    "type, such as a missed field goal run back."
)
_SEASON_TYPE = "Count regular season games, postseason games, or all."
_CURSOR = "next_cursor from the previous page of the same search."

# Bounds keep nonsense out of the queries. They are wider than the data on purpose: a season
# of 1985 is a valid argument that gets a helpful "not covered", not a schema error.
Season = Annotated[int, Field(description=_SEASON, ge=1900, le=2100)]
OptionalSeason = Annotated[int | None, Field(description=_SEASON, ge=1900, le=2100)]
OptionalWeek = Annotated[int | None, Field(description=_WEEK, ge=1, le=30)]
OptionalTeam = Annotated[str | None, Field(description=_TEAM, max_length=60)]
OptionalTdType = Annotated[TdType | None, Field(description=_TD_TYPE)]
SeasonTypeArg = Annotated[SeasonType, Field(description=_SEASON_TYPE)]
OptionalCursor = Annotated[str | None, Field(description=_CURSOR, max_length=400)]
PageSize = Annotated[int, Field(description="Rows per page.", ge=1, le=50)]


def describe_server(coverage: Coverage) -> str:
    """The instructions a client model reads before using the tools.

    They are written once, when the server starts. The data is replaced daily, so the season
    range here can fall behind when a new season begins; results always carry the current one.
    """
    return f"""AllTouchdowns holds every NFL touchdown, as structured facts.

Scope: touchdowns only, for seasons {coverage.describe()}. Every result says under `source` \
which seasons are loaded now, which is the list to trust if the two differ. It has no field \
goals, yardage totals, standings or other statistics. A season is named by the year it \
starts in, and its playoff games in the following January and February belong to it.

Credit: each touchdown is credited to the player who scored it. On a passing play that is \
the receiver. Passes thrown are not recorded, so a question about touchdown passes thrown by \
a quarterback cannot be answered from this data. Say so instead of using these counts.

Counts are split into regular season and postseason. State which one you are reporting. \
Read a result's `notes`: they say when a count is known to be incomplete.

Teams are identified by city and abbreviation, never by nickname. Pass a city or an \
abbreviation as a `team` argument, and use the city when you answer.

Weeks keep counting into the postseason. Describe postseason games only as "Playoffs", and \
a season's final game as the "Championship".

Players are identified by a slug. Call find_players to get it. Never guess a slug: the \
best-known player of a name is often not the one with the plain slug.

This server answers questions; it does not hand out the dataset. Page through results when \
a question needs it. A request for every touchdown of a season or more, or for a file of the \
data, is not offered: say so and suggest a narrower question.

{ATTRIBUTION}"""


def create_server(settings: Settings) -> MCPServer:
    with open_database(settings.db_path, settings.query_step_limit) as db:
        coverage = db.coverage

    gate = ExitGate()
    mcp = MCPServer(
        "alltouchdowns",
        title="AllTouchdowns",
        description=f"Every NFL touchdown since 1999, as structured facts. {ATTRIBUTION}",
        instructions=describe_server(coverage),
        website_url="https://alltouchdowns.com",
        version=version("alltouchdowns-mcp"),
        middleware=[gate],
        log_level="WARNING",
    )

    def tool(title: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a read-only tool. Its docstring, with indentation removed, is what the
        client model reads as the tool's description."""

        def register(function: Callable[..., Any]) -> Callable[..., Any]:
            description = inspect.cleandoc(function.__doc__ or "")
            # Tell the gate which arguments exist, so that it can refuse any other.
            gate.arguments[function.__name__] = frozenset(inspect.signature(function).parameters)
            return mcp.tool(title=title, description=description, annotations=_READ_ONLY)(function)

        return register

    def database():
        return open_database(settings.db_path, settings.query_step_limit)

    @tool("Find players")
    def find_players(
        query: Annotated[str, Field(description="All or part of a player's name.", max_length=100)],
        limit: PageSize = 10,
    ) -> FindPlayersResult:
        """Find players by name and get their slugs.

        Only players with at least one touchdown are listed. Several players can share a name:
        tell them apart by position, seasons and teams before choosing a slug.
        """
        with database() as db:
            return players.find_players(db, query, limit)

    @tool("Get player")
    def get_player(
        slug: Annotated[str, Field(description=_SLUG, max_length=100)],
    ) -> GetPlayerResult:
        """Touchdown summary for one player: totals by type, by season and by team.

        Counts are touchdowns the player scored himself. Passes thrown are never credited, so
        a quarterback's total here is not his passing total.
        """
        with database() as db:
            return players.get_player(db, slug)

    @tool("Search touchdowns")
    def search_touchdowns(
        player: Annotated[str | None, Field(description=_SLUG, max_length=100)] = None,
        team: Annotated[
            str | None, Field(description=f"The team that scored. {_TEAM}", max_length=60)
        ] = None,
        season: OptionalSeason = None,
        week: OptionalWeek = None,
        td_type: OptionalTdType = None,
        quarter: Annotated[
            int | None, Field(description="1 to 4, or 5 for overtime.", ge=1, le=5)
        ] = None,
        min_yards: Annotated[
            int | None,
            Field(description="Only touchdowns of at least this many yards.", ge=0, le=150),
        ] = None,
        season_type: SeasonTypeArg = "all",
        order: Annotated[
            Order,
            Field(
                description="By kick-off time, 'oldest' or 'newest' first, or 'longest' by yards."
            ),
        ] = "oldest",
        limit: PageSize = 25,
        cursor: OptionalCursor = None,
    ) -> SearchTouchdownsResult:
        """List individual touchdowns that match the filters.

        `matched` is the full count across all pages, split by regular season and postseason,
        so a "how many" question needs one call and no paging. With season_type 'regular' or
        'postseason' the other side is null: it was not counted, which does not mean there
        were none.
        """
        with database() as db:
            return touchdowns.search_touchdowns(
                db,
                player=player,
                team=team,
                season=season,
                week=week,
                td_type=td_type,
                quarter=quarter,
                min_yards=min_yards,
                season_type=season_type,
                order=order,
                limit=limit,
                cursor=cursor,
            )

    @tool("Get game")
    def get_game(
        game_id: Annotated[
            str | None,
            Field(description="e.g. '2019-05-KC-DEN'. Use this or the other three.", max_length=40),
        ] = None,
        season: OptionalSeason = None,
        week: OptionalWeek = None,
        team: Annotated[
            str | None, Field(description=f"Either team in the game. {_TEAM}", max_length=60)
        ] = None,
    ) -> GetGameResult:
        """One finished game: teams, final score and every touchdown in order.

        Pass game_id on its own, or season, week and team together.
        """
        with database() as db:
            return games.get_game(db, game_id=game_id, season=season, week=week, team=team)

    @tool("List games")
    def list_games(
        season: Season,
        week: OptionalWeek = None,
        team: OptionalTeam = None,
        season_type: SeasonTypeArg = "all",
        limit: PageSize = 50,
        cursor: OptionalCursor = None,
    ) -> ListGamesResult:
        """Finished games in one season, in the order they were played, with final scores and
        touchdown counts."""
        with database() as db:
            return games.list_games(
                db,
                season=season,
                week=week,
                team=team,
                season_type=season_type,
                limit=limit,
                cursor=cursor,
            )

    @tool("Get leaderboard")
    def get_leaderboard(
        season: Annotated[
            int | None,
            Field(description=f"{_SEASON} Omit to rank across all seasons.", ge=1900, le=2100),
        ] = None,
        td_type: OptionalTdType = None,
        team: OptionalTeam = None,
        season_type: SeasonTypeArg = "regular",
        per_season: Annotated[
            bool,
            Field(
                description="With no season given: false adds each player's seasons together; "
                "true ranks single seasons, for 'most in one season' questions."
            ),
        ] = False,
        limit: Annotated[int, Field(description="How many ranks to return.", ge=1, le=50)] = 10,
    ) -> LeaderboardResult:
        """Rank players by touchdowns scored.

        Ranks by regular-season touchdowns unless season_type says otherwise; `ranked_by` in
        the result states exactly what was counted. Players tied at the last rank are all
        included, so more rows than `limit` can come back. Passes thrown are not counted.
        """
        with database() as db:
            return leaderboard.get_leaderboard(
                db,
                season=season,
                td_type=td_type,
                team=team,
                season_type=season_type,
                per_season=per_season,
                limit=limit,
            )

    return mcp
