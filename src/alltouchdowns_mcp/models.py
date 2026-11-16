"""The shapes the tools return.

These Pydantic models are also the tools' published output schemas, so field names and
descriptions here are read by the client model. Keep them short and literal.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

TdType = Literal[
    "receiving", "rushing", "int_return", "fumble_return", "punt_return", "kickoff_return", "other"
]
SeasonType = Literal["all", "regular", "postseason"]
Stage = Literal["Regular season", "Playoffs", "Championship"]
MatchKind = Literal["exact", "all_words", "word_prefix", "last_name_only"]


def _drop_field_titles(schema: dict[str, Any]) -> None:
    """Pydantic gives every field a title ("Game Id" for game_id). It says nothing the name
    does not, and it nearly doubles the size of the schemas every client downloads."""
    for field in schema.get("properties", {}).values():
        field.pop("title", None)


class Model(BaseModel):
    """Base class for everything a tool returns."""

    model_config = ConfigDict(json_schema_extra=_drop_field_titles)


class Source(Model):
    """Where the data comes from and how much of it is loaded. On every result."""

    attribution: str
    seasons: str = Field(description="Seasons this server covers, e.g. '1999 to 2026'.")
    data_through: str = Field(description="Date of the most recent game included.")
    season_in_progress: bool = Field(
        description="True while the latest season has not had its Championship game."
    )


class TdCount(Model):
    """Touchdown counts are always split, because the two are easy to confuse."""

    regular_season: int
    postseason: int
    total: int


class Team(Model):
    abbr: str = Field(description="Abbreviation used in that season, e.g. 'SD'.")
    name: str = Field(description="City as of that season, e.g. 'San Diego'.")


class Scorer(Model):
    slug: str | None = Field(description="Player id. Null if the scorer is not a known player.")
    name: str | None = Field(description="Null when the touchdown was credited to the team.")


class Score(Model):
    away: int
    home: int


class Touchdown(Model):
    game_id: str
    game_seq: int = Field(description="Position among the game's touchdowns, starting at 1.")
    season: int
    week_label: str = Field(description="'Week 7' or 'Championship'.")
    stage: Stage
    date: str
    team: Team = Field(description="The team that scored.")
    opponent: Team
    scorer: Scorer
    td_type: TdType
    quarter: int | None
    overtime: bool
    clock: str | None = Field(description="Time left in the quarter, MM:SS.")
    yards: int | None
    score_after: Score | None = Field(
        description="Score right after the touchdown, before the extra-point try."
    )


class TeamScore(Model):
    abbr: str
    name: str
    score: int


class Game(Model):
    game_id: str
    season: int
    week: int
    week_label: str
    stage: Stage
    date: str
    away: TeamScore
    home: TeamScore
    touchdown_count: int
    touchdowns_incomplete: bool = Field(
        description="True if the source is missing this game's touchdown records."
    )


class PlayerMatch(Model):
    slug: str
    name: str
    position: str | None
    first_season: int = Field(description="First season with a touchdown.")
    last_season: int
    teams: list[Team]
    touchdowns: TdCount
    match: MatchKind | None = Field(default=None, description="How the name matched the query.")


class TypeCount(Model):
    td_type: TdType
    touchdowns: TdCount


class SeasonCount(Model):
    season: int
    teams: list[Team]
    touchdowns: TdCount


class TeamCount(Model):
    team: Team
    first_season: int
    last_season: int
    touchdowns: TdCount


class ResolvedTeam(Model):
    """How a `team` argument was understood. A franchise can span several cities."""

    franchise: str
    names: list[str]


class AppliedFilters(Model):
    """The filters as the server understood them. Check they are what you meant: an argument
    the server does not know is refused, but a filter you forgot to pass is simply absent."""

    player: PlayerMatch | None = None
    team: ResolvedTeam | None = None
    season: int | None = None
    week: int | None = None
    td_type: TdType | None = None
    quarter: int | None = None
    min_yards: int | None = None
    season_type: SeasonType = "all"


class LeaderboardRow(Model):
    rank: int
    season: int | None = Field(default=None, description="Set when ranking single seasons.")
    player: Scorer
    position: str | None
    touchdowns: TdCount


class TieOmitted(Model):
    """A tied group too large to list."""

    rank: int
    players: int
    touchdowns: int


# --- One result model per tool ---------------------------------------------------------------


class FindPlayersResult(Model):
    matches: list[PlayerMatch]
    total_matches: int
    notes: list[str]
    source: Source


class GetPlayerResult(Model):
    player: PlayerMatch
    by_type: list[TypeCount]
    by_season: list[SeasonCount]
    by_team: list[TeamCount]
    first_touchdown: Touchdown
    last_touchdown: Touchdown
    same_name_players: list[PlayerMatch] = Field(
        description="Other players with the same name. Check you have the right one."
    )
    notes: list[str]
    source: Source


class SearchTouchdownsResult(Model):
    matched: TdCount = Field(description="How many touchdowns match in total, on all pages.")
    touchdowns: list[Touchdown]
    next_cursor: str | None
    filters: AppliedFilters
    notes: list[str]
    source: Source


class GetGameResult(Model):
    game: Game
    touchdowns: list[Touchdown]
    notes: list[str]
    source: Source


class ListGamesResult(Model):
    total_matches: int
    games: list[Game]
    next_cursor: str | None
    source: Source


class LeaderboardResult(Model):
    ranked_by: str = Field(description="Which count the ranking uses.")
    rows: list[LeaderboardRow]
    tie_omitted: TieOmitted | None = None
    filters: AppliedFilters
    notes: list[str]
    source: Source
