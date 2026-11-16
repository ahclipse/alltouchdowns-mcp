"""get_game and list_games, against the two-season fixture."""

from __future__ import annotations

import pytest

from alltouchdowns_mcp.db import Database
from alltouchdowns_mcp.errors import ToolRefusal
from alltouchdowns_mcp.tools.games import get_game, list_games
from tests.conftest import FIXTURE_DB, rows

# --- get_game --------------------------------------------------------------------------------


def test_game_by_id_lists_touchdowns_in_scoring_order(db: Database) -> None:
    result = get_game(db, game_id="2007-10-STL-NO")

    expected = rows(
        FIXTURE_DB,
        "SELECT player_slug FROM touchdowns WHERE game_id = '2007-10-STL-NO' ORDER BY game_seq",
    )
    assert [td.scorer.slug for td in result.touchdowns] == [slug for (slug,) in expected]
    assert [td.game_seq for td in result.touchdowns] == list(range(1, len(expected) + 1))
    assert result.game.touchdown_count == len(expected) == 8
    assert (result.game.away.name, result.game.home.name) == ("St. Louis", "New Orleans")


def test_game_by_season_week_and_either_team(db: Database) -> None:
    by_away = get_game(db, season=2007, week=21, team="NYG")
    by_home = get_game(db, season=2007, week=21, team="New England")

    assert by_away.game.game_id == by_home.game.game_id == "2007-21-NYG-NE"
    game = by_away.game
    assert (game.week_label, game.stage) == ("Championship", "Championship")
    assert (game.away.name, game.away.score) == ("New York (NYG)", 17)
    assert (game.home.name, game.home.score) == ("New England", 14)
    assert game.date == "2008-02-03"  # played the following February, still the 2007 season


def test_game_can_be_found_by_a_team_s_old_city(db: Database) -> None:
    assert get_game(db, season=2007, week=9, team="San Diego").game.game_id == "2007-09-SD-MIN"


def test_score_after_follows_the_game(db: Database) -> None:
    touchdowns = get_game(db, game_id="2007-21-NYG-NE").touchdowns

    assert touchdowns[0].score_after is not None
    last = touchdowns[-1].score_after
    assert last is not None and (last.away, last.home) == (16, 14)  # before the extra point


def test_game_without_touchdowns_says_so(db: Database) -> None:
    result = get_game(db, game_id="2007-12-MIA-PIT")

    assert result.touchdowns == [] and result.game.touchdown_count == 0
    assert (result.game.away.score, result.game.home.score) == (0, 3)
    assert result.notes == ["No touchdowns were scored in this game."]


def test_game_id_is_matched_loosely(db: Database) -> None:
    assert get_game(db, game_id=" 2007-12-mia-pit ").game.game_id == "2007-12-MIA-PIT"


@pytest.mark.parametrize(
    ("arguments", "code"),
    [
        ({}, "invalid_argument"),
        ({"game_id": "2007-12-MIA-PIT", "season": 2007}, "invalid_argument"),
        ({"season": 2007, "week": 12}, "invalid_argument"),
        ({"game_id": "2007-12-MIA-KC"}, "not_found"),
        ({"season": 2007, "week": 30, "team": "KC"}, "not_found"),
        ({"season": 2012, "week": 1, "team": "KC"}, "not_covered"),
        ({"season": 2007, "week": 1, "team": "Dolphins"}, "unknown_team"),
    ],
)
def test_get_game_refusals(db: Database, arguments: dict, code: str) -> None:
    with pytest.raises(ToolRefusal) as refusal:
        get_game(db, **arguments)
    assert refusal.value.code == code


# --- list_games ------------------------------------------------------------------------------


def test_postseason_games_end_with_the_championship(db: Database) -> None:
    result = list_games(db, season=2019, season_type="postseason")

    assert result.total_matches == len(result.games) == 11
    assert [g.stage for g in result.games[:-1]] == ["Playoffs"] * 10
    assert (result.games[-1].game_id, result.games[-1].stage) == ("2019-21-SF-KC", "Championship")
    assert result.next_cursor is None


def test_games_in_a_week_agree_with_plain_sql(db: Database) -> None:
    result = list_games(db, season=2007, week=3)

    expected = rows(
        FIXTURE_DB,
        "SELECT game_id, touchdown_count FROM games WHERE season = 2007 AND week = 3"
        " ORDER BY played_at, game_id",
    )
    assert [(g.game_id, g.touchdown_count) for g in result.games] == expected


def test_a_team_s_season_is_in_order(db: Database) -> None:
    result = list_games(db, season=2019, team="Kansas City")

    weeks = [g.week for g in result.games]
    assert weeks == sorted(weeks) and result.total_matches == 19  # 16 games plus 3 playoff wins
    assert all("KC" in (g.away.abbr, g.home.abbr) for g in result.games)


def test_paging_visits_every_game_once(db: Database) -> None:
    seen, cursor = [], None
    while True:
        page = list_games(db, season=2007, limit=40, cursor=cursor)
        seen += [g.game_id for g in page.games]
        cursor = page.next_cursor
        if cursor is None:
            break

    assert len(seen) == len(set(seen)) == page.total_matches == 267


@pytest.mark.parametrize(
    ("arguments", "code"),
    [
        ({"season": 2012}, "not_covered"),
        ({"season": 2007, "team": "Los Angeles"}, "ambiguous_team"),
        ({"season": 2007, "limit": 51}, "invalid_argument"),
        ({"season": 2007, "cursor": "garbage"}, "bad_cursor"),
    ],
)
def test_list_games_refusals(db: Database, arguments: dict, code: str) -> None:
    with pytest.raises(ToolRefusal) as refusal:
        list_games(db, **arguments)
    assert refusal.value.code == code


# --- Review findings -------------------------------------------------------------------------


def test_games_are_listed_in_the_order_they_were_played(db: Database) -> None:
    week_one = list_games(db, season=2019, week=1).games

    assert week_one[0].game_id == "2019-01-GB-CHI"  # Thursday night, before Sunday's games
    assert [g.date for g in week_one] == sorted(g.date for g in week_one)


def test_a_team_s_games_are_found_under_its_old_city(db: Database) -> None:
    by_old_city = list_games(db, season=2007, team="San Diego")
    by_franchise = list_games(db, season=2007, team="LAC")

    assert by_old_city.total_matches == by_franchise.total_matches == 19
    assert all("SD" in (g.away.abbr, g.home.abbr) for g in by_old_city.games)


def test_cursor_from_another_search_is_refused(db: Database) -> None:
    cursor = list_games(db, season=2007, limit=5).next_cursor

    with pytest.raises(ToolRefusal) as refusal:
        list_games(db, season=2007, season_type="regular", limit=5, cursor=cursor)
    assert refusal.value.code == "bad_cursor"


def test_game_with_missing_records_is_not_reported_as_scoreless(edge_db: Database) -> None:
    result = get_game(edge_db, game_id="2025-02-KC-NE")

    assert result.game.touchdowns_incomplete and result.touchdowns == []
    assert "Do not report it as a game without touchdowns" in result.notes[0]
    assert list_games(edge_db, season=2025).games[1].touchdowns_incomplete
