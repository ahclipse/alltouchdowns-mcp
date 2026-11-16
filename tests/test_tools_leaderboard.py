"""get_leaderboard, against the two-season fixture."""

from __future__ import annotations

import pytest

from alltouchdowns_mcp.db import Database
from alltouchdowns_mcp.errors import ToolRefusal
from alltouchdowns_mcp.tools import leaderboard
from alltouchdowns_mcp.tools.leaderboard import get_leaderboard


def ranking(result) -> list[tuple[int, str | None, int]]:
    return [(r.rank, r.player.slug, r.touchdowns.regular_season) for r in result.rows]


def test_season_leaders(db: Database) -> None:
    result = get_leaderboard(db, season=2007, limit=3)

    assert ranking(result) == [
        (1, "randy-moss", 23),
        (2, "ladainian-tomlinson", 18),
        (3, "braylon-edwards", 16),
    ]
    assert result.ranked_by == "regular-season touchdowns in the 2007 season"


def test_everyone_tied_at_the_cut_off_is_included(db: Database) -> None:
    result = get_leaderboard(db, season=2007, limit=4)

    # Two players share fourth place with 15, so "top 4" has five rows.
    assert ranking(result)[3:] == [(4, "joseph-addai", 15), (4, "terrell-owens", 15)]


def test_ties_can_return_many_more_rows_than_the_limit(db: Database) -> None:
    result = get_leaderboard(db, season=2007, td_type="int_return", limit=3)

    # One player has 3, then five are tied on 2. The next rank after a five-way tie for 2nd
    # would be 7th, so all six are within "top 3".
    assert [(r.rank, r.touchdowns.regular_season) for r in result.rows] == [(1, 3)] + [(2, 2)] * 5
    tied = [r.player.name or "" for r in result.rows[1:]]
    assert tied == sorted(tied)  # ties are listed alphabetically


def test_regular_season_and_all_games_can_have_different_leaders(db: Database) -> None:
    regular = get_leaderboard(db, season=2019, limit=1)
    everything = get_leaderboard(db, season=2019, season_type="all", limit=1)
    postseason = get_leaderboard(db, season=2019, season_type="postseason", limit=1)

    assert [r.player.slug for r in regular.rows] == ["aaron-jones-2", "christian-mccaffrey"]
    assert [(r.player.slug, r.touchdowns.total) for r in everything.rows] == [("aaron-jones-2", 23)]
    assert everything.ranked_by == (
        "regular-season and postseason touchdowns combined in the 2019 season"
    )
    assert postseason.rows[0].touchdowns.postseason >= 4
    assert all(r.touchdowns.postseason > 0 for r in postseason.rows)


def test_every_row_reports_both_counts_whatever_the_ranking_uses(db: Database) -> None:
    moss = get_leaderboard(db, season=2007, limit=1).rows[0]

    assert moss.touchdowns.model_dump() == {"regular_season": 23, "postseason": 1, "total": 24}


def test_filters_by_team_across_eras_and_by_type(db: Database) -> None:
    result = get_leaderboard(db, team="San Diego", td_type="rushing", limit=1)

    assert result.filters.team is not None and result.filters.team.franchise == "LAC"
    assert result.rows[0].player.slug == "ladainian-tomlinson"


def test_across_all_seasons_when_no_season_is_given(db: Database) -> None:
    one_season = get_leaderboard(db, season=2007, limit=1).rows[0].touchdowns.regular_season
    both = get_leaderboard(db, limit=1).rows[0].touchdowns.regular_season

    assert both >= one_season


def test_a_tied_group_too_large_to_list_is_reported_as_a_count(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(leaderboard, "MAX_ROWS", 3)

    result = get_leaderboard(db, season=2007, td_type="int_return", limit=3)

    # The six rows exceed the cap, so the five-way tie is dropped and described instead.
    assert [r.player.slug for r in result.rows] == ["antrel-rolle"]
    assert result.tie_omitted is not None
    assert result.tie_omitted.model_dump() == {"rank": 2, "players": 5, "touchdowns": 2}


@pytest.mark.parametrize(
    ("arguments", "code"),
    [
        ({"season": 2012}, "not_covered"),
        ({"limit": 0}, "invalid_argument"),
        ({"limit": 51}, "invalid_argument"),
        ({"team": "Patriots"}, "unknown_team"),
    ],
)
def test_refusals(db: Database, arguments: dict, code: str) -> None:
    with pytest.raises(ToolRefusal) as refusal:
        get_leaderboard(db, **arguments)
    assert refusal.value.code == code


# --- Review findings -------------------------------------------------------------------------


def test_without_a_season_the_result_says_the_seasons_were_added_together(db: Database) -> None:
    result = get_leaderboard(db, limit=1)

    assert result.ranked_by == "regular-season touchdowns added up over seasons 2007 and 2019"
    assert result.rows[0].season is None


def test_per_season_ranks_single_seasons(db: Database) -> None:
    """ "Most touchdowns in one season" is a different question from "most in total"."""
    result = get_leaderboard(db, per_season=True, limit=2)

    assert result.ranked_by == "regular-season touchdowns in a single season"
    assert [(r.player.slug, r.season, r.touchdowns.regular_season) for r in result.rows] == [
        ("randy-moss", 2007, 23),
        ("aaron-jones-2", 2019, 19),
        ("christian-mccaffrey", 2019, 19),
    ]


def test_a_team_that_missed_the_playoffs_has_an_empty_postseason_leaderboard(db: Database) -> None:
    result = get_leaderboard(db, season=2019, team="MIA", season_type="postseason")

    assert result.rows == [] and result.tie_omitted is None


def test_leaderboard_warns_when_a_game_s_records_are_missing(edge_db: Database) -> None:
    assert any("2025-02-KC-NE" in note for note in get_leaderboard(edge_db, season=2025).notes)
    assert get_leaderboard(edge_db, season=2025, team="DEN").notes == []
