"""find_players and get_player, against the two-season fixture."""

from __future__ import annotations

import pytest

from alltouchdowns_mcp.db import Database
from alltouchdowns_mcp.errors import ToolRefusal
from alltouchdowns_mcp.tools.players import find_players, get_player
from tests.conftest import FIXTURE_DB, rows


def slugs(result) -> list[str]:
    return [match.slug for match in result.matches]


# --- find_players ----------------------------------------------------------------------------


def test_namesakes_come_back_together_with_the_prominent_one_first(db: Database) -> None:
    result = find_players(db, "Adrian Peterson")

    # Two different players. The famous one has the suffixed slug, so guessing would be wrong.
    assert slugs(result) == ["adrian-peterson-2", "adrian-peterson"]
    assert [m.match for m in result.matches] == ["exact", "exact"]
    famous, other = result.matches
    assert [team.abbr for team in famous.teams] == ["MIN", "WAS"]
    assert famous.touchdowns.total == 18 and other.touchdowns.total == 3
    assert result.total_matches == 2


@pytest.mark.parametrize(
    ("query", "expected_first", "kind"),
    [
        ("mccaffrey", "christian-mccaffrey", "all_words"),
        ("McCaffrey, Christian", "christian-mccaffrey", "all_words"),
        ("chris mccaff", "christian-mccaffrey", "word_prefix"),
        ("LADAINIAN TOMLINSON", "ladainian-tomlinson", "exact"),
        ("t.j. houshmandzadeh", "t-j-houshmandzadeh", "exact"),
        ("tj houshmandzadeh", "t-j-houshmandzadeh", "exact"),
    ],
)
def test_matching_ignores_case_punctuation_and_word_order(
    db: Database, query: str, expected_first: str, kind: str
) -> None:
    result = find_players(db, query)

    assert result.matches[0].slug == expected_first
    assert result.matches[0].match == kind


def test_better_kinds_of_match_rank_above_more_touchdowns(db: Database) -> None:
    result = find_players(db, "smith", limit=50)

    best_first = ["exact", "all_words", "word_prefix"]
    kinds = [best_first.index(m.match or "") for m in result.matches]
    assert kinds == sorted(kinds)
    whole_word = [m for m in result.matches if m.match == "all_words"]
    totals = [m.touchdowns.total for m in whole_word]
    assert totals == sorted(totals, reverse=True)
    assert whole_word[0].slug == "steve-smith-2"


def test_unknown_first_name_falls_back_to_the_last_name(db: Database) -> None:
    result = find_players(db, "Zebulon Tomlinson")

    assert slugs(result) == ["ladainian-tomlinson"]
    assert result.matches[0].match == "last_name_only"
    assert any("last name" in note for note in result.notes)


def test_no_match_explains_that_only_scorers_are_listed(db: Database) -> None:
    result = find_players(db, "Zzyzx Qqqqq")

    assert result.matches == [] and result.total_matches == 0
    assert any("at least one touchdown" in note for note in result.notes)


def test_limit_caps_the_page_but_not_the_count(db: Database) -> None:
    result = find_players(db, "smith", limit=2)

    assert len(result.matches) == 2
    assert result.total_matches > 2


@pytest.mark.parametrize("query", ["", "a", ".", "  ' "])
def test_query_needs_two_letters(db: Database, query: str) -> None:
    with pytest.raises(ToolRefusal) as refusal:
        find_players(db, query)
    assert refusal.value.code == "invalid_argument"


# --- get_player ------------------------------------------------------------------------------


def test_player_summary_agrees_with_the_touchdown_rows(db: Database) -> None:
    result = get_player(db, "adrian-peterson-2")

    by_season = rows(
        FIXTURE_DB,
        """SELECT season, SUM(postseason = 0), SUM(postseason = 1) FROM touchdowns
           WHERE player_slug = 'adrian-peterson-2' GROUP BY season ORDER BY season""",
    )
    assert [
        (s.season, s.touchdowns.regular_season, s.touchdowns.postseason) for s in result.by_season
    ] == by_season
    assert result.player.touchdowns.total == sum(regular + post for _, regular, post in by_season)
    assert sum(t.touchdowns.total for t in result.by_type) == result.player.touchdowns.total
    assert sum(t.touchdowns.total for t in result.by_team) == result.player.touchdowns.total


def test_player_summary_shows_each_team_in_the_order_he_played_for_them(db: Database) -> None:
    result = get_player(db, "adrian-peterson-2")

    assert [(t.team.abbr, t.first_season, t.last_season) for t in result.by_team] == [
        ("MIN", 2007, 2007),
        ("WAS", 2019, 2019),
    ]
    assert [team.name for team in result.by_season[0].teams] == ["Minnesota"]


def test_first_and_last_touchdown_are_in_order(db: Database) -> None:
    result = get_player(db, "adrian-peterson-2")

    first, last = result.first_touchdown, result.last_touchdown
    assert (first.season, last.season) == (2007, 2019)
    assert (first.game_id, first.game_seq) < (last.game_id, last.game_seq)
    assert first.scorer.slug == "adrian-peterson-2"


def test_player_summary_points_out_the_other_player_of_the_same_name(db: Database) -> None:
    result = get_player(db, "adrian-peterson")

    assert [p.slug for p in result.same_name_players] == ["adrian-peterson-2"]
    assert get_player(db, "randy-moss").same_name_players == []


def test_notes_explain_scorer_credit_and_where_coverage_starts(db: Database) -> None:
    notes = " ".join(get_player(db, "randy-moss").notes)

    assert "Passes thrown are not credited" in notes
    assert "Coverage starts in 2007" in notes  # true of this fixture; 1999 in production


def test_slug_is_matched_loosely_but_never_guessed(db: Database) -> None:
    assert get_player(db, " Randy-Moss ").player.slug == "randy-moss"
    with pytest.raises(ToolRefusal) as refusal:
        get_player(db, "randy moss")
    assert refusal.value.code == "unknown_player"


# --- Review findings -------------------------------------------------------------------------


@pytest.mark.parametrize("query", ["s " * 2000, "x" * 101, "a b c d e f g"])
def test_overlong_queries_are_refused_before_any_work_is_done(db: Database, query: str) -> None:
    with pytest.raises(ToolRefusal) as refusal:
        find_players(db, query)
    assert refusal.value.code == "invalid_argument"


def test_same_name_players_ignores_punctuation_like_the_search_does(edge_db: Database) -> None:
    """ "DJ Example" and "D.J. Example" are an exact match for each other in find_players, so
    get_player must warn about the other one too."""
    assert slugs(find_players(edge_db, "dj example")) == ["dj-example", "d-j-example"]
    assert [p.slug for p in get_player(edge_db, "dj-example").same_name_players] == ["d-j-example"]
    assert [p.slug for p in get_player(edge_db, "d-j-example").same_name_players] == ["dj-example"]


def test_player_summary_warns_when_his_team_has_a_game_with_missing_records(
    edge_db: Database,
) -> None:
    kansas_city_player = " ".join(get_player(edge_db, "dj-example").notes)
    denver_player = " ".join(get_player(edge_db, "d-j-example").notes)

    assert "2025-02-KC-NE" in kansas_city_player
    assert "2025-02-KC-NE" not in denver_player
