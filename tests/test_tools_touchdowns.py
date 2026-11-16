"""search_touchdowns, against the two-season fixture."""

from __future__ import annotations

import pytest

from alltouchdowns_mcp.db import Database
from alltouchdowns_mcp.errors import ToolRefusal
from alltouchdowns_mcp.tools.touchdowns import search_touchdowns
from tests.conftest import FIXTURE_DB, rows


def count(where: str, *params: object) -> int:
    return rows(FIXTURE_DB, f"SELECT COUNT(*) FROM touchdowns WHERE {where}", *params)[0][0]


def every_page(db: Database, **filters) -> list[tuple[str, int]]:
    """Follow next_cursor to the end and return every touchdown's key, in order."""
    keys, cursor = [], None
    for _ in range(200):
        result = search_touchdowns(db, cursor=cursor, **filters)
        keys += [(td.game_id, td.game_seq) for td in result.touchdowns]
        cursor = result.next_cursor
        if cursor is None:
            return keys
    raise AssertionError("paging never ended")


# --- Filters ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filters", "where"),
    [
        ({"season": 2007}, "season = 2007"),
        ({"season": 2019, "week": 5}, "season = 2019 AND week = 5"),
        ({"season": 2007, "td_type": "punt_return"}, "season = 2007 AND td_type = 'punt_return'"),
        ({"season": 2019, "quarter": 2}, "season = 2019 AND quarter = 2"),
        ({"season": 2019, "quarter": 5}, "season = 2019 AND quarter >= 5"),  # 5 means overtime
        ({"season": 2007, "min_yards": 80}, "season = 2007 AND yards >= 80"),
        ({"season": 2019, "season_type": "regular"}, "season = 2019 AND postseason = 0"),
        ({"season": 2019, "season_type": "postseason"}, "season = 2019 AND postseason = 1"),
        ({"player": "randy-moss"}, "player_slug = 'randy-moss'"),
        ({"team": "KC"}, "franchise = 'KC'"),
        ({"team": "San Diego", "td_type": "rushing"}, "franchise = 'LAC' AND td_type = 'rushing'"),
    ],
)  # fmt: skip
def test_matched_count_agrees_with_plain_sql(db: Database, filters: dict, where: str) -> None:
    result = search_touchdowns(db, **filters)

    assert result.matched.total == count(where)
    assert result.matched.regular_season == count(f"({where}) AND postseason = 0")
    assert result.matched.postseason == count(f"({where}) AND postseason = 1")


def test_matched_is_the_whole_count_not_the_page(db: Database) -> None:
    result = search_touchdowns(db, season=2007, limit=5)

    assert len(result.touchdowns) == 5
    assert result.matched.total == count("season = 2007") > 1000


def test_team_matches_the_franchise_and_labels_each_era(db: Database) -> None:
    result = search_touchdowns(db, team="SD", limit=50)

    assert result.filters.team is not None
    assert result.filters.team.names == ["San Diego", "Los Angeles (LAC)"]
    assert result.matched.total == count("franchise = 'LAC'") == 55 + 37
    assert {td.team.name for td in result.touchdowns} == {"San Diego"}  # first page is 2007

    recent = search_touchdowns(db, team="SD", season=2019, limit=1).touchdowns[0]
    assert (recent.team.abbr, recent.team.name) == ("LAC", "Los Angeles (LAC)")


def test_touchdown_rows_carry_opponent_stage_and_overtime(db: Database) -> None:
    result = search_touchdowns(db, season=2019, quarter=5, season_type="regular")

    assert {td.game_id for td in result.touchdowns} == {
        "2019-01-IND-LAC",
        "2019-09-TB-SEA",
        "2019-14-NYG-PHI",
        "2019-16-NYG-WAS",
        "2019-17-ATL-TB",
    }
    for td in result.touchdowns:
        assert td.overtime and td.stage == "Regular season"
        away, home = td.game_id.split("-")[2:]
        assert {td.team.abbr, td.opponent.abbr} == {away, home}


def test_player_filter_echoes_who_it_understood(db: Database) -> None:
    result = search_touchdowns(db, player="adrian-peterson")

    assert result.filters.player is not None
    assert result.filters.player.teams[0].abbr == "CHI"  # the other Adrian Peterson
    assert result.matched.total == 3


# --- Order and paging ------------------------------------------------------------------------


@pytest.mark.parametrize("order", ["oldest", "newest", "longest"])
def test_paging_visits_every_row_exactly_once(db: Database, order: str) -> None:
    keys = every_page(db, team="KC", order=order, limit=7)

    assert len(keys) == len(set(keys)) == count("franchise = 'KC'")


def test_orders(db: Database) -> None:
    oldest = every_page(db, team="KC", order="oldest", limit=50)
    assert oldest == sorted(oldest)
    assert every_page(db, team="KC", order="newest", limit=50) == oldest[::-1]

    longest = search_touchdowns(db, season=2007, order="longest", limit=3).touchdowns
    assert [(td.game_id, td.yards) for td in longest][:2] == [
        ("2007-09-SD-MIN", 109),
        ("2007-01-NE-NYJ", 108),
    ]


def test_cursor_belongs_to_one_search(db: Database) -> None:
    cursor = search_touchdowns(db, team="KC", limit=5).next_cursor
    assert cursor is not None

    with pytest.raises(ToolRefusal) as refusal:
        search_touchdowns(db, team="DEN", limit=5, cursor=cursor)
    assert refusal.value.code == "bad_cursor"
    with pytest.raises(ToolRefusal):
        search_touchdowns(db, team="KC", order="newest", limit=5, cursor=cursor)


def test_unfiltered_search_cannot_be_paged_through(db: Database) -> None:
    """The longest touchdowns ever is a fair question. Walking the whole dataset is not."""
    result = search_touchdowns(db, order="longest", limit=5)

    assert len(result.touchdowns) == 5
    assert result.next_cursor is None
    assert any("Add a player, team or season" in note for note in result.notes)


def test_last_page_has_no_cursor(db: Database) -> None:
    result = search_touchdowns(db, season=2019, quarter=5, season_type="regular", limit=50)

    assert result.next_cursor is None and result.notes == []


# --- Refusals --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filters", "code"),
    [
        ({"season": 2012}, "not_covered"),  # the fixture holds 2007 and 2019 only
        ({"season": 1998}, "not_covered"),
        ({"team": "Chiefs"}, "unknown_team"),
        ({"team": "New York"}, "ambiguous_team"),
        ({"player": "nobody-at-all"}, "unknown_player"),
        ({"quarter": 6}, "invalid_argument"),
        ({"limit": 0}, "invalid_argument"),
        ({"limit": 51}, "invalid_argument"),
        ({"team": "KC", "cursor": "garbage"}, "bad_cursor"),
    ],
)
def test_refusals(db: Database, filters: dict, code: str) -> None:
    with pytest.raises(ToolRefusal) as refusal:
        search_touchdowns(db, **filters)
    assert refusal.value.code == code


def test_no_matches_is_an_answer_not_an_error(db: Database) -> None:
    result = search_touchdowns(db, season=2007, season_type="postseason", quarter=5)

    assert result.matched.total == 0 and result.touchdowns == []


# --- Review findings: time order, cursors, echoed filters, missing records -------------------


def test_oldest_first_means_by_kick_off_not_by_team_name(db: Database) -> None:
    """The 2019 season opened on a Thursday with Green Bay at Chicago. Ordering by game id
    would put Atlanta at Minnesota (the following Sunday) first, because A sorts before G."""
    first = search_touchdowns(db, season=2019, limit=1).touchdowns[0]
    last = search_touchdowns(db, season=2019, order="newest", limit=1).touchdowns[0]

    assert (first.game_id, first.date) == ("2019-01-GB-CHI", "2019-09-05")
    assert (last.game_id, last.stage) == ("2019-21-SF-KC", "Championship")


def test_dates_never_go_backwards_across_pages(db: Database) -> None:
    dates, cursor = [], None
    for _ in range(4):
        page = search_touchdowns(db, season=2019, limit=50, cursor=cursor)
        dates += [td.date for td in page.touchdowns]
        cursor = page.next_cursor
    assert dates == sorted(dates) and len(dates) == 200


def test_result_says_back_which_filters_were_applied(db: Database) -> None:
    filters = search_touchdowns(
        db, season=2019, week=5, td_type="rushing", quarter=4, min_yards=10, season_type="regular"
    ).filters

    assert filters.model_dump(exclude={"player", "team"}) == {
        "season": 2019,
        "week": 5,
        "td_type": "rushing",
        "quarter": 4,
        "min_yards": 10,
        "season_type": "regular",
    }


def test_a_made_up_cursor_cannot_page_through_an_unfiltered_search(db: Database) -> None:
    from alltouchdowns_mcp import cursor as cursors

    search = {"season_type": "all", "order": "oldest", "data": db.coverage.exported_at}
    forged = cursors.encode(("2007-09-06T20:30", "2007-01-NO-IND", 1), search)

    with pytest.raises(ToolRefusal) as refusal:
        search_touchdowns(db, cursor=forged)
    assert refusal.value.code == "bad_cursor"


def test_a_cursor_does_not_survive_a_new_export(db: Database, tmp_path) -> None:
    import shutil
    import sqlite3

    from alltouchdowns_mcp.db import open_database

    cursor = search_touchdowns(db, team="KC", limit=5).next_cursor
    newer = tmp_path / "newer.db"
    shutil.copy(FIXTURE_DB, newer)
    conn = sqlite3.connect(newer)
    conn.execute("UPDATE meta SET value = '2030-01-01T00:00:00Z' WHERE key = 'exported_at'")
    conn.commit()
    conn.close()

    with (
        open_database(newer, step_limit=20_000_000) as replaced,
        pytest.raises(ToolRefusal) as refusal,
    ):
        search_touchdowns(replaced, team="KC", limit=5, cursor=cursor)
    assert refusal.value.code == "bad_cursor"


@pytest.mark.parametrize("min_yards", [-1, 151, 10**20])
def test_min_yards_must_be_a_plausible_length(db: Database, min_yards: int) -> None:
    with pytest.raises(ToolRefusal) as refusal:
        search_touchdowns(db, min_yards=min_yards)
    assert refusal.value.code == "invalid_argument"


def test_counts_that_cover_a_game_with_missing_records_say_so(edge_db: Database) -> None:
    covering = search_touchdowns(edge_db, team="KC", season=2025)
    not_covering = search_touchdowns(edge_db, team="DEN", season=2025)

    assert covering.matched.total == 2
    assert any("2025-02-KC-NE" in note and "lower than the true" in note for note in covering.notes)
    assert not_covering.notes == []
