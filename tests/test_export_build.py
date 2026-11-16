"""What the export publishes, and what it leaves behind."""

from __future__ import annotations

from pathlib import Path

from alltouchdowns_mcp.export.build import build
from tests.conftest import DESCRIPTION, OLD, RECENT, TODAY, SiteDb, rows


def export(site: SiteDb, tmp_path: Path, **options) -> Path:
    out = tmp_path / "mcp.db"
    build(site.save(), out, today=TODAY, **options)
    return out


def game_ids(out: Path) -> list[str]:
    return [game_id for (game_id,) in rows(out, "SELECT game_id FROM games ORDER BY game_id")]


# --- Which games and touchdowns are published ------------------------------------------------


def test_only_finished_nflverse_games_are_exported(site: SiteDb, tmp_path: Path) -> None:
    player = site.player("Jamaal Example")
    final = site.game(2025, 1, "KC", "DEN")
    site.touchdown(final, player)
    site.game(2025, 2, "KC", "NE", status="scheduled", score=None)
    site.game(2025, 3, "KC", "LV", status="live", score=(7, 0))
    other_provider = site.game(2025, 4, "KC", "NYJ", source="other-provider")
    site.touchdown(other_provider, player, source="other-provider")

    assert game_ids(export(site, tmp_path)) == [final]


def test_touchdowns_from_another_provider_are_never_exported(site: SiteDb, tmp_path: Path) -> None:
    player = site.player("Jamaal Example")
    # A low score, so the only thing that can flag the game is the other provider's row.
    game = site.game(2025, 1, "KC", "DEN", score=(3, 7), date=OLD)
    site.touchdown(game, player, source="other-provider")

    out = export(site, tmp_path)

    # The game is old enough that nflverse is the final word, so it is published, but without
    # the other provider's row and flagged as incomplete.
    assert rows(out, "SELECT touchdown_count, touchdowns_incomplete FROM games") == [(0, 1)]
    assert rows(out, "SELECT COUNT(*) FROM touchdowns") == [(0,)]
    assert rows(out, "SELECT COUNT(*) FROM players") == [(0,)]


def test_recent_game_is_held_back_until_nflverse_touchdowns_arrive(
    site: SiteDb, tmp_path: Path
) -> None:
    player = site.player("Jamaal Example")
    ready = site.game(2026, 10, "KC", "DEN", date=RECENT)
    site.touchdown(ready, player)
    other_provider_only = site.game(2026, 10, "NE", "LV", date=RECENT)
    site.touchdown(other_provider_only, player, source="other-provider")
    site.game(2026, 10, "NYG", "NYJ", date=RECENT)  # no touchdown rows at all yet

    out = tmp_path / "mcp.db"
    result = build(site.save(), out, today=TODAY)

    assert game_ids(out) == [ready]
    assert result.held_back == 2


def test_old_game_without_touchdowns_is_a_real_zero_unless_the_score_says_otherwise(
    site: SiteDb, tmp_path: Path
) -> None:
    site.game(2025, 1, "KC", "DEN", score=(9, 6), date=OLD)
    site.game(2025, 2, "KC", "NE", score=(18, 0), date=OLD)  # six field goals: possible
    site.game(2025, 3, "KC", "LV", score=(0, 19), date=OLD)  # not without a touchdown
    site.game(2025, 4, "KC", "NYJ", score=(10, 27), date=OLD)

    flags = rows(
        export(site, tmp_path), "SELECT game_id, touchdowns_incomplete FROM games ORDER BY game_id"
    )

    assert flags == [
        ("2025-01-KC-DEN", 0),
        ("2025-02-KC-NE", 0),
        ("2025-03-KC-LV", 1),
        ("2025-04-KC-NYJ", 1),
    ]


def test_recent_game_with_rows_from_both_providers_is_held_back(
    site: SiteDb, tmp_path: Path
) -> None:
    """Mid-restatement: publishing now would show only part of the game's touchdowns."""
    player = site.player("Jamaal Example")
    game = site.game(2026, 10, "KC", "DEN", date=RECENT)
    site.touchdown(game, player)
    site.touchdown(game, player, source="other-provider")

    out = tmp_path / "mcp.db"
    result = build(site.save(), out, today=TODAY)

    assert game_ids(out) == []
    assert result.held_back == 1


def test_hold_back_window_is_seven_days(site: SiteDb, tmp_path: Path) -> None:
    # TODAY is 2026-11-16. Neither game has touchdown rows yet.
    site.game(2026, 9, "KC", "DEN", score=(9, 6), date="2026-11-09")  # 7 days ago: still held
    eight_days_old = site.game(2026, 9, "NE", "LV", score=(9, 6), date="2026-11-08")

    assert game_ids(export(site, tmp_path)) == [eight_days_old]


def test_finished_game_still_sourced_from_another_provider_is_counted_as_held_back(
    site: SiteDb, tmp_path: Path
) -> None:
    site.touchdown(site.game(2025, 1, "KC", "DEN"), site.player("Jamaal Example"))
    site.game(2025, 2, "KC", "NE", source="other-provider", date=OLD)

    out = tmp_path / "mcp.db"
    result = build(site.save(), out, today=TODAY)

    assert game_ids(out) == ["2025-01-KC-DEN"]
    assert result.held_back == 1
    assert dict(rows(out, "SELECT key, value FROM meta"))["held_back"] == "1"


# --- Labels ----------------------------------------------------------------------------------


def test_championship_is_the_postseason_week_with_one_game(site: SiteDb, tmp_path: Path) -> None:
    player = site.player("Jamaal Example")
    regular = site.game(2025, 18, "KC", "DEN")
    semifinal_a = site.game(2025, 21, "KC", "NE", postseason=True)
    semifinal_b = site.game(2025, 21, "NYG", "LA", postseason=True)
    final = site.game(2025, 22, "KC", "NYG", postseason=True)
    for game in (regular, semifinal_a, semifinal_b, final):
        site.touchdown(game, player)

    out = export(site, tmp_path)

    assert rows(out, "SELECT game_id, week_label, stage FROM games ORDER BY game_id") == [
        (regular, "Week 18", "Regular season"),
        (semifinal_a, "Week 21", "Playoffs"),
        (semifinal_b, "Week 21", "Playoffs"),
        (final, "Championship", "Championship"),
    ]
    assert rows(out, "SELECT season, final_week, complete FROM seasons") == [(2025, 22, 1)]


def test_a_held_back_game_cannot_make_another_week_look_like_the_championship(
    site: SiteDb, tmp_path: Path
) -> None:
    """Two semifinals, one held back, and the real final not played yet. Counting only the
    exported games would see a postseason week with exactly one game and mislabel it."""
    player = site.player("Jamaal Example")
    exported = site.game(2026, 21, "KC", "NE", postseason=True, date=RECENT)
    site.touchdown(exported, player)
    site.game(2026, 21, "NYG", "LA", postseason=True, date=RECENT)  # held back: no rows yet
    site.game(2026, 22, "KC", "NYG", postseason=True, status="scheduled", score=None)

    out = export(site, tmp_path)

    assert rows(out, "SELECT game_id, week_label, stage FROM games") == [
        (exported, "Week 21", "Playoffs")
    ]
    assert rows(out, "SELECT season, final_week, complete FROM seasons") == [(2026, 22, 0)]


def test_no_championship_until_a_single_game_week_is_scheduled(
    site: SiteDb, tmp_path: Path
) -> None:
    """The last postseason week so far has two games, so it is not the final."""
    player = site.player("Jamaal Example")
    for away, home in (("KC", "NE"), ("NYG", "LA")):
        site.touchdown(site.game(2025, 21, away, home, postseason=True), player)

    out = export(site, tmp_path)

    assert rows(out, "SELECT DISTINCT week_label, stage FROM games") == [("Week 21", "Playoffs")]
    assert rows(out, "SELECT season, final_week, complete FROM seasons") == [(2025, None, 0)]


def test_championship_is_the_last_single_game_week_not_the_first(
    site: SiteDb, tmp_path: Path
) -> None:
    player = site.player("Jamaal Example")
    early = site.game(2025, 19, "KC", "NE", postseason=True)  # a lone game in an early round
    final = site.game(2025, 22, "KC", "NYG", postseason=True)
    for game in (early, final):
        site.touchdown(game, player)

    out = export(site, tmp_path)

    assert rows(out, "SELECT game_id, stage FROM games ORDER BY game_id") == [
        (early, "Playoffs"),
        (final, "Championship"),
    ]


def test_season_is_not_complete_while_its_championship_is_held_back(
    site: SiteDb, tmp_path: Path
) -> None:
    """The final has been played, but its touchdowns have not arrived, so it is not exported.
    The season must not claim to be complete without it."""
    player = site.player("Jamaal Example")
    site.touchdown(site.game(2026, 21, "KC", "NE", postseason=True, date=RECENT), player)
    site.touchdown(site.game(2026, 21, "NYG", "LA", postseason=True, date=RECENT), player)
    site.game(2026, 22, "KC", "NYG", postseason=True, date=RECENT)  # final, no rows yet

    out = export(site, tmp_path)

    assert rows(out, "SELECT COUNT(*) FROM games WHERE stage = 'Championship'") == [(0,)]
    assert rows(out, "SELECT season, final_week, complete FROM seasons") == [(2026, 22, 0)]


def test_team_labels_use_era_cities_and_mark_shared_cities(site: SiteDb, tmp_path: Path) -> None:
    labels = dict(rows(export(site, tmp_path), "SELECT abbr, label FROM teams"))

    assert labels["KC"] == "Kansas City"
    assert labels["SD"] == "San Diego"
    assert labels["LAC"] == "Los Angeles (LAC)"
    assert labels["LA"] == "Los Angeles (LA)"
    assert labels["NYG"] == "New York (NYG)"
    assert rows(tmp_path / "mcp.db", "SELECT franchise FROM teams WHERE abbr = 'SD'") == [("LAC",)]


# --- Touchdown rows --------------------------------------------------------------------------


def test_touchdown_types_are_relabelled_and_clocks_padded(site: SiteDb, tmp_path: Path) -> None:
    player = site.player("Jamaal Example")
    game = site.game(2025, 1, "SD", "DEN")
    site.touchdown(game, player, td_type="pass", clock="2:27")
    site.touchdown(game, player, td_type="rush", clock="14:03")
    site.touchdown(game, player, td_type="int_return", team="DEN", quarter=5)

    out = export(site, tmp_path)

    assert rows(
        out,
        "SELECT game_seq, td_type, clock, team_abbr, franchise, quarter"
        " FROM touchdowns ORDER BY game_seq",
    ) == [
        (1, "receiving", "02:27", "SD", "LAC", 1),
        (2, "rushing", "14:03", "SD", "LAC", 1),
        (3, "int_return", "10:00", "DEN", "DEN", 5),
    ]


def test_unlinked_scorer_keeps_the_play_by_play_name(site: SiteDb, tmp_path: Path) -> None:
    linked = site.player("Jamaal Example")
    game = site.game(2025, 1, "KC", "DEN")
    site.touchdown(game, linked, name="J.Example")
    site.touchdown(game, None, name="O.Amey")  # scorer the site could not link to a player
    site.touchdown(game, None, name=None)  # credited to the team

    out = export(site, tmp_path)

    assert rows(out, "SELECT player_slug, scorer_name FROM touchdowns ORDER BY game_seq") == [
        ("jamaal-example", "Jamaal Example"),
        (None, "O.Amey"),
        (None, None),
    ]


def test_score_after_can_be_missing(site: SiteDb, tmp_path: Path) -> None:
    game = site.game(2025, 1, "KC", "DEN")
    site.touchdown(game, site.player("Jamaal Example"), score_after=(None, None))

    assert rows(
        export(site, tmp_path), "SELECT away_score_after, home_score_after FROM touchdowns"
    ) == [(None, None)]


# --- Players ---------------------------------------------------------------------------------


def test_only_scorers_are_exported_with_their_totals(site: SiteDb, tmp_path: Path) -> None:
    scorer = site.player("Odell Example Jr.", position="WR")
    site.player("Never Scored")
    first = site.game(2024, 1, "SD", "DEN")
    second = site.game(2025, 1, "NYJ", "DEN")
    playoff = site.game(2025, 19, "NYJ", "KC", postseason=True)
    site.touchdown(first, scorer)
    site.touchdown(second, scorer)
    site.touchdown(second, scorer)
    site.touchdown(playoff, scorer)

    assert rows(export(site, tmp_path), "SELECT * FROM players") == [
        (
            "odell-example-jr",
            "Odell Example Jr.",
            "odell example",
            "jr",
            "WR",
            2024,
            2025,
            3,
            1,
            "SD,NYJ",
        ),
    ]


# --- What must never be published ------------------------------------------------------------


def test_descriptions_and_private_columns_are_left_behind(site: SiteDb, tmp_path: Path) -> None:
    game = site.game(2025, 1, "KC", "DEN")
    site.touchdown(game, site.player("Jamaal Example"))

    out = export(site, tmp_path)

    columns = {
        name
        for (name,) in rows(
            out,
            """
        SELECT p.name FROM sqlite_master m, pragma_table_info(m.name) p WHERE m.type = 'table'""",
        )
    }
    forbidden = {
        "description",
        "id",
        "source",
        "source_id",
        "player_id",
        "first_name",
        "last_name",
        "kickoff_time",
    }
    assert columns.isdisjoint(forbidden)
    # Belt and braces: the description text is nowhere in the file, in any form.
    assert b"SECRET-PLAY-TEXT" not in out.read_bytes()
    assert b"LegalFirst" not in out.read_bytes()
    assert DESCRIPTION.encode() not in out.read_bytes()


# --- Seasons and metadata --------------------------------------------------------------------


def test_season_filter_and_metadata(site: SiteDb, tmp_path: Path) -> None:
    player = site.player("Jamaal Example")
    site.touchdown(site.game(2007, 1, "SD", "DEN", date="2007-09-09"), player)
    site.touchdown(site.game(2019, 1, "LAC", "DEN", date="2019-09-08"), player)
    site.touchdown(site.game(2025, 1, "KC", "DEN"), player)

    out = export(site, tmp_path, seasons=[2007, 2019])

    assert game_ids(out) == ["2007-01-SD-DEN", "2019-01-LAC-DEN"]
    meta = dict(rows(out, "SELECT key, value FROM meta"))
    assert meta["coverage"] == "partial"
    assert meta["data_through"] == "2019-09-08"
    assert (meta["games"], meta["touchdowns"], meta["players"]) == ("2", "2", "1")


def test_a_new_season_with_no_finished_games_is_not_listed(site: SiteDb, tmp_path: Path) -> None:
    site.touchdown(site.game(2025, 1, "KC", "DEN"), site.player("Jamaal Example"))
    site.game(2026, 1, "KC", "DEN", status="scheduled", score=None, date="2026-09-13")

    out = export(site, tmp_path)

    assert rows(out, "SELECT season FROM seasons") == [(2025,)]
    assert dict(rows(out, "SELECT key, value FROM meta"))["coverage"] == "full"
