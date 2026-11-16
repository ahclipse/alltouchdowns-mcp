"""Checks that run on a freshly built database, and the swap into place."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from alltouchdowns_mcp.export import validate
from alltouchdowns_mcp.export.build import build
from alltouchdowns_mcp.export.validate import Expectations
from tests.conftest import TODAY, SiteDb


def built(site: SiteDb, tmp_path: Path, name: str = "mcp.db") -> Path:
    out = tmp_path / name
    build(site.save(), out, today=TODAY)
    return out


def one_game(site: SiteDb, scorer: str = "Jamaal Example", touchdowns: int = 2) -> None:
    player = site.player(scorer)
    game = site.game(2025, 1, "KC", "DEN")
    for _ in range(touchdowns):
        site.touchdown(game, player)


def tamper(path: Path, sql: str) -> None:
    db = sqlite3.connect(path)
    db.execute(sql)
    db.commit()
    db.close()


# --- Legal scan ------------------------------------------------------------------------------


def test_legal_scan_passes_clean_data_and_allowlisted_names(site: SiteDb, tmp_path: Path) -> None:
    one_game(site, scorer="Colt McCoy")

    assert validate.check_legal(built(site, tmp_path)) == []


def test_legal_scan_fails_on_a_player_name_that_is_not_allowlisted(
    site: SiteDb, tmp_path: Path
) -> None:
    one_game(site, scorer="Raven Greene")

    problems = validate.check_legal(built(site, tmp_path))

    assert any("players.display_name" in p and "Raven Greene" in p for p in problems)
    assert any("touchdowns.scorer_name" in p for p in problems)


def test_legal_scan_gives_cities_no_allowlist(site: SiteDb, tmp_path: Path) -> None:
    one_game(site)
    out = built(site, tmp_path)
    tamper(out, "UPDATE teams SET label = 'Colt McCoy' WHERE abbr = 'KC'")

    assert validate.check_legal(out) == ["banned term in teams.label: 'Colt McCoy'"]


def test_legal_scan_covers_every_text_column_not_a_hand_kept_list(
    site: SiteDb, tmp_path: Path
) -> None:
    one_game(site)
    out = built(site, tmp_path)
    tamper(out, "UPDATE games SET away_franchise = 'Chiefs'")
    tamper(out, "UPDATE players SET search_name = 'super bowl jones'")

    problems = validate.check_legal(out)

    assert "banned term in games.away_franchise: 'Chiefs'" in problems
    assert "banned term in players.search_name: 'super bowl jones'" in problems


# --- Structure -------------------------------------------------------------------------------


def test_structure_check_passes_a_consistent_database(site: SiteDb, tmp_path: Path) -> None:
    one_game(site)

    assert validate.check_structure(built(site, tmp_path), only_seasons=[2025]) == []


def test_structure_check_notices_a_missing_touchdown_row(site: SiteDb, tmp_path: Path) -> None:
    one_game(site, touchdowns=3)
    out = built(site, tmp_path)
    tamper(out, "DELETE FROM touchdowns WHERE game_seq = 2")

    problems = validate.check_structure(out, only_seasons=[2025])

    assert any("touchdown_count disagrees" in p for p in problems)
    assert any("not numbered 1..n" in p for p in problems)
    assert any("stored totals disagree" in p for p in problems)


def test_structure_check_notices_a_dangling_reference(site: SiteDb, tmp_path: Path) -> None:
    one_game(site)
    out = built(site, tmp_path)
    tamper(out, "DELETE FROM players")

    problems = validate.check_structure(out, only_seasons=[2025])

    assert any("was not exported" in p for p in problems)


def test_structure_check_notices_a_touchdown_filed_under_the_wrong_week(
    site: SiteDb, tmp_path: Path
) -> None:
    one_game(site)
    out = built(site, tmp_path)
    tamper(out, "UPDATE touchdowns SET week = 99 WHERE game_seq = 1")

    problems = validate.check_structure(out, only_seasons=[2025])

    assert any("disagrees with its game" in p for p in problems)


def test_structure_check_notices_an_incomplete_flag_on_a_game_with_touchdowns(
    site: SiteDb, tmp_path: Path
) -> None:
    one_game(site)
    out = built(site, tmp_path)
    tamper(out, "UPDATE games SET touchdowns_incomplete = 1")

    problems = validate.check_structure(out, only_seasons=[2025])

    assert any("flagged incomplete has touchdown rows" in p for p in problems)


def test_structure_check_reports_a_damaged_file(site: SiteDb, tmp_path: Path) -> None:
    one_game(site)
    out = built(site, tmp_path)
    data = bytearray(out.read_bytes())
    data[4096:8192] = b"\xff" * 4096  # overwrite the second page
    out.write_bytes(bytes(data))

    problems = validate.check_structure(out, only_seasons=[2025])

    assert len(problems) == 1 and "damaged" in problems[0]


def test_structure_check_compares_seasons_with_what_was_asked_for(
    site: SiteDb, tmp_path: Path
) -> None:
    one_game(site)

    problems = validate.check_structure(built(site, tmp_path), only_seasons=[2007, 2019])

    assert problems == ["expected seasons [2007, 2019], found [2025]"]


def test_full_export_must_cover_every_season_and_be_a_plausible_size(
    site: SiteDb, tmp_path: Path
) -> None:
    one_game(site)

    problems = validate.check_structure(built(site, tmp_path))

    assert "seasons are not continuous from 1999: [2025]" in problems
    assert any("franchises" in p for p in problems)
    assert "only 1 games, expected at least 7400" in problems


def test_size_ranges_apply_to_completed_seasons_only(site: SiteDb, tmp_path: Path) -> None:
    player = site.player("Jamaal Example")
    site.touchdown(site.game(2024, 22, "KC", "NYG", postseason=True), player)  # 2024 complete
    site.touchdown(site.game(2025, 1, "KC", "DEN"), player)  # 2025 still in progress

    problems = validate.check_structure(built(site, tmp_path), only_seasons=[2024, 2025])

    assert problems == ["season 2024 has 1 games", "season 2024 has 1 touchdowns"]
    relaxed = Expectations(season_games=(1, 290), season_touchdowns=(1, 1700))
    assert (
        validate.check_structure(tmp_path / "mcp.db", only_seasons=[2024, 2025], expect=relaxed)
        == []
    )


# --- Comparison with the database being served -----------------------------------------------


def test_previous_comparison_accepts_growth_and_rejects_shrinkage(
    site: SiteDb, tmp_path: Path
) -> None:
    player = site.player("Jamaal Example")
    for week in range(1, 11):
        game = site.game(2025, week, "KC", "DEN")
        for _ in range(4):
            site.touchdown(game, player)
    bigger = built(site, tmp_path, "bigger.db")

    smaller = tmp_path / "smaller.db"
    smaller.write_bytes(bigger.read_bytes())
    tamper(smaller, "DELETE FROM touchdowns WHERE week > 5")
    tamper(smaller, "DELETE FROM games WHERE week > 5")

    assert validate.check_against_previous(bigger, smaller) == []
    assert validate.check_against_previous(smaller, bigger) == [
        "games went from 10 to 5",
        "touchdowns went from 40 to 20",
    ]


def test_previous_comparison_rejects_implausible_growth_and_lost_players(
    site: SiteDb, tmp_path: Path
) -> None:
    for number in range(1, 43):  # 42 games, one scorer each, 7 touchdowns a game
        game = site.game(2025, number, "KC", "DEN")
        scorer = site.player(f"Scorer Number{number}")
        for _ in range(7):
            site.touchdown(game, scorer)
    big = built(site, tmp_path, "big.db")

    small = tmp_path / "small.db"
    small.write_bytes(big.read_bytes())
    tamper(small, "DELETE FROM touchdowns WHERE week > 1")
    tamper(small, "DELETE FROM games WHERE week > 1")
    tamper(small, "DELETE FROM players WHERE slug <> 'scorer-number1'")

    # 1 game -> 42 is more than a day's worth; 7 touchdowns -> 294 likewise.
    assert validate.check_against_previous(big, small) == [
        "games went from 1 to 42",
        "touchdowns went from 7 to 294",
    ]
    assert "players went from 42 to 1" in validate.check_against_previous(small, big)


# --- Publishing ------------------------------------------------------------------------------


def test_publish_swaps_the_file_and_keeps_the_previous_one(tmp_path: Path) -> None:
    out = tmp_path / "mcp.db"
    out.write_text("old")
    candidate = tmp_path / "mcp.db.tmp"
    candidate.write_text("new")

    validate.publish(candidate, out)

    assert out.read_text() == "new"
    assert (tmp_path / "mcp.db.prev").read_text() == "old"
    assert not candidate.exists()


def test_publish_works_the_first_time(tmp_path: Path) -> None:
    candidate = tmp_path / "mcp.db.tmp"
    candidate.write_text("new")

    validate.publish(candidate, tmp_path / "mcp.db")

    assert (tmp_path / "mcp.db").read_text() == "new"
    assert not (tmp_path / "mcp.db.prev").exists()
