"""The committed two-season fixture is what it claims to be."""

from __future__ import annotations

import hashlib

from alltouchdowns_mcp.export import validate
from tests.conftest import FIXTURE_DB, rows


def test_fixture_matches_its_recorded_hash() -> None:
    recorded = (FIXTURE_DB.parent / "fixture.sha256").read_text().split()[0]

    assert hashlib.sha256(FIXTURE_DB.read_bytes()).hexdigest() == recorded


def test_fixture_holds_two_complete_seasons() -> None:
    assert rows(FIXTURE_DB, "SELECT season, complete FROM seasons") == [(2007, 1), (2019, 1)]
    assert rows(FIXTURE_DB, "SELECT COUNT(*) FROM games") == [(534,)]
    assert rows(FIXTURE_DB, "SELECT COUNT(*) FROM touchdowns") == [(2695,)]
    assert rows(FIXTURE_DB, "SELECT COUNT(*) FROM players") == [(819,)]
    assert dict(rows(FIXTURE_DB, "SELECT key, value FROM meta"))["coverage"] == "partial"


def test_fixture_covers_both_eras_of_the_relocated_teams() -> None:
    def abbreviations(season: int) -> set[str]:
        return {
            a
            for (a,) in rows(
                FIXTURE_DB, "SELECT DISTINCT team_abbr FROM touchdowns WHERE season = ?", season
            )
        }

    assert {"SD", "STL", "OAK"} <= abbreviations(2007)
    assert {"LAC", "LA", "OAK"} <= abbreviations(2019)


def test_fixture_passes_the_export_checks() -> None:
    assert validate.check_legal(FIXTURE_DB) == []
    assert validate.check_structure(FIXTURE_DB, only_seasons=[2007, 2019]) == []


def test_fixture_contains_no_play_descriptions() -> None:
    data = FIXTURE_DB.read_bytes()

    # Play-by-play text always spells the result in capitals; nothing else here does.
    assert b"TOUCHDOWN" not in data
    assert b"nflverse" not in data  # nor any provider name: the source column is not copied
