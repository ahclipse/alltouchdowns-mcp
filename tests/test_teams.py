"""Turning what a caller typed into a franchise, and abbreviations into labels."""

from __future__ import annotations

import pytest

from alltouchdowns_mcp.db import Database
from alltouchdowns_mcp.errors import ToolRefusal


@pytest.mark.parametrize(
    ("typed", "franchise"),
    [
        ("KC", "KC"),
        ("kc", "KC"),
        ("  Kansas City ", "KC"),
        ("SD", "LAC"),  # an old abbreviation finds today's franchise
        ("San Diego", "LAC"),
        ("LAC", "LAC"),
        ("Los Angeles (LAC)", "LAC"),  # the label the server itself shows
        ("St. Louis", "LA"),
        ("St Louis", "LA"),
        ("Oakland", "LV"),
        ("LAR", "LA"),  # an alternative spelling some sources use
    ],
)
def test_resolve(db: Database, typed: str, franchise: str) -> None:
    assert db.teams.resolve(typed) == franchise


@pytest.mark.parametrize(
    ("city", "options"),
    [
        ("Los Angeles", "Los Angeles (LA) or Los Angeles (LAC)"),
        ("new york", "New York (NYG) or New York (NYJ)"),
    ],
)
def test_city_with_two_teams_asks_for_the_abbreviation(
    db: Database, city: str, options: str
) -> None:
    with pytest.raises(ToolRefusal) as refusal:
        db.teams.resolve(city)

    assert refusal.value.code == "ambiguous_team"
    assert options in str(refusal.value)


@pytest.mark.parametrize("typed", ["Chiefs", "the Raiders", "Springfield", ""])
def test_unknown_team_lists_valid_abbreviations_and_never_repeats_the_input(
    db: Database, typed: str
) -> None:
    with pytest.raises(ToolRefusal) as refusal:
        db.teams.resolve(typed)

    message = str(refusal.value)
    assert refusal.value.code == "unknown_team"
    assert "KC" in message and "LAC" in message
    assert "Chiefs" not in message and "Raiders" not in message and "Springfield" not in message


def test_labels_show_the_city_of_the_era(db: Database) -> None:
    assert db.teams.label("SD") == "San Diego"
    assert db.teams.label("LAC") == "Los Angeles (LAC)"
    assert db.teams.label("NYG") == "New York (NYG)"
    assert db.teams.team("STL").model_dump() == {"abbr": "STL", "name": "St. Louis"}


def test_describe_lists_every_city_a_franchise_has_played_in(db: Database) -> None:
    assert db.teams.describe("LAC").names == ["San Diego", "Los Angeles (LAC)"]
    assert db.teams.describe("LA").names == ["St. Louis", "Los Angeles (LA)"]
    assert db.teams.describe("KC").names == ["Kansas City"]
