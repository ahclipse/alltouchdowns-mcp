"""The banned-term filter. These tests necessarily spell out the terms they ban."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from alltouchdowns_mcp import ATTRIBUTION, legal


@pytest.mark.parametrize(
    "text",
    [
        "Longest run in Super Bowl history.",
        "super bowl",  # the event names are case-insensitive
        "Wild Card weekend",
        "a Pro Bowl receiver",
        "the Divisional round",
        "AFC Championship",
        "nfc championship",
        "SB XLII",
        "sets a Raiders record",
        "the Chiefs",
        "a Chief",
        "49ers",
        "Washington Football Team",
    ],
)
def test_banned_terms_are_found(text: str) -> None:
    assert legal.find_banned(text)


@pytest.mark.parametrize(
    "text",
    [
        "Kansas City",
        "Los Angeles (LAC)",
        "Championship",  # the site's own label for a season's final game
        "Playoffs",
        "Week 19",
        "Antonio Brown",  # singular surname forms are deliberately not banned
        "Bill Belichick",
        "chiefs",  # nickname patterns are case-sensitive, exactly as on the site
        "sb xlii",
        "Cardinale",  # a banned word inside a longer word is not a match
        ATTRIBUTION,
    ],
)
def test_clean_text_passes(text: str) -> None:
    assert legal.find_banned(text) == []


def test_allowlisted_player_names_pass() -> None:
    assert legal.name_allowlist() == {"Bear Pascoe", "Colt McCoy"}
    assert legal.is_clean("Colt McCoy")
    assert legal.is_clean("Bear Pascoe scored; so did Colt McCoy.")


def test_allowlist_never_hides_another_banned_term() -> None:
    assert legal.find_banned("Colt McCoy of the Chiefs") == ["Chiefs"]


def test_allowlist_matches_whole_names_only() -> None:
    assert legal.find_banned("Colt McCoyote") == ["Colt"]
    assert legal.find_banned("Colt") == ["Colt"]


def test_allowlist_can_be_switched_off_for_text_that_is_never_a_name() -> None:
    assert legal.find_banned("Colt McCoy", allow_names=False) == ["Colt"]


def test_patterns_match_javascript_ascii_word_boundaries() -> None:
    # In JavaScript without the `u` flag, "é" is not a word character, so a word boundary
    # falls between "é" and "Chiefs". Python needs re.ASCII to agree.
    assert legal.find_banned("éChiefs") == ["Chiefs"]


# --- Drift check ---------------------------------------------------------------------------

_TYPESCRIPT = r"""
/** Header comment mentioning /slashes/ and [brackets]. */
const BANNED_PATTERNS: RegExp[] = [
  /\bSuper Bowl\b/i,
  /\bSB\s+[IVXLC]+\b/, // a trailing comment after a pattern
  // A full-line comment with a / in it.
  /\b(Raiders?|Chiefs?)\b/,
  /a[/]b/g,
];

export function unrelated() { return [/not-in-the-list/]; }
"""


def test_typescript_parser_reads_sources_and_flags() -> None:
    assert legal.parse_typescript_patterns(_TYPESCRIPT) == (
        legal.Pattern(r"\bSuper Bowl\b", "i"),
        legal.Pattern(r"\bSB\s+[IVXLC]+\b", ""),
        legal.Pattern(r"\b(Raiders?|Chiefs?)\b", ""),
        legal.Pattern("a[/]b", "g"),
    )


def test_typescript_parser_rejects_a_file_without_the_list() -> None:
    with pytest.raises(ValueError):
        legal.parse_typescript_patterns("export const somethingElse = 1;")


def test_drift_is_reported_in_both_directions() -> None:
    problems = legal.drift_from_site(_TYPESCRIPT)
    assert any("site has a pattern this project lacks" in p for p in problems)
    assert any("this project has a pattern the site lacks" in p for p in problems)


@pytest.mark.skipif(
    not os.environ.get("ATD_SITE_PATTERNS"),
    reason="needs the private site's pattern source file: set ATD_SITE_PATTERNS to its path",
)
def test_pattern_list_matches_the_site() -> None:
    site_source = Path(os.environ["ATD_SITE_PATTERNS"])
    assert legal.drift_from_site(site_source.read_text(encoding="utf-8")) == []


def test_whitespace_in_patterns_matches_what_javascript_matches() -> None:
    # JavaScript's \s includes Unicode spaces even without the `u` flag.
    assert legal.find_banned("SB\N{NO-BREAK SPACE}XLII")
    assert legal.find_banned("SB\N{EM SPACE}XLII")
    assert legal.find_banned("SBXLII") == []
