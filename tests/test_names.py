from __future__ import annotations

import pytest

from alltouchdowns_mcp.names import name_tokens, search_name


@pytest.mark.parametrize(
    ("name", "tokens"),
    [
        ("LaDainian Tomlinson", ["ladainian", "tomlinson"]),
        ("C.J. Anderson", ["cj", "anderson"]),
        ("D'Andre Swift", ["dandre", "swift"]),
        ("Audric Estimé", ["audric", "estime"]),
        ("Ha Ha Clinton-Dix", ["ha", "ha", "clinton", "dix"]),
        ("  Extra   Spaces ", ["extra", "spaces"]),
        ("", []),
    ],
)
def test_name_tokens(name: str, tokens: list[str]) -> None:
    assert name_tokens(name) == tokens


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Odell Beckham Jr.", ("odell beckham", "jr")),
        ("Robert Griffin III", ("robert griffin", "iii")),
        ("Steve Smith Sr.", ("steve smith", "sr")),
        ("Patrick Mahomes", ("patrick mahomes", None)),
        ("V", ("v", None)),  # a lone token is a name, never a suffix
    ],
)
def test_search_name_splits_off_the_suffix(name: str, expected: tuple[str, str | None]) -> None:
    assert search_name(name) == expected
