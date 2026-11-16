"""Player-name normalisation, shared by the export (which stores it) and player search."""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})

# Characters that vanish inside a name: periods, straight and curly apostrophes, backticks.
_SILENT = str.maketrans("", "", ".'`\N{RIGHT SINGLE QUOTATION MARK}")


def name_tokens(name: str) -> list[str]:
    """Split a name into lower-case ASCII words.

    Accents are stripped ("Estimé" -> "estime"). Periods and apostrophes vanish, so initials and
    contractions join up ("C.J." -> "cj", "D'Andre" -> "dandre"). Any other punctuation
    separates words ("Clinton-Dix" -> "clinton", "dix").
    """
    decomposed = unicodedata.normalize("NFKD", name)
    without_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    joined = without_accents.casefold().translate(_SILENT)
    return re.findall(r"[a-z0-9]+", joined)


def split_suffix(tokens: list[str]) -> tuple[list[str], str | None]:
    """Separate a trailing generational suffix: ["odell", "beckham", "jr"] -> (..., "jr")."""
    if len(tokens) > 1 and tokens[-1] in SUFFIXES:
        return tokens[:-1], tokens[-1]
    return tokens, None


def search_name(display_name: str) -> tuple[str, str | None]:
    """The stored search form of a display name, and its suffix if it has one."""
    tokens, suffix = split_suffix(name_tokens(display_name))
    return " ".join(tokens), suffix


def match_kind(
    query: list[str], name: list[str]
) -> Literal["exact", "all_words", "word_prefix"] | None:
    """How well a query's words match a name's words: the best kind that applies, or None.

    exact        the same letters, ignoring how they are split ("c j anderson" = "cj anderson")
    all_words    every query word is one of the name's words ("mahomes")
    word_prefix  every query word starts one of the name's words ("pat mah")
    """
    if "".join(query) == "".join(name):
        return "exact"
    if all(word in name for word in query):
        return "all_words"
    if all(any(part.startswith(word) for part in name) for word in query):
        return "word_prefix"
    return None
