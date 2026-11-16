"""Legal-policy filter.

The site never shows team nicknames or trademarked event names. This module holds the same
banned-term patterns (ported from the site's TypeScript list) and checks every piece of text
this project can emit against them.

Some real players' names contain a banned word, "Colt McCoy" for example. Those exact names are
on a small reviewed allowlist. An allowlisted name is masked before the patterns run, so it can
never hide a banned term elsewhere in the same text.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cache
from importlib import resources

_DATA = resources.files("alltouchdowns_mcp") / "data"

# Everything JavaScript's \s matches (ECMAScript WhiteSpace and LineTerminator).
_JS_WHITESPACE = (
    "[\\t\\n\\v\\f\\r \\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000\\ufeff]"
)


@dataclass(frozen=True)
class Pattern:
    """One banned-term regex, stored exactly as the site's TypeScript writes it."""

    source: str
    flags: str  # JavaScript flags: "" or "i"

    def compile(self) -> re.Pattern[str]:
        """Compile for Python so that it matches exactly what the JavaScript regex matches."""
        # Without the `u` flag, JavaScript treats \b and \w as ASCII-only. re.ASCII gives
        # Python the same behaviour.
        flags = re.ASCII
        if "i" in self.flags:
            flags |= re.IGNORECASE
        # \s is the exception: in JavaScript it always includes Unicode spaces such as the
        # no-break space, which re.ASCII would exclude. So \s is spelled out.
        source = re.sub(r"(?<!\\)\\s", lambda _: _JS_WHITESPACE, self.source)
        return re.compile(source, flags)


@cache
def banned_patterns() -> tuple[Pattern, ...]:
    raw = json.loads((_DATA / "banned_patterns.json").read_text(encoding="utf-8"))
    return tuple(Pattern(p["source"], p["flags"]) for p in raw["patterns"])


@cache
def name_allowlist() -> frozenset[str]:
    lines = (_DATA / "name_allowlist.txt").read_text(encoding="utf-8").splitlines()
    return frozenset(s for line in lines if (s := line.strip()) and not s.startswith("#"))


@cache
def _compiled() -> tuple[re.Pattern[str], ...]:
    return tuple(p.compile() for p in banned_patterns())


@cache
def _allowlist_mask() -> re.Pattern[str] | None:
    names = sorted(name_allowlist(), key=len, reverse=True)
    if not names:
        return None
    alternation = "|".join(re.escape(name) for name in names)
    # Whole names only: "Colt McCoy" is masked, "Colt McCoyote" is not.
    return re.compile(rf"(?<!\w)(?:{alternation})(?!\w)")


def find_banned(text: str, *, allow_names: bool = True) -> list[str]:
    """Return the banned terms found in `text`. An empty list means the text is clean.

    Pass `allow_names=False` for text that is never a person's name (cities, labels), so the
    allowlist cannot excuse anything there.
    """
    if allow_names and (mask := _allowlist_mask()) is not None:
        text = mask.sub(" ", text)
    return [match.group(0) for pattern in _compiled() if (match := pattern.search(text))]


def is_clean(text: str, *, allow_names: bool = True) -> bool:
    return not find_banned(text, allow_names=allow_names)


# --- Drift check against the site's TypeScript source -------------------------------------


def parse_typescript_patterns(ts_source: str) -> tuple[Pattern, ...]:
    """Extract the regex literals from the `BANNED_PATTERNS` array in the site's TypeScript."""
    name_at = ts_source.find("BANNED_PATTERNS")
    array_at = ts_source.find("= [", name_at)
    if name_at == -1 or array_at == -1:
        raise ValueError("BANNED_PATTERNS array not found")

    patterns: list[Pattern] = []
    i = array_at + len("= [")
    while i < len(ts_source):
        if ts_source.startswith("//", i):
            i = _index_or_end(ts_source, "\n", i)
        elif ts_source.startswith("/*", i):
            i = _index_or_end(ts_source, "*/", i) + 2
        elif ts_source[i] == "/":
            pattern, i = _read_regex_literal(ts_source, i)
            patterns.append(pattern)
        elif ts_source[i] == "]":
            return tuple(patterns)
        else:
            i += 1
    raise ValueError("BANNED_PATTERNS array is not closed")


def _index_or_end(text: str, needle: str, start: int) -> int:
    found = text.find(needle, start)
    return len(text) if found == -1 else found


def _read_regex_literal(text: str, start: int) -> tuple[Pattern, int]:
    """Read `/source/flags` beginning at `start`. Returns the pattern and the index after it."""
    i = start + 1
    in_class = False  # inside [...], where an unescaped "/" does not end the literal
    while i < len(text):
        char = text[i]
        if char == "\\":
            i += 2
            continue
        if char == "\n":
            break
        if char == "[":
            in_class = True
        elif char == "]":
            in_class = False
        elif char == "/" and not in_class:
            end = i + 1
            while end < len(text) and text[end].isalpha():
                end += 1
            return Pattern(text[start + 1 : i], text[i + 1 : end]), end
        i += 1
    raise ValueError(f"unterminated regex literal at offset {start}")


def drift_from_site(ts_source: str) -> list[str]:
    """Describe how this project's pattern list differs from the site's. Empty means in sync."""
    site = parse_typescript_patterns(ts_source)
    ours = banned_patterns()
    if site == ours:
        return []
    problems = [
        f"site has a pattern this project lacks: /{p.source}/{p.flags}"
        for p in site
        if p not in ours
    ]
    problems += [
        f"this project has a pattern the site lacks: /{p.source}/{p.flags}"
        for p in ours
        if p not in site
    ]
    return problems or ["same patterns in a different order"]
