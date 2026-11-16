"""Team lookup: turning what a caller typed into a franchise, and abbreviations into labels.

A franchise keeps one id for its whole history (the abbreviation it uses today), while each
game records the abbreviation of its own era. So the San Diego years and the Los Angeles years
of one franchise are found together and labelled separately.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from alltouchdowns_mcp.errors import ToolRefusal
from alltouchdowns_mcp.models import ResolvedTeam, Team


@dataclass(frozen=True)
class _Row:
    abbr: str
    franchise: str
    city: str
    label: str
    first_season: int


class Teams:
    def __init__(self, rows: list[_Row]) -> None:
        self._rows = sorted(rows, key=lambda row: (row.first_season, row.abbr))
        self._by_abbr = {row.abbr: row for row in rows}

    @classmethod
    def load(cls, conn: sqlite3.Connection) -> Teams:
        query = "SELECT abbr, franchise, city, label, first_season FROM teams"
        return cls([_Row(*row) for row in conn.execute(query)])

    def team(self, abbr: str) -> Team:
        """The abbreviation and era city to show for a team."""
        return Team(abbr=abbr, name=self._by_abbr[abbr].label)

    def label(self, abbr: str) -> str:
        return self._by_abbr[abbr].label

    def resolve(self, text: str) -> str:
        """Return the franchise a caller means by an abbreviation or a city."""
        wanted = _fold(text)
        for field in ("abbr", "label"):
            franchises = {r.franchise for r in self._rows if _fold(getattr(r, field)) == wanted}
            if len(franchises) == 1:
                return franchises.pop()

        franchises = {row.franchise for row in self._rows if _fold(row.city) == wanted}
        if len(franchises) == 1:
            return franchises.pop()
        if len(franchises) > 1:
            options = " or ".join(sorted(self.label(franchise) for franchise in franchises))
            raise ToolRefusal("ambiguous_team", f"Choose {options}.")

        valid = ", ".join(sorted({row.franchise for row in self._rows}))
        raise ToolRefusal("unknown_team", f"Valid abbreviations: {valid}.")

    def describe(self, franchise: str) -> ResolvedTeam:
        """Every city a franchise has played in, oldest first."""
        names: dict[str, str] = {}  # city -> label; a second abbreviation for a city is skipped
        for row in self._rows:
            if row.franchise == franchise:
                names.setdefault(row.city, row.label)
        return ResolvedTeam(franchise=franchise, names=list(names.values()))


def _fold(text: str) -> str:
    """Compare loosely: case, surrounding space and periods ("St Louis") do not matter."""
    return " ".join(text.replace(".", "").casefold().split())
