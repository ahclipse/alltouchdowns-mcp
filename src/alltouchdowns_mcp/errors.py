"""Errors a tool can return to the client.

Every message is a fixed sentence written here. A message never repeats what the caller sent,
because the caller's text may contain a term this project must not emit. Where a message needs
detail (a list of valid abbreviations, the seasons covered), the detail comes from the
database, which has already passed the legal scan.
"""

from __future__ import annotations

from typing import Literal

from mcp.server.mcpserver.exceptions import ToolError

ErrorCode = Literal[
    "invalid_argument",
    "not_covered",
    "unknown_team",
    "ambiguous_team",
    "unknown_player",
    "not_found",
    "bad_cursor",
    "too_broad",
    "busy",
    "internal_error",
]

_MESSAGES: dict[ErrorCode, str] = {
    "invalid_argument": "An argument is missing or not allowed.",
    "not_covered": "That is outside what this server covers.",
    "unknown_team": "No team matches. Use a city or an abbreviation, not a nickname.",
    "ambiguous_team": "More than one team plays in that city. Use the abbreviation.",
    "unknown_player": "No player has that slug. Call find_players to look the slug up.",
    "not_found": "No finished game matches.",
    "bad_cursor": "The cursor is not valid for this search. Start again without a cursor.",
    "too_broad": "That search covers too much to answer. Add a filter to narrow it.",
    "busy": "The server is busy. Try again in a moment.",
    "internal_error": "The response could not be produced.",
}


class ToolRefusal(ToolError):
    """Raised by a tool to answer with an error the client model can act on.

    It subclasses the SDK's ToolError, so the SDK reports it as a normal tool error result
    instead of logging a crash.
    """

    def __init__(self, code: ErrorCode, detail: str | None = None) -> None:
        self.code: ErrorCode = code
        message = _MESSAGES[code] if detail is None else f"{_MESSAGES[code]} {detail}"
        super().__init__(f"{code}: {message}")
