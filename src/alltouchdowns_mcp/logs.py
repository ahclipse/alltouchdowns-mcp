"""Logging: what this server writes down, and what it never does.

What is logged: one JSON line per tool call (which tool, whether the caller was signed in, how
it ended, how long it took), and a count of requests refused by each rate limit. What is never
logged: IP addresses, tokens, user ids, or anything a caller typed. Arguments and headers are
free text, and a log is a place data tends to be forgotten in.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import UTC, datetime
from typing import Any

_REQUESTS = logging.getLogger("alltouchdowns.requests")

SECONDS_BETWEEN_REFUSAL_LINES = 60


def configure() -> None:
    """Send request lines to standard output, where the container runtime collects them."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _REQUESTS.addHandler(handler)
    _REQUESTS.setLevel(logging.INFO)
    _REQUESTS.propagate = False

    # The SDK logs a refused Host or Origin header word for word, and a header is caller-written
    # text. Those requests are already answered with 421 or 403; there is nothing to act on.
    logging.getLogger("mcp.server.transport_security").setLevel(logging.ERROR)


def record(event: str, **fields: Any) -> None:
    line = {"at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "event": event, **fields}
    _REQUESTS.info(json.dumps(line, separators=(",", ":")))


class RefusalCount:
    """Reports how many requests a rate limit has refused, in at most one line a minute.

    One line per refused request would let a flood of requests become a flood of log lines.
    Each line carries the number refused since the previous line.
    """

    def __init__(self, limit: str) -> None:
        self.limit = limit
        self._refused = 0
        self._last_line_at: float | None = None

    def add(self) -> None:
        self._refused += 1
        now = time.monotonic()
        due = (
            self._last_line_at is None or now - self._last_line_at >= SECONDS_BETWEEN_REFUSAL_LINES
        )
        if due:
            record("rate_limited", limit=self.limit, refused=self._refused)
            self._refused, self._last_line_at = 0, now
