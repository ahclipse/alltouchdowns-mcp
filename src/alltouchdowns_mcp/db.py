"""Read-only access to the slim database.

The database is opened fresh for every tool call. Opening takes microseconds, and it means the
daily export can swap in a new file at any moment without the server noticing: a call in flight
keeps reading the file it opened, and the next call opens the new one.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from alltouchdowns_mcp import ATTRIBUTION
from alltouchdowns_mcp.errors import ToolRefusal
from alltouchdowns_mcp.models import Source
from alltouchdowns_mcp.teams import Teams

# Tool calls run on worker threads. Letting them all query at once does not make any of them
# faster (Python runs one thread at a time), it only makes each one slower until they all hit
# the deadline together. So a few run and the rest wait their turn.
_QUERY_SLOTS = threading.BoundedSemaphore(4)
_MAX_WAIT_FOR_SLOT_SECONDS = 5.0


@dataclass(frozen=True)
class Coverage:
    """How much data is loaded. The test fixture holds two seasons; production holds all."""

    seasons: tuple[int, ...]
    data_through: str
    exported_at: str  # also serves as the version of the data
    season_in_progress: bool

    @classmethod
    def load(cls, conn: sqlite3.Connection) -> Coverage:
        seasons = conn.execute("SELECT season, complete FROM seasons ORDER BY season").fetchall()
        meta = dict(conn.execute("SELECT key, value FROM meta"))
        return cls(
            seasons=tuple(season for season, _ in seasons),
            data_through=meta["data_through"],
            exported_at=meta["exported_at"],
            season_in_progress=bool(seasons) and not seasons[-1][1],
        )

    def describe(self) -> str:
        """'1999 to 2026' when the seasons are consecutive, otherwise a list."""
        first, last = self.seasons[0], self.seasons[-1]
        if len(self.seasons) > 2 and len(self.seasons) == last - first + 1:
            return f"{first} to {last}"
        return " and ".join(str(season) for season in self.seasons)

    def require(self, season: int) -> None:
        """Refuse a season that is not loaded, rather than answer with an empty result."""
        if season not in self.seasons:
            raise ToolRefusal("not_covered", f"Seasons covered: {self.describe()}.")

    def source(self) -> Source:
        return Source(
            attribution=ATTRIBUTION,
            seasons=self.describe(),
            data_through=self.data_through,
            season_in_progress=self.season_in_progress,
        )


@dataclass(frozen=True)
class Database:
    """Everything a tool needs for one call."""

    conn: sqlite3.Connection
    teams: Teams
    coverage: Coverage


def _connect(path: Path) -> sqlite3.Connection:
    # immutable=1: the file is only ever replaced whole, never modified, so SQLite can skip
    # locking and never creates journal files beside it. mode=ro refuses any write.
    return sqlite3.connect(f"file:{quote(str(path))}?mode=ro&immutable=1", uri=True)


def read_meta(path: Path) -> dict[str, str]:
    """What the export recorded about itself (when it ran, how far the data goes).

    For the health check. It is one tiny read, so it does not wait for a query slot: the
    health check has to answer even when every slot is busy.
    """
    conn = _connect(path)
    try:
        return dict(conn.execute("SELECT key, value FROM meta"))
    finally:
        conn.close()


@contextmanager
def open_database(path: Path, timeout_ms: int) -> Iterator[Database]:
    """Open the database for one tool call, with a deadline on the queries it runs."""
    if not _QUERY_SLOTS.acquire(timeout=_MAX_WAIT_FOR_SLOT_SECONDS):
        raise ToolRefusal("busy")
    try:
        conn = _connect(path)
        try:
            # The clock starts here, after the wait for a slot, so a queue is not mistaken
            # for a slow query. SQLite calls the handler every 1,000 steps and abandons the
            # query when it returns true.
            deadline = time.monotonic() + timeout_ms / 1000
            conn.set_progress_handler(lambda: time.monotonic() > deadline, 1000)
            yield Database(conn, Teams.load(conn), Coverage.load(conn))
        except sqlite3.OperationalError as error:
            if "interrupted" in str(error):
                raise ToolRefusal("timeout") from error
            raise
        finally:
            conn.close()
    finally:
        _QUERY_SLOTS.release()
