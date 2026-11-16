"""Export step 1: take a private, verified copy of the site's database.

SQLite never opens a file in the site's data directory. The source is copied byte for byte into
a scratch directory first, and every later step works on that copy. That keeps the export from
ever creating lock or journal files beside the site's database.
"""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
import time
from pathlib import Path
from urllib.parse import quote

SQLITE_MAGIC = b"SQLite format 3\x00"
# Header bytes 18-19 are the file-format write and read versions: 1 = rollback journal, 2 = WAL.
ROLLBACK_JOURNAL = b"\x01\x01"
WAL_JOURNAL = b"\x02\x02"

_POINTER_VALUE = re.compile(r"[A-Za-z0-9]{1,16}")


class SourceError(Exception):
    """The source could not be read safely right now. A later run may succeed."""


def open_immutable(path: Path) -> sqlite3.Connection:
    """Open a database file that nothing will write to. Takes no locks, creates no files."""
    return sqlite3.connect(f"file:{quote(str(path))}?mode=ro&immutable=1", uri=True)


def copy_local_file(source: Path, scratch: Path) -> Path:
    """Copy a database file that is not being written to, such as a developer's local copy."""
    wal = source.with_name(source.name + "-wal")
    if wal.exists() and wal.stat().st_size > 0:
        raise SourceError(
            f"{wal.name} is not empty, so recent changes are not in the main file yet. "
            "Stop whatever is writing to the database, or export from a backup copy."
        )
    copy = _copy_unchanged(source, scratch)
    verify_copy(copy, allow_wal_header=True)
    return copy


def copy_current_snapshot(
    pointer_file: Path,
    path_template: str,
    scratch: Path,
    *,
    attempts: int = 3,
    wait_seconds: float = 5.0,
) -> Path:
    """Copy the snapshot that a pointer file currently names.

    The site keeps more than one snapshot of its database and records which one is being
    served in a small pointer file. `path_template` turns the pointer's value into a path, for
    example "/site-data/site.{slot}.db". A snapshot is never modified once written,
    but the site may replace the one that is *not* being served at any moment, and may move
    the pointer. So the pointer is read before and after the copy, and the copy is retried if
    anything moved.
    """
    if "{slot}" not in path_template:
        raise ValueError("path template must contain {slot}")

    error: Exception = SourceError("no attempt made")
    for attempt in range(attempts):
        if attempt:
            time.sleep(wait_seconds)
        try:
            slot = _read_pointer(pointer_file)
            copy = _copy_unchanged(Path(path_template.format(slot=slot)), scratch)
            if _read_pointer(pointer_file) != slot:
                raise SourceError("the pointer moved while the snapshot was being copied")
            verify_copy(copy, allow_wal_header=False)
            return copy
        except (SourceError, OSError) as caught:
            error = caught
    raise SourceError(f"no stable snapshot after {attempts} attempts: {error}")


def verify_copy(path: Path, *, allow_wal_header: bool) -> None:
    """Check that a copied file is a whole, healthy SQLite database."""
    size = path.stat().st_size
    with path.open("rb") as file:
        header = file.read(100)

    if len(header) < 100 or not header.startswith(SQLITE_MAGIC):
        raise SourceError("not a SQLite database")

    journal = header[18:20]
    if journal != ROLLBACK_JOURNAL and not (allow_wal_header and journal == WAL_JOURNAL):
        raise SourceError(f"unexpected journal mode in header: {journal.hex()}")

    page_size = int.from_bytes(header[16:18], "big")
    if page_size == 1:  # the header's way of writing 65536
        page_size = 65536
    if size % page_size:
        raise SourceError(f"file is {size} bytes, not a whole number of {page_size}-byte pages")
    # The header also records the page count, but SQLite only trusts that field when the
    # "version-valid-for" number beside it matches the change counter. Do the same.
    page_count = int.from_bytes(header[28:32], "big")
    count_is_current = header[92:96] == header[24:28] and page_count > 0
    if count_is_current and page_size * page_count != size:
        raise SourceError(f"file is {size} bytes but its header describes {page_size * page_count}")

    db = open_immutable(path)
    try:
        result = db.execute("PRAGMA integrity_check").fetchall()
    except sqlite3.DatabaseError as error:
        raise SourceError(f"integrity check could not run: {error}") from error
    finally:
        db.close()
    if result != [("ok",)]:
        raise SourceError(f"integrity check failed: {result[:3]}")


def _read_pointer(pointer_file: Path) -> str:
    # The site rewrites this file in place, so a reader can catch it empty for an instant.
    value = pointer_file.read_text(encoding="utf-8").strip()
    if not _POINTER_VALUE.fullmatch(value):
        raise SourceError("the pointer file does not hold a snapshot name right now")
    return value


def _copy_unchanged(source: Path, scratch: Path) -> Path:
    """Copy `source` into `scratch`, failing if the file changed while it was being read."""
    destination = scratch / "source.db"
    with source.open("rb") as reader:
        before = os.fstat(reader.fileno())
        with destination.open("wb") as writer:
            shutil.copyfileobj(reader, writer)
        after = os.fstat(reader.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise SourceError("the source file changed while it was being copied")
    return destination
