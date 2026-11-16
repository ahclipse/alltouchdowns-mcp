"""Copying and verifying the source database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from alltouchdowns_mcp.export import source
from alltouchdowns_mcp.export.source import SourceError


def make_db(path: Path, *, wal: bool = False) -> Path:
    db = sqlite3.connect(path)
    if wal:
        db.execute("PRAGMA journal_mode = WAL")
    db.execute("CREATE TABLE t (x)")
    db.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(500)])
    db.commit()
    db.close()
    return path


@pytest.fixture
def scratch(tmp_path: Path) -> Path:
    directory = tmp_path / "scratch"
    directory.mkdir()
    return directory


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(source.time, "sleep", lambda seconds: None)


# --- A local file ----------------------------------------------------------------------------


def test_local_copy_leaves_the_source_directory_untouched(tmp_path: Path, scratch: Path) -> None:
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    original = make_db(site_dir / "site.db", wal=True)
    before = sorted(p.name for p in site_dir.iterdir())

    copy = source.copy_local_file(original, scratch)

    assert copy.parent == scratch
    assert copy.read_bytes() == original.read_bytes()
    assert sorted(p.name for p in site_dir.iterdir()) == before


def test_local_copy_refuses_a_database_with_unsaved_changes(tmp_path: Path, scratch: Path) -> None:
    original = make_db(tmp_path / "site.db")
    (tmp_path / "site.db-wal").write_bytes(b"pending changes")

    with pytest.raises(SourceError, match="not empty"):
        source.copy_local_file(original, scratch)


# --- Verification ----------------------------------------------------------------------------


def test_verify_rejects_a_file_missing_whole_pages(tmp_path: Path) -> None:
    path = make_db(tmp_path / "site.db")
    path.write_bytes(path.read_bytes()[:-4096])

    with pytest.raises(SourceError, match="header describes"):
        source.verify_copy(path, allow_wal_header=False)


def test_verify_rejects_a_file_cut_off_mid_page(tmp_path: Path) -> None:
    path = make_db(tmp_path / "site.db")
    path.write_bytes(path.read_bytes()[:-1024])

    with pytest.raises(SourceError, match="not a whole number"):
        source.verify_copy(path, allow_wal_header=False)


def test_verify_ignores_a_page_count_sqlite_itself_would_not_trust(tmp_path: Path) -> None:
    """An older tool can leave the header's page count stale. SQLite spots that by comparing
    two counters in the header and then goes by the file size, so the file is healthy."""
    path = make_db(tmp_path / "site.db")
    data = bytearray(path.read_bytes())
    data[28:32] = (0).to_bytes(4, "big")  # page count
    data[92:96] = (int.from_bytes(data[24:28], "big") - 1).to_bytes(4, "big")  # now out of step
    path.write_bytes(bytes(data))

    source.verify_copy(path, allow_wal_header=False)


def test_verify_handles_the_largest_page_size(tmp_path: Path) -> None:
    path = tmp_path / "site.db"
    db = sqlite3.connect(path)
    db.execute("PRAGMA page_size = 65536")  # stored in the header as 1
    db.execute("CREATE TABLE t (x)")
    db.commit()
    db.close()

    source.verify_copy(path, allow_wal_header=False)


def test_verify_rejects_a_file_too_short_to_have_a_header(tmp_path: Path) -> None:
    path = tmp_path / "site.db"
    path.write_bytes(source.SQLITE_MAGIC)

    with pytest.raises(SourceError, match="not a SQLite database"):
        source.verify_copy(path, allow_wal_header=True)


def test_copy_fails_if_the_source_changes_while_being_read(
    tmp_path: Path, scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = make_db(tmp_path / "site.db")
    real_copy = source.shutil.copyfileobj

    def copy_then_grow(reader, writer) -> None:
        real_copy(reader, writer)
        with original.open("ab") as file:  # something wrote to the source mid-copy
            file.write(b"\0" * 4096)

    monkeypatch.setattr(source.shutil, "copyfileobj", copy_then_grow)

    with pytest.raises(SourceError, match="changed while it was being copied"):
        source.copy_local_file(original, scratch)


def test_verify_rejects_a_file_that_is_not_sqlite(tmp_path: Path) -> None:
    path = tmp_path / "site.db"
    path.write_bytes(b"x" * 4096)

    with pytest.raises(SourceError, match="not a SQLite database"):
        source.verify_copy(path, allow_wal_header=True)


def test_verify_rejects_a_live_wal_database_where_a_snapshot_is_expected(tmp_path: Path) -> None:
    path = make_db(tmp_path / "site.db", wal=True)

    with pytest.raises(SourceError, match="journal mode"):
        source.verify_copy(path, allow_wal_header=False)
    source.verify_copy(path, allow_wal_header=True)  # fine for a local file


def test_verify_rejects_a_corrupt_page(tmp_path: Path) -> None:
    path = make_db(tmp_path / "site.db")
    data = bytearray(path.read_bytes())
    data[4096:8192] = b"\xff" * 4096  # overwrite the second page
    path.write_bytes(bytes(data))

    with pytest.raises(SourceError, match="integrity check"):
        source.verify_copy(path, allow_wal_header=False)


# --- Snapshots chosen by a pointer file ------------------------------------------------------


@pytest.fixture
def snapshots(tmp_path: Path) -> tuple[Path, str]:
    directory = tmp_path / "snapshots"
    directory.mkdir()
    make_db(directory / "site.a.db")
    make_db(directory / "site.b.db")
    return tmp_path / "current", str(directory / "site.{slot}.db")


def test_snapshot_copy_follows_the_pointer(snapshots: tuple[Path, str], scratch: Path) -> None:
    pointer, template = snapshots
    pointer.write_text("b\n")

    copy = source.copy_current_snapshot(pointer, template, scratch)

    assert copy.read_bytes() == Path(template.format(slot="b")).read_bytes()


def test_snapshot_copy_retries_when_the_pointer_moves(
    snapshots: tuple[Path, str], scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pointer, template = snapshots
    # First attempt: reads "a", then finds "b" after copying. Second attempt: stable on "b".
    readings = iter(["a", "b", "b", "b"])
    monkeypatch.setattr(source, "_read_pointer", lambda path: next(readings))

    copy = source.copy_current_snapshot(pointer, template, scratch)

    assert copy.read_bytes() == Path(template.format(slot="b")).read_bytes()
    assert next(readings, None) is None  # both attempts read the pointer before and after


def test_snapshot_copy_gives_up_on_an_unreadable_pointer(
    snapshots: tuple[Path, str], scratch: Path
) -> None:
    pointer, template = snapshots
    pointer.write_text("")  # caught mid-rewrite, or simply wrong

    with pytest.raises(SourceError, match="no stable snapshot after 3 attempts"):
        source.copy_current_snapshot(pointer, template, scratch)


def test_snapshot_copy_gives_up_when_the_snapshot_is_missing(
    snapshots: tuple[Path, str], scratch: Path
) -> None:
    pointer, template = snapshots
    pointer.write_text("c")

    with pytest.raises(SourceError, match="no stable snapshot"):
        source.copy_current_snapshot(pointer, template, scratch)


def test_pointer_value_cannot_escape_the_template(
    snapshots: tuple[Path, str], scratch: Path
) -> None:
    pointer, template = snapshots
    pointer.write_text("../../etc/passwd")

    with pytest.raises(SourceError):
        source.copy_current_snapshot(pointer, template, scratch)


def test_template_must_name_the_slot(snapshots: tuple[Path, str], scratch: Path) -> None:
    pointer, _ = snapshots
    with pytest.raises(ValueError):
        source.copy_current_snapshot(pointer, "/somewhere/site.db", scratch)
