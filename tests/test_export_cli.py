"""The export end to end, through its command line."""

from __future__ import annotations

import fcntl
import signal
from pathlib import Path

import pytest

from alltouchdowns_mcp.export import __main__ as export_main
from alltouchdowns_mcp.export import source, validate
from alltouchdowns_mcp.export.__main__ import (
    EXIT_INVALID,
    EXIT_LEGAL,
    EXIT_OK,
    EXIT_SOURCE,
    main,
)
from alltouchdowns_mcp.export.build import build
from tests.conftest import TODAY, SiteDb, rows


@pytest.fixture
def scratch(tmp_path: Path) -> Path:
    directory = tmp_path / "scratch"
    directory.mkdir()
    return directory


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(source.time, "sleep", lambda seconds: None)


def site_with_one_game(site: SiteDb, scorer: str = "Jamaal Example") -> Path:
    site.touchdown(site.game(2025, 1, "KC", "DEN"), site.player(scorer))
    return site.save()


def run(*args: object) -> int:
    return main([str(a) for a in args], today=TODAY)


def test_successful_export_publishes_and_cleans_up(
    site: SiteDb, tmp_path: Path, scratch: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "mcp.db"

    code = run(
        "--source-file",
        site_with_one_game(site),
        "--out",
        out,
        "--seasons",
        "2025",
        "--scratch",
        scratch,
    )

    assert code == EXIT_OK
    assert rows(out, "SELECT COUNT(*) FROM touchdowns") == [(1,)]
    assert list(scratch.iterdir()) == []  # the full site copy is gone
    assert not (tmp_path / "mcp.db.tmp").exists()
    assert "done: 1 games, 1 touchdowns, 1 players" in capsys.readouterr().out


def test_banned_name_fails_the_export_and_keeps_the_served_file(
    site: SiteDb, tmp_path: Path, scratch: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "mcp.db"
    out.write_text("yesterday's database")

    code = run(
        "--source-file",
        site_with_one_game(site, scorer="Raven Greene"),
        "--out",
        out,
        "--seasons",
        "2025",
        "--scratch",
        scratch,
    )

    assert code == EXIT_LEGAL
    assert out.read_text() == "yesterday's database"
    assert list(scratch.iterdir()) == []
    assert not (tmp_path / "mcp.db.tmp").exists()
    assert "FAILED: nothing was published" in capsys.readouterr().out


def test_implausible_result_is_not_published(site: SiteDb, tmp_path: Path, scratch: Path) -> None:
    out = tmp_path / "mcp.db"

    # Without --seasons this is a full export, and one game is nowhere near a full history.
    code = run("--source-file", site_with_one_game(site), "--out", out, "--scratch", scratch)

    assert code == EXIT_INVALID
    assert not out.exists()


def test_unreadable_source_is_a_retryable_failure(
    site: SiteDb, tmp_path: Path, scratch: Path
) -> None:
    path = site_with_one_game(site)
    path.with_name("site.db-wal").write_bytes(b"pending changes")

    code = run("--source-file", path, "--out", tmp_path / "mcp.db", "--scratch", scratch)

    assert code == EXIT_SOURCE
    assert list(scratch.iterdir()) == []


def test_pattern_drift_stops_the_export_before_anything_is_read(
    site: SiteDb, tmp_path: Path, scratch: Path
) -> None:
    site_patterns = tmp_path / "patterns.ts"
    site_patterns.write_text("const BANNED_PATTERNS: RegExp[] = [\n  /\\bNewTerm\\b/,\n];\n")

    code = run(
        "--source-file",
        site_with_one_game(site),
        "--out",
        tmp_path / "mcp.db",
        "--seasons",
        "2025",
        "--scratch",
        scratch,
        "--site-patterns",
        site_patterns,
    )

    assert code == EXIT_LEGAL
    assert not (tmp_path / "mcp.db").exists()


def test_snapshot_mode_reads_the_file_the_pointer_names(
    site: SiteDb, tmp_path: Path, scratch: Path
) -> None:
    snapshot = site_with_one_game(site)
    snapshot.rename(snapshot.with_name("site.b.db"))
    pointer = snapshot.with_name("current")
    pointer.write_text("b")
    out = tmp_path / "mcp.db"

    code = run(
        "--pointer-file", pointer,
        "--source-template", snapshot.with_name("site.{slot}.db"),
        "--out", out, "--seasons", "2025", "--scratch", scratch,
    )  # fmt: skip

    assert code == EXIT_OK
    assert rows(out, "SELECT COUNT(*) FROM games") == [(1,)]


@pytest.mark.parametrize(
    "args",
    [
        ["--out", "x.db"],
        ["--out", "x.db", "--pointer-file", "p"],
        [
            "--out",
            "x.db",
            "--source-file",
            "a.db",
            "--pointer-file",
            "p",
            "--source-template",
            "t{slot}",
        ],
    ],
)
def test_exactly_one_source_must_be_given(args: list[str]) -> None:
    with pytest.raises(SystemExit):
        main(args)


# --- The site's directory is never written to ------------------------------------------------


def test_output_beside_the_source_is_refused(site: SiteDb, scratch: Path) -> None:
    path = site_with_one_game(site)
    before = path.read_bytes()

    code = run("--source-file", path, "--out", path.with_name("mcp.db"),
               "--seasons", "2025", "--scratch", scratch)  # fmt: skip

    assert code == EXIT_INVALID
    assert sorted(p.name for p in path.parent.iterdir()) == ["site.db"]
    assert path.read_bytes() == before


def test_output_over_the_source_itself_is_refused(site: SiteDb, scratch: Path) -> None:
    path = site_with_one_game(site)
    before = path.read_bytes()

    code = run("--source-file", path, "--out", path, "--seasons", "2025", "--scratch", scratch)

    assert code == EXIT_INVALID
    assert path.read_bytes() == before


def test_scratch_inside_the_source_directory_is_refused(site: SiteDb, tmp_path: Path) -> None:
    path = site_with_one_game(site)

    code = run("--source-file", path, "--out", tmp_path / "mcp.db",
               "--seasons", "2025", "--scratch", path.parent)  # fmt: skip

    assert code == EXIT_INVALID
    assert sorted(p.name for p in path.parent.iterdir()) == ["site.db"]


# --- Comparison with the database being served -----------------------------------------------


@pytest.fixture
def any_size_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let a tiny database through the size checks, so a full (not --seasons) export of the
    synthetic data reaches the comparison with the served file."""
    monkeypatch.setattr(validate, "check_structure", lambda path, only_seasons: [])


def served_database_with_more_games(site: SiteDb, tmp_path: Path) -> Path:
    player = site.player("Jamaal Example")
    for week in range(1, 6):
        site.touchdown(site.game(2025, week, "KC", "DEN"), player)
    served = tmp_path / "mcp.db"
    build(site.save(), served, today=TODAY)
    return served


def test_shrunken_result_is_not_published_unless_accepted(
    site: SiteDb, tmp_path: Path, scratch: Path, any_size_passes: None
) -> None:
    served = served_database_with_more_games(site, tmp_path)
    smaller_source = SiteDb(tmp_path / "site" / "later.db")
    smaller_source.touchdown(
        smaller_source.game(2025, 1, "KC", "DEN"), smaller_source.player("Jamaal Example")
    )
    args = ["--source-file", smaller_source.save(), "--out", served, "--scratch", scratch]

    assert run(*args) == EXIT_INVALID
    assert rows(served, "SELECT COUNT(*) FROM games") == [(5,)]

    assert run(*args, "--accept-restatement") == EXIT_OK
    assert rows(served, "SELECT COUNT(*) FROM games") == [(1,)]


def test_unreadable_served_file_is_reported_with_the_way_out(
    site: SiteDb, tmp_path: Path, scratch: Path, any_size_passes: None,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    served = tmp_path / "mcp.db"
    served.write_text("not a database")
    args = ["--source-file", site_with_one_game(site), "--out", served, "--scratch", scratch]

    assert run(*args) == EXIT_INVALID
    assert "--accept-restatement" in capsys.readouterr().out
    assert served.read_text() == "not a database"

    assert run(*args, "--accept-restatement") == EXIT_OK
    assert rows(served, "SELECT COUNT(*) FROM games") == [(1,)]


# --- Failures always end in a FAILED line and a documented exit code -------------------------


@pytest.mark.parametrize("missing", ["out_directory", "source_file", "site_patterns"])
def test_missing_paths_fail_cleanly(
    missing: str, site: SiteDb, tmp_path: Path, scratch: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = site_with_one_game(site)
    args = {
        "out_directory": ["--source-file", path, "--out", tmp_path / "nowhere" / "mcp.db"],
        "source_file": ["--source-file", path.with_name("gone.db"), "--out", tmp_path / "m.db"],
        "site_patterns": ["--source-file", path, "--out", tmp_path / "m.db",
                          "--site-patterns", tmp_path / "gone.ts"],
    }[missing]  # fmt: skip

    code = run(*args, "--seasons", "2025", "--scratch", scratch)

    assert code == EXIT_INVALID
    assert "FAILED: nothing was published (exit 3)" in capsys.readouterr().out
    assert list(scratch.iterdir()) == []


def test_second_export_at_the_same_time_backs_off(
    site: SiteDb, tmp_path: Path, scratch: Path
) -> None:
    out = tmp_path / "mcp.db"
    with open(tmp_path / "mcp.db.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # another export is running

        code = run("--source-file", site_with_one_game(site), "--out", out,
                   "--seasons", "2025", "--scratch", scratch)  # fmt: skip

    assert code == EXIT_SOURCE
    assert not out.exists()


def test_termination_signal_still_deletes_the_scratch_copy(
    site: SiteDb, tmp_path: Path, scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handler_before = signal.getsignal(signal.SIGTERM)

    def killed_mid_build(*args: object, **kwargs: object) -> None:
        assert any(scratch.iterdir())  # the full site copy exists at this point
        signal.raise_signal(signal.SIGTERM)

    monkeypatch.setattr(export_main, "build", killed_mid_build)

    with pytest.raises(SystemExit) as exit_info:
        run("--source-file", site_with_one_game(site), "--out", tmp_path / "mcp.db",
            "--seasons", "2025", "--scratch", scratch)  # fmt: skip

    assert exit_info.value.code == 128 + signal.SIGTERM
    assert list(scratch.iterdir()) == []
    assert signal.getsignal(signal.SIGTERM) is handler_before
