"""Command line for the export: `python -m alltouchdowns_mcp.export --help`."""

from __future__ import annotations

import argparse
import fcntl
import shutil
import signal
import sqlite3
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from alltouchdowns_mcp import legal
from alltouchdowns_mcp.export import source, validate
from alltouchdowns_mcp.export.build import HOLD_BACK_DAYS, BuildResult, build

EXIT_OK = 0
EXIT_LEGAL = 2  # a banned term, or the pattern list no longer matches the site's
EXIT_INVALID = 3  # the result or the set-up is wrong; needs a person to look
EXIT_SOURCE = 75  # the source was not readable right now; a later run may succeed

# Game dates are recorded in US Eastern time, so "the last 7 days" is counted there too.
EASTERN = ZoneInfo("America/New_York")


class ExportFailure(Exception):
    """Stops the export. Nothing is published."""

    def __init__(self, code: int, problems: list[str]) -> None:
        super().__init__(problems[0])
        self.code = code
        self.problems = problems


def main(argv: Sequence[str] | None = None, *, today: date | None = None) -> int:
    args = _parse_args(argv)
    try:
        _refuse_to_write_beside_the_source(args)
        with (
            _terminate_cleanly(),
            _single_run(args.out),
            _scratch_directory(args.scratch) as scratch,
        ):
            result = _export(args, scratch, today or datetime.now(EASTERN).date())
    except ExportFailure as failure:
        return _fail(failure.code, failure.problems)
    except source.SourceError as error:
        return _fail(EXIT_SOURCE, [str(error)])
    except (OSError, sqlite3.Error) as error:
        return _fail(EXIT_INVALID, [f"{type(error).__name__}: {error}"])

    _log(
        f"done: {result.games} games, {result.touchdowns} touchdowns, "
        f"{result.players} players, seasons {_span(result.seasons)}, "
        f"held_back={result.held_back}, incomplete={result.incomplete}"
    )
    return EXIT_OK


def _export(args: argparse.Namespace, scratch: Path, today: date) -> BuildResult:
    out: Path = args.out
    candidate = out.with_name(out.name + ".tmp")
    try:
        if args.site_patterns:
            drift = legal.drift_from_site(args.site_patterns.read_text(encoding="utf-8"))
            if drift:
                raise ExportFailure(
                    EXIT_LEGAL, ["banned-pattern list differs from the site's", *drift]
                )

        if args.source_file:
            copy = source.copy_local_file(args.source_file, scratch)
        else:
            copy = source.copy_current_snapshot(args.pointer_file, args.source_template, scratch)

        result = build(copy, candidate, today=today, seasons=args.seasons)

        if problems := validate.check_legal(candidate):
            raise ExportFailure(EXIT_LEGAL, problems)

        problems = validate.check_structure(candidate, only_seasons=args.seasons)
        if out.exists() and args.seasons is None and not args.accept_restatement:
            problems += _compare_with_served(candidate, out)
        if problems:
            raise ExportFailure(EXIT_INVALID, problems)

        validate.publish(candidate, out)
        if result.held_back:
            _log(
                f"note: {result.held_back} finished games were left out. A game appears once "
                f"its nflverse touchdowns have arrived, or after {HOLD_BACK_DAYS} days."
            )
        return result
    finally:
        candidate.unlink(missing_ok=True)


def _compare_with_served(candidate: Path, served: Path) -> list[str]:
    try:
        return validate.check_against_previous(candidate, served)
    except sqlite3.Error as error:
        return [
            f"the database currently at {served.name} could not be read ({error}). "
            "If it is safe to replace, run again with --accept-restatement"
        ]


# --- Safety around the run -------------------------------------------------------------------


def _refuse_to_write_beside_the_source(args: argparse.Namespace) -> None:
    """This export must never create a file in the site's data directory.

    On the server that directory is mounted read-only, but a mistyped --out on a developer's
    machine would otherwise replace the site's own database with the slim one.
    """
    if args.source_file:
        protected = [args.source_file.parent]
    else:
        protected = [args.pointer_file.parent, Path(args.source_template).parent]
    protected = [directory.resolve() for directory in protected]

    writes = {"--out": args.out.resolve().parent}
    if args.scratch:
        writes["--scratch"] = args.scratch.resolve()
    for option, directory in writes.items():
        if any(directory.is_relative_to(source_dir) for source_dir in protected):
            raise ExportFailure(
                EXIT_INVALID, [f"{option} points into the directory the source is read from"]
            )


@contextmanager
def _terminate_cleanly() -> Iterator[None]:
    """Turn a termination signal into a normal exit, so the scratch copy is always deleted.

    By default Python dies on SIGTERM without running `finally` blocks, which would leave the
    full site database in the scratch directory.
    """

    def stop(signum: int, frame: object) -> None:
        raise SystemExit(128 + signum)

    previous = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGHUP)}
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


@contextmanager
def _single_run(out: Path) -> Iterator[None]:
    """Hold a lock beside the output file, so two exports cannot overwrite each other's work."""
    with open(out.with_name(out.name + ".lock"), "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise source.SourceError("another export is already running") from None
        yield


@contextmanager
def _scratch_directory(parent: Path | None) -> Iterator[Path]:
    """A private directory for the copy of the site database.

    That copy holds everything, play descriptions included, so it must never outlive the run.
    """
    scratch = Path(tempfile.mkdtemp(prefix="atd-export-", dir=parent))
    try:
        yield scratch
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# --- Arguments and logging -------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m alltouchdowns_mcp.export",
        description="Build the slim public database from a copy of the site database.",
    )
    where = parser.add_argument_group("source (choose one)")
    where.add_argument("--source-file", type=Path, help="a database file nothing is writing to")
    where.add_argument("--pointer-file", type=Path, help="file naming the current snapshot")
    where.add_argument(
        "--source-template", help="snapshot path with {slot} where the pointer's value goes"
    )
    parser.add_argument("--out", type=Path, required=True, help="where to publish the result")
    parser.add_argument("--seasons", type=_season_list, help="only these seasons, e.g. 2007,2019")
    parser.add_argument("--scratch", type=Path, help="directory for the temporary source copy")
    parser.add_argument(
        "--site-patterns", type=Path, help="the site's pattern source file, to check for drift"
    )
    parser.add_argument(
        "--accept-restatement",
        action="store_true",
        help="publish even if counts shrank, or the current database cannot be read",
    )
    args = parser.parse_args(argv)

    snapshot_mode = bool(args.pointer_file and args.source_template)
    if bool(args.source_file) == snapshot_mode:
        parser.error("give either --source-file, or --pointer-file with --source-template")
    return args


def _season_list(text: str) -> list[int]:
    return [int(part) for part in text.split(",")]


def _span(seasons: tuple[int, ...]) -> str:
    """'1999-2026' for consecutive seasons, otherwise '2007,2019'."""
    if len(seasons) == seasons[-1] - seasons[0] + 1:
        return f"{seasons[0]}-{seasons[-1]}"
    return ",".join(map(str, seasons))


def _fail(code: int, problems: list[str]) -> int:
    for problem in problems:
        _log(f"FAILED: {problem}")
    _log(f"FAILED: nothing was published (exit {code})")
    return code


def _log(message: str) -> None:
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[mcp-export {stamp}] {message}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
