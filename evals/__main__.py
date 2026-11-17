"""Command line for the eval suite: `python -m evals --help`.

preflight   free checks on the cases and grader (run on every pull request)
selftest    free check of the whole harness, using scripted stand-ins for the model
sign-in     free check that the API accepts our credentials and offers the models we use
run         a real run against a model. This one costs money.
report      rebuild the summary of a finished run
share       copy a run without the model's own words (what CI uploads)
baseline    set the pass mark from several runs of the current code
gate        compare a run with the baseline
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import anthropic
import anyio

from evals import preflight, report, runner, scripted
from evals import truth as truths
from evals.schema import CASES_DIR, load_cases

ROOT = Path(__file__).parent
FIXTURE = ROOT.parent / "tests" / "fixtures" / "seasons-2007-2019.db"
BASELINES = ROOT / "baselines"  # one file per model: baselines/claude-opus-5-5.json


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cases", type=Path, default=CASES_DIR, help="directory of case files")
    parser.add_argument("--db", type=Path, default=FIXTURE, help="database to evaluate against")
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("preflight", help="free checks on the cases and grader")
    check.add_argument("--only", help="only cases whose category or id starts with this")

    commands.add_parser("selftest", help="free check of the harness with scripted agents")

    commands.add_parser("sign-in", help="free check of the API credentials and models")

    run = commands.add_parser("run", help="run the suite against a model (costs money)")
    run.add_argument("--model", required=True, help="e.g. claude-opus-5-5")
    run.add_argument("--out", type=Path, required=True, help="directory for this run's output")
    run.add_argument("--effort", default="medium", help="model effort level (default: medium)")
    run.add_argument("--reps", type=int, default=3, help="trials per case (default: 3)")
    run.add_argument("--no-tools", action="store_true", help="baseline: no tools, guess")
    run.add_argument("--only", help="only cases whose category or id starts with this")
    run.add_argument("--split", choices=["dev", "heldout"])
    run.add_argument("--limit", type=int, help="first N cases only (for a pilot)")
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--max-cost-usd", type=float, default=5.0, help="stop before spending more")

    rebuild = commands.add_parser("report", help="rebuild summary.json and summary.md")
    rebuild.add_argument("run_dir", type=Path)

    share = commands.add_parser("share", help="copy a run without the model's own words")
    share.add_argument("run_dir", type=Path)
    share.add_argument("out_dir", type=Path)

    baseline = commands.add_parser("baseline", help="set the pass mark from several runs")
    baseline.add_argument("run_dirs", type=Path, nargs="+")
    baseline.add_argument("--out", type=Path, help="default: evals/baselines/<model>.json")

    gate = commands.add_parser("gate", help="compare a run with the baseline")
    gate.add_argument("run_dir", type=Path)
    gate.add_argument("--baseline", type=Path, help="default: evals/baselines/<model>.json")

    args = parser.parse_args(argv)
    return {
        "preflight": _preflight,
        "selftest": _selftest,
        "sign-in": _sign_in,
        "run": _run,
        "report": _report,
        "share": _share,
        "baseline": _baseline,
        "gate": _gate,
    }[args.command](args)


def _preflight(args: argparse.Namespace) -> int:
    conn = truths.open_database(args.db)
    try:
        cases = truths.cases_that_hold(conn, load_cases(args.cases))
        if args.only:
            cases = [
                c for c in cases if c.category.startswith(args.only) or c.id.startswith(args.only)
            ]
        print(preflight.describe(cases))
        franchises = truths.franchises(conn)
        problems = preflight.validate(cases, conn)
        if not problems:  # the later checks need every truth query to run
            problems += preflight.oracle_and_null(cases, conn, franchises)
            problems += anyio.run(preflight.reference, cases, args.db, conn, franchises)
    finally:
        conn.close()

    for problem in problems:
        print(f"FAILED: {problem}")
    print("pre-flight passed" if not problems else f"pre-flight found {len(problems)} problems")
    return 1 if problems else 0


def _selftest(args: argparse.Namespace) -> int:
    """The oracle must score 100 % and the null agent 0 %, through the real runner."""
    problems = []
    with tempfile.TemporaryDirectory() as scratch:
        for model, expected in ((scripted.ORACLE, 1.0), (scripted.NULL, 0.0)):
            out = Path(scratch) / model
            config = runner.RunConfig(
                model=model, out=out, cases_dir=args.cases, db=args.db, reps=1
            )
            code = runner.run(config)
            summary = json.loads((out / "summary.json").read_text())
            errors = report.read_jsonl(out / "errors.jsonl")
            if code != 0 or errors:
                kinds = sorted({f"{e['class']}: {e['detail'][:120]}" for e in errors})
                problems.append(
                    f"{model}: not a clean run, {len(errors)} errors: {'; '.join(kinds[:3])}"
                )
            if summary["correct_and_grounded"] != expected:
                rows = report.read_jsonl(out / "results.jsonl")
                off = [
                    _what_went_wrong(row)
                    for row in rows
                    if (row["correct"] and row["grounded"]) != bool(expected)
                ]
                problems.append(f"{model}: expected {expected:.0%}, cases off: {off[:6]}")
    for problem in problems:
        print(f"FAILED: {problem}")
    print("self-test passed" if not problems else "self-test FAILED")
    return 1 if problems else 0


def _what_went_wrong(row: dict) -> str:
    """One trial's outcome in a few words, for the self-test's failure report."""
    outcome = row["failure"] or ("right but not grounded" if not row["grounded"] else "passed")
    failed_calls = sorted({str(c["error_code"]) for c in row["tool_calls"] if c["is_error"]})
    return f"{row['case_id']}: {outcome}" + (
        f", tool errors {failed_calls}" if failed_calls else ""
    )


def _sign_in(args: argparse.Namespace) -> int:
    """Ask the API which models we may use. It costs nothing and calls no model, so it is the
    way to find out that a key or a federation rule is wrong before any money is spent."""
    try:
        available = {model.id for model in anthropic.Anthropic().models.list(limit=1000)}
    except Exception as error:
        # Any failure here is the answer to the question asked, so it is reported, not raised.
        # (With no credentials at all the SDK raises a plain TypeError, not one of its own.)
        print(f"FAILED: could not sign in: {type(error).__name__}: {str(error)[:400]}")
        return 1
    wanted = [model for model in report.PRICES if not runner.is_scripted(model)]
    print(f"signed in: {len(available)} models are available")
    for model in wanted:
        print(f"  {'available    ' if model in available else 'NOT AVAILABLE'}  {model}")
    return 0 if all(model in available for model in wanted) else 1


def _run(args: argparse.Namespace) -> int:
    config = runner.RunConfig(
        model=args.model,
        out=args.out,
        cases_dir=args.cases,
        db=args.db,
        effort=args.effort,
        reps=args.reps,
        tools=not args.no_tools,
        only=args.only,
        split=args.split,
        limit=args.limit,
        concurrency=args.concurrency,
        max_cost_usd=args.max_cost_usd,
    )
    return runner.run(config)


def _report(args: argparse.Namespace) -> int:
    summary = report.summarise(args.run_dir)
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.run_dir / "summary.md").write_text(report.markdown(summary))
    print(report.markdown(summary))
    return 0 if summary["valid"] else 2


def _share(args: argparse.Namespace) -> int:
    report.shareable_copy(args.run_dir, args.out_dir)
    return 0


def _baseline(args: argparse.Namespace) -> int:
    summaries = [report.summarise(run_dir) for run_dir in args.run_dirs]
    invalid = [str(d) for d, s in zip(args.run_dirs, summaries, strict=True) if not s["valid"]]
    if invalid:
        print(f"FAILED: not valid runs: {', '.join(invalid)}")
        return 2
    try:
        baseline = report.make_baseline(summaries)
    except ValueError as error:
        print(f"FAILED: {error}")
        return 2
    out = args.out or BASELINES / f"{baseline['model']}.json"
    out.write_text(json.dumps(baseline, indent=2) + "\n")
    print(
        f"baseline {baseline['mean']:.1%} over {baseline['runs']} runs; gate {baseline['gate']:.1%}"
    )
    return 0


def _gate(args: argparse.Namespace) -> int:
    summary = report.summarise(args.run_dir)
    baseline = args.baseline or BASELINES / f"{summary['model']}.json"
    code, message = report.gate_verdict(summary, json.loads(baseline.read_text()))
    print(message)
    return code


if __name__ == "__main__":
    sys.exit(main())
