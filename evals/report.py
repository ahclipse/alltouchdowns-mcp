"""Turn a run's raw rows into numbers and a summary.

Everything here is recomputed from `results.jsonl`, one row per trial. Nothing is read from a
stored aggregate, so the headline can always be traced back to individual trials.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# US dollars per million tokens: (input, output, cache read). Cache writes cost 1.25 x input.
# From Anthropic's published prices on 2026-11-03. Cost is always derived from the model that
# actually served a row and the tokens it reported, never assumed.
PRICES: dict[str, tuple[float, float, float]] = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    # The scripted stand-ins used to test the harness itself (see scripted.py).
    "scripted-oracle": (0.0, 0.0, 0.0),
    "scripted-null": (0.0, 0.0, 0.0),
}
CACHE_WRITE_FACTOR = 1.25

VALID_SHARE = 0.98  # of the planned trials that must have been graded


def cost_usd(model: str, usage: dict[str, int]) -> float:
    price_in, price_out, price_cache_read = PRICES[model]
    return (
        usage["input_tokens"] * price_in
        + usage["output_tokens"] * price_out
        + usage["cache_read_input_tokens"] * price_cache_read
        + usage["cache_creation_input_tokens"] * price_in * CACHE_WRITE_FACTOR
    ) / 1_000_000


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def total_cost(rows: Sequence[dict[str, Any]], errors: Sequence[dict[str, Any]]) -> float:
    """Everything a run has spent: graded trials, and trials that failed part-way through
    (those had already been billed for the requests they did make)."""
    graded = sum(cost_usd(row["model"], row["usage"]) for row in rows)
    return graded + sum(error.get("cost_usd", 0.0) for error in errors)


def summarise(run_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(run_dir / "results.jsonl")
    errors = read_jsonl(run_dir / "errors.jsonl")
    manifest = json.loads((run_dir / "manifest.json").read_text())

    # Every trial that produced a reply is in every denominator. One that was cut off at the
    # token limit, or refused, gave no answer and so counts as wrong; leaving those out would
    # score a model only on the questions it managed to finish.
    answerable = [row for row in rows if row["expect"] == "answer"]
    declines = [row for row in rows if row["expect"] == "decline"]
    said_not_covered = [row for row in rows if row["answer_kind_given"] == "not_covered"]
    calls = [call for row in rows for call in row["tool_calls"]]

    by_case: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row["correct"] and row["grounded"])

    # A trial that failed and was completed by a later run of the same directory is settled.
    done = {(row["case_id"], row["rep"]) for row in rows}
    never_completed = {(error["case_id"], error["rep"]) for error in errors} - done
    planned = manifest["cases"] * manifest["reps"]

    return {
        "model": manifest["model"],
        "effort": manifest["effort"],
        "tools": manifest["tools"],
        "cases": manifest["cases"],
        "cases_hash": manifest["cases_hash"],
        "trials_per_case": manifest["reps"],
        "trials_planned": planned,
        "trials": len(rows),
        "truncated": sum(row["status"] == "truncated" for row in rows),
        "refused": sum(row["status"] == "refused" for row in rows),
        "errors": len(never_completed),
        "error_attempts": len(errors),
        # A run is only a measurement if nearly every planned trial was graded.
        "valid": len(rows) >= VALID_SHARE * planned > 0 and not manifest.get("stopped"),
        # The headline: right, and the answer came from the tools rather than from memory.
        "correct_and_grounded": _rate(row["correct"] and row["grounded"] for row in rows),
        "correct_and_grounded_ci95": _bootstrap_ci(by_case),
        "correct": _rate(row["correct"] for row in rows),
        "answer_accuracy": _rate(row["correct"] for row in answerable),
        # Of the questions that should be declined, how many were. And of the declines given,
        # how many were warranted. Reported apart because one can be traded for the other.
        "decline_recall": _rate(row["correct"] for row in declines),
        "decline_precision": _rate(row["expect"] == "decline" for row in said_not_covered),
        "expected_tools_used": _rate(row["expected_tools_used"] for row in answerable),
        "pass_any_trial": _rate(any(results) for results in by_case.values()),
        "pass_every_trial": _rate(all(results) for results in by_case.values()),
        "tool_calls_per_trial": _mean(len(row["tool_calls"]) for row in rows),
        "over_call_budget": _rate(row["over_budget"] for row in rows),
        "tool_error_rate": _rate(call["is_error"] for call in calls),
        "invalid_argument_rate": _rate(call["error_code"] == "invalid_argument" for call in calls),
        "forced_final_rate": _rate(row["forced_final"] for row in rows),
        "model_seconds_p50": _percentile([row["model_seconds"] for row in rows], 50),
        "model_seconds_p95": _percentile([row["model_seconds"] for row in rows], 95),
        "tool_seconds_p50": _percentile([call["seconds"] for call in calls], 50),
        "input_tokens": sum(row["usage"]["input_tokens"] for row in rows),
        "output_tokens": sum(row["usage"]["output_tokens"] for row in rows),
        "cache_read_tokens": sum(row["usage"]["cache_read_input_tokens"] for row in rows),
        "cost_usd": round(total_cost(rows, errors), 4),
        "failures": _count(row["failure"] for row in rows if row["failure"]),
        "by_category": {
            category: {
                "trials": len(group),
                "correct_and_grounded": _rate(row["correct"] and row["grounded"] for row in group),
            }
            for category, group in sorted(_group(rows, "category").items())
        },
        "banned_terms_from_model": sum(row["banned_terms_from_model"] for row in rows),
    }


def shareable_copy(run_dir: Path, out_dir: Path) -> None:
    """Copy a run without any of the model's own words: scores and identifiers only.

    This is what CI uploads. Anyone signed in to GitHub can download the artifacts of a public
    repository's workflows, and free text written by a model may contain terms this project
    never publishes. The full run, with transcripts, stays on the machine that made it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "summary.json", "summary.md"):
        if (run_dir / name).exists():
            (out_dir / name).write_text((run_dir / name).read_text())
    for name, model_text in (("results.jsonl", "answer"), ("errors.jsonl", "detail")):
        rows = [
            {k: v for k, v in row.items() if k != model_text} for row in read_jsonl(run_dir / name)
        ]
        (out_dir / name).write_text("".join(json.dumps(row) + "\n" for row in rows))


def markdown(summary: dict[str, Any]) -> str:
    """A short report for the GitHub job summary. Numbers and ids only, no model text."""
    low, high = summary["correct_and_grounded_ci95"]
    lines = [
        f"## Eval: {summary['model']} (effort {summary['effort']})",
        "",
        f"**Correct and grounded: {_pct(summary['correct_and_grounded'])}** "
        f"(95% CI {_pct(low)} to {_pct(high)}; {summary['cases']} cases, "
        f"{summary['trials']} of {summary['trials_planned']} planned trials graded)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Answer accuracy (questions the data answers) | {_pct(summary['answer_accuracy'])} |",
        f"| Decline recall (questions it must decline) | {_pct(summary['decline_recall'])} |",
        f"| Decline precision | {_pct(summary['decline_precision'])} |",
        f"| Passed in every trial / in any trial | {_pct(summary['pass_every_trial'])} / "
        f"{_pct(summary['pass_any_trial'])} |",
        f"| Used an expected tool | {_pct(summary['expected_tools_used'])} |",
        f"| Tool calls per trial | {summary['tool_calls_per_trial']:.1f} |",
        f"| Invalid-argument rate (per tool call) | {_pct(summary['invalid_argument_rate'])} |",
        f"| Model time per trial, p50 / p95 | {summary['model_seconds_p50']:.1f} s / "
        f"{summary['model_seconds_p95']:.1f} s |",
        f"| Cost | ${summary['cost_usd']:.2f} |",
        f"| Cut off at the token limit / refused (both count as wrong) | "
        f"{summary['truncated']} / {summary['refused']} |",
        f"| Trials never completed (failed attempts in all) | {summary['errors']} "
        f"({summary['error_attempts']}) |",
        f"| Banned terms written by the model | {summary['banned_terms_from_model']} |",
        "",
        "| Category | Trials | Correct and grounded |",
        "|---|---|---|",
        *(
            f"| {name} | {group['trials']} | {_pct(group['correct_and_grounded'])} |"
            for name, group in summary["by_category"].items()
        ),
    ]
    if summary["failures"]:
        failures = ", ".join(f"{name} {count}" for name, count in summary["failures"].items())
        lines += ["", f"Failures by kind: {failures}."]
    if not summary["valid"]:
        lines += [
            "",
            "**This run is not valid**: too few of the planned trials were graded, or it was "
            "stopped early. Its numbers are not a measurement.",
        ]
    return "\n".join(lines) + "\n"


def noise_floor(cases: int, trials_per_case: int) -> float:
    """Roughly how far a pass rate can move between identical runs, as a fraction.

    The rule of thumb 1/sqrt(trials) treats every trial as independent. Trials of the same
    case are not, so the truth lies between this and 1/sqrt(cases).
    """
    return 1 / (cases * trials_per_case) ** 0.5


# --- small helpers ---------------------------------------------------------------------------


def _rate(flags) -> float:
    flags = list(flags)
    return sum(bool(flag) for flag in flags) / len(flags) if flags else 0.0


def _mean(values) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def _percentile(values: Sequence[float], percent: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(percent / 100 * (len(ordered) - 1)))]


def _count(items) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in items:
        counts[item] += 1
    return dict(sorted(counts.items(), key=lambda pair: -pair[1]))


def _group(rows: Sequence[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return groups


def _bootstrap_ci(by_case: dict[str, list[bool]], resamples: int = 2000) -> tuple[float, float]:
    """95% interval for the pass rate, resampling whole cases.

    Trials of one case rise and fall together, so the cases, not the trials, are what vary
    from one imagined run to the next. The seed is fixed so a report is reproducible.
    """
    cases = list(by_case.values())
    if not cases:
        return (0.0, 0.0)
    rng = random.Random(0)
    rates = []
    for _ in range(resamples):
        sample = [trial for case in rng.choices(cases, k=len(cases)) for trial in case]
        rates.append(sum(sample) / len(sample))
    rates.sort()
    return (rates[int(0.025 * resamples)], rates[int(0.975 * resamples) - 1])


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


# --- Baseline and gate -----------------------------------------------------------------------


# Two runs can be compared only if they asked the same questions the same number of times.
_SAME_EXPERIMENT = ("model", "effort", "cases", "cases_hash", "trials_per_case")


def make_baseline(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Set the pass mark from measured runs of the current code, not from a round number.

    The gate sits below the baseline mean by the larger of twice the run-to-run spread and the
    noise floor for this many trials. A later run below the gate is unlikely to be chance.
    """
    first = summaries[0]
    if any(run[key] != first[key] for run in summaries for key in _SAME_EXPERIMENT):
        raise ValueError("the runs differ in model, effort, cases or trials per case")
    scores = [summary["correct_and_grounded"] for summary in summaries]
    floor = noise_floor(first["cases"], first["trials_per_case"])
    spread = statistics.stdev(scores) if len(scores) > 1 else 0.0
    mean = statistics.fmean(scores)
    return {
        **{key: first[key] for key in _SAME_EXPERIMENT},
        "runs": len(scores),
        "scores": scores,
        "mean": mean,
        "run_to_run_stdev": spread,
        "noise_floor": floor,
        "gate": mean - max(2 * spread, floor),
    }


def gate_verdict(summary: dict[str, Any], baseline: dict[str, Any]) -> tuple[int, str]:
    """Exit code and message: 0 passed, 1 regression, 2 not a valid measurement."""
    if not summary["valid"]:
        return 2, "invalid run: no score reported"
    if any(summary[key] != baseline.get(key) for key in _SAME_EXPERIMENT):
        return 2, "the baseline was measured on a different model, effort, case set or trial count"
    score, gate = summary["correct_and_grounded"], baseline["gate"]
    verdict = "passed" if score >= gate else "REGRESSION"
    message = (
        f"{verdict}: {_pct(score)} against a gate of {_pct(gate)} "
        f"(baseline {_pct(baseline['mean'])})"
    )
    return (0 if score >= gate else 1), message
