"""Free checks that the eval suite itself is sound. No model is called, so they cost nothing
and run on every pull request.

An eval that is broken still prints a confident number. These checks catch the usual ways:

    validate   every case is well formed and its truth SQL runs and returns the right shape
    oracle     the true answer, fed through the grader, scores 100 %
    null       no answer, a wrong answer and "not covered" score what they should
    reference  each case's scripted tool calls really do produce its true answer, which proves
               the question can be answered through the tools (and checks the tools against
               independent SQL at the same time). The answer is either one field of the last
               result, or what a worked solution computes from all of the results
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mcp import Client
from mcp.types import TextContent

from alltouchdowns_mcp.config import Settings
from alltouchdowns_mcp.server import create_server
from evals import grade as grading
from evals import truth as truths
from evals.schema import Case, FinalAnswer
from evals.solutions import SOLUTIONS

TOOLS = (
    "find_players",
    "get_player",
    "search_touchdowns",
    "get_game",
    "list_games",
    "get_leaderboard",
)


# --- validate --------------------------------------------------------------------------------


def validate(cases: Sequence[Case], conn: sqlite3.Connection) -> list[str]:
    problems: list[str] = []
    franchises = truths.franchises(conn)
    for case in cases:
        where = f"{case.id}:"
        for group in case.expected_tools:
            problems += [f"{where} unknown tool '{t}'" for t in group if t not in TOOLS]
        problems += [
            f"{where} unknown tool '{call.tool}'"
            for call in case.reference_calls
            if call.tool not in TOOLS
        ]
        problems += [f"{where} {p}" for p in _proof_problems(case)]
        for name in ("truth_sql", "trap_sql"):
            sql = getattr(case, name)
            if sql:
                problems += [f"{where} {name} {p}" for p in truths.sql_problems(sql)]
        if case.expect == "answer" and not truths.sql_problems(case.truth_sql or ""):
            try:
                truth = truths.compute(conn, case)
            except (sqlite3.Error, ValueError) as error:
                problems.append(f"{where} truth_sql failed: {error}")
            else:
                problems += [f"{where} {p}" for p in _one_team_twice(case, truth, franchises)]
        if case.trap_sql and not truths.sql_problems(case.trap_sql):
            try:
                truths.trap_value(conn, case)
            except (sqlite3.Error, ValueError, TypeError) as error:
                problems.append(f"{where} trap_sql failed: {error}")

    questions = Counter(case.question.strip().lower() for case in cases)
    problems += [f"duplicate question: {q!r}" for q, n in questions.items() if n > 1]
    return problems


def _proof_problems(case: Case) -> list[str]:
    """A case that expects an answer proves it can be answered in exactly one of two ways:
    the answer is a field of the last result (`reference_path`), or a worked solution
    computes it from the results (evals/solutions.py)."""
    if case.expect != "answer":
        return []
    proofs = (case.reference_path is not None) + (case.id in SOLUTIONS)
    if proofs == 0:
        return ["needs a reference_path or a worked solution in evals/solutions.py"]
    if proofs == 2:
        return ["has both a reference_path and a worked solution"]
    return []


def _one_team_twice(case: Case, truth: truths.Truth, franchises: Mapping[str, str]) -> list[str]:
    """The grader treats every abbreviation of a franchise as the same team, which is right
    for "SD" against "LAC" but would merge two items of an answer that lists both. Such a
    case cannot be graded fairly, so it is refused here."""
    if case.item_type != "team_abbr" or isinstance(truth, bool | int):
        return []
    items = truth if isinstance(truth, frozenset) else set().union(*truth)
    count = Counter(grading.normalise(item, case, franchises) for item in items)
    twice = sorted(franchise for franchise, n in count.items() if n > 1)
    return [f"the answer names the franchise {name} more than once" for name in twice]


# --- oracle and null -------------------------------------------------------------------------


def oracle_answer(case: Case, truth: truths.Truth | None) -> FinalAnswer:
    """The answer a perfect agent would give."""
    blank: dict[str, Any] = {
        "number": None,
        "items": [],
        "boolean": None,
        "not_covered_reason": None,
        "note": "",
    }
    if case.expect == "decline":
        return FinalAnswer(
            kind="not_covered", **{**blank, "not_covered_reason": case.decline_reason}
        )
    if isinstance(truth, bool):
        return FinalAnswer(kind="boolean", **{**blank, "boolean": truth})
    if isinstance(truth, int):
        return FinalAnswer(kind="number", **{**blank, "number": truth})
    if isinstance(truth, frozenset):
        return FinalAnswer(kind="list", **{**blank, "items": sorted(truth)})
    assert truth is not None
    ordered = [item for group in truth for item in sorted(group)]
    return FinalAnswer(kind="list", **{**blank, "items": ordered})


def wrong_answers(case: Case, right: FinalAnswer) -> list[FinalAnswer | None]:
    """Answers that must all be graded wrong: nothing at all, the opposite of a decline, and
    near misses of the true answer."""
    not_covered = right.model_copy(update={"kind": "not_covered", "not_covered_reason": "other"})
    a_number = right.model_copy(update={"kind": "number", "number": 7, "items": []})
    if case.expect == "decline":
        return [None, a_number]

    wrong: list[FinalAnswer | None] = [None, not_covered]
    if right.kind == "number" and right.number is not None:
        wrong += [right.model_copy(update={"number": right.number + step}) for step in (1, -1)]
    elif right.kind == "boolean":
        wrong.append(right.model_copy(update={"boolean": not right.boolean}))
    else:
        wrong.append(right.model_copy(update={"items": [*right.items, "not-a-real-item"]}))
        if len(right.items) > 1:
            wrong.append(right.model_copy(update={"items": right.items[:-1]}))
    return wrong


def oracle_and_null(
    cases: Sequence[Case], conn: sqlite3.Connection, franchises: Mapping[str, str]
) -> list[str]:
    problems: list[str] = []
    for case in cases:
        truth = truths.compute(conn, case) if case.expect == "answer" else None
        trap = truths.trap_value(conn, case)
        right = oracle_answer(case, truth)

        if not grading.grade(case, right, truth, franchises=franchises, trap=trap).correct:
            problems.append(f"{case.id}: the true answer is graded wrong")
        for answer in wrong_answers(case, right):
            if grading.grade(case, answer, truth, franchises=franchises, trap=trap).correct:
                problems.append(f"{case.id}: a wrong answer is graded right: {answer}")
        if trap is not None:
            fooled = right.model_copy(update={"kind": "number", "number": trap, "items": []})
            verdict = grading.grade(case, fooled, truth, franchises=franchises, trap=trap)
            if verdict.failure != "fell_for_trap":
                problems.append(f"{case.id}: answering the trap value is not recognised")
    return problems


# --- reference -------------------------------------------------------------------------------


def resolve_path(data: Any, path: str) -> Any:
    """Follow a dotted path into nested data. `name[*]` steps into every element of a list.

    resolve_path(result, "rows[*].player.slug") -> ["randy-moss", "ladainian-tomlinson", ...]
    """
    values = [data]
    many = False
    for step in path.split("."):
        each = step.endswith("[*]")
        key = step.removesuffix("[*]")
        values = [value[key] for value in values]
        if each:
            values = [item for value in values for item in value]
            many = True
    return values if many else values[0]


async def reference(
    cases: Sequence[Case], fixture: Path, conn: sqlite3.Connection, franchises: Mapping[str, str]
) -> list[str]:
    """Run each case's scripted tool calls against the real server and compare with the truth."""
    problems: list[str] = []
    server = create_server(Settings(db_path=fixture))
    async with Client(server) as client:
        tools = (await client.list_tools()).tools
        shown_up_front = (client.instructions or "") + json.dumps(
            [tool.model_dump(mode="json") for tool in tools]
        )
        for case in cases:
            if case.expect != "answer":
                continue
            truth = truths.compute(conn, case)
            problems += _leaked(case, truth, shown_up_front)
            structured: dict[str, Any] = {}
            results: list[dict[str, Any]] = []
            texts: list[str] = []
            failed = False
            for call in case.reference_calls:
                result = await client.call_tool(call.tool, call.args_after(structured))
                text = "".join(b.text for b in result.content if isinstance(b, TextContent))
                if result.is_error:
                    problems.append(f"{case.id}: reference call {call.tool} failed: {text[:120]}")
                    failed = True
                    break
                structured = result.structured_content or {}
                results.append(structured)
                texts.append(text)
            if failed:
                continue
            if len(case.reference_calls) > case.max_calls:
                problems.append(f"{case.id}: the reference solution exceeds max_calls")
            if case.id in SOLUTIONS:
                problem = _solve(case, truth, results, franchises)
            else:
                problem = _read_path(case, truth, structured, franchises)
            if problem:
                problems.append(f"{case.id}: {problem}")
            right = oracle_answer(case, truth)
            if case.evidence_field and not grading.grounded(case, right, texts, franchises):
                problems.append(
                    f"{case.id}: the reference calls do not show '{case.evidence_field}' "
                    "beside every player in the answer"
                )
    return problems


def _leaked(case: Case, truth: truths.Truth, shown_up_front: str) -> list[str]:
    """An answer that already appears in the tool descriptions or server instructions can be
    copied without calling anything. Checked for slugs and game ids, which are unguessable;
    team abbreviations and touchdown types are a small vocabulary the descriptions must name."""
    if case.item_type not in ("player_slug", "game_id") or isinstance(truth, bool | int):
        return []
    items = truth if isinstance(truth, frozenset) else set().union(*truth)
    return [
        f"{case.id}: the answer '{item}' appears in what the agent is shown before any call"
        for item in sorted(items)
        if item.lower() in shown_up_front.lower()
    ]


def _read_path(
    case: Case, truth: truths.Truth, last_result: Mapping[str, Any], franchises: Mapping[str, str]
) -> str | None:
    """The answer is one field of the last call's result: read it there and grade it."""
    if case.reference_path is None:
        return "has neither a reference_path nor a worked solution"
    try:
        value = resolve_path(last_result, case.reference_path)
    except (KeyError, TypeError, IndexError) as error:
        return f"reference_path '{case.reference_path}' does not resolve: {error!r}"
    return _judge(case, truth, value, franchises)


def _solve(
    case: Case,
    truth: truths.Truth,
    results: Sequence[Mapping[str, Any]],
    franchises: Mapping[str, str],
) -> str | None:
    """No one field holds the answer: work it out from every result, and grade it."""
    try:
        value = SOLUTIONS[case.id](results)
    except Exception as error:
        # A solution is a few lines over nested results. Whatever goes wrong inside one is a
        # finding about that case, to be listed beside the others, not a crash of the check.
        return f"the worked solution failed on the tool results: {error!r}"
    return _judge(case, truth, value, franchises)


def _judge(
    case: Case, truth: truths.Truth, value: Any, franchises: Mapping[str, str]
) -> str | None:
    """Grade a value taken from the tool results exactly as an agent's answer would be.

    The value must first be the right kind of thing. A null, such as the side of a count
    that a filter left out, is never an answer: read as "no" it would pass a yes/no case
    whose truth is no without proving anything.
    """
    blank: dict[str, Any] = {"number": None, "items": [], "boolean": None,
                             "not_covered_reason": None, "note": ""}  # fmt: skip
    if case.answer_kind == "number":
        if isinstance(value, bool) or not isinstance(value, int):
            return f"the tools gave {value!r}, which is not a whole number"
        answer = FinalAnswer(kind="number", **{**blank, "number": value})
    elif case.answer_kind == "boolean":
        if not isinstance(value, bool | int):  # a count stands for "is there any"
            return f"the tools gave {value!r}, which is not a yes/no or a count"
        answer = FinalAnswer(kind="boolean", **{**blank, "boolean": bool(value)})
    else:
        in_order = isinstance(value, list | tuple)
        unordered = case.answer_kind == "set" and isinstance(value, set | frozenset)
        if not (in_order or unordered) or not all(isinstance(item, str) for item in value):
            return f"the tools gave {value!r}, which is not a list of identifiers"
        items = list(value) if in_order else sorted(value)
        answer = FinalAnswer(kind="list", **{**blank, "items": items})
    verdict = grading.grade(case, answer, truth, franchises=franchises)
    if verdict.correct:
        return None
    return f"the tools returned {value!r} but the truth is {_show(truth)} ({verdict.failure})"


def _show(truth: truths.Truth) -> str:
    if isinstance(truth, list):
        return str([sorted(group) for group in truth])
    return str(sorted(truth)) if isinstance(truth, frozenset) else str(truth)


# --- A summary of the case set ---------------------------------------------------------------


def describe(cases: Sequence[Case]) -> str:
    by_category = Counter(case.category for case in cases)
    declines = sum(case.expect == "decline" for case in cases)
    heldout = sum(case.split == "heldout" for case in cases)
    lines = [
        f"{len(cases)} cases: {len(cases) - declines} to answer, {declines} to decline; "
        f"{heldout} held out",
        *(f"  {count:3}  {category}" for category, count in sorted(by_category.items())),
    ]
    if cases:
        share = declines / len(cases)
        lines.append(f"answering 'not covered' to everything would score {share:.1%}")
    return "\n".join(lines)
