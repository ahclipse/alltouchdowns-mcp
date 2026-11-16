"""Free checks that the eval suite itself is sound. No model is called, so they cost nothing
and run on every pull request.

An eval that is broken still prints a confident number. These checks catch the usual ways:

    validate   every case is well formed and its truth SQL runs and returns the right shape
    oracle     the true answer, fed through the grader, scores 100 %
    null       no answer, a wrong answer and "not covered" score what they should
    reference  each case's scripted tool calls really do produce its true answer, which proves
               the question can be answered through the tools (and checks the tools against
               independent SQL at the same time)
"""

from __future__ import annotations

import json
import re
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
                texts.append(text)
            if failed:
                continue
            if len(case.reference_calls) > case.max_calls:
                problems.append(f"{case.id}: the reference solution exceeds max_calls")
            problem = _compare_reference(case, truth, structured, texts, franchises)
            if problem:
                problems.append(f"{case.id}: {problem}")
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


def _compare_reference(
    case: Case,
    truth: truths.Truth,
    structured: dict[str, Any],
    texts: Sequence[str],
    franchises: Mapping[str, str],
) -> str | None:
    if case.reference_path is None:
        # No single field holds the answer (it takes combining calls). Settle for checking
        # that the pieces of the answer all appear in what the tools returned.
        haystack = "\n".join(texts)
        if isinstance(truth, bool):
            return None
        if isinstance(truth, int):
            found = re.search(rf"(?<![\w.]){truth}(?![\w.])", haystack)
            return None if found else f"the true answer {truth} appears in no tool result"
        items = truth if isinstance(truth, frozenset) else set().union(*truth)
        missing = sorted(item for item in items if item.lower() not in haystack.lower())
        return f"not in any tool result: {missing}" if missing else None

    try:
        value = resolve_path(structured, case.reference_path)
    except (KeyError, TypeError, IndexError) as error:
        return f"reference_path '{case.reference_path}' does not resolve: {error!r}"

    blank: dict[str, Any] = {"number": None, "items": [], "boolean": None,
                             "not_covered_reason": None, "note": ""}  # fmt: skip
    if case.answer_kind == "number":
        answer = FinalAnswer(kind="number", **{**blank, "number": value})
    elif case.answer_kind == "boolean":
        answer = FinalAnswer(kind="boolean", **{**blank, "boolean": bool(value)})
    else:
        answer = FinalAnswer(kind="list", **{**blank, "items": [str(item) for item in value]})
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
