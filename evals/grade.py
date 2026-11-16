"""Grading: compare an agent's answer with the truth. Pure functions, no model involved.

Every answer here is a number, a yes/no, or a list of identifiers, so exact comparison is
both possible and fair. Identifiers are normalised first (case, spaces, a team's old and new
abbreviation), because those differences say nothing about whether the agent was right.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from evals.schema import Case, FinalAnswer
from evals.truth import Truth, fold

Failure = Literal[
    "no_answer",  # nothing parseable came back
    "malformed",  # the wrong kind of answer for the question
    "wrong_value",
    "wrong_identity",  # a list with the wrong members
    "wrong_order",
    "tie_truncation",  # a ranked list cut off in the middle of a tie
    "false_decline",  # said "not covered" for a question the data answers
    "false_answer",  # answered a question that should have been declined
    "fell_for_trap",  # ... with exactly the misleading number the data offers
]


@dataclass(frozen=True)
class Grade:
    correct: bool
    failure: Failure | None = None


RIGHT = Grade(correct=True)


def wrong(failure: Failure) -> Grade:
    return Grade(correct=False, failure=failure)


def grade(
    case: Case,
    answer: FinalAnswer | None,
    truth: Truth | None,
    *,
    franchises: Mapping[str, str],
    trap: int | None = None,
) -> Grade:
    if answer is None:
        # No answer is never a correct decline: silence and "this is not covered" are
        # different things, and only the second is right for a decline case.
        return wrong("no_answer")

    if case.expect == "decline":
        if answer.kind == "not_covered":
            return RIGHT
        fell = trap is not None and answer.kind == "number" and answer.number == trap
        return wrong("fell_for_trap" if fell else "false_answer")

    if answer.kind == "not_covered":
        return wrong("false_decline")

    if case.answer_kind == "number":
        if answer.kind != "number" or answer.number is None:
            return wrong("malformed")
        return RIGHT if answer.number == truth else wrong("wrong_value")

    if case.answer_kind == "boolean":
        if answer.kind != "boolean" or answer.boolean is None:
            return wrong("malformed")
        return RIGHT if answer.boolean == truth else wrong("wrong_value")

    if answer.kind != "list":
        return wrong("malformed")
    items = [normalise(item, case, franchises) for item in answer.items]

    if case.answer_kind == "set":
        assert isinstance(truth, frozenset)
        expected = {normalise(item, case, franchises) for item in truth}
        # Naming an item twice (or one team by two of its names) does no harm in a set.
        return RIGHT if set(items) == expected else wrong("wrong_identity")

    assert isinstance(truth, list)
    groups = [{normalise(item, case, franchises) for item in group} for group in truth]
    return _grade_ranked(items, groups)


def _grade_ranked(items: Sequence[str], groups: Sequence[set[str]]) -> Grade:
    """A ranked list is right when each tie group appears whole, in rank order. The order of
    the items inside a tie group is free."""
    everything = set().union(*groups)
    if set(items) != everything or len(items) != len(everything):
        if set(items) < everything and _is_prefix_cut_inside_a_tie(items, groups):
            return wrong("tie_truncation")
        return wrong("wrong_identity")

    position = 0
    for group in groups:
        if set(items[position : position + len(group)]) != group:
            return wrong("wrong_order")
        position += len(group)
    return RIGHT


def _is_prefix_cut_inside_a_tie(items: Sequence[str], groups: Sequence[set[str]]) -> bool:
    given = set(items)
    for group in groups:
        if group <= given:
            given -= group
        else:
            # The first group not fully present: a cut is "inside the tie" if part of the
            # group was given and nothing from a later group was.
            return bool(given) and given < group
    return False


def normalise(item: str, case: Case, franchises: Mapping[str, str]) -> str:
    """Reduce an identifier to one canonical form, so that only real differences count."""
    text = item.strip()
    if case.item_type == "team_abbr":
        return franchises.get(fold(text), fold(text))
    if case.item_type == "game_id":
        return text.upper()
    if case.item_type == "td_type":
        # "Fumble return" and "fumble-return" are the tools' "fumble_return".
        value = text.lower().replace("-", "_").replace(" ", "_")
        return {"interception_return": "int_return"}.get(value, value)
    return text.lower()


def grounded(
    case: Case,
    answer: FinalAnswer | None,
    tool_results: Sequence[str],
    franchises: Mapping[str, str],
) -> bool:
    """Did the answer come from the tools rather than the model's memory?

    For a list, every item must appear in something a tool returned. A number cannot be traced
    that way, so it only needs at least one successful tool call behind it. A decline needs
    nothing: recognising that a question is out of scope is the point.
    """
    if answer is None:
        return False
    if answer.kind == "not_covered":
        return True
    if not tool_results:
        return False
    if answer.kind != "list":
        return True

    seen = fold("\n".join(tool_results))
    for item in answer.items:
        if case.item_type == "team_abbr":
            # A team counts as seen under any of its names: the tools show the abbreviation of
            # the era, and an agent may reasonably answer with today's.
            franchise = normalise(item, case, franchises)
            spellings = [name for name, owner in franchises.items() if owner == franchise]
            found = any(re.search(rf"\b{re.escape(name)}\b", seen) for name in spellings)
        else:
            found = fold(normalise(item, case, franchises)) in seen
        if not found:
            return False
    return True


def used_expected_tools(case: Case, called: Sequence[str]) -> bool:
    """Every group of acceptable tools had at least one of its tools called."""
    return all(any(tool in called for tool in group) for group in case.expected_tools)
