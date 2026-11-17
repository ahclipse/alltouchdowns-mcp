"""Grading: compare an agent's answer with the truth. Pure functions, no model involved.

Every answer here is a number, a yes/no, or a list of identifiers, so exact comparison is
both possible and fair. Identifiers are normalised first (case, spaces, a team's old and new
abbreviation), because those differences say nothing about whether the agent was right.
"""

from __future__ import annotations

import json
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
    if case.item_type in ("player_slug", "game_id"):
        # A slug or a game id never contains a space, so anything after one is commentary
        # ("torry-holt (WR, 7 TD)"). The identifier is graded; `decorated` reports the rest.
        text = text.split()[0] if text.split() else ""
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

    Appearing is not enough when the question picks players by position. Asked which of a
    team's scorers are quarterbacks, an agent can list the scorers with one tool and choose
    among them from memory. Such a case sets `evidence_field`, and each player in the answer
    must then have been shown with a position beside his slug. That proves the players named
    and nothing about the ones left out, and it asks for a position, not for a particular
    one. Neither can turn a wrong list into a pass, because the list is graded exactly.
    """
    if answer is None:
        return False
    if answer.kind == "not_covered":
        return True
    if not tool_results:
        return False
    if answer.kind != "list":
        return True

    shown = "\n".join(tool_results)
    seen = fold(shown)
    for item in answer.items:
        if case.item_type == "team_abbr":
            # A team counts as seen under any of its names: the tools show the abbreviation of
            # the era, and an agent may reasonably answer with today's.
            franchise = normalise(item, case, franchises)
            spellings = [name for name, owner in franchises.items() if owner == franchise]
            found = any(_team_shown(name, shown, seen) for name in spellings)
        else:
            # The whole identifier, not the start of a longer one: a result that holds only
            # "adrian-peterson-2" has not shown "adrian-peterson".
            identifier = re.escape(fold(normalise(item, case, franchises)))
            found = re.search(rf"(?<![\w-]){identifier}(?![\w-])", seen) is not None
        if not found:
            return False

    if case.evidence_field:
        beside = _players_shown_with(case.evidence_field, tool_results)
        return all(normalise(item, case, franchises) in beside for item in answer.items)
    return True


def _team_shown(name: str, shown: str, folded: str) -> bool:
    """Did the tool results show this way of writing a team?

    A city or a label is compared loosely. An abbreviation counts only in capitals, as the
    tools write it: folded, "NO", "WAS" and "TEN" are ordinary words, and a note that begins
    "No touchdown scorer matches" would otherwise show a team.
    """
    if name.isalpha() and len(name) <= 3:
        return re.search(rf"\b{name.upper()}\b", shown) is not None
    return re.search(rf"\b{re.escape(name)}\b", folded) is not None


def _players_shown_with(field: str, tool_results: Sequence[str]) -> set[str]:
    """The slugs of every player that some tool result showed together with `field`.

    The field sits either on the player's own object (a find_players match) or one level up
    from it (a leaderboard row, whose `player` holds the slug).
    """
    shown: set[str] = set()

    def visit(node: object) -> None:
        if isinstance(node, list):
            for child in node:
                visit(child)
        elif isinstance(node, dict):
            if node.get(field) is not None:
                for holder in (node, *node.values()):
                    if isinstance(holder, dict) and isinstance(holder.get("slug"), str):
                        shown.add(holder["slug"].lower())
            for child in node.values():
                visit(child)

    for text in tool_results:
        try:
            visit(json.loads(text))
        except ValueError:
            continue  # not a structured result, so it shows nothing in a named field
    return shown


def decorated(case: Case, answer: FinalAnswer | None) -> bool:
    """Did a list answer put more than the identifier in an item?

    That breaks the answer format but says nothing about whether the agent found the right
    thing, so it is counted on its own and not as a wrong answer.
    """
    if answer is None or answer.kind != "list" or case.item_type not in ("player_slug", "game_id"):
        return False
    return any(len(item.split()) > 1 for item in answer.items)


def decline_reason_matches(case: Case, answer: FinalAnswer | None) -> bool | None:
    """For a question that was rightly declined: was it declined for the right reason?
    None when the question does not apply. Reported, not graded: any decline is a pass."""
    if case.expect != "decline" or answer is None or answer.kind != "not_covered":
        return None
    return answer.not_covered_reason in {case.decline_reason, *case.decline_reason_also}


def repeated_calls(calls: Sequence[tuple[str, Mapping[str, object]]]) -> int:
    """How many tool calls were exact repeats of an earlier call in the same trial. The server
    is deterministic, so a repeat can never return anything new: it is pure cost."""
    seen: set[str] = set()
    repeats = 0
    for name, arguments in calls:
        key = f"{name} {sorted(arguments.items())!r}"
        repeats += key in seen
        seen.add(key)
    return repeats


def used_expected_tools(case: Case, called: Sequence[str]) -> bool:
    """Every group of acceptable tools had at least one of its tools called."""
    return all(any(tool in called for tool in group) for group in case.expected_tools)
