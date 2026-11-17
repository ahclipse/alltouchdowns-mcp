"""The shape of an eval case, and of the answer an agent must give."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

CASES_DIR = Path(__file__).parent / "cases"

AnswerKind = Literal["number", "ordered_list", "set", "boolean"]
ItemType = Literal["player_slug", "game_id", "team_abbr", "td_type"]
DeclineReason = Literal[
    "passer_not_credited",  # the data credits scorers, never passers
    "outside_seasons",  # before 1999, another league, or a season not loaded
    "not_a_touchdown_stat",  # field goals, yards, wins, ...
    "excluded_content",  # play descriptions, logos, video
    "bulk_export",  # "give me everything"
    "other",
]


# As an argument of a reference call: "the cursor the previous call returned". A real cursor
# cannot be written into a case file, because it is only valid for one export of the data.
NEXT_CURSOR = "$next_cursor"


class ReferenceCall(BaseModel):
    """One tool call of a scripted solution."""

    model_config = ConfigDict(extra="forbid")
    tool: str
    args: dict[str, Any]

    def args_after(self, previous_result: Mapping[str, Any]) -> dict[str, Any]:
        """The arguments to send, given the structured result of the call before this one."""
        return {
            name: previous_result.get("next_cursor") if value == NEXT_CURSOR else value
            for name, value in self.args.items()
        }


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    category: str
    # "heldout" cases are scored but never read while tuning tool descriptions, so the
    # descriptions cannot be fitted to them.
    split: Literal["dev", "heldout"]
    question: str
    # "fixture" marks a case that is only right against the two-season test database, such as
    # declining a season that the full dataset does cover.
    requires: Literal["any", "fixture"] = "any"
    # "hard" cases need something the tools do not hand over: combining calls, counting rows,
    # or resisting a number that is on screen and wrong. They are scored on their own as well,
    # because every model passed nearly all of the "core" cases.
    tier: Literal["core", "hard"] = "core"

    expect: Literal["answer", "decline"]
    # What kind of answer the question calls for. A decline case states the kind a naive
    # answer would have, so the answer format never gives the right behaviour away.
    answer_kind: AnswerKind
    item_type: ItemType | None = None

    # Ground truth, as SQL against the fixture database. It may use only columns copied
    # unchanged from the source (see truth.py), never the server's own derived columns.
    truth_sql: str | None = None
    decline_reason: DeclineReason | None = None
    # Other labels that fit the same decline. The agent sees the labels with no explanation,
    # and for a question about another league "other" describes it as well as the first does.
    decline_reason_also: list[DeclineReason] = []
    # For a decline case: SQL for the tempting wrong number, to detect an agent that fell for it.
    trap_sql: str | None = None

    # Tools a sensible solution uses: every inner list must have at least one tool called.
    # A soft signal. An agent that answers correctly another way is still correct.
    expected_tools: list[list[str]] = []
    max_calls: int = 4

    # For a list of players picked by position ("which quarterbacks"): the answer counts as
    # grounded only if some tool result showed a position beside each player named. Without
    # this, a list of scorers filtered from memory looks as if it came from the tools.
    # Position is the only attribute supported: it is the one that sits beside a player's
    # slug in a result, where the grader can pair the two.
    evidence_field: Literal["position"] | None = None

    # A scripted solution, run in the free pre-flight to prove the case can be answered
    # through the tools. `reference_path` points at the answer inside the last call's result,
    # e.g. "matched.regular_season" or "rows[*].player.slug". A case whose answer is no single
    # field leaves it out and has a worked solution in solutions.py instead.
    reference_calls: list[ReferenceCall] = []
    reference_path: str | None = None

    notes: str

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.expect == "answer":
            if not self.truth_sql:
                raise ValueError("a case that expects an answer needs truth_sql")
            if not self.reference_calls:
                raise ValueError("a case that expects an answer needs reference_calls")
            if self.decline_reason or self.decline_reason_also or self.trap_sql:
                raise ValueError("decline_reason and trap_sql belong to decline cases")
        else:
            if not self.decline_reason:
                raise ValueError("a decline case needs decline_reason")
            if self.truth_sql or self.reference_calls:
                raise ValueError("a decline case has no truth_sql or reference_calls")
        if self.answer_kind in ("set", "ordered_list") and self.item_type is None:
            raise ValueError("a list answer needs item_type")
        if self.evidence_field and (self.expect != "answer" or self.item_type != "player_slug"):
            raise ValueError("evidence_field belongs to a case that answers with players")
        return self


class FinalAnswer(BaseModel):
    """What the agent returns. The grader reads only the field that `kind` names."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["number", "list", "boolean", "not_covered"]
    number: int | None
    items: list[str]
    boolean: bool | None
    not_covered_reason: DeclineReason | None
    note: str


# The same shape as a JSON schema, sent to the model as its required output format.
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "number", "items", "boolean", "not_covered_reason", "note"],
    "properties": {
        "kind": {"type": "string", "enum": ["number", "list", "boolean", "not_covered"]},
        "number": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "items": {
            "type": "array",
            "description": "Bare identifiers only: one per item, with no other text.",
            "items": {"type": "string"},
        },
        "boolean": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
        "not_covered_reason": {
            "anyOf": [
                {
                    "type": "string",
                    "enum": [
                        "passer_not_credited",
                        "outside_seasons",
                        "not_a_touchdown_stat",
                        "excluded_content",
                        "bulk_export",
                        "other",
                    ],
                },
                {"type": "null"},
            ]
        },
        "note": {"type": "string"},
    },
}


def load_cases(directory: Path = CASES_DIR) -> list[Case]:
    """Read every case file. Ids must be unique across files."""
    cases: list[Case] = []
    for path in sorted(directory.glob("*.toml")):
        with path.open("rb") as file:
            cases += [Case.model_validate(raw) for raw in tomllib.load(file).get("case", [])]

    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise ValueError(f"duplicate case id: {case.id}")
        seen.add(case.id)
    return cases
