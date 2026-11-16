"""A stand-in for the model, so the whole harness can be tested for free.

Before trusting what an eval says about a model, check what it says about two agents whose
scores are known in advance:

    oracle  replays each case's scripted tool calls against the real server, then gives the
            true answer. It must score 100 %. If it does not, the runner or grader is broken.
    null    calls nothing and answers -1 to everything. It must score 0 %. If it does not,
            the grader is too lenient.

Both run through exactly the same runner, server and grader as a real model.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from evals.preflight import oracle_answer
from evals.schema import Case, FinalAnswer
from evals.truth import Truth

ORACLE = "scripted-oracle"
NULL = "scripted-null"


@dataclass
class _Text:
    text: str
    type: Literal["text"] = "text"


@dataclass
class _ToolUse:
    id: str
    name: str
    input: dict[str, Any]
    type: Literal["tool_use"] = "tool_use"


@dataclass
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class _Response:
    model: str
    stop_reason: str
    content: list[Any]
    usage: _Usage = field(default_factory=_Usage)


class ScriptedApi:
    """Has the one method of the Anthropic client that the agent loop uses."""

    def __init__(self, cases: Sequence[Case], answers: Mapping[str, Truth]) -> None:
        self._cases = {case.question: case for case in cases}
        self._answers = answers
        self.messages = self  # so that `api.messages.create(...)` works

    async def create(self, *, model: str, messages: list[dict[str, Any]], **_: Any) -> _Response:
        case = self._cases[messages[0]["content"]]
        calls_made = sum(message["role"] == "assistant" for message in messages)

        if model == ORACLE:
            if case.expect == "answer" and calls_made < len(case.reference_calls):
                call = case.reference_calls[calls_made]
                arguments = call.args_after(_last_tool_result(messages))
                tool_use = _ToolUse(id=f"call-{calls_made}", name=call.tool, input=arguments)
                return _Response(model, "tool_use", [tool_use])
            answer = oracle_answer(case, self._answers.get(case.id))
        else:
            answer = FinalAnswer(
                kind="number", number=-1, items=[], boolean=None, not_covered_reason=None, note=""
            )
        return _Response(model, "end_turn", [_Text(answer.model_dump_json())])

    async def close(self) -> None:
        pass


def _last_tool_result(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """The structured result of the most recent tool call, read back from the conversation.
    Empty before the first call, when the last message is still the question."""
    content = messages[-1]["content"]
    if not isinstance(content, list):
        return {}
    return json.loads(content[-1]["content"])
