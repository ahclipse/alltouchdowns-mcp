"""One trial: a model, the live server's tools, one question.

The loop is written out by hand rather than delegated to an SDK helper, because the eval needs
to see inside it: how long each tool call took, which calls failed and why, how many tokens
each model request used, and whether the final answer had to be forced.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import anthropic
from mcp import Client
from mcp.types import TextContent, Tool
from pydantic import ValidationError

from evals.schema import ANSWER_SCHEMA, FinalAnswer

MAX_TOKENS = 16_000
RETRY_ATTEMPTS = 6
MAX_RETRY_WAIT_SECONDS = 60.0

# Tells the agent how to report its answer, and nothing about the subject. Everything it knows
# about touchdowns comes from the server's own instructions and tool descriptions, which is
# what a real client would give it.
ANSWER_RULES = """\
Answer the user's question using the tools. Your final reply must be a JSON object:

- kind "number": put the count in `number`.
- kind "list": put the answers in `items`, one identifier per item and nothing else: a \
player's slug, a game's game_id, a team's abbreviation or a touchdown type, exactly as the \
tools return it. Do not add a name, count, position, date or comment to an item; anything \
like that goes in `note`. If the question asks for an order, list the items in that order.
- kind "boolean": put true or false in `boolean`.
- kind "not_covered": use this when the question cannot be answered from this data, and give \
the closest `not_covered_reason`. Do not answer such a question from memory.

Fill the fields you are not using with null (or an empty list for `items`). Use `note` for a \
one-sentence explanation."""

NO_TOOLS_RULES = """\
No tools are available. Answer from what you already know, and give your best guess rather \
than declining, unless the question is impossible in principle."""

Status = Literal["ok", "truncated", "refused"]


class ModelMismatch(Exception):
    """The API answered with a different model than the one asked for."""


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    is_error: bool
    error_code: str | None  # e.g. "invalid_argument", from the server's error text
    seconds: float


@dataclass
class Trial:
    answer: FinalAnswer | None = None
    status: Status = "ok"
    stop_reason: str | None = None
    forced_final: bool = False  # the turn limit ran out and the model was told to answer now
    turns: int = 0
    api_retries: int = 0
    model: str = ""
    model_seconds: float = 0.0
    usage: dict[str, int] = field(
        default_factory=lambda: {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
    )
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[str] = field(default_factory=list)  # successful results only
    transcript: list[dict[str, Any]] = field(default_factory=list)


def tool_definitions(tools: list[Tool]) -> list[dict[str, Any]]:
    """The server's tools in the form the Messages API takes."""
    return [
        {"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
        for t in tools
    ]


async def run_trial(
    api: Any,  # anthropic.AsyncAnthropic, or the scripted stand-in used to test the harness
    mcp: Client | None,
    *,
    model: str,
    effort: str,
    question: str,
    system: str,
    tools: list[dict[str, Any]],
    max_turns: int,
    trial: Trial | None = None,
) -> Trial:
    """Ask one question and let the model call tools until it answers.

    `mcp` is None for the no-tools baseline, where `tools` is empty. Pass a `trial` to fill in
    when the caller needs what was spent even if the trial is cut short: tokens are added to
    it as each model request completes.
    """
    trial = trial if trial is not None else Trial()
    trial.transcript += [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]

    for turn in range(max_turns + 1):
        # After max_turns rounds of tool use, one last request with tools switched off makes
        # the model answer with what it has, so a slow agent is graded rather than dropped.
        trial.forced_final = turn == max_turns
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": MAX_TOKENS,
            # Two cache markers. This one, on the system prompt, covers what every trial of a
            # run shares: the tool definitions and the instructions, which come first in what
            # the model reads. Only the first trial pays to write that; the rest read it. The
            # top-level marker further down caches the conversation as it grows within a
            # trial. With that one alone, every trial wrote the shared part again, because
            # its cache entry ended with that trial's own question.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": messages,
            "output_config": {
                "effort": effort,
                "format": {"type": "json_schema", "schema": ANSWER_SCHEMA},
            },
            "cache_control": {"type": "ephemeral"},
        }
        if tools:
            request["tools"] = tools
            request["tool_choice"] = {"type": "none" if trial.forced_final else "auto"}

        response = await _create(api, request, trial)
        if response.model != model:
            raise ModelMismatch(f"asked for {model}, served by {response.model}")
        trial.turns += 1
        trial.stop_reason = response.stop_reason
        messages.append({"role": "assistant", "content": response.content})

        tool_uses = [block for block in response.content if block.type == "tool_use"]
        for block in response.content:
            if block.type == "text" and block.text.strip():
                trial.transcript.append({"role": "assistant", "content": block.text})

        if response.stop_reason != "tool_use" or mcp is None or trial.forced_final:
            break
        # Every tool result goes back in ONE user message, as the API expects.
        results = [await _call_tool(mcp, block, trial) for block in tool_uses]
        messages.append({"role": "user", "content": results})

    if trial.stop_reason == "max_tokens":
        trial.status = "truncated"
    elif trial.stop_reason == "refusal":
        trial.status = "refused"
    else:
        trial.answer = _parse_answer(messages[-1]["content"])
    return trial


async def _create(api: Any, request: dict[str, Any], trial: Trial) -> Any:
    """One model request, retried when the API is overloaded or unreachable. Retries are
    counted, and only the successful attempt is timed, so a busy API does not look like a slow
    model."""
    for attempt in range(RETRY_ATTEMPTS):
        started = time.perf_counter()
        try:
            response = await api.messages.create(**request)
        except anthropic.APIStatusError as error:
            if error.status_code != 429 and error.status_code < 500:
                raise  # the request itself is wrong; sending it again would not help
            failure: Exception = error
        except anthropic.APIConnectionError as error:
            failure = error
        else:
            trial.model_seconds += time.perf_counter() - started
            trial.model = response.model
            for key in trial.usage:
                trial.usage[key] += getattr(response.usage, key, 0) or 0
            return response
        if attempt == RETRY_ATTEMPTS - 1:
            raise RuntimeError("the API kept failing after retries") from failure
        trial.api_retries += 1
        await asyncio.sleep(_retry_wait(failure, attempt))
    raise AssertionError("unreachable")


def _retry_wait(failure: Exception, attempt: int) -> float:
    """Seconds to wait before trying again: what the API asked for when it said, otherwise a
    wait that doubles each time, with jitter so parallel trials do not all retry together."""
    headers = getattr(getattr(failure, "response", None), "headers", None) or {}
    try:
        wait = float(headers.get("retry-after", ""))
    except ValueError:
        wait = random.uniform(0.5, 1.5) * 2**attempt
    return min(wait, MAX_RETRY_WAIT_SECONDS)


async def _call_tool(mcp: Client, block: Any, trial: Trial) -> dict[str, Any]:
    started = time.perf_counter()
    result = await mcp.call_tool(block.name, block.input)
    seconds = time.perf_counter() - started
    text = "".join(part.text for part in result.content if isinstance(part, TextContent))
    is_error = bool(result.is_error)

    # The server's errors start with a code and a colon: "unknown_team: No team matches..."
    code = text.split(":", 1)[0] if is_error and ":" in text[:40] else None
    trial.tool_calls.append(ToolCall(block.name, dict(block.input), is_error, code, seconds))
    if not is_error:
        trial.tool_results.append(text)
    trial.transcript += [
        {"role": "tool_call", "name": block.name, "content": json.dumps(block.input, indent=2)},
        {"role": "tool_result", "content": text},
    ]
    return {"type": "tool_result", "tool_use_id": block.id, "content": text, "is_error": is_error}


def _parse_answer(content: list[Any]) -> FinalAnswer | None:
    """The final reply is one JSON text block. Anything else counts as no answer."""
    text = next((block.text for block in content if block.type == "text"), None)
    if text is None:
        return None
    try:
        return FinalAnswer.model_validate_json(text)
    except ValidationError:
        return None
