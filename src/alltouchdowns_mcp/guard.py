"""The exit gate: every tool call passes through here on its way in and on its way out.

It is the one place that guarantees four things about a tool result, whichever tool produced
it and whether it succeeded or failed:

* an argument the tool does not have is refused, never silently ignored (a misspelt filter
  would otherwise return an unfiltered answer that looks perfectly normal),
* the result carries the attribution sentence,
* an error says nothing the caller typed (the SDK's own validation errors quote the input),
* it contains no banned term. A result that does is withheld and replaced by a plain error.

Errors of the protocol itself (an unknown method, a malformed request) never become tool
results, so they are handled one layer further out, in `envelope.py`.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.context import CallNext, HandlerResult, ServerRequestContext
from mcp.types import CallToolResult, TextContent
from pydantic import ValidationError

from alltouchdowns_mcp import ATTRIBUTION, legal, logs

logger = logging.getLogger(__name__)

_WITHHELD = "internal_error: The response could not be produced."

# The SDK prefixes every tool error with this.
_SDK_PREFIX = re.compile(r"^Error executing tool [\w.-]+:?\s*")
# Pydantic appends the rejected value to each message, e.g.
#   Input should be a valid integer [type=int_parsing, input_value='abc', input_type=str]
_PYDANTIC_DETAIL = re.compile(r"\s*\[type=.*\]\s*$")
_ERROR_CODE = re.compile(r"^([a-z_]+): ")


class ExitGate:
    """MCP middleware. `arguments` maps each tool to the argument names it accepts."""

    def __init__(self) -> None:
        self.arguments: dict[str, frozenset[str]] = {}

    async def __call__(
        self, ctx: ServerRequestContext[Any, Any], call_next: CallNext
    ) -> HandlerResult:
        if ctx.method != "tools/call":
            return await call_next(ctx)

        started = time.perf_counter()
        final = await self._checked_call(ctx, call_next)
        if final is not None:
            self._log(ctx.params, final, time.perf_counter() - started)
        return final

    async def _checked_call(
        self, ctx: ServerRequestContext[Any, Any], call_next: CallNext
    ) -> CallToolResult | None:
        tool = self._tool_name(ctx.params)
        refusal = self._unknown_argument(ctx.params)
        if refusal is not None:
            return finalise(refusal, tool)

        result = await call_next(ctx)
        if result is None:
            return None
        try:
            # The SDK hands middleware the result already serialised to a dict.
            call_result = (
                result
                if isinstance(result, CallToolResult)
                else CallToolResult.model_validate(result)
            )
        except ValidationError:
            # Not a shape this gate understands, so it cannot be vouched for.
            logger.error("tool result withheld (%s): unexpected result shape", tool)
            return _error(_WITHHELD)
        return finalise(call_result, tool)

    def _tool_name(self, params: Any) -> str:
        """The tool being called, for the log. The name comes from the caller, so only a name
        this server knows is ever written."""
        name = str(params.get("name")) if isinstance(params, dict) else ""
        return name if name in self.arguments else "unknown"

    def _log(self, params: Any, result: CallToolResult, seconds: float) -> None:
        """One line per call. Nothing the caller sent is logged except a known tool name."""
        outcome = "ok"
        if result.is_error and isinstance(result.structured_content, dict):
            outcome = result.structured_content["error"]["code"]
        logs.record(
            "tool_call",
            tool=self._tool_name(params),
            signed_in=get_access_token() is not None,
            outcome=outcome,
            ms=round(seconds * 1000, 1),
        )

    def _unknown_argument(self, params: Any) -> CallToolResult | None:
        """An error result if the call names an argument its tool does not have."""
        if not isinstance(params, dict):
            return None
        allowed = self.arguments.get(str(params.get("name")))
        given = params.get("arguments") or {}
        if allowed is None or not isinstance(given, dict) or set(given) <= allowed:
            return None
        # The unknown name is the caller's text, so it is not repeated. The valid names are.
        message = (
            "invalid_argument: This tool has no argument by one of the names given. "
            f"Its arguments are: {', '.join(sorted(allowed))}."
        )
        return CallToolResult(content=[TextContent(type="text", text=message)], is_error=True)


def finalise(result: CallToolResult, tool: str = "unknown") -> CallToolResult:
    final = _error(_error_message(result), result) if result.is_error else _success(result)
    if legal.find_banned(_all_text(final)):
        # Never log the text itself: the log must not hold the term either.
        logger.error("tool result withheld (%s): it contained a banned term", tool)
        return _error(_WITHHELD, result)
    return final


def _success(result: CallToolResult) -> CallToolResult:
    """Replace the SDK's indented JSON text with compact JSON. Same content, fewer tokens."""
    if result.structured_content is None:
        text = "\n".join(block.text for block in result.content if isinstance(block, TextContent))
    else:
        text = json.dumps(result.structured_content, separators=(",", ":"), ensure_ascii=False)
    if ATTRIBUTION not in text:
        text = f"{text}\n\n{ATTRIBUTION}"
    # model_copy keeps everything else the SDK put on the result, such as its metadata.
    return result.model_copy(update={"content": [TextContent(type="text", text=text)]})


def _error(message: str, original: CallToolResult | None = None) -> CallToolResult:
    """An error result: the message and attribution as text, and the same in structured form
    so a program can read the code without parsing the sentence."""
    code = match.group(1) if (match := _ERROR_CODE.match(message)) else "error"
    update = {
        "content": [TextContent(type="text", text=f"{message}\n\n{ATTRIBUTION}")],
        "structured_content": {
            "error": {"code": code, "message": message},
            "attribution": ATTRIBUTION,
        },
        "is_error": True,
    }
    return (original or CallToolResult(content=[])).model_copy(update=update)


def _error_message(result: CallToolResult) -> str:
    """The message to send for a failed call, with anything the caller typed removed."""
    raw = "\n".join(block.text for block in result.content if isinstance(block, TextContent))
    message = _SDK_PREFIX.sub("", raw).strip()
    if not message:
        return _WITHHELD  # a crash: the SDK already kept the details on the server
    if "validation error" in message.splitlines()[0]:
        return _argument_error(message)
    if message.startswith("Unknown tool"):  # the SDK's message repeats the name it was sent
        return "unknown_tool: There is no tool with that name."
    return message


def _argument_error(pydantic_message: str) -> str:
    """Rebuild an argument-validation error from field names and rules only.

    Pydantic's message looks like this, and the bracketed part repeats the caller's input:

        1 validation error for search_touchdownsArguments
        season
          Input should be a valid integer [type=int_parsing, input_value='abc', input_type=str]
            For further information visit https://errors.pydantic.dev/...
    """
    details: list[str] = []
    field = ""
    for line in pydantic_message.splitlines()[1:]:
        if not line.startswith(" "):
            field = line.strip()
        elif "For further information" not in line:
            details.append(f"{field}: {_PYDANTIC_DETAIL.sub('', line.strip())}.")
    return " ".join(["invalid_argument: An argument is missing or not allowed.", *details])


def _all_text(result: CallToolResult) -> str:
    parts = [block.text for block in result.content if isinstance(block, TextContent)]
    if result.structured_content is not None:
        parts.append(json.dumps(result.structured_content, ensure_ascii=False))
    return "\n".join(parts)
