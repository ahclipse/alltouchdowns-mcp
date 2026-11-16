"""The outer check on everything the MCP endpoint says.

It has two protections: every error message is replaced by a fixed sentence, and anything
holding a banned term is withheld. Each is tested here with input the other would not catch,
so that neither can be removed unnoticed.
"""

from __future__ import annotations

import json
from typing import Any

from alltouchdowns_mcp import ATTRIBUTION
from alltouchdowns_mcp.envelope import checked


def error(message: str, *, request_id: Any = 1, code: Any = -32601, **extra: Any) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message, **extra},
    }


def sent(payload: Any) -> Any:
    """What the endpoint would actually send for a response the SDK wrote."""
    return json.loads(checked(json.dumps(payload).encode()))


# --- Errors say a fixed sentence -------------------------------------------------------------


def test_an_error_message_is_replaced_even_when_nothing_in_it_is_banned() -> None:
    """The SDK's messages repeat the caller's own text, such as the method it did not find."""
    reply = sent(error("Method not found: xyzzy-echo", data={"method": "xyzzy-echo"}))

    assert reply == {
        "jsonrpc": "2.0",
        "id": 1,
        "error": {
            "code": -32601,
            "message": "Method not found.",
            "data": {"attribution": ATTRIBUTION},
        },
    }


def test_every_error_in_a_batch_is_replaced() -> None:
    result = {"jsonrpc": "2.0", "id": 2, "result": {"ok": True}}

    replies = sent([error("Unknown prompt: hunter2", code=-32602), result])

    assert "hunter2" not in json.dumps(replies)
    assert replies[0]["error"]["message"] == "Invalid params."
    assert replies[1] == result  # a success in the same batch is left alone


def test_an_unfamiliar_error_code_gets_a_general_message() -> None:
    assert sent(error("Custom failure: hunter2", code=-32000))["error"] == {
        "code": -32000,
        "message": "Request failed.",
        "data": {"attribution": ATTRIBUTION},
    }
    assert sent(error("hunter2", code="not-a-number"))["error"]["code"] == -32603


# --- Nothing banned leaves -------------------------------------------------------------------


def test_a_result_holding_a_banned_term_is_withheld() -> None:
    """A success, so the error rewriting above does not apply: only the scan stops it."""
    leaked = {"content": [{"type": "text", "text": "Kansas City Chiefs"}]}

    reply = sent({"jsonrpc": "2.0", "id": 7, "result": leaked})

    assert "Chiefs" not in json.dumps(reply) and "result" not in reply
    assert reply["id"] == 7 and reply["error"]["message"] == "Internal error."


def test_a_clean_result_is_sent_as_it_is() -> None:
    payload = {"jsonrpc": "2.0", "id": "request-7", "result": {"team": "Kansas City"}}

    assert sent(payload) == payload


def test_a_banned_term_in_the_request_id_is_not_repeated() -> None:
    reply = sent({"jsonrpc": "2.0", "id": "Chiefs", "result": {"ok": True}})

    assert reply["id"] is None and "Chiefs" not in json.dumps(reply)


def test_text_that_is_not_json_is_sent_only_if_it_is_clean() -> None:
    assert checked(b"Invalid Host header") == b"Invalid Host header"
    assert checked(b"Not found: Chiefs") == b""
    assert checked(b"") == b""
