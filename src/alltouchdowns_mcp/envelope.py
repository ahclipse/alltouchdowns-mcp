"""The outermost layer around the MCP endpoint: what goes in must be a POST, and whatever
comes out has been checked.

Tool results are finalised by `guard.py`. This layer covers everything else the endpoint can
say: protocol errors (an unknown method, a malformed request, a bad resource name) are written
by the SDK and quote the caller's own text back. Here every error object is replaced by a
fixed sentence, and the whole response body is scanned for banned terms before it is sent. So
no path through the endpoint, known or not yet thought of, can echo input or emit a banned term.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from alltouchdowns_mcp import ATTRIBUTION, legal

logger = logging.getLogger(__name__)

# JSON-RPC error codes and the only thing this server says for each.
_MESSAGES = {
    -32700: "Parse error.",
    -32600: "Invalid request.",
    -32601: "Method not found.",
    -32602: "Invalid params.",
    -32603: "Internal error.",
}
_OTHER = "Request failed."

MCP_PATH = "/mcp"


class CheckedEndpoint:
    """ASGI wrapper for the SDK's MCP app."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        # The path as this app sees it, with the mount point removed.
        if scope["path"].removeprefix(scope.get("root_path", "")) != MCP_PATH:
            await _respond(send, 404, {"error": "Not found."}, [])
            return
        if scope["method"] != "POST":
            # The SDK would answer a GET from an older client by opening a stream that never
            # sends anything and never closes. This server has nothing to stream, so refuse.
            await _respond(send, 405, {"error": "Method not allowed."}, [(b"allow", b"POST")])
            return

        # Responses are single JSON bodies (the app runs with json_response=True), so the
        # whole body can be held back, checked and then sent.
        start: Message | None = None
        chunks: list[bytes] = []

        async def hold(message: Message) -> None:
            nonlocal start
            if message["type"] == "http.response.start":
                start = message
                return
            chunks.append(message.get("body", b""))
            if not message.get("more_body"):
                assert start is not None
                body = checked(b"".join(chunks))
                headers = [(k, v) for k, v in start.get("headers", []) if k != b"content-length"]
                headers.append((b"content-length", str(len(body)).encode()))
                await send({**start, "headers": headers})
                await send({"type": "http.response.body", "body": body})

        await self.app(scope, receive, hold)


def checked(body: bytes) -> bytes:
    """The body that may be sent: error objects rewritten, and nothing banned in it."""
    if not body:
        return body
    try:
        payload = json.loads(body)
    except ValueError:
        # Not JSON (for example a plain-text "Invalid Host header"). Short fixed texts written
        # by the SDK are fine; anything carrying a banned term is not.
        return body if legal.is_clean(body.decode("utf-8", "replace")) else b""

    payload = _plain_errors(payload)
    text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    if legal.find_banned(text):
        request_id = payload.get("id") if isinstance(payload, dict) else None
        if isinstance(request_id, str) and not legal.is_clean(request_id):
            # A response repeats the request's id, and this caller put a banned term in it.
            # Their doing, not a leak from the data, so it is not logged as an error. The id
            # cannot be repeated; JSON-RPC allows null for an id that cannot be returned.
            logger.warning("response withheld: the request id contained a banned term")
            request_id = None
        else:
            logger.error("response withheld: it contained a banned term")
        text = json.dumps(_error_object(request_id, -32603))
    return text.encode()


def _plain_errors(payload: Any) -> Any:
    """Replace every JSON-RPC error with a fixed message. The SDK's own messages and `data`
    repeat what the caller sent, such as the name of the method it did not find."""
    if isinstance(payload, list):
        return [_plain_errors(item) for item in payload]
    if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
        code = payload["error"].get("code")
        return _error_object(payload.get("id"), code if isinstance(code, int) else -32603)
    return payload


def _error_object(request_id: Any, code: int) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": code,
            "message": _MESSAGES.get(code, _OTHER),
            "data": {"attribution": ATTRIBUTION},
        },
    }


async def _respond(
    send: Send, status: int, payload: dict[str, Any], extra_headers: list[tuple[bytes, bytes]]
) -> None:
    body = json.dumps(payload).encode()
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
        *extra_headers,
    ]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})
