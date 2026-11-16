"""Rate limiting.

A token bucket per caller: each caller has a bucket that holds up to a minute's allowance and
refills steadily. A request takes one token; an empty bucket means 429 with a Retry-After
header saying when the next token arrives.

Everything is kept in memory in this one process. That is deliberate: the server runs as a
single worker, so the count is exact, and there is no second service to operate. Running
several workers would need a shared store such as Redis.
"""

from __future__ import annotations

import ipaddress
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from starlette.types import ASGIApp, Receive, Scope, Send

from alltouchdowns_mcp.logs import RefusalCount

MAX_CALLERS = 10_000


@dataclass
class _Bucket:
    tokens: float
    updated: float


class TokenBuckets:
    """One bucket per key, each allowing `per_minute` requests a minute."""

    def __init__(
        self,
        per_minute: float,
        *,
        burst: float | None = None,
        max_keys: int = MAX_CALLERS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        # A bucket holds a whole minute's allowance unless a smaller burst is asked for.
        self.capacity = float(per_minute if burst is None else burst)
        self.refill_per_second = per_minute / 60
        self._max_keys = max_keys
        self._clock = clock
        self._buckets: dict[str, _Bucket] = {}

    def take(self, key: str) -> float:
        """Take one token for `key`. Returns 0 if allowed, else seconds until a token is free."""
        now = self._clock()
        bucket = self._buckets.get(key)
        if bucket is None:
            if len(self._buckets) >= self._max_keys and not self._forget_idle_callers(now):
                return 60.0  # the table is full of active callers: turn new ones away
            bucket = self._buckets[key] = _Bucket(self.capacity, now)

        elapsed = now - bucket.updated
        bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.refill_per_second)
        bucket.updated = now
        if bucket.tokens >= 1:
            bucket.tokens -= 1
            return 0.0
        return (1 - bucket.tokens) / self.refill_per_second

    def _forget_idle_callers(self, now: float) -> bool:
        """Drop buckets that have refilled completely. Forgetting a full bucket loses nothing:
        a caller who returns simply gets a new full one."""
        seconds_to_refill = self.capacity / self.refill_per_second
        idle = [k for k, b in self._buckets.items() if now - b.updated >= seconds_to_refill]
        for key in idle:
            del self._buckets[key]
        return bool(idle)


class RateLimit:
    """ASGI middleware: charge each request to the caller `key_of` names. `name` says which
    limit this is in the log."""

    def __init__(
        self, app: ASGIApp, name: str, buckets: TokenBuckets, key_of: Callable[[Scope], str]
    ) -> None:
        self.app = app
        self.buckets = buckets
        self.key_of = key_of
        self.refused = RefusalCount(name)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        wait = self.buckets.take(self.key_of(scope))
        if wait:
            self.refused.add()
            await too_many_requests(send, wait)
            return
        await self.app(scope, receive, send)


class FailedSignInLimit:
    """ASGI middleware for the signed-in endpoint: slow down a caller whose tokens keep being
    refused.

    It sits just inside token verification, so it knows the outcome. A verified caller passes
    untouched, whatever anyone else at the same address is doing. So does a request with no
    token at all: that is how every client begins, and the 401 it gets is what starts sign-in.
    A request whose token was refused takes from its address's bucket, and once the bucket is
    empty gets 429 instead of 401.

    Checking a token is cheap and needs no network, so this is not here to save work. It tells
    a client stuck retrying a dead token to back off rather than loop.
    """

    def __init__(self, app: ASGIApp, buckets: TokenBuckets, key_of: Callable[[Scope], str]) -> None:
        self.app = app
        self.buckets = buckets
        self.key_of = key_of
        self.refused = RefusalCount("failed_sign_in")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and _token_was_refused(scope):
            wait = self.buckets.take(self.key_of(scope))
            if wait:
                self.refused.add()
                await too_many_requests(send, wait)
                return
        await self.app(scope, receive, send)


def _token_was_refused(scope: Scope) -> bool:
    """A token was presented, and verification (which has already run) did not accept it."""
    presented = any(name == b"authorization" for name, _ in scope.get("headers") or [])
    return presented and not scope["user"].is_authenticated


async def too_many_requests(send: Send, wait_seconds: float) -> None:
    retry_after = str(max(1, math.ceil(wait_seconds)))
    body = json.dumps(
        {"error": "rate_limited", "message": f"Too many requests. Retry in {retry_after} seconds."}
    ).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 429,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"retry-after", retry_after.encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def client_address(scope: Scope, *, trust_proxy_header: bool) -> str:
    """Who is calling, for rate limiting only. Never logged.

    Behind the production proxy chain every connection comes from the proxy, and the real
    address is in the CF-Connecting-IP header. That header is believed only when the setting
    says a trusted proxy is in front AND the connection itself comes from a private address,
    because anyone who can reach the server directly could otherwise forge it.

    IPv6 callers are grouped by /64, the smallest block an ISP hands to one customer, so one
    customer cannot multiply their allowance by rotating addresses.
    """
    peer = (scope.get("client") or ("unknown", 0))[0]
    address = peer
    if trust_proxy_header and _is_private(peer):
        headers = dict(scope.get("headers") or [])
        forwarded = headers.get(b"cf-connecting-ip", b"").decode("latin-1").strip()
        if forwarded:
            address = forwarded
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return "unknown"
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped:
            return str(ip.ipv4_mapped)
        return str(ipaddress.ip_network(f"{ip}/64", strict=False).network_address) + "/64"
    return str(ip)


# Where a trusted reverse proxy on the same host or Docker network connects from.
_PROXY_NETWORKS = [
    ipaddress.ip_network(block)
    for block in (
        "127.0.0.0/8",
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "::1/128",
        "fc00::/7",
    )
]


def _is_private(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    return any(ip in network for network in _PROXY_NETWORKS)
