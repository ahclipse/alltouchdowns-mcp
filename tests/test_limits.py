"""Rate limiting: the token buckets, who a caller is, and the 429 response."""

from __future__ import annotations

import json
from typing import Any

import pytest
from starlette.types import Message

from alltouchdowns_mcp import logs
from alltouchdowns_mcp.limits import RateLimit, TokenBuckets, client_address, too_many_requests


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


# --- Buckets ---------------------------------------------------------------------------------


def test_a_caller_gets_a_minute_s_allowance_at_once_then_must_wait() -> None:
    clock = Clock()
    buckets = TokenBuckets(per_minute=30, clock=clock)

    assert [buckets.take("a") for _ in range(30)] == [0.0] * 30
    assert buckets.take("a") == pytest.approx(2.0)  # one token every two seconds


def test_tokens_come_back_steadily() -> None:
    clock = Clock()
    buckets = TokenBuckets(per_minute=30, clock=clock)
    for _ in range(30):
        buckets.take("a")

    clock.now += 2
    assert buckets.take("a") == 0.0
    assert buckets.take("a") > 0

    clock.now += 3600  # a long rest refills the bucket, but never beyond its size
    assert [buckets.take("a") for _ in range(30)] == [0.0] * 30
    assert buckets.take("a") > 0


def test_callers_do_not_share_an_allowance() -> None:
    buckets = TokenBuckets(per_minute=1, clock=Clock())

    assert buckets.take("a") == 0.0
    assert buckets.take("a") > 0
    assert buckets.take("b") == 0.0


def test_a_full_table_forgets_idle_callers_but_never_active_ones() -> None:
    clock = Clock()
    buckets = TokenBuckets(per_minute=60, max_keys=2, clock=clock)
    buckets.take("a")
    buckets.take("b")

    # Both callers were just active, so a third is turned away rather than let in for free
    # by evicting someone who is in the middle of their allowance.
    assert buckets.take("c") == 60.0

    clock.now += 61  # a and b have refilled completely; forgetting them loses nothing
    assert buckets.take("c") == 0.0


# --- Who is calling --------------------------------------------------------------------------


def scope(peer: str, **headers: str) -> dict[str, Any]:
    raw = [(name.replace("_", "-").encode(), value.encode()) for name, value in headers.items()]
    return {"type": "http", "client": (peer, 5555), "headers": raw}


@pytest.mark.parametrize("peer", ["203.0.113.9", "172.18.0.5", "127.0.0.1"])
def test_without_a_trusted_proxy_the_connection_is_the_caller(peer: str) -> None:
    """With the setting off the header is never believed, not even from a private address:
    in local development anything on the machine could send it."""
    forged = scope(peer, cf_connecting_ip="198.51.100.1")

    assert client_address(forged, trust_proxy_header=False) == peer


@pytest.mark.parametrize(
    "proxy", ["127.0.0.1", "10.1.2.3", "172.18.0.5", "192.168.1.9", "::1", "fd00::5"]
)
def test_proxy_header_is_believed_from_a_private_peer(proxy: str) -> None:
    through_proxy = scope(proxy, cf_connecting_ip="198.51.100.1")

    assert client_address(through_proxy, trust_proxy_header=True) == "198.51.100.1"


@pytest.mark.parametrize("peer", ["203.0.113.9", "8.8.8.8", "172.32.0.1", "2001:db8::7"])
def test_proxy_header_from_the_open_internet_is_ignored(peer: str) -> None:
    """Anyone who reaches the server directly can write any header they like."""
    direct = scope(peer, cf_connecting_ip="198.51.100.1")

    assert client_address(direct, trust_proxy_header=True) != "198.51.100.1"


def test_missing_header_behind_the_proxy_falls_back_to_the_peer() -> None:
    assert client_address(scope("172.18.0.5"), trust_proxy_header=True) == "172.18.0.5"


def test_ipv6_callers_are_grouped_by_their_64_block() -> None:
    one = scope("172.18.0.5", cf_connecting_ip="2001:db8:1:2:aaaa:bbbb:cccc:dddd")
    same_customer = scope("172.18.0.5", cf_connecting_ip="2001:db8:1:2::1")
    another = scope("172.18.0.5", cf_connecting_ip="2001:db8:1:3::1")

    assert client_address(one, trust_proxy_header=True) == "2001:db8:1:2::/64"
    assert client_address(same_customer, trust_proxy_header=True) == "2001:db8:1:2::/64"
    assert client_address(another, trust_proxy_header=True) != "2001:db8:1:2::/64"


def test_ipv4_written_as_ipv6_is_still_that_ipv4_caller() -> None:
    mapped = scope("172.18.0.5", cf_connecting_ip="::ffff:198.51.100.1")

    assert client_address(mapped, trust_proxy_header=True) == "198.51.100.1"


@pytest.mark.parametrize("garbage", ["not-an-address", "198.51.100.1, 10.0.0.1", "' OR 1=1"])
def test_unparseable_addresses_share_one_bucket(garbage: str) -> None:
    forged = scope("172.18.0.5", cf_connecting_ip=garbage)

    assert client_address(forged, trust_proxy_header=True) == "unknown"


# --- The middleware --------------------------------------------------------------------------


async def call(app: Any, http_scope: dict[str, Any]) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b""}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await app(http_scope, receive, send)
    return sent


@pytest.mark.anyio
async def test_over_the_limit_gets_429_with_retry_after() -> None:
    async def inner(scope: Any, receive: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    buckets = TokenBuckets(per_minute=1, clock=Clock())
    limited = RateLimit(inner, "test", buckets, lambda s: s["client"][0])

    first = await call(limited, scope("203.0.113.9"))
    second = await call(limited, scope("203.0.113.9"))
    someone_else = await call(limited, scope("203.0.113.10"))

    assert first[0]["status"] == 200 and someone_else[0]["status"] == 200
    assert second[0]["status"] == 429
    assert dict(second[0]["headers"])[b"retry-after"] == b"60"
    assert json.loads(second[1]["body"])["error"] == "rate_limited"


@pytest.mark.anyio
@pytest.mark.parametrize(("wait", "header"), [(0.02, "1"), (1.0, "1"), (1.2, "2"), (59.5, "60")])
async def test_retry_after_is_whole_seconds_and_never_zero(wait: float, header: str) -> None:
    """Retry-After: 0 would invite the caller straight back."""
    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    await too_many_requests(send, wait)

    assert dict(sent[0]["headers"])[b"retry-after"] == header.encode()


def test_refusals_are_reported_as_a_count_not_a_line_each(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A flood of refused requests must not become a flood of log lines."""
    now = [1000.0]
    monkeypatch.setattr(logs.time, "monotonic", lambda: now[0])
    refused = logs.RefusalCount("anonymous")

    with caplog.at_level("INFO", logger="alltouchdowns.requests"):
        for _ in range(500):
            refused.add()
        now[0] += logs.SECONDS_BETWEEN_REFUSAL_LINES
        refused.add()

    lines = [json.loads(record.getMessage()) for record in caplog.records]
    assert [(line["event"], line["limit"], line["refused"]) for line in lines] == [
        ("rate_limited", "anonymous", 1),  # the first refusal is reported at once
        ("rate_limited", "anonymous", 500),  # the rest a minute later, as one number
    ]
