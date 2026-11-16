"""Token verification, with keys generated locally. No network, no real provider."""

from __future__ import annotations

import base64
import hashlib
import hmac
import http.server
import json
import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import anyio
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from alltouchdowns_mcp import auth
from alltouchdowns_mcp.auth import AuthSettings, JwtVerifier

SETTINGS = AuthSettings(
    issuer="https://login.example.test",
    jwks_url="https://login.example.test/oauth2/jwks",
    resource_url="https://mcp.example.test/auth/mcp",
)

pytestmark = pytest.mark.anyio


def new_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


PROVIDER_KEY = new_key()


class FakeProvider:
    """Stands in for the provider's published key list. Calling it is one fetch."""

    def __init__(self, **keys: rsa.RSAPrivateKey) -> None:
        self.published: dict[str, Any] = {key_id: key.public_key() for key_id, key in keys.items()}
        self.fetches = 0
        self.failure: Exception | None = None

    def __call__(self) -> dict[str, Any]:
        self.fetches += 1
        if self.failure is not None:
            raise self.failure
        return dict(self.published)


def token(key: rsa.RSAPrivateKey = PROVIDER_KEY, *, kid: str = "key-1", **changes: Any) -> str:
    claims: dict[str, Any] = {
        "iss": SETTINGS.issuer,
        "aud": SETTINGS.resource_url,
        "sub": "user_123",
        "exp": int(time.time()) + 300,
        "scope": "read",
        "client_id": "client_abc",
    }
    claims.update(changes)
    claims = {name: value for name, value in claims.items() if value is not None}
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": kid})


@pytest.fixture
def keys() -> FakeProvider:
    return FakeProvider(**{"key-1": PROVIDER_KEY})


@pytest.fixture
def verifier(keys: FakeProvider) -> JwtVerifier:
    return JwtVerifier(SETTINGS, fetch_keys=keys)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """A clock the test moves by hand: clock[0] += seconds."""
    now = [5000.0]
    monkeypatch.setattr(auth.time, "monotonic", lambda: now[0])
    return now


async def test_a_genuine_token_identifies_its_owner(verifier: JwtVerifier) -> None:
    access = await verifier.verify_token(token())

    assert access is not None
    assert (access.subject, access.client_id, access.scopes) == ("user_123", "client_abc", ["read"])
    assert access.resource == SETTINGS.resource_url


@pytest.mark.parametrize(
    "changes",
    [
        {"exp": int(time.time()) - 600},  # expired
        {"aud": "https://some-other-api.example.test"},  # issued for a different server
        {"iss": "https://evil.example.test"},  # issued by someone else
        {"sub": None},  # says nothing about who it is
        {"exp": None},  # never expires
        {"aud": None},
    ],
)
async def test_tokens_that_must_be_refused(verifier: JwtVerifier, changes: dict[str, Any]) -> None:
    assert await verifier.verify_token(token(**changes)) is None


async def test_a_token_signed_by_someone_else_is_refused(verifier: JwtVerifier) -> None:
    forged = token(new_key())  # right key id, wrong key

    assert await verifier.verify_token(forged) is None


async def test_an_unsigned_token_is_refused(verifier: JwtVerifier) -> None:
    claims = {"iss": SETTINGS.issuer, "aud": SETTINGS.resource_url, "sub": "x",
              "exp": int(time.time()) + 300}  # fmt: skip
    unsigned = jwt.encode(claims, key="", algorithm="none", headers={"kid": "key-1"})

    assert await verifier.verify_token(unsigned) is None


async def test_the_public_key_cannot_be_used_as_a_shared_secret(verifier: JwtVerifier) -> None:
    """The classic algorithm-confusion attack: sign with HS256 using the server's PUBLIC key
    as the secret, and hope the server verifies with whatever algorithm the token names.
    The token is built by hand because PyJWT itself refuses to create one."""
    public_pem = PROVIDER_KEY.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )

    def part(data: dict[str, Any]) -> bytes:
        return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=")

    claims = {"iss": SETTINGS.issuer, "aud": SETTINGS.resource_url, "sub": "x",
              "exp": int(time.time()) + 300}  # fmt: skip
    signed = part({"alg": "HS256", "typ": "JWT", "kid": "key-1"}) + b"." + part(claims)
    signature = hmac.new(public_pem, signed, hashlib.sha256).digest()
    forged = (signed + b"." + base64.urlsafe_b64encode(signature).rstrip(b"=")).decode()

    assert await verifier.verify_token(forged) is None


@pytest.mark.parametrize("garbage", ["", "not.a.token", "a.b.c", "Bearer abc"])
async def test_garbage_is_refused_without_an_error(verifier: JwtVerifier, garbage: str) -> None:
    assert await verifier.verify_token(garbage) is None


async def test_a_token_without_a_key_id_is_refused(verifier: JwtVerifier) -> None:
    claims = {"iss": SETTINGS.issuer, "aud": SETTINGS.resource_url, "sub": "x",
              "exp": int(time.time()) + 300}  # fmt: skip

    assert await verifier.verify_token(jwt.encode(claims, PROVIDER_KEY, algorithm="RS256")) is None


async def test_keys_are_fetched_once_and_then_remembered(
    verifier: JwtVerifier, keys: FakeProvider
) -> None:
    for _ in range(5):
        assert await verifier.verify_token(token()) is not None

    assert keys.fetches == 1


async def test_unknown_key_ids_cannot_trigger_a_fetch_storm(
    verifier: JwtVerifier, keys: FakeProvider, clock: list[float]
) -> None:
    await verifier.verify_token(token())

    for number in range(20):  # forged tokens, each naming a key that does not exist
        assert await verifier.verify_token(token(new_key(), kid=f"made-up-{number}")) is None
        clock[0] += 1
    assert keys.fetches == 1  # 20 seconds, 20 unknown ids, no extra fetch

    clock[0] += 60
    await verifier.verify_token(token(new_key(), kid="made-up-later"))
    assert keys.fetches == 2  # at most one a minute


async def test_a_rotated_key_is_picked_up(
    verifier: JwtVerifier, keys: FakeProvider, clock: list[float]
) -> None:
    await verifier.verify_token(token())

    rotated = new_key()
    keys.published["key-2"] = rotated.public_key()
    clock[0] += 120

    assert await verifier.verify_token(token(rotated, kid="key-2")) is not None


async def test_a_withdrawn_key_stops_being_trusted_within_the_hour(
    verifier: JwtVerifier, keys: FakeProvider, clock: list[float]
) -> None:
    assert await verifier.verify_token(token()) is not None
    del keys.published["key-1"]
    keys.published["key-2"] = new_key().public_key()

    clock[0] += auth.KEY_LIST_LIFETIME_SECONDS - 1
    assert await verifier.verify_token(token()) is not None  # the list is not re-read yet
    clock[0] += 1
    assert await verifier.verify_token(token()) is None
    assert keys.fetches == 2


async def test_when_the_provider_is_down_the_keys_already_held_keep_working(
    verifier: JwtVerifier, keys: FakeProvider, clock: list[float]
) -> None:
    assert await verifier.verify_token(token()) is not None
    keys.failure = TimeoutError()
    clock[0] += auth.KEY_LIST_LIFETIME_SECONDS

    for _ in range(5):
        assert await verifier.verify_token(token()) is not None
    assert keys.fetches == 2  # one failed attempt, not one for every request

    clock[0] += auth.MIN_SECONDS_BETWEEN_KEY_FETCHES
    assert await verifier.verify_token(token()) is not None
    assert keys.fetches == 3


@pytest.mark.parametrize(
    "failure", [TimeoutError(), ConnectionResetError(), ValueError("not JSON")]
)
async def test_a_failed_first_fetch_refuses_the_token_instead_of_crashing(
    verifier: JwtVerifier, keys: FakeProvider, failure: Exception
) -> None:
    keys.failure = failure

    assert await verifier.verify_token(token()) is None
    assert await verifier.verify_token(token()) is None
    assert keys.fetches == 1  # and a failure is not retried at once either


async def test_a_key_this_server_cannot_use_refuses_the_token_instead_of_crashing(
    verifier: JwtVerifier, keys: FakeProvider
) -> None:
    """Not every failure is one of the JWT library's own errors: this one is a TypeError."""
    keys.published["key-1"] = 12345

    assert await verifier.verify_token(token()) is None


async def test_requests_arriving_together_share_one_fetch(keys: FakeProvider) -> None:
    def slow_fetch() -> dict[str, Any]:
        time.sleep(0.05)
        return keys()

    verifier = JwtVerifier(SETTINGS, fetch_keys=slow_fetch)
    results: list[object] = []

    async def verify() -> None:
        results.append(await verifier.verify_token(token()))

    async with anyio.create_task_group() as group:
        for _ in range(10):
            group.start_soon(verify)

    assert keys.fetches == 1
    assert len(results) == 10 and all(results)  # the nine that waited were not refused


# --- The real fetch, against a local web server ----------------------------------------------


@pytest.fixture
def provider_url() -> Iterator[tuple[str, dict[str, bytes]]]:
    """A local web server standing in for the provider's key endpoint. Returns its URL and a
    dict whose "body" entry is what it serves."""
    served = {"body": b""}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(served["body"])

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/jwks", served
    server.shutdown()
    server.server_close()


async def test_keys_are_read_from_the_provider_s_published_document(
    provider_url: tuple[str, dict[str, bytes]],
) -> None:
    url, served = provider_url
    jwk = RSAAlgorithm.to_jwk(PROVIDER_KEY.public_key(), as_dict=True)
    published = {"keys": [{**jwk, "kid": "key-1", "use": "sig", "alg": "RS256"}]}
    served["body"] = json.dumps(published).encode()
    verifier = JwtVerifier(replace(SETTINGS, jwks_url=url))

    assert await verifier.verify_token(token()) is not None
    assert await verifier.verify_token(token(new_key())) is None


async def test_an_error_page_instead_of_keys_refuses_the_token(
    provider_url: tuple[str, dict[str, bytes]],
) -> None:
    url, served = provider_url
    served["body"] = b"<html>Service unavailable</html>"

    assert await JwtVerifier(replace(SETTINGS, jwks_url=url)).verify_token(token()) is None
