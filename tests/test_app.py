"""The web application over real HTTP: a server on a local port, as in production."""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import replace
from email.message import Message
from http.client import HTTPConnection
from pathlib import Path

import pytest
import uvicorn
from mcp import Client
from mcp.server.auth.provider import AccessToken
from starlette.requests import Request
from starlette.responses import JSONResponse

from alltouchdowns_mcp import ATTRIBUTION, logs
from alltouchdowns_mcp import app as app_module
from alltouchdowns_mcp import db as db_module
from alltouchdowns_mcp.__main__ import server_config
from alltouchdowns_mcp.app import MAX_REQUEST_BYTES, HealthCheck, create_app
from alltouchdowns_mcp.auth import AuthSettings
from alltouchdowns_mcp.config import Settings
from tests.conftest import FIXTURE_DB

# The fixture was exported once, long ago by the time most tests run, so "fresh" has to be
# generous here. Staleness itself is tested separately.
SETTINGS = Settings(db_path=FIXTURE_DB, stale_after_hours=24 * 365 * 100, rate_limits=False)


@pytest.fixture(scope="module")
def base_url() -> Iterator[str]:
    """Run the real app with uvicorn on a free local port for the tests in this module."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(create_app(SETTINGS), host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "server did not start"
        time.sleep(0.02)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


def http(url: str, *, data: bytes | None = None, headers: dict[str, str] | None = None):
    """Make a request and return (status, body), whether or not the status is an error."""
    request = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


# --- MCP over HTTP ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_a_real_client_can_list_and_call_tools(base_url: str) -> None:
    async with Client(f"{base_url}/mcp") as client:
        tools = (await client.list_tools()).tools
        result = await client.call_tool("get_leaderboard", {"season": 2007, "limit": 1})

    assert len(tools) == 6
    assert result.structured_content is not None
    assert result.structured_content["rows"][0]["player"]["slug"] == "randy-moss"


def test_foreign_host_header_is_rejected(base_url: str) -> None:
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).encode()
    headers = {"Content-Type": "application/json", "Host": "evil.example"}

    status, _ = http(f"{base_url}/mcp", data=body, headers=headers)

    assert status == 421


def test_oversized_request_is_rejected(base_url: str) -> None:
    """The server refuses on the size a request declares, before any of the body arrives.

    So the test declares a body one byte over the limit and sends none of it. (Sending it
    would race with the refusal: a server that closes on unread data resets the connection,
    and the client may then lose the very response being tested.)
    """
    host, port = base_url.removeprefix("http://").split(":")
    connection = HTTPConnection(host, int(port), timeout=5)
    try:
        connection.putrequest("POST", "/mcp")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Accept", "application/json, text/event-stream")
        connection.putheader("Content-Length", str(MAX_REQUEST_BYTES + 1))
        connection.endheaders()

        assert connection.getresponse().status == 413
    finally:
        connection.close()


def test_a_request_at_the_size_limit_is_accepted(base_url: str) -> None:
    padding = "x" * (MAX_REQUEST_BYTES - 200)
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "pad": padding}).encode()
    assert len(body) <= MAX_REQUEST_BYTES
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}

    status, _ = http(f"{base_url}/mcp", data=body, headers=headers)

    assert status == 200


def test_unknown_path_is_not_found(base_url: str) -> None:
    assert http(f"{base_url}/admin")[0] == 404


# --- Health check ----------------------------------------------------------------------------


def test_healthz_reports_the_export_it_is_serving(base_url: str) -> None:
    status, body = http(f"{base_url}/healthz")

    report = json.loads(body)
    assert status == 200
    assert report["status"] == "ok" and report["data_through"] == "2020-02-02"
    assert report["exported_at"].endswith("Z")


async def health(settings: Settings) -> JSONResponse:
    request = Request({"type": "http", "method": "GET", "headers": []})
    return await HealthCheck(settings).respond(request)


@pytest.mark.anyio
async def test_healthz_fails_once_the_export_is_stale() -> None:
    response = await health(replace(SETTINGS, stale_after_hours=0))

    assert response.status_code == 503
    assert json.loads(bytes(response.body))["status"] == "stale"


@pytest.mark.anyio
async def test_healthz_fails_when_the_database_is_missing(tmp_path: Path) -> None:
    response = await health(replace(SETTINGS, db_path=tmp_path / "gone.db"))

    assert response.status_code == 503
    assert json.loads(bytes(response.body)) == {"status": "unavailable"}


@pytest.mark.anyio
async def test_healthz_answers_even_when_every_query_slot_is_busy() -> None:
    slots = db_module._QUERY_SLOTS
    held = [slots.acquire(timeout=1) for _ in range(4)]
    try:
        response = await health(SETTINGS)
    finally:
        for _ in held:
            slots.release()

    assert all(held) and response.status_code == 200


@pytest.mark.anyio
async def test_healthz_reads_the_database_rarely_however_often_it_is_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads = []
    real = app_module.read_meta
    monkeypatch.setattr(app_module, "read_meta", lambda path: reads.append(path) or real(path))
    check = HealthCheck(SETTINGS)
    request = Request({"type": "http", "method": "GET", "headers": []})

    statuses = [(await check.respond(request)).status_code for _ in range(50)]

    assert statuses == [200] * 50 and len(reads) == 1


# --- Settings --------------------------------------------------------------------------------


def test_settings_from_environment() -> None:
    settings = Settings.from_env(
        {
            "ATD_DB_PATH": "/data/mcp.db",
            "ATD_HOST": "0.0.0.0",
            "ATD_PORT": "9000",
            "ATD_ALLOWED_HOSTS": "mcp.example.com, mcp.example.com:*",
            "ATD_QUERY_STEP_LIMIT": "100000",
        }
    )

    assert settings.db_path == Path("/data/mcp.db")
    assert (settings.host, settings.port, settings.query_step_limit) == ("0.0.0.0", 9000, 100000)
    assert settings.allowed_hosts == ("mcp.example.com", "mcp.example.com:*")
    assert settings.allowed_origins  # falls back to the local defaults


def test_only_the_database_path_is_required() -> None:
    assert Settings.from_env({"ATD_DB_PATH": "x.db"}).port == 8000
    with pytest.raises(KeyError):
        Settings.from_env({})


# --- Review findings: methods, protocol errors -----------------------------------------------

JSON_HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}


def rpc(base_url: str, payload: object, **headers: str) -> tuple[int, dict]:
    status, body = http(f"{base_url}/mcp", data=json.dumps(payload).encode(),
                        headers={**JSON_HEADERS, **headers})  # fmt: skip
    return status, json.loads(body) if body else {}


@pytest.mark.parametrize("method", ["GET", "DELETE", "PUT"])
def test_only_post_is_served_on_the_mcp_path(base_url: str, method: str) -> None:
    """A GET from an older client would otherwise open a stream that never ends."""
    request = urllib.request.Request(f"{base_url}/mcp", method=method)
    with pytest.raises(urllib.error.HTTPError) as refused:
        urllib.request.urlopen(request, timeout=5)

    assert refused.value.code == 405
    assert refused.value.headers["Allow"] == "POST"


@pytest.mark.parametrize(
    "payload",
    [
        {"jsonrpc": "2.0", "id": 1, "method": "Super Bowl Chiefs"},
        {"jsonrpc": "2.0", "id": 1, "method": "prompts/get",
         "params": {"name": "Chiefs Super Bowl"}},
        {"jsonrpc": "2.0", "id": 1, "method": "resources/read",
         "params": {"uri": "file:///Chiefs/Super%20Bowl"}},
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": "Chiefs"},
        {"not": "a request", "Chiefs": "Super Bowl"},
    ],
)  # fmt: skip
def test_protocol_errors_say_nothing_the_caller_sent(base_url: str, payload: dict) -> None:
    status, reply = rpc(base_url, payload)

    text = json.dumps(reply)
    assert "Chiefs" not in text and "Super" not in text and "pydantic" not in text
    assert status in (200, 400, 404)
    if "error" in reply:
        assert reply["error"]["message"].endswith(".")
        assert reply["error"]["data"] == {"attribution": ATTRIBUTION}


@pytest.mark.parametrize("method", ["tools/list", "no/such/method"])
def test_a_banned_term_in_the_request_id_is_not_repeated(base_url: str, method: str) -> None:
    """A response repeats the id of its request, so the id is caller-written text too."""
    _, reply = rpc(base_url, {"jsonrpc": "2.0", "id": "Super Bowl Chiefs", "method": method})

    assert "Super" not in json.dumps(reply) and "Chiefs" not in json.dumps(reply)
    assert reply["id"] is None and reply["error"]["message"] == "Internal error."


def test_an_ordinary_request_id_is_repeated_unchanged(base_url: str) -> None:
    for request_id in ("request-7", 7):
        _, reply = rpc(base_url, {"jsonrpc": "2.0", "id": request_id, "method": "tools/list"})

        assert reply["id"] == request_id and "result" in reply


@pytest.mark.parametrize(
    "payload",
    [
        {"jsonrpc": "2.0", "id": 1, "method": "xyzzy-echo"},
        {"jsonrpc": "2.0", "id": 1, "method": "prompts/get", "params": {"name": "hunter2"}},
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
         "params": {"name": "hunter2", "arguments": {"xyzzy": 1}}},
    ],
)  # fmt: skip
def test_errors_repeat_nothing_the_caller_sent_banned_or_not(base_url: str, payload: dict) -> None:
    """The banned-term scan would not catch these words; only fixed error messages do."""
    _, reply = rpc(base_url, payload)

    assert "xyzzy" not in json.dumps(reply) and "hunter2" not in json.dumps(reply)


def test_invalid_json_gets_a_plain_error(base_url: str) -> None:
    status, body = http(f"{base_url}/mcp", data=b'{"Chiefs": ', headers=JSON_HEADERS)

    assert status == 400
    assert b"Chiefs" not in body


# --- Rate limits -----------------------------------------------------------------------------


@pytest.fixture
def serve():
    """Start an app for one test and stop it afterwards."""
    servers = []

    def start(settings: Settings, **kwargs) -> str:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        app = create_app(replace(settings, port=port), **kwargs)
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        while not server.started:
            time.sleep(0.02)
        servers.append((server, thread))
        return f"http://127.0.0.1:{port}"

    yield start
    for server, thread in servers:
        server.should_exit = True
        thread.join(timeout=5)


LIST_TOOLS = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}


def test_anonymous_callers_are_limited_per_address(serve) -> None:
    url = serve(replace(SETTINGS, rate_limits=True, anonymous_per_minute=3))

    statuses = [rpc(url, LIST_TOOLS)[0] for _ in range(5)]

    assert statuses == [200, 200, 200, 429, 429]
    request = urllib.request.Request(f"{url}/mcp", data=b"{}", headers=JSON_HEADERS)
    with pytest.raises(urllib.error.HTTPError) as limited:
        urllib.request.urlopen(request, timeout=5)
    assert int(limited.value.headers["Retry-After"]) >= 1


def test_health_check_is_never_rate_limited(serve) -> None:
    url = serve(replace(SETTINGS, rate_limits=True, anonymous_per_minute=1))

    assert [http(f"{url}/healthz")[0] for _ in range(5)] == [200] * 5


def test_everyone_together_has_a_ceiling(serve) -> None:
    url = serve(replace(SETTINGS, rate_limits=True, anonymous_per_minute=1000, total_per_second=2))

    statuses = [rpc(url, LIST_TOOLS)[0] for _ in range(8)]

    assert statuses[:4] == [200] * 4  # a burst of twice the per-second rate
    assert 429 in statuses[4:]


def test_rate_limit_refusals_are_logged_as_a_count(serve, caplog) -> None:
    url = serve(replace(SETTINGS, rate_limits=True, anonymous_per_minute=2))

    with caplog.at_level("INFO", logger="alltouchdowns.requests"):
        statuses = [rpc(url, LIST_TOOLS)[0] for _ in range(6)]

    lines = [json.loads(record.getMessage()) for record in caplog.records]
    assert statuses.count(429) == 4
    assert [(line["event"], line["limit"], line["refused"]) for line in lines] == [
        ("rate_limited", "anonymous", 1)  # one line, not four; the rest are counted for the next
    ]
    assert "127.0.0.1" not in caplog.text


def test_one_busy_caller_cannot_lock_everyone_else_out(serve) -> None:
    """A request refused by its own caller's limit must not count against the total."""
    settings = replace(SETTINGS, rate_limits=True, trust_proxy_header=True,
                       anonymous_per_minute=3, total_per_second=5)  # fmt: skip
    url = serve(settings)

    def call(address: str) -> int:
        return rpc(url, LIST_TOOLS, **{"CF-Connecting-IP": address})[0]

    busy = [call("203.0.113.5") for _ in range(40)]

    assert busy.count(200) == 3 and busy.count(429) == 37
    assert [call(f"198.51.100.{n}") for n in range(1, 6)] == [200] * 5


# --- The signed-in endpoint ------------------------------------------------------------------

AUTH = AuthSettings(
    issuer="https://login.example.test",
    jwks_url="https://login.example.test/oauth2/jwks",
    resource_url="https://mcp.example.test/auth/mcp",
)


class StubVerifier:
    """Accepts tokens of the form "good-<user>"."""

    async def verify_token(self, token: str) -> AccessToken | None:
        if not token.startswith("good-"):
            return None
        return AccessToken(token=token, client_id="client", scopes=[], subject=token,
                           resource=AUTH.resource_url)  # fmt: skip


def signed_in(url: str, token: str | None) -> tuple[int, Message, dict]:
    headers = dict(JSON_HEADERS)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(f"{url}/auth/mcp", data=json.dumps(LIST_TOOLS).encode(),
                                     headers=headers)  # fmt: skip
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.headers, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, error.headers, {}  # header lookup on these ignores case


def test_without_oauth_settings_there_is_no_signed_in_endpoint(base_url: str) -> None:
    assert http(f"{base_url}/auth/mcp", data=b"{}", headers=JSON_HEADERS)[0] == 404
    assert http(f"{base_url}/.well-known/oauth-protected-resource")[0] == 404


def test_signed_in_endpoint_demands_a_token_and_says_where_to_get_one(serve) -> None:
    url = serve(replace(SETTINGS, auth=AUTH), token_verifier=StubVerifier())

    status, headers, _ = signed_in(url, None)

    assert status == 401
    challenge = headers["WWW-Authenticate"]
    assert challenge.startswith("Bearer ")
    # Built from settings, never from the request's Host header.
    expected = "https://mcp.example.test/.well-known/oauth-protected-resource/auth/mcp"
    assert f'resource_metadata="{expected}"' in challenge


def test_a_bad_token_is_refused_and_a_good_one_reaches_the_tools(serve) -> None:
    url = serve(replace(SETTINGS, auth=AUTH), token_verifier=StubVerifier())

    assert signed_in(url, "forged")[0] == 401
    status, _, reply = signed_in(url, "good-alice")
    assert status == 200
    assert len(reply["result"]["tools"]) == 6


def test_anonymous_endpoint_still_needs_no_token(serve) -> None:
    url = serve(replace(SETTINGS, auth=AUTH), token_verifier=StubVerifier())

    assert rpc(url, LIST_TOOLS)[0] == 200


@pytest.mark.parametrize(
    "path",
    ["/.well-known/oauth-protected-resource/auth/mcp", "/.well-known/oauth-protected-resource"],
)
def test_metadata_names_the_resource_and_the_authorization_server(serve, path: str) -> None:
    url = serve(replace(SETTINGS, auth=AUTH), token_verifier=StubVerifier())

    status, body = http(url + path)

    metadata = json.loads(body)
    assert status == 200
    assert metadata["resource"] == "https://mcp.example.test/auth/mcp"
    assert metadata["authorization_servers"] == ["https://login.example.test/"]
    assert metadata["bearer_methods_supported"] == ["header"]


def test_signed_in_users_each_have_their_own_allowance(serve) -> None:
    settings = replace(SETTINGS, auth=AUTH, rate_limits=True, signed_in_per_minute=2)
    url = serve(settings, token_verifier=StubVerifier())

    alice = [signed_in(url, "good-alice")[0] for _ in range(3)]
    bob = signed_in(url, "good-bob")[0]

    assert alice == [200, 200, 429]
    assert bob == 200  # same address, different user


def test_repeated_bad_tokens_are_slowed_down_without_touching_valid_users(serve) -> None:
    settings = replace(SETTINGS, auth=AUTH, rate_limits=True, failed_sign_in_per_minute=3)
    url = serve(settings, token_verifier=StubVerifier())

    assert [signed_in(url, "good-alice")[0] for _ in range(5)] == [200] * 5  # never charged
    assert [signed_in(url, "forged")[0] for _ in range(5)] == [401, 401, 401, 429, 429]
    # From the same address, straight afterwards:
    assert signed_in(url, "good-alice")[0] == 200  # a valid token is never held back
    assert signed_in(url, None)[0] == 401  # no token is how sign-in starts; never counted


def test_callers_without_a_valid_token_cannot_lock_signed_in_users_out(serve) -> None:
    settings = replace(SETTINGS, auth=AUTH, rate_limits=True, total_per_second=2)
    url = serve(settings, token_verifier=StubVerifier())

    assert {signed_in(url, None)[0] for _ in range(20)} == {401}
    assert {signed_in(url, "forged")[0] for _ in range(20)} == {401}

    assert signed_in(url, "good-alice")[0] == 200  # the total allowance was not touched


def test_oauth_settings_come_as_a_set() -> None:
    env = {"ATD_DB_PATH": "x.db", "ATD_AUTH_ISSUER": "https://login.example.test"}

    with pytest.raises(ValueError, match="all of"):
        Settings.from_env(env)
    complete = {**env, "ATD_AUTH_JWKS_URL": "https://login.example.test/jwks",
                "ATD_AUTH_RESOURCE_URL": "https://mcp.example.test/auth/mcp",
                "ATD_RATE_LIMITS": "off", "ATD_TRUST_PROXY_HEADER": "on"}  # fmt: skip
    settings = Settings.from_env(complete)
    assert settings.auth is not None and settings.auth.issuer == "https://login.example.test"
    assert settings.rate_limits is False and settings.trust_proxy_header is True


# --- Request log -----------------------------------------------------------------------------


def test_request_log_says_what_happened_and_nothing_about_who(
    serve, caplog: pytest.LogCaptureFixture
) -> None:
    url = serve(replace(SETTINGS, auth=AUTH), token_verifier=StubVerifier())
    arguments = {"season": 2007, "limit": 1}
    call = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "get_leaderboard", "arguments": arguments}}  # fmt: skip
    bad = {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
           "params": {"name": "Chiefs", "arguments": {"secret": "hunter2"}}}  # fmt: skip

    with caplog.at_level("INFO", logger="alltouchdowns.requests"):
        rpc(url, call)
        rpc(url, bad)
        request = urllib.request.Request(
            f"{url}/auth/mcp", data=json.dumps(call).encode(),
            headers={**JSON_HEADERS, "Authorization": "Bearer good-alice"},
        )  # fmt: skip
        urllib.request.urlopen(request, timeout=5).read()

    lines = [json.loads(record.getMessage()) for record in caplog.records]
    assert [(line["tool"], line["signed_in"], line["outcome"]) for line in lines] == [
        ("get_leaderboard", False, "ok"),
        ("unknown", False, "unknown_tool"),
        ("get_leaderboard", True, "ok"),
    ]
    logged = " ".join(record.getMessage() for record in caplog.records)
    for secret in ("127.0.0.1", "good-alice", "hunter2", "Chiefs", "2007"):
        assert secret not in logged
    assert all(set(line) == {"at", "event", "tool", "signed_in", "outcome", "ms"} for line in lines)


# --- Logging rules ---------------------------------------------------------------------------


@pytest.fixture
def production_logging() -> Iterator[None]:
    """Apply the logging set-up of the real entry point, and undo it afterwards."""
    requests = logging.getLogger("alltouchdowns.requests")
    sdk = logging.getLogger("mcp.server.transport_security")
    before = (requests.handlers[:], requests.level, requests.propagate, sdk.level)
    logs.configure()
    yield
    requests.handlers[:] = before[0]
    requests.setLevel(before[1])
    requests.propagate = before[2]
    sdk.setLevel(before[3])


def test_a_refused_host_or_origin_header_is_not_written_to_the_log(
    base_url: str, production_logging: None, caplog: pytest.LogCaptureFixture
) -> None:
    """The SDK would log both headers word for word, and a header is the caller's text."""
    body = json.dumps(LIST_TOOLS).encode()

    def post(**header: str) -> int:
        return http(f"{base_url}/mcp", data=body, headers={**JSON_HEADERS, **header})[0]

    with caplog.at_level(logging.DEBUG):
        host = post(Host="chiefs-hunter2.example")
        origin = post(Origin="http://Chiefs-hunter2.example")

    assert (host, origin) == (421, 403)
    assert "hunter2" not in caplog.text and "hiefs" not in caplog.text


def test_request_lines_are_not_repeated_by_the_root_logger(production_logging: None) -> None:
    assert logging.getLogger("alltouchdowns.requests").propagate is False


def test_the_server_keeps_no_access_log() -> None:
    """An access log would record the IP address of every caller."""
    assert server_config(SETTINGS).access_log is False


@pytest.mark.parametrize(
    ("value", "trusted"),
    [("on", True), ("1", True), ("true", True), ("off", False), ("0", False), ("", False),
     (None, False)],
)  # fmt: skip
def test_the_proxy_header_is_trusted_only_when_switched_on(
    value: str | None, trusted: bool
) -> None:
    env = {"ATD_DB_PATH": "x.db"}
    if value is not None:
        env["ATD_TRUST_PROXY_HEADER"] = value

    assert Settings.from_env(env).trust_proxy_header is trusted
