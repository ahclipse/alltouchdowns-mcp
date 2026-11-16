"""The web application: the MCP endpoint, twice, plus a health check.

    /mcp        anyone, no sign-in, a small allowance per IP address
    /auth/mcp   OAuth access token required, a larger allowance per user
    /healthz    is the data fresh?

Why two endpoints rather than one with optional sign-in: an MCP client only offers to sign in
after the server answers 401. A URL that also serves anonymous callers never answers 401, so
its sign-in could never be reached. Both endpoints serve the same six tools.

`create_app` is the single entry point. Production, the tests and the eval runner all build
the app through it, so they exercise the same code.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import anyio
from mcp.server.auth.middleware.auth_context import AuthContextMiddleware
from mcp.server.auth.middleware.bearer_auth import BearerAuthBackend, RequireAuthMiddleware
from mcp.server.auth.provider import TokenVerifier
from mcp.server.auth.routes import create_protected_resource_routes
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import AnyHttpUrl
from starlette.applications import Starlette
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import BaseRoute, Mount, Route
from starlette.types import ASGIApp, Scope

from alltouchdowns_mcp.auth import JwtVerifier
from alltouchdowns_mcp.config import Settings
from alltouchdowns_mcp.db import read_meta
from alltouchdowns_mcp.envelope import MCP_PATH, CheckedEndpoint
from alltouchdowns_mcp.limits import FailedSignInLimit, RateLimit, TokenBuckets, client_address
from alltouchdowns_mcp.server import create_server

MAX_REQUEST_BYTES = 64 * 1024  # tool calls are a few hundred bytes; nothing needs more
REPOSITORY = "https://github.com/ahclipse/alltouchdowns-mcp"
_METADATA_PATH = "/.well-known/oauth-protected-resource"
HEALTH_READ_INTERVAL_SECONDS = 10


def create_app(settings: Settings, *, token_verifier: TokenVerifier | None = None) -> Starlette:
    """Build the application. `token_verifier` replaces the real one in tests, which have no
    authorization server to fetch keys from."""
    server = create_server(settings)
    mcp_app = server.streamable_http_app(
        streamable_http_path=MCP_PATH,
        # One JSON body per request and no server-side sessions: nothing here streams, and a
        # stateless server needs no sticky routing.
        json_response=True,
        stateless_http=True,
        max_request_body_size=MAX_REQUEST_BYTES,
        # Always explicit. The SDK only applies its own Host check when bound to localhost.
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=list(settings.allowed_hosts),
            allowed_origins=list(settings.allowed_origins),
        ),
        host=settings.host,
    )
    endpoint = CheckedEndpoint(mcp_app)

    routes: list[BaseRoute] = [Route("/healthz", HealthCheck(settings).respond, methods=["GET"])]
    if settings.auth is not None:
        routes += _metadata_routes(settings)
        verifier = token_verifier or JwtVerifier(settings.auth)
        routes.append(Mount("/auth", app=_signed_in(endpoint, settings, verifier)))
    routes.append(Mount("/", app=_anonymous(endpoint, settings)))

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        # A mounted app's own lifespan never runs, so the MCP session manager starts here.
        async with server.session_manager.run():
            yield

    return Starlette(routes=routes, lifespan=lifespan)


# --- The two tiers ---------------------------------------------------------------------------


def _anonymous(endpoint: ASGIApp, settings: Settings) -> ASGIApp:
    if not settings.rate_limits:
        return endpoint

    def caller(scope: Scope) -> str:
        return client_address(scope, trust_proxy_header=settings.trust_proxy_header)

    # The caller's own limit comes first, so a request refused by it never counts against
    # everyone's. Otherwise one busy caller could use up the total and lock the rest out.
    everyone = _total_limit(endpoint, "anonymous_total", settings)
    return RateLimit(everyone, "anonymous", TokenBuckets(settings.anonymous_per_minute), caller)


def _signed_in(endpoint: ASGIApp, settings: Settings, verifier: TokenVerifier) -> ASGIApp:
    """The endpoint behind OAuth. Read from the bottom up: that is the order a request meets
    the layers in."""
    assert settings.auth is not None
    resource = AnyHttpUrl(settings.auth.resource_url)
    metadata = AnyHttpUrl(_origin(settings.auth.resource_url) + _METADATA_PATH + _path(resource))

    # Lets code deeper in (the request log) know the call is signed in.
    app: ASGIApp = AuthContextMiddleware(endpoint)
    if settings.rate_limits:
        # Both limits run after the token is verified, so the first can count per user. As on
        # the anonymous endpoint, a user over their own limit never counts against the total.
        app = _total_limit(app, "signed_in_total", settings)
        buckets = TokenBuckets(settings.signed_in_per_minute)
        app = RateLimit(app, "signed_in", buckets, _signed_in_user)
    # No valid token: 401 with a pointer to the metadata, which is what starts sign-in.
    app = RequireAuthMiddleware(app, required_scopes=[], resource_metadata_url=metadata)
    if settings.rate_limits:

        def caller(scope: Scope) -> str:
            return client_address(scope, trust_proxy_header=settings.trust_proxy_header)

        # A caller whose tokens keep being refused gets 429 in place of that 401.
        app = FailedSignInLimit(app, TokenBuckets(settings.failed_sign_in_per_minute), caller)
    # Verifies the bearer token, if any, and records who it belongs to.
    return AuthenticationMiddleware(
        app, backend=BearerAuthBackend(verifier, resource_server_url=resource)
    )


def _total_limit(app: ASGIApp, name: str, settings: Settings) -> ASGIApp:
    """A ceiling on everyone using one endpoint, so the load this server puts on its host is
    bounded however many callers there are. Each endpoint has its own, so a rush of anonymous
    callers cannot shut signed-in users out; the two together allow twice the setting."""
    per_minute = settings.total_per_second * 60
    buckets = TokenBuckets(per_minute, burst=settings.total_per_second * 2)
    return RateLimit(app, name, buckets, lambda scope: "everyone")


def _signed_in_user(scope: Scope) -> str:
    token = scope["user"].access_token  # set by AuthenticationMiddleware; always present here
    return token.subject or token.client_id


def _metadata_routes(settings: Settings) -> list[BaseRoute]:
    """OAuth protected-resource metadata (RFC 9728): tells a client which authorization server
    to sign in with. Served at the path derived from the resource URL, and at the bare
    well-known path too, since some clients look there."""
    assert settings.auth is not None
    routes: list[BaseRoute] = list(
        create_protected_resource_routes(
            resource_url=AnyHttpUrl(settings.auth.resource_url),
            authorization_servers=[AnyHttpUrl(settings.auth.issuer)],
            resource_name="AllTouchdowns MCP",
            resource_documentation=AnyHttpUrl(REPOSITORY),
        )
    )
    first = routes[0]
    assert isinstance(first, Route)
    routes.append(Route(_METADATA_PATH, first.endpoint, methods=["GET", "OPTIONS"]))
    return routes


def _origin(url: str) -> str:
    parsed = AnyHttpUrl(url)
    port = f":{parsed.port}" if parsed.port and parsed.port not in (80, 443) else ""
    return f"{parsed.scheme}://{parsed.host}{port}"


def _path(url: AnyHttpUrl) -> str:
    return (url.path or "").rstrip("/")


# --- Health ----------------------------------------------------------------------------------


class HealthCheck:
    """200 while the data is fresh. 503 when the daily export has stopped arriving, which is
    the one failure nothing else would reveal: the server keeps answering from old data.

    The endpoint is public and has no rate limit, so it must cost nothing to call: the
    database is read at most once every few seconds, and that read does not queue behind tool
    calls.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._read_again_at = 0.0
        self._meta: dict[str, str] | None = None

    async def respond(self, request: Request) -> JSONResponse:
        now = time.monotonic()
        if now >= self._read_again_at:
            self._read_again_at = now + HEALTH_READ_INTERVAL_SECONDS  # set first: one reader
            self._meta = await anyio.to_thread.run_sync(self._read)
        return self.report(self._meta)

    def _read(self) -> dict[str, str] | None:
        try:
            return read_meta(self.settings.db_path)
        except (sqlite3.Error, OSError):
            return None

    def report(self, meta: dict[str, str] | None) -> JSONResponse:
        if meta is None:
            return JSONResponse({"status": "unavailable"}, status_code=503)
        exported_at = datetime.strptime(meta["exported_at"], "%Y-%m-%dT%H:%M:%SZ")
        age = datetime.now(UTC) - exported_at.replace(tzinfo=UTC)
        stale = age > timedelta(hours=self.settings.stale_after_hours)
        body = {
            "status": "stale" if stale else "ok",
            "exported_at": meta["exported_at"],
            "data_through": meta["data_through"],
        }
        return JSONResponse(body, status_code=503 if stale else 200)
