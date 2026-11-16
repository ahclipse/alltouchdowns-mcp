"""OAuth for the signed-in endpoint.

This server is an OAuth *resource server*: it never signs anyone in. A hosted authorization
server does that and issues a signed token (a JWT); all this module does is check that a token
is genuine and was issued for this server.

The check is plain JWT validation against the provider's published keys, so it works with any
standards-following provider. Switching provider is a change of three settings.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import anyio
import jwt
from mcp.server.auth.provider import AccessToken

logger = logging.getLogger(__name__)

# Signature algorithms accepted. Listing them explicitly matters: accepting whatever the token
# itself names is a classic JWT vulnerability.
ALGORITHMS = ("RS256", "ES256")
CLOCK_SKEW_SECONDS = 30

KEY_LIST_LIFETIME_SECONDS = 3600
MIN_SECONDS_BETWEEN_KEY_FETCHES = 60
KEY_FETCH_TIMEOUT_SECONDS = 5

KeyFetcher = Callable[[], dict[str, Any]]  # returns key id -> public key


@dataclass(frozen=True)
class AuthSettings:
    issuer: str  # the authorization server, e.g. https://example.authkit.app
    jwks_url: str  # where it publishes its signing keys
    resource_url: str  # this server's protected endpoint; tokens must be issued for it


def fetch_published_keys(jwks_url: str) -> dict[str, Any]:
    """Download the provider's signing keys (its JWKS document). A blocking network call."""
    client = jwt.PyJWKClient(jwks_url, cache_jwk_set=False, timeout=KEY_FETCH_TIMEOUT_SECONDS)
    return {key.key_id: key.key for key in client.get_signing_keys() if key.key_id}


class JwtVerifier:
    """Checks bearer tokens. Implements the SDK's `TokenVerifier` protocol."""

    def __init__(self, settings: AuthSettings, *, fetch_keys: KeyFetcher | None = None) -> None:
        self.settings = settings
        self._fetch_keys = fetch_keys or partial(fetch_published_keys, settings.jwks_url)
        self._keys: dict[str, Any] = {}  # from the last fetch that worked
        self._keys_fetched_at: float | None = None
        self._last_fetch_attempt: float | None = None
        self._fetching = anyio.Lock()

    async def verify_token(self, token: str) -> AccessToken | None:
        """Return who the token belongs to, or None if it should not be trusted.

        Every reason for refusing looks the same to the caller (a 401), on purpose: telling an
        attacker *why* a forged token failed helps them forge a better one.
        """
        try:
            key_id = jwt.get_unverified_header(token).get("kid")
            key = await self._key(key_id) if isinstance(key_id, str) else None
            if key is None:
                return None
            claims = jwt.decode(
                token,
                key,
                algorithms=list(ALGORITHMS),
                issuer=self.settings.issuer,
                audience=self.settings.resource_url,
                leeway=CLOCK_SKEW_SECONDS,
                options={"require": ["exp", "iss", "aud", "sub"]},
            )
        except Exception as error:
            # Deliberately broad: whatever went wrong, the answer is "not trusted", never a
            # crash. Only the kind of error is logged; the token never is.
            logger.info("token refused: %s", type(error).__name__)
            return None
        return AccessToken(
            token=token,
            client_id=str(claims.get("client_id") or claims.get("azp") or claims["sub"]),
            scopes=str(claims.get("scope") or "").split(),
            expires_at=int(claims["exp"]),
            resource=self.settings.resource_url,
            subject=str(claims["sub"]),
        )

    async def _key(self, key_id: str) -> Any | None:
        """The provider's public key with this id, fetching the key list when needed.

        The list is fetched again when a token names a key that is not in it (providers rotate
        keys), and when it is an hour old (so a key the provider has withdrawn stops being
        trusted). But a forged token can name any key id it likes, and the provider can be
        slow or down, so fetches are rationed: one at a time, at most one a minute whatever
        the reason, and when a fetch fails the keys already held stay in use.
        """
        if self._fetch_wanted(key_id):
            # Requests that arrive during a fetch wait here, then use what it brought back.
            async with self._fetching:
                if self._fetch_wanted(key_id) and self._fetch_allowed():
                    await self._fetch()
        return self._keys.get(key_id)

    def _fetch_wanted(self, key_id: str) -> bool:
        if key_id not in self._keys or self._keys_fetched_at is None:
            return True
        return time.monotonic() - self._keys_fetched_at >= KEY_LIST_LIFETIME_SECONDS

    def _fetch_allowed(self) -> bool:
        last = self._last_fetch_attempt
        return last is None or time.monotonic() - last >= MIN_SECONDS_BETWEEN_KEY_FETCHES

    async def _fetch(self) -> None:
        self._last_fetch_attempt = time.monotonic()
        try:
            # Off the event loop, because it blocks. The lock above means it never occupies
            # more than one worker thread, however many requests are waiting.
            self._keys = await anyio.to_thread.run_sync(self._fetch_keys)
            self._keys_fetched_at = time.monotonic()
        except Exception as error:
            # A timeout, a refused connection, an error page instead of JSON: all the same.
            logger.warning("could not fetch signing keys: %s", type(error).__name__)
