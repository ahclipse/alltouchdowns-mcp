"""Settings, read once from environment variables at start-up."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from alltouchdowns_mcp.auth import AuthSettings

_LOCAL_HOSTS = ("127.0.0.1:*", "localhost:*", "[::1]:*")
_LOCAL_ORIGINS = ("http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*")


@dataclass(frozen=True)
class Settings:
    db_path: Path
    host: str = "127.0.0.1"
    port: int = 8000

    # Host and Origin headers the server accepts. They are always listed explicitly: the SDK
    # only applies its own defaults when bound to localhost, and accepts anything otherwise.
    allowed_hosts: tuple[str, ...] = _LOCAL_HOSTS
    allowed_origins: tuple[str, ...] = _LOCAL_ORIGINS

    query_timeout_ms: int = 250
    # /healthz reports a problem once the newest export is older than this.
    stale_after_hours: int = 36

    # Rate limits. On unless switched off, which the tests and the eval runner do: they make
    # far more calls per minute than any one real client.
    rate_limits: bool = True
    anonymous_per_minute: int = 30  # per IP address, on /mcp
    signed_in_per_minute: int = 300  # per signed-in user, on /auth/mcp
    failed_sign_in_per_minute: int = 60  # per IP address: rejected tokens on /auth/mcp
    total_per_second: int = 20  # everyone together on one endpoint: a ceiling on host load
    # True only behind the production proxy chain. See limits.client_address.
    trust_proxy_header: bool = False

    # The signed-in endpoint exists only when OAuth is configured.
    auth: AuthSettings | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> Settings:
        """Build settings from ATD_* variables. Only ATD_DB_PATH is required."""
        base = cls(db_path=Path(env["ATD_DB_PATH"]))

        def number(name: str, default: int) -> int:
            return int(env.get(name, default))

        auth = None
        oauth = [env.get(f"ATD_AUTH_{part}") for part in ("ISSUER", "JWKS_URL", "RESOURCE_URL")]
        if any(oauth):
            if not all(oauth):
                raise ValueError(
                    "set all of ATD_AUTH_ISSUER, ATD_AUTH_JWKS_URL and ATD_AUTH_RESOURCE_URL, "
                    "or none of them"
                )
            auth = AuthSettings(issuer=oauth[0], jwks_url=oauth[1], resource_url=oauth[2])  # type: ignore[arg-type]

        return cls(
            db_path=base.db_path,
            host=env.get("ATD_HOST", base.host),
            port=number("ATD_PORT", base.port),
            allowed_hosts=_csv(env.get("ATD_ALLOWED_HOSTS")) or base.allowed_hosts,
            allowed_origins=_csv(env.get("ATD_ALLOWED_ORIGINS")) or base.allowed_origins,
            query_timeout_ms=number("ATD_QUERY_TIMEOUT_MS", base.query_timeout_ms),
            stale_after_hours=number("ATD_STALE_AFTER_HOURS", base.stale_after_hours),
            rate_limits=env.get("ATD_RATE_LIMITS", "on").lower() not in ("off", "0", "false"),
            anonymous_per_minute=number("ATD_ANONYMOUS_PER_MINUTE", base.anonymous_per_minute),
            signed_in_per_minute=number("ATD_SIGNED_IN_PER_MINUTE", base.signed_in_per_minute),
            failed_sign_in_per_minute=number(
                "ATD_FAILED_SIGN_IN_PER_MINUTE", base.failed_sign_in_per_minute
            ),
            total_per_second=number("ATD_TOTAL_PER_SECOND", base.total_per_second),
            trust_proxy_header=env.get("ATD_TRUST_PROXY_HEADER", "off").lower()
            in ("on", "1", "true"),
            auth=auth,
        )


def _csv(value: str | None) -> tuple[str, ...]:
    return tuple(part.strip() for part in (value or "").split(",") if part.strip())
