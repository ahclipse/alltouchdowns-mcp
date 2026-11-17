"""The script the eval workflow uses to fetch GitHub's identity token, against a local stand-in
for GitHub's token endpoint."""

from __future__ import annotations

import base64
import http.server
import json
import os
import shutil
import subprocess
import threading
from collections.abc import Iterator, Mapping
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

ACTION = Path(__file__).parent.parent / ".github" / "actions" / "claude-api-sign-in"
SCRIPT = ACTION / "identity-token.sh"
CLAIMS_SCRIPT = ACTION / "identity-claims.sh"

pytestmark = pytest.mark.skipif(
    not (shutil.which("bash") and shutil.which("curl") and shutil.which("jq")),
    reason="needs bash, curl and jq, as on a GitHub runner",
)


class GitHub:
    """What the stand-in endpoint answers next, and what it was asked."""

    def __init__(self) -> None:
        self.reply: tuple[int, object] = (200, {"value": "token-1"})
        self.requests: list[dict[str, object]] = []
        self.url = ""


@pytest.fixture
def github() -> Iterator[GitHub]:
    state = GitHub()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            query = parse_qs(urlparse(self.path).query)
            state.requests.append(
                {"authorization": self.headers.get("Authorization"), "audience": query["audience"]}
            )
            status, payload = state.reply
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state.url = f"http://127.0.0.1:{server.server_port}/token?api-version=2.0"
    yield state
    server.shutdown()
    server.server_close()


def run(
    github: GitHub, token_file: Path, **changes: str | None
) -> subprocess.CompletedProcess[str]:
    env: dict[str, str | None] = {
        "PATH": os.environ["PATH"],
        "ANTHROPIC_IDENTITY_TOKEN_FILE": str(token_file),
        "ACTIONS_ID_TOKEN_REQUEST_URL": github.url,
        "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "request-token",
        **changes,
    }
    given = {name: value for name, value in env.items() if value is not None}
    return subprocess.run(["bash", str(SCRIPT)], env=given, capture_output=True, text=True)


def test_the_token_is_requested_for_the_claude_api_and_written_to_the_file(
    github: GitHub, tmp_path: Path
) -> None:
    token_file = tmp_path / "identity-token"

    result = run(github, token_file)

    assert result.returncode == 0, result.stderr
    assert token_file.read_text().strip() == "token-1"
    assert github.requests == [
        {"authorization": "Bearer request-token", "audience": ["https://api.anthropic.com"]}
    ]
    assert list(tmp_path.iterdir()) == [token_file]  # nothing half-written is left beside it


def test_a_later_run_replaces_the_token(github: GitHub, tmp_path: Path) -> None:
    token_file = tmp_path / "identity-token"
    run(github, token_file)
    github.reply = (200, {"value": "token-2"})

    assert run(github, token_file).returncode == 0
    assert token_file.read_text().strip() == "token-2"


@pytest.mark.parametrize(
    "reply",
    [(500, {"message": "unavailable"}), (200, {"value": None}), (200, {"unexpected": "shape"})],
)
def test_a_failed_request_keeps_the_previous_token_and_says_so(
    github: GitHub, tmp_path: Path, reply: tuple[int, object]
) -> None:
    """The loop that calls this ignores a failure, so a failure must never damage the file."""
    token_file = tmp_path / "identity-token"
    run(github, token_file)
    github.reply = reply

    assert run(github, token_file).returncode != 0
    assert token_file.read_text().strip() == "token-1"


@pytest.mark.parametrize(
    "missing",
    [
        "ANTHROPIC_IDENTITY_TOKEN_FILE",
        "ACTIONS_ID_TOKEN_REQUEST_URL",
        "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
    ],
)
def test_a_job_without_the_permission_is_told_what_is_missing(
    github: GitHub, tmp_path: Path, missing: str
) -> None:
    result = run(github, tmp_path / "identity-token", **{missing: None})

    assert result.returncode != 0 and github.requests == []
    assert "id-token permission" in result.stderr or "is not set" in result.stderr


# --- Showing who the token says the job is -----------------------------------------------------


def fake_token(claims: Mapping[str, object]) -> str:
    """Shaped like GitHub's: three base64url parts without padding, joined by dots."""

    def part(data: object) -> str:
        return base64.urlsafe_b64encode(json.dumps(data).encode()).decode().rstrip("=")

    return f"{part({'alg': 'RS256'})}.{part(claims)}.THE-SIGNATURE-MUST-NEVER-BE-PRINTED"


@pytest.mark.parametrize("filler", ["", "x", "xx", "xxx"])  # every length the padding can take
def test_the_claims_are_shown_and_the_token_is_not(tmp_path: Path, filler: str) -> None:
    claims = {
        "sub": "repo:someone@123/some-repo@456:environment:evals",
        "aud": "https://api.anthropic.com",
        "ref": "refs/heads/main",
        "environment": "evals",
        "repository_owner": "someone",
        "workflow_ref": "someone/some-repo/.github/workflows/evals.yml@refs/heads/main",
        "jti": "a-value-that-is-not-shown" + filler,
        "url-unsafe": "<<<???>>>",  # makes the encoding use the characters - and _
    }
    token_file = tmp_path / "identity-token"
    token_file.write_text(fake_token(claims) + "\n")
    env = {"PATH": os.environ["PATH"], "ANTHROPIC_IDENTITY_TOKEN_FILE": str(token_file)}

    result = subprocess.run(["bash", str(CLAIMS_SCRIPT)], env=env, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    shown = dict(line.split("=", 1) for line in result.stdout.splitlines())
    assert shown["sub"] == "repo:someone@123/some-repo@456:environment:evals"
    assert shown["ref"] == "refs/heads/main" and shown["environment"] == "evals"
    assert shown["workflow_ref"].endswith("evals.yml@refs/heads/main")
    assert shown["repository_id"] == "null"  # a claim the token lacks is shown as missing
    assert "SIGNATURE" not in result.stdout and "not-shown" not in result.stdout
