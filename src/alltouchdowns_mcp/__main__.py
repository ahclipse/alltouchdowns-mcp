"""Run the server: `python -m alltouchdowns_mcp` with ATD_DB_PATH set."""

from __future__ import annotations

import os

import uvicorn

from alltouchdowns_mcp import logs
from alltouchdowns_mcp.app import create_app
from alltouchdowns_mcp.config import Settings


def server_config(settings: Settings) -> uvicorn.Config:
    return uvicorn.Config(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        # No access log: it would record client addresses, which this project does not keep.
        access_log=False,
        log_level="warning",
    )


def main() -> None:
    settings = Settings.from_env(os.environ)
    logs.configure()
    uvicorn.Server(server_config(settings)).run()


if __name__ == "__main__":
    main()
