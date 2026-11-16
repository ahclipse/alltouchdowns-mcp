#!/usr/bin/env bash
# Daily export, run from cron. Safe to run at any time and as often as you like: it reads a
# snapshot, never the live database, and only replaces mcp.db when the new one checks out.
#
# Exit codes come from the export itself:
#   0   published
#   2   a banned term, or the banned-pattern list no longer matches the website's
#   3   the result looked wrong; needs a person
#   75  the source was not readable right now; the next run will try again
set -euo pipefail
cd "$(dirname "$0")"

log() { echo "[mcp-export $(date -u +%FT%TZ)] $*"; }

# </dev/null: without it, docker would swallow the rest of a script fed over ssh.
if docker compose run --rm --no-deps -T export < /dev/null; then
  exit 0
else
  code=$?
  log "FAILED: export exited with $code"
  exit "$code"
fi
