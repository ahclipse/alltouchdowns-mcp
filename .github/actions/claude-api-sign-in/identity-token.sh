#!/usr/bin/env bash
# Asks GitHub for an identity token for this job and writes it to the file named by
# ANTHROPIC_IDENTITY_TOKEN_FILE.
#
# The eval jobs sign in to the Claude API by workload identity federation. GitHub vouches for
# the job with a short-lived signed token that says which repository and environment it runs
# in; the Anthropic SDK reads that token from the file and exchanges it for an access token of
# its own. No API key is stored in GitHub.
#
# The token is good for about five minutes, so the workflow runs this again every two minutes.
set -euo pipefail

file="${ANTHROPIC_IDENTITY_TOKEN_FILE:?ANTHROPIC_IDENTITY_TOKEN_FILE is not set}"
# GitHub sets these two only for a job that has the "id-token: write" permission.
url="${ACTIONS_ID_TOKEN_REQUEST_URL:?this job has no id-token permission}"
bearer="${ACTIONS_ID_TOKEN_REQUEST_TOKEN:?this job has no id-token permission}"

curl --silent --show-error --fail --max-time 20 \
  --header "Authorization: Bearer $bearer" \
  "$url&audience=https://api.anthropic.com" |
  jq --raw-output --exit-status .value > "$file.new"

# Replaced in one step, so the SDK never reads half a token. If the request above failed,
# this line is not reached and the previous token stays in place.
mv "$file.new" "$file"
