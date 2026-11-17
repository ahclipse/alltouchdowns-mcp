#!/usr/bin/env bash
# Prints who GitHub says this job is: the claims inside the identity token.
#
# A federation rule on the Anthropic side has to match these values exactly, so this is where
# to copy them from. They are identifiers (repository, branch, environment, workflow file), not
# secrets, and on a public repository they are public already. The token itself, which is what
# has to stay private, is never printed: only its middle part is decoded, and without the
# signature that part proves nothing to anyone.
set -euo pipefail

file="${ANTHROPIC_IDENTITY_TOKEN_FILE:?ANTHROPIC_IDENTITY_TOKEN_FILE is not set}"

# A token is three parts joined by dots; the second is the claims, as base64url JSON.
jq --raw-input --raw-output '
  split(".")[1] | gsub("-"; "+") | gsub("_"; "/") | @base64d | fromjson
  | {sub, aud, ref, environment, repository_owner, repository_owner_id, repository,
     repository_id, workflow_ref}
  | to_entries[] | "\(.key)=\(.value)"
' "$file"
