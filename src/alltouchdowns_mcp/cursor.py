"""Opaque pagination cursors.

A cursor records the last row a page ended on (a "keyset"), not a row offset, so the next page
is found with an index instead of by counting rows. It also carries a fingerprint of the
search it belongs to and of the data it was issued against. A cursor presented with different
filters, or after the daily export has replaced the database, is refused, and the caller
starts again. That is simpler and safer than trying to continue across a change in the data,
where a newly published game could sort before the cursor and be skipped.

A cursor is not a secret and is not signed. It grants nothing a fresh search would not.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Mapping, Sequence

from alltouchdowns_mcp.errors import ToolRefusal

Key = Sequence[str | int]
_SQLITE_INTEGERS = range(-(2**63), 2**63)


def encode(key: Key, search: Mapping[str, object]) -> str:
    payload = json.dumps({"k": list(key), "f": _fingerprint(search)}, separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")


def decode(cursor: str, search: Mapping[str, object], shape: Sequence[type]) -> list[str | int]:
    """Return the key inside `cursor`, or refuse it.

    `search` is everything that defines the result set: the filters, the order and the version
    of the data. `shape` lists the expected type of each part of the key, for example
    (str, str, int). Anything else is treated as a tampered or mismatched cursor.
    """
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode()))
        key, fingerprint = payload["k"], payload["f"]
    except (ValueError, KeyError, TypeError, binascii.Error) as error:
        raise ToolRefusal("bad_cursor") from error

    well_formed = (
        isinstance(key, list)
        and len(key) == len(shape)
        and all(type(part) is kind for part, kind in zip(key, shape, strict=True))
        and all(part in _SQLITE_INTEGERS for part in key if isinstance(part, int))
    )
    if not well_formed or fingerprint != _fingerprint(search):
        raise ToolRefusal("bad_cursor")
    return key


def _fingerprint(search: Mapping[str, object]) -> str:
    canonical = json.dumps(search, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]
