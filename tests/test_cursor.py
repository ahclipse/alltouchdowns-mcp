from __future__ import annotations

import base64
import json

import pytest

from alltouchdowns_mcp import cursor
from alltouchdowns_mcp.errors import ToolRefusal

FILTERS = {"season": 2019, "franchise": "KC", "order": "oldest"}


def test_round_trip() -> None:
    token = cursor.encode(("2019-05-IND-KC", 3), FILTERS)

    assert cursor.decode(token, FILTERS, (str, int)) == ["2019-05-IND-KC", 3]
    assert "=" not in token and "2019" not in token  # opaque, and safe in a URL


def test_cursor_from_a_different_search_is_refused() -> None:
    token = cursor.encode(("2019-05-IND-KC", 3), FILTERS)

    with pytest.raises(ToolRefusal) as refusal:
        cursor.decode(token, {**FILTERS, "season": 2007}, (str, int))
    assert refusal.value.code == "bad_cursor"


@pytest.mark.parametrize("token", ["", "not-a-cursor", "e30", "!!!!", "eyJrIjpbXX0"])
def test_garbage_is_refused(token: str) -> None:
    with pytest.raises(ToolRefusal):
        cursor.decode(token, FILTERS, (str, int))


@pytest.mark.parametrize("key", [["only-one-part"], [3, "wrong-order"], ["x", "3"], ["x", 1.5]])
def test_key_of_the_wrong_shape_is_refused(key: list) -> None:
    payload = json.dumps({"k": key, "f": cursor._fingerprint(FILTERS)})
    token = base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")

    with pytest.raises(ToolRefusal):
        cursor.decode(token, FILTERS, (str, int))
