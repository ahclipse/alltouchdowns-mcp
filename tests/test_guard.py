"""The exit gate every tool result passes through."""

from __future__ import annotations

import json

from mcp.types import CallToolResult, TextContent

from alltouchdowns_mcp import ATTRIBUTION
from alltouchdowns_mcp.guard import finalise


def success(structured: dict) -> CallToolResult:
    """A result the way the SDK builds it: indented JSON text plus the structured form."""
    text = json.dumps(structured, indent=2)
    return CallToolResult(
        content=[TextContent(type="text", text=text)], structured_content=structured
    )


def failure(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], is_error=True)


def text_of(result: CallToolResult) -> str:
    (block,) = result.content
    assert isinstance(block, TextContent)
    return block.text


# --- Successful results ----------------------------------------------------------------------


def test_success_text_becomes_compact_json_with_the_same_content() -> None:
    structured = {
        "rows": [{"name": "Audric Estimé", "n": 5}],
        "source": {"attribution": ATTRIBUTION},
    }

    result = finalise(success(structured))

    assert not result.is_error
    assert result.structured_content == structured
    assert json.loads(text_of(result)) == structured
    assert "\n" not in text_of(result) and "Estimé" in text_of(result)


def test_success_without_attribution_gets_it_appended() -> None:
    result = finalise(success({"rows": []}))

    assert text_of(result).endswith(ATTRIBUTION)


def test_allowlisted_player_name_is_not_withheld() -> None:
    structured = {"scorer": {"slug": "colt-mccoy", "name": "Colt McCoy"}, "note": ATTRIBUTION}

    assert not finalise(success(structured)).is_error


def test_result_containing_a_banned_term_is_withheld() -> None:
    result = finalise(success({"team": {"abbr": "KC", "name": "Kansas City Chiefs"}}))

    assert result.is_error
    assert result.structured_content is not None
    assert result.structured_content["error"]["code"] == "internal_error"
    assert "Chiefs" not in str(result.structured_content)
    assert text_of(result).startswith("internal_error:")
    assert "Chiefs" not in text_of(result)
    assert ATTRIBUTION in text_of(result)


# --- Errors ----------------------------------------------------------------------------------


def test_tool_refusal_loses_the_sdk_prefix_and_gains_attribution() -> None:
    result = finalise(
        failure("Error executing tool get_player: unknown_player: No player has that slug.")
    )

    assert result.is_error
    assert text_of(result) == f"unknown_player: No player has that slug.\n\n{ATTRIBUTION}"
    # The same, in a form a program can read without parsing the sentence.
    assert result.structured_content == {
        "error": {"code": "unknown_player", "message": "unknown_player: No player has that slug."},
        "attribution": ATTRIBUTION,
    }


def test_argument_errors_are_rebuilt_without_the_caller_s_input() -> None:
    sdk_message = (
        "Error executing tool search_touchdowns: "
        "2 validation errors for search_touchdownsArguments\n"
        "season\n"
        "  Input should be a valid integer, unable to parse string as an integer "
        "[type=int_parsing, input_value='the Chiefs in 2019', input_type=str]\n"
        "    For further information visit https://errors.pydantic.dev/2.14/v/int_parsing\n"
        "limit\n"
        "  Input should be less than or equal to 50 "
        "[type=less_than_equal, input_value=500, input_type=int]\n"
        "    For further information visit https://errors.pydantic.dev/2.14/v/less_than_equal"
    )

    text = text_of(finalise(failure(sdk_message)))

    assert text.startswith("invalid_argument: An argument is missing or not allowed.")
    assert "season: Input should be a valid integer" in text
    assert "limit: Input should be less than or equal to 50." in text
    assert "Chiefs" not in text and "500" not in text and "pydantic" not in text
    assert text.endswith(ATTRIBUTION)


def test_a_crash_says_nothing_about_what_went_wrong() -> None:
    result = finalise(failure("Error executing tool get_game"))

    assert text_of(result).startswith("internal_error:")


def test_unknown_tool_does_not_repeat_the_name_it_was_sent() -> None:
    text = text_of(finalise(failure("Unknown tool: Chiefs")))

    assert text.startswith("unknown_tool:") and "Chiefs" not in text


def test_error_carrying_a_banned_term_is_withheld_too() -> None:
    text = text_of(finalise(failure("Error executing tool x: not_found: No Super Bowl here.")))

    assert text.startswith("internal_error:") and "Super Bowl" not in text
