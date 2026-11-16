"""The server as an MCP client sees it, through the SDK's in-memory client."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import ClassVar

import pytest
from mcp import Client
from mcp.types import CallToolResult, TextContent

from alltouchdowns_mcp import ATTRIBUTION, legal
from alltouchdowns_mcp.config import Settings
from alltouchdowns_mcp.server import create_server
from tests.conftest import FIXTURE_DB

pytestmark = pytest.mark.anyio

TOOLS = [
    "find_players",
    "get_player",
    "search_touchdowns",
    "get_game",
    "list_games",
    "get_leaderboard",
]

# One ordinary call per tool.
CALLS: dict[str, dict] = {
    "find_players": {"query": "tomlinson"},
    "get_player": {"slug": "ladainian-tomlinson"},
    "search_touchdowns": {"team": "San Diego", "season": 2007, "limit": 3},
    "get_game": {"game_id": "2007-21-NYG-NE"},
    "list_games": {"season": 2019, "week": 1},
    "get_leaderboard": {"season": 2019, "limit": 3},
}


@pytest.fixture
async def client() -> AsyncIterator[Client]:
    server = create_server(Settings(db_path=FIXTURE_DB))
    async with Client(server) as connected:
        yield connected


def text_of(result: CallToolResult) -> str:
    (block,) = result.content
    assert isinstance(block, TextContent)
    return block.text


# --- What the client model reads before calling anything -------------------------------------


async def test_six_read_only_tools_with_output_schemas(client: Client) -> None:
    tools = (await client.list_tools()).tools

    assert [tool.name for tool in tools] == TOOLS
    for tool in tools:
        assert tool.description and tool.title
        assert tool.output_schema is not None
        assert tool.annotations is not None
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.open_world_hint is False


async def test_everything_a_client_reads_up_front_is_clean_and_attributed(client: Client) -> None:
    tools = (await client.list_tools()).tools
    published = json.dumps([tool.model_dump(mode="json") for tool in tools])

    # No allowlist here: tool descriptions and schemas never contain a person's name.
    assert legal.find_banned(published, allow_names=False) == []
    assert legal.find_banned(client.instructions or "", allow_names=False) == []
    assert ATTRIBUTION in (client.instructions or "")
    assert client.server_info is not None
    assert ATTRIBUTION in (client.server_info.description or "")


async def test_instructions_state_the_seasons_actually_loaded(client: Client) -> None:
    assert "for seasons 2007 and 2019" in (client.instructions or "")
    assert "Passes thrown are not recorded" in (client.instructions or "")


# --- Every result ----------------------------------------------------------------------------


@pytest.mark.parametrize("tool", TOOLS)
async def test_every_tool_returns_attributed_structured_output(client: Client, tool: str) -> None:
    result = await client.call_tool(tool, CALLS[tool])

    assert not result.is_error
    assert result.structured_content is not None
    assert result.structured_content["source"]["attribution"] == ATTRIBUTION
    assert result.structured_content["source"]["seasons"] == "2007 and 2019"
    # The text a model reads is the same data, as compact JSON.
    assert json.loads(text_of(result)) == result.structured_content
    assert legal.find_banned(text_of(result)) == []


@pytest.mark.parametrize(
    ("tool", "arguments", "code"),
    [
        ("search_touchdowns", {"team": "Chiefs"}, "unknown_team"),
        ("search_touchdowns", {"team": "Los Angeles"}, "ambiguous_team"),
        ("search_touchdowns", {"season": "the Chiefs in 2019"}, "invalid_argument"),
        ("search_touchdowns", {"td_type": "passing"}, "invalid_argument"),
        ("search_touchdowns", {"limit": 500}, "invalid_argument"),
        ("get_leaderboard", {"season": 1998}, "not_covered"),
        ("get_player", {"slug": "Chiefs"}, "unknown_player"),
        ("get_player", {}, "invalid_argument"),
        ("get_game", {"season": 2007}, "invalid_argument"),
        ("list_games", {"season": 2007, "cursor": "Chiefs"}, "bad_cursor"),
        ("Chiefs", {}, "unknown_tool"),
    ],
)
async def test_every_error_is_coded_attributed_and_never_repeats_input(
    client: Client, tool: str, arguments: dict, code: str
) -> None:
    result = await client.call_tool(tool, arguments)

    text = text_of(result)
    assert result.is_error
    assert result.structured_content is not None
    assert result.structured_content["error"]["code"] == code
    assert result.structured_content["attribution"] == ATTRIBUTION
    assert "Chiefs" not in str(result.structured_content)
    assert text.startswith(f"{code}: ")
    assert text.endswith(ATTRIBUTION)
    assert "Chiefs" not in text and "500" not in text
    assert legal.find_banned(text) == []


async def test_error_for_a_wrong_enum_value_lists_the_valid_ones(client: Client) -> None:
    result = await client.call_tool("search_touchdowns", {"td_type": "passing"})

    assert "'receiving'" in text_of(result) and "'rushing'" in text_of(result)


async def test_client_and_server_agree_on_the_current_protocol(client: Client) -> None:
    assert client.protocol_version == "2026-07-28"


async def test_descriptions_are_tidy_and_every_parameter_is_described(client: Client) -> None:
    for tool in (await client.list_tools()).tools:
        assert tool.description is not None
        assert "\n    " not in tool.description  # docstring indentation was removed
        for name, schema in tool.input_schema["properties"].items():
            assert schema.get("description"), f"{tool.name}.{name} has no description"


# --- Review findings -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("get_leaderboard", {"year": 2019, "limit": 1}),  # a natural guess for "season"
        ("get_leaderboard", {"season": 2019, "position": "TE"}),  # not something it can filter
        ("search_touchdowns", {"player_name": "Patrick Mahomes"}),
        ("list_games", {"season": 2019, "Chiefs": True}),
    ],
)
async def test_an_argument_the_tool_does_not_have_is_refused_not_ignored(
    client: Client, tool: str, arguments: dict
) -> None:
    """Ignoring it would return the unfiltered answer, which looks exactly like a real one."""
    result = await client.call_tool(tool, arguments)

    text = text_of(result)
    assert result.is_error and text.startswith("invalid_argument: ")
    assert "season" in text  # the valid names are listed
    for unknown in ("year", "position", "player_name", "Chiefs"):
        assert unknown not in text.replace("player, ", "")  # the caller's name is not repeated


async def test_per_season_leaderboard_through_the_client(client: Client) -> None:
    result = await client.call_tool("get_leaderboard", {"per_season": True, "limit": 1})

    assert result.structured_content is not None
    assert result.structured_content["rows"][0]["season"] == 2007
    assert "single season" in result.structured_content["ranked_by"]


async def test_exit_gate_refuses_a_result_it_does_not_recognise() -> None:
    from alltouchdowns_mcp.guard import ExitGate

    class Context:
        method = "tools/call"
        params: ClassVar[dict] = {"name": "get_game", "arguments": {}}

    async def unexpected_shape(ctx: object) -> dict:
        return {"something": "else"}

    result = await ExitGate()(Context(), unexpected_shape)  # type: ignore[arg-type]

    assert isinstance(result, CallToolResult) and result.is_error
    assert text_of(result).startswith("internal_error:")
