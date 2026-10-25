"""Проверка MCP-сервера независимым клиентом: настоящий процесс и протокол по stdio."""

import sys
from contextlib import asynccontextmanager

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PARAMS = StdioServerParameters(command=sys.executable, args=["-m", "app.mcp_server"])


@asynccontextmanager
async def open_session():
    async with stdio_client(PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def test_server_exposes_weather_tool_with_schemas():
    async with open_session() as session:
        tools = {t.name: t for t in (await session.list_tools()).tools}

    tool = tools["get_weather"]
    assert tool.inputSchema["required"] == ["city"]
    assert tool.inputSchema["properties"]["city"]["maxLength"] == 100
    assert tool.inputSchema["additionalProperties"] is False
    assert tool.outputSchema["properties"]["status"]["enum"] == ["ok", "not_found", "ambiguous"]
    assert tool.annotations.readOnlyHint is True


@pytest.mark.parametrize("city", ["", "   ", "x" * 101])
async def test_invalid_city_is_rejected_before_network(city):
    async with open_session() as session:
        result = await session.call_tool("get_weather", {"city": city})

    assert result.isError is True
    assert result.structuredContent is None


async def test_extra_argument_is_rejected_before_network():
    async with open_session() as session:
        result = await session.call_tool("get_weather", {"city": "Казань", "user_id": 1})

    assert result.isError is True
