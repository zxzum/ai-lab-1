"""Проверка MCP-сервера независимым клиентом: настоящий процесс и протокол по stdio."""

import json
import sys
from contextlib import asynccontextmanager
from datetime import date

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


async def test_schedule_tool_and_resource_are_discoverable():
    async with open_session() as session:
        tools = {t.name: t for t in (await session.list_tools()).tools}
        resources = (await session.list_resources()).resources

    assert tools["get_schedule"].inputSchema["required"] == ["date"]
    assert tools["get_schedule"].annotations.readOnlyHint is True
    assert tools["get_schedule"].inputSchema["additionalProperties"] is False
    assert [str(r.uri) for r in resources] == ["schedule://current-week"]


async def test_get_schedule_returns_lessons_from_bundled_file():
    async with open_session() as session:
        result = await session.call_tool("get_schedule", {"date": "2026-10-26"})

    assert result.isError is False
    lessons = result.structuredContent["lessons"]
    assert result.structuredContent["date"] == "2026-10-26"
    assert [le["start"] for le in lessons] == ["09:00", "10:40"]
    assert lessons[0]["title"] == "Алгоритмы и структуры данных (лекция)"


async def test_get_schedule_empty_day_is_valid_result():
    async with open_session() as session:
        result = await session.call_tool("get_schedule", {"date": "2026-10-25"})

    assert result.isError is False
    assert result.structuredContent["lessons"] == []


@pytest.mark.parametrize("value", ["26.10.2026", "2026-10-26T09:00", "2026-02-30", "tomorrow"])
async def test_get_schedule_rejects_bad_date(value):
    async with open_session() as session:
        result = await session.call_tool("get_schedule", {"date": value})

    assert result.isError is True


async def test_current_week_resource_has_seven_days_monday_to_sunday():
    async with open_session() as session:
        content = (await session.read_resource("schedule://current-week")).contents[0]

    week = json.loads(content.text)
    assert len(week["days"]) == 7
    assert date.fromisoformat(week["week_start"]).weekday() == 0
    assert date.fromisoformat(week["week_end"]).weekday() == 6
