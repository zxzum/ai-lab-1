"""Тесты MCP-клиента бота и команды /tools: настоящий сервер и заведомо сломанный."""

import sys
from unittest.mock import AsyncMock

import pytest
from mcp import StdioServerParameters

from app.config import ConfigError, Settings
from app.handlers.commands import cmd_tools, format_tools
from app.mcp_client.client import MCPToolbox, ToolInfo, server_params

TOKEN = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk"


def settings(tmp_path, **env):
    path = tmp_path / ".env"
    path.write_text(f"BOT_TOKEN={TOKEN}\nPOSTGRES_PASSWORD=pw\n", encoding="utf-8")
    return Settings.load(path, environ=env)


async def test_toolbox_discovers_tools_from_real_server(tmp_path):
    toolbox = MCPToolbox(server_params(settings(tmp_path)))

    try:
        await toolbox.start()

        assert toolbox.available
        assert {t.name: t.read_only for t in toolbox.tools} == {
            "get_weather": True,
            "get_schedule": True,
        }
    finally:
        await toolbox.close()
    assert not toolbox.available
    assert toolbox.tools == []


async def test_unreachable_server_does_not_break_startup():
    broken = StdioServerParameters(command=sys.executable, args=["-c", "raise SystemExit(1)"])
    toolbox = MCPToolbox(broken)

    await toolbox.start()  # исключения быть не должно

    assert not toolbox.available
    assert toolbox.tools == []


def test_server_gets_only_its_own_settings(tmp_path):
    params = server_params(settings(tmp_path, SCHEDULE_TIMEZONE="Asia/Tokyo", LLM_API_KEY="k"))

    assert params.env == {"SCHEDULE_TIMEZONE": "Asia/Tokyo"}


def test_unknown_schedule_timezone_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="SCHEDULE_TIMEZONE"):
        settings(tmp_path, SCHEDULE_TIMEZONE="Mars/Base")


def fake_toolbox(available, tools=()):
    toolbox = MCPToolbox(StdioServerParameters(command="x"))
    toolbox.tools = list(tools)
    toolbox.session = object() if available else None
    return toolbox


def test_format_tools_lists_name_summary_and_kind():
    toolbox = fake_toolbox(
        True,
        [
            ToolInfo("get_weather", "Текущая погода в городе. Вернёт варианты.", True, {}),
            ToolInfo("add_reminder", "Создание напоминания", False, {}),
        ],
    )

    assert format_tools(toolbox) == (
        "Доступные инструменты:\n"
        "• get_weather — Текущая погода в городе (чтение)\n"
        "• add_reminder — Создание напоминания (изменение, нужно подтверждение)"
    )


def test_format_tools_when_server_unavailable():
    assert format_tools(fake_toolbox(False)) == "Инструменты сейчас недоступны."


async def test_tools_command_answers_with_plain_text():
    message = AsyncMock()

    await cmd_tools(message, fake_toolbox(False))

    message.answer.assert_awaited_once_with("Инструменты сейчас недоступны.", parse_mode=None)
