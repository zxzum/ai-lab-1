"""MCP-клиент: запускает сервер дочерним процессом (stdio) и узнаёт его инструменты."""

import asyncio
import logging
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.config import Settings

logger = logging.getLogger("app.mcp")

CONNECT_TIMEOUT = 15


@dataclass(frozen=True)
class ToolInfo:
    name: str
    description: str
    read_only: bool
    input_schema: dict = field(repr=False)


def server_params(settings: Settings) -> StdioServerParameters:
    """Сервер получает только свои настройки, без токенов бота и ключей LLM."""
    env = {"SCHEDULE_TIMEZONE": settings.schedule_timezone}
    if settings.schedule_file:
        env["SCHEDULE_FILE"] = settings.schedule_file
    return StdioServerParameters(command=sys.executable, args=["-m", "app.mcp_server"], env=env)


class MCPToolbox:
    """Соединение с MCP-сервером и кэш обнаруженных инструментов.

    start() и close() нужно вызывать из одной задачи asyncio: так требует anyio
    для процесса и сессии, открытых через AsyncExitStack.
    """

    def __init__(self, params: StdioServerParameters):
        self._params = params
        self._stack = AsyncExitStack()
        self.session: ClientSession | None = None
        self.tools: list[ToolInfo] = []

    @property
    def available(self) -> bool:
        return self.session is not None

    async def start(self) -> None:
        """Подключается к серверу; при сбое бот продолжает работать без инструментов."""
        try:
            async with asyncio.timeout(CONNECT_TIMEOUT):
                read, write = await self._stack.enter_async_context(stdio_client(self._params))
                session = await self._stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                listed = await session.list_tools()
        except Exception as exc:
            logger.warning("MCP-сервер недоступен: %s: %s", type(exc).__name__, exc)
            await self.close()
            return
        self.tools = [
            ToolInfo(
                name=t.name,
                description=t.description or "",
                read_only=bool(t.annotations and t.annotations.readOnlyHint),
                input_schema=t.inputSchema,
            )
            for t in listed.tools
        ]
        self.session = session
        logger.info("MCP-сервер подключён, инструменты: %s", [t.name for t in self.tools])

    async def close(self) -> None:
        self.session = None
        self.tools = []
        try:
            await self._stack.aclose()
        except Exception as exc:
            logger.warning("Ошибка при закрытии MCP-соединения: %s", type(exc).__name__)
