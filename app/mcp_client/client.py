"""MCP-клиент: запускает сервер дочерним процессом (stdio), узнаёт и вызывает его инструменты."""

import asyncio
import logging
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.config import Settings
from app.mcp_trust import SECRET_ENV

logger = logging.getLogger("app.mcp")

CONNECT_TIMEOUT = 15
CALL_TIMEOUT = 20


class ToolUnavailableError(Exception):
    """Сервер недоступен, оборвал соединение или не ответил вовремя."""


@dataclass(frozen=True)
class ToolInfo:
    name: str
    description: str
    read_only: bool
    input_schema: dict = field(repr=False)


@dataclass(frozen=True)
class ToolResult:
    is_error: bool
    data: dict | None
    text: str


def server_params(settings: Settings, secret: str = "") -> StdioServerParameters:
    """Серверу передаются только его настройки: без токена бота и ключей LLM."""
    env = {
        "SCHEDULE_TIMEZONE": settings.schedule_timezone,
        "POSTGRES_HOST": settings.postgres_host,
        "POSTGRES_PORT": str(settings.postgres_port),
        "POSTGRES_DB": settings.postgres_db,
        "POSTGRES_USER": settings.postgres_user,
        "POSTGRES_PASSWORD": settings.postgres_password,
    }
    if settings.schedule_file:
        env["SCHEDULE_FILE"] = settings.schedule_file
    if secret:
        env[SECRET_ENV] = secret
    return StdioServerParameters(command=sys.executable, args=["-m", "app.mcp_server"], env=env)


class MCPToolbox:
    """Соединение с MCP-сервером и кэш обнаруженных инструментов.

    start() и close() вызываются из одной задачи (run в __main__): этого требует anyio.
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

    async def call_tool(self, name: str, arguments: dict, meta: dict | None = None) -> ToolResult:
        if self.session is None:
            raise ToolUnavailableError("MCP-сервер недоступен")
        try:
            async with asyncio.timeout(CALL_TIMEOUT):
                result = await self.session.call_tool(name, arguments, meta=meta)
        except Exception as exc:
            logger.warning("Сбой вызова инструмента %s: %s", name, type(exc).__name__)
            self.session = None  # соединение потеряно: бот отвечает без инструментов
            raise ToolUnavailableError(type(exc).__name__) from exc
        text = " ".join(c.text for c in result.content if c.type == "text")
        if not result.isError and not isinstance(result.structuredContent, dict):
            return ToolResult(True, None, "неожиданная структура результата")
        return ToolResult(result.isError, result.structuredContent, text)
