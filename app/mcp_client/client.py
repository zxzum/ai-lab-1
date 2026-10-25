"""MCP-клиент: запускает сервер дочерним процессом (stdio), узнаёт и вызывает его инструменты."""

import asyncio
import logging
import sys
import time
from dataclasses import dataclass, field

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.config import Settings
from app.mcp_trust import SECRET_ENV

logger = logging.getLogger("app.mcp")

CONNECT_TIMEOUT = 15
CALL_TIMEOUT = 20
RECONNECT_INTERVAL = 30  # не чаще одной попытки переподключения за это время, секунд


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

    Соединение живёт в отдельной задаче: процесс и сессия anyio должны открываться и
    закрываться в одной задаче, а использовать их могут любые обработчики.
    """

    def __init__(self, params: StdioServerParameters):
        self._params = params
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._lock = asyncio.Lock()
        self._last_attempt = 0.0
        self.session: ClientSession | None = None
        self.tools: list[ToolInfo] = []

    @property
    def available(self) -> bool:
        return self.session is not None

    async def start(self) -> None:
        """Подключается к серверу; при сбое бот продолжает работать без инструментов."""
        self._last_attempt = time.monotonic()
        self._stop = asyncio.Event()
        ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._task = asyncio.create_task(self._run(ready))
        try:
            await asyncio.wait_for(asyncio.shield(ready), CONNECT_TIMEOUT)
        except Exception as exc:
            logger.warning("MCP-сервер недоступен: %s: %s", type(exc).__name__, exc)
            await self.close()

    async def ensure_started(self) -> bool:
        """Повторное подключение после сбоя, не чаще раза в RECONNECT_INTERVAL."""
        async with self._lock:
            if not self.available and time.monotonic() - self._last_attempt > RECONNECT_INTERVAL:
                await self.close()
                await self.start()
            return self.available

    async def _run(self, ready: asyncio.Future[None]) -> None:
        try:
            async with stdio_client(self._params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listed = await session.list_tools()
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
                    logger.info(
                        "MCP-сервер подключён, инструменты: %s", [t.name for t in self.tools]
                    )
                    if not ready.done():
                        ready.set_result(None)
                    await self._stop.wait()
        except Exception as exc:
            if not ready.done():
                ready.set_exception(exc)
            else:
                logger.warning("MCP-соединение прервано: %s", type(exc).__name__)
        finally:
            self.session = None
            self.tools = []
            if not ready.done():
                ready.set_exception(ConnectionError("соединение закрыто"))

    async def close(self) -> None:
        self._stop.set()
        task, self._task = self._task, None
        self.session = None
        self.tools = []
        if task is not None:
            try:
                await asyncio.wait_for(task, 5)
            except (TimeoutError, Exception):
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def call_tool(self, name: str, arguments: dict, meta: dict | None = None) -> ToolResult:
        session = self.session
        if session is None:
            raise ToolUnavailableError("MCP-сервер недоступен")
        try:
            async with asyncio.timeout(CALL_TIMEOUT):
                result = await session.call_tool(name, arguments, meta=meta)
        except Exception as exc:
            logger.warning("Сбой вызова инструмента %s: %s", name, type(exc).__name__)
            self._stop.set()  # соединение считаем потерянным; ensure_started переподключит
            raise ToolUnavailableError(type(exc).__name__) from exc
        text = " ".join(c.text for c in result.content if getattr(c, "type", "") == "text")
        data = result.structuredContent
        if not result.isError and not isinstance(data, dict):
            return ToolResult(True, None, "неожиданная структура результата")
        return ToolResult(bool(result.isError), data, text)
