"""MCP-сервер: описывает инструменты и делегирует работу модулям с логикой."""

import datetime as dt
import logging
from typing import Annotated

import aiohttp
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field, StringConstraints

from app.mcp_server.reminders import (
    ReminderRefusedError,
    ReminderResult,
    ReminderStore,
    parse_remind_at,
)
from app.mcp_server.schedule import (
    DaySchedule,
    ScheduleUnavailableError,
    day_schedule,
    load_entries,
    schedule_timezone,
    today_in,
    week_schedule,
)
from app.mcp_server.weather import WeatherResult, WeatherUpstreamError, fetch_weather
from app.mcp_trust import read_trusted

logger = logging.getLogger("app.mcp_server")

mcp = FastMCP("student-assistant")

READ_ONLY = ToolAnnotations(readOnlyHint=True)
WRITE_IDEMPOTENT = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True)

store = ReminderStore()


@mcp.tool(annotations=READ_ONLY)
async def get_weather(
    city: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=100),
        Field(description="Название города; для уточнения через запятую: 'Москва, Айдахо'"),
    ],
) -> WeatherResult:
    """Текущая погода в городе. При нескольких одноимённых городах вернёт варианты на выбор."""
    try:
        async with aiohttp.ClientSession() as session:
            return await fetch_weather(session, city)
    except WeatherUpstreamError as exc:
        logger.warning("weather upstream error: %s", exc)
        raise ToolError("Погодный сервис сейчас недоступен, попробуйте позже.") from exc


@mcp.tool(annotations=READ_ONLY)
def get_schedule(
    date: Annotated[
        str,
        StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}$"),
        Field(description="Дата в формате YYYY-MM-DD"),
    ],
) -> DaySchedule:
    """Занятия на выбранную дату. Пустой список lessons означает, что занятий нет."""
    try:
        day = dt.date.fromisoformat(date)
    except ValueError as exc:  # формат верный, но даты не существует, например 2026-02-30
        raise ToolError("Такой даты не существует.") from exc
    return day_schedule(_entries(), day, schedule_timezone())


@mcp.tool(annotations=WRITE_IDEMPOTENT)
async def add_reminder(
    text: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
        Field(description="Текст напоминания"),
    ],
    remind_at: Annotated[
        str,
        Field(
            description=(
                "Будущий момент времени в ISO 8601 с явным смещением часового пояса, "
                "например 2026-10-27T09:00:00+03:00. Передавай локальное время пользователя: "
                "приложение само подставляет его часовой пояс и проверяет результат."
            )
        ),
    ],
    ctx: Context,
) -> ReminderResult:
    """Создание напоминания. Выполняется только после подтверждения пользователя."""
    meta = ctx.request_context.meta
    trusted = read_trusted(meta.model_dump() if meta else None)
    if trusted is None:
        # Владельца нельзя получить из аргументов: без подписанного контекста приложения отказ.
        logger.warning("add_reminder refused: no trusted context")
        raise ToolError("Нет доверенного контекста приложения: создание напоминания запрещено.")
    try:
        moment = parse_remind_at(remind_at, dt.datetime.now(dt.UTC))
        return await store.create(trusted, text, moment)
    except ReminderRefusedError as exc:
        raise ToolError(str(exc)) from exc


@mcp.resource("schedule://current-week", mime_type="application/json")
def current_week() -> str:
    """Расписание текущей недели (понедельник-воскресенье), сгруппированное по датам."""
    timezone = schedule_timezone()
    return week_schedule(_entries(), today_in(timezone), timezone).model_dump_json()


def _entries():
    try:
        return load_entries()
    except ScheduleUnavailableError as exc:
        logger.warning("schedule error: %s", exc)
        raise ToolError("Расписание сейчас недоступно.") from exc


def forbid_extra_arguments(server: FastMCP) -> None:
    """SDK по умолчанию молча отбрасывает лишние поля; для нас это нарушение схемы.

    shortcut: используется внутренний API SDK (версия закреплена, тест это проверяет);
    при обновлении mcp заменить на публичную настройку, если она появится.
    """
    for tool in server._tool_manager.list_tools():
        model = tool.fn_metadata.arg_model
        model.model_config["extra"] = "forbid"
        model.model_rebuild(force=True)
        tool.parameters = model.model_json_schema(by_alias=True)  # схема для клиентов


forbid_extra_arguments(mcp)
