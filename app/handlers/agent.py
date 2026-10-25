"""Команды /timezone и /why и кнопки подтверждения действий агента."""

import logging
from datetime import UTC, datetime
from functools import cache
from zoneinfo import ZoneInfo, available_timezones

import asyncpg
from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from app.agent.tools import format_moment
from app.config import Settings
from app.mcp_client.client import MCPToolbox, ToolUnavailableError
from app.mcp_trust import TrustedContext, build_meta
from app.storage.agent_repository import AgentRepository, PendingAction

logger = logging.getLogger("app.agent")

TIMEZONE_HINT = "Нужно имя из базы IANA, например: /timezone Europe/Moscow"
ACTION_STATES = {
    "expired": "Время подтверждения истекло. Отправьте запрос ещё раз.",
    "executed": "Это действие уже выполнено.",
    "cancelled": "Это действие отменено или заменено новым запросом.",
    "confirmed": "Это действие уже выполняется или выполнено.",
    "failed": "Это действие уже завершилось ошибкой. Отправьте запрос ещё раз.",
}
ACTIONS = {
    "call_tool": "вызов инструмента",
    "respond": "ответ без инструмента",
    "clarify": "уточнение",
}
VALIDATION = {"passed": "пройдена", "failed": "не пройдена", "n/a": "не требовалась"}
EXECUTION = {
    "success": "успешно",
    "error": "ошибка",
    "pending_confirmation": "ожидает подтверждения",
    "not_executed": "не выполнялось",
    "limit": "остановлено по лимиту вызовов",
    "n/a": "—",
}


@cache
def _zones() -> frozenset[str]:
    return frozenset(available_timezones())


def confirm_keyboard(action_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Подтвердить", callback_data=f"act:ok:{action_id}"),
                InlineKeyboardButton(text="Отменить", callback_data=f"act:no:{action_id}"),
            ]
        ]
    )


async def cmd_timezone(message: Message, command: CommandObject, db: asyncpg.Pool) -> None:
    if not message.from_user:
        return
    repo = AgentRepository(db)
    name = (command.args or "").strip()
    if not name:
        current = await repo.get_timezone(message.from_user.id)
        text = (
            f"Ваш часовой пояс: {current}."
            if current
            else f"Часовой пояс не задан. {TIMEZONE_HINT}"
        )
        await message.answer(text, parse_mode=None)
        return
    if name not in _zones():
        await message.answer(f"Неизвестный часовой пояс. {TIMEZONE_HINT}", parse_mode=None)
        return
    await repo.set_timezone(message.from_user.id, name)
    local = datetime.now(UTC).astimezone(ZoneInfo(name))
    await message.answer(
        f"Часовой пояс сохранён: {name}. Сейчас там {local:%d.%m.%Y %H:%M}.", parse_mode=None
    )


async def cmd_why(message: Message, db: asyncpg.Pool, settings: Settings) -> None:
    """Последнее решение агента для этого пользователя; без промптов и чужих данных."""
    if not message.from_user:
        return
    repo = AgentRepository(db)
    record = await repo.last_audit(message.from_user.id)
    if record is None:
        await message.answer("Вы ещё не отправляли запросов агенту.", parse_mode=None)
        return
    zone = await repo.get_timezone(message.from_user.id) or settings.schedule_timezone
    when = record.created_at.astimezone(ZoneInfo(zone))
    action = ACTIONS.get(record.action, record.action)
    if record.tool_name:
        action += f" {record.tool_name}"
    lines = [f"Последний запрос: {when:%d.%m.%Y %H:%M} ({zone})", f"Действие: {action}"]
    if record.args_summary:
        lines.append(f"Аргументы: {record.args_summary}")
    lines.append(f"Проверка: {VALIDATION.get(record.validation, record.validation)}")
    lines.append(f"Выполнение: {EXECUTION.get(record.execution, record.execution)}")
    if record.reason:
        lines.append(f"Причина: {record.reason}")
    await message.answer("\n".join(lines), parse_mode=None)


async def _execute(
    action: PendingAction, repo: AgentRepository, toolbox: MCPToolbox, secret: str
) -> tuple[bool, str]:
    """Вызывает изменяющий инструмент с подписанным контекстом; успех только по факту."""
    meta = build_meta(secret, TrustedContext(action.user_id, action.timezone, action.id))
    try:
        result = await toolbox.call_tool(action.tool_name, action.arguments, meta=meta)
    except ToolUnavailableError:
        return False, "Не удалось создать напоминание: сервис сейчас недоступен. Повторите запрос."
    data = result.data or {}
    if result.is_error or not isinstance(data.get("reminder_id"), int):
        reason = result.text[:200] if result.is_error else "неожиданный ответ сервиса"
        return False, f"Не удалось создать напоминание: {reason}"
    when = f"{format_moment(data['remind_at'])} ({data['timezone']})"
    if data.get("created"):
        body = f"Текст: {action.arguments['text']}\nКогда: {when}"
        return True, f"Напоминание №{data['reminder_id']} создано.\n{body}"
    return True, f"Это напоминание уже было создано (№{data['reminder_id']}). Когда: {when}"


async def callback_action(
    query: CallbackQuery, db: asyncpg.Pool, toolbox: MCPToolbox, mcp_secret: str
) -> None:
    if not query.data or not query.message:
        await query.answer()
        return
    _, verb, action_id = query.data.split(":", 2)
    user_id = query.from_user.id
    repo = AgentRepository(db)

    if verb == "no":
        done = await repo.cancel_action(action_id, user_id)
        text = "Действие отменено." if done else await _why_not(repo, action_id, user_id)
    else:
        action = await repo.claim_action(action_id, user_id)
        if action is None:
            text = await _why_not(repo, action_id, user_id)
        else:
            ok, text = await _execute(action, repo, toolbox, mcp_secret)
            await repo.finish_action(action.id, user_id, ok)
            await repo.add_audit(
                user_id,
                action.request_id,
                "call_tool",
                tool_name=action.tool_name,
                validation="passed",
                execution="success" if ok else "error",
                reason="подтверждено пользователем",
            )
    await query.answer()
    await query.message.edit_text(text, reply_markup=None, parse_mode=None)


async def _why_not(repo: AgentRepository, action_id: str, user_id: int) -> str:
    state = await repo.action_state(action_id, user_id)
    return ACTION_STATES.get(state or "", "Действие не найдено.")


def create_agent_router() -> Router:
    r = Router(name="agent")
    r.message.filter(F.chat.type == "private")
    r.message.register(cmd_timezone, Command("timezone"))
    r.message.register(cmd_why, Command("why"))
    r.callback_query.register(callback_action, F.data.regexp(r"^act:(ok|no):[0-9a-f]{32}$"))
    return r


router = create_agent_router()
