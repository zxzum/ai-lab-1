"""Ограниченный агентный цикл: модель предлагает действие, приложение проверяет и выполняет."""

import json
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from app.agent.prompts import AGENT_RULES, NO_TIMEZONE_NOTE, NO_TOOLS_NOTE
from app.agent.tools import (
    CLARIFY,
    MAX_MCP_CALLS,
    RATIONALE,
    ArgumentError,
    ReminderTimeError,
    describe_now,
    expose_tools,
    format_moment,
    localize_reminder,
    summarize_arguments,
    validate_arguments,
)
from app.llm.client import LLMClient, LLMTurn, ToolCall
from app.llm.exceptions import LLMEmptyResponseError
from app.mcp_client.client import MCPToolbox, ToolResult, ToolUnavailableError
from app.storage.agent_repository import AgentRepository, PendingAction

logger = logging.getLogger("app.agent")

REMINDER_TOOL = "add_reminder"
MAX_LLM_ROUNDS = 4
TOOL_RESULT_LIMIT = 6000

NO_TIMEZONE_REPLY = (
    "Сначала задайте часовой пояс командой /timezone <зона>, например "
    "/timezone Europe/Moscow: от него зависит время напоминания."
)
LIMIT_REPLY = (
    f"За один запрос можно обратиться к инструментам не более {MAX_MCP_CALLS} раз. "
    "Я остановился; уточните запрос или разбейте его на части."
)
UNAVAILABLE_REPLY = "Инструмент сейчас недоступен. Попробуйте позже."
REFUSED_REPLY = "Не могу выполнить это действие: такого инструмента нет среди доступных."


@dataclass
class AgentOutcome:
    """reply — ответ; refused — отказ/ошибка; confirm — нужна кнопка.

    from_tools: ответ построен на данных инструментов. Такие обмены в историю не пишутся:
    модель копирует прошлые ответы с цифрами вместо нового вызова инструмента.
    """

    kind: Literal["reply", "refused", "confirm"]
    text: str
    action: PendingAction | None = None
    from_tools: bool = False


def build_system_prompt(
    mode_prompt: str, now: datetime, timezone: str | None, default_timezone: str, has_tools: bool
) -> str:
    """Инструкция режима + правила работы с инструментами + текущее время пользователя."""
    shown = timezone or default_timezone
    rules = AGENT_RULES.format(
        now=describe_now(now, shown),
        timezone=shown,
        tz_note=("" if has_tools else NO_TOOLS_NOTE) + ("" if timezone else NO_TIMEZONE_NOTE),
    )
    return f"{mode_prompt}\n\n{rules}"


def preview_text(action: PendingAction) -> str:
    args = action.arguments
    return (
        "Создать напоминание?\n"
        f"Текст: {args['text']}\n"
        f"Когда: {format_moment(args['remind_at'])} ({action.timezone})\n"
        "Подтвердить можно в течение 5 минут."
    )


def _tool_message(call: ToolCall, name: str, result: ToolResult) -> dict:
    # Результат — данные с указанием источника, а не инструкция для системы.
    payload = {"source": f"mcp:{name}"}
    payload["error" if result.is_error else "untrusted_data"] = (
        result.text if result.is_error else result.data
    )
    content = json.dumps(payload, ensure_ascii=False)[:TOOL_RESULT_LIMIT]
    return {"role": "tool", "tool_call_id": call.id, "content": content}


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        toolbox: MCPToolbox,
        repo: AgentRepository,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self.llm = llm
        self.toolbox = toolbox
        self.repo = repo
        self.clock = clock

    async def _next_turn(
        self, messages: list[dict], exposed: list[dict], temperature: float
    ) -> LLMTurn:
        try:
            return await self.llm.chat_with_tools(messages, exposed, temperature)
        except LLMEmptyResponseError:  # модель иногда отвечает пустотой: один повтор
            return await self.llm.chat_with_tools(messages, exposed, temperature)

    async def run(
        self,
        *,
        user_id: int,
        update_id: int,
        text: str,
        history: list[dict],
        mode_prompt: str,
        temperature: float,
        timezone: str | None,
        default_timezone: str,
    ) -> AgentOutcome:
        started = time.monotonic()
        request_id = uuid.uuid4().hex[:8]

        # Повторная доставка того же update: прежняя подготовленная операция, а не новая.
        existing = await self.repo.get_action_by_update(user_id, update_id)
        if existing is not None:
            if existing.status == "pending":
                return AgentOutcome("confirm", preview_text(existing), existing)
            return AgentOutcome("refused", "Этот запрос уже обработан.")

        tools = {t.name: t for t in self.toolbox.tools}

        now = self.clock()
        system = build_system_prompt(mode_prompt, now, timezone, default_timezone, bool(tools))
        messages: list[dict] = [{"role": "system", "content": system}]
        messages.extend(history)
        messages.append({"role": "user", "content": text})
        exposed = expose_tools(list(tools.values()))

        def duration() -> int:
            return int((time.monotonic() - started) * 1000)

        async def audit(action: str, **fields: str | None) -> None:
            await self.repo.add_audit(user_id, request_id, action, duration_ms=duration(), **fields)
            logger.info(
                "agent: req_id=%s action=%s tool=%s validation=%s execution=%s time_ms=%d",
                request_id,
                action,
                fields.get("tool_name"),
                fields.get("validation", "n/a"),
                fields.get("execution", "n/a"),
                duration(),
            )

        mcp_calls = 0
        decided = False  # было ли уже записано решение о вызове инструмента

        for _ in range(MAX_LLM_ROUNDS):
            turn = await self._next_turn(messages, exposed, temperature)
            if not turn.tool_calls:
                if not decided:
                    await audit("respond", reason="ответ без инструмента", execution="success")
                return AgentOutcome("reply", turn.content, from_tools=mcp_calls > 0)

            messages.append(turn.assistant_message)
            for call in turn.tool_calls:
                name = call.name
                if name == CLARIFY:
                    question = _parse_arguments(call.arguments).get("question")
                    if not isinstance(question, str) or not question.strip():
                        await audit("clarify", validation="failed", execution="not_executed")
                        return AgentOutcome("refused", REFUSED_REPLY)
                    await audit(
                        "clarify",
                        reason="не хватает данных для ответа",
                        validation="passed",
                        execution="success",
                    )
                    return AgentOutcome("reply", question.strip())

                tool = tools.get(name)
                try:
                    if tool is None:
                        raise ArgumentError(REFUSED_REPLY)
                    arguments = _parse_arguments(call.arguments)
                    reason = str(arguments.pop(RATIONALE, ""))[:200]
                    validate_arguments(tool, arguments)
                    if not tool.read_only:
                        if name != REMINDER_TOOL:  # других изменяющих действий агент не выполняет
                            raise ArgumentError(REFUSED_REPLY)
                        if timezone is None:
                            raise ReminderTimeError(NO_TIMEZONE_REPLY)
                        arguments = localize_reminder(arguments, timezone, now)
                except (ArgumentError, ReminderTimeError) as exc:
                    await audit(
                        "call_tool",
                        tool_name=name[:64],
                        validation="failed",
                        execution="not_executed",
                        reason=str(exc)[:200],
                    )
                    return AgentOutcome("refused", str(exc))

                summary = summarize_arguments(arguments)
                common = {"tool_name": name, "args_summary": summary, "reason": reason}
                if not tool.read_only:
                    action, created = await self.repo.create_action(
                        user_id, update_id, request_id, name, arguments, timezone
                    )
                    if created:
                        await audit(
                            "call_tool",
                            validation="passed",
                            execution="pending_confirmation",
                            **common,
                        )
                    return AgentOutcome("confirm", preview_text(action), action)

                if mcp_calls >= MAX_MCP_CALLS:
                    await audit("call_tool", validation="passed", execution="limit", **common)
                    return AgentOutcome("refused", LIMIT_REPLY)
                mcp_calls += 1
                decided = True
                try:
                    result = await self.toolbox.call_tool(name, arguments)
                except ToolUnavailableError:
                    await audit("call_tool", validation="passed", execution="error", **common)
                    return AgentOutcome("refused", UNAVAILABLE_REPLY)
                await audit(
                    "call_tool",
                    validation="passed",
                    execution="error" if result.is_error else "success",
                    **common,
                )
                messages.append(_tool_message(call, name, result))

        return AgentOutcome("refused", "Не удалось завершить запрос. Попробуйте переформулировать.")


def _parse_arguments(raw: str) -> dict:
    try:
        parsed = json.loads(raw or "{}")
    except ValueError:
        raise ArgumentError("Некорректные аргументы инструмента.") from None
    if not isinstance(parsed, dict):
        raise ArgumentError("Некорректные аргументы инструмента.")
    return parsed
