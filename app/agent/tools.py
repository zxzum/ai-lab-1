"""Схемы инструментов для модели, проверка аргументов и работа с локальным временем."""

import copy
from datetime import datetime
from zoneinfo import ZoneInfo

from jsonschema import Draft202012Validator

from app.mcp_client.client import ToolInfo

RATIONALE = "rationale"
CLARIFY = "ask_clarification"
MAX_MCP_CALLS = 2

WEEKDAYS = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
LABELS = {"city": "город", "date": "дата", "text": "текст", "remind_at": "время"}


class ArgumentError(Exception):
    """Аргументы не прошли проверку; сообщение безопасно показывать пользователю."""


class ReminderTimeError(Exception):
    """Время напоминания нельзя однозначно определить; сообщение для пользователя."""


def expose_tools(tools: list[ToolInfo]) -> list[dict]:
    """Описание инструментов для модели: схемы сервера как есть + поле rationale."""
    exposed = []
    for tool in tools:
        schema = copy.deepcopy(tool.input_schema)
        schema.setdefault("properties", {})[RATIONALE] = {
            "type": "string",
            "maxLength": 200,
            "description": "Одна короткая фраза: почему нужен этот инструмент",
        }
        schema["required"] = [*schema.get("required", []), RATIONALE]
        exposed.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": schema,
                },
            }
        )
    if tools:
        exposed.append(
            {
                "type": "function",
                "function": {
                    "name": CLARIFY,
                    "description": "Задать пользователю уточняющий вопрос, если не хватает "
                    "данных для вызова инструмента или запрос неоднозначен.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string", "minLength": 1, "maxLength": 300}
                        },
                        "required": ["question"],
                        "additionalProperties": False,
                    },
                },
            }
        )
    return exposed


def validate_arguments(tool: ToolInfo, arguments: dict) -> None:
    """Проверка по схеме, полученной от MCP-сервера (типы, длины, лишние поля)."""
    error = next(iter(Draft202012Validator(tool.input_schema).iter_errors(arguments)), None)
    if error is not None:
        field = ".".join(str(p) for p in error.path) or "аргументы"
        raise ArgumentError(f"Некорректные аргументы инструмента ({field}).")


def summarize_arguments(arguments: dict) -> str:
    """Безопасная краткая запись аргументов для /why и аудита."""
    parts = [f"{LABELS.get(k, k)} «{str(v)[:80]}»" for k, v in arguments.items()]
    return "; ".join(parts)


def localize_reminder(arguments: dict, timezone: str, now: datetime) -> dict:
    """Приводит время напоминания к часовому поясу пользователя или просит уточнить.

    Модель не выбирает зону: локальное время трактуется только в сохранённой зоне,
    а смещение, не совпадающее с зоной пользователя, отклоняется.
    """
    zone = ZoneInfo(timezone)
    try:
        moment = datetime.fromisoformat(arguments["remind_at"])
    except (KeyError, ValueError):
        raise ArgumentError("Не удалось разобрать время напоминания.") from None
    if moment.tzinfo is None:
        first, second = moment.replace(tzinfo=zone, fold=0), moment.replace(tzinfo=zone, fold=1)
        if first.utcoffset() != second.utcoffset():
            raise ReminderTimeError(
                "Это время неоднозначно или не существует из-за перехода на летнее/зимнее "
                "время. Уточните, пожалуйста, другое время."
            )
        moment = first
    elif moment.utcoffset() != moment.astimezone(zone).utcoffset():
        raise ReminderTimeError(
            "Часовой пояс времени не совпадает с вашим. Назовите время ещё раз."
        )
    moment = moment.astimezone(zone)
    if moment <= now:
        raise ReminderTimeError("Это время уже прошло. Назовите время в будущем.")
    return {"text": arguments["text"], "remind_at": moment.isoformat()}


def format_moment(iso: str) -> str:
    return datetime.fromisoformat(iso).strftime("%d.%m.%Y %H:%M")


def describe_now(now: datetime, timezone: str) -> str:
    local = now.astimezone(ZoneInfo(timezone))
    return f"{WEEKDAYS[local.weekday()]}, {local:%Y-%m-%d %H:%M}"
