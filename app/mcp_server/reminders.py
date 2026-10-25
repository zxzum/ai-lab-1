"""Создание напоминаний: проверка аргументов и идемпотентная запись в PostgreSQL."""

import os
from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import asyncpg
from pydantic import BaseModel, Field

from app.mcp_trust import TrustedContext


class ReminderRefusedError(Exception):
    """Вызов отклонён; сообщение безопасно показывать пользователю."""


class ReminderResult(BaseModel):
    reminder_id: int
    remind_at: str = Field(description="Нормализованное время в часовом поясе пользователя")
    timezone: str = Field(description="Часовой пояс пользователя (IANA)")
    created: bool = Field(description="False, если это повтор уже подтверждённого действия")


def parse_remind_at(value: str, now: datetime) -> datetime:
    """ISO 8601 с явным смещением, строго в будущем."""
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        raise ReminderRefusedError("Время напоминания должно быть в формате ISO 8601.") from None
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ReminderRefusedError("У времени напоминания должен быть явный часовой пояс.")
    if moment <= now:
        raise ReminderRefusedError("Время напоминания уже прошло.")
    return moment


def check_timezone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        raise ReminderRefusedError("Неизвестный часовой пояс.") from None


class ReminderStore:
    """Пул соединений создаётся при первом обращении; реквизиты берутся из окружения сервера."""

    def __init__(self, pool: asyncpg.Pool | None = None):
        self._pool = pool

    async def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            try:
                self._pool = await asyncpg.create_pool(
                    host=os.environ.get("POSTGRES_HOST", "127.0.0.1"),
                    port=int(os.environ.get("POSTGRES_PORT", "5432")),
                    database=os.environ.get("POSTGRES_DB", "bot"),
                    user=os.environ.get("POSTGRES_USER", "bot"),
                    password=os.environ["POSTGRES_PASSWORD"],
                    min_size=1,
                    max_size=2,
                    timeout=5,
                    command_timeout=5,
                )
            except (KeyError, OSError, asyncpg.PostgresError, ValueError):
                raise ReminderRefusedError("Хранилище напоминаний сейчас недоступно.") from None
        return self._pool

    async def create(
        self, ctx: TrustedContext, text: str, remind_at: datetime, now: datetime | None = None
    ) -> ReminderResult:
        zone = check_timezone(ctx.timezone)
        moment = parse_remind_at(remind_at.isoformat(), now or datetime.now(UTC))
        pool = await self.pool()
        try:
            async with pool.acquire() as conn:
                # Повтор с тем же ключом возвращает прежнюю запись, а не создаёт новую.
                row = await conn.fetchrow(
                    """
                    INSERT INTO reminders (owner_id, text, remind_at, timezone, idempotency_key)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (owner_id, idempotency_key) DO NOTHING
                    RETURNING id, remind_at
                    """,
                    ctx.owner_id,
                    text,
                    moment,
                    ctx.timezone,
                    ctx.idempotency_key,
                )
                created = row is not None
                if row is None:
                    row = await conn.fetchrow(
                        "SELECT id, remind_at FROM reminders"
                        " WHERE owner_id = $1 AND idempotency_key = $2",
                        ctx.owner_id,
                        ctx.idempotency_key,
                    )
        except (OSError, asyncpg.PostgresError):
            raise ReminderRefusedError("Хранилище напоминаний сейчас недоступно.") from None
        return ReminderResult(
            reminder_id=row["id"],
            remind_at=row["remind_at"].astimezone(zone).isoformat(),
            timezone=ctx.timezone,
            created=created,
        )
