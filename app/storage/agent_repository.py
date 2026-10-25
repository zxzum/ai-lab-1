"""Данные агента в PostgreSQL: часовой пояс, подтверждения действий и журнал аудита."""

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

import asyncpg

CONFIRMATION_TTL = timedelta(minutes=5)


@dataclass(frozen=True)
class PendingAction:
    id: str
    user_id: int
    request_id: str
    tool_name: str
    arguments: dict
    timezone: str
    status: str
    expires_at: datetime


@dataclass(frozen=True)
class AuditRecord:
    created_at: datetime
    action: str
    tool_name: str | None
    args_summary: str
    validation: str
    execution: str
    reason: str


def _action(row: asyncpg.Record) -> PendingAction:
    return PendingAction(
        id=row["id"],
        user_id=row["user_id"],
        request_id=row["request_id"],
        tool_name=row["tool_name"],
        arguments=json.loads(row["arguments"]),
        timezone=row["timezone"],
        status=row["status"],
        expires_at=row["expires_at"],
    )


class AgentRepository:
    """Все запросы фильтруются по user_id: чужие данные недоступны по построению."""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def get_timezone(self, user_id: int) -> str | None:
        return await self.pool.fetchval(
            "SELECT timezone FROM user_settings WHERE user_id = $1", user_id
        )

    async def set_timezone(self, user_id: int, timezone: str) -> None:
        await self.pool.execute(
            """
            INSERT INTO user_settings (user_id, timezone) VALUES ($1, $2)
            ON CONFLICT (user_id) DO UPDATE SET timezone = EXCLUDED.timezone, updated_at = NOW()
            """,
            user_id,
            timezone,
        )

    async def get_action_by_update(self, user_id: int, update_id: int) -> PendingAction | None:
        row = await self.pool.fetchrow(
            "SELECT * FROM pending_actions WHERE user_id = $1 AND update_id = $2",
            user_id,
            update_id,
        )
        return _action(row) if row else None

    async def create_action(
        self,
        user_id: int,
        update_id: int,
        request_id: str,
        tool_name: str,
        arguments: dict,
        timezone: str,
    ) -> tuple[PendingAction, bool]:
        """Идемпотентно по (user_id, update_id). Возвращает действие и признак «создано сейчас»."""
        action_id = uuid.uuid4().hex
        async with self.pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                """
                INSERT INTO pending_actions
                    (id, user_id, update_id, request_id, tool_name, arguments, timezone, expires_at)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, NOW() + $8)
                ON CONFLICT (user_id, update_id) DO NOTHING
                RETURNING *
                """,
                action_id,
                user_id,
                update_id,
                request_id,
                tool_name,
                json.dumps(arguments, ensure_ascii=False),
                timezone,
                CONFIRMATION_TTL,
            )
            if row is None:  # тот же update уже обработан: возвращаем прежнюю операцию
                existing = await conn.fetchrow(
                    "SELECT * FROM pending_actions WHERE user_id = $1 AND update_id = $2",
                    user_id,
                    update_id,
                )
                return _action(existing), False
            # Новое действие вытесняет старые неподтверждённые: подтверждение не переносится.
            await conn.execute(
                """
                UPDATE pending_actions SET status = 'cancelled'
                WHERE user_id = $1 AND status = 'pending' AND id <> $2
                """,
                user_id,
                action_id,
            )
            return _action(row), True

    async def claim_action(self, action_id: str, user_id: int) -> PendingAction | None:
        """Атомарно переводит pending -> confirmed. Выигрывает ровно один из конкурентов."""
        row = await self.pool.fetchrow(
            """
            UPDATE pending_actions SET status = 'confirmed'
            WHERE id = $1 AND user_id = $2 AND status = 'pending' AND expires_at > NOW()
            RETURNING *
            """,
            action_id,
            user_id,
        )
        return _action(row) if row else None

    async def action_state(self, action_id: str, user_id: int) -> str | None:
        """Причина, по которой claim не удался: not_found | expired | <статус>."""
        row = await self.pool.fetchrow(
            "SELECT status, expires_at < NOW() AS late FROM pending_actions"
            " WHERE id = $1 AND user_id = $2",
            action_id,
            user_id,
        )
        if row is None:
            return None
        if row["status"] == "pending" and row["late"]:
            await self.pool.execute(
                "UPDATE pending_actions SET status = 'expired'"
                " WHERE id = $1 AND user_id = $2 AND status = 'pending'",
                action_id,
                user_id,
            )
            return "expired"
        return row["status"]

    async def cancel_action(self, action_id: str, user_id: int) -> bool:
        result = await self.pool.execute(
            "UPDATE pending_actions SET status = 'cancelled'"
            " WHERE id = $1 AND user_id = $2 AND status = 'pending'",
            action_id,
            user_id,
        )
        return result.endswith(" 1")

    async def finish_action(self, action_id: str, user_id: int, ok: bool) -> None:
        await self.pool.execute(
            "UPDATE pending_actions SET status = $3 WHERE id = $1 AND user_id = $2",
            action_id,
            user_id,
            "executed" if ok else "failed",
        )

    async def add_audit(
        self,
        user_id: int,
        request_id: str,
        action: str,
        *,
        tool_name: str | None = None,
        args_summary: str = "",
        validation: str = "n/a",
        execution: str = "n/a",
        reason: str = "",
        duration_ms: int = 0,
    ) -> None:
        await self.pool.execute(
            """
            INSERT INTO agent_audit
                (user_id, request_id, action, tool_name, args_summary, validation,
                 execution, reason, duration_ms)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            user_id,
            request_id,
            action,
            tool_name,
            args_summary,
            validation,
            execution,
            reason,
            duration_ms,
        )

    async def last_audit(self, user_id: int) -> AuditRecord | None:
        row = await self.pool.fetchrow(
            """
            SELECT created_at, action, tool_name, args_summary, validation, execution, reason
            FROM agent_audit WHERE user_id = $1 ORDER BY id DESC LIMIT 1
            """,
            user_id,
        )
        return AuditRecord(**dict(row)) if row else None
