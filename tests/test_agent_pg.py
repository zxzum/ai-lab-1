"""Проверки на настоящем PostgreSQL: идемпотентность, конкурентные повторы, изоляция.

RUN_INTEGRATION=1 python -m pytest tests/test_agent_pg.py -v
"""

import asyncio
import os
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.filters import CommandObject

from app.db import create_pool, init_db
from app.handlers.agent import callback_action, cmd_timezone, cmd_why
from app.mcp_client.client import MCPToolbox, server_params
from app.mcp_server.reminders import ReminderStore
from app.mcp_trust import TrustedContext, build_meta, new_secret
from app.storage.agent_repository import AgentRepository

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1",
        reason="Включите RUN_INTEGRATION=1 для Docker-тестов",
    ),
]

ALICE, BOB = 1001, 2002


def in_days(days: int) -> datetime:
    return datetime.now(UTC) + timedelta(days=days)


def new_action(repo, user_id, update_id, text="Сдать лабу"):
    args = {"text": text, "remind_at": in_days(2).astimezone().isoformat()}
    return repo.create_action(user_id, update_id, "req00001", "add_reminder", args, "Europe/Moscow")


async def test_repository_and_store_guarantees_on_real_postgres(database):
    # Arrange
    settings, *_ = database
    pool = await create_pool(settings)
    try:
        await init_db(pool)
        await init_db(pool)  # повторный запуск приложения не ломает схему
        repo, store = AgentRepository(pool), ReminderStore(pool)

        # Act / Assert: один и тот же update, доставленный конкурентно, даёт одну операцию
        results = await asyncio.gather(*[new_action(repo, ALICE, 500) for _ in range(5)])
        assert len({a.id for a, _ in results}) == 1
        assert sum(created for _, created in results) == 1
        assert await pool.fetchval("SELECT count(*) FROM pending_actions") == 1
        action = results[0][0]

        # конкурентное двойное подтверждение: ровно один выигрывает
        claims = await asyncio.gather(*[repo.claim_action(action.id, ALICE) for _ in range(5)])
        assert sum(c is not None for c in claims) == 1

        # чужой пользователь не может ни подтвердить, ни отменить, ни увидеть состояние
        other, _ = await new_action(repo, ALICE, 501)
        assert await repo.claim_action(other.id, BOB) is None
        assert await repo.cancel_action(other.id, BOB) is False
        assert await repo.action_state(other.id, BOB) is None
        assert await repo.action_state(other.id, ALICE) == "pending"

        # новое действие вытесняет старое неподтверждённое
        newer, _ = await new_action(repo, ALICE, 502, text="Другое")
        assert await repo.action_state(other.id, ALICE) == "cancelled"
        assert await repo.claim_action(other.id, ALICE) is None
        assert await repo.claim_action(newer.id, ALICE) is not None

        # просроченное подтверждение недействительно
        stale, _ = await new_action(repo, BOB, 600)
        await pool.execute("UPDATE pending_actions SET expires_at = NOW() - interval '1 second'")
        assert await repo.claim_action(stale.id, BOB) is None
        assert await repo.action_state(stale.id, BOB) == "expired"

        # напоминания: повтор с тем же ключом не создаёт вторую запись
        ctx = TrustedContext(ALICE, "Europe/Moscow", "key-1")
        made = await asyncio.gather(*[store.create(ctx, "текст", in_days(1)) for _ in range(5)])
        assert len({r.reminder_id for r in made}) == 1
        assert sum(r.created for r in made) == 1
        assert made[0].remind_at.endswith("+03:00")
        assert await pool.fetchval("SELECT count(*) FROM reminders") == 1
        # тот же ключ у другого владельца — отдельное напоминание
        other_owner = await store.create(TrustedContext(BOB, "UTC", "key-1"), "t", in_days(1))
        assert other_owner.created and other_owner.reminder_id != made[0].reminder_id

        # аудит и часовой пояс изолированы по пользователям
        await repo.add_audit(ALICE, "r1", "respond", execution="success")
        await repo.add_audit(BOB, "r2", "call_tool", tool_name="get_weather")
        assert (await repo.last_audit(ALICE)).action == "respond"
        assert (await repo.last_audit(BOB)).tool_name == "get_weather"
        await repo.set_timezone(ALICE, "Asia/Tokyo")
        assert await repo.get_timezone(ALICE) == "Asia/Tokyo"
        assert await repo.get_timezone(BOB) is None
    finally:
        await pool.close()


async def test_add_reminder_through_real_mcp_and_confirmation_flow(database):
    # Arrange
    settings, *_ = database
    secret = new_secret()
    pool = await create_pool(settings)
    await init_db(pool)
    repo = AgentRepository(pool)
    toolbox = MCPToolbox(server_params(settings, secret))
    await toolbox.start()
    try:
        assert toolbox.available
        when = in_days(3).astimezone().isoformat()
        args = {"text": "Сдать лабу", "remind_at": when}
        meta = build_meta(secret, TrustedContext(ALICE, "Europe/Moscow", "k1"))

        # Act / Assert: подписанный вызов создаёт запись, повтор возвращает прежнюю
        first = await toolbox.call_tool("add_reminder", args, meta)
        again = await toolbox.call_tool("add_reminder", args, meta)
        assert (first.is_error, first.data["created"]) == (False, True)
        assert again.data["created"] is False
        assert again.data["reminder_id"] == first.data["reminder_id"]
        row = await pool.fetchrow("SELECT owner_id, text FROM reminders")
        assert (row["owner_id"], row["text"]) == (ALICE, "Сдать лабу")

        # независимый клиент без доверенного контекста получает отказ и ничего не пишет
        for bad_meta in (None, build_meta("чужой-секрет", TrustedContext(BOB, "UTC", "k2"))):
            refused = await toolbox.call_tool("add_reminder", args, bad_meta)
            assert refused.is_error
        # владелец в аргументах не принимается
        forged = await toolbox.call_tool("add_reminder", {**args, "owner_id": BOB}, meta)
        assert forged.is_error
        # прошлое время и пустой текст отклоняются
        past = {"text": "x", "remind_at": (datetime.now(UTC) - timedelta(days=1)).isoformat()}
        assert (await toolbox.call_tool("add_reminder", past, meta)).is_error
        assert (await toolbox.call_tool("add_reminder", {**args, "text": "  "}, meta)).is_error
        assert await pool.fetchval("SELECT count(*) FROM reminders") == 1

        # Act / Assert: кнопка «Подтвердить» через обработчик
        action, _ = await repo.create_action(
            ALICE, 700, "req00002", "add_reminder", {"text": "Через кнопку", "remind_at": when},
            "Europe/Moscow",
        )  # fmt: skip

        def press(verb, user_id, action_id):
            message = SimpleNamespace(edit_text=AsyncMock())
            query = SimpleNamespace(
                data=f"act:{verb}:{action_id}",
                from_user=SimpleNamespace(id=user_id),
                message=message,
                answer=AsyncMock(),
            )
            return query, message

        query, message = press("ok", BOB, action.id)  # чужой пользователь
        await callback_action(query, pool, toolbox, secret)
        assert message.edit_text.await_args.args[0] == "Действие не найдено."

        results = []
        for _ in range(2):  # двойное нажатие
            query, message = press("ok", ALICE, action.id)
            await callback_action(query, pool, toolbox, secret)
            results.append(message.edit_text.await_args.args[0])
        assert results[0].startswith("Напоминание №")
        assert "Через кнопку" in results[0]
        assert results[1] == "Это действие уже выполнено."
        assert (
            await pool.fetchval("SELECT count(*) FROM reminders WHERE text = 'Через кнопку'") == 1
        )
        assert (
            await pool.fetchval("SELECT status FROM pending_actions WHERE id = $1", action.id)
            == "executed"
        )

        # отмена до подтверждения
        cancel, _ = await repo.create_action(
            ALICE, 701, "req00003", "add_reminder", {"text": "Не создавать", "remind_at": when},
            "Europe/Moscow",
        )  # fmt: skip
        query, message = press("no", ALICE, cancel.id)
        await callback_action(query, pool, toolbox, secret)
        assert message.edit_text.await_args.args[0] == "Действие отменено."
        query, message = press("ok", ALICE, cancel.id)
        await callback_action(query, pool, toolbox, secret)
        assert "отменено" in message.edit_text.await_args.args[0]
        assert (
            await pool.fetchval("SELECT count(*) FROM reminders WHERE text = 'Не создавать'") == 0
        )
    finally:
        await toolbox.close()
        await pool.close()


async def test_timezone_and_why_commands(database):
    # Arrange
    settings, *_ = database
    pool = await create_pool(settings)
    await init_db(pool)

    def message(user_id):
        return SimpleNamespace(from_user=SimpleNamespace(id=user_id), answer=AsyncMock())

    async def reply(m):
        return m.answer.await_args.args[0]

    try:
        # Act / Assert: /timezone
        m = message(ALICE)
        await cmd_timezone(m, CommandObject(command="timezone", args=None), pool)
        assert (await reply(m)).startswith("Часовой пояс не задан.")
        m = message(ALICE)
        await cmd_timezone(m, CommandObject(command="timezone", args="Mars/Base"), pool)
        assert (await reply(m)).startswith("Неизвестный часовой пояс.")
        m = message(ALICE)
        await cmd_timezone(m, CommandObject(command="timezone", args="Europe/Moscow"), pool)
        assert (await reply(m)).startswith("Часовой пояс сохранён: Europe/Moscow. Сейчас там ")
        m = message(ALICE)
        await cmd_timezone(m, CommandObject(command="timezone", args=None), pool)
        assert await reply(m) == "Ваш часовой пояс: Europe/Moscow."
        m = message(BOB)  # у другого пользователя зона не задана
        await cmd_timezone(m, CommandObject(command="timezone", args=None), pool)
        assert (await reply(m)).startswith("Часовой пояс не задан.")

        # Act / Assert: /why
        repo = AgentRepository(pool)
        m = message(ALICE)
        await cmd_why(m, pool, SimpleNamespace(schedule_timezone="Europe/Moscow"))
        assert await reply(m) == "Вы ещё не отправляли запросов агенту."
        await repo.add_audit(
            ALICE, "r1", "call_tool", tool_name="get_weather", args_summary="город «Казань»",
            validation="passed", execution="success", reason="нужны актуальные погодные данные",
        )  # fmt: skip
        await repo.add_audit(BOB, "r2", "respond", execution="success", reason="секрет Боба")
        m = message(ALICE)
        await cmd_why(m, pool, SimpleNamespace(schedule_timezone="Europe/Moscow"))
        text = await reply(m)
        lines = text.split("\n")
        assert lines[0].startswith("Последний запрос: ") and lines[0].endswith("(Europe/Moscow)")
        assert lines[1:] == [
            "Действие: вызов инструмента get_weather",
            "Аргументы: город «Казань»",
            "Проверка: пройдена",
            "Выполнение: успешно",
            "Причина: нужны актуальные погодные данные",
        ]
        assert "Боба" not in text
    finally:
        await pool.close()
