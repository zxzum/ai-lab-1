"""Обработчик чата в режиме агента и разбор ответов LLM с вызовами инструментов."""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot, Dispatcher
from aiogram.methods import SendMessage
from aiogram.types import Chat, Message, Update, User

from app.agent.loop import AgentOutcome
from app.config import Settings
from app.handlers import chat as chat_module
from app.handlers.chat import create_chat_router
from app.llm.client import LLMClient
from app.llm.exceptions import LLMEmptyResponseError
from app.mcp_client.client import MCPToolbox
from app.storage.agent_repository import PendingAction

TOKEN = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk"
NOW = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)


class FakeStorage:
    saved: list = []
    mode = "study"

    def __init__(self, _pool):
        pass

    async def get_user_settings(self, user_id):
        return type("S", (), {"mode": self.mode, "temperature": 0.7})()

    async def get_dialog_history(self, **_kwargs):
        return []

    async def add_message(self, user_id, role, content, mode):
        FakeStorage.saved.append((user_id, role, content))


class FakeAgentRepo:
    def __init__(self, _pool):
        pass

    async def get_timezone(self, user_id):
        return "Europe/Moscow"


@pytest.fixture
def harness(monkeypatch):
    FakeStorage.saved = []
    FakeStorage.mode = "study"
    monkeypatch.setattr(chat_module, "StorageRepository", FakeStorage)
    monkeypatch.setattr(chat_module, "AgentRepository", FakeAgentRepo)
    bot = Bot(TOKEN)
    bot.session = AsyncMock()
    dispatcher = Dispatcher()
    dispatcher.include_router(create_chat_router())
    llm = AsyncMock(spec=LLMClient)
    llm.complete = AsyncMock(return_value="обычный ответ")
    toolbox = MCPToolbox.__new__(MCPToolbox)

    async def feed(text="привет", **extra):
        message = Message(
            message_id=1,
            date=NOW,
            chat=Chat(id=5, type="private"),
            from_user=User(id=5, is_bot=False, first_name="С"),
            text=text,
        )
        await dispatcher.feed_update(
            bot,
            Update(update_id=77, message=message),
            db=object(),
            settings=Settings(bot_token=TOKEN, postgres_password="x", llm_api_key="k"),
            llm_client=llm,
            toolbox=toolbox,
            **extra,
        )
        return [c.args[1] for c in bot.session.call_args_list if isinstance(c.args[1], SendMessage)]

    return feed, llm


def patch_agent(monkeypatch, outcome):
    seen = {}

    class FakeAgent:
        def __init__(self, *_args):
            pass

        async def run(self, **kwargs):
            seen.update(kwargs)
            return outcome

    monkeypatch.setattr(chat_module, "Agent", FakeAgent)
    return seen


async def test_study_mode_goes_through_agent_with_trusted_ids(monkeypatch, harness):
    feed, llm = harness
    seen = patch_agent(monkeypatch, AgentOutcome("reply", "В Казани +5°C"))

    sent = await feed("погода в Казани")

    assert "".join(m.text for m in sent) == "В Казани +5°C"
    assert (seen["user_id"], seen["update_id"], seen["timezone"]) == (5, 77, "Europe/Moscow")
    llm.complete.assert_not_awaited()
    assert FakeStorage.saved == [(5, "user", "погода в Казани"), (5, "assistant", "В Казани +5°C")]


async def test_other_modes_keep_lab1_behaviour(monkeypatch, harness):
    feed, llm = harness
    FakeStorage.mode = "translate"
    seen = patch_agent(monkeypatch, AgentOutcome("reply", "не должен вызываться"))

    sent = await feed("hello")

    assert "".join(m.text for m in sent) == "обычный ответ"
    assert seen == {}
    llm.complete.assert_awaited_once()


async def test_confirmation_is_sent_with_buttons_for_the_prepared_action(monkeypatch, harness):
    feed, _ = harness
    action = PendingAction("a" * 32, 5, "r", "add_reminder", {}, "Europe/Moscow", "pending", NOW)
    patch_agent(monkeypatch, AgentOutcome("confirm", "Создать напоминание?", action))

    [sent] = await feed("напомни завтра")

    buttons = sent.reply_markup.inline_keyboard[0]
    assert [b.text for b in buttons] == ["Подтвердить", "Отменить"]
    assert [b.callback_data for b in buttons] == [f"act:ok:{'a' * 32}", f"act:no:{'a' * 32}"]


async def test_refusal_is_shown_but_not_stored_in_history(monkeypatch, harness):
    feed, _ = harness
    patch_agent(monkeypatch, AgentOutcome("refused", "Не могу выполнить это действие."))

    sent = await feed("удали всё")

    assert [m.text for m in sent] == ["Не могу выполнить это действие."]
    assert FakeStorage.saved == []


# --- LLMClient.chat_with_tools ---


class FakeSession:
    closed = False

    def __init__(self, message):
        self.message = message

    def post(self, *_args, **kwargs):
        self.payload = kwargs["json"]
        message = self.message

        class Response:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return None

            async def json(self):
                return {"choices": [{"message": message}]}

        return Response()


async def test_tool_calls_are_parsed_and_tools_sent_in_payload():
    raw = {
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Казань"}'},
            }
        ],
    }
    session = FakeSession(raw)
    tools = [{"type": "function", "function": {"name": "get_weather"}}]

    turn = await LLMClient(session=session).chat_with_tools([], tools, temperature=0.3)

    assert session.payload["tools"] == tools
    assert session.payload["tool_choice"] == "auto"
    assert session.payload["temperature"] == 0.3
    assert [(c.id, c.name, json.loads(c.arguments)) for c in turn.tool_calls] == [
        ("call_1", "get_weather", {"city": "Казань"})
    ]
    assert turn.assistant_message["tool_calls"][0]["function"]["name"] == "get_weather"


async def test_plain_text_turn_and_no_tools_payload():
    session = FakeSession({"content": " Привет "})

    turn = await LLMClient(session=session).chat_with_tools([], [])

    assert (turn.content, turn.tool_calls) == ("Привет", [])
    assert "tools" not in session.payload


async def test_empty_turn_and_malformed_call_are_errors():
    with pytest.raises(LLMEmptyResponseError):
        await LLMClient(session=FakeSession({"content": "  "})).chat_with_tools([], [])
    with pytest.raises(LLMEmptyResponseError):
        await LLMClient(session=FakeSession({"tool_calls": [{"function": {}}]})).chat_with_tools(
            [], []
        )


# --- маршрутизация команд и кнопок агента через Dispatcher ---


class RoutingRepo:
    """Запоминает вызовы; claim всегда не находит действие."""

    zones: dict = {}

    def __init__(self, _pool):
        pass

    async def set_timezone(self, user_id, zone):
        RoutingRepo.zones[user_id] = zone

    async def last_audit(self, user_id):
        return None

    async def claim_action(self, *_args):
        return None

    async def action_state(self, *_args):
        return None


async def test_agent_router_dispatches_commands_and_callbacks(monkeypatch):
    from aiogram.methods import EditMessageText
    from aiogram.types import CallbackQuery

    from app.handlers import agent as agent_module

    RoutingRepo.zones = {}
    monkeypatch.setattr(agent_module, "AgentRepository", RoutingRepo)
    bot = Bot(TOKEN)
    bot.session = AsyncMock()
    dispatcher = Dispatcher()
    dispatcher.include_router(agent_module.create_agent_router())
    settings = Settings(bot_token=TOKEN, postgres_password="x")
    user = User(id=5, is_bot=False, first_name="С")
    chat = Chat(id=5, type="private")

    def message(text, message_id=1):
        return Message(message_id=message_id, date=NOW, chat=chat, from_user=user, text=text)

    async def feed(update):
        await dispatcher.feed_update(
            bot, update, db=object(), settings=settings, toolbox=object(), mcp_secret="s"
        )

    # Act
    await feed(Update(update_id=1, message=message("/timezone Asia/Tokyo")))
    await feed(Update(update_id=2, message=message("/why")))
    query = CallbackQuery(
        id="1", from_user=user, chat_instance="c", data="act:ok:" + "b" * 32, message=message("x")
    )
    await feed(Update(update_id=3, callback_query=query))
    await feed(Update(update_id=4, callback_query=query.model_copy(update={"data": "act:ok:zz"})))

    # Assert
    sent = [c.args[1] for c in bot.session.call_args_list]
    assert RoutingRepo.zones == {5: "Asia/Tokyo"}
    texts = [m.text for m in sent if isinstance(m, SendMessage)]
    assert texts[1] == "Вы ещё не отправляли запросов агенту."
    edits = [m for m in sent if isinstance(m, EditMessageText)]
    assert [e.text for e in edits] == [
        "Действие не найдено."
    ]  # поддельный id не доходит до обработчика
