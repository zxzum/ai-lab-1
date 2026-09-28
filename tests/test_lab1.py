"""Автоматические тесты для Лабораторной работы № 1.

Покрывают все обязательные сценарии ТЗ:
1. Обычное сообщение передаётся модели с системной инструкцией активного режима.
2. Истории двух пользователей не смешиваются.
3. При превышении лимита применяется стратегия сокращения
   (без потери текущего запроса и нарушения порядка).
4. Смена режима очищает историю предыдущего режима и сохраняет новую настройку.
5. /reset очищает только историю вызвавшего пользователя.
6. /settings принимает разрешённое и отклоняет некорректное значение.
7. Ошибка LLM преобразуется в безопасное сообщение без сохранения ответа ассистента в историю.
8. Ответ длиннее 4096 символов разбивается и отправляется по частям без потери текста.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot, Dispatcher
from aiogram.methods import SendMessage
from aiogram.types import Chat, Message, Update, User

from app.config import ConfigError, Settings
from app.handlers.chat import create_chat_router
from app.handlers.commands import create_commands_router
from app.llm.client import LLMClient
from app.llm.exceptions import (
    LLMAuthError,
    LLMEmptyResponseError,
    LLMRateLimitError,
    LLMServiceError,
    LLMTimeoutError,
)
from app.llm.prompts import CODE_REVIEW_PROMPT, STUDY_PROMPT
from app.storage.repository import StorageRepository
from app.utils.text import markdown_to_telegram_html, split_message

TOKEN = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk"


class FakePool:
    """Легковесный in-memory пул для тестирования репозитория без внешнего PostgreSQL."""

    def __init__(self):
        self.settings = {}  # user_id -> {"mode": str, "temperature": float}
        self.messages = []  # list of dicts

    async def fetchrow(self, query, *args):
        if "user_settings" in query:
            user_id = args[0]
            if user_id in self.settings:
                return {
                    "mode": self.settings[user_id]["mode"],
                    "temperature": self.settings[user_id]["temperature"],
                }
            return None
        return None

    async def execute(self, query, *args):
        if "INSERT INTO user_settings" in query:
            user_id, mode, temp = args[0], args[1], args[2]
            self.settings[user_id] = {"mode": mode, "temperature": temp}
            return "INSERT 0 1"
        elif "INSERT INTO dialog_messages" in query:
            user_id, role, content, mode = args[0], args[1], args[2], args[3]
            self.messages.append(
                {
                    "id": len(self.messages) + 1,
                    "user_id": user_id,
                    "role": role,
                    "content": content,
                    "mode": mode,
                }
            )
            return "INSERT 0 1"
        elif "DELETE FROM dialog_messages" in query:
            user_id = args[0]
            if len(args) > 1:
                mode = args[1]
                before = len(self.messages)
                self.messages = [
                    m for m in self.messages if not (m["user_id"] == user_id and m["mode"] == mode)
                ]
                return f"DELETE {before - len(self.messages)}"
            else:
                before = len(self.messages)
                self.messages = [m for m in self.messages if m["user_id"] != user_id]
                return f"DELETE {before - len(self.messages)}"
        return "OK"

    async def fetch(self, query, *args):
        if "dialog_messages" in query:
            user_id, mode, limit = args[0], args[1], args[2]
            matched = [m for m in self.messages if m["user_id"] == user_id and m["mode"] == mode]
            # Имитируем ORDER BY created_at DESC LIMIT limit
            matched = list(reversed(matched))[:limit]
            return [{"role": m["role"], "content": m["content"]} for m in matched]
        return []


def create_message(
    user_id: int = 42,
    text: str = "Привет",
    message_id: int = 1,
) -> Message:
    """Создает объект текстового сообщения от пользователя в приватном чате."""
    return Message(
        message_id=message_id,
        date=datetime.now(UTC),
        chat=Chat(id=user_id, type="private"),
        from_user=User(id=user_id, is_bot=False, first_name="Студент"),
        text=text,
    )


@pytest.fixture
def fake_pool():
    return FakePool()


@pytest.fixture
def test_settings():
    return Settings(
        bot_token=TOKEN,
        postgres_password="secret",
        llm_api_key="sk-test-key",
        llm_model="gpt-4o-mini",
        max_history_messages=10,
        max_history_chars=4000,
    )


@pytest.fixture
def test_dispatcher():
    dp = Dispatcher()
    dp.include_router(create_commands_router())
    dp.include_router(create_chat_router())
    return dp


async def test_normal_message_sent_with_active_mode_instruction(
    fake_pool, test_settings, test_dispatcher
):
    """Сценарий 1: Обычное сообщение передаётся модели с инструкцией активного режима."""
    # Arrange
    bot = Bot(TOKEN)
    bot.session = AsyncMock()
    llm_client = AsyncMock(spec=LLMClient)
    llm_client.complete = AsyncMock(return_value="Рекурсия — это вызов функции из самой себя.")

    message = create_message(user_id=100, text="Что такое рекурсия?")

    # Act
    await test_dispatcher.feed_update(
        bot,
        Update(update_id=1, message=message),
        db=fake_pool,
        settings=test_settings,
        llm_client=llm_client,
    )

    # Assert
    llm_client.complete.assert_awaited_once()
    call_kwargs = llm_client.complete.call_args.kwargs
    messages = call_kwargs["messages"]
    temperature = call_kwargs["temperature"]

    # Проверяем, что активный режим по умолчанию — study
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == STUDY_PROMPT
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "Что такое рекурсия?"
    assert temperature == 0.7

    # Проверяем ответ пользователю
    method = bot.session.call_args_list[-1].args[1]
    assert isinstance(method, SendMessage)
    assert method.chat_id == 100
    assert "Рекурсия" in method.text


async def test_two_users_histories_do_not_mix(fake_pool, test_settings, test_dispatcher):
    """Сценарий 2: Истории двух пользователей не смешиваются."""
    # Arrange
    bot = Bot(TOKEN)
    bot.session = AsyncMock()
    llm_client = AsyncMock(spec=LLMClient)
    llm_client.complete = AsyncMock(side_effect=["Ответ юзеру 1", "Ответ юзеру 2"])

    msg_user1 = create_message(user_id=1, text="Вопрос первого пользователя")
    msg_user2 = create_message(user_id=2, text="Вопрос второго пользователя")

    # Act
    await test_dispatcher.feed_update(
        bot,
        Update(update_id=1, message=msg_user1),
        db=fake_pool,
        settings=test_settings,
        llm_client=llm_client,
    )
    await test_dispatcher.feed_update(
        bot,
        Update(update_id=2, message=msg_user2),
        db=fake_pool,
        settings=test_settings,
        llm_client=llm_client,
    )

    # Assert
    call2_messages = llm_client.complete.call_args_list[1].kwargs["messages"]
    # В запросе пользователя 2 не должно быть данных пользователя 1
    contents = [m["content"] for m in call2_messages]
    assert "Вопрос первого пользователя" not in contents
    assert "Ответ юзеру 1" not in contents
    assert "Вопрос второго пользователя" in contents


async def test_context_truncation_preserves_order_and_current_query(fake_pool):
    """Сценарий 3: При превышении лимита сохраняется порядок и не теряется текущий запрос."""
    # Arrange
    repo = StorageRepository(fake_pool)
    user_id = 55
    mode = "study"

    # Добавляем 8 сообщений в историю
    for i in range(1, 9):
        role = "user" if i % 2 == 1 else "assistant"
        await repo.add_message(user_id, role, f"Сообщение {i}", mode)

    # Act 1: Проверяем ограничение по числу сообщений (max_messages=4)
    history = await repo.get_dialog_history(user_id, mode, max_messages=4, max_chars=4000)
    assert len(history) == 4
    # Порядок должен быть строго хронологическим (от старых к новым): 5, 6, 7, 8
    assert [m["content"] for m in history] == [
        "Сообщение 5",
        "Сообщение 6",
        "Сообщение 7",
        "Сообщение 8",
    ]

    # Act 2: Проверяем ограничение по объему символов
    # Каждое сообщение около 11 символов, ограничим max_chars=25 -> влезут только 2 последних
    history_chars = await repo.get_dialog_history(user_id, mode, max_messages=10, max_chars=25)
    assert len(history_chars) == 2
    assert [m["content"] for m in history_chars] == ["Сообщение 7", "Сообщение 8"]


async def test_context_does_not_include_single_message_over_char_limit(fake_pool):
    repo = StorageRepository(fake_pool)
    await repo.add_message(55, "user", "x" * 100, "study")

    history = await repo.get_dialog_history(55, "study", max_messages=10, max_chars=10)

    assert history == []


async def test_mode_switching_clears_history_and_saves_new_mode(
    fake_pool, test_settings, test_dispatcher
):
    """Сценарий 4: Смена режима очищает историю и сохраняет новую настройку."""
    # Arrange
    bot = Bot(TOKEN)
    bot.session = AsyncMock()
    repo = StorageRepository(fake_pool)
    user_id = 77

    # Заполняем историю в режиме study
    await repo.add_message(user_id, "user", "Вопрос по учебе", "study")
    await repo.add_message(user_id, "assistant", "Ответ по учебе", "study")

    # Act 1: Переключаемся на /translate
    cmd_translate = create_message(user_id=user_id, text="/translate")
    await test_dispatcher.feed_update(
        bot,
        Update(update_id=1, message=cmd_translate),
        db=fake_pool,
        settings=test_settings,
    )

    # Assert 1
    settings_after = await repo.get_user_settings(user_id)
    assert settings_after.mode == "translate"
    study_history = await repo.get_dialog_history(user_id, "study")
    assert len(study_history) == 0  # История старого режима очищена

    # Act 2: Переключаемся на собственный режим /code_review
    cmd_review = create_message(user_id=user_id, text="/code_review")
    await test_dispatcher.feed_update(
        bot,
        Update(update_id=2, message=cmd_review),
        db=fake_pool,
        settings=test_settings,
    )

    # Assert 2
    settings_review = await repo.get_user_settings(user_id)
    assert settings_review.mode == "code_review"

    # Проверяем, что следующему сообщению передастся CODE_REVIEW_PROMPT
    llm_client = AsyncMock(spec=LLMClient)
    llm_client.complete = AsyncMock(return_value="Код чистый, багов нет.")
    code_msg = create_message(user_id=user_id, text="def foo(): pass")
    await test_dispatcher.feed_update(
        bot,
        Update(update_id=3, message=code_msg),
        db=fake_pool,
        settings=test_settings,
        llm_client=llm_client,
    )
    prompt_sent = llm_client.complete.call_args.kwargs["messages"][0]["content"]
    assert prompt_sent == CODE_REVIEW_PROMPT


async def test_reset_clears_only_calling_user_history(fake_pool, test_settings, test_dispatcher):
    """Сценарий 5: /reset очищает только историю вызвавшего пользователя."""
    # Arrange
    bot = Bot(TOKEN)
    bot.session = AsyncMock()
    repo = StorageRepository(fake_pool)
    user1, user2 = 11, 22

    await repo.set_user_mode(user1, "translate")
    await repo.set_user_temperature(user1, 0.3)
    await repo.add_message(user1, "user", "Привет от 1", "translate")

    await repo.add_message(user2, "user", "Привет от 2", "study")

    # Act
    reset_msg = create_message(user_id=user1, text="/reset")
    await test_dispatcher.feed_update(
        bot,
        Update(update_id=1, message=reset_msg),
        db=fake_pool,
        settings=test_settings,
    )

    # Assert
    # История user1 очищена
    u1_history = await repo.get_dialog_history(user1, "translate")
    assert len(u1_history) == 0

    # Режим и температура user1 сохранились
    u1_settings = await repo.get_user_settings(user1)
    assert u1_settings.mode == "translate"
    assert u1_settings.temperature == 0.3

    # История user2 не затронута
    u2_history = await repo.get_dialog_history(user2, "study")
    assert len(u2_history) == 1
    assert u2_history[0]["content"] == "Привет от 2"


async def test_settings_accepts_allowed_and_rejects_invalid(
    fake_pool, test_settings, test_dispatcher
):
    """Сценарий 6: /settings принимает разрешённое и отклоняет некорректное значение."""
    # Arrange
    bot = Bot(TOKEN)
    bot.session = AsyncMock()
    repo = StorageRepository(fake_pool)
    user_id = 99

    # Act 1: Установка валидной температуры 0.3
    msg_valid = create_message(user_id=user_id, text="/settings 0.3")
    await test_dispatcher.feed_update(
        bot,
        Update(update_id=1, message=msg_valid),
        db=fake_pool,
        settings=test_settings,
    )

    # Assert 1
    current = await repo.get_user_settings(user_id)
    assert current.temperature == 0.3
    reply_valid = bot.session.call_args_list[-1].args[1]
    assert "Температура успешно установлена: 0.3" in reply_valid.text

    # Act 2: Отклонение неразрешенного значения (например, 0.5)
    msg_invalid = create_message(user_id=user_id, text="/settings 0.5")
    await test_dispatcher.feed_update(
        bot,
        Update(update_id=2, message=msg_invalid),
        db=fake_pool,
        settings=test_settings,
    )

    # Assert 2: Значение в БД не изменилось
    current_after = await repo.get_user_settings(user_id)
    assert current_after.temperature == 0.3
    reply_invalid = bot.session.call_args_list[-1].args[1]
    assert (
        "Недопустимое значение температуры. Разрешены только: 0.0, 0.3, 0.7, 1.0."
        in reply_invalid.text
    )

    # Act 3: Отклонение текстового мусора
    msg_text_garbage = create_message(user_id=user_id, text="/settings hot")
    await test_dispatcher.feed_update(
        bot,
        Update(update_id=3, message=msg_text_garbage),
        db=fake_pool,
        settings=test_settings,
    )
    reply_garbage = bot.session.call_args_list[-1].args[1]
    assert "Недопустимое значение температуры" in reply_garbage.text


@pytest.mark.parametrize(
    ("exception_cls", "expected_user_snippet"),
    [
        (
            LLMTimeoutError,
            "Превышено время ожидания ответа от языковой модели",
        ),
        (
            LLMAuthError,
            "Ошибка авторизации сервиса языковой модели",
        ),
        (
            LLMRateLimitError,
            "Превышен лимит запросов к сервису языковой модели",
        ),
        (
            LLMEmptyResponseError,
            "Языковая модель вернула пустой ответ",
        ),
        (
            LLMServiceError,
            "Сервис языковой модели временно недоступен",
        ),
    ],
)
async def test_llm_error_converted_to_safe_message_and_not_saved(
    fake_pool,
    test_settings,
    test_dispatcher,
    exception_cls,
    expected_user_snippet,
):
    """Сценарий 7: Ошибка LLM преобразуется в безопасное пользовательское сообщение."""
    # Arrange
    bot = Bot(TOKEN)
    bot.session = AsyncMock()
    llm_client = AsyncMock(spec=LLMClient)
    llm_client.complete = AsyncMock(side_effect=exception_cls("Техническая ошибка"))

    user_id = 123
    repo = StorageRepository(fake_pool)
    msg = create_message(user_id=user_id, text="Привет, модель!")

    # Act
    await test_dispatcher.feed_update(
        bot,
        Update(update_id=1, message=msg),
        db=fake_pool,
        settings=test_settings,
        llm_client=llm_client,
    )

    # Assert
    reply = bot.session.call_args_list[-1].args[1]
    assert expected_user_snippet in reply.text
    # Проверяем, что в истории нет сообщения ассистента
    history = await repo.get_dialog_history(user_id, "study")
    assert len(history) == 0


async def test_long_response_split_without_loss(fake_pool, test_settings, test_dispatcher):
    """Сценарий 8: Ответ длиннее 4096 символов разбивается и отправляется без потери текста."""
    # Arrange
    bot = Bot(TOKEN)
    bot.session = AsyncMock()

    # Формируем длинный текст: 3 абзаца по 2000 символов = 6000 символов
    para1 = "А" * 2000
    para2 = "Б" * 2000
    para3 = "В" * 2000
    long_text = f"{para1}\n\n{para2}\n\n{para3}"
    assert len(long_text) > 4096

    llm_client = AsyncMock(spec=LLMClient)
    llm_client.complete = AsyncMock(return_value=long_text)

    msg = create_message(user_id=10, text="Сгенерируй длинный текст")

    # Act
    await test_dispatcher.feed_update(
        bot,
        Update(update_id=1, message=msg),
        db=fake_pool,
        settings=test_settings,
        llm_client=llm_client,
    )

    # Assert
    # Фильтруем вызовы SendMessage
    sent_messages = [
        call.args[1] for call in bot.session.call_args_list if isinstance(call.args[1], SendMessage)
    ]
    assert len(sent_messages) >= 2

    # Каждое сообщение не должно превышать 4096 символов
    for sent in sent_messages:
        assert len(sent.text) <= 4096

    # Суммарный текст должен полностью совпадать с исходным
    reconstructed = "".join(sent.text for sent in sent_messages)
    assert reconstructed == long_text


async def test_long_markdown_response_is_sent_without_markup(
    fake_pool, test_settings, test_dispatcher
):
    bot = Bot(TOKEN)
    bot.session = AsyncMock()
    long_text = "```python\n" + ("x" * 5000) + "\n```"
    llm_client = AsyncMock(spec=LLMClient)
    llm_client.complete = AsyncMock(return_value=long_text)

    await test_dispatcher.feed_update(
        bot,
        Update(update_id=1, message=create_message(user_id=10, text="Код")),
        db=fake_pool,
        settings=test_settings,
        llm_client=llm_client,
    )

    sent_messages = [
        call.args[1] for call in bot.session.call_args_list if isinstance(call.args[1], SendMessage)
    ]
    assert "".join(message.text for message in sent_messages) == long_text
    assert all(message.parse_mode is None for message in sent_messages)


async def test_llm_client_sends_experiment_parameters_and_usage():
    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def json(self):
            return {"choices": [{"message": {"content": "Ответ"}}], "usage": {"total_tokens": 7}}

    class Session:
        closed = False

        def post(self, *_args, **kwargs):
            self.kwargs = kwargs
            return Response()

    session = Session()
    client = LLMClient(session=session)

    assert await client.complete([], temperature=0.3, top_p=1.0, max_tokens=1000) == "Ответ"
    assert session.kwargs["json"]["top_p"] == 1.0
    assert session.kwargs["json"]["max_tokens"] == 1000
    assert client.last_usage == {"total_tokens": 7}


def test_split_message_utility():
    """Тестирование функции split_message на граничных случаях."""
    # Пустая строка
    assert split_message("") == []

    # Текст меньше лимита
    assert split_message("Привет", 10) == ["Привет"]

    # Длинный непрерывный текст без пробелов
    long_word = "x" * 100
    chunks = split_message(long_word, 30)
    assert len(chunks) == 4
    assert "".join(chunks) == long_word
    assert all(len(c) <= 30 for c in chunks)


def test_missing_llm_api_key_when_required(tmp_path):
    """Проверка, что отсутствие LLM_API_KEY при require_llm=True вызывает понятную ошибку."""
    env_file = tmp_path / ".env"
    env_file.write_text(f"BOT_TOKEN={TOKEN}\nPOSTGRES_PASSWORD=db\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="LLM_API_KEY"):
        Settings.load(env_file, environ={}, require_llm=True)


def test_markdown_to_telegram_html():
    """Тестирование безопасной конвертации Markdown в HTML для Telegram."""
    md = (
        "### Заголовок\n\n"
        "**Жирный текст** и *курсив*\n"
        "* Пункт 1\n"
        "- Пункт 2\n"
        "`inline_code`\n\n"
        "```python\nif a > b:\n    return True\n```"
    )
    res = markdown_to_telegram_html(md)
    assert "<b>Заголовок</b>" in res
    assert "<b>Жирный текст</b>" in res
    assert "<i>курсив</i>" in res
    assert "• Пункт 1" in res
    assert "• Пункт 2" in res
    assert "<code>inline_code</code>" in res
    assert '<pre><code class="language-python">if a &gt; b:\n    return True</code></pre>' in res

    # Проверка незакрытого блока кода
    unclosed = "```python\nprint(1)"
    fixed = markdown_to_telegram_html(unclosed)
    assert "<pre><code" in fixed
    assert "</code></pre>" in fixed


def test_llm_proxy_fallback_and_validation(tmp_path):
    """Проверка, что LLM_PROXY_URL берет TELEGRAM_PROXY_URL по умолчанию и валидируется."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"BOT_TOKEN={TOKEN}\nPOSTGRES_PASSWORD=db\nTELEGRAM_PROXY_URL=socks5://proxy:1080\n",
        encoding="utf-8",
    )
    # 1. Fallback к TELEGRAM_PROXY_URL
    s = Settings.load(env_file, environ={})
    assert s.llm_proxy_url == "socks5://proxy:1080"

    # 2. Явный LLM_PROXY_URL переопределяет TELEGRAM_PROXY_URL
    s2 = Settings.load(
        env_file,
        environ={"LLM_PROXY_URL": "http://other:8080"},
    )
    assert s2.llm_proxy_url == "http://other:8080"

    # 3. Некорректный LLM_PROXY_URL отклоняется с понятной ошибкой
    with pytest.raises(ConfigError, match="LLM_PROXY_URL"):
        Settings.load(
            env_file,
            environ={"LLM_PROXY_URL": "invalid-url"},
        )


async def test_llm_client_creates_proxy_connector():
    """Проверка, что при передаче proxy в LLMClient создается ProxyConnector."""
    client = LLMClient(proxy="socks5://proxy:1080")
    try:
        session = await client.get_session()
        assert session.connector is not None
        from aiohttp_socks import ProxyConnector

        assert isinstance(session.connector, ProxyConnector)
    finally:
        await client.close()
