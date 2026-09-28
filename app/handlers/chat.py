"""Обработчик обычных текстовых сообщений пользователя в приватном чате."""

import logging

import asyncpg
from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import Message
from aiogram.utils.chat_action import ChatActionSender

from app.config import Settings
from app.llm.client import LLMClient
from app.llm.exceptions import (
    LLMAuthError,
    LLMEmptyResponseError,
    LLMError,
    LLMRateLimitError,
    LLMServiceError,
    LLMTimeoutError,
)
from app.llm.prompts import get_system_prompt
from app.storage.repository import StorageRepository
from app.utils.text import markdown_to_telegram_html, split_message

logger = logging.getLogger("app.chat")


async def handle_chat_message(
    message: Message,
    db: asyncpg.Pool,
    settings: Settings,
    llm_client: LLMClient,
) -> None:
    """Обрабатывает текстовые сообщения пользователя, обращаясь к языковой модели."""
    if not message.text or not message.from_user or not message.bot:
        return

    user_id = message.from_user.id
    repo = StorageRepository(db)

    # Получаем настройки пользователя (режим и температуру)
    user_settings = await repo.get_user_settings(user_id)

    # Получаем историю диалога текущего режима с учетом ограничений
    history = await repo.get_dialog_history(
        user_id=user_id,
        mode=user_settings.mode,
        max_messages=settings.max_history_messages,
        max_chars=settings.max_history_chars,
    )

    # Формируем список сообщений для LLM
    system_prompt = get_system_prompt(user_settings.mode)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": message.text})

    # Показываем статус "печатает..." во время генерации
    try:
        async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
            reply_text = await llm_client.complete(
                messages=messages,
                temperature=user_settings.temperature,
            )
    except LLMTimeoutError:
        await message.answer(
            "Превышено время ожидания ответа от языковой модели. "
            "Пожалуйста, попробуйте повторить запрос позже.",
            parse_mode=None,
        )
        return
    except LLMAuthError:
        await message.answer(
            "Ошибка авторизации сервиса языковой модели. Пожалуйста, обратитесь к администратору.",
            parse_mode=None,
        )
        return
    except LLMRateLimitError:
        await message.answer(
            "Превышен лимит запросов к сервису языковой модели. "
            "Пожалуйста, подождите немного и повторите запрос.",
            parse_mode=None,
        )
        return
    except LLMEmptyResponseError:
        await message.answer(
            "Языковая модель вернула пустой ответ. Попробуйте переформулировать ваш вопрос.",
            parse_mode=None,
        )
        return
    except LLMServiceError:
        await message.answer(
            "Сервис языковой модели временно недоступен. "
            "Пожалуйста, попробуйте повторить запрос позже.",
            parse_mode=None,
        )
        return
    except (LLMError, Exception):
        logger.exception("Непредвиденная ошибка при вызове LLM")
        await message.answer(
            "Произошла ошибка при обработке вашего сообщения. Пожалуйста, попробуйте позже.",
            parse_mode=None,
        )
        return

    # Сохраняем в историю диалога только успешный диалог
    await repo.add_message(user_id, "user", message.text, user_settings.mode)
    await repo.add_message(user_id, "assistant", reply_text, user_settings.mode)

    # Разметка может разорваться при разбиении; длинный ответ отправляем как обычный текст.
    if len(reply_text) > 4000:
        for chunk in split_message(reply_text):
            await message.answer(chunk, parse_mode=None)
        return

    formatted_chunk = markdown_to_telegram_html(reply_text)
    try:
        await message.answer(formatted_chunk, parse_mode="HTML")
    except TelegramBadRequest:
        # Безопасный fallback: отправляем без parse_mode, если разметка не прошла валидацию
        await message.answer(reply_text, parse_mode=None)


def create_chat_router() -> Router:
    """Создает роутер чата с зарегистрированным обработчиком текста."""
    r = Router(name="chat")
    r.message.filter(F.chat.type == "private")
    r.message.register(handle_chat_message, F.text & ~F.text.startswith("/"))
    return r


router = create_chat_router()
