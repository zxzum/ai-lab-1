"""Обработчики команд бота: /start, /study, /translate, /code_review, /settings, /reset."""

import asyncpg
from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from app.config import Settings
from app.llm.prompts import get_mode_name
from app.storage.repository import StorageRepository

ALLOWED_TEMPERATURES = {0.0, 0.3, 0.7, 1.0}


def get_settings_keyboard(current_temp: float) -> InlineKeyboardMarkup:
    """Формирует клавиатуру с допустимыми значениями температуры."""
    buttons = []
    for temp in sorted(ALLOWED_TEMPERATURES):
        # Отмечаем текущее значение галочкой
        is_selected = abs(temp - current_temp) < 1e-4
        text = f"✓ {temp}" if is_selected else str(temp)
        buttons.append(InlineKeyboardButton(text=text, callback_data=f"set_temp:{temp}"))
    return InlineKeyboardMarkup(inline_keyboard=[buttons])


async def cmd_start(message: Message, db: asyncpg.Pool) -> None:
    """Приветственное сообщение со списком команд и текущим режимом."""
    if not message.from_user:
        return
    repo = StorageRepository(db)
    user_settings = await repo.get_user_settings(message.from_user.id)
    mode_name = get_mode_name(user_settings.mode)

    text = (
        "👋 Привет! Я твой учебный AI-ассистент.\n\n"
        "Я помогу разобраться с программированием, перевести текст "
        "или провести ревью твоего кода.\n\n"
        "📋 Доступные режимы работы:\n"
        "• /study — объяснение учебных тем и алгоритмов (по умолчанию)\n"
        "• /translate — перевод текстов (с few-shot примерами)\n"
        "• /code_review — ревью кода, поиск багов и рекомендации\n\n"
        "⚙️ Управление:\n"
        "• /settings — просмотр и изменение температуры генерации\n"
        "• /reset — сбросить историю диалога текущего пользователя\n"
        "• /start — показать эту справку\n\n"
        f"Текущий активный режим: {mode_name}\n"
        "Отправь мне текстовое сообщение, чтобы начать диалог!"
    )
    await message.answer(text, parse_mode=None)


async def cmd_study(message: Message, db: asyncpg.Pool) -> None:
    """Переключение на режим /study."""
    if not message.from_user:
        return
    repo = StorageRepository(db)
    await repo.set_user_mode(message.from_user.id, "study")
    await message.answer(
        "Режим переключен: Учеба и программирование (/study).\n"
        "История диалога предыдущего режима очищена.",
        parse_mode=None,
    )


async def cmd_translate(message: Message, db: asyncpg.Pool) -> None:
    """Переключение на режим /translate."""
    if not message.from_user:
        return
    repo = StorageRepository(db)
    await repo.set_user_mode(message.from_user.id, "translate")
    await message.answer(
        "Режим переключен: Переводчик (/translate).\nИстория диалога предыдущего режима очищена.",
        parse_mode=None,
    )


async def cmd_code_review(message: Message, db: asyncpg.Pool) -> None:
    """Переключение на собственный режим /code_review."""
    if not message.from_user:
        return
    repo = StorageRepository(db)
    await repo.set_user_mode(message.from_user.id, "code_review")
    await message.answer(
        "Режим переключен: Ревью кода (/code_review).\nИстория диалога предыдущего режима очищена.",
        parse_mode=None,
    )


async def cmd_reset(message: Message, db: asyncpg.Pool) -> None:
    """Сброс контекста диалога текущего пользователя."""
    if not message.from_user:
        return
    repo = StorageRepository(db)
    await repo.clear_user_history(message.from_user.id)
    await message.answer(
        "История диалога очищена. Текущий режим и настройки температуры сохранены.",
        parse_mode=None,
    )


async def cmd_settings(
    message: Message,
    command: CommandObject,
    db: asyncpg.Pool,
    settings: Settings,
) -> None:
    """Просмотр и изменение настроек генерации."""
    if not message.from_user:
        return
    repo = StorageRepository(db)
    user_settings = await repo.get_user_settings(message.from_user.id)

    # Если аргумент передан текстом: /settings 0.7
    if command.args:
        arg = command.args.strip()
        try:
            val = float(arg)
            if val not in ALLOWED_TEMPERATURES:
                raise ValueError
        except ValueError:
            await message.answer(
                "Недопустимое значение температуры. Разрешены только: 0.0, 0.3, 0.7, 1.0.",
                parse_mode=None,
            )
            return

        await repo.set_user_temperature(message.from_user.id, val)
        await message.answer(
            f"Температура успешно установлена: {val:.1f}\n"
            "Новое значение будет применяться начиная со следующего запроса.",
            parse_mode=None,
        )
        return

    # Если аргумент не передан, показываем текущие настройки и инлайн-клавиатуру
    mode_name = get_mode_name(user_settings.mode)
    text = (
        "⚙️ Текущие настройки:\n\n"
        f"• Активный режим: {mode_name}\n"
        f"• Модель: {settings.llm_model}\n"
        f"• Температура (temperature): {user_settings.temperature:.1f}\n\n"
        "Выберите новое значение температуры с помощью кнопок ниже "
        "или отправьте команду вида /settings 0.3:"
    )
    keyboard = get_settings_keyboard(user_settings.temperature)
    await message.answer(text, reply_markup=keyboard, parse_mode=None)


async def callback_set_temp(
    callback: CallbackQuery,
    db: asyncpg.Pool,
    settings: Settings,
) -> None:
    """Обработка нажатия на кнопку выбора температуры."""
    if not callback.data or not callback.message or not callback.from_user:
        return

    raw_temp = callback.data.split(":", 1)[1]
    try:
        new_temp = float(raw_temp)
        if new_temp not in ALLOWED_TEMPERATURES:
            raise ValueError
    except ValueError:
        await callback.answer(
            "Недопустимое значение температуры.",
            show_alert=True,
        )
        return

    repo = StorageRepository(db)
    await repo.set_user_temperature(callback.from_user.id, new_temp)
    await callback.answer(f"Температура изменена на {new_temp:.1f}")

    user_settings = await repo.get_user_settings(callback.from_user.id)
    mode_name = get_mode_name(user_settings.mode)
    text = (
        "⚙️ Текущие настройки:\n\n"
        f"• Активный режим: {mode_name}\n"
        f"• Модель: {settings.llm_model}\n"
        f"• Температура (temperature): {new_temp:.1f}\n\n"
        "Значение сохранено в базе данных и вступит в силу со следующего запроса."
    )
    keyboard = get_settings_keyboard(new_temp)
    if isinstance(callback.message, Message):
        await callback.message.edit_text(text, reply_markup=keyboard, parse_mode=None)


def create_commands_router() -> Router:
    """Создает роутер команд с зарегистрированными обработчиками."""
    r = Router(name="commands")
    r.message.filter(F.chat.type == "private")
    r.callback_query.filter(F.message.chat.type == "private")

    r.message.register(cmd_start, Command("start"))
    r.message.register(cmd_study, Command("study"))
    r.message.register(cmd_translate, Command("translate"))
    r.message.register(cmd_code_review, Command("code_review"))
    r.message.register(cmd_reset, Command("reset"))
    r.message.register(cmd_settings, Command("settings"))
    r.callback_query.register(callback_set_temp, F.data.startswith("set_temp:"))
    return r


router = create_commands_router()
