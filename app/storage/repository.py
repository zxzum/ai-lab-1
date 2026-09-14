"""Репозиторий для работы с пользовательскими настройками и историей диалогов в PostgreSQL."""

from dataclasses import dataclass

import asyncpg


@dataclass
class UserSettings:
    user_id: int
    mode: str = "study"
    temperature: float = 0.7


class StorageRepository:
    """Изолированный класс доступа к данным в PostgreSQL."""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def get_user_settings(self, user_id: int) -> UserSettings:
        """Получает настройки пользователя или значения по умолчанию."""
        row = await self.pool.fetchrow(
            """
            SELECT mode, temperature
            FROM user_settings
            WHERE user_id = $1
            """,
            user_id,
        )
        if row is None:
            return UserSettings(user_id=user_id, mode="study", temperature=0.7)
        return UserSettings(
            user_id=user_id,
            mode=row["mode"],
            temperature=float(row["temperature"]),
        )

    async def set_user_mode(self, user_id: int, new_mode: str) -> None:
        """Устанавливает новый режим пользователя и очищает историю предыдущего режима."""
        current = await self.get_user_settings(user_id)
        old_mode = current.mode

        # Очищаем историю старого режима для этого пользователя
        await self.clear_user_history(user_id, mode=old_mode)

        # Сохраняем новый режим
        await self.pool.execute(
            """
            INSERT INTO user_settings (user_id, mode, temperature, updated_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (user_id)
            DO UPDATE SET mode = EXCLUDED.mode, updated_at = NOW()
            """,
            user_id,
            new_mode,
            current.temperature,
        )

    async def set_user_temperature(self, user_id: int, temperature: float) -> None:
        """Устанавливает температуру генерации для пользователя."""
        current = await self.get_user_settings(user_id)
        await self.pool.execute(
            """
            INSERT INTO user_settings (user_id, mode, temperature, updated_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (user_id)
            DO UPDATE SET temperature = EXCLUDED.temperature, updated_at = NOW()
            """,
            user_id,
            current.mode,
            temperature,
        )

    async def add_message(self, user_id: int, role: str, content: str, mode: str) -> None:
        """Сохраняет сообщение в историю диалога."""
        await self.pool.execute(
            """
            INSERT INTO dialog_messages (user_id, role, content, mode, created_at)
            VALUES ($1, $2, $3, $4, NOW())
            """,
            user_id,
            role,
            content,
            mode,
        )

    async def clear_user_history(self, user_id: int, mode: str | None = None) -> int:
        """Удаляет историю диалога пользователя (для конкретного режима или всю)."""
        if mode is not None:
            result = await self.pool.execute(
                """
                DELETE FROM dialog_messages
                WHERE user_id = $1 AND mode = $2
                """,
                user_id,
                mode,
            )
        else:
            result = await self.pool.execute(
                """
                DELETE FROM dialog_messages
                WHERE user_id = $1
                """,
                user_id,
            )
        # asyncpg возвращает строку вида "DELETE <count>"
        try:
            return int(result.split()[-1])
        except (IndexError, ValueError):
            return 0

    async def get_dialog_history(
        self,
        user_id: int,
        mode: str,
        max_messages: int = 10,
        max_chars: int = 4000,
    ) -> list[dict[str, str]]:
        """Возвращает историю сообщений с учетом ограничений по числу и объему.

        Выбирает последние сообщения, сохраняя хронологический порядок (от старых к новым).
        """
        if max_messages <= 0 or max_chars <= 0:
            return []

        # Выбираем последние N сообщений от новых к старым
        rows = await self.pool.fetch(
            """
            SELECT role, content
            FROM dialog_messages
            WHERE user_id = $1 AND mode = $2
            ORDER BY created_at DESC, id DESC
            LIMIT $3
            """,
            user_id,
            mode,
            max_messages,
        )

        # Ограничиваем по суммарному объёму символов (начиная с самых свежих)
        selected: list[dict[str, str]] = []
        current_chars = 0

        for row in rows:
            content = row["content"]
            content_len = len(content)
            if current_chars + content_len > max_chars and selected:
                # Если добавление сообщения превысит лимит объема, останавливаемся
                break
            selected.append({"role": row["role"], "content": content})
            current_chars += content_len

        # Разворачиваем обратно в хронологический порядок (от старых к новым)
        selected.reverse()
        return selected
