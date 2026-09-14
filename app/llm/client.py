"""Асинхронный клиент для взаимодействия с OpenAI-совместимым API языковых моделей."""

import asyncio
import logging
import time
import uuid

import aiohttp

from app.llm.exceptions import (
    LLMAuthError,
    LLMEmptyResponseError,
    LLMRateLimitError,
    LLMServiceError,
    LLMTimeoutError,
)

logger = logging.getLogger("app.llm")


class LLMClient:
    """Клиент для обращения к OpenAI-совместимому API модели."""

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        model: str = "gpt-4o-mini",
        timeout: float = 30.0,
        session: aiohttp.ClientSession | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._session = session
        self._owns_session = session is None

    async def get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        model: str | None = None,
    ) -> str:
        """Отправляет запрос на генерацию ответа в OpenAI-совместимый API.

        Логирует только безопасный request_id, имя модели, длительность и результат.
        Не логирует текст промпта, историю и API-ключи.
        """
        request_id = uuid.uuid4().hex[:8]
        target_model = model or self.model
        start_time = time.monotonic()

        logger.info(
            "LLM запрос начат: req_id=%s, model=%s, temperature=%.2f",
            request_id,
            target_model,
            temperature,
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
        }

        url = f"{self.base_url}/chat/completions"
        session = await self.get_session()

        try:
            async with asyncio.timeout(self.timeout):
                async with session.post(url, json=payload, headers=headers) as response:
                    duration = time.monotonic() - start_time
                    status = response.status

                    if status in (401, 403):
                        logger.error(
                            "LLM ошибка авторизации: req_id=%s, status=%d, time=%.2fs",
                            request_id,
                            status,
                            duration,
                        )
                        raise LLMAuthError("Неверный ключ доступа к языковой модели.")

                    if status == 429:
                        logger.warning(
                            "LLM превышение лимита: req_id=%s, status=%d, time=%.2fs",
                            request_id,
                            status,
                            duration,
                        )
                        raise LLMRateLimitError(
                            "Превышен лимит запросов к сервису языковой модели."
                        )

                    if status >= 500:
                        logger.error(
                            "LLM ошибка сервиса: req_id=%s, status=%d, time=%.2fs",
                            request_id,
                            status,
                            duration,
                        )
                        raise LLMServiceError(
                            f"Сервис языковой модели временно недоступен (HTTP {status})."
                        )

                    if status != 200:
                        logger.error(
                            "LLM неожиданный статус: req_id=%s, status=%d, time=%.2fs",
                            request_id,
                            status,
                            duration,
                        )
                        raise LLMServiceError(
                            f"Ошибка обращения к языковой модели (HTTP {status})."
                        )

                    data = await response.json()
                    choices = data.get("choices")
                    if not choices or not isinstance(choices, list):
                        logger.warning(
                            "LLM пустой ответ choices: req_id=%s, time=%.2fs",
                            request_id,
                            duration,
                        )
                        raise LLMEmptyResponseError("Модель вернула пустой ответ.")

                    message_obj = choices[0].get("message", {})
                    content = message_obj.get("content", "")
                    if not content or not content.strip():
                        logger.warning(
                            "LLM пустой текст ответа: req_id=%s, time=%.2fs",
                            request_id,
                            duration,
                        )
                        raise LLMEmptyResponseError("Модель вернула пустой текст.")

                    logger.info(
                        "LLM запрос завершен успешно: req_id=%s, model=%s, time=%.2fs, chars=%d",
                        request_id,
                        target_model,
                        duration,
                        len(content),
                    )
                    return content.strip()

        except TimeoutError:
            duration = time.monotonic() - start_time
            logger.error(
                "LLM таймаут: req_id=%s, model=%s, time=%.2fs",
                request_id,
                target_model,
                duration,
            )
            raise LLMTimeoutError("Превышено время ожидания ответа от языковой модели.") from None
        except aiohttp.ClientError as exc:
            duration = time.monotonic() - start_time
            logger.error(
                "LLM сетевая ошибка: req_id=%s, error=%s, time=%.2fs",
                request_id,
                type(exc).__name__,
                duration,
            )
            raise LLMServiceError("Сетевая ошибка при обращении к языковой модели.") from exc
