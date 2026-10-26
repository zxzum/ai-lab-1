"""Асинхронный клиент для взаимодействия с OpenAI-совместимым API языковых моделей."""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field

import aiohttp

from app.llm.exceptions import (
    LLMAuthError,
    LLMEmptyResponseError,
    LLMRateLimitError,
    LLMServiceError,
    LLMTimeoutError,
)

logger = logging.getLogger("app.llm")


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str = field(
        repr=False
    )  # JSON-строка как её вернула модель; разбор и проверка в агенте


@dataclass(frozen=True)
class LLMTurn:
    """Один ход модели: либо текст, либо запросы на вызов инструментов."""

    content: str
    tool_calls: list[ToolCall]
    assistant_message: dict = field(repr=False)  # для возврата в историю запроса как есть


class LLMClient:
    """Клиент для обращения к OpenAI-совместимому API модели."""

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        model: str = "gpt-4o-mini",
        timeout: float = 30.0,
        proxy: str = "",
        session: aiohttp.ClientSession | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.proxy = proxy
        self._session = session
        self._owns_session = session is None
        self.last_usage: dict | None = None
        self.last_finish_reason: str | None = None

    async def get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = None
            if self.proxy:
                from aiohttp_socks import ProxyConnector

                connector = ProxyConnector.from_url(self.proxy)
            self._session = aiohttp.ClientSession(connector=connector)
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
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Отправляет запрос на генерацию ответа в OpenAI-совместимый API.

        Логирует только безопасный request_id, имя модели, длительность и результат.
        Не логирует текст промпта, историю и API-ключи.
        """
        payload: dict = {"messages": messages, "temperature": temperature}
        if top_p is not None:
            payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        message_obj, request_id, start_time = await self._chat_request(payload, model)
        content = message_obj.get("content", "")
        if not isinstance(content, str) or not content.strip():
            logger.warning(
                "LLM пустой текст ответа: req_id=%s, time=%.2fs",
                request_id,
                time.monotonic() - start_time,
            )
            raise LLMEmptyResponseError("Модель вернула пустой текст.")
        logger.info(
            "LLM запрос завершен успешно: req_id=%s, model=%s, time=%.2fs, chars=%d",
            request_id,
            model or self.model,
            time.monotonic() - start_time,
            len(content),
        )
        return content.strip()

    async def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.7,
    ) -> LLMTurn:
        """Нативный вызов инструментов (function calling) в формате OpenAI.

        tools пуст — запрос идёт без инструментов. Проверка имени и аргументов вызова
        выполняется вызывающим кодом, здесь только разбор формы ответа.
        """
        payload: dict = {"messages": messages, "temperature": temperature}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        message_obj, request_id, start_time = await self._chat_request(payload, None)

        raw_calls = message_obj.get("tool_calls") or []
        calls: list[ToolCall] = []
        for raw in raw_calls if isinstance(raw_calls, list) else []:
            function = raw.get("function") if isinstance(raw, dict) else None
            if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                logger.warning("LLM некорректный вызов инструмента: req_id=%s", request_id)
                raise LLMEmptyResponseError("Модель вернула некорректный вызов инструмента.")
            arguments = function.get("arguments") or "{}"
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            calls.append(ToolCall(str(raw.get("id") or ""), function["name"], arguments))

        content = message_obj.get("content")
        content = content.strip() if isinstance(content, str) else ""
        if not calls and not content:
            logger.warning(
                "LLM пустой ход: req_id=%s, finish_reason=%s",
                request_id,
                self.last_finish_reason,
            )
            raise LLMEmptyResponseError("Модель вернула пустой текст.")
        logger.info(
            "LLM ход завершен: req_id=%s, model=%s, time=%.2fs, tool_calls=%d, chars=%d",
            request_id,
            self.model,
            time.monotonic() - start_time,
            len(calls),
            len(content),
        )
        assistant: dict = {"role": "assistant", "content": content or None}
        if calls:
            assistant["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments},
                }
                for c in calls
            ]
        return LLMTurn(content, calls, assistant)

    async def _chat_request(self, payload: dict, model: str | None) -> tuple[dict, str, float]:
        """POST /chat/completions: общая часть — статусы, таймаут, форма ответа."""
        request_id = uuid.uuid4().hex[:8]
        target_model = model or self.model
        start_time = time.monotonic()
        self.last_usage = None
        self.last_finish_reason = None
        payload = {"model": target_model, **payload}

        logger.info(
            "LLM запрос начат: req_id=%s, model=%s, temperature=%.2f",
            request_id,
            target_model,
            payload.get("temperature", 0.0),
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if "openrouter" in self.base_url.lower():
            headers["HTTP-Referer"] = "https://github.com/LegionerSV/itmo-tg-template"
            headers["X-Title"] = "ITMO Student AI Assistant"

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
                    if not isinstance(data, dict):
                        raise LLMEmptyResponseError("Модель вернула некорректный ответ.")
                    usage = data.get("usage")
                    self.last_usage = usage if isinstance(usage, dict) else None
                    choices = data.get("choices")
                    if (
                        not choices
                        or not isinstance(choices, list)
                        or not isinstance(choices[0], dict)
                    ):
                        error = data.get("error")
                        logger.warning(
                            "LLM пустой ответ choices: req_id=%s, time=%.2fs, provider_error=%s",
                            request_id,
                            duration,
                            error.get("message") if isinstance(error, dict) else error,
                        )
                        raise LLMEmptyResponseError("Модель вернула пустой ответ.")

                    self.last_finish_reason = choices[0].get("finish_reason")
                    message_obj = choices[0].get("message", {})
                    if not isinstance(message_obj, dict):
                        message_obj = {}
                    return message_obj, request_id, start_time

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
