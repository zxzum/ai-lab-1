"""Исключения при работе с языковой моделью (LLM)."""


class LLMError(Exception):
    """Базовое исключение для ошибок вызова LLM."""


class LLMTimeoutError(LLMError):
    """Превышено время ожидания ответа от языковой модели."""


class LLMAuthError(LLMError):
    """Ошибка авторизации или неверный API-ключ."""


class LLMRateLimitError(LLMError):
    """Превышен лимит запросов к сервису LLM."""


class LLMServiceError(LLMError):
    """Ошибка сервиса LLM или сетевой сбой."""


class LLMEmptyResponseError(LLMError):
    """Модель вернула пустой или некорректный ответ."""
