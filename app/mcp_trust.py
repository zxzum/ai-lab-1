"""Доверенный контекст вызова MCP-инструмента с побочным эффектом.

Владельца напоминания знает только приложение (из Telegram update). Оно подписывает
контекст секретом, который передаёт серверу при запуске дочернего процесса. Независимый
MCP-клиент (например Inspector) секрета не знает и не может подделать владельца.
"""

import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass

SECRET_ENV = "MCP_TRUSTED_SECRET"
META_KEY = "trusted"


@dataclass(frozen=True)
class TrustedContext:
    owner_id: int
    timezone: str
    idempotency_key: str


def new_secret() -> str:
    return secrets.token_urlsafe(32)


def _digest(secret: str, ctx: TrustedContext) -> str:
    message = f"{ctx.owner_id}|{ctx.timezone}|{ctx.idempotency_key}".encode()
    return hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()


def build_meta(secret: str, ctx: TrustedContext) -> dict:
    """Значение для поля _meta запроса tools/call."""
    return {
        META_KEY: {
            "owner_id": ctx.owner_id,
            "timezone": ctx.timezone,
            "idempotency_key": ctx.idempotency_key,
            "signature": _digest(secret, ctx),
        }
    }


def read_trusted(meta: dict | None, secret: str | None = None) -> TrustedContext | None:
    """Возвращает проверенный контекст или None (нет секрета, нет подписи, подделка)."""
    secret = secret if secret is not None else os.environ.get(SECRET_ENV, "")
    raw = (meta or {}).get(META_KEY)
    if not secret or not isinstance(raw, dict):
        return None
    try:
        ctx = TrustedContext(
            int(raw["owner_id"]), str(raw["timezone"]), str(raw["idempotency_key"])
        )
        signature = str(raw["signature"])
    except (KeyError, TypeError, ValueError):
        return None
    return ctx if hmac.compare_digest(signature, _digest(secret, ctx)) else None
