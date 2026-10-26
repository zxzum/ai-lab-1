"""Чистые функции агента: схемы для модели, проверка аргументов, локальное время, подпись."""

from datetime import UTC, datetime

import pytest

from app.agent.tools import (
    CLARIFY,
    RATIONALE,
    ArgumentError,
    ReminderTimeError,
    expose_tools,
    localize_reminder,
    summarize_arguments,
    validate_arguments,
)
from app.mcp_client.client import ToolInfo
from app.mcp_server.reminders import ReminderRefusedError, parse_remind_at
from app.mcp_trust import TrustedContext, build_meta, read_trusted

WEATHER = ToolInfo(
    "get_weather",
    "Текущая погода",
    True,
    {
        "type": "object",
        "properties": {"city": {"type": "string", "minLength": 1, "maxLength": 100}},
        "required": ["city"],
        "additionalProperties": False,
    },
)
NOW = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)


def test_exposed_schema_adds_rationale_without_touching_original():
    exposed = expose_tools([WEATHER])

    params = exposed[0]["function"]["parameters"]
    assert params["required"] == ["city", RATIONALE]
    assert RATIONALE in params["properties"]
    assert exposed[1]["function"]["name"] == CLARIFY
    assert RATIONALE not in WEATHER.input_schema["properties"]  # исходная схема не изменена


def test_no_tools_means_nothing_is_exposed():
    assert expose_tools([]) == []


@pytest.mark.parametrize(
    "arguments",
    [{}, {"city": ""}, {"city": "x" * 101}, {"city": 5}, {"city": "Казань", "user_id": 1}],
)
def test_invalid_arguments_are_rejected(arguments):
    with pytest.raises(ArgumentError):
        validate_arguments(WEATHER, arguments)


def test_valid_arguments_pass():
    validate_arguments(WEATHER, {"city": "Казань"})


def test_summary_uses_labels_and_truncates():
    assert summarize_arguments({"city": "Казань"}) == "город «Казань»"
    assert summarize_arguments({"text": "я" * 200}) == "текст «" + "я" * 80 + "»"


def test_naive_local_time_gets_user_timezone():
    result = localize_reminder(
        {"text": "Сдать лабу", "remind_at": "2026-10-27T09:00:00"}, "Europe/Moscow", NOW
    )

    assert result == {"text": "Сдать лабу", "remind_at": "2026-10-27T09:00:00+03:00"}


def test_model_cannot_choose_another_timezone():
    args = {"text": "x", "remind_at": "2026-10-27T09:00:00+09:00"}

    with pytest.raises(ReminderTimeError):
        localize_reminder(args, "Europe/Moscow", NOW)


def test_matching_offset_is_accepted():
    args = {"text": "x", "remind_at": "2026-10-27T09:00:00+03:00"}

    assert localize_reminder(args, "Europe/Moscow", NOW)["remind_at"] == "2026-10-27T09:00:00+03:00"


@pytest.mark.parametrize(
    "local",
    ["2026-11-01T01:30:00", "2026-03-08T02:30:00"],  # повтор часа и несуществующий час, США
)
def test_dst_ambiguous_or_missing_local_time_requires_clarification(local):
    with pytest.raises(ReminderTimeError, match="неоднозначно"):
        localize_reminder(
            {"text": "x", "remind_at": local}, "America/New_York", datetime(2026, 1, 1, tzinfo=UTC)
        )


def test_past_time_is_rejected():
    with pytest.raises(ReminderTimeError, match="прошло"):
        localize_reminder({"text": "x", "remind_at": "2026-10-25T10:00:00"}, "Europe/Moscow", NOW)


def test_garbage_time_is_rejected():
    with pytest.raises(ArgumentError):
        localize_reminder({"text": "x", "remind_at": "завтра"}, "Europe/Moscow", NOW)


# --- сервер: проверка времени напоминания ---


def test_server_requires_explicit_offset_and_future():
    assert parse_remind_at("2026-10-27T09:00:00+03:00", NOW).hour == 9
    for bad in ("2026-10-27T09:00:00", "2026-10-25T10:00:00+00:00", "завтра"):
        with pytest.raises(ReminderRefusedError):
            parse_remind_at(bad, NOW)


# --- подпись доверенного контекста ---

CTX = TrustedContext(42, "Europe/Moscow", "abc123")


def test_signed_context_is_accepted():
    meta = build_meta("secret", CTX)

    assert read_trusted(meta, "secret") == CTX


@pytest.mark.parametrize(
    ("field", "value"),
    [("owner_id", 43), ("timezone", "UTC"), ("idempotency_key", "other"), ("signature", "0" * 64)],
)
def test_tampered_context_is_rejected(field, value):
    meta = build_meta("secret", CTX)
    meta["trusted"][field] = value

    assert read_trusted(meta, "secret") is None


def test_wrong_empty_or_missing_secret_is_rejected():
    meta = build_meta("secret", CTX)

    assert read_trusted(meta, "other") is None
    assert read_trusted(meta, "") is None  # сервер без секрета отказывает всегда
    assert read_trusted(None, "secret") is None
    assert read_trusted({"trusted": {"owner_id": "x"}}, "secret") is None
