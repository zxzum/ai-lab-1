"""Тесты расписания: чтение файла, выбор дня и недели без обращения к системным часам."""

import json
from datetime import UTC, date, datetime

import pytest

from app.mcp_server import schedule
from app.mcp_server.schedule import (
    ScheduleUnavailableError,
    day_schedule,
    load_entries,
    today_in,
    week_schedule,
)

TZ = "Europe/Moscow"


@pytest.fixture
def entries(tmp_path):
    items = [
        {"date": "2026-12-31", "start": "11:00", "end": "12:30", "title": "Б", "location": "2"},
        {"date": "2026-12-31", "start": "09:00", "end": "10:30", "title": "А", "location": "1"},
        {"date": "2027-01-04", "start": "09:00", "end": "10:30", "title": "В", "location": "3"},
    ]
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"lessons": items}), encoding="utf-8")
    return load_entries(path)


def test_day_with_lessons_is_sorted_by_start(entries):
    result = day_schedule(entries, date(2026, 12, 31), TZ)

    assert [(le.start, le.end, le.title, le.location) for le in result.lessons] == [
        ("09:00", "10:30", "А", "1"),
        ("11:00", "12:30", "Б", "2"),
    ]


def test_day_without_lessons_is_empty_list(entries):
    result = day_schedule(entries, date(2027, 1, 1), TZ)

    assert result.lessons == []
    assert result.date == "2027-01-01"


def test_week_across_year_boundary(entries):
    # 31.12.2026 — четверг; неделя 28.12.2026-03.01.2027 пересекает границу года
    result = week_schedule(entries, date(2026, 12, 31), TZ)

    assert (result.week_start, result.week_end) == ("2026-12-28", "2027-01-03")
    assert list(result.days) == [f"2026-12-{d}" for d in (28, 29, 30, 31)] + [
        "2027-01-01",
        "2027-01-02",
        "2027-01-03",
    ]
    assert len(result.days["2026-12-31"]) == 2
    assert "2027-01-04" not in result.days  # следующий понедельник уже другая неделя


def test_week_on_sunday_and_month_boundary(entries):
    # воскресенье 31.05.2026 относится к неделе 25-31 мая
    result = week_schedule(entries, date(2026, 5, 31), TZ)

    assert (result.week_start, result.week_end) == ("2026-05-25", "2026-05-31")


def test_today_uses_schedule_timezone_not_system_clock():
    # 23:30 UTC 24 октября — это уже 25 октября в Москве (UTC+3)
    clock = lambda: datetime(2026, 10, 24, 23, 30, tzinfo=UTC)  # noqa: E731

    assert today_in(TZ, clock) == date(2026, 10, 25)


def test_missing_and_broken_files_are_reported(tmp_path):
    broken = tmp_path / "b.json"
    broken.write_text("{not json", encoding="utf-8")

    with pytest.raises(ScheduleUnavailableError):
        load_entries(tmp_path / "missing.json")
    with pytest.raises(ScheduleUnavailableError):
        load_entries(broken)


def test_bundled_schedule_covers_at_least_two_weeks():
    dates = {e.date for e in load_entries(schedule.DEFAULT_FILE)}

    assert (max(dates) - min(dates)).days >= 13
