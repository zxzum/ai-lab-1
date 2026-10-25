"""Расписание занятий: чтение из фиксированного JSON-файла и выбор по дате/неделе."""

import json
import os
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

DEFAULT_FILE = Path(__file__).parent / "data" / "schedule.json"
DEFAULT_TIMEZONE = "Europe/Moscow"


class ScheduleUnavailableError(Exception):
    """Файл расписания отсутствует или повреждён."""


class Lesson(BaseModel):
    start: str = Field(description="Начало, HH:MM по времени расписания")
    end: str = Field(description="Окончание, HH:MM")
    title: str
    location: str


class DaySchedule(BaseModel):
    date: str = Field(description="Дата, YYYY-MM-DD")
    timezone: str
    lessons: list[Lesson] = Field(description="Пустой список означает отсутствие занятий")


class WeekSchedule(BaseModel):
    week_start: str = Field(description="Понедельник, YYYY-MM-DD")
    week_end: str = Field(description="Воскресенье, YYYY-MM-DD")
    timezone: str
    days: dict[str, list[Lesson]] = Field(description="Занятия по датам, все 7 дней недели")


class _Entry(Lesson):
    date: date


def schedule_timezone() -> str:
    """Часовой пояс расписания задаётся конфигурацией, не аргументами инструментов."""
    return os.environ.get("SCHEDULE_TIMEZONE", DEFAULT_TIMEZONE)


def load_entries(path: Path | None = None) -> list[_Entry]:
    path = path or Path(os.environ.get("SCHEDULE_FILE") or DEFAULT_FILE)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [_Entry.model_validate(item) for item in raw["lessons"]]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ScheduleUnavailableError(f"расписание недоступно: {type(exc).__name__}") from exc


def _lessons_on(entries: list[_Entry], day: date) -> list[Lesson]:
    found = sorted((e for e in entries if e.date == day), key=lambda e: e.start)
    return [Lesson(**e.model_dump(exclude={"date"})) for e in found]


def day_schedule(entries: list[_Entry], day: date, timezone: str) -> DaySchedule:
    return DaySchedule(date=day.isoformat(), timezone=timezone, lessons=_lessons_on(entries, day))


def week_schedule(entries: list[_Entry], today: date, timezone: str) -> WeekSchedule:
    monday = today - timedelta(days=today.weekday())
    days = [monday + timedelta(days=i) for i in range(7)]
    return WeekSchedule(
        week_start=days[0].isoformat(),
        week_end=days[-1].isoformat(),
        timezone=timezone,
        days={d.isoformat(): _lessons_on(entries, d) for d in days},
    )


def today_in(timezone: str, clock: Callable[[], datetime] = datetime.now) -> date:
    """Текущая дата в часовом поясе расписания; clock подменяется в тестах."""
    return clock().astimezone(ZoneInfo(timezone)).date()
