"""Получение текущей погоды через Open-Meteo (геокодинг + прогноз). Ключ API не нужен."""

import asyncio
from typing import Literal

import aiohttp
from pydantic import BaseModel, Field

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HTTP_TIMEOUT = aiohttp.ClientTimeout(total=10)
READ_ATTEMPTS = 2  # повтор только для безопасных GET-запросов чтения

# Коды погоды WMO, которые возвращает Open-Meteo.
WMO_DESCRIPTIONS = {
    0: "ясно",
    1: "преимущественно ясно",
    2: "переменная облачность",
    3: "пасмурно",
    45: "туман",
    48: "изморозевый туман",
    51: "слабая морось",
    53: "морось",
    55: "сильная морось",
    56: "ледяная морось",
    57: "сильная ледяная морось",
    61: "слабый дождь",
    63: "дождь",
    65: "сильный дождь",
    66: "ледяной дождь",
    67: "сильный ледяной дождь",
    71: "слабый снег",
    73: "снег",
    75: "сильный снег",
    77: "снежные зёрна",
    80: "слабый ливень",
    81: "ливень",
    82: "сильный ливень",
    85: "слабый снегопад",
    86: "сильный снегопад",
    95: "гроза",
    96: "гроза с градом",
    99: "сильная гроза с градом",
}


class WeatherUpstreamError(Exception):
    """Погодный сервис недоступен, превысил тайм-аут или вернул некорректный ответ."""


class Place(BaseModel):
    name: str
    country: str
    region: str | None = Field(default=None, description="Административная область")


class CurrentWeather(BaseModel):
    place: Place
    temperature_c: float = Field(description="Температура воздуха, °C")
    condition: str = Field(description="Описание погодных условий")
    weather_code: int = Field(description="Код погоды WMO")
    wind_speed: float
    wind_speed_unit: str
    observed_at: str = Field(description="Время наблюдения, локальное время места (ISO 8601)")
    timezone: str = Field(description="Часовой пояс данных (IANA)")


class WeatherResult(BaseModel):
    status: Literal["ok", "not_found", "ambiguous"]
    weather: CurrentWeather | None = None
    candidates: list[Place] = Field(
        default_factory=list,
        description="Варианты при status=ambiguous: уточните город, например 'Москва, Idaho'",
    )


async def _get_json(session: aiohttp.ClientSession, url: str, params: dict) -> dict:
    last_error: Exception | None = None
    for _ in range(READ_ATTEMPTS):
        try:
            async with session.get(url, params=params, timeout=HTTP_TIMEOUT) as resp:
                resp.raise_for_status()
                data = await resp.json()
            if not isinstance(data, dict):
                raise WeatherUpstreamError("неожиданная структура ответа")
            return data
        except WeatherUpstreamError:
            raise
        except (TimeoutError, aiohttp.ClientError, ValueError) as exc:
            last_error = exc
            await asyncio.sleep(0.2)
    raise WeatherUpstreamError(f"погодный сервис недоступен: {type(last_error).__name__}")


def _split_query(city: str) -> tuple[str, str]:
    """'Москва, Idaho' -> ('Москва', 'idaho'); уточнение после запятой фильтрует варианты."""
    name, _, qualifier = city.partition(",")
    return name.strip(), qualifier.strip().casefold()


def _matches(item: dict, qualifier: str) -> bool:
    code = (item.get("country_code") or "").casefold()
    text = [(item.get(k) or "").casefold() for k in ("country", "admin1")]
    return qualifier == code or (len(qualifier) >= 3 and any(qualifier in t for t in text))


def _place(item: dict) -> Place:
    return Place(name=item["name"], country=item.get("country", ""), region=item.get("admin1"))


async def fetch_weather(session: aiohttp.ClientSession, city: str) -> WeatherResult:
    name, qualifier = _split_query(city)
    if not name:
        return WeatherResult(status="not_found")

    geo = await _get_json(
        session, GEOCODING_URL, {"name": name, "count": 10, "language": "ru", "format": "json"}
    )
    found = geo.get("results") or []
    if qualifier:
        found = [item for item in found if _matches(item, qualifier)]
    if not found:
        return WeatherResult(status="not_found")
    # «Новосибирск» не неоднозначен из-за «Новосибирское»: точные имена в приоритете.
    found = [i for i in found if i["name"].casefold() == name.casefold()] or found

    # Разные места с одним названием: первый результат автоматически не выбираем.
    places = list(dict.fromkeys(_place(i).model_dump_json() for i in found))
    if len(places) > 1:
        return WeatherResult(
            status="ambiguous",
            candidates=[Place.model_validate_json(p) for p in places[:5]],
        )

    place = found[0]
    forecast = await _get_json(
        session,
        FORECAST_URL,
        {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,weather_code,wind_speed_10m",
            "wind_speed_unit": "ms",
            "timezone": "auto",
        },
    )
    try:
        current = forecast["current"]
        code = int(current["weather_code"])
        weather = CurrentWeather(
            place=_place(place),
            temperature_c=current["temperature_2m"],
            condition=WMO_DESCRIPTIONS.get(code, f"код погоды {code}"),
            weather_code=code,
            wind_speed=current["wind_speed_10m"],
            wind_speed_unit=forecast["current_units"]["wind_speed_10m"],
            observed_at=current["time"],
            timezone=forecast["timezone"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise WeatherUpstreamError("погодный сервис вернул некорректный ответ") from exc
    return WeatherResult(status="ok", weather=weather)
