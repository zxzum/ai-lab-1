"""Тесты погодного инструмента MCP-сервера: сеть заменена контролируемой заглушкой."""

import pytest

from app.mcp_server import weather
from app.mcp_server.weather import WeatherUpstreamError, fetch_weather


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    async def __aenter__(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    async def json(self):
        return self.payload


class FakeSession:
    """Отдаёт заранее заданные ответы по URL и считает обращения."""

    def __init__(self, geo, forecast=None):
        self.responses = {weather.GEOCODING_URL: geo, weather.FORECAST_URL: forecast}
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        return FakeResponse(self.responses[url])


def place(name="Казань", admin1="Татарстан", country="Россия", code="RU"):
    return {
        "name": name,
        "latitude": 55.78,
        "longitude": 49.12,
        "admin1": admin1,
        "country": country,
        "country_code": code,
    }


FORECAST = {
    "timezone": "Europe/Moscow",
    "current_units": {"wind_speed_10m": "m/s"},
    "current": {
        "time": "2026-10-25T21:00",
        "temperature_2m": 4.9,
        "weather_code": 61,
        "wind_speed_10m": 1.1,
    },
}


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def instant(_):
        pass

    monkeypatch.setattr(weather.asyncio, "sleep", instant)


async def test_weather_ok():
    session = FakeSession({"results": [place()]}, FORECAST)

    result = await fetch_weather(session, "Казань")

    assert result.status == "ok"
    assert result.weather.place.name == "Казань"
    assert result.weather.temperature_c == 4.9
    assert result.weather.condition == "слабый дождь"
    assert result.weather.wind_speed_unit == "m/s"
    assert result.weather.timezone == "Europe/Moscow"


async def test_city_not_found():
    session = FakeSession({})

    result = await fetch_weather(session, "Zzzxqq")

    assert result.status == "not_found"
    assert result.weather is None


async def test_same_name_cities_are_ambiguous_and_first_is_not_chosen():
    geo = {"results": [place("Москва", "Москва"), place("Москва", "Айдахо", "США", "US")]}
    session = FakeSession(geo, FORECAST)

    result = await fetch_weather(session, "Москва")

    assert result.status == "ambiguous"
    assert [c.region for c in result.candidates] == ["Москва", "Айдахо"]
    assert session.calls == 1  # прогноз не запрашивался


async def test_qualifier_after_comma_narrows_candidates():
    geo = {"results": [place("Москва", "Москва"), place("Москва", "Айдахо", "США", "US")]}
    session = FakeSession(geo, FORECAST)

    result = await fetch_weather(session, "Москва, Айдахо")

    assert result.status == "ok"
    assert result.weather.place.region == "Айдахо"


async def test_several_qualifiers_after_commas_all_must_match():
    geo = {"results": [place("Москва", "Москва"), place("Москва", "Айдахо", "США", "US")]}

    ok = await fetch_weather(FakeSession(geo, FORECAST), "Москва, Айдахо, США")
    none = await fetch_weather(FakeSession(geo, FORECAST), "Москва, Айдахо, Россия")

    assert ok.status == "ok" and ok.weather.place.region == "Айдахо"
    assert none.status == "not_found"


async def test_exact_name_beats_derived_names():
    geo = {"results": [place("Новосибирск", "Новосибирская"), place("Новосибирское", "Сахалин")]}
    session = FakeSession(geo, FORECAST)

    result = await fetch_weather(session, "Новосибирск")

    assert result.status == "ok"


async def test_timeout_is_retried_then_reported():
    session = FakeSession(TimeoutError())

    with pytest.raises(WeatherUpstreamError):
        await fetch_weather(session, "Казань")

    assert session.calls == weather.READ_ATTEMPTS


async def test_malformed_forecast_is_rejected():
    session = FakeSession({"results": [place()]}, {"current": {}})

    with pytest.raises(WeatherUpstreamError):
        await fetch_weather(session, "Казань")
