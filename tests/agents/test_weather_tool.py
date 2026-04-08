from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from synapsekit.agents.tools.weather import WeatherTool


def _mock_response(data: dict):
    resp = MagicMock()
    resp.read.return_value = json.dumps(data).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


_CURRENT_RESP = {
    "name": "London",
    "sys": {"country": "GB"},
    "weather": [{"description": "overcast clouds"}],
    "main": {"temp": 14.2, "humidity": 72},
    "wind": {"speed": 5.1},
}

_FORECAST_RESP = {
    "city": {"name": "London", "country": "GB"},
    "list": [
        {
            "dt_txt": "2026-04-08 12:00:00",
            "weather": [{"description": "light rain"}],
            "main": {"temp": 13.0, "humidity": 80},
            "wind": {"speed": 4.5},
        },
        {
            "dt_txt": "2026-04-08 18:00:00",
            "weather": [{"description": "clear sky"}],
            "main": {"temp": 15.5, "humidity": 65},
            "wind": {"speed": 3.2},
        },
        {
            "dt_txt": "2026-04-09 12:00:00",
            "weather": [{"description": "scattered clouds"}],
            "main": {"temp": 16.0, "humidity": 58},
            "wind": {"speed": 2.8},
        },
    ],
}


class TestWeatherTool:
    @pytest.mark.asyncio
    async def test_current(self, monkeypatch):
        monkeypatch.setenv("OPENWEATHERMAP_API_KEY", "test-key")
        tool = WeatherTool()
        with patch("urllib.request.urlopen", return_value=_mock_response(_CURRENT_RESP)):
            res = await tool.run(action="current", city="London")
        assert not res.is_error
        assert "London, GB" in res.output
        assert "14.2" in res.output
        assert "72" in res.output
        assert "overcast clouds" in res.output

    @pytest.mark.asyncio
    async def test_forecast(self, monkeypatch):
        monkeypatch.setenv("OPENWEATHERMAP_API_KEY", "test-key")
        tool = WeatherTool()
        with patch("urllib.request.urlopen", return_value=_mock_response(_FORECAST_RESP)):
            res = await tool.run(action="forecast", city="London", days=2)
        assert not res.is_error
        assert "London, GB" in res.output
        assert "light rain" in res.output

    @pytest.mark.asyncio
    async def test_forecast_clamps_days(self, monkeypatch):
        monkeypatch.setenv("OPENWEATHERMAP_API_KEY", "test-key")
        tool = WeatherTool()
        with patch("urllib.request.urlopen", return_value=_mock_response(_FORECAST_RESP)):
            res = await tool.run(action="forecast", city="London", days=10)
        assert not res.is_error

    @pytest.mark.asyncio
    async def test_missing_city(self, monkeypatch):
        monkeypatch.setenv("OPENWEATHERMAP_API_KEY", "test-key")
        tool = WeatherTool()
        res = await tool.run(action="current")
        assert res.is_error
        assert "City" in res.error

    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENWEATHERMAP_API_KEY", raising=False)
        tool = WeatherTool()
        res = await tool.run(action="current", city="London")
        assert res.is_error
        assert "OPENWEATHERMAP_API_KEY" in res.error

    @pytest.mark.asyncio
    async def test_unknown_action(self, monkeypatch):
        monkeypatch.setenv("OPENWEATHERMAP_API_KEY", "test-key")
        tool = WeatherTool()
        res = await tool.run(action="hourly", city="London")
        assert res.is_error
        assert "Unknown action" in res.error

    @pytest.mark.asyncio
    async def test_network_error(self, monkeypatch):
        monkeypatch.setenv("OPENWEATHERMAP_API_KEY", "test-key")
        tool = WeatherTool()
        with patch("urllib.request.urlopen", side_effect=Exception("connection refused")):
            res = await tool.run(action="current", city="London")
        assert res.is_error

    @pytest.mark.asyncio
    async def test_default_action_is_current(self, monkeypatch):
        monkeypatch.setenv("OPENWEATHERMAP_API_KEY", "test-key")
        tool = WeatherTool()
        with patch("urllib.request.urlopen", return_value=_mock_response(_CURRENT_RESP)):
            res = await tool.run(city="London")
        assert not res.is_error
        assert "London" in res.output

    @pytest.mark.asyncio
    async def test_empty_forecast(self, monkeypatch):
        monkeypatch.setenv("OPENWEATHERMAP_API_KEY", "test-key")
        tool = WeatherTool()
        empty = {"city": {"name": "Nowhere", "country": "XX"}, "list": []}
        with patch("urllib.request.urlopen", return_value=_mock_response(empty)):
            res = await tool.run(action="forecast", city="Nowhere")
        assert not res.is_error
        assert "No forecast" in res.output

    def test_schema(self):
        tool = WeatherTool()
        s = tool.schema()
        assert s["function"]["name"] == "weather"
        props = s["function"]["parameters"]["properties"]
        assert "action" in props
        assert "city" in props
        assert "days" in props
