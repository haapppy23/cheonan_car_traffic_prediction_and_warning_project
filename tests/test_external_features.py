from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.download_cheonan_weather import (
    WeatherDownloadError,
    parse_weather_response,
)
from src.features.build_external_features import (
    ExternalFeatureError,
    combine_external_features,
    korean_holiday_flags,
)


def weather_payload() -> dict[str, object]:
    return {
        "hourly": {
            "time": ["2024-01-01T00:00", "2024-01-01T01:00"],
            "temperature_2m": [-2.0, -1.5],
            "precipitation": [0.0, 0.1],
            "wind_speed_10m": [3.0, 4.0],
        }
    }


def test_weather_response_parses_hourly_columns() -> None:
    result = parse_weather_response(weather_payload())
    assert result.columns.tolist() == [
        "timestamp",
        "weather_temperature_2m",
        "weather_precipitation",
        "weather_wind_speed_10m",
    ]
    assert result.iloc[1]["weather_precipitation"] == 0.1


def test_weather_response_rejects_non_hourly_gap() -> None:
    payload = weather_payload()
    payload["hourly"]["time"][1] = "2024-01-01T02:00"
    with pytest.raises(WeatherDownloadError):
        parse_weather_response(payload)


def test_korean_holiday_flags_include_new_year_and_exclude_next_day() -> None:
    flags = korean_holiday_flags(
        pd.Series(pd.to_datetime(["2024-01-01 12:00", "2024-01-02 12:00"]))
    )
    assert flags.tolist() == [True, False]


def test_external_features_join_weather_and_replace_holiday() -> None:
    timestamps = pd.date_range("2024-01-01", periods=2, freq="h")
    features = pd.DataFrame(
        {"target_timestamp": timestamps, "target_is_holiday": [np.nan, np.nan]}
    )
    weather = parse_weather_response(weather_payload())
    result = combine_external_features(features, weather)
    assert result["target_is_holiday"].tolist() == [True, True]
    assert result["weather_temperature_2m"].tolist() == [-2.0, -1.5]


def test_external_features_reject_missing_weather_hour() -> None:
    features = pd.DataFrame(
        {"target_timestamp": pd.to_datetime(["2024-01-01 03:00"])}
    )
    with pytest.raises(ExternalFeatureError):
        combine_external_features(features, parse_weather_response(weather_payload()))
