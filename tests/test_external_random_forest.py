from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_external_features import WEATHER_COLUMNS
from src.models.external_random_forest import (
    ExternalRandomForestError,
    ExternalRandomForestModel,
)
from src.models.ridge import MODEL_FEATURES


def sample_model_data(rows: int = 240) -> pd.DataFrame:
    index = np.arange(rows)
    frame = pd.DataFrame(
        {
            "target_timestamp": pd.date_range("2025-01-01", periods=rows, freq="h"),
            "intersection": np.where(index % 2 == 0, "교차로A", "교차로B"),
            "approach": np.where(index % 3 == 0, "북", "남"),
            "target_month": 1,
            "target_is_weekend": index % 7 >= 5,
            "hour_sin": np.sin(2 * np.pi * (index % 24) / 24),
            "hour_cos": np.cos(2 * np.pi * (index % 24) / 24),
            "weekday_sin": np.sin(2 * np.pi * (index % 7) / 7),
            "weekday_cos": np.cos(2 * np.pi * (index % 7) / 7),
            "lag_1": 100 + index,
            "lag_2": 99 + index,
            "lag_3": 98 + index,
            "lag_24": 90 + index,
            "lag_168": 80 + index,
            "rolling_mean_3": 99 + index,
            "rolling_mean_6": 97 + index,
            "rolling_std_6": 2.0,
            "target_is_holiday": index % 48 < 24,
            "weather_temperature_2m": 5 + np.sin(index / 24),
            "weather_precipitation": np.where(index % 30 == 0, 1.0, 0.0),
            "weather_wind_speed_10m": 3.0,
            "target_volume": 105 + index,
        }
    )
    return frame


def test_external_random_forest_fits_and_predicts_nonnegative_values() -> None:
    frame = sample_model_data()
    model = ExternalRandomForestModel().fit(frame, frame["target_volume"])
    prediction = model.predict(frame.head(8))
    assert prediction.shape == (8,)
    assert np.isfinite(prediction).all()
    assert (prediction >= 0).all()


def test_external_random_forest_requires_every_external_feature() -> None:
    frame = sample_model_data().drop(columns=[WEATHER_COLUMNS[0]])
    with pytest.raises(ExternalRandomForestError):
        ExternalRandomForestModel().fit(frame, frame["target_volume"])


def test_external_model_data_contains_all_base_features() -> None:
    assert set(MODEL_FEATURES).issubset(sample_model_data().columns)
