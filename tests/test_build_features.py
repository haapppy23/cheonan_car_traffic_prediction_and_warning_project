from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    FeatureValidationError,
    assign_time_split,
    build_feature_table,
    mark_all_zero_target_days,
    validate_hourly_data,
)


def hourly_sample(periods: int = 240) -> pd.DataFrame:
    timestamp = pd.date_range("2024-01-01", periods=periods, freq="h")
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "intersection": "교차로A",
            "approach": "북",
            "volume": np.arange(periods, dtype=int),
            "is_holiday": pd.Series(pd.NA, index=range(periods), dtype="boolean"),
        }
    )


def test_lags_and_rolling_means_use_only_past_values() -> None:
    features, _ = build_feature_table(
        hourly_sample(),
        train_end=pd.Timestamp("2024-01-08"),
        validation_end=pd.Timestamp("2024-01-09"),
    )
    row = features.loc[
        features["target_timestamp"].eq(pd.Timestamp("2024-01-08 00:00"))
    ].iloc[0]
    assert row["target_volume"] == 168
    assert row["lag_1"] == 167
    assert row["lag_2"] == 166
    assert row["lag_3"] == 165
    assert row["lag_24"] == 144
    assert row["lag_168"] == 0
    assert row["rolling_mean_3"] == pytest.approx(166)
    assert row["rolling_mean_6"] == pytest.approx(164.5)


def test_changing_future_volume_does_not_change_earlier_features() -> None:
    source = hourly_sample()
    original, _ = build_feature_table(
        source,
        train_end=pd.Timestamp("2024-01-08"),
        validation_end=pd.Timestamp("2024-01-09"),
    )
    source.loc[source["timestamp"].gt("2024-01-08 00:00"), "volume"] = 999_999
    changed, _ = build_feature_table(
        source,
        train_end=pd.Timestamp("2024-01-08"),
        validation_end=pd.Timestamp("2024-01-09"),
    )
    feature_columns = [
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_24",
        "lag_168",
        "rolling_mean_3",
        "rolling_mean_6",
        "rolling_std_6",
    ]
    target = original["target_timestamp"].eq(pd.Timestamp("2024-01-08 00:00"))
    assert original.loc[target, feature_columns].reset_index(drop=True).equals(
        changed.loc[target, feature_columns].reset_index(drop=True)
    )


def test_threshold_uses_training_period_only() -> None:
    source = hourly_sample()
    first, first_thresholds = build_feature_table(
        source,
        train_end=pd.Timestamp("2024-01-08"),
        validation_end=pd.Timestamp("2024-01-09"),
    )
    source.loc[source["timestamp"].ge("2024-01-08"), "volume"] = 1_000_000
    second, second_thresholds = build_feature_table(
        source,
        train_end=pd.Timestamp("2024-01-08"),
        validation_end=pd.Timestamp("2024-01-09"),
    )
    assert first_thresholds["high_traffic_threshold"].iloc[0] == pytest.approx(
        second_thresholds["high_traffic_threshold"].iloc[0]
    )
    assert first["high_traffic_threshold"].iloc[0] == pytest.approx(
        second["high_traffic_threshold"].iloc[0]
    )


def test_split_boundaries_use_target_timestamp() -> None:
    timestamps = pd.Series(
        pd.to_datetime(
            ["2025-12-31 23:00", "2026-01-01 00:00", "2026-06-01 00:00"]
        )
    )
    result = assign_time_split(timestamps)
    assert result.tolist() == ["train", "validation", "test"]


def test_all_zero_target_day_is_not_model_ready() -> None:
    source = hourly_sample()
    zero_day = source["timestamp"].dt.normalize().eq(pd.Timestamp("2024-01-09"))
    source.loc[zero_day, "volume"] = 0
    features, _ = build_feature_table(
        source,
        train_end=pd.Timestamp("2024-01-08"),
        validation_end=pd.Timestamp("2024-01-09"),
    )
    result = features.loc[
        features["target_timestamp"].dt.normalize().eq(pd.Timestamp("2024-01-09"))
    ]
    assert result["target_all_zero_approach_day"].all()
    assert not result["is_model_ready"].any()
    assert result["target_high_volume"].isna().all()


def test_partial_zero_day_is_not_misclassified_as_full_zero_day() -> None:
    source = hourly_sample(periods=12)
    source["volume"] = 0
    assert not mark_all_zero_target_days(source).any()


def test_gap_in_hourly_series_is_rejected() -> None:
    source = hourly_sample().drop(index=20)
    with pytest.raises(FeatureValidationError, match="한 시간 간격"):
        validate_hourly_data(source)
