from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.finalize_random_forest import select_final_partitions


def sample_features() -> pd.DataFrame:
    rows = []
    starts = {
        "train": pd.Timestamp("2024-01-01"),
        "validation": pd.Timestamp("2026-01-01"),
        "test": pd.Timestamp("2026-06-01"),
    }
    for split, start in starts.items():
        for offset in range(2):
            value = offset + 10
            rows.append(
                {
                    "split": split,
                    "is_model_ready": True,
                    "target_timestamp": start + pd.Timedelta(hours=offset),
                    "target_volume": value,
                    "target_high_volume": int(value > 10),
                    "high_traffic_threshold": 10,
                    "target_month": start.month,
                    "target_is_weekend": False,
                    "hour_sin": 0.0,
                    "hour_cos": 1.0,
                    "weekday_sin": 0.0,
                    "weekday_cos": 1.0,
                    "lag_1": value - 1,
                    "lag_2": value - 2,
                    "lag_3": value - 3,
                    "lag_24": value - 1,
                    "lag_168": value - 1,
                    "rolling_mean_3": value - 2,
                    "rolling_mean_6": value - 3,
                    "rolling_std_6": 1.0,
                    "intersection": "교차로A",
                    "approach": "북",
                }
            )
    return pd.DataFrame(rows)


def test_final_training_combines_train_and_validation_only() -> None:
    final_train, test = select_final_partitions(sample_features())
    assert set(final_train["split"]) == {"train", "validation"}
    assert set(test["split"]) == {"test"}
    assert len(final_train) == 4
    assert len(test) == 2


def test_final_training_time_precedes_test_time() -> None:
    final_train, test = select_final_partitions(sample_features())
    assert final_train["target_timestamp"].max() < test["target_timestamp"].min()


def test_not_ready_rows_are_excluded_from_final_partitions() -> None:
    frame = sample_features()
    frame.loc[0, "is_model_ready"] = False
    final_train, test = select_final_partitions(frame)
    assert len(final_train) == 3
    assert len(test) == 2
