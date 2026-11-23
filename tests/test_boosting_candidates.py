from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.boosting_candidates import (
    BoostingCandidateValidationError,
    CatBoostTrafficModel,
    XGBoostTrafficModel,
)
from src.models.ridge import MODEL_FEATURES
from src.models.train_extended_models import chronological_tuning_partitions
from src.models.train_ridge import select_ready_split


def sample_model_data(rows: int = 240) -> pd.DataFrame:
    index = np.arange(rows)
    timestamp = pd.date_range("2025-01-01", periods=rows, freq="h")
    frame = pd.DataFrame(
        {
            "target_timestamp": timestamp,
            "split": np.where(index < 180, "train", "validation"),
            "is_model_ready": True,
            "target_month": timestamp.month,
            "target_is_weekend": timestamp.dayofweek >= 5,
            "hour_sin": np.sin(2 * np.pi * timestamp.hour / 24),
            "hour_cos": np.cos(2 * np.pi * timestamp.hour / 24),
            "weekday_sin": np.sin(2 * np.pi * timestamp.dayofweek / 7),
            "weekday_cos": np.cos(2 * np.pi * timestamp.dayofweek / 7),
            "lag_1": index + 10,
            "lag_2": index + 9,
            "lag_3": index + 8,
            "lag_24": index,
            "lag_168": index / 2,
            "rolling_mean_3": index + 9,
            "rolling_mean_6": index + 7.5,
            "rolling_std_6": 1.7,
            "intersection": np.where(index % 2 == 0, "교차로A", "교차로B"),
            "approach": np.where(index % 2 == 0, "북", "남"),
        }
    )
    frame["target_volume"] = 2 * frame["lag_1"] + 5
    return frame


@pytest.mark.parametrize(
    "model",
    [
        XGBoostTrafficModel(n_estimators=10, max_depth=3, n_jobs=1),
        CatBoostTrafficModel(iterations=10, depth=4, thread_count=1),
    ],
)
def test_additional_boosting_models_predict_nonnegative_volume(model) -> None:
    frame = sample_model_data()
    train = select_ready_split(frame, "train")
    validation = select_ready_split(frame, "validation")
    model.fit(train, train["target_volume"])
    prediction = model.predict(validation)
    assert len(prediction) == len(validation)
    assert np.isfinite(prediction).all()
    assert (prediction >= 0).all()


def test_additional_boosting_model_rejects_missing_feature() -> None:
    frame = sample_model_data().drop(columns=[MODEL_FEATURES[0]])
    with pytest.raises(BoostingCandidateValidationError, match="필요한 열"):
        XGBoostTrafficModel(n_estimators=5, n_jobs=1).fit(
            frame, frame["target_volume"]
        )


def test_internal_tuning_split_is_chronological_and_ignores_test() -> None:
    frame = sample_model_data()
    test_row = frame.iloc[[0]].copy()
    test_row["split"] = "test"
    test_row["target_timestamp"] = pd.Timestamp("2026-06-01")
    extended = pd.concat([frame, test_row], ignore_index=True)
    tuning_start = frame.loc[120, "target_timestamp"]
    full_train, tuning_train, tuning_validation = chronological_tuning_partitions(
        extended, tuning_start=tuning_start
    )
    assert len(full_train) == 180
    assert tuning_train["target_timestamp"].max() < tuning_start
    assert tuning_validation["target_timestamp"].min() >= tuning_start
    assert tuning_train["target_timestamp"].max() < tuning_validation[
        "target_timestamp"
    ].min()
