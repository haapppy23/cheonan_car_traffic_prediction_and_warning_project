from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.random_forest import (
    RandomForestTrafficModel,
    RandomForestValidationError,
)
from src.models.ridge import MODEL_FEATURES
from src.models.train_random_forest import train_and_validate_random_forest
from src.models.train_ridge import select_ready_split


def sample_model_data(rows: int = 240) -> pd.DataFrame:
    index = np.arange(rows)
    frame = pd.DataFrame(
        {
            "split": np.where(index < 180, "train", "validation"),
            "is_model_ready": True,
            "target_month": 1,
            "target_is_weekend": index % 7 >= 5,
            "hour_sin": np.sin(2 * np.pi * (index % 24) / 24),
            "hour_cos": np.cos(2 * np.pi * (index % 24) / 24),
            "weekday_sin": np.sin(2 * np.pi * (index % 7) / 7),
            "weekday_cos": np.cos(2 * np.pi * (index % 7) / 7),
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


def test_random_forest_fits_and_predicts_nonnegative_volume() -> None:
    frame = sample_model_data()
    train = select_ready_split(frame, "train")
    validation = select_ready_split(frame, "validation")
    model = RandomForestTrafficModel(n_estimators=5, max_depth=5).fit(
        train, train["target_volume"]
    )
    prediction = model.predict(validation)
    assert len(prediction) == len(validation)
    assert (prediction >= 0).all()


def test_random_forest_explains_single_prediction() -> None:
    frame = sample_model_data()
    train = select_ready_split(frame, "train")
    validation = select_ready_split(frame, "validation")
    model = RandomForestTrafficModel(n_estimators=5, max_depth=5).fit(
        train, train["target_volume"]
    )

    factors = model.explain(validation.iloc[[0]], top_k=3)

    assert len(factors) == 3
    assert all(
        set(factor) == {"feature", "label", "contribution", "effect"}
        for factor in factors
    )
    assert all(factor["effect"] in {"increase", "decrease"} for factor in factors)
    assert [abs(float(factor["contribution"])) for factor in factors] == sorted(
        [abs(float(factor["contribution"])) for factor in factors], reverse=True
    )


def test_random_forest_rejects_missing_feature() -> None:
    frame = sample_model_data().drop(columns=[MODEL_FEATURES[0]])
    with pytest.raises(RandomForestValidationError, match="필요한 열"):
        RandomForestTrafficModel(n_estimators=5).fit(
            frame, frame["target_volume"]
        )


def test_saved_random_forest_preserves_predictions(tmp_path) -> None:
    frame = sample_model_data()
    train = select_ready_split(frame, "train")
    validation = select_ready_split(frame, "validation")
    model = RandomForestTrafficModel(n_estimators=5, max_depth=5).fit(
        train, train["target_volume"]
    )
    expected = model.predict(validation)
    path = tmp_path / "random_forest.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    assert np.allclose(expected, loaded.predict(validation))


def test_random_forest_comparison_ignores_test_rows() -> None:
    frame = sample_model_data()
    test_row = frame.iloc[[0]].copy()
    test_row["split"] = "test"
    test_row["target_volume"] = 10_000_000
    extended = pd.concat([frame, test_row], ignore_index=True)
    _, original_metrics, original_details = train_and_validate_random_forest(frame)
    _, extended_metrics, extended_details = train_and_validate_random_forest(extended)
    original = original_metrics.loc[original_metrics["model"].eq("random_forest")]
    changed = extended_metrics.loc[extended_metrics["model"].eq("random_forest")]
    assert np.allclose(
        original[["mae", "wape", "rmse"]].to_numpy(),
        changed[["mae", "wape", "rmse"]].to_numpy(),
    )
    assert original_details == extended_details
