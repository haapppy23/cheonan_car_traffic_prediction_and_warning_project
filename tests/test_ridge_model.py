from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import joblib

from src.models.ridge import MODEL_FEATURES, RidgeTrafficModel, RidgeValidationError
from src.models.train_ridge import select_ready_split, train_and_validate


def sample_model_data(rows: int = 240) -> pd.DataFrame:
    index = np.arange(rows)
    split = np.where(index < 180, "train", "validation")
    frame = pd.DataFrame(
        {
            "split": split,
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


def test_ridge_model_fits_and_predicts_nonnegative_volume() -> None:
    frame = sample_model_data()
    train = select_ready_split(frame, "train")
    validation = select_ready_split(frame, "validation")
    model = RidgeTrafficModel(alpha=1.0).fit(train, train["target_volume"])
    prediction = model.predict(validation)
    assert len(prediction) == len(validation)
    assert (prediction >= 0).all()


def test_ridge_rejects_missing_model_feature() -> None:
    frame = sample_model_data().drop(columns=[MODEL_FEATURES[0]])
    with pytest.raises(RidgeValidationError, match="필요한 열"):
        RidgeTrafficModel().fit(frame, frame["target_volume"])


def test_saved_ridge_model_preserves_predictions(tmp_path) -> None:
    frame = sample_model_data()
    train = select_ready_split(frame, "train")
    validation = select_ready_split(frame, "validation")
    model = RidgeTrafficModel().fit(train, train["target_volume"])
    expected = model.predict(validation)
    path = tmp_path / "ridge.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    assert np.allclose(expected, loaded.predict(validation))


def test_training_compares_ridge_without_test_rows() -> None:
    frame = sample_model_data()
    test_row = frame.iloc[[0]].copy()
    test_row["split"] = "test"
    test_row["target_volume"] = 10_000_000
    extended = pd.concat([frame, test_row], ignore_index=True)
    _, original_metrics, original_details = train_and_validate(frame)
    _, extended_metrics, extended_details = train_and_validate(extended)
    ridge_original = original_metrics.loc[original_metrics["model"].eq("ridge")]
    ridge_extended = extended_metrics.loc[extended_metrics["model"].eq("ridge")]
    assert ridge_original[["mae", "wape", "rmse"]].reset_index(drop=True).equals(
        ridge_extended[["mae", "wape", "rmse"]].reset_index(drop=True)
    )
    assert original_details == extended_details
