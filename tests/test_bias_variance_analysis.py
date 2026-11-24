from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.run_bias_variance_analysis import (
    BiasVarianceAnalysisError,
    evaluate_bias_variance,
)


def sample_features(rows: int = 180) -> pd.DataFrame:
    index = np.arange(rows)
    split = np.where(index < 120, "train", "validation")
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


def test_bias_variance_uses_only_train_and_validation() -> None:
    frame = sample_features()
    test_row = frame.iloc[[0]].copy()
    test_row["split"] = "test"
    test_row["target_volume"] = 10_000_000
    extended = pd.concat([frame, test_row], ignore_index=True)

    expected = evaluate_bias_variance(
        frame,
        train_sizes=(40, 80),
        depths=(4, 8),
        validation_rows=30,
        n_estimators=3,
    )
    actual = evaluate_bias_variance(
        extended,
        train_sizes=(40, 80),
        depths=(4, 8),
        validation_rows=30,
        n_estimators=3,
    )

    pd.testing.assert_frame_equal(expected, actual)
    assert set(actual["max_depth"]) == {4, 8}
    assert set(actual["train_rows"]) == {40, 80}
    assert (actual["validation_rows"] == 30).all()


def test_bias_variance_rejects_oversized_training_samples() -> None:
    with pytest.raises(BiasVarianceAnalysisError, match="작은 표본"):
        evaluate_bias_variance(
            sample_features(), train_sizes=(500,), depths=(4,), validation_rows=20
        )
