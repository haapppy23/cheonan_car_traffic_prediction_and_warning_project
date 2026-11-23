from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.run_shap_analysis import (
    ShapAnalysisError,
    clean_feature_names,
    select_validation_sample,
    summarize_shap_values,
)
from src.models.ridge import MODEL_FEATURES


def sample_features() -> pd.DataFrame:
    rows = 12
    frame = pd.DataFrame(
        {
            "split": ["validation"] * 8 + ["train"] * 2 + ["test"] * 2,
            "is_model_ready": True,
            "target_timestamp": pd.date_range("2026-01-01", periods=rows, freq="h"),
        }
    )
    for feature in MODEL_FEATURES:
        if feature == "intersection":
            frame[feature] = "교차로A"
        elif feature == "approach":
            frame[feature] = "북"
        else:
            frame[feature] = np.arange(rows, dtype="float64")
    return frame


def test_validation_sample_excludes_train_and_test_and_is_deterministic() -> None:
    first = select_validation_sample(sample_features(), sample_size=4)
    second = select_validation_sample(sample_features(), sample_size=4)
    assert len(first) == 4
    assert set(first["split"]) == {"validation"}
    pd.testing.assert_frame_equal(first, second)


def test_validation_sample_rejects_nonpositive_size() -> None:
    with pytest.raises(ShapAnalysisError):
        select_validation_sample(sample_features(), sample_size=0)


def test_feature_names_remove_transformer_prefix() -> None:
    assert clean_feature_names(["numeric__lag_1", "categorical__approach"]) == [
        "lag_1",
        "approach",
    ]


def test_shap_summary_orders_mean_absolute_contribution() -> None:
    values = np.array([[1.0, -4.0], [-1.0, 2.0]])
    result = summarize_shap_values(values, ["lag_1", "lag_24"])
    assert result.iloc[0]["feature"] == "lag_24"
    assert result.iloc[0]["mean_abs_shap"] == 3.0
    assert result["importance_share"].sum() == pytest.approx(1.0)
