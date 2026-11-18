from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.run_error_analysis import analyze_validation_errors


class PredictionColumnModel:
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return features["prediction"].to_numpy(dtype="float64")


def sample_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["validation", "validation", "validation", "test"],
            "is_model_ready": True,
            "target_timestamp": pd.to_datetime(
                [
                    "2026-01-01 08:00",
                    "2026-01-01 09:00",
                    "2026-01-02 08:00",
                    "2026-06-01 00:00",
                ]
            ),
            "intersection": ["교차로A", "교차로A", "교차로B", "교차로A"],
            "approach": ["북", "북", "남", "북"],
            "target_hour": [8, 9, 8, 0],
            "target_is_weekend": [False, False, False, False],
            "target_volume": [120, 80, 150, 10_000],
            "target_high_volume": pd.Series([1, 0, 1, 1], dtype="Int8"),
            "high_traffic_threshold": [100, 100, 100, 100],
            "prediction": [110, 90, 95, 10_000],
        }
    )


def test_error_analysis_uses_validation_rows_only() -> None:
    tables, summary = analyze_validation_errors(
        sample_features(), PredictionColumnModel()
    )
    assert summary["rows"] == 3
    assert tables["intersection"]["rows"].sum() == 3
    assert tables["largest_errors"]["target_volume"].max() == 150


def test_error_analysis_collects_false_negatives() -> None:
    tables, summary = analyze_validation_errors(
        sample_features(), PredictionColumnModel()
    )
    assert summary["false_negatives"] == 1
    missed = tables["false_negatives"].iloc[0]
    assert missed["intersection"] == "교차로B"
    assert missed["underprediction"] == 55


def test_error_analysis_groups_intersection_metrics() -> None:
    tables, _ = analyze_validation_errors(sample_features(), PredictionColumnModel())
    intersection_a = tables["intersection"].loc[
        tables["intersection"]["intersection"].eq("교차로A")
    ].iloc[0]
    assert intersection_a["rows"] == 2
    assert intersection_a["mae"] == 10
