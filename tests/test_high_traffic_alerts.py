from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import MetricValidationError, classification_metrics
from src.evaluation.run_high_traffic_alerts import (
    AlertEvaluationError,
    evaluate_high_traffic_alerts,
    evaluate_model_alerts,
)


def test_classification_metrics_have_expected_values() -> None:
    result = classification_metrics(
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [0.9, 0.4, 0.8, 0.1],
    )
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)
    assert result["pr_auc"] == pytest.approx(5 / 6)
    assert result["true_positive"] == 1
    assert result["false_negative"] == 1


def test_classification_metrics_require_both_actual_classes() -> None:
    with pytest.raises(MetricValidationError, match="양성과 음성"):
        classification_metrics([0, 0], [0, 0], [0.1, 0.2])


def sample_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["validation", "validation", "validation", "test"],
            "is_model_ready": True,
            "target_volume": [120, 80, 130, 10_000],
            "target_high_volume": pd.Series([1, 0, 1, 1], dtype="Int8"),
            "high_traffic_threshold": [100, 100, 100, 100],
            "prediction": [110, 90, 95, 10_000],
        }
    )


class PredictionColumnModel:
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return features["prediction"].to_numpy(dtype="float64")


def test_alert_evaluation_uses_approach_threshold() -> None:
    validation = sample_features().iloc[:3]
    result, details = evaluate_high_traffic_alerts(
        validation, validation["prediction"]
    )
    row = result.iloc[0]
    assert row["actual_high_volume"] == 2
    assert row["predicted_alerts"] == 1
    assert row["true_positive"] == 1
    assert row["false_negative"] == 1
    assert details["rows"] == 3


def test_alert_evaluation_rejects_inconsistent_target_label() -> None:
    validation = sample_features().iloc[:3].copy()
    validation.loc[0, "target_high_volume"] = 0
    with pytest.raises(AlertEvaluationError, match="일치하지 않습니다"):
        evaluate_high_traffic_alerts(validation, validation["prediction"])


def test_model_alert_evaluation_ignores_test_rows() -> None:
    result, details = evaluate_model_alerts(sample_features(), PredictionColumnModel())
    assert result.loc[0, "rows"] == 3
    assert details["rows"] == 3
