from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import MetricValidationError, regression_metrics
from src.evaluation.run_baselines import BaselineEvaluationError, evaluate_baselines


def test_regression_metrics_have_expected_values() -> None:
    result = regression_metrics([100, 200], [90, 230])
    assert result["mae"] == pytest.approx(20.0)
    assert result["wape"] == pytest.approx(40 / 300)
    assert result["rmse"] == pytest.approx(np.sqrt(500))


def test_regression_metrics_reject_non_finite_values() -> None:
    with pytest.raises(MetricValidationError, match="결측"):
        regression_metrics([100, np.nan], [90, 110])


def sample_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["validation", "validation", "validation", "train"],
            "is_model_ready": [True, True, False, True],
            "target_volume": [100, 200, 999, 50],
            "lag_1": [90, 210, 0, 50],
            "lag_24": [80, 220, 0, 40],
            "lag_168": [70, 230, 0, 30],
        }
    )


def test_baselines_use_only_ready_rows_from_requested_split() -> None:
    result = evaluate_baselines(sample_features())
    assert set(result["model"]) == {
        "persistence",
        "daily_seasonal",
        "weekly_seasonal",
    }
    assert result["rows"].eq(2).all()
    persistence = result.loc[result["model"].eq("persistence")].iloc[0]
    assert persistence["mae"] == pytest.approx(10.0)


def test_test_split_is_blocked_by_default() -> None:
    frame = sample_features().copy()
    frame.loc[0, "split"] = "test"
    with pytest.raises(BaselineEvaluationError, match="최종 모델"):
        evaluate_baselines(frame, split="test")
