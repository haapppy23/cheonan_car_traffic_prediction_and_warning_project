"""교통량 회귀 모델의 공통 평가 지표."""

from __future__ import annotations

import numpy as np


class MetricValidationError(ValueError):
    """평가 대상 배열이 유효하지 않을 때 발생한다."""


def regression_metrics(
    actual: object,
    predicted: object,
) -> dict[str, float]:
    """MAE, WAPE, RMSE를 동일한 검증 규칙으로 계산한다."""
    y_true = np.asarray(actual, dtype="float64").reshape(-1)
    y_pred = np.asarray(predicted, dtype="float64").reshape(-1)
    if y_true.size == 0:
        raise MetricValidationError("평가할 행이 없습니다.")
    if y_true.shape != y_pred.shape:
        raise MetricValidationError("실제값과 예측값의 행 수가 다릅니다.")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise MetricValidationError("실제값 또는 예측값에 결측·무한대가 있습니다.")

    absolute_error = np.abs(y_true - y_pred)
    denominator = np.abs(y_true).sum()
    if denominator == 0:
        raise MetricValidationError("실제 교통량 합계가 0이어서 WAPE를 계산할 수 없습니다.")
    return {
        "mae": float(absolute_error.mean()),
        "wape": float(absolute_error.sum() / denominator),
        "rmse": float(np.sqrt(np.mean(np.square(y_true - y_pred)))),
    }
