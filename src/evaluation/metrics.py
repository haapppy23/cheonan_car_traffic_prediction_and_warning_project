"""교통량 회귀 모델의 공통 평가 지표."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


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


def classification_metrics(
    actual: object,
    predicted: object,
    score: object,
) -> dict[str, float | int]:
    """이진 경보의 Precision, Recall, F1, PR-AUC와 혼동행렬을 계산한다."""
    y_true = np.asarray(actual, dtype="int8").reshape(-1)
    y_pred = np.asarray(predicted, dtype="int8").reshape(-1)
    y_score = np.asarray(score, dtype="float64").reshape(-1)
    if y_true.size == 0:
        raise MetricValidationError("평가할 경보 행이 없습니다.")
    if y_true.shape != y_pred.shape or y_true.shape != y_score.shape:
        raise MetricValidationError("경보 실제값·예측값·점수의 행 수가 다릅니다.")
    if not np.isfinite(y_score).all():
        raise MetricValidationError("경보 점수에 결측·무한대가 있습니다.")
    if not set(np.unique(y_true)).issubset({0, 1}) or not set(
        np.unique(y_pred)
    ).issubset({0, 1}):
        raise MetricValidationError("경보 실제값과 예측값은 0 또는 1이어야 합니다.")
    if set(np.unique(y_true)) != {0, 1}:
        raise MetricValidationError("PR-AUC 계산에는 실제 양성과 음성이 모두 필요합니다.")

    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "true_positive": int(true_positive),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_negative": int(true_negative),
    }
