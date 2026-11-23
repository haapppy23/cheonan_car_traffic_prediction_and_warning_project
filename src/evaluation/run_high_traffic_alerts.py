"""최고 회귀 모델의 검증 예측을 접근로별 고교통량 경보로 평가한다."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import classification_metrics


class VolumePredictor(Protocol):
    def predict(self, features: pd.DataFrame) -> np.ndarray: ...


class AlertEvaluationError(ValueError):
    """경보 기준·정답·예측값이 서로 일치하지 않을 때 발생한다."""


MODEL_LABELS = {
    "hist_gradient_boosting": "HistGradientBoosting",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
}


def evaluate_high_traffic_alerts(
    validation: pd.DataFrame,
    predicted_volume: object,
    *,
    model_name: str = "hist_gradient_boosting",
    split_name: str = "validation",
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    required = {
        "target_volume",
        "target_high_volume",
        "high_traffic_threshold",
    }
    missing = sorted(required - set(validation.columns))
    if missing:
        raise AlertEvaluationError(f"경보 평가에 필요한 열이 없습니다: {missing}")
    if validation["target_high_volume"].isna().any():
        raise AlertEvaluationError("경보 정답에 결측값이 있습니다.")
    if validation["high_traffic_threshold"].le(0).any():
        raise AlertEvaluationError("고교통량 기준값은 0보다 커야 합니다.")

    prediction = np.asarray(predicted_volume, dtype="float64").reshape(-1)
    if len(prediction) != len(validation):
        raise AlertEvaluationError("경보 예측값과 검증 행 수가 다릅니다.")
    if not np.isfinite(prediction).all():
        raise AlertEvaluationError("예측 교통량에 결측·무한대가 있습니다.")

    expected_label = validation["target_volume"].gt(
        validation["high_traffic_threshold"]
    )
    actual_label = validation["target_high_volume"].astype("int8")
    if not expected_label.astype("int8").equals(actual_label):
        raise AlertEvaluationError("교통량과 고교통량 정답이 일치하지 않습니다.")

    threshold = validation["high_traffic_threshold"].to_numpy(dtype="float64")
    predicted_label = prediction > threshold
    risk_score = prediction / threshold
    metrics = classification_metrics(actual_label, predicted_label, risk_score)
    result = pd.DataFrame(
        [
            {
                "model": model_name,
                "split": split_name,
                "rows": len(validation),
                "actual_high_volume": int(actual_label.sum()),
                "predicted_alerts": int(predicted_label.sum()),
                **metrics,
            }
        ]
    )
    details: dict[str, int | float] = {
        "rows": len(validation),
        "actual_positive_rate": float(actual_label.mean()),
        "predicted_alert_rate": float(predicted_label.mean()),
    }
    return result, details


def evaluate_model_alerts(
    features: pd.DataFrame,
    model: VolumePredictor,
    *,
    model_name: str = "hist_gradient_boosting",
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """시험 구간을 건드리지 않고 검증 행만 모델에 전달한다."""
    required = {"split", "is_model_ready"}
    missing = sorted(required - set(features.columns))
    if missing:
        raise AlertEvaluationError(f"검증 구간 선택에 필요한 열이 없습니다: {missing}")
    validation = features.loc[
        features["split"].eq("validation") & features["is_model_ready"]
    ].copy()
    if validation.empty:
        raise AlertEvaluationError("검증 구간에 경보 평가 가능한 행이 없습니다.")
    prediction = model.predict(validation)
    return evaluate_high_traffic_alerts(
        validation, prediction, model_name=model_name
    )


def write_report(
    metrics: pd.DataFrame,
    details: dict[str, int | float],
    report_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    row = metrics.iloc[0]
    model_label = MODEL_LABELS.get(str(row["model"]), str(row["model"]))
    report = f"""# {model_label} 고교통량 경보 검증 결과

## 평가 기준

{model_label}가 예측한 교통량이 해당 접근로의 학습 구간 90백분위수를 초과하면 경보로 판단했다. PR-AUC 점수는 `예측 교통량 ÷ 접근로별 기준값`을 사용했다. 이 점수는 확률이 아니라 서로 다른 접근로의 기준을 맞춘 상대 위험 점수이다.

- 검증 행: {details['rows']:,}개
- 실제 고교통량: {int(row['actual_high_volume']):,}건, {details['actual_positive_rate'] * 100:.2f}%
- 발생 경보: {int(row['predicted_alerts']):,}건, {details['predicted_alert_rate'] * 100:.2f}%
- 시험 구간: 사용하지 않음

## 경보 지표

| Precision | Recall | F1 | PR-AUC |
|---:|---:|---:|---:|
| {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['pr_auc']:.3f} |

## 혼동행렬

| 구분 | 건수 |
|---|---:|
| 실제 고교통량을 맞춘 경보, TP | {int(row['true_positive']):,} |
| 잘못 발생한 경보, FP | {int(row['false_positive']):,} |
| 놓친 고교통량, FN | {int(row['false_negative']):,} |
| 정상 시간을 정상으로 판단, TN | {int(row['true_negative']):,} |

## 해석 원칙

현재 결과는 학습 구간에서 정한 90백분위수를 그대로 적용한 기준선이다. Recall을 높이기 위해 임의로 경보 기준을 낮추지 않았으며, 기준 조정이 필요하면 검증 구간에서 별도 비교한 뒤 결정한다.
"""
    report_path.write_text(report, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/features/model_features.parquet"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/hist_gradient_boosting.joblib"),
    )
    parser.add_argument("--model-name", default="hist_gradient_boosting")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/evaluation/high_traffic_alert_metrics.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/high_traffic_alert_results.md"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.model.exists():
        raise FileNotFoundError(
            f"학습 모델이 없습니다: {args.model}. 해당 모델 학습 명령을 먼저 실행하십시오."
        )
    features = pd.read_parquet(args.input)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Could not find the number of physical cores",
            category=UserWarning,
            module=r"joblib\.externals\.loky\.backend\.context",
        )
        model = joblib.load(args.model)
    metrics, details = evaluate_model_alerts(
        features, model, model_name=args.model_name
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output, index=False, encoding="utf-8-sig")
    write_report(metrics, details, args.report)
    print(metrics.to_string(index=False))
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
