"""최고 모델의 검증 오차와 놓친 고교통량 사례를 구간별로 분석한다."""

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

from src.evaluation.metrics import regression_metrics
from src.evaluation.run_high_traffic_alerts import MODEL_LABELS, evaluate_high_traffic_alerts


class VolumePredictor(Protocol):
    def predict(self, features: pd.DataFrame) -> np.ndarray: ...


class ErrorAnalysisError(ValueError):
    """오류 분석 입력이나 예측값이 유효하지 않을 때 발생한다."""


PRECIPITATION_COLUMN = "weather_precipitation"


def summarize_regression_by(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    grouped = (
        frame.groupby(columns, observed=True)
        .agg(
            rows=("target_volume", "size"),
            actual_volume=("target_volume", "sum"),
            absolute_error=("absolute_error", "sum"),
            mae=("absolute_error", "mean"),
            mean_error=("error", "mean"),
            mean_squared_error=("squared_error", "mean"),
        )
        .reset_index()
    )
    grouped["wape"] = grouped["absolute_error"] / grouped["actual_volume"]
    grouped["rmse"] = np.sqrt(grouped["mean_squared_error"])
    return grouped.drop(
        columns=["actual_volume", "absolute_error", "mean_squared_error"]
    ).sort_values("mae", ascending=False, kind="stable")


def analyze_validation_errors(
    features: pd.DataFrame,
    model: VolumePredictor,
) -> tuple[dict[str, pd.DataFrame], dict[str, float | int]]:
    required = {
        "split",
        "is_model_ready",
        "target_timestamp",
        "intersection",
        "approach",
        "target_hour",
        "target_is_weekend",
        "target_volume",
        "target_high_volume",
        "high_traffic_threshold",
    }
    missing = sorted(required - set(features.columns))
    if missing:
        raise ErrorAnalysisError(f"오류 분석에 필요한 열이 없습니다: {missing}")
    validation = features.loc[
        features["split"].eq("validation") & features["is_model_ready"]
    ].copy()
    if validation.empty:
        raise ErrorAnalysisError("검증 구간에 분석 가능한 행이 없습니다.")

    prediction = np.asarray(model.predict(validation), dtype="float64").reshape(-1)
    if len(prediction) != len(validation) or not np.isfinite(prediction).all():
        raise ErrorAnalysisError("검증 예측값의 행 수 또는 값이 유효하지 않습니다.")
    alert_metrics, alert_details = evaluate_high_traffic_alerts(validation, prediction)

    analysis_columns = [
        "target_timestamp",
        "intersection",
        "approach",
        "target_hour",
        "target_is_weekend",
        "target_volume",
        "target_high_volume",
        "high_traffic_threshold",
    ]
    if PRECIPITATION_COLUMN in validation:
        analysis_columns.append(PRECIPITATION_COLUMN)
    analysis = validation[analysis_columns].copy()
    analysis["predicted_volume"] = prediction
    analysis["error"] = prediction - analysis["target_volume"]
    analysis["absolute_error"] = analysis["error"].abs()
    analysis["squared_error"] = analysis["error"].pow(2)
    analysis["predicted_high_volume"] = prediction > analysis[
        "high_traffic_threshold"
    ].to_numpy()
    analysis["day_type"] = np.where(
        analysis["target_is_weekend"], "주말", "평일"
    )
    if PRECIPITATION_COLUMN in analysis:
        analysis["rain_status"] = np.where(
            analysis[PRECIPITATION_COLUMN].gt(0), "강수", "비강수"
        )

    positive = analysis.loc[analysis["target_high_volume"].eq(1)].copy()
    intersection_alerts = (
        positive.groupby("intersection", observed=True)
        .agg(
            actual_high_volume=("target_high_volume", "size"),
            detected=("predicted_high_volume", "sum"),
        )
        .reset_index()
    )
    intersection_alerts["missed"] = (
        intersection_alerts["actual_high_volume"] - intersection_alerts["detected"]
    )
    intersection_alerts["recall"] = (
        intersection_alerts["detected"]
        / intersection_alerts["actual_high_volume"]
    )
    intersection_alerts = intersection_alerts.sort_values(
        ["recall", "missed"], ascending=[True, False], kind="stable"
    )

    false_negatives = analysis.loc[
        analysis["target_high_volume"].eq(1)
        & ~analysis["predicted_high_volume"]
    ].copy()
    false_negatives["underprediction"] = (
        false_negatives["target_volume"] - false_negatives["predicted_volume"]
    )
    false_negatives = false_negatives.sort_values(
        "underprediction", ascending=False, kind="stable"
    )
    largest_errors = analysis.sort_values(
        "absolute_error", ascending=False, kind="stable"
    )

    regression = regression_metrics(analysis["target_volume"], prediction)
    alert_row = alert_metrics.iloc[0]
    summary: dict[str, float | int] = {
        "rows": len(analysis),
        **regression,
        "precision": float(alert_row["precision"]),
        "recall": float(alert_row["recall"]),
        "f1": float(alert_row["f1"]),
        "pr_auc": float(alert_row["pr_auc"]),
        "false_negatives": int(alert_row["false_negative"]),
        "actual_positive_rate": float(alert_details["actual_positive_rate"]),
    }
    tables = {
        "intersection": summarize_regression_by(analysis, ["intersection"]),
        "approach": summarize_regression_by(
            analysis, ["intersection", "approach"]
        ),
        "hour": summarize_regression_by(analysis, ["target_hour"]),
        "day_type": summarize_regression_by(analysis, ["day_type"]),
        "intersection_alerts": intersection_alerts,
        "false_negatives": false_negatives,
        "largest_errors": largest_errors,
    }
    if "rain_status" in analysis:
        tables["rain_status"] = summarize_regression_by(
            analysis, ["rain_status"]
        )
        tables["rain_largest_errors"] = analysis.loc[
            analysis["rain_status"].eq("강수")
        ].sort_values("absolute_error", ascending=False, kind="stable")
    return tables, summary


def _regression_table(frame: pd.DataFrame, label_column: str) -> str:
    rows = []
    for row in frame.itertuples():
        label = getattr(row, label_column)
        rows.append(
            f"| {label} | {row.rows:,} | {row.mae:.2f} | "
            f"{row.wape * 100:.2f}% | {row.rmse:.2f} | {row.mean_error:.2f} |"
        )
    return "\n".join(rows)


def write_report(
    tables: dict[str, pd.DataFrame],
    summary: dict[str, float | int],
    report_path: Path,
    *,
    model_name: str = "hist_gradient_boosting",
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    intersection = tables["intersection"]
    hour = tables["hour"]
    worst_intersection = intersection.iloc[0]
    best_intersection = intersection.iloc[-1]
    worst_hour = hour.iloc[0]

    alert_rows = []
    for row in tables["intersection_alerts"].itertuples():
        alert_rows.append(
            f"| {row.intersection} | {row.actual_high_volume:,} | "
            f"{row.detected:,} | {row.missed:,} | {row.recall:.3f} |"
        )
    false_negative_rows = []
    for row in tables["false_negatives"].head(10).itertuples():
        false_negative_rows.append(
            f"| {row.target_timestamp} | {row.intersection} | {row.approach} | "
            f"{row.target_volume:.0f} | {row.predicted_volume:.1f} | "
            f"{row.high_traffic_threshold:.1f} | {row.underprediction:.1f} |"
        )

    rain_section = ""
    if "rain_status" in tables:
        rain_error_rows = []
        for row in tables["rain_largest_errors"].head(10).itertuples():
            rain_error_rows.append(
                f"| {row.target_timestamp} | {row.intersection} | {row.approach} | "
                f"{row.weather_precipitation:.1f} | {row.target_volume:.0f} | "
                f"{row.predicted_volume:.1f} | {row.absolute_error:.1f} |"
            )
        rain_section = f"""
## 강수 여부별 회귀 오차

강수량이 0mm보다 큰 예측 시각을 `강수`로 구분했다. 기상값은 천안 중심 좌표의 시간별 재분석 자료이므로 각 교차로의 현장 관측값은 아니다.

| 구분 | 행 | MAE | WAPE | RMSE | mean_error |
|---|---:|---:|---:|---:|---:|
{_regression_table(tables['rain_status'], 'rain_status')}

## 강수 시 큰 오차 사례

| 시각 | 교차로 | 접근로 | 강수량(mm) | 실제 | 예측 | 절대오차 |
|---|---|---|---:|---:|---:|---:|
{chr(10).join(rain_error_rows)}
"""

    model_label = MODEL_LABELS.get(model_name, model_name)
    report = f"""# {model_label} 오류 분석

## 분석 범위

검증 구간의 `is_model_ready=True`인 {summary['rows']:,}개 행만 분석했다. 시험 구간은 사용하지 않았다. 전체 MAE는 {summary['mae']:.2f}대, WAPE는 {summary['wape'] * 100:.2f}%, RMSE는 {summary['rmse']:.2f}대이다.

## 주요 결과

- 교차로별 MAE가 가장 큰 곳: {worst_intersection['intersection']} {worst_intersection['mae']:.2f}대
- 교차로별 MAE가 가장 작은 곳: {best_intersection['intersection']} {best_intersection['mae']:.2f}대
- 시간대별 MAE가 가장 큰 시각: {int(worst_hour['target_hour'])}시 {worst_hour['mae']:.2f}대
- 놓친 고교통량: {summary['false_negatives']:,}건

`mean_error`는 예측값에서 실제값을 뺀 평균이다. 음수이면 평균적으로 과소예측했다는 뜻이다.

## 교차로별 회귀 오차

| 교차로 | 행 | MAE | WAPE | RMSE | mean_error |
|---|---:|---:|---:|---:|---:|
{_regression_table(intersection, 'intersection')}

## 평일·주말 회귀 오차

| 구분 | 행 | MAE | WAPE | RMSE | mean_error |
|---|---:|---:|---:|---:|---:|
{_regression_table(tables['day_type'], 'day_type')}

{rain_section}

## 교차로별 고교통량 탐지

| 교차로 | 실제 고교통량 | 탐지 | 놓침 | Recall |
|---|---:|---:|---:|---:|
{chr(10).join(alert_rows)}

## 과소예측이 큰 놓친 고교통량 사례

| 시각 | 교차로 | 접근로 | 실제 | 예측 | 기준 | 과소예측 |
|---|---|---|---:|---:|---:|---:|
{chr(10).join(false_negative_rows)}

## 해석 주의

교차로마다 교통량 규모가 다르므로 MAE만으로 지점의 품질을 단정하지 않고 WAPE도 함께 본다. 이 결과는 검증 구간의 약점을 설명하기 위한 자료이며 시험 성능으로 표현하지 않는다.
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
        "--output-dir",
        type=Path,
        default=Path("data/processed/evaluation/error_analysis"),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("docs/error_analysis.md")
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
    tables, summary = analyze_validation_errors(features, model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(
            args.output_dir / f"{name}.csv", index=False, encoding="utf-8-sig"
        )
    write_report(tables, summary, args.report, model_name=args.model_name)
    print(f"rows: {summary['rows']:,}")
    print(f"MAE: {summary['mae']:.2f}")
    print(f"false negatives: {summary['false_negatives']:,}")
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
