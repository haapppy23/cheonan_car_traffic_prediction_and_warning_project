"""XGBoost와 CatBoost를 시간순 Randomized Search로 비교한다."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import regression_metrics
from src.evaluation.run_high_traffic_alerts import evaluate_high_traffic_alerts
from src.models.boosting_candidates import (
    CatBoostTrafficModel,
    XGBoostTrafficModel,
)
from src.models.train_ridge import select_ready_split


TUNING_START_DEFAULT = pd.Timestamp("2025-10-01")
MODEL_LABELS = {"xgboost": "XGBoost", "catboost": "CatBoost"}
PARAMETER_SPACES: dict[str, dict[str, list[Any]]] = {
    "xgboost": {
        "n_estimators": [160, 240, 320],
        "max_depth": [5, 7, 9],
        "learning_rate": [0.03, 0.05, 0.08],
        "min_child_weight": [5.0, 10.0, 20.0],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
    },
    "catboost": {
        "iterations": [180, 260, 340],
        "depth": [6, 8, 10],
        "learning_rate": [0.03, 0.05, 0.08],
        "l2_leaf_reg": [3.0, 5.0, 10.0],
        "random_strength": [0.0, 0.5, 1.0],
    },
}


class ExtendedTrainingError(ValueError):
    """추가 후보 학습 조건이 시간순 검증 규칙을 만족하지 않을 때 발생한다."""


def chronological_tuning_partitions(
    features: pd.DataFrame,
    *,
    tuning_start: pd.Timestamp = TUNING_START_DEFAULT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """기존 학습 구간 끝부분을 내부 튜닝 검증으로 분리한다."""
    if "target_timestamp" not in features:
        raise ExtendedTrainingError("시간순 탐색에 target_timestamp 열이 필요합니다.")
    full_train = select_ready_split(features, "train").copy()
    full_train["target_timestamp"] = pd.to_datetime(
        full_train["target_timestamp"], errors="coerce"
    )
    if full_train["target_timestamp"].isna().any():
        raise ExtendedTrainingError("학습 구간에 해석할 수 없는 시각이 있습니다.")
    tuning_start = pd.Timestamp(tuning_start)
    tuning_train = full_train.loc[
        full_train["target_timestamp"].lt(tuning_start)
    ].copy()
    tuning_validation = full_train.loc[
        full_train["target_timestamp"].ge(tuning_start)
    ].copy()
    if tuning_train.empty or tuning_validation.empty:
        raise ExtendedTrainingError("내부 학습 또는 내부 검증 구간이 비어 있습니다.")
    if (
        tuning_train["target_timestamp"].max()
        >= tuning_validation["target_timestamp"].min()
    ):
        raise ExtendedTrainingError("내부 학습과 검증의 시간 순서가 잘못되었습니다.")
    return full_train, tuning_train, tuning_validation


def _python_value(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _make_model(model_name: str, params: dict[str, Any]):
    normalized = {key: _python_value(value) for key, value in params.items()}
    if model_name == "xgboost":
        return XGBoostTrafficModel(**normalized)
    if model_name == "catboost":
        return CatBoostTrafficModel(**normalized)
    raise ExtendedTrainingError(f"지원하지 않는 후보 모델입니다: {model_name}")


def search_and_validate_candidate(
    features: pd.DataFrame,
    *,
    model_name: str,
    n_iter: int = 4,
    random_state: int = 42,
    tuning_start: pd.Timestamp = TUNING_START_DEFAULT,
) -> tuple[Any, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """내부 시간순 탐색 후 전체 학습 구간으로 재학습하고 검증한다."""
    if model_name not in PARAMETER_SPACES:
        raise ExtendedTrainingError(f"지원하지 않는 후보 모델입니다: {model_name}")
    if n_iter <= 0:
        raise ExtendedTrainingError("Randomized Search 반복 횟수는 1 이상이어야 합니다.")

    full_train, tuning_train, tuning_validation = chronological_tuning_partitions(
        features, tuning_start=tuning_start
    )
    validation = select_ready_split(features, "validation")
    sampled = list(
        ParameterSampler(
            PARAMETER_SPACES[model_name],
            n_iter=n_iter,
            random_state=random_state,
        )
    )
    search_rows: list[dict[str, Any]] = []
    for candidate_id, params in enumerate(sampled, start=1):
        started = time.perf_counter()
        model = _make_model(model_name, params).fit(
            tuning_train, tuning_train["target_volume"]
        )
        prediction = model.predict(tuning_validation)
        metrics = regression_metrics(tuning_validation["target_volume"], prediction)
        search_rows.append(
            {
                "model": model_name,
                "candidate": candidate_id,
                "split": "internal_validation",
                "train_rows": len(tuning_train),
                "validation_rows": len(tuning_validation),
                "seconds": time.perf_counter() - started,
                "parameters": json.dumps(
                    {k: _python_value(v) for k, v in params.items()},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                **metrics,
            }
        )
    search_results = pd.DataFrame(search_rows).sort_values(
        ["mae", "rmse"], kind="stable"
    ).reset_index(drop=True)
    search_results.insert(2, "rank", np.arange(1, len(search_results) + 1))
    best_params = json.loads(search_results.iloc[0]["parameters"])

    started = time.perf_counter()
    best_model = _make_model(model_name, best_params).fit(
        full_train, full_train["target_volume"]
    )
    raw_prediction = best_model.predict_raw(validation)
    prediction = np.maximum(raw_prediction, 0.0)
    metrics = regression_metrics(validation["target_volume"], prediction)
    regression_row = pd.DataFrame(
        [
            {
                "model": model_name,
                "label": MODEL_LABELS[model_name],
                "source_feature": "공통 입력 변수",
                "split": "validation",
                "rows": len(validation),
                **metrics,
            }
        ]
    )
    alert_row, _ = evaluate_high_traffic_alerts(
        validation, prediction, model_name=model_name
    )
    details = {
        "model": model_name,
        "n_iter": n_iter,
        "tuning_start": pd.Timestamp(tuning_start).isoformat(),
        "internal_train_rows": len(tuning_train),
        "internal_validation_rows": len(tuning_validation),
        "full_train_rows": len(full_train),
        "validation_rows": len(validation),
        "best_parameters": best_params,
        "refit_seconds": time.perf_counter() - started,
        "raw_negative_predictions": int((raw_prediction < 0).sum()),
    }
    return best_model, regression_row, alert_row, search_results, details


def _load_existing_rows(path: Path, excluded: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"기존 비교 결과가 없습니다: {path}. 기존 모델 평가 명령을 먼저 실행하십시오."
        )
    frame = pd.read_csv(path, encoding="utf-8-sig")
    return frame.loc[~frame["model"].isin(excluded)].copy()


def write_report(
    comparison: pd.DataFrame,
    alert_comparison: pd.DataFrame,
    search_results: pd.DataFrame,
    details: list[dict[str, Any]],
    report_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    regression_rows = "\n".join(
        f"| {row.label} | {row.mae:.2f} | {row.wape * 100:.2f}% | {row.rmse:.2f} |"
        for row in comparison.itertuples()
    )
    alert_labels = {**MODEL_LABELS, "random_forest": "Random Forest", "hist_gradient_boosting": "HistGradientBoosting"}
    alert_rows = "\n".join(
        f"| {alert_labels.get(str(row.model), row.model)} | {row.precision:.3f} | "
        f"{row.recall:.3f} | {row.f1:.3f} | {row.pr_auc:.3f} |"
        for row in alert_comparison.itertuples()
    )
    search_rows = "\n".join(
        f"| {MODEL_LABELS[str(row.model)]} | {int(row.rank)} | {row.mae:.2f} | "
        f"{row.wape * 100:.2f}% | {row.rmse:.2f} | {row.seconds:.1f}초 |"
        for row in search_results.itertuples()
    )
    settings = "\n".join(
        f"- {MODEL_LABELS[item['model']]}: `{json.dumps(item['best_parameters'], ensure_ascii=False, sort_keys=True)}`"
        for item in details
    )
    report = f"""# 추가 부스팅 모델과 시간순 Randomized Search 결과

## 탐색 규칙

- 내부 학습: 기존 학습 구간 중 2025-10-01 이전
- 내부 검증: 기존 학습 구간 중 2025-10-01~2025-12-31
- 공식 검증: 2026-01-01~2026-05-31
- 보관 시험: 사용하지 않음
- 모델별 무작위 설정 후보: {details[0]['n_iter']}개

Randomized Search는 미래 행을 과거 학습에 섞지 않았다. 내부 검증 MAE가 가장 낮은 설정을 고른 뒤 기존 전체 학습 구간으로 다시 학습하고 공식 검증 구간에서 한 번 비교했다.

## 내부 탐색 결과

| 모델 | 내부 순위 | MAE | WAPE | RMSE | 학습·예측 시간 |
|---|---:|---:|---:|---:|---:|
{search_rows}

## 선택된 설정

{settings}

## 공식 검증 회귀 비교

| 모델 | MAE | WAPE | RMSE |
|---|---:|---:|---:|
{regression_rows}

## 공식 검증 고교통량 경보 비교

| 모델 | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|
{alert_rows}

## 해석 제한

추가 실험은 기존 보관 시험 결과를 본 뒤 수행했으므로 시험 성능을 다시 열어 최종 모델을 바꾸지 않는다. 이 결과는 XGBoost와 CatBoost를 같은 시간순 검증 구간에서 보충 비교한 값으로만 사용한다.
"""
    report_path.write_text(report, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/features/model_features.parquet"),
    )
    parser.add_argument("--n-iter", type=int, default=4)
    parser.add_argument("--tuning-start", default="2025-10-01")
    parser.add_argument(
        "--existing-comparison",
        type=Path,
        default=Path("data/processed/evaluation/random_forest_comparison.csv"),
    )
    parser.add_argument(
        "--existing-alerts",
        type=Path,
        nargs="*",
        default=[
            Path("data/processed/evaluation/random_forest_alert_metrics.csv"),
            Path("data/processed/evaluation/high_traffic_alert_metrics.csv"),
        ],
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=Path("data/processed/evaluation/extended_model_comparison.csv"),
    )
    parser.add_argument(
        "--alert-output",
        type=Path,
        default=Path("data/processed/evaluation/extended_alert_comparison.csv"),
    )
    parser.add_argument(
        "--search-output",
        type=Path,
        default=Path("data/processed/evaluation/randomized_search_results.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/extended_model_comparison.md"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    features = pd.read_parquet(args.input)
    model_rows = []
    alert_rows = []
    search_frames = []
    details = []
    for model_name in ("xgboost", "catboost"):
        model, regression, alert, search, detail = search_and_validate_candidate(
            features,
            model_name=model_name,
            n_iter=args.n_iter,
            tuning_start=pd.Timestamp(args.tuning_start),
        )
        output = Path("models") / f"{model_name}.joblib"
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output)
        model_rows.append(regression)
        alert_rows.append(alert)
        search_frames.append(search)
        details.append(detail)
        print(f"{model_name}: {regression.iloc[0].to_dict()}")

    existing = _load_existing_rows(
        args.existing_comparison, {"xgboost", "catboost"}
    )
    comparison = pd.concat([*model_rows, existing], ignore_index=True).sort_values(
        ["mae", "rmse"], kind="stable"
    )
    existing_alert_frames = [
        pd.read_csv(path, encoding="utf-8-sig")
        for path in args.existing_alerts
        if path.exists()
    ]
    alerts = pd.concat(
        [*alert_rows, *existing_alert_frames], ignore_index=True
    ).drop_duplicates("model", keep="first")
    alerts = alerts.sort_values(["f1", "pr_auc"], ascending=False, kind="stable")
    search_results = pd.concat(search_frames, ignore_index=True)

    for path, frame in (
        (args.comparison_output, comparison),
        (args.alert_output, alerts),
        (args.search_output, search_results),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    write_report(comparison, alerts, search_results, details, args.report)
    print(comparison.to_string(index=False))
    print(alerts.to_string(index=False))
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
