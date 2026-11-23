"""공휴일·기상 변수를 추가한 Random Forest를 시간순 검증 구간에서 비교한다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import regression_metrics
from src.evaluation.run_high_traffic_alerts import evaluate_high_traffic_alerts
from src.models.external_random_forest import ExternalRandomForestModel
from src.models.random_forest import RandomForestTrafficModel
from src.models.train_ridge import select_ready_split


def run_external_feature_experiment(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, ExternalRandomForestModel]:
    """시험 구간을 제외하고 기본·외부 변수 모델을 같은 설정으로 비교한다."""
    train = select_ready_split(features, "train")
    validation = select_ready_split(features, "validation")
    models = {
        "random_forest_base": RandomForestTrafficModel(),
        "random_forest_external": ExternalRandomForestModel(),
    }
    regression_rows: list[dict[str, object]] = []
    alert_tables: list[pd.DataFrame] = []
    for name, model in models.items():
        model.fit(train, train["target_volume"])
        prediction = model.predict(validation)
        metrics = regression_metrics(validation["target_volume"], prediction)
        regression_rows.append(
            {
                "model": name,
                "split": "validation",
                "rows": len(validation),
                **metrics,
            }
        )
        alerts, _ = evaluate_high_traffic_alerts(
            validation,
            prediction,
            model_name=name,
            split_name="validation",
        )
        alert_tables.append(alerts)
    return (
        pd.DataFrame(regression_rows).sort_values("mae").reset_index(drop=True),
        pd.concat(alert_tables, ignore_index=True).sort_values(
            "f1", ascending=False
        ).reset_index(drop=True),
        models["random_forest_external"],
    )


def write_report(
    regression: pd.DataFrame,
    alerts: pd.DataFrame,
    report_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    labels = {
        "random_forest_base": "기존 입력 Random Forest",
        "random_forest_external": "공휴일·기상 추가 Random Forest",
    }
    regression_rows = []
    for _, row in regression.iterrows():
        regression_rows.append(
            f"| {labels[row['model']]} | {row['mae']:.2f} | "
            f"{row['wape'] * 100:.2f}% | {row['rmse']:.2f} |"
        )
    alert_rows = []
    for _, row in alerts.iterrows():
        alert_rows.append(
            f"| {labels[row['model']]} | {row['precision']:.3f} | "
            f"{row['recall']:.3f} | {row['f1']:.3f} | {row['pr_auc']:.3f} |"
        )
    base = regression.set_index("model").loc["random_forest_base"]
    external = regression.set_index("model").loc["random_forest_external"]
    mae_change = float(external["mae"] - base["mae"])
    decision = "개선" if mae_change < 0 else "악화"
    report = f"""# 공휴일·기상 변수 보충 실험

## 실험 원칙

- 학습: 2024-01-01~2025-12-31
- 검증: 2026-01-01~2026-05-31
- 보관 시험: 사용하지 않음
- Random Forest 설정은 동일하게 고정하고 공휴일·기상 변수 추가 여부만 비교

## 회귀 성능

| 모델 | MAE | WAPE | RMSE |
|---|---:|---:|---:|
{chr(10).join(regression_rows)}

## 고교통량 경보 성능

| 모델 | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|
{chr(10).join(alert_rows)}

## 결론

외부 변수 추가 모델의 검증 MAE는 기존 입력 모델보다 {abs(mae_change):.2f}만큼 {decision}했다. 이 결과만으로 이미 보관 시험을 평가한 최종 모델을 다시 선택하지 않는다.

기상값은 천안 각 교차로의 현장 관측이 아닌 격자 재분석 자료다. 실제 서비스에서는 1시간 후 대상 시각에 사용할 수 있는 예보값으로 교체해야 한다.
"""
    report_path.write_text(report, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/features/model_features_external.parquet"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("data/processed/evaluation/external_feature_comparison.csv"),
    )
    parser.add_argument(
        "--alerts",
        type=Path,
        default=Path("data/processed/evaluation/external_feature_alerts.csv"),
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/random_forest_external.joblib"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/external_feature_experiment.md"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    features = pd.read_parquet(args.input)
    regression, alerts, model = run_external_feature_experiment(features)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    regression.to_csv(args.metrics, index=False, encoding="utf-8-sig")
    alerts.to_csv(args.alerts, index=False, encoding="utf-8-sig")
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_output)
    write_report(regression, alerts, args.report)
    print(regression.to_string(index=False))
    print(alerts.to_string(index=False))
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
