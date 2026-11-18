"""HistGradientBoosting을 학습하고 검증 구간에서 Ridge·단순 기준과 비교한다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import regression_metrics
from src.models.hist_gradient_boosting import HistGradientBoostingTrafficModel
from src.models.train_ridge import select_ready_split, train_and_validate


def train_and_validate_hist_gradient_boosting(
    features: pd.DataFrame,
) -> tuple[
    HistGradientBoostingTrafficModel,
    pd.DataFrame,
    dict[str, int | float],
]:
    """시험 구간을 사용하지 않고 고정 부스팅·Ridge·단순 기준을 비교한다."""
    train = select_ready_split(features, "train")
    validation = select_ready_split(features, "validation")
    model = HistGradientBoostingTrafficModel().fit(train, train["target_volume"])
    raw_prediction = model.predict_raw(validation)
    prediction = model.predict(validation)
    metrics = regression_metrics(validation["target_volume"], prediction)

    _, ridge_comparison, _ = train_and_validate(features)
    boosting_row = pd.DataFrame(
        [
            {
                "model": "hist_gradient_boosting",
                "label": "HistGradientBoosting",
                "source_feature": "공통 입력 변수",
                "split": "validation",
                "rows": len(validation),
                **metrics,
            }
        ]
    )
    comparison = pd.concat(
        [boosting_row, ridge_comparison], ignore_index=True
    ).sort_values("mae", kind="stable")
    details: dict[str, int | float] = {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "raw_negative_predictions": int((raw_prediction < 0).sum()),
        "max_iter": model.max_iter,
        "learning_rate": model.learning_rate,
        "max_leaf_nodes": model.max_leaf_nodes,
    }
    return model, comparison.reset_index(drop=True), details


def write_report(
    comparison: pd.DataFrame,
    details: dict[str, int | float],
    report_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in comparison.itertuples():
        rows.append(
            f"| {row.label} | {row.rows:,} | {row.mae:.2f} | "
            f"{row.wape * 100:.2f}% | {row.rmse:.2f} |"
        )
    boosting = comparison.loc[
        comparison["model"].eq("hist_gradient_boosting")
    ].iloc[0]
    ridge = comparison.loc[comparison["model"].eq("ridge")].iloc[0]
    mae_change = (ridge["mae"] - boosting["mae"]) / ridge["mae"]
    rmse_change = (ridge["rmse"] - boosting["rmse"]) / ridge["rmse"]
    report = f"""# HistGradientBoosting 검증 결과

## 학습 조건

- 학습 행: {details['train_rows']:,}개
- 검증 행: {details['validation_rows']:,}개
- 설정: `max_iter={details['max_iter']}`, `learning_rate={details['learning_rate']}`, `max_leaf_nodes={details['max_leaf_nodes']}`
- 내부 무작위 조기 종료: 사용하지 않음
- 원시 음수 예측: {details['raw_negative_predictions']:,}개, 최종 출력에서는 0으로 보정
- 시험 구간: 사용하지 않음

## 검증 결과

| 모델 | 검증 행 | MAE | WAPE | RMSE |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## 비교

HistGradientBoosting의 MAE는 {boosting['mae']:.2f}대로 Ridge보다 {mae_change * 100:.1f}% {'개선' if mae_change >= 0 else '악화'}되었다. RMSE는 {boosting['rmse']:.2f}대로 Ridge보다 {rmse_change * 100:.1f}% {'개선' if rmse_change >= 0 else '악화'}되었다.

이번 결과는 고정 설정의 첫 트리 기반 기준선이다. 설정 탐색이나 시험 구간 평가는 아직 수행하지 않았다.
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
        "--model-output",
        type=Path,
        default=Path("models/hist_gradient_boosting.joblib"),
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("data/processed/evaluation/hist_gradient_boosting_comparison.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/hist_gradient_boosting_results.md"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    features = pd.read_parquet(args.input)
    model, comparison, details = train_and_validate_hist_gradient_boosting(features)

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_output)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.metrics_output, index=False, encoding="utf-8-sig")
    write_report(comparison, details, args.report)

    print(comparison.to_string(index=False))
    print(f"model: {args.model_output}")
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
