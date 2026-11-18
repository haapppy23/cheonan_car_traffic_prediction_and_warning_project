"""학습 구간으로 Ridge를 학습하고 검증 구간에서 단순 기준과 비교한다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import regression_metrics
from src.evaluation.run_baselines import evaluate_baselines
from src.models.ridge import MODEL_FEATURES, RidgeTrafficModel


class RidgeTrainingError(ValueError):
    """공통 학습 데이터에 Ridge 학습 구간이 없을 때 발생한다."""


def select_ready_split(features: pd.DataFrame, split: str) -> pd.DataFrame:
    required = {"split", "is_model_ready", "target_volume", *MODEL_FEATURES}
    missing = sorted(required - set(features.columns))
    if missing:
        raise RidgeTrainingError(f"Ridge 학습에 필요한 열이 없습니다: {missing}")
    selected = features.loc[
        features["split"].eq(split) & features["is_model_ready"]
    ].copy()
    if selected.empty:
        raise RidgeTrainingError(f"{split} 구간에 사용 가능한 행이 없습니다.")
    return selected


def train_and_validate(
    features: pd.DataFrame,
    *,
    alpha: float = 1.0,
) -> tuple[RidgeTrafficModel, pd.DataFrame, dict[str, int | float]]:
    """시험 구간을 사용하지 않고 고정 Ridge와 단순 기준을 비교한다."""
    train = select_ready_split(features, "train")
    validation = select_ready_split(features, "validation")
    model = RidgeTrafficModel(alpha=alpha).fit(train, train["target_volume"])
    raw_prediction = model.predict_raw(validation)
    prediction = model.predict(validation)
    ridge_metrics = regression_metrics(validation["target_volume"], prediction)

    baselines = evaluate_baselines(features, split="validation")
    ridge_row = pd.DataFrame(
        [
            {
                "model": "ridge",
                "label": f"Ridge(alpha={alpha:g})",
                "source_feature": "공통 입력 변수",
                "split": "validation",
                "rows": len(validation),
                **ridge_metrics,
            }
        ]
    )
    comparison = pd.concat([ridge_row, baselines], ignore_index=True).sort_values(
        "mae", kind="stable"
    )
    details: dict[str, int | float] = {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "raw_negative_predictions": int((raw_prediction < 0).sum()),
        "alpha": alpha,
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
    ridge = comparison.loc[comparison["model"].eq("ridge")].iloc[0]
    baselines = comparison.loc[~comparison["model"].eq("ridge")]
    best_baseline_mae = baselines.loc[baselines["mae"].idxmin()]
    best_baseline_rmse = baselines.loc[baselines["rmse"].idxmin()]
    mae_change = (best_baseline_mae["mae"] - ridge["mae"]) / best_baseline_mae[
        "mae"
    ]
    rmse_change = (best_baseline_rmse["rmse"] - ridge["rmse"]) / best_baseline_rmse[
        "rmse"
    ]
    report = f"""# Ridge 회귀 검증 결과

## 학습 조건

- 학습 행: {details['train_rows']:,}개
- 검증 행: {details['validation_rows']:,}개
- 설정: `alpha={details['alpha']:g}`, `solver=lsqr`
- 입력: 시간 주기·월·주말, 과거 교통량, 이동 통계, 교차로와 접근로
- 원시 음수 예측: {details['raw_negative_predictions']:,}개, 최종 출력에서는 0으로 보정
- 시험 구간: 사용하지 않음

## 검증 결과

| 모델 | 검증 행 | MAE | WAPE | RMSE |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## 비교

Ridge의 MAE는 {ridge['mae']:.2f}대로 가장 좋은 단순 기준인 {best_baseline_mae['label']}보다 {mae_change * 100:.1f}% {'개선' if mae_change >= 0 else '악화'}되었다. RMSE는 {ridge['rmse']:.2f}대로 가장 좋은 단순 기준인 {best_baseline_rmse['label']}보다 {rmse_change * 100:.1f}% {'개선' if rmse_change >= 0 else '악화'}되었다.

이번 값은 Ridge의 고정 기준선이며 하이퍼파라미터 탐색 결과가 아니다. 다음 후보 모델도 동일한 학습·검증 행과 지표로 비교한다.
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
        "--model-output", type=Path, default=Path("models/ridge_baseline.joblib")
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("data/processed/evaluation/ridge_comparison.csv"),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("docs/ridge_results.md")
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    features = pd.read_parquet(args.input)
    model, comparison, details = train_and_validate(features, alpha=args.alpha)

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
