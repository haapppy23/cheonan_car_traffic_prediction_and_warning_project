"""검증 구간에서 지속성·일간·주간 단순 기준을 비교한다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import regression_metrics
from src.models.baselines import BASELINE_LABELS, BASELINES, get_baseline_predictions


class BaselineEvaluationError(ValueError):
    """평가 구간이나 공통 데이터가 안전하지 않을 때 발생한다."""


def evaluate_baselines(
    features: pd.DataFrame,
    *,
    split: str = "validation",
    allow_test: bool = False,
) -> pd.DataFrame:
    """같은 모델 사용 가능 행에서 세 단순 기준의 회귀 오차를 계산한다."""
    if split not in {"train", "validation", "test"}:
        raise BaselineEvaluationError(f"지원하지 않는 평가 구간입니다: {split}")
    if split == "test" and not allow_test:
        raise BaselineEvaluationError(
            "시험 구간은 최종 모델 확정 전에는 평가하지 않습니다."
        )
    required = {"split", "is_model_ready", "target_volume", *BASELINES.values()}
    missing = sorted(required - set(features.columns))
    if missing:
        raise BaselineEvaluationError(f"평가에 필요한 열이 없습니다: {missing}")

    evaluation = features.loc[
        features["split"].eq(split) & features["is_model_ready"]
    ].copy()
    if evaluation.empty:
        raise BaselineEvaluationError(f"{split} 구간에 평가 가능한 행이 없습니다.")

    results: list[dict[str, object]] = []
    for model, prediction in get_baseline_predictions(evaluation).items():
        metrics = regression_metrics(evaluation["target_volume"], prediction)
        results.append(
            {
                "model": model,
                "label": BASELINE_LABELS[model],
                "source_feature": BASELINES[model],
                "split": split,
                "rows": len(evaluation),
                **metrics,
            }
        )
    return pd.DataFrame(results).sort_values("mae", kind="stable").reset_index(drop=True)


def write_report(metrics: pd.DataFrame, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    table_rows = []
    for row in metrics.itertuples():
        table_rows.append(
            f"| {row.label} | `{row.source_feature}` | {row.rows:,} | "
            f"{row.mae:.2f} | {row.wape * 100:.2f}% | {row.rmse:.2f} |"
        )
    best_mae = metrics.sort_values("mae", kind="stable").iloc[0]
    best_rmse = metrics.sort_values("rmse", kind="stable").iloc[0]
    report = f"""# 단순 기준 모델 검증 결과

## 평가 원칙

2026년 1월부터 5월까지의 검증 구간만 사용했다. 세 기준 모두 `is_model_ready=True`인 같은 행에서 비교했으며 2026년 6월부터 8월까지의 시험 구간은 확인하지 않았다.

| 기준 | 사용값 | 검증 행 | MAE | WAPE | RMSE |
|---|---|---:|---:|---:|---:|
{chr(10).join(table_rows)}

## 해석

가장 낮은 MAE와 WAPE는 {best_mae['label']} 기준의 {best_mae['mae']:.2f}대와 {best_mae['wape'] * 100:.2f}%이다. 가장 낮은 RMSE는 {best_rmse['label']} 기준의 {best_rmse['rmse']:.2f}대이다. 평균적인 오차에서는 전주 패턴이 유리하지만 큰 오차 억제에서는 최근 한 시간 값이 유리하므로, 이후 모델도 세 지표를 함께 비교한다.

## 시험 구간 보호

실행 코드는 기본적으로 `split=test` 평가를 거부한다. 최종 모델과 설정이 확정된 뒤에만 명시적으로 시험 평가를 허용한다.
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
        "--output",
        type=Path,
        default=Path("data/processed/evaluation/baseline_metrics.csv"),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("docs/baseline_results.md")
    )
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="validation"
    )
    parser.add_argument("--allow-test", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    features = pd.read_parquet(args.input)
    metrics = evaluate_baselines(
        features, split=args.split, allow_test=args.allow_test
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output, index=False, encoding="utf-8-sig")
    write_report(metrics, args.report)
    print(metrics.to_string(index=False))
    print(f"output: {args.output}")
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
