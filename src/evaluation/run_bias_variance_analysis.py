"""학습량과 트리 깊이에 따른 학습·검증 오차로 Bias–Variance를 점검한다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import regression_metrics
from src.models.random_forest import RandomForestTrafficModel
from src.models.train_ridge import select_ready_split


class BiasVarianceAnalysisError(ValueError):
    """Bias–Variance 분석 조건이나 입력이 올바르지 않을 때 발생한다."""


def evaluate_bias_variance(
    features: pd.DataFrame,
    *,
    train_sizes: tuple[int, ...] = (20_000, 60_000, 120_000),
    depths: tuple[int, ...] = (6, 12, 18),
    validation_rows: int = 30_000,
    n_estimators: int = 8,
    random_state: int = 42,
) -> pd.DataFrame:
    """시험 구간 없이 중첩 학습 표본의 학습·검증 MAE를 비교한다."""
    if not train_sizes or any(size <= 0 for size in train_sizes):
        raise BiasVarianceAnalysisError("학습 표본 크기는 1 이상이어야 합니다.")
    if not depths or any(depth <= 0 for depth in depths):
        raise BiasVarianceAnalysisError("트리 깊이는 1 이상이어야 합니다.")
    if validation_rows <= 0 or n_estimators <= 0:
        raise BiasVarianceAnalysisError(
            "검증 표본 크기와 트리 수는 1 이상이어야 합니다."
        )

    train = select_ready_split(features, "train").reset_index(drop=True)
    validation = select_ready_split(features, "validation").reset_index(drop=True)
    usable_sizes = tuple(sorted({size for size in train_sizes if size <= len(train)}))
    if not usable_sizes:
        raise BiasVarianceAnalysisError(
            f"학습 행 {len(train):,}개보다 작은 표본 크기가 없습니다."
        )

    rng = np.random.default_rng(random_state)
    train_order = rng.permutation(len(train))
    if len(validation) > validation_rows:
        validation = validation.sample(
            n=validation_rows, random_state=random_state
        ).reset_index(drop=True)

    results: list[dict[str, int | float]] = []
    for depth in sorted(set(depths)):
        for size in usable_sizes:
            sample = train.iloc[train_order[:size]].copy()
            model = RandomForestTrafficModel(
                n_estimators=n_estimators,
                max_depth=depth,
                min_samples_leaf=10,
            ).fit(sample, sample["target_volume"])
            train_prediction = model.predict(sample)
            validation_prediction = model.predict(validation)
            train_metrics = regression_metrics(
                sample["target_volume"], train_prediction
            )
            validation_metrics = regression_metrics(
                validation["target_volume"], validation_prediction
            )
            results.append(
                {
                    "max_depth": depth,
                    "train_rows": size,
                    "validation_rows": len(validation),
                    "train_mae": train_metrics["mae"],
                    "validation_mae": validation_metrics["mae"],
                    "mae_gap": validation_metrics["mae"] - train_metrics["mae"],
                    "validation_wape": validation_metrics["wape"],
                    "validation_rmse": validation_metrics["rmse"],
                }
            )
    return pd.DataFrame(results).sort_values(
        ["max_depth", "train_rows"], kind="stable"
    ).reset_index(drop=True)


def write_plot(results: pd.DataFrame, output_path: Path) -> None:
    """트리 깊이별 학습량과 학습·검증 MAE 관계를 저장한다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True)
    for depth, group in results.groupby("max_depth", sort=True):
        label = f"depth={depth}"
        axes[0].plot(group["train_rows"], group["train_mae"], marker="o", label=label)
        axes[1].plot(
            group["train_rows"], group["validation_mae"], marker="o", label=label
        )
    axes[0].set_title("Training MAE")
    axes[1].set_title("Validation MAE")
    for ax in axes:
        ax.set_xlabel("Training rows")
        ax.set_ylabel("MAE")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Bias–Variance diagnostic without the held-out test split")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(
    results: pd.DataFrame,
    *,
    report_path: Path,
    figure_path: Path,
) -> None:
    """Bias–Variance 비교표와 해석 원칙을 문서로 저장한다."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in results.itertuples(index=False):
        rows.append(
            f"| {row.max_depth} | {row.train_rows:,} | {row.validation_rows:,} | "
            f"{row.train_mae:.2f} | {row.validation_mae:.2f} | {row.mae_gap:.2f} | "
            f"{row.validation_wape * 100:.2f}% | {row.validation_rmse:.2f} |"
        )
    largest = results.loc[
        results["train_rows"].eq(results["train_rows"].max())
    ].sort_values("validation_mae", kind="stable")
    best = largest.iloc[0]
    figure_link = Path("figures") / figure_path.name
    report = f"""# Random Forest Bias–Variance 점검

## 분석 원칙

- 학습 구간과 공식 검증 구간만 사용
- 보관 시험 구간은 사용하지 않음
- 학습 표본은 고정 난수 42의 중첩 표본으로 구성
- 같은 검증 표본에서 학습량과 `max_depth`에 따른 오차 비교
- 이 분석은 설명용 진단이며 이미 확정한 최종 모델을 다시 선택하지 않음

## 결과

| max_depth | 학습 행 | 검증 행 | 학습 MAE | 검증 MAE | MAE 차이 | 검증 WAPE | 검증 RMSE |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

![Bias–Variance 학습·검증 오차]({figure_link.as_posix()})

## 해석

학습 MAE와 검증 MAE가 함께 높으면 모델이 패턴을 충분히 학습하지 못한 높은 Bias 상태로 본다. 학습 MAE는 낮지만 검증 MAE와의 차이가 커지면 학습 표본에 지나치게 맞춘 높은 Variance 가능성을 의심한다.

시간순 학습·검증 구간의 교통량 분포와 난이도가 다르므로 일부 설정에서는 검증 MAE가 학습 MAE보다 낮을 수 있다. 따라서 오차 차이의 부호 하나만으로 과적합을 판단하지 않고 학습량 증가에 따른 검증 오차 추세를 함께 본다.

가장 큰 학습 표본에서 검증 MAE가 가장 낮은 진단 설정은 `max_depth={int(best['max_depth'])}`이며 검증 MAE는 {best['validation_mae']:.2f}대, 학습·검증 MAE 차이는 {best['mae_gap']:.2f}대다. 이 값은 진단용 표본 결과이므로 최종 모델의 공식 검증·시험 성능을 대체하지 않는다.
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
        default=Path("data/processed/evaluation/bias_variance_metrics.csv"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("docs/figures/bias_variance_analysis.png"),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("docs/bias_variance_analysis.md")
    )
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[20_000, 60_000, 120_000])
    parser.add_argument("--depths", type=int, nargs="+", default=[6, 12, 18])
    parser.add_argument("--validation-rows", type=int, default=30_000)
    parser.add_argument("--n-estimators", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    features = pd.read_parquet(args.input)
    results = evaluate_bias_variance(
        features,
        train_sizes=tuple(args.train_sizes),
        depths=tuple(args.depths),
        validation_rows=args.validation_rows,
        n_estimators=args.n_estimators,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False, encoding="utf-8-sig")
    write_plot(results, args.figure)
    write_report(results, report_path=args.report, figure_path=args.figure)
    print(results.to_string(index=False))
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
