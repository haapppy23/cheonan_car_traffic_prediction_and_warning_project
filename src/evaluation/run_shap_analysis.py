"""최종 Random Forest가 검증 예측에 사용한 입력 변수의 SHAP 중요도를 계산한다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.random_forest import (
    FEATURE_LABELS,
    RandomForestTrafficModel,
    clean_feature_names,
)
from src.models.ridge import MODEL_FEATURES


class ShapAnalysisError(ValueError):
    """SHAP 분석 입력이나 저장 모델이 유효하지 않을 때 발생한다."""


def select_validation_sample(
    features: pd.DataFrame,
    *,
    sample_size: int = 2_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """모델 사용 가능 검증 행만 고정 난수로 표본 추출한다."""
    if sample_size <= 0:
        raise ShapAnalysisError("SHAP 표본 크기는 1 이상이어야 합니다.")
    required = {"split", "is_model_ready", *MODEL_FEATURES}
    missing = sorted(required - set(features.columns))
    if missing:
        raise ShapAnalysisError(f"SHAP 분석에 필요한 열이 없습니다: {missing}")
    validation = features.loc[
        features["split"].eq("validation") & features["is_model_ready"]
    ].copy()
    if validation.empty:
        raise ShapAnalysisError("SHAP 분석에 사용할 검증 행이 없습니다.")
    if len(validation) > sample_size:
        validation = validation.sample(n=sample_size, random_state=random_state)
    if "target_timestamp" in validation.columns:
        validation = validation.sort_values("target_timestamp", kind="stable")
    return validation.reset_index(drop=True)


def summarize_shap_values(
    values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """행별 SHAP 절댓값을 평균내어 전역 중요도 표를 만든다."""
    array = np.asarray(values, dtype="float64")
    if array.ndim != 2 or array.shape[1] != len(feature_names):
        raise ShapAnalysisError("SHAP 값과 입력 변수 이름의 크기가 맞지 않습니다.")
    mean_absolute = np.mean(np.abs(array), axis=0)
    total = float(mean_absolute.sum())
    share = mean_absolute / total if total > 0 else np.zeros_like(mean_absolute)
    result = pd.DataFrame(
        {
            "feature": feature_names,
            "feature_label": [FEATURE_LABELS.get(name, name) for name in feature_names],
            "mean_abs_shap": mean_absolute,
            "importance_share": share,
        }
    )
    return result.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def analyze_random_forest_shap(
    features: pd.DataFrame,
    model: RandomForestTrafficModel,
    *,
    sample_size: int = 2_000,
) -> tuple[pd.DataFrame, int]:
    """검증 표본을 전처리하고 Tree SHAP 전역 중요도를 반환한다."""
    sample = select_validation_sample(features, sample_size=sample_size)
    pipeline = getattr(model, "pipeline", None)
    if pipeline is None:
        raise ShapAnalysisError("저장 모델에 Random Forest 파이프라인이 없습니다.")
    try:
        preprocessing = pipeline.named_steps["preprocessing"]
        regressor = pipeline.named_steps["random_forest"]
    except KeyError as exc:
        raise ShapAnalysisError("저장 모델의 파이프라인 구성이 예상과 다릅니다.") from exc

    transformed = preprocessing.transform(sample[MODEL_FEATURES])
    feature_names = clean_feature_names(preprocessing.get_feature_names_out())
    explainer = shap.TreeExplainer(regressor)
    explanation = explainer(transformed, check_additivity=False)
    importance = summarize_shap_values(explanation.values, feature_names)
    return importance, len(sample)


def write_importance_plot(importance: pd.DataFrame, output_path: Path) -> None:
    """상위 12개 변수의 평균 절대 SHAP 막대그래프를 저장한다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top = importance.head(12).sort_values("mean_abs_shap", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#5367B1")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("")
    ax.set_title("Random Forest feature importance on validation sample")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(
    importance: pd.DataFrame,
    *,
    sample_rows: int,
    report_path: Path,
    figure_path: Path,
) -> None:
    """발표와 보고서에서 바로 쓸 수 있는 해석 문서를 저장한다."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, item in importance.head(12).iterrows():
        rows.append(
            f"| {rank + 1} | {item['feature_label']} | `{item['feature']}` | "
            f"{item['mean_abs_shap']:.2f} | {item['importance_share'] * 100:.1f}% |"
        )
    figure_link = Path("figures") / figure_path.name
    report = f"""# Random Forest SHAP 입력 변수 분석

## 분석 조건

- 대상 모델: 최종 Random Forest
- 사용 구간: 검증 구간만 사용
- 분석 표본: {sample_rows:,}행, 고정 난수 42
- 보관 시험 구간: 사용하지 않음

## 주요 영향 변수

| 순위 | 설명 | 입력 열 | 평균 절대 SHAP | 비중 |
|---:|---|---|---:|---:|
{chr(10).join(rows)}

![SHAP 입력 변수 중요도]({figure_link.as_posix()})

## 읽는 방법

평균 절대 SHAP이 클수록 해당 변수가 여러 검증 행의 예측값을 크게 움직였다는 뜻이다. 예를 들어 `lag_1`이 상위라면 직전 시간 교통량이 1시간 후 교통량 예측에 많이 사용됐다는 의미다.

이 값은 모델 내부의 예측 기여도이지 원인과 결과를 증명하지 않는다. 또한 절댓값 평균이므로 교통량을 올렸는지 내렸는지는 이 표만으로 판단하지 않는다.
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
        default=Path("models/final_random_forest.joblib"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/evaluation/shap_feature_importance.csv"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("docs/figures/shap_feature_importance.png"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/shap_feature_importance.md"),
    )
    parser.add_argument("--sample-size", type=int, default=2_000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    features = pd.read_parquet(args.input)
    model = joblib.load(args.model)
    importance, sample_rows = analyze_random_forest_shap(
        features, model, sample_size=args.sample_size
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(args.output, index=False, encoding="utf-8-sig")
    write_importance_plot(importance, args.figure)
    write_report(
        importance,
        sample_rows=sample_rows,
        report_path=args.report,
        figure_path=args.figure,
    )
    print(importance.head(12).to_string(index=False))
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
