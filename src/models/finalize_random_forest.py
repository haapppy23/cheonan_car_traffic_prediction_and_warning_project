"""선정된 Random Forest를 재학습하고 보관한 시험 구간을 한 번 평가한다."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import regression_metrics
from src.evaluation.run_high_traffic_alerts import evaluate_high_traffic_alerts
from src.models.random_forest import RandomForestTrafficModel
from src.models.ridge import MODEL_FEATURES


FINAL_MODEL_NAME = "random_forest"
SELECTION_DATE = "2026-11-20"
VALIDATION_METRICS = {
    "mae": 35.995214,
    "wape": 0.061471,
    "rmse": 77.732099,
    "precision": 0.831288,
    "recall": 0.720717,
    "f1": 0.772064,
    "pr_auc": 0.847369,
}


class FinalizationError(ValueError):
    """최종 학습·시험 분리가 안전하지 않을 때 발생한다."""


def select_final_partitions(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "split",
        "is_model_ready",
        "target_timestamp",
        "target_volume",
        "target_high_volume",
        "high_traffic_threshold",
        *MODEL_FEATURES,
    }
    missing = sorted(required - set(features.columns))
    if missing:
        raise FinalizationError(f"최종 모델에 필요한 열이 없습니다: {missing}")
    ready = features.loc[features["is_model_ready"]].copy()
    final_train = ready.loc[ready["split"].isin(["train", "validation"])].copy()
    test = ready.loc[ready["split"].eq("test")].copy()
    if final_train.empty or test.empty:
        raise FinalizationError("최종 학습 또는 시험 구간에 사용 가능한 행이 없습니다.")
    if final_train["target_timestamp"].max() >= test["target_timestamp"].min():
        raise FinalizationError("최종 학습 시각과 시험 시각이 겹칩니다.")
    return final_train, test


def train_and_test_final_model(
    features: pd.DataFrame,
) -> tuple[
    RandomForestTrafficModel,
    dict[str, float | int],
    pd.DataFrame,
]:
    """고정 Random Forest를 학습+검증으로 재학습하고 시험 구간을 평가한다."""
    final_train, test = select_final_partitions(features)
    model = RandomForestTrafficModel()
    started = time.perf_counter()
    model.fit(final_train, final_train["target_volume"])
    fit_seconds = time.perf_counter() - started
    prediction = model.predict(test)

    regression = regression_metrics(test["target_volume"], prediction)
    alerts, _ = evaluate_high_traffic_alerts(
        test,
        prediction,
        model_name=FINAL_MODEL_NAME,
        split_name="test",
    )
    alert = alerts.iloc[0]
    summary: dict[str, float | int] = {
        "train_rows": len(final_train),
        "test_rows": len(test),
        "train_end": final_train["target_timestamp"].max().isoformat(),
        "test_start": test["target_timestamp"].min().isoformat(),
        "test_end": test["target_timestamp"].max().isoformat(),
        "fit_seconds": fit_seconds,
        **regression,
        "precision": float(alert["precision"]),
        "recall": float(alert["recall"]),
        "f1": float(alert["f1"]),
        "pr_auc": float(alert["pr_auc"]),
        "true_positive": int(alert["true_positive"]),
        "false_positive": int(alert["false_positive"]),
        "false_negative": int(alert["false_negative"]),
        "true_negative": int(alert["true_negative"]),
    }
    return model, summary, alerts


def write_final_report(summary: dict[str, float | int], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = f"""# 최종 모델 선정과 시험 결과

## 선정 결정

최종 모델은 Random Forest로 고정했다. 시험 구간을 보기 전에 검증 구간에서 MAE·WAPE·RMSE와 Precision·Recall·F1·PR-AUC가 비교 후보 중 모두 가장 좋았기 때문이다. 모델 크기는 HistGradientBoosting보다 크지만 약 31.6MB로 FastAPI 서버에서 사용할 수 있는 범위라고 판단했다.

저장소에 재현 가능한 결과가 없는 XGBoost와 CatBoost는 이번 최종 비교에 포함하지 않았다. 이후 추가 실험을 하더라도 아래 시험 결과를 보고 설정을 조정하지 않는다.

## 최종 학습

- 사용 자료: 기존 학습+검증 구간
- 학습 행: {summary['train_rows']:,}개
- 학습 마지막 시각: {summary['train_end']}
- 시험 기간: {str(summary['test_start'])[:10]}~{str(summary['test_end'])[:10]}
- 시험 행: {summary['test_rows']:,}개
- 재학습 시간: {summary['fit_seconds']:.1f}초

## 보관 시험 구간 회귀 성능

| MAE | WAPE | RMSE |
|---:|---:|---:|
| {summary['mae']:.2f} | {summary['wape'] * 100:.2f}% | {summary['rmse']:.2f} |

## 보관 시험 구간 경보 성능

| Precision | Recall | F1 | PR-AUC |
|---:|---:|---:|---:|
| {summary['precision']:.3f} | {summary['recall']:.3f} | {summary['f1']:.3f} | {summary['pr_auc']:.3f} |

| TP | FP | FN | TN |
|---:|---:|---:|---:|
| {summary['true_positive']:,} | {summary['false_positive']:,} | {summary['false_negative']:,} | {summary['true_negative']:,} |

## 해석 원칙

이 결과는 2026년 6~8월 시험 구간을 한 번 평가한 최종 보고값이다. 검증 결과보다 나쁘거나 좋다는 이유로 모델 설정을 다시 선택하지 않는다.
"""
    report_path.write_text(report, encoding="utf-8")


def write_handoff(handoff_path: Path) -> None:
    handoff_path.parent.mkdir(parents=True, exist_ok=True)
    features = ", ".join(f"`{name}`" for name in MODEL_FEATURES)
    handoff = f"""# 최종 모델 전달 규격

## 모델

- 최종 모델: Random Forest
- 로컬 파일: `models/final_random_forest.joblib`
- 재생성 명령: `python src/models/finalize_random_forest.py`
- 모델 파일은 용량과 재생성 가능성 때문에 Git에 올리지 않는다.

## 입력

한 행은 예측하려는 교차로·접근로·시각이다. 모델 입력 열은 다음과 같다.

{features}

사용자가 이 숫자를 직접 입력하는 구조가 아니다. 백엔드는 저장된 최근 교통량에서 `src/features/build_features.py`와 같은 규칙으로 입력을 만든다. 168시간 이력이 없거나 센서 품질 플래그가 유효하지 않으면 예측 요청을 거부하거나 데이터 부족 상태를 반환한다.

## 출력

- `predicted_volume`: 1시간 후 예상 교통량, 0 이상의 실수
- `high_traffic_threshold`: 선택한 접근로의 학습 구간 90백분위수
- `high_traffic_alert`: `predicted_volume > high_traffic_threshold`
- `risk_score`: `predicted_volume / high_traffic_threshold`

접근로별 기준값은 `data/processed/features/high_traffic_thresholds.csv`에서 읽는다.

## API 담당자 주의사항

- 모델과 기준값을 서버 시작 시 한 번만 불러온다.
- 교차로·접근로 이름은 학습 데이터의 문자열을 그대로 사용한다.
- 시험 결과를 서비스 화면에서 실시간 정확도로 표현하지 않는다.
- 속도 자료가 없으므로 출력 명칭은 `정체 확정`이 아니라 `고교통량 위험`으로 사용한다.
"""
    handoff_path.write_text(handoff, encoding="utf-8")


def write_metadata(
    summary: dict[str, float | int],
    model_path: Path,
    metadata_path: Path,
) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_name": FINAL_MODEL_NAME,
        "selected_on": SELECTION_DATE,
        "selection_made_before_test": True,
        "selection_candidates": [
            "persistence",
            "daily_seasonal",
            "weekly_seasonal",
            "ridge",
            "hist_gradient_boosting",
            "random_forest",
        ],
        "not_included_without_reproducible_result": ["xgboost", "catboost"],
        "parameters": {
            "n_estimators": 12,
            "max_depth": 18,
            "min_samples_leaf": 10,
            "max_features": 0.7,
            "random_state": 42,
        },
        "validation_metrics": VALIDATION_METRICS,
        "test_metrics": {
            key: summary[key]
            for key in ("mae", "wape", "rmse", "precision", "recall", "f1", "pr_auc")
        },
        "training_rows": summary["train_rows"],
        "test_rows": summary["test_rows"],
        "artifact": str(model_path).replace("\\", "/"),
        "artifact_size_bytes": model_path.stat().st_size,
        "reproduce": "python src/models/finalize_random_forest.py",
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


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
        default=Path("models/final_random_forest.joblib"),
    )
    parser.add_argument(
        "--metadata", type=Path, default=Path("config/final_model.json")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("docs/final_test_results.md")
    )
    parser.add_argument(
        "--handoff", type=Path, default=Path("docs/model_handoff.md")
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    features = pd.read_parquet(args.input)
    model, summary, _ = train_and_test_final_model(features)

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_output)
    write_final_report(summary, args.report)
    write_handoff(args.handoff)
    write_metadata(summary, args.model_output, args.metadata)

    print(f"train rows: {summary['train_rows']:,}")
    print(f"test rows: {summary['test_rows']:,}")
    print(f"test MAE: {summary['mae']:.2f}")
    print(f"test WAPE: {summary['wape'] * 100:.2f}%")
    print(f"test RMSE: {summary['rmse']:.2f}")
    print(f"test F1: {summary['f1']:.3f}")
    print(f"model: {args.model_output}")
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
