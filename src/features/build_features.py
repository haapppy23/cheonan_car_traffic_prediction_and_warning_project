"""시간별 교통량에서 누출 없는 모델 입력 변수와 시간순 분할을 만든다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.schema import write_table


GROUP_COLUMNS = ["intersection", "approach"]
TRAIN_END_DEFAULT = pd.Timestamp("2026-01-01")
VALIDATION_END_DEFAULT = pd.Timestamp("2026-06-01")
LAG_HOURS = (1, 2, 3, 24, 168)
ROLLING_HOURS = tuple(range(1, 7))


class FeatureValidationError(ValueError):
    """입력 자료나 시간 분할이 안전한 모델링 조건을 만족하지 않을 때 발생한다."""


def load_hourly_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["timestamp"])
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    return validate_hourly_data(frame)


def validate_hourly_data(frame: pd.DataFrame) -> pd.DataFrame:
    """필수 열과 접근로별 한 시간 연속성을 확인하고 시간순으로 정렬한다."""
    required = {"timestamp", "intersection", "approach", "volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise FeatureValidationError(f"입력 변수 생성에 필요한 열이 없습니다: {missing}")

    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    result["volume"] = pd.to_numeric(result["volume"], errors="coerce")
    if result[list(GROUP_COLUMNS)].isna().any().any():
        raise FeatureValidationError("교차로 또는 접근로 이름에 결측값이 있습니다.")
    if result["timestamp"].isna().any():
        raise FeatureValidationError("해석할 수 없는 시각이 있습니다.")
    if result["volume"].isna().any():
        raise FeatureValidationError("교통량 결측값이 있습니다.")
    if result["volume"].lt(0).any():
        raise FeatureValidationError("음수 교통량이 있습니다.")

    duplicate_count = int(
        result.duplicated(["timestamp", *GROUP_COLUMNS], keep=False).sum()
    )
    if duplicate_count:
        raise FeatureValidationError(
            f"시간·교차로·접근로 중복 행이 {duplicate_count}개 있습니다."
        )

    result = result.sort_values(
        [*GROUP_COLUMNS, "timestamp"], kind="stable"
    ).reset_index(drop=True)
    gaps = result.groupby(GROUP_COLUMNS, sort=False, observed=True)["timestamp"].diff()
    invalid_gaps = gaps.notna() & gaps.ne(pd.Timedelta(hours=1))
    if invalid_gaps.any():
        examples = result.loc[
            invalid_gaps, ["timestamp", *GROUP_COLUMNS]
        ].head(5)
        raise FeatureValidationError(
            "접근로별 시각이 한 시간 간격으로 연속되지 않습니다: "
            + examples.to_dict(orient="records").__str__()
        )
    return result


def mark_all_zero_target_days(frame: pd.DataFrame) -> pd.Series:
    """같은 접근로의 하루 24시간이 모두 0인 예측 대상을 표시한다."""
    date_key = frame["timestamp"].dt.normalize()
    grouped = frame.groupby(
        [date_key, frame["intersection"], frame["approach"]],
        sort=False,
        observed=True,
    )["volume"]
    day_max = grouped.transform("max")
    day_size = grouped.transform("size")
    return day_size.eq(24) & day_max.eq(0)


def assign_time_split(
    target_timestamp: pd.Series,
    *,
    train_end: pd.Timestamp = TRAIN_END_DEFAULT,
    validation_end: pd.Timestamp = VALIDATION_END_DEFAULT,
) -> pd.Series:
    """예측 대상 시각을 기준으로 학습·검증·시험 구간을 배정한다."""
    train_end = pd.Timestamp(train_end)
    validation_end = pd.Timestamp(validation_end)
    if train_end >= validation_end:
        raise FeatureValidationError("학습 종료일은 검증 종료일보다 빨라야 합니다.")
    values = np.select(
        [target_timestamp.lt(train_end), target_timestamp.lt(validation_end)],
        ["train", "validation"],
        default="test",
    )
    return pd.Series(values, index=target_timestamp.index, dtype="string")


def build_feature_table(
    frame: pd.DataFrame,
    *,
    train_end: pd.Timestamp = TRAIN_END_DEFAULT,
    validation_end: pd.Timestamp = VALIDATION_END_DEFAULT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """과거 교통량·시간 특성·학습 전용 경보 기준을 생성한다."""
    source = validate_hourly_data(frame)
    train_end = pd.Timestamp(train_end)
    validation_end = pd.Timestamp(validation_end)

    result = pd.DataFrame(
        {
            "target_timestamp": source["timestamp"],
            "intersection": source["intersection"].astype("string"),
            "approach": source["approach"].astype("string"),
            "target_volume": source["volume"],
        }
    )
    result["target_hour"] = source["timestamp"].dt.hour.astype("int8")
    result["target_day_of_week"] = source["timestamp"].dt.dayofweek.astype("int8")
    result["target_month"] = source["timestamp"].dt.month.astype("int8")
    result["target_is_weekend"] = result["target_day_of_week"].ge(5)
    if "is_holiday" in source:
        result["target_is_holiday"] = source["is_holiday"].astype("boolean")
    else:
        result["target_is_holiday"] = pd.Series(
            pd.NA, index=result.index, dtype="boolean"
        )

    result["hour_sin"] = np.sin(2 * np.pi * result["target_hour"] / 24).astype(
        "float32"
    )
    result["hour_cos"] = np.cos(2 * np.pi * result["target_hour"] / 24).astype(
        "float32"
    )
    result["weekday_sin"] = np.sin(
        2 * np.pi * result["target_day_of_week"] / 7
    ).astype("float32")
    result["weekday_cos"] = np.cos(
        2 * np.pi * result["target_day_of_week"] / 7
    ).astype("float32")

    grouped_volume = source.groupby(
        GROUP_COLUMNS, sort=False, observed=True
    )["volume"]
    short_lags: dict[int, pd.Series] = {}
    for lag in ROLLING_HOURS:
        shifted = grouped_volume.shift(lag).astype("float32")
        short_lags[lag] = shifted
        if lag in LAG_HOURS:
            result[f"lag_{lag}"] = shifted
    for lag in (24, 168):
        result[f"lag_{lag}"] = grouped_volume.shift(lag).astype("float32")

    lag_frame = pd.DataFrame(short_lags)
    result["rolling_mean_3"] = lag_frame[[1, 2, 3]].mean(
        axis=1, skipna=False
    ).astype("float32")
    result["rolling_mean_6"] = lag_frame.mean(axis=1, skipna=False).astype("float32")
    result["rolling_std_6"] = lag_frame.std(
        axis=1, ddof=0, skipna=False
    ).astype("float32")

    result["target_all_zero_approach_day"] = mark_all_zero_target_days(source)
    result["split"] = assign_time_split(
        result["target_timestamp"],
        train_end=train_end,
        validation_end=validation_end,
    )

    threshold_source = result.loc[
        result["target_timestamp"].lt(train_end)
        & ~result["target_all_zero_approach_day"]
    ]
    thresholds = (
        threshold_source.groupby(GROUP_COLUMNS, observed=True)["target_volume"]
        .quantile(0.90)
        .rename("high_traffic_threshold")
        .reset_index()
    )
    if thresholds.empty:
        raise FeatureValidationError("학습 구간에 경보 기준을 계산할 자료가 없습니다.")
    result = result.merge(
        thresholds, on=GROUP_COLUMNS, how="left", validate="many_to_one"
    )

    labels = pd.Series(pd.NA, index=result.index, dtype="Int8")
    valid_target = (
        ~result["target_all_zero_approach_day"]
        & result["high_traffic_threshold"].notna()
    )
    labels.loc[valid_target] = (
        result.loc[valid_target, "target_volume"]
        > result.loc[valid_target, "high_traffic_threshold"]
    ).astype("int8")
    result["target_high_volume"] = labels

    required_history = [
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_24",
        "lag_168",
        "rolling_mean_3",
        "rolling_mean_6",
        "rolling_std_6",
    ]
    result["history_available"] = result[required_history].notna().all(axis=1)
    result["is_model_ready"] = (
        result["history_available"]
        & ~result["target_all_zero_approach_day"]
        & result["high_traffic_threshold"].notna()
    )

    column_order = [
        "target_timestamp",
        "intersection",
        "approach",
        "split",
        "is_model_ready",
        "target_all_zero_approach_day",
        "history_available",
        "target_volume",
        "target_high_volume",
        "high_traffic_threshold",
        "target_hour",
        "target_day_of_week",
        "target_month",
        "target_is_weekend",
        "target_is_holiday",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_24",
        "lag_168",
        "rolling_mean_3",
        "rolling_mean_6",
        "rolling_std_6",
    ]
    return result[column_order], thresholds


def summarize_features(
    features: pd.DataFrame,
    thresholds: pd.DataFrame,
    *,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> dict[str, object]:
    split_summary: dict[str, dict[str, int | float]] = {}
    for split in ("train", "validation", "test"):
        part = features.loc[features["split"].eq(split)]
        ready = part.loc[part["is_model_ready"]]
        high_count = int(ready["target_high_volume"].sum())
        split_summary[split] = {
            "total_rows": int(len(part)),
            "model_ready_rows": int(len(ready)),
            "high_volume_rows": high_count,
            "high_volume_rate": high_count / len(ready) if len(ready) else 0.0,
        }
    return {
        "rows": int(len(features)),
        "model_ready_rows": int(features["is_model_ready"].sum()),
        "quality_excluded_rows": int(features["target_all_zero_approach_day"].sum()),
        "history_unavailable_rows": int((~features["history_available"]).sum()),
        "train_end_exclusive": pd.Timestamp(train_end).isoformat(),
        "validation_end_exclusive": pd.Timestamp(validation_end).isoformat(),
        "threshold_min": float(thresholds["high_traffic_threshold"].min()),
        "threshold_median": float(thresholds["high_traffic_threshold"].median()),
        "threshold_max": float(thresholds["high_traffic_threshold"].max()),
        "split_summary": split_summary,
        "columns": list(features.columns),
    }


def write_feature_report(summary: dict[str, object], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    labels = {"train": "학습", "validation": "검증", "test": "시험"}
    for split in ("train", "validation", "test"):
        item = summary["split_summary"][split]
        rows.append(
            f"| {labels[split]} | {item['total_rows']:,} | "
            f"{item['model_ready_rows']:,} | {item['high_volume_rows']:,} | "
            f"{item['high_volume_rate'] * 100:.1f}% |"
        )
    split_table = "\n".join(rows)
    report = f"""# 모델 입력 변수 생성 결과

## 생성 기준

한 행의 `target_timestamp`는 예측하려는 시각이다. `target_volume`은 그 시각의 실제 교통량이고, `lag_1`은 바로 한 시간 전까지 관측된 교통량이다. 따라서 현재 행의 정답이나 미래 교통량은 입력 변수에 포함되지 않는다.

- 전체 행: {summary['rows']:,}개
- 모델 사용 가능 행: {summary['model_ready_rows']:,}개
- 24시간 전체가 0이어서 제외한 예측 대상: {summary['quality_excluded_rows']:,}개
- 168시간 이력이 완성되지 않은 행: {summary['history_unavailable_rows']:,}개
- 접근로별 학습 구간 90백분위수: {summary['threshold_min']:.1f}~{summary['threshold_max']:.1f}대, 중앙값 {summary['threshold_median']:.1f}대

## 시간순 분할 결과

분할은 예측 대상 시각을 기준으로 한다. 학습 종료 경계는 {str(summary['train_end_exclusive'])[:10]}, 검증 종료 경계는 {str(summary['validation_end_exclusive'])[:10]}이며 각 경계일은 다음 구간에 포함된다.

| 구간 | 전체 행 | 모델 사용 가능 | 고교통량 | 고교통량 비율 |
|---|---:|---:|---:|---:|
{split_table}

## 생성한 입력 변수

- 시간: `target_hour`, `target_day_of_week`, `target_month`, `target_is_weekend`
- 주기 표현: `hour_sin`, `hour_cos`, `weekday_sin`, `weekday_cos`
- 과거 교통량: `lag_1`, `lag_2`, `lag_3`, `lag_24`, `lag_168`
- 최근 흐름: `rolling_mean_3`, `rolling_mean_6`, `rolling_std_6`
- 범주: `intersection`, `approach`
- 공휴일: 외부 자료를 결합하기 전까지 `target_is_holiday`는 결측 상태로 유지

## 누출 방지 규칙

- 모든 지연 변수와 이동 통계는 예측 대상 시각보다 과거인 값만 사용한다.
- 고교통량 기준은 2024~2025년 학습 구간의 유효 자료만 사용해 접근로별로 계산한다.
- 24시간 전체가 0인 접근로·날짜의 정답은 센서 중단 가능성이 있어 모델 행에서 제외한다.
- 첫 168시간처럼 필요한 과거 이력이 부족한 행은 임의로 채우지 않는다.
- 학습·검증·시험 구간은 무작위로 섞지 않는다.

## 출력 파일

- `data/processed/features/model_features.parquet`: 모델 입력 변수와 정답
- `data/processed/features/high_traffic_thresholds.csv`: 접근로별 학습 구간 90백분위수
- `data/processed/features/feature_metadata.json`: 행 수·분할·열 메타데이터
"""
    report_path.write_text(report, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/selected_traffic_long.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/features/model_features.parquet"),
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path("data/processed/features/high_traffic_thresholds.csv"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/processed/features/feature_metadata.json"),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("docs/feature_dataset.md")
    )
    parser.add_argument("--train-end", default="2026-01-01")
    parser.add_argument("--validation-end", default="2026-06-01")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    train_end = pd.Timestamp(args.train_end)
    validation_end = pd.Timestamp(args.validation_end)
    source = load_hourly_data(args.input)
    features, thresholds = build_feature_table(
        source, train_end=train_end, validation_end=validation_end
    )
    summary = summarize_features(
        features,
        thresholds,
        train_end=train_end,
        validation_end=validation_end,
    )

    write_table(features, args.output)
    write_table(thresholds, args.thresholds)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_feature_report(summary, args.report)

    print(f"rows: {summary['rows']:,}")
    print(f"model-ready rows: {summary['model_ready_rows']:,}")
    print(f"quality-excluded rows: {summary['quality_excluded_rows']:,}")
    for split, item in summary["split_summary"].items():
        print(f"{split}: {item['model_ready_rows']:,}")
    print(f"output: {args.output}")
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
