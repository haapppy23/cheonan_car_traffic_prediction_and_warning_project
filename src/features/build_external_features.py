"""기존 모델 입력 자료에 대한민국 공휴일과 천안 시간별 기상을 결합한다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import holidays
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


WEATHER_COLUMNS = [
    "weather_temperature_2m",
    "weather_precipitation",
    "weather_wind_speed_10m",
]


class ExternalFeatureError(ValueError):
    """공휴일·기상 결합 자료가 안전한 모델 입력 조건을 만족하지 않을 때 발생한다."""


def korean_holiday_flags(timestamps: pd.Series) -> pd.Series:
    """시각의 날짜가 대한민국 법정공휴일인지 반환한다."""
    parsed = pd.to_datetime(timestamps, errors="coerce")
    if parsed.isna().any():
        raise ExternalFeatureError("공휴일 변환 중 해석할 수 없는 시각이 있습니다.")
    years = sorted(parsed.dt.year.unique().tolist())
    calendar = holidays.country_holidays("KR", years=years)
    return parsed.dt.date.map(lambda day: day in calendar).astype("bool")


def combine_external_features(
    features: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """시간별 모델 행에 공휴일과 기상을 many-to-one으로 결합한다."""
    if "target_timestamp" not in features:
        raise ExternalFeatureError("모델 자료에 target_timestamp 열이 없습니다.")
    required_weather = {"timestamp", *WEATHER_COLUMNS}
    missing = sorted(required_weather - set(weather.columns))
    if missing:
        raise ExternalFeatureError(f"기상 자료에 필요한 열이 없습니다: {missing}")
    result = features.copy()
    result["target_timestamp"] = pd.to_datetime(
        result["target_timestamp"], errors="coerce"
    )
    weather_table = weather[["timestamp", *WEATHER_COLUMNS]].copy()
    weather_table["timestamp"] = pd.to_datetime(
        weather_table["timestamp"], errors="coerce"
    )
    if result["target_timestamp"].isna().any() or weather_table["timestamp"].isna().any():
        raise ExternalFeatureError("외부 변수 결합 중 해석할 수 없는 시각이 있습니다.")
    if weather_table["timestamp"].duplicated().any():
        raise ExternalFeatureError("기상 자료에 중복 시각이 있습니다.")
    result = result.drop(columns=[*WEATHER_COLUMNS], errors="ignore")
    result = result.merge(
        weather_table,
        left_on="target_timestamp",
        right_on="timestamp",
        how="left",
        validate="many_to_one",
    ).drop(columns="timestamp")
    result["target_is_holiday"] = korean_holiday_flags(result["target_timestamp"])
    missing_weather = result[WEATHER_COLUMNS].isna().any(axis=1)
    if missing_weather.any():
        examples = result.loc[missing_weather, "target_timestamp"].head(5).tolist()
        raise ExternalFeatureError(
            f"교통량 시각과 일치하는 기상값이 {int(missing_weather.sum()):,}행 없습니다: {examples}"
        )
    return result


def write_external_report(features: pd.DataFrame, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    holiday_rows = int(features["target_is_holiday"].sum())
    unique_holiday_hours = int(
        features.loc[features["target_is_holiday"], "target_timestamp"].nunique()
    )
    report = f"""# 공휴일·기상 외부 입력 변수

## 결합 결과

- 전체 모델 행: {len(features):,}개
- 공휴일 표시 행: {holiday_rows:,}개
- 공휴일 시간: {unique_holiday_hours:,}시간
- 기상 결측 행: {int(features[WEATHER_COLUMNS].isna().any(axis=1).sum()):,}개

추가 변수는 `target_is_holiday`, `weather_temperature_2m`, `weather_precipitation`, `weather_wind_speed_10m`이다. 대한민국 공휴일은 `holidays` 패키지의 KR 달력으로 만들었고, 기상은 천안 중심 좌표의 Open-Meteo 시간별 재분석 자료를 사용했다.

## 해석 제한

Open-Meteo 값은 천안 각 교차로의 현장 관측값이 아니라 격자 재분석 값이다. 실제 1시간 후 서비스에서는 대상 시각의 관측값 대신 당시 이용 가능한 초단기 예보값을 사용해야 한다. 이번 결과는 외부 변수가 검증 성능을 개선하는지 확인하는 보충 실험으로만 사용한다.
"""
    report_path.write_text(report, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/features/model_features.parquet"),
    )
    parser.add_argument(
        "--weather",
        type=Path,
        default=Path("data/external/cheonan_weather_hourly.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/features/model_features_external.parquet"),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("docs/external_features.md")
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    features = pd.read_parquet(args.features)
    weather = pd.read_csv(args.weather, encoding="utf-8-sig")
    result = combine_external_features(features, weather)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    write_external_report(result, args.report)
    print(f"rows: {len(result):,}")
    print(f"holiday rows: {int(result['target_is_holiday'].sum()):,}")
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
