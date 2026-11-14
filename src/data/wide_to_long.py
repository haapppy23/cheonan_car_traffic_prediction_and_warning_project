"""하루 24시간 열의 교통량 원본을 시간당 한 행으로 변환한다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.schema import (
    DataSchemaError,
    read_table,
    resolve_wide_schema,
    write_table,
)


class DataValidationError(ValueError):
    """내용 중복이나 형식 오류 때문에 안전하게 변환할 수 없을 때 발생한다."""


def wide_to_long(
    frame: pd.DataFrame,
    *,
    date_column: str | None = None,
    intersection_column: str | None = None,
    approach_column: str | None = None,
    total_column: str | None = None,
) -> pd.DataFrame:
    source = frame.copy(deep=True)
    schema = resolve_wide_schema(
        source,
        date_column=date_column,
        intersection_column=intersection_column,
        approach_column=approach_column,
        total_column=total_column,
    )
    key_columns = [schema.date, schema.intersection, schema.approach]
    duplicate_mask = source.duplicated(subset=key_columns, keep=False)
    if duplicate_mask.any():
        raise DataValidationError(
            "일자·교차로·접근로 조합이 중복되어 timestamp 키를 고유하게 만들 수 없습니다. "
            f"중복 행 수: {int(duplicate_mask.sum())}"
        )

    hour_columns = [schema.hours[hour] for hour in range(24)]
    hour_by_column = {column: hour for hour, column in schema.hours.items()}
    long_frame = source.melt(
        id_vars=key_columns,
        value_vars=hour_columns,
        var_name="_source_hour",
        value_name="volume",
    )
    long_frame["hour"] = long_frame["_source_hour"].map(hour_by_column).astype("int8")

    parsed_date = pd.to_datetime(long_frame[schema.date], errors="coerce")
    invalid_date_count = int(parsed_date.isna().sum())
    if invalid_date_count:
        raise DataValidationError(
            f"날짜로 해석할 수 없는 값이 {invalid_date_count}개 있습니다: {schema.date}"
        )

    raw_volume = long_frame["volume"]
    numeric_volume = pd.to_numeric(raw_volume, errors="coerce")
    invalid_volume = raw_volume.notna() & numeric_volume.isna()
    if invalid_volume.any():
        examples = raw_volume[invalid_volume].astype(str).drop_duplicates().head(5).tolist()
        raise DataValidationError(
            "숫자로 변환할 수 없는 교통량 값이 있습니다. 예시: " + ", ".join(examples)
        )

    timestamp = parsed_date.dt.normalize() + pd.to_timedelta(long_frame["hour"], unit="h")
    result = pd.DataFrame(
        {
            "timestamp": timestamp,
            "intersection": long_frame[schema.intersection].astype("string"),
            "approach": long_frame[schema.approach].astype("string"),
            "volume": numeric_volume,
            "date": timestamp.dt.date,
            "hour": long_frame["hour"],
            "day_of_week": timestamp.dt.dayofweek.astype("int8"),
            "is_weekend": timestamp.dt.dayofweek.ge(5),
            "is_holiday": pd.Series(pd.NA, index=long_frame.index, dtype="boolean"),
        }
    )
    result = result.sort_values(
        ["intersection", "approach", "timestamp"], kind="stable"
    ).reset_index(drop=True)

    expected_rows = len(source) * 24
    if len(result) != expected_rows:
        raise DataValidationError(
            f"변환 행 수가 예상과 다릅니다. 예상 {expected_rows}, 실제 {len(result)}"
        )

    long_duplicate_mask = result.duplicated(
        subset=["timestamp", "intersection", "approach"], keep=False
    )
    if long_duplicate_mask.any():
        raise DataValidationError(
            "변환 결과에 timestamp·intersection·approach 중복이 있습니다. "
            f"중복 행 수: {int(long_duplicate_mask.sum())}"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="입력 CSV 또는 XLSX 파일")
    parser.add_argument("--output", required=True, help="출력 CSV 또는 Parquet 파일")
    parser.add_argument("--date-column", help="일자 열 이름을 명시적으로 지정")
    parser.add_argument("--intersection-column", help="교차로 열 이름을 명시적으로 지정")
    parser.add_argument("--approach-column", help="접근로 열 이름을 명시적으로 지정")
    parser.add_argument("--total-column", help="합계 열 이름을 명시적으로 지정")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        source = read_table(args.input)
        result = wide_to_long(
            source,
            date_column=args.date_column,
            intersection_column=args.intersection_column,
            approach_column=args.approach_column,
            total_column=args.total_column,
        )
        write_table(result, args.output)
    except (DataSchemaError, DataValidationError, FileNotFoundError) as error:
        print(f"변환 실패: {error}", file=sys.stderr)
        return 2

    print(f"변환 전 행 수: {len(source)}")
    print(f"변환 후 행 수: {len(result)}")
    print("결과 열: " + ", ".join(result.columns))
    print(f"저장 위치: {Path(args.output).resolve()}")
    print("주의: is_holiday는 외부 공휴일 정보를 결합하기 전이므로 결측값입니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
