"""천안시 스마트교차로 원본 파일의 구조와 기본 품질을 검사한다."""

from __future__ import annotations

import argparse
import sys
from itertools import groupby
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.schema import DataSchemaError, read_table, resolve_wide_schema


def _longest_zero_run(values: list[object]) -> int:
    flags = [pd.notna(value) and value == 0 for value in values]
    return max(
        (sum(1 for _ in group) for is_zero, group in groupby(flags) if is_zero),
        default=0,
    )


def inspect_dataframe(
    frame: pd.DataFrame,
    *,
    date_column: str | None = None,
    intersection_column: str | None = None,
    approach_column: str | None = None,
    total_column: str | None = None,
) -> dict[str, Any]:
    schema = resolve_wide_schema(
        frame,
        date_column=date_column,
        intersection_column=intersection_column,
        approach_column=approach_column,
        total_column=total_column,
    )
    hour_columns = [schema.hours[hour] for hour in range(24)]
    raw_hours = frame[hour_columns]
    numeric_hours = raw_hours.apply(pd.to_numeric, errors="coerce")
    invalid_numeric = raw_hours.notna() & numeric_hours.isna()

    key_columns = [schema.date, schema.intersection, schema.approach]
    duplicate_key_rows = int(frame.duplicated(subset=key_columns, keep=False).sum())
    zero_counts = (numeric_hours == 0).sum(axis=1)
    longest_runs = [
        _longest_zero_run(row.tolist()) for _, row in numeric_hours.iterrows()
    ]

    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "column_names": list(frame.columns),
        "schema": schema,
        "missing_by_column": {
            column: int(count) for column, count in frame.isna().sum().items()
        },
        "exact_duplicate_rows": int(frame.duplicated(keep=False).sum()),
        "duplicate_key_rows": duplicate_key_rows,
        "invalid_numeric_cells": int(invalid_numeric.sum().sum()),
        "negative_volume_cells": int((numeric_hours < 0).sum().sum()),
        "zero_volume_cells": int((numeric_hours == 0).sum().sum()),
        "all_zero_hour_rows": int((zero_counts == 24).sum()),
        "max_consecutive_zero_hours": int(max(longest_runs, default=0)),
        "intersection_count": int(frame[schema.intersection].nunique(dropna=True)),
        "approach_count": int(frame[schema.approach].nunique(dropna=True)),
    }


def print_report(report: dict[str, Any]) -> None:
    schema = report["schema"]
    print("[파일 구조]")
    print(f"행 수: {report['rows']}")
    print(f"열 수: {report['columns']}")
    print("열 이름: " + ", ".join(report["column_names"]))
    print(f"일자 열: {schema.date}")
    print(f"교차로 열: {schema.intersection}")
    print(f"접근로 열: {schema.approach}")
    print(f"합계 열: {schema.total or '없음(선택 열)'}")
    print(f"시간 열 수: {len(schema.hours)}")

    print("\n[고유값]")
    print(f"교차로 수: {report['intersection_count']}")
    print(f"접근로 수: {report['approach_count']}")

    print("\n[중복]")
    print(f"완전히 동일한 중복 행 수: {report['exact_duplicate_rows']}")
    print(f"일자·교차로·접근로 중복 행 수: {report['duplicate_key_rows']}")

    print("\n[교통량 품질]")
    print(f"숫자로 변환할 수 없는 셀 수: {report['invalid_numeric_cells']}")
    print(f"음수 교통량 셀 수: {report['negative_volume_cells']}")
    print(f"0 교통량 셀 수: {report['zero_volume_cells']}")
    print(f"24시간 전체가 0인 행 수: {report['all_zero_hour_rows']}")
    print(f"한 행에서 연속된 0의 최대 길이: {report['max_consecutive_zero_hours']}시간")

    print("\n[열별 결측값]")
    for column, count in report["missing_by_column"].items():
        print(f"{column}: {count}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="검사할 CSV 또는 XLSX 파일")
    parser.add_argument("--date-column", help="일자 열 이름을 명시적으로 지정")
    parser.add_argument("--intersection-column", help="교차로 열 이름을 명시적으로 지정")
    parser.add_argument("--approach-column", help="접근로 열 이름을 명시적으로 지정")
    parser.add_argument("--total-column", help="합계 열 이름을 명시적으로 지정")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        frame = read_table(args.input)
        report = inspect_dataframe(
            frame,
            date_column=args.date_column,
            intersection_column=args.intersection_column,
            approach_column=args.approach_column,
            total_column=args.total_column,
        )
        print_report(report)
    except (DataSchemaError, FileNotFoundError) as error:
        print(f"검사 실패: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
