"""월별 스마트교차로 JSON을 병합하고 관측 연속성 보고서를 만든다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


class MonthlyDataError(ValueError):
    """월별 원본을 안전하게 병합할 수 없을 때 발생한다."""


BASE_FIELDS = {
    "일자": "STAT_DT",
    "교차로명": "LCTN_NM",
    "접근로명": "ACRD_NM",
    "합계": "TOTAL_TFVL",
}
HOUR_FIELDS = {f"{hour:02d}시": f"IXR_STAT_DT_{hour:02d}" for hour in range(24)}
CANONICAL_FIELDS = {**BASE_FIELDS, **HOUR_FIELDS}
KEY_COLUMNS = ["일자", "교차로명", "접근로명"]


def canonicalize_rows(rows: list[dict[str, object]], source: str) -> pd.DataFrame:
    if not rows:
        raise MonthlyDataError(f"행이 없는 원본입니다: {source}")

    required = set(CANONICAL_FIELDS.values())
    missing = sorted(required - set(rows[0]))
    if missing:
        raise MonthlyDataError(f"필수 필드가 없습니다: {missing} ({source})")

    frame = pd.DataFrame(
        {column: [row.get(field) for row in rows] for column, field in CANONICAL_FIELDS.items()}
    )
    parsed_date = pd.to_datetime(frame["일자"], errors="coerce")
    if parsed_date.isna().any():
        raise MonthlyDataError(
            f"날짜로 해석할 수 없는 행이 {int(parsed_date.isna().sum())}개 있습니다: {source}"
        )
    frame["일자"] = parsed_date.dt.normalize()

    numeric_columns = ["합계", *HOUR_FIELDS]
    for column in numeric_columns:
        raw = frame[column]
        converted = pd.to_numeric(raw, errors="coerce")
        invalid = raw.notna() & converted.isna()
        if invalid.any():
            raise MonthlyDataError(
                f"{column}에 숫자가 아닌 값이 {int(invalid.sum())}개 있습니다: {source}"
            )
        frame[column] = converted

    frame["교차로명"] = frame["교차로명"].astype("string").str.strip()
    frame["접근로명"] = frame["접근로명"].astype("string").str.strip()
    if frame[["교차로명", "접근로명"]].isna().any().any():
        raise MonthlyDataError(f"교차로명 또는 접근로명이 비어 있습니다: {source}")
    return frame


def load_monthly_files(raw_dir: Path) -> pd.DataFrame:
    paths = sorted(raw_dir.glob("cheonan_smart_intersection_traffic_????-??.json"))
    if not paths:
        raise MonthlyDataError(f"월별 JSON을 찾을 수 없습니다: {raw_dir}")

    frames: list[pd.DataFrame] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("resultList") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise MonthlyDataError(f"resultList 배열이 없습니다: {path}")
        frames.append(canonicalize_rows(rows, str(path)))

    merged = pd.concat(frames, ignore_index=True)
    duplicate_mask = merged.duplicated(KEY_COLUMNS, keep=False)
    if duplicate_mask.any():
        raise MonthlyDataError(
            "병합 결과에 일자·교차로·접근로 중복이 있습니다. "
            f"중복 행 수: {int(duplicate_mask.sum())}"
        )
    return merged.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)


def build_continuity_reports(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        raise MonthlyDataError("연속성을 계산할 데이터가 없습니다.")

    hours = list(HOUR_FIELDS)
    start_date = frame["일자"].min()
    end_date = frame["일자"].max()
    full_period_days = int((end_date - start_date).days + 1)

    quality = frame.copy()
    quality["missing_hour_cells"] = quality[hours].isna().sum(axis=1)
    quality["negative_hour_cells"] = quality[hours].lt(0).sum(axis=1)
    quality["zero_hour_cells"] = quality[hours].eq(0).sum(axis=1)
    quality["all_zero_day"] = quality[hours].eq(0).all(axis=1)
    quality["sum_mismatch"] = (
        quality[hours].sum(axis=1, min_count=1) - quality["합계"]
    ).abs().gt(0.001)

    pair = (
        quality.groupby(["교차로명", "접근로명"], sort=True, observed=True)
        .agg(
            first_date=("일자", "min"),
            last_date=("일자", "max"),
            observed_days=("일자", "nunique"),
            missing_hour_cells=("missing_hour_cells", "sum"),
            negative_hour_cells=("negative_hour_cells", "sum"),
            zero_hour_cells=("zero_hour_cells", "sum"),
            all_zero_days=("all_zero_day", "sum"),
            sum_mismatch_days=("sum_mismatch", "sum"),
            total_volume=("합계", "sum"),
        )
        .reset_index()
    )
    pair["full_period_days"] = full_period_days
    pair["active_span_days"] = (
        pair["last_date"] - pair["first_date"]
    ).dt.days.add(1)
    pair["coverage_full_period"] = pair["observed_days"] / full_period_days
    pair["coverage_active_span"] = pair["observed_days"] / pair["active_span_days"]

    intersection_base = (
        quality.groupby("교차로명", sort=True, observed=True)
        .agg(
            first_date=("일자", "min"),
            last_date=("일자", "max"),
            approach_count=("접근로명", "nunique"),
            observed_approach_days=("일자", "size"),
            missing_hour_cells=("missing_hour_cells", "sum"),
            negative_hour_cells=("negative_hour_cells", "sum"),
            zero_hour_cells=("zero_hour_cells", "sum"),
            all_zero_days=("all_zero_day", "sum"),
            sum_mismatch_days=("sum_mismatch", "sum"),
            total_volume=("합계", "sum"),
        )
        .reset_index()
    )
    pair_summary = (
        pair.groupby("교차로명", sort=True, observed=True)
        .agg(
            min_pair_coverage_full=("coverage_full_period", "min"),
            mean_pair_coverage_full=("coverage_full_period", "mean"),
            min_pair_coverage_active=("coverage_active_span", "min"),
            mean_pair_coverage_active=("coverage_active_span", "mean"),
        )
        .reset_index()
    )
    intersection = intersection_base.merge(pair_summary, on="교차로명", validate="one_to_one")
    intersection["full_period_days"] = full_period_days
    intersection["expected_approach_days"] = (
        intersection["approach_count"] * full_period_days
    )
    intersection["coverage_full_period"] = (
        intersection["observed_approach_days"]
        / intersection["expected_approach_days"]
    )
    intersection["zero_hour_rate"] = (
        intersection["zero_hour_cells"]
        / (intersection["observed_approach_days"] * 24)
    )
    intersection = intersection.sort_values(
        ["coverage_full_period", "total_volume"], ascending=[False, False]
    ).reset_index(drop=True)
    return pair, intersection


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    for column in output.select_dtypes(include=["datetime", "datetimetz"]).columns:
        output[column] = output[column].dt.strftime("%Y-%m-%d")
    output.to_csv(path, index=False, encoding="utf-8-sig")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--wide-output",
        type=Path,
        default=Path("data/interim/cheonan_traffic_wide_2024-01_2026-08.csv"),
    )
    parser.add_argument(
        "--pair-report",
        type=Path,
        default=Path("data/processed/approach_continuity.csv"),
    )
    parser.add_argument(
        "--intersection-report",
        type=Path,
        default=Path("data/processed/intersection_continuity.csv"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    merged = load_monthly_files(args.raw_dir)
    pair_report, intersection_report = build_continuity_reports(merged)

    write_csv(merged, args.wide_output)
    write_csv(pair_report, args.pair_report)
    write_csv(intersection_report, args.intersection_report)

    print(f"wide rows: {len(merged):,}")
    print(f"period: {merged['일자'].min():%Y-%m-%d} ~ {merged['일자'].max():%Y-%m-%d}")
    print(f"intersections: {merged['교차로명'].nunique():,}")
    print(f"approach pairs: {len(pair_report):,}")
    print(f"wide output: {args.wide_output}")
    print(f"pair report: {args.pair_report}")
    print(f"intersection report: {args.intersection_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
