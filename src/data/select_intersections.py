"""선정 목록과 연속성 지표를 결합해 모델링 대상 데이터를 만든다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.schema import write_table
from src.data.wide_to_long import wide_to_long


class SelectionError(ValueError):
    """교차로 선정 설정이나 입력 데이터가 일치하지 않을 때 발생한다."""


def load_selection(path: Path) -> pd.DataFrame:
    selection = pd.read_csv(path, encoding="utf-8-sig")
    required = ["생활권", "교차로명", "선정근거"]
    missing = [column for column in required if column not in selection]
    if missing:
        raise SelectionError(f"선정 파일에 필요한 열이 없습니다: {missing}")
    if selection["교차로명"].duplicated().any():
        duplicates = selection.loc[
            selection["교차로명"].duplicated(keep=False), "교차로명"
        ].tolist()
        raise SelectionError(f"선정 교차로가 중복되었습니다: {duplicates}")
    return selection[required]


def combine_selection_metrics(
    selection: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    minimum_coverage: float = 0.99,
) -> pd.DataFrame:
    result = selection.merge(metrics, on="교차로명", how="left", validate="one_to_one")
    missing_metrics = result.loc[result["coverage_full_period"].isna(), "교차로명"].tolist()
    if missing_metrics:
        raise SelectionError(f"연속성 보고서에 없는 교차로입니다: {missing_metrics}")

    insufficient = result.loc[
        result["min_pair_coverage_full"] < minimum_coverage,
        ["교차로명", "min_pair_coverage_full"],
    ]
    if not insufficient.empty:
        raise SelectionError(
            "접근로 최소 관측률이 기준보다 낮습니다: "
            + insufficient.to_dict(orient="records").__str__()
        )

    result["품질상태"] = "사용"
    result.loc[result["zero_hour_rate"] >= 0.10, "품질상태"] = "0값 주의"
    return result


def filter_wide_data(wide: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    selected_names = selection["교차로명"].tolist()
    available = set(wide["교차로명"].dropna().astype(str))
    missing = [name for name in selected_names if name not in available]
    if missing:
        raise SelectionError(f"병합 데이터에 없는 교차로입니다: {missing}")

    result = wide.loc[wide["교차로명"].isin(selected_names)].copy()
    result["_order"] = pd.Categorical(
        result["교차로명"], categories=selected_names, ordered=True
    )
    result = result.sort_values(
        ["_order", "접근로명", "일자"], kind="stable"
    ).drop(columns="_order")
    return result.reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection", type=Path, default=Path("config/selected_intersections.csv")
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("data/processed/intersection_continuity.csv"),
    )
    parser.add_argument(
        "--wide-input",
        type=Path,
        default=Path("data/interim/cheonan_traffic_wide_2024-01_2026-08.csv"),
    )
    parser.add_argument(
        "--selected-metrics",
        type=Path,
        default=Path("data/processed/selected_intersection_metrics.csv"),
    )
    parser.add_argument(
        "--selected-wide",
        type=Path,
        default=Path("data/processed/selected_traffic_wide.csv"),
    )
    parser.add_argument(
        "--selected-long",
        type=Path,
        default=Path("data/processed/selected_traffic_long.csv"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    selection = load_selection(args.selection)
    metrics = pd.read_csv(args.metrics, encoding="utf-8-sig")
    selected_metrics = combine_selection_metrics(selection, metrics)

    wide = pd.read_csv(args.wide_input, encoding="utf-8-sig")
    selected_wide = filter_wide_data(wide, selection)
    selected_long = wide_to_long(selected_wide)

    args.selected_metrics.parent.mkdir(parents=True, exist_ok=True)
    selected_metrics.to_csv(args.selected_metrics, index=False, encoding="utf-8-sig")
    selected_wide.to_csv(args.selected_wide, index=False, encoding="utf-8-sig")
    write_table(selected_long, args.selected_long)

    print(f"selected intersections: {len(selection):,}")
    print(f"selected approaches: {selected_wide[['교차로명', '접근로명']].drop_duplicates().shape[0]:,}")
    print(f"selected wide rows: {len(selected_wide):,}")
    print(f"selected long rows: {len(selected_long):,}")
    print(f"quality warnings: {(selected_metrics['품질상태'] != '사용').sum():,}")
    print(f"metrics output: {args.selected_metrics}")
    print(f"long output: {args.selected_long}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
