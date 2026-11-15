"""선정된 스마트교차로 시간별 데이터의 재현 가능한 EDA를 실행한다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class EDAValidationError(ValueError):
    """EDA 입력 데이터의 키나 값이 안전하지 않을 때 발생한다."""


AREA_COLORS = {
    "불당": "#3B6FB6",
    "두정": "#E07A2D",
    "천안역": "#2E8B57",
    "터미널": "#C6534F",
}


def load_long_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["timestamp", "date"])
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    validate_long_data(frame)
    return frame


def validate_long_data(frame: pd.DataFrame) -> None:
    """시간별 데이터의 필수 열과 모델링에 치명적인 품질 오류를 검사한다."""
    required = {
        "timestamp",
        "intersection",
        "approach",
        "volume",
        "date",
        "hour",
        "day_of_week",
        "is_weekend",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise EDAValidationError(f"EDA 입력에 필요한 열이 없습니다: {missing}")

    duplicate_count = int(
        frame.duplicated(["timestamp", "intersection", "approach"], keep=False).sum()
    )
    if duplicate_count:
        raise EDAValidationError(f"시간·교차로·접근로 중복 행이 {duplicate_count}개 있습니다.")
    if frame["volume"].isna().any():
        raise EDAValidationError(
            f"교통량 결측값이 {int(frame['volume'].isna().sum())}개 있습니다."
        )
    if frame["volume"].lt(0).any():
        raise EDAValidationError(
            f"음수 교통량이 {int(frame['volume'].lt(0).sum())}개 있습니다."
        )


def attach_area(frame: pd.DataFrame, selection_path: Path) -> pd.DataFrame:
    selection = pd.read_csv(selection_path, encoding="utf-8-sig")
    mapping = selection[["교차로명", "생활권"]].rename(
        columns={"교차로명": "intersection", "생활권": "area"}
    )
    result = frame.merge(mapping, on="intersection", how="left", validate="many_to_one")
    missing = sorted(result.loc[result["area"].isna(), "intersection"].unique())
    if missing:
        raise EDAValidationError(f"생활권 매핑이 없는 교차로입니다: {missing}")
    return result


def mark_all_zero_approach_days(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["all_zero_approach_day"] = result.groupby(
        ["date", "intersection", "approach"], observed=True
    )["volume"].transform(lambda values: bool(values.eq(0).all()))
    return result


def build_summary_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if "all_zero_approach_day" not in frame:
        frame = mark_all_zero_approach_days(frame)

    intersection = (
        frame.groupby(["area", "intersection"], observed=True)
        .agg(
            rows=("volume", "size"),
            approach_count=("approach", "nunique"),
            total_volume=("volume", "sum"),
            mean_volume=("volume", "mean"),
            median_volume=("volume", "median"),
            p90_volume=("volume", lambda values: values.quantile(0.90)),
            zero_hours=("volume", lambda values: int(values.eq(0).sum())),
            all_zero_hours=("all_zero_approach_day", "sum"),
        )
        .reset_index()
    )
    intersection["zero_rate"] = intersection["zero_hours"] / intersection["rows"]
    intersection["all_zero_approach_days"] = (
        intersection["all_zero_hours"] / 24
    ).round().astype(int)
    intersection = intersection.drop(columns="all_zero_hours")

    hourly_area = (
        frame.groupby(["area", "hour"], observed=True)["volume"]
        .agg(mean_volume="mean", median_volume="median")
        .reset_index()
    )

    weekend = (
        frame.groupby(["is_weekend", "hour"], observed=True)["volume"]
        .agg(mean_volume="mean", median_volume="median")
        .reset_index()
    )
    weekend["day_type"] = weekend["is_weekend"].map(
        {False: "평일", True: "주말"}
    )

    monthly_source = frame.assign(month=frame["timestamp"].dt.to_period("M").astype(str))
    monthly = (
        monthly_source.groupby("month", observed=True)
        .agg(
            mean_hourly_volume=("volume", "mean"),
            median_hourly_volume=("volume", "median"),
            zero_rate=("volume", lambda values: values.eq(0).mean()),
            total_volume=("volume", "sum"),
        )
        .reset_index()
    )
    return {
        "intersection": intersection,
        "hourly_area": hourly_area,
        "weekend": weekend,
        "monthly": monthly,
    }


def make_overall_summary(frame: pd.DataFrame) -> dict[str, object]:
    train = frame.loc[frame["timestamp"] < pd.Timestamp("2026-01-01")]
    approach_thresholds = train.groupby(
        ["intersection", "approach"], observed=True
    )["volume"].quantile(0.90)
    return {
        "rows": int(len(frame)),
        "start": frame["timestamp"].min().isoformat(),
        "end": frame["timestamp"].max().isoformat(),
        "intersections": int(frame["intersection"].nunique()),
        "approach_pairs": int(
            frame[["intersection", "approach"]].drop_duplicates().shape[0]
        ),
        "total_volume": int(frame["volume"].sum()),
        "mean_volume": float(frame["volume"].mean()),
        "median_volume": float(frame["volume"].median()),
        "zero_hours": int(frame["volume"].eq(0).sum()),
        "zero_rate": float(frame["volume"].eq(0).mean()),
        "all_zero_approach_days": int(
            frame.loc[frame["all_zero_approach_day"]]
            .drop_duplicates(["date", "intersection", "approach"])
            .shape[0]
        ),
        "train_p90_threshold_min": float(approach_thresholds.min()),
        "train_p90_threshold_median": float(approach_thresholds.median()),
        "train_p90_threshold_max": float(approach_thresholds.max()),
    }


def configure_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "Malgun Gothic",
            "axes.unicode_minus": False,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    )


def save_figures(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    configure_plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    hourly = tables["hourly_area"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for area in AREA_COLORS:
        group = hourly.loc[hourly["area"] == area]
        ax.plot(
            group["hour"],
            group["mean_volume"],
            marker="o",
            markersize=3,
            linewidth=2,
            label=area,
            color=AREA_COLORS[area],
        )
    ax.set(title="생활권별 시간대 평균 교통량", xlabel="시각", ylabel="시간당 차량 수")
    ax.set_xticks(range(24))
    ax.legend(ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "eda_hourly_by_area.png", dpi=170)
    plt.close(fig)

    weekend = tables["weekend"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = {"평일": "#355C7D", "주말": "#C06C84"}
    for day_type in ("평일", "주말"):
        group = weekend.loc[weekend["day_type"] == day_type]
        ax.plot(
            group["hour"],
            group["mean_volume"],
            marker="o",
            markersize=3,
            linewidth=2,
            label=day_type,
            color=colors[day_type],
        )
    ax.set(title="평일과 주말의 시간대 평균 교통량", xlabel="시각", ylabel="시간당 차량 수")
    ax.set_xticks(range(24))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "eda_weekday_weekend.png", dpi=170)
    plt.close(fig)

    intersection = tables["intersection"].sort_values("zero_rate")
    fig, ax = plt.subplots(figsize=(10, 6.2))
    bar_colors = ["#C6534F" if value >= 0.10 else "#4C78A8" for value in intersection["zero_rate"]]
    ax.barh(intersection["intersection"], intersection["zero_rate"] * 100, color=bar_colors)
    ax.axvline(10, color="#8B1E1E", linestyle="--", linewidth=1, label="주의 기준 10%")
    ax.set(title="교차로별 시간값 0 비율", xlabel="0값 비율 (%)", ylabel="")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "eda_zero_rate_by_intersection.png", dpi=170)
    plt.close(fig)

    monthly = tables["monthly"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(
        monthly["month"],
        monthly["mean_hourly_volume"],
        color="#3B6FB6",
        linewidth=2,
        marker="o",
        markersize=3,
    )
    ax.set(title="월별 시간당 평균 교통량", xlabel="월", ylabel="시간당 차량 수")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(output_dir / "eda_monthly_mean_volume.png", dpi=170)
    plt.close(fig)


def write_report(
    summary: dict[str, object],
    tables: dict[str, pd.DataFrame],
    report_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    intersection = tables["intersection"]
    warning_rows = intersection.loc[intersection["zero_rate"] >= 0.10]
    warnings = ", ".join(
        f"{row.intersection} {row.zero_rate * 100:.1f}%"
        for row in warning_rows.itertuples()
    ) or "없음"
    hourly = tables["hourly_area"]
    area_peak_rows = hourly.loc[
        hourly.groupby("area", observed=True)["mean_volume"].idxmax()
    ].sort_values("area")
    area_peaks = ", ".join(
        f"{row.area} {int(row.hour)}시({row.mean_volume:.1f}대)"
        for row in area_peak_rows.itertuples()
    )
    day_type = tables["weekend"]
    day_peak_rows = day_type.loc[
        day_type.groupby("day_type", observed=True)["mean_volume"].idxmax()
    ].sort_values("day_type")
    day_peaks = ", ".join(
        f"{row.day_type} {int(row.hour)}시({row.mean_volume:.1f}대)"
        for row in day_peak_rows.itertuples()
    )
    monthly = tables["monthly"]
    monthly_low = monthly.loc[monthly["mean_hourly_volume"].idxmin()]
    monthly_high = monthly.loc[monthly["mean_hourly_volume"].idxmax()]
    report = f"""# 선정 교차로 EDA 결과

## 데이터 범위

- 기간: {str(summary['start'])[:10]}~{str(summary['end'])[:10]}
- 시간별 행: {summary['rows']:,}개
- 교차로: {summary['intersections']}개
- 접근로 조합: {summary['approach_pairs']}개
- 평균 교통량: {summary['mean_volume']:.1f}대/시간
- 중앙값 교통량: {summary['median_volume']:.1f}대/시간
- 0값: {summary['zero_hours']:,}개, 전체의 {summary['zero_rate'] * 100:.1f}%
- 24시간 전체가 0인 접근로·날짜: {summary['all_zero_approach_days']:,}건

## 고교통량 기준 확인

고교통량 기준은 학습 기간인 2024년 1월부터 2025년 12월까지만 사용해 접근로별 90백분위수로 계산한다. 접근로별 기준값의 최솟값은 {summary['train_p90_threshold_min']:.1f}대, 중앙값은 {summary['train_p90_threshold_median']:.1f}대, 최댓값은 {summary['train_p90_threshold_max']:.1f}대이다. 하나의 공통 차량 수를 모든 접근로에 적용하지 않는다.

## 품질 판단

- 중복 키, 교통량 결측값과 음수 교통량은 없다.
- 단일 0값은 실제 저교통량일 수 있으므로 자동 삭제하지 않는다.
- 24시간 전체가 0인 접근로·날짜는 센서 중단 가능성이 있어 모델 학습 대상에서 제외할 품질 플래그로 사용한다.
- 0값 비율이 10% 이상인 교차로: {warnings}
- `터미널4`는 생활권 대표성 때문에 유지하되, 모델 성능을 교차로별로 따로 보고한다.

## 주요 관찰

- 생활권별 평균 교통량의 최고 시간대: {area_peaks}
- 평일과 주말의 최고 시간대: {day_peaks}
- 월별 시간당 평균은 {monthly_low['month']} {monthly_low['mean_hourly_volume']:.1f}대로 가장 낮고 {monthly_high['month']} {monthly_high['mean_hourly_volume']:.1f}대로 가장 높다. 월별 차이의 원인은 이 자료만으로 단정하지 않고 이후 월별 오차 분석 대상으로 둔다.

## 생성 그림

### 생활권별 시간대 평균

![생활권별 시간대 평균](figures/eda_hourly_by_area.png)

### 평일·주말 시간대 비교

![평일·주말 시간대 비교](figures/eda_weekday_weekend.png)

### 교차로별 0값 비율

![교차로별 0값 비율](figures/eda_zero_rate_by_intersection.png)

### 월별 시간당 평균

![월별 시간당 평균](figures/eda_monthly_mean_volume.png)

## 다음 단계

품질 플래그를 반영한 입력 변수 생성 코드를 작성한다. 시간·요일·공휴일, 과거 1·2·24·168시간 교통량과 최근 3·6시간 평균은 현재 시점 이전의 값만 사용한다. 모델 학습 전에 학습·검증·시험 기간을 시간순으로 분리한다.
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
        "--selection",
        type=Path,
        default=Path("config/selected_intersections.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/eda")
    )
    parser.add_argument(
        "--figure-dir", type=Path, default=Path("docs/figures")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("docs/eda_report.md")
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    frame = load_long_data(args.input)
    frame = attach_area(frame, args.selection)
    frame = mark_all_zero_approach_days(frame)
    tables = build_summary_tables(frame)
    summary = make_overall_summary(frame)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(
            args.output_dir / f"{name}.csv", index=False, encoding="utf-8-sig"
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_figures(tables, args.figure_dir)
    write_report(summary, tables, args.report)

    print(f"rows: {summary['rows']:,}")
    print(f"zero rate: {summary['zero_rate'] * 100:.2f}%")
    print(f"all-zero approach-days: {summary['all_zero_approach_days']:,}")
    print(f"report: {args.report}")
    print(f"figures: {args.figure_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
