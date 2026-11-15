from __future__ import annotations

import pandas as pd
import pytest

from src.data.prepare_monthly_traffic import (
    MonthlyDataError,
    build_continuity_reports,
    canonicalize_rows,
)


def api_row(day: str, intersection: str, approach: str, volume: int) -> dict[str, object]:
    row: dict[str, object] = {
        "STAT_DT": day,
        "LCTN_NM": intersection,
        "ACRD_NM": approach,
        "TOTAL_TFVL": volume * 24,
    }
    row.update({f"IXR_STAT_DT_{hour:02d}": volume for hour in range(24)})
    return row


def test_canonicalize_rows_maps_official_fields() -> None:
    result = canonicalize_rows([api_row("2024-01-01", "교차로A", "북", 3)], "test")
    assert list(result.columns[:4]) == ["일자", "교차로명", "접근로명", "합계"]
    assert result.loc[0, "00시"] == 3
    assert result.loc[0, "23시"] == 3


def test_canonicalize_rows_rejects_missing_hour_field() -> None:
    row = api_row("2024-01-01", "교차로A", "북", 3)
    del row["IXR_STAT_DT_23"]
    with pytest.raises(MonthlyDataError, match="필수 필드"):
        canonicalize_rows([row], "test")


def test_continuity_reports_calculate_full_and_active_coverage() -> None:
    rows = [
        api_row("2024-01-01", "교차로A", "북", 1),
        api_row("2024-01-03", "교차로A", "북", 1),
        api_row("2024-01-01", "교차로A", "남", 2),
        api_row("2024-01-02", "교차로A", "남", 2),
        api_row("2024-01-03", "교차로A", "남", 2),
    ]
    frame = canonicalize_rows(rows, "test")
    pair, intersection = build_continuity_reports(frame)

    north = pair.loc[pair["접근로명"] == "북"].iloc[0]
    assert north["observed_days"] == 2
    assert north["coverage_full_period"] == pytest.approx(2 / 3)
    assert north["coverage_active_span"] == pytest.approx(2 / 3)

    result = intersection.iloc[0]
    assert result["approach_count"] == 2
    assert result["coverage_full_period"] == pytest.approx(5 / 6)
    assert result["sum_mismatch_days"] == 0


def test_continuity_report_detects_zero_day_and_sum_mismatch() -> None:
    row = api_row("2024-01-01", "교차로A", "북", 0)
    row["TOTAL_TFVL"] = 1
    frame = canonicalize_rows([row], "test")
    pair, _ = build_continuity_reports(frame)
    assert pair.loc[0, "all_zero_days"] == 1
    assert pair.loc[0, "sum_mismatch_days"] == 1
