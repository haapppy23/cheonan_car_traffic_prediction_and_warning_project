from __future__ import annotations

import pandas as pd
import pytest

from src.data.run_eda import (
    EDAValidationError,
    build_summary_tables,
    mark_all_zero_approach_days,
    validate_long_data,
)


def sample_long() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for hour in range(24):
        rows.append(
            {
                "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=hour),
                "date": pd.Timestamp("2024-01-01"),
                "intersection": "교차로A",
                "approach": "북",
                "volume": 0,
                "hour": hour,
                "day_of_week": 0,
                "is_weekend": False,
                "area": "불당",
            }
        )
    for hour in range(24):
        rows.append(
            {
                "timestamp": pd.Timestamp("2024-01-02") + pd.Timedelta(hours=hour),
                "date": pd.Timestamp("2024-01-02"),
                "intersection": "교차로A",
                "approach": "북",
                "volume": hour,
                "hour": hour,
                "day_of_week": 1,
                "is_weekend": False,
                "area": "불당",
            }
        )
    return pd.DataFrame(rows)


def test_mark_all_zero_approach_days_marks_only_full_zero_day() -> None:
    result = mark_all_zero_approach_days(sample_long())
    assert result.loc[result["date"] == pd.Timestamp("2024-01-01"), "all_zero_approach_day"].all()
    assert not result.loc[result["date"] == pd.Timestamp("2024-01-02"), "all_zero_approach_day"].any()


def test_build_summary_tables_counts_zero_days() -> None:
    result = mark_all_zero_approach_days(sample_long())
    tables = build_summary_tables(result)
    intersection = tables["intersection"].iloc[0]
    assert intersection["all_zero_approach_days"] == 1
    assert intersection["zero_hours"] == 25
    assert len(tables["hourly_area"]) == 24


def test_duplicate_key_validation_error_message() -> None:
    frame = sample_long().iloc[[0, 0]].copy()
    with pytest.raises(EDAValidationError, match="중복"):
        validate_long_data(frame)
