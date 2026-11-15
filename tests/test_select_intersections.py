from __future__ import annotations

import pandas as pd
import pytest

from src.data.select_intersections import (
    SelectionError,
    combine_selection_metrics,
    filter_wide_data,
)


def test_combine_selection_metrics_marks_high_zero_rate() -> None:
    selection = pd.DataFrame(
        [{"생활권": "터미널", "교차로명": "터미널4", "선정근거": "핵심"}]
    )
    metrics = pd.DataFrame(
        [
            {
                "교차로명": "터미널4",
                "coverage_full_period": 1.0,
                "min_pair_coverage_full": 1.0,
                "zero_hour_rate": 0.12,
            }
        ]
    )
    result = combine_selection_metrics(selection, metrics)
    assert result.loc[0, "품질상태"] == "0값 주의"


def test_combine_selection_metrics_rejects_low_coverage() -> None:
    selection = pd.DataFrame(
        [{"생활권": "두정", "교차로명": "두정역사거리", "선정근거": "이름"}]
    )
    metrics = pd.DataFrame(
        [
            {
                "교차로명": "두정역사거리",
                "coverage_full_period": 0.63,
                "min_pair_coverage_full": 0.63,
                "zero_hour_rate": 0.01,
            }
        ]
    )
    with pytest.raises(SelectionError, match="최소 관측률"):
        combine_selection_metrics(selection, metrics)


def test_filter_wide_data_preserves_selection_order() -> None:
    selection = pd.DataFrame(
        [
            {"생활권": "A", "교차로명": "교차로B", "선정근거": "test"},
            {"생활권": "A", "교차로명": "교차로A", "선정근거": "test"},
        ]
    )
    wide = pd.DataFrame(
        [
            {"일자": "2024-01-01", "교차로명": "교차로A", "접근로명": "북"},
            {"일자": "2024-01-01", "교차로명": "교차로B", "접근로명": "남"},
            {"일자": "2024-01-01", "교차로명": "미선정", "접근로명": "동"},
        ]
    )
    result = filter_wide_data(wide, selection)
    assert result["교차로명"].tolist() == ["교차로B", "교차로A"]
