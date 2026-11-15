from __future__ import annotations

from datetime import date

import pytest

from src.data.download_cheonan_traffic import (
    iter_months,
    month_period,
    parse_month,
    validate_payload,
)


def test_iter_months_crosses_year_boundary() -> None:
    assert list(iter_months(date(2024, 11, 1), date(2025, 2, 1))) == [
        date(2024, 11, 1),
        date(2024, 12, 1),
        date(2025, 1, 1),
        date(2025, 2, 1),
    ]


def test_month_period_handles_leap_february() -> None:
    assert month_period(date(2024, 2, 1)) == ("20240201", "20240229")


def test_parse_month_rejects_invalid_format() -> None:
    with pytest.raises(Exception, match="YYYY-MM"):
        parse_month("2024/02")


def test_validate_payload_accepts_matching_month() -> None:
    rows = validate_payload(
        {"resultList": [{"STAT_DT": "2024-02-01"}, {"STAT_DT": "2024-02-29"}]},
        date(2024, 2, 1),
    )
    assert len(rows) == 2


def test_validate_payload_rejects_other_month() -> None:
    with pytest.raises(ValueError, match="범위를 벗어난"):
        validate_payload(
            {"resultList": [{"STAT_DT": "2024-03-01"}]},
            date(2024, 2, 1),
        )
