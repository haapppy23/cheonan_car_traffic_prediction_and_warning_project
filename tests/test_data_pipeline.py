from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.data.inspect_raw import inspect_dataframe
from src.data.schema import DataSchemaError, read_table, resolve_wide_schema
from src.data.wide_to_long import DataValidationError, wide_to_long


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample" / "cheonan_traffic_wide_sample.csv"


@pytest.fixture()
def sample_frame() -> pd.DataFrame:
    return read_table(SAMPLE)


def test_valid_schema_has_all_24_hour_columns(sample_frame: pd.DataFrame) -> None:
    schema = resolve_wide_schema(sample_frame)
    assert list(schema.hours) == list(range(24))
    assert len(schema.hours) == 24


def test_missing_hour_column_raises_clear_error(sample_frame: pd.DataFrame) -> None:
    missing_hour = sample_frame.drop(columns=["23시"])
    with pytest.raises(DataSchemaError, match="누락된 시간 열: 23시"):
        resolve_wide_schema(missing_hour)


def test_one_wide_row_becomes_24_long_rows(sample_frame: pd.DataFrame) -> None:
    one_row = sample_frame.iloc[[0]].copy()
    result = wide_to_long(one_row)
    assert len(result) == 24


def test_timestamp_is_created_from_date_and_hour(sample_frame: pd.DataFrame) -> None:
    one_row = sample_frame.iloc[[0]].copy()
    result = wide_to_long(one_row)
    assert result.loc[0, "timestamp"] == pd.Timestamp("2024-01-01 00:00:00")
    assert result.loc[23, "timestamp"] == pd.Timestamp("2024-01-01 23:00:00")


def test_volume_is_numeric(sample_frame: pd.DataFrame) -> None:
    result = wide_to_long(sample_frame)
    assert pd.api.types.is_numeric_dtype(result["volume"])


def test_conversion_does_not_modify_source(sample_frame: pd.DataFrame) -> None:
    original = sample_frame.copy(deep=True)
    wide_to_long(sample_frame)
    assert_frame_equal(sample_frame, original)


def test_duplicate_keys_are_detected(sample_frame: pd.DataFrame) -> None:
    duplicated = pd.concat([sample_frame, sample_frame.iloc[[0]]], ignore_index=True)
    report = inspect_dataframe(duplicated)
    assert report["duplicate_key_rows"] == 2
    with pytest.raises(DataValidationError, match="중복"):
        wide_to_long(duplicated)
