"""원본 교통량 파일의 공통 스키마와 입출력 규칙."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


class DataSchemaError(ValueError):
    """원본 열 구조가 예상 형식과 다를 때 발생한다."""


DATE_ALIASES = ("일자", "날짜", "date")
INTERSECTION_ALIASES = ("교차로명", "교차로", "intersection")
APPROACH_ALIASES = ("접근로명", "접근로", "진입로명", "진입로", "approach")
TOTAL_ALIASES = ("합계", "total")


@dataclass(frozen=True)
class WideSchema:
    date: str
    intersection: str
    approach: str
    total: str | None
    hours: dict[int, str]


def read_table(path: str | Path) -> pd.DataFrame:
    """UTF-8/CP949 CSV 또는 XLSX를 읽고 열 이름의 바깥 공백을 제거한다."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {source}")

    suffix = source.suffix.lower()
    if suffix == ".csv":
        last_error: UnicodeDecodeError | None = None
        for encoding in ("utf-8-sig", "utf-8", "cp949"):
            try:
                frame = pd.read_csv(source, encoding=encoding)
                break
            except UnicodeDecodeError as error:
                last_error = error
        else:
            raise DataSchemaError(
                f"CSV 인코딩을 읽지 못했습니다. UTF-8 또는 CP949인지 확인하십시오: {source}"
            ) from last_error
    elif suffix == ".xlsx":
        frame = pd.read_excel(source, engine="openpyxl")
    else:
        raise DataSchemaError(
            f"지원하지 않는 파일 형식입니다: {suffix or '(확장자 없음)'}. CSV 또는 XLSX를 사용하십시오."
        )

    normalized = [str(column).strip() for column in frame.columns]
    if len(normalized) != len(set(normalized)):
        raise DataSchemaError("열 이름의 공백을 제거한 뒤 중복된 열 이름이 발견되었습니다.")
    frame.columns = normalized
    return frame


def _resolve_column(
    columns: Iterable[str],
    explicit: str | None,
    aliases: tuple[str, ...],
    label: str,
    required: bool = True,
) -> str | None:
    available = list(columns)
    if explicit:
        if explicit not in available:
            raise DataSchemaError(
                f"지정한 {label} 열 '{explicit}'이 없습니다. 실제 열: {available}"
            )
        return explicit

    lower_map = {column.lower(): column for column in available}
    matches = [lower_map[alias.lower()] for alias in aliases if alias.lower() in lower_map]
    matches = list(dict.fromkeys(matches))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise DataSchemaError(
            f"{label} 후보가 여러 개입니다: {matches}. 명령행 옵션으로 사용할 열을 지정하십시오."
        )
    if required:
        raise DataSchemaError(
            f"필수 {label} 열을 찾지 못했습니다. 후보 이름은 {list(aliases)}이며 실제 열은 {available}입니다."
        )
    return None


def detect_hour_columns(columns: Iterable[str]) -> dict[int, str]:
    """각 시각에 대응하는 열을 탐지하되 누락과 중복 후보를 명확히 거부한다."""
    available = list(columns)
    result: dict[int, str] = {}
    missing: list[str] = []

    for hour in range(24):
        candidates = (f"{hour:02d}시", f"{hour}시", f"{hour:02d}", str(hour))
        matches = [candidate for candidate in candidates if candidate in available]
        matches = list(dict.fromkeys(matches))
        if not matches:
            missing.append(f"{hour:02d}시")
        elif len(matches) > 1:
            raise DataSchemaError(
                f"{hour:02d}시에 해당하는 열 후보가 여러 개입니다: {matches}. 원본 열 이름을 정리하십시오."
            )
        else:
            result[hour] = matches[0]

    if missing:
        raise DataSchemaError(
            "24개 시간 열이 모두 필요합니다. 누락된 시간 열: " + ", ".join(missing)
        )
    return result


def resolve_wide_schema(
    frame: pd.DataFrame,
    *,
    date_column: str | None = None,
    intersection_column: str | None = None,
    approach_column: str | None = None,
    total_column: str | None = None,
) -> WideSchema:
    columns = list(frame.columns)
    date = _resolve_column(columns, date_column, DATE_ALIASES, "일자")
    intersection = _resolve_column(
        columns, intersection_column, INTERSECTION_ALIASES, "교차로"
    )
    approach = _resolve_column(columns, approach_column, APPROACH_ALIASES, "접근로")
    total = _resolve_column(
        columns, total_column, TOTAL_ALIASES, "합계", required=False
    )
    hours = detect_hour_columns(columns)
    return WideSchema(
        date=date,
        intersection=intersection,
        approach=approach,
        total=total,
        hours=hours,
    )


def write_table(frame: pd.DataFrame, path: str | Path) -> None:
    """결과를 CSV 또는 Parquet으로 저장한다."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(destination, index=False, encoding="utf-8-sig")
    elif suffix in {".parquet", ".pq"}:
        try:
            frame.to_parquet(destination, index=False)
        except ImportError as error:
            raise DataSchemaError(
                "Parquet 저장에는 pyarrow가 필요합니다. 'python -m pip install -r requirements.txt'를 실행하십시오."
            ) from error
    else:
        raise DataSchemaError("출력 확장자는 .csv, .parquet 또는 .pq여야 합니다.")
