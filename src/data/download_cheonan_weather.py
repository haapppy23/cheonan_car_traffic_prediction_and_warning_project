"""Open-Meteo에서 천안시 시간별 과거 기상 자료를 내려받는다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import requests

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"
CHEONAN_LATITUDE = 36.8151
CHEONAN_LONGITUDE = 127.1139
HOURLY_VARIABLES = (
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
)


class WeatherDownloadError(ValueError):
    """기상 API 응답이 학습 자료로 쓸 수 없을 때 발생한다."""


def parse_weather_response(payload: dict[str, object]) -> pd.DataFrame:
    """Open-Meteo JSON을 중복 없는 시간별 표로 검증·변환한다."""
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise WeatherDownloadError("기상 응답에 hourly 자료가 없습니다.")
    required = {"time", *HOURLY_VARIABLES}
    missing = sorted(required - set(hourly))
    if missing:
        raise WeatherDownloadError(f"기상 응답에 필요한 항목이 없습니다: {missing}")
    lengths = {name: len(hourly[name]) for name in required}
    if len(set(lengths.values())) != 1:
        raise WeatherDownloadError(f"기상 응답 배열 길이가 다릅니다: {lengths}")

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(hourly["time"], errors="coerce"),
            "weather_temperature_2m": pd.to_numeric(
                hourly["temperature_2m"], errors="coerce"
            ),
            "weather_precipitation": pd.to_numeric(
                hourly["precipitation"], errors="coerce"
            ),
            "weather_wind_speed_10m": pd.to_numeric(
                hourly["wind_speed_10m"], errors="coerce"
            ),
        }
    )
    if frame.isna().any().any():
        raise WeatherDownloadError("기상 응답에 해석할 수 없는 시각이나 결측값이 있습니다.")
    if frame["timestamp"].duplicated().any():
        raise WeatherDownloadError("기상 응답에 중복 시각이 있습니다.")
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    gaps = frame["timestamp"].diff().dropna()
    if not gaps.eq(pd.Timedelta(hours=1)).all():
        raise WeatherDownloadError("기상 응답이 한 시간 간격으로 연속되지 않습니다.")
    return frame


def download_weather(
    *,
    start_date: str,
    end_date: str,
    timeout: int = 120,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """천안 중심 좌표의 시간별 재분석 기상 자료를 요청한다."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if start > end:
        raise WeatherDownloadError("기상 시작일은 종료일보다 늦을 수 없습니다.")
    params = {
        "latitude": CHEONAN_LATITUDE,
        "longitude": CHEONAN_LONGITUDE,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "Asia/Seoul",
    }
    response = requests.get(ARCHIVE_API_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    frame = parse_weather_response(payload)
    metadata = {
        "source": ARCHIVE_API_URL,
        "source_kind": "Open-Meteo historical reanalysis",
        "latitude_requested": CHEONAN_LATITUDE,
        "longitude_requested": CHEONAN_LONGITUDE,
        "latitude_returned": payload.get("latitude"),
        "longitude_returned": payload.get("longitude"),
        "elevation": payload.get("elevation"),
        "timezone": payload.get("timezone"),
        "start": frame["timestamp"].min().isoformat(),
        "end": frame["timestamp"].max().isoformat(),
        "rows": len(frame),
        "hourly_units": payload.get("hourly_units", {}),
        "variables": list(HOURLY_VARIABLES),
    }
    return frame, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2026-08-31")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/cheonan_weather_hourly.csv"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/external/cheonan_weather_metadata.json"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    frame, metadata = download_weather(
        start_date=args.start_date,
        end_date=args.end_date,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding="utf-8-sig")
    args.metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"weather rows: {len(frame):,}")
    print(f"weather period: {metadata['start']} ~ {metadata['end']}")
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
