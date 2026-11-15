"""천안교통정보센터의 스마트교차로 교통량을 월별 JSON으로 내려받는다."""

from __future__ import annotations

import argparse
import json
import time
from calendar import monthrange
from datetime import date
from pathlib import Path
from typing import Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ENDPOINT = "https://its.cheonan.go.kr/stat/getIxrStatExcelList.do"


def parse_month(value: str) -> date:
    try:
        year_text, month_text = value.split("-", maxsplit=1)
        return date(int(year_text), int(month_text), 1)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            f"월은 YYYY-MM 형식이어야 합니다: {value!r}"
        ) from error


def iter_months(start: date, end: date) -> Iterator[date]:
    if start > end:
        raise ValueError("시작 월은 종료 월보다 늦을 수 없습니다.")

    current = start
    while current <= end:
        yield current
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)


def month_period(month: date) -> tuple[str, str]:
    last_day = monthrange(month.year, month.month)[1]
    start = month.strftime("%Y%m%d")
    end = date(month.year, month.month, last_day).strftime("%Y%m%d")
    return start, end


def validate_payload(payload: object, month: date) -> list[dict[str, object]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("resultList"), list):
        raise ValueError("공식 응답에 resultList 배열이 없습니다.")

    rows = payload["resultList"]
    if not rows:
        raise ValueError(f"{month:%Y-%m} 응답에 교통량 행이 없습니다.")

    expected_prefix = month.strftime("%Y-%m")
    invalid_dates = [
        row.get("STAT_DT")
        for row in rows
        if not isinstance(row, dict)
        or not str(row.get("STAT_DT", "")).startswith(expected_prefix)
    ]
    if invalid_dates:
        raise ValueError(
            f"{month:%Y-%m} 범위를 벗어난 행이 있습니다. 예시: {invalid_dates[:3]}"
        )
    return rows


def download_month(month: date, timeout: int) -> bytes:
    start_date, end_date = month_period(month)
    body = urlencode(
        {"ixrName": "", "startDate": start_date, "endDate": end_date}
    ).encode("utf-8")
    request = Request(
        ENDPOINT,
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "User-Agent": "cheonan-traffic-course-project/1.0",
        },
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def save_month(
    month: date,
    output_dir: Path,
    *,
    timeout: int,
    force: bool,
    retries: int = 3,
) -> tuple[Path, int, bool]:
    output = output_dir / f"cheonan_smart_intersection_traffic_{month:%Y-%m}.json"
    if output.exists() and not force:
        existing = json.loads(output.read_text(encoding="utf-8"))
        rows = validate_payload(existing, month)
        return output, len(rows), True

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            raw = download_month(month, timeout)
            payload = json.loads(raw.decode("utf-8"))
            rows = validate_payload(payload, month)
            output_dir.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(output.suffix + ".part")
            temporary.write_bytes(raw)
            temporary.replace(output)
            return output, len(rows), False
        except (HTTPError, URLError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(attempt * 2)

    raise RuntimeError(f"{month:%Y-%m} 다운로드에 실패했습니다: {last_error}") from last_error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-month", type=parse_month, required=True)
    parser.add_argument("--end-month", type=parse_month, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    months = list(iter_months(args.start_month, args.end_month))
    total_rows = 0

    for index, month in enumerate(months, start=1):
        path, rows, skipped = save_month(
            month,
            args.output_dir,
            timeout=args.timeout,
            force=args.force,
        )
        total_rows += rows
        status = "verified existing" if skipped else "downloaded"
        print(f"[{index}/{len(months)}] {month:%Y-%m}: {status}, {rows:,} rows -> {path}", flush=True)
        if index < len(months) and not skipped and args.delay > 0:
            time.sleep(args.delay)

    print(f"complete: {len(months)} months, {total_rows:,} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
