from __future__ import annotations

import os
from dataclasses import dataclass

import requests

DEFAULT_API_URL = "http://127.0.0.1:8000"
TIMEOUT_SECONDS = 10

FIELD_REQ_INTERSECTION = "intersection"
FIELD_REQ_APPROACH = "approach"
FIELD_REQ_TARGET_TIME = "target_timestamp"
FIELD_RES_PREDICTION = "predicted_volume"
FIELD_RES_THRESHOLD = "high_traffic_threshold"
FIELD_RES_ALERT = "high_traffic_alert"
FIELD_RES_RISK = "risk_score"
FIELD_RES_TARGET_TIME = "target_timestamp"
FIELD_RES_FACTORS = "major_factors"


class ApiError(Exception):
    pass


@dataclass
class PredictionFactor:
    feature: str
    label: str
    contribution: float
    effect: str


@dataclass
class Prediction:
    predicted_volume: float
    threshold: float
    high_traffic_alert: bool
    risk_score: float
    target_time: str | None = None
    major_factors: tuple[PredictionFactor, ...] = ()


DIRECTION_WORDS = {"동": "동쪽", "서": "서쪽", "남": "남쪽", "북": "북쪽"}


def approach_label(approach: str) -> str:
    suffix = approach.rsplit("_", 1)[-1] if "_" in approach else ""
    word = DIRECTION_WORDS.get(suffix)
    return f"{word}에서 진입 ({approach})" if word else approach


def api_url() -> str:
    return os.environ.get("TRAFFIC_API_URL", DEFAULT_API_URL).rstrip("/")


def _request(method: str, path: str, base_url: str | None = None,
             not_found_message: str | None = None, **kwargs) -> dict | list:
    url = f"{base_url or api_url()}{path}"
    try:
        resp = requests.request(method, url, timeout=TIMEOUT_SECONDS, **kwargs)
    except requests.exceptions.Timeout:
        raise ApiError("예측 서버의 응답이 늦습니다. 잠시 후 다시 시도해 주세요.")
    except requests.exceptions.RequestException:
        raise ApiError(f"예측 서버({base_url or api_url()})에 연결할 수 없습니다. "
                       "FastAPI 서버가 실행 중인지 확인해 주세요.")

    if resp.status_code == 404 and not_found_message:
        raise ApiError(not_found_message)
    if resp.status_code == 404:
        raise ApiError("선택한 교차로·접근로 또는 시각의 자료가 없어 예측할 수 없습니다. "
                       "다른 접근로나 시각을 선택해 주세요." + _detail(resp))
    if resp.status_code == 422:
        raise ApiError("이 시각은 예측할 수 없습니다. 예측 직전 1주(168시간) 교통량 이력이 부족하거나 "
                       "입력값이 올바르지 않습니다. 다른 시각이나 접근로를 선택해 주세요."
                       + _detail(resp))
    if resp.status_code >= 400:
        raise ApiError(f"예측 서버에서 오류가 발생했습니다. (HTTP {resp.status_code})")
    try:
        return resp.json()
    except ValueError:
        raise ApiError("예측 서버의 응답을 읽을 수 없습니다. 백엔드 상태를 확인해 주세요.")


def _detail(resp) -> str:
    try:
        detail = resp.json().get("detail")
    except (ValueError, AttributeError):
        return ""
    if isinstance(detail, str) and detail:
        return f" (서버 메시지: {detail})"
    if isinstance(detail, list) and detail:
        msgs = [d.get("msg", "") for d in detail if isinstance(d, dict)]
        return f" (서버 메시지: {'; '.join(m for m in msgs if m)})" if any(msgs) else ""
    return ""


def check_health(base_url: str | None = None) -> bool:
    _request("GET", "/health", base_url)
    return True


def get_locations(base_url: str | None = None) -> dict[str, list[str]]:
    data = _request("GET", "/locations", base_url, not_found_message=(
        "예측 서버에 교차로·접근로 목록 기능(/locations)이 없습니다. "
        "이나율 님의 백엔드 작업이 반영된 최신 main을 받았는지 확인해 주세요."))
    if isinstance(data, dict):
        for key in ("locations", "intersections", "items", "data"):
            if key in data and isinstance(data[key], (list, dict)):
                data = data[key]
                break
    result: dict[str, list[str]] = {}
    if isinstance(data, dict):
        for inter, apprs in data.items():
            if isinstance(apprs, list):
                result[str(inter)] = [str(a) for a in apprs]
    elif isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue
            inter = row.get(FIELD_REQ_INTERSECTION) or row.get("name")
            if inter is None:
                continue
            apprs = row.get("approaches")
            if apprs is None and row.get(FIELD_REQ_APPROACH) is not None:
                apprs = [row[FIELD_REQ_APPROACH]]
            for a in apprs or []:
                name = a.get("name", a.get(FIELD_REQ_APPROACH)) if isinstance(a, dict) else a
                if name is not None and str(name) not in result.setdefault(str(inter), []):
                    result[str(inter)].append(str(name))
    result = {k: v for k, v in result.items() if v}
    if not result:
        raise ApiError("예측 서버에서 교차로·접근로 목록을 받지 못했습니다.")
    return result


def predict(intersection: str, approach: str, target_timestamp: str | None = None,
            base_url: str | None = None) -> Prediction:
    payload = {FIELD_REQ_INTERSECTION: intersection, FIELD_REQ_APPROACH: approach}
    if target_timestamp:
        payload[FIELD_REQ_TARGET_TIME] = target_timestamp
    data = _request("POST", "/predict", base_url, json=payload)
    if not isinstance(data, dict):
        raise ApiError("예측 서버의 응답 형식이 약속과 다릅니다.")
    missing = [k for k in (FIELD_RES_PREDICTION, FIELD_RES_THRESHOLD, FIELD_RES_ALERT,
                           FIELD_RES_RISK) if k not in data]
    if missing:
        reason = next((data[k] for k in ("detail", "message", "status", "reason")
                       if isinstance(data.get(k), str) and data[k]), None)
        if reason:
            raise ApiError("이 접근로·시각은 예측에 필요한 과거 교통량 자료가 부족해 예측할 수 없습니다. "
                           f"다른 접근로나 시각을 선택해 주세요. (서버 메시지: {reason})")
        raise ApiError(f"예측 서버의 응답에 필요한 값이 없습니다: {', '.join(missing)}")
    try:
        raw_factors = data.get(FIELD_RES_FACTORS, [])
        if not isinstance(raw_factors, list):
            raise TypeError
        factors = tuple(
            PredictionFactor(
                feature=str(item["feature"]),
                label=str(item["label"]),
                contribution=float(item["contribution"]),
                effect=str(item["effect"]),
            )
            for item in raw_factors
            if isinstance(item, dict)
        )
        return Prediction(
            predicted_volume=float(data[FIELD_RES_PREDICTION]),
            threshold=float(data[FIELD_RES_THRESHOLD]),
            high_traffic_alert=bool(data[FIELD_RES_ALERT]),
            risk_score=float(data[FIELD_RES_RISK]),
            target_time=data.get(FIELD_RES_TARGET_TIME),
            major_factors=factors,
        )
    except (KeyError, TypeError, ValueError):
        raise ApiError("예측 서버의 응답 값 형식이 올바르지 않습니다.")
