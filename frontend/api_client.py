"""FastAPI 호출 전용 모듈. 화면(app.py)과 분리해 Streamlit 없이 테스트할 수 있다.

프론트엔드는 모델 파일을 직접 불러오거나 lag·이동 평균·경보 기준을 계산하지 않는다.
모든 값은 FastAPI(/health, /locations, /predict)에서 받는다.

요청·응답 필드는 backend/README.md와 docs/model_handoff.md의 규격을 따른다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import requests

DEFAULT_API_URL = "http://127.0.0.1:8000"
TIMEOUT_SECONDS = 10

# --- 요청 필드 (POST /predict) ---------------------------------------------
FIELD_REQ_INTERSECTION = "intersection"
FIELD_REQ_APPROACH = "approach"
FIELD_REQ_TARGET_TIME = "target_timestamp"  # 예측할 시각(한국 현지 정각, 시간대 표기 없음).
                                            # 생략하면 서버가 마지막 관측의 1시간 후를 예측한다
# --- 응답 필드 (POST /predict, docs/model_handoff.md 기준) ------------------
FIELD_RES_PREDICTION = "predicted_volume"
FIELD_RES_THRESHOLD = "high_traffic_threshold"
FIELD_RES_ALERT = "high_traffic_alert"      # 프롬프트에 명시된 이름
FIELD_RES_RISK = "risk_score"               # 프롬프트에 명시된 이름
FIELD_RES_TARGET_TIME = "target_timestamp"  # 서버가 실제로 예측한 시각


class ApiError(Exception):
    """사용자에게 그대로 보여줄 수 있는 문장을 담는 오류."""


@dataclass
class Prediction:
    predicted_volume: float
    threshold: float
    high_traffic_alert: bool
    risk_score: float
    target_time: str | None = None


DIRECTION_WORDS = {"동": "동쪽", "서": "서쪽", "남": "남쪽", "북": "북쪽"}


def approach_label(approach: str) -> str:
    """화면 표시용 이름. 원본 접근로명 '불당4_남' -> '남쪽에서 진입 (불당4_남)'.

    docs의 데이터 규칙상 접근로명은 진입 방향을 포함한다. API에는 원래 이름을 그대로 보낸다.
    형식이 다르면 원래 이름을 그대로 돌려준다.
    """
    suffix = approach.rsplit("_", 1)[-1] if "_" in approach else ""
    word = DIRECTION_WORDS.get(suffix)
    return f"{word}에서 진입 ({approach})" if word else approach


def api_url() -> str:
    """환경변수 TRAFFIC_API_URL, 없으면 기본값. 끝의 / 는 제거한다."""
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
        # 백엔드는 입력 형식 오류와 '예측 직전 168시간 이력 부족'을 모두 422로 돌려준다
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
    """FastAPI가 준 detail을 짧게 덧붙인다. 422의 리스트 형식도 처리한다."""
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
    """GET /health. 연결되면 True, 실패하면 ApiError."""
    _request("GET", "/health", base_url)
    return True


def get_locations(base_url: str | None = None) -> dict[str, list[str]]:
    """GET /locations -> {교차로: [접근로, ...]}.

    서버 응답이 아래 중 어떤 모양이어도 같은 dict로 바꾼다.
      {"교차로": ["접근로", ...]}
      {"locations": [...]} 또는 {"intersections": [...]}
      [{"intersection": "교차로", "approaches": ["접근로", ...]}, ...]
      [{"intersection": "교차로", "approach": "접근로"}, ...]
    """
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
    """POST /predict. 화면에 필요한 네 가지 값을 Prediction으로 돌려준다."""
    payload = {FIELD_REQ_INTERSECTION: intersection, FIELD_REQ_APPROACH: approach}
    if target_timestamp:
        payload[FIELD_REQ_TARGET_TIME] = target_timestamp
    data = _request("POST", "/predict", base_url, json=payload)
    if not isinstance(data, dict):
        raise ApiError("예측 서버의 응답 형식이 약속과 다릅니다.")
    missing = [k for k in (FIELD_RES_PREDICTION, FIELD_RES_THRESHOLD, FIELD_RES_ALERT,
                           FIELD_RES_RISK) if k not in data]
    if missing:
        # 백엔드는 168시간 이력이 없거나 센서 품질이 나쁘면 '데이터 부족' 상태를 돌려줄 수 있다
        reason = next((data[k] for k in ("detail", "message", "status", "reason")
                       if isinstance(data.get(k), str) and data[k]), None)
        if reason:
            raise ApiError("이 접근로·시각은 예측에 필요한 과거 교통량 자료가 부족해 예측할 수 없습니다. "
                           f"다른 접근로나 시각을 선택해 주세요. (서버 메시지: {reason})")
        raise ApiError(f"예측 서버의 응답에 필요한 값이 없습니다: {', '.join(missing)}")
    try:
        return Prediction(
            predicted_volume=float(data[FIELD_RES_PREDICTION]),
            threshold=float(data[FIELD_RES_THRESHOLD]),
            high_traffic_alert=bool(data[FIELD_RES_ALERT]),
            risk_score=float(data[FIELD_RES_RISK]),
            target_time=data.get(FIELD_RES_TARGET_TIME),
        )
    except (TypeError, ValueError):
        raise ApiError("예측 서버의 응답 값 형식이 올바르지 않습니다.")
