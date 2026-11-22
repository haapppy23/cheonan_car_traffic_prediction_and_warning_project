"""천안시 스마트교차로 교통량 예측 · 고교통량 조기경보 Streamlit 화면.

실행:  streamlit run frontend/app.py
API 주소는 환경변수 TRAFFIC_API_URL (기본 http://127.0.0.1:8000).
이 화면은 FastAPI만 호출하며 모델이나 경보 기준을 직접 계산하지 않는다.
"""
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import api_client as api  # noqa: E402

# docs 기준 자료 기간. 첫 1주(168시간)는 과거 이력이 없어 예측 불가라 제외 (feature_dataset.md)
DATA_START = date(2024, 1, 8)
DATA_END = date(2026, 8, 31)

st.set_page_config(page_title="천안 교통량 예측", page_icon="🚦", layout="centered")
st.title("천안시 스마트교차로 교통량 예측")
st.caption("선택한 접근로의 1시간 후 차량 통행량과 고교통량 위험 여부를 보여줍니다.")


@st.cache_data(ttl=300, show_spinner=False)
def load_locations(base_url: str) -> dict:
    return api.get_locations(base_url)


base_url = api.api_url()

# 1) 연결 상태
try:
    api.check_health(base_url)
    st.success(f"예측 서버 연결됨 ({base_url})")
except api.ApiError as e:
    st.error(str(e))
    st.stop()

# 2) 교차로·접근로 선택 (목록은 /locations에서 받음)
try:
    locations = load_locations(base_url)
except api.ApiError as e:
    st.error(str(e))
    st.stop()

col1, col2 = st.columns(2)
intersection = col1.selectbox("교차로", list(locations))
approach = col2.selectbox("접근로 (진입 방향)", locations[intersection],
                          format_func=api.approach_label)
st.caption("방향은 차량이 교차로로 들어오기 전에 있던 쪽입니다. 예: '남쪽에서 진입'은 교차로 남쪽 도로에서 들어오는 차량입니다.")

# 기준 시각: 키보드 입력 없이 달력과 목록으로 고른다.
# 백엔드는 '예측할 시각'(target_timestamp)을 받으므로 기준 시각 + 1시간을 보낸다.
use_latest = st.toggle("서버에 있는 가장 최근 자료 시각으로 예측", value=True)
target_timestamp = None
if not use_latest:
    d_col, h_col = st.columns(2)
    day = d_col.date_input("기준 날짜", value=DATA_END, min_value=DATA_START, max_value=DATA_END)
    hour = h_col.selectbox("기준 시각", list(range(24)), index=12, format_func=lambda h: f"{h:02d}시")
    target_timestamp = (datetime.combine(day, time(hour)) + timedelta(hours=1)).isoformat()
    st.caption(f"자료 기간: {DATA_START:%Y-%m-%d} ~ {DATA_END:%Y-%m-%d}. "
               "선택한 시각의 1시간 뒤 교통량을 예측합니다.")

# 3) 예측
if st.button("예측하기", type="primary", use_container_width=True):
    try:
        with st.spinner("예측 중..."):
            result = api.predict(intersection, approach, target_timestamp=target_timestamp, base_url=base_url)
    except api.ApiError as e:
        st.error(str(e))
        st.stop()

    st.subheader(f"{intersection} · {api.approach_label(approach)}")
    if result.target_time:
        st.caption(f"예측 대상 시각: {result.target_time.replace('T', ' ')[:16]}")

    c1, c2 = st.columns(2)
    c1.metric("1시간 후 예상 교통량", f"{result.predicted_volume:,.0f} 대")
    c2.metric("고교통량 기준 (평소 상위 10%)", f"{result.threshold:,.0f} 대")
    c3, c4 = st.columns(2)
    c3.metric("고교통량 위험", "위험" if result.high_traffic_alert else "보통")
    # risk_score = 예측 교통량 ÷ 접근로별 기준값 (docs의 경보 문서 기준). 확률이 아니다.
    c4.metric("위험 비율 (기준 대비)", f"{result.risk_score * 100:.0f}%",
              help="예상 교통량을 이 접근로의 고교통량 기준으로 나눈 값입니다. "
                   "100% 이상이면 고교통량 위험입니다. 발생 확률이 아닙니다.")

    if result.high_traffic_alert:
        st.warning("고교통량 위험: 예상 교통량이 이 접근로의 평소 상위 10% 수준을 넘습니다.")
    else:
        st.info("예상 교통량이 이 접근로의 평소 범위 안에 있습니다.")

    # 결과 시각화: API 응답값 두 개를 막대로 비교한다 (화면에서 새로 계산하는 값 없음)
    st.bar_chart(
        pd.DataFrame({"교통량(대/시간)": [result.predicted_volume, result.threshold]},
                     index=["1시간 후 예상", "고교통량 기준"]),
        horizontal=True, height=180,
    )

st.divider()
st.caption("이 결과는 차량 통행량 기반의 고교통량 위험 신호이며, 속도·통행시간 자료가 없어 "
           "실제 정체를 확정하지 않습니다. 행사·사고 등 관측되지 않은 요인은 반영되지 않습니다.")
