# Streamlit 프론트엔드

FastAPI 예측 서버를 호출해 교차로·접근로별 1시간 후 교통량과 고교통량 위험을 보여주는 화면이다.
화면은 모델 파일을 불러오거나 경보 기준을 계산하지 않고 FastAPI만 호출한다.

## 파일

| 파일 | 역할 |
|---|---|
| `frontend/app.py` | Streamlit 화면 |
| `frontend/api_client.py` | `/health`, `/locations`, `/predict` 호출과 오류 문장 처리 (화면과 분리) |
| `tests/test_frontend_api_client.py` | API 호출 함수의 정상·오류 응답 자동 테스트 |

## 실행 순서

저장소 최상위 폴더에서 실행한다. 실행파일 압축(`천안교통_실행파일.zip`)을 먼저 풀어 둔다.

1. 의존성 설치

   ```bash
   python -m pip install -r requirements.txt
   ```

2. 터미널 1: FastAPI 서버 실행

   ```bash
   python -m uvicorn backend.app:app --reload
   ```

3. 터미널 2: Streamlit 화면 실행

   ```bash
   streamlit run frontend/app.py
   ```

   브라우저에서 `http://localhost:8501`이 열린다.

API 주소를 바꾸려면 환경변수 `TRAFFIC_API_URL`을 지정한다. 기본값은 `http://127.0.0.1:8000`이다.

```powershell
# Windows PowerShell
$env:TRAFFIC_API_URL = "http://127.0.0.1:8000"; streamlit run frontend/app.py
```

## 화면 동작

1. `GET /health`로 서버 연결을 확인한다. 실패하면 안내 문장을 보여주고 멈춘다.
2. `GET /locations`로 교차로·접근로 목록을 받아 선택 항목을 만든다. 교차로를 바꾸면 접근로 목록도 바뀐다.
3. 기준 시각은 기본으로 서버의 가장 최근 자료 시각을 쓴다. 토글을 끄면 달력과 시각 목록으로 직접 고를 수 있다.
4. **예측하기**를 누르면 `POST /predict`를 호출해 예상 교통량, 고교통량 기준, 위험 여부(`high_traffic_alert`), 위험 비율(`risk_score`)을 표시한다.
5. 같은 응답의 `recent_traffic`으로 예측 직전 24시간 교통량과 평상시 상위 10% 기준을 선 그래프로 비교한다.
6. `major_factors`로 이번 예측의 Tree SHAP 주요 영향 변수 세 개와 증가·감소 방향을 표시한다.
7. 연결 실패, 404, 422 응답은 사용자가 이해할 수 있는 문장으로 표시한다.

위험일 때는 "고교통량 위험"으로 표시하며 정체를 확정하지 않는다.

## 테스트

```bash
python -m pytest -q
python -m compileall -q frontend tests
```
