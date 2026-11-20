# 김문일 Streamlit 작업용 AI 프롬프트

이나율의 백엔드 작업이 `main`에 반영된 것을 확인한 뒤 아래 내용을 AI에게 그대로 전달한다.

```text
천안시 스마트교차로 교통량 예측 프로젝트의 Streamlit 담당 작업을 수행한다.

저장소:
https://github.com/haapppy23/cheonan_car_traffic_prediction_and_warning_project

내 역할:
김문일, Streamlit 프론트엔드 담당

작업 전에 저장소 루트의 AGENTS.md, frontend/README.md, backend/README.md, docs/model_handoff.md, docs/team_handoff/README.md를 전부 읽어라. git status와 현재 브랜치를 확인하고, 이나율의 /locations 작업이 반영된 최신 main에서 feat/streamlit-ui 브랜치를 만들어라. 이미 같은 브랜치가 있으면 기존 작업을 보존하고 이어서 진행하라.

프론트엔드는 모델 파일을 직접 불러오거나 lag, 이동 평균, 경보 기준을 계산하면 안 된다. FastAPI만 호출한다.

이번 작업 범위:
- frontend/app.py에 단일 Streamlit 화면을 만든다.
- API 주소는 환경변수 TRAFFIC_API_URL을 사용하고 기본값은 http://127.0.0.1:8000으로 한다.
- GET /health로 연결 상태를 확인한다.
- GET /locations로 교차로와 접근로 선택 항목을 만든다.
- 사용자가 예측 버튼을 누르면 POST /predict를 호출한다.
- 예상 교통량, 고교통량 기준, high_traffic_alert와 risk_score를 읽기 쉽게 표시한다.
- 위험일 때는 고교통량 위험이라고 표시하고 정체 확정이라고 쓰지 않는다.
- API 연결 실패, 404, 422 응답을 사용자가 이해할 수 있는 문장으로 표시한다.
- API 호출 코드는 화면 코드와 분리해 테스트할 수 있게 만든다.
- 필요한 최소 의존성만 requirements.txt에 추가한다.
- frontend/README.md에 백엔드와 프론트엔드 실행 순서를 작성한다.

화면은 발표용으로 단정하게 만들되 지도, 로그인, 데이터베이스, 애니메이션, 과도한 CSS는 추가하지 마라.

검증:
- API 호출 함수의 정상 응답과 오류 응답 자동 테스트
- python -m pytest -q
- python -m compileall -q frontend tests
- git diff --check
- 실행파일이 있으면 FastAPI와 Streamlit을 함께 실행해 실제 예측 한 건을 확인한다.

완료 조건을 만족하면 변경 파일만 커밋하라. 커밋 메시지는 feat(frontend): add traffic prediction dashboard 로 한다. main에는 push하지 말고 feat/streamlit-ui 브랜치를 origin에 push하라.

마지막 보고에는 브랜치명, 커밋 해시, 테스트 개수, 실제 연동 확인 여부, 실행 명령과 남은 문제를 짧게 적어라.
```

AI가 작업을 끝냈다고 하면 다음 내용을 직접 확인한다.

- 본인 GitHub 브랜치에 커밋이 있는가
- 교차로를 바꾸면 접근로 목록도 맞게 바뀌는가
- 예측 버튼을 눌렀을 때 네 가지 결과가 표시되는가
- FastAPI가 꺼졌을 때 오류 문장이 표시되는가
