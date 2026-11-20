# 이나율 FastAPI 작업용 AI 프롬프트

저장소를 최신 상태로 받은 뒤 아래 내용을 AI에게 그대로 전달한다.

```text
천안시 스마트교차로 교통량 예측 프로젝트의 FastAPI 담당 작업을 수행한다.

저장소:
https://github.com/haapppy23/cheonan_car_traffic_prediction_and_warning_project

내 역할:
이나율, FastAPI 백엔드 담당

이 요청은 방법 설명이나 계획 작성 요청이 아니라 저장소에서 실제 작업을 끝내라는 요청이다. 계획만 제시하고 멈추지 말고 명령 실행, 파일 수정, 검증, 커밋과 본인 브랜치 push까지 직접 진행하라. 범위 안의 안전한 결정은 저장소 근거로 판단하고 불필요한 확인 질문을 하지 마라. 명령이 실패하면 원인을 확인하고 안전한 대안을 시도한 뒤 계속 진행하라.

GitHub 로그인·권한 승인처럼 사용자가 직접 해야 하는 조치만 짧게 요청할 수 있다. 실행파일이 없어도 코드와 합성 테스트 작업은 중단하지 않는다. 원격 브랜치에서 최종 커밋이 확인되기 전에는 완료했다고 보고하지 마라.

작업 전에 저장소 루트의 AGENTS.md, backend/README.md, docs/model_handoff.md, docs/team_handoff/README.md를 전부 읽어라. git status와 현재 브랜치를 확인하고, 최신 main에서 feat/backend-locations 브랜치를 만들어라. 이미 같은 브랜치가 있으면 기존 작업을 보존하고 이어서 진행하라.

현재 /health와 /predict, 모델 로드, 최근 168시간 입력 변수 생성은 구현되어 있다. 기존 기능을 다시 만들거나 모델을 재학습하지 마라.

이번 작업 범위:
- PredictionService에 사용 가능한 교차로와 접근로 목록을 반환하는 공개 기능을 추가한다.
- GET /locations 엔드포인트를 추가한다.
- 응답 형식은 아래 계약을 사용한다.
  {"locations":[{"intersection":"도로원점3","approaches":["도로원점3_남","도로원점3_동"]}]}
- 교차로와 접근로는 중복 없이 이름순으로 정렬한다.
- 기존 /health와 /predict 요청·응답을 변경하지 않는다.
- 서비스 계층과 API 계층 자동 테스트를 추가한다.
- backend/README.md에 /locations 사용법을 추가한다.

검증:
- python -m pytest -q
- python -m compileall -q backend tests
- git diff --check
- 실행파일이 있으면 Uvicorn을 띄워 /health, /locations, /predict를 실제 HTTP로 확인한다.

완료 조건을 만족하면 변경 파일만 커밋하라. 커밋 메시지는 feat(api): list traffic locations 로 한다. main에는 push하지 말고 feat/backend-locations 브랜치를 origin에 push하라.

push 후 원격 브랜치의 커밋 해시가 로컬 HEAD와 같은지 확인하라. 마지막 보고에는 브랜치명, 커밋 해시, 테스트 개수, 실제 HTTP 확인 여부, 남은 문제를 짧게 적어라. 실행파일이 없더라도 코드와 합성 테스트는 계속 진행하고 실제 HTTP 확인만 미완료로 보고하라.
```

AI가 작업을 끝냈다고 하면 다음 내용을 직접 확인한다.

- 본인 GitHub 브랜치에 커밋이 있는가
- `python -m pytest -q`가 통과하는가
- `http://127.0.0.1:8000/docs`에 `/locations`가 보이는가
- 기존 `/predict`가 여전히 동작하는가
