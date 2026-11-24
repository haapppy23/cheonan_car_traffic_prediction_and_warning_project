# 클라우드 배포 준비 상태

## 현재 상태

FastAPI와 Streamlit은 로컬에서 연동되며 `TRAFFIC_API_URL`로 프론트엔드가 호출할 API 주소를 바꿀 수 있다. 그러나 GitHub 저장소만 연결한 클라우드 배포는 현재 상태로 실행되지 않는다.

백엔드 실행에 필요한 다음 파일을 보안·용량·재현성 원칙에 따라 Git에서 제외했기 때문이다.

| 파일 | 로컬 크기 | 용도 |
|---|---:|---|
| `models/final_random_forest.joblib` | 약 35.4MB | 최종 Random Forest |
| `data/processed/selected_traffic_long.csv` | 약 82.4MB | 예측 직전 교통량 이력 |
| `data/processed/features/high_traffic_thresholds.csv` | 약 1.9KB | 접근로별 고교통량 기준 |

## 실제 배포에 필요한 결정

- FastAPI 실행 환경에 세 파일을 비공개 저장소나 영구 디스크로 전달
- 외부 공개가 허용되는 데이터 범위와 이용 조건 확인
- 배포된 FastAPI 주소를 Streamlit의 `TRAFFIC_API_URL`에 설정
- 배포 후 예시 입력의 로컬·클라우드 예측값 일치 확인

## 권장 구성

FastAPI는 모델과 자료 파일을 보관할 수 있는 서버에 배포하고 Streamlit Community Cloud에는 프론트엔드만 배포한다. 실제 자료와 모델을 공개 Git 저장소에 직접 커밋하지 않는다.

현재 단계에서는 계정과 비공개 파일 저장 위치가 정해지지 않아 외부 URL 생성까지 자동화하지 않는다.
