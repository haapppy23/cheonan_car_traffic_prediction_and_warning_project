# 최종 모델 전달 규격

## 모델

- 최종 모델: Random Forest
- 로컬 파일: `models/final_random_forest.joblib`
- 재생성 명령: `python src/models/finalize_random_forest.py`
- 모델 파일은 용량과 재생성 가능성 때문에 Git에 올리지 않는다.

## 입력

한 행은 예측하려는 교차로·접근로·시각이다. 모델 입력 열은 다음과 같다.

`target_month`, `target_is_weekend`, `hour_sin`, `hour_cos`, `weekday_sin`, `weekday_cos`, `lag_1`, `lag_2`, `lag_3`, `lag_24`, `lag_168`, `rolling_mean_3`, `rolling_mean_6`, `rolling_std_6`, `intersection`, `approach`

사용자가 이 숫자를 직접 입력하는 구조가 아니다. 백엔드는 저장된 최근 교통량에서 `src/features/build_features.py`와 같은 규칙으로 입력을 만든다. 168시간 이력이 없거나 센서 품질 플래그가 유효하지 않으면 예측 요청을 거부하거나 데이터 부족 상태를 반환한다.

## 출력

- `predicted_volume`: 1시간 후 예상 교통량, 0 이상의 실수
- `high_traffic_threshold`: 선택한 접근로의 학습 구간 90백분위수
- `high_traffic_alert`: `predicted_volume > high_traffic_threshold`
- `risk_score`: `predicted_volume / high_traffic_threshold`

접근로별 기준값은 `data/processed/features/high_traffic_thresholds.csv`에서 읽는다.

## API 담당자 주의사항

- 모델과 기준값을 서버 시작 시 한 번만 불러온다.
- 교차로·접근로 이름은 학습 데이터의 문자열을 그대로 사용한다.
- 시험 결과를 서비스 화면에서 실시간 정확도로 표현하지 않는다.
- 속도 자료가 없으므로 출력 명칭은 `정체 확정`이 아니라 `고교통량 위험`으로 사용한다.
