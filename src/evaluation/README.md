# evaluation

모델별로 같은 정의의 지표를 적용하고 시험 구간 사용을 통제한다.

- `metrics.py`: MAE, WAPE와 RMSE 계산
- `run_baselines.py`: 검증 구간의 지속성·일간·주간 단순 기준 비교
- `run_high_traffic_alerts.py`: 최고 회귀 모델의 Precision·Recall·F1·PR-AUC 계산

```bash
python src/evaluation/run_baselines.py
python src/evaluation/run_high_traffic_alerts.py
```

세부 시간대·교차로별 오류 분석은 후보 모델 비교가 끝난 뒤 추가한다.
