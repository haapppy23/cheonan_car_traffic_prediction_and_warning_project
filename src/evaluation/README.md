# evaluation

모델별로 같은 정의의 지표를 적용하고 시험 구간 사용을 통제한다.

- `metrics.py`: MAE, WAPE와 RMSE 계산
- `run_baselines.py`: 검증 구간의 지속성·일간·주간 단순 기준 비교

```bash
python src/evaluation/run_baselines.py
```

고교통량 경보 지표와 세부 오류 분석은 후보 모델 단계에서 추가한다.
