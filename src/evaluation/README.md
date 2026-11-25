# evaluation

모델별로 같은 정의의 지표를 적용하고 시험 구간 사용을 통제한다.

- `metrics.py`: MAE, WAPE와 RMSE 계산
- `run_baselines.py`: 검증 구간의 지속성·일간·주간 단순 기준 비교
- `run_high_traffic_alerts.py`: 최고 회귀 모델의 Precision·Recall·F1·PR-AUC 계산
- `run_error_analysis.py`: 교차로·시간대·평일/주말·강수 여부 오차와 놓친 경보 분석
- `run_shap_analysis.py`: 최종 Random Forest의 검증 표본 SHAP 입력 변수 분석

```bash
python src/evaluation/run_baselines.py
python src/evaluation/run_high_traffic_alerts.py
python src/evaluation/run_error_analysis.py
python src/evaluation/run_shap_analysis.py
```

강수 여부별 오류까지 재생성할 때는 외부 입력 자료와 학습 구간 Random Forest를 사용한다.

```bash
python src/evaluation/run_error_analysis.py --input data/processed/features/model_features_external.parquet --model models/random_forest.joblib --model-name random_forest --output-dir data/processed/evaluation/random_forest_error_analysis --report docs/random_forest_error_analysis.md
```

시험 구간은 최종 모델과 설정이 확정되기 전까지 분석하지 않는다.
