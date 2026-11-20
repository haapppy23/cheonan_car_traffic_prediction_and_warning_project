# models

단순 기준, Ridge와 트리 기반 후보 모델의 코드를 둔다.

- `baselines.py`: 1시간 전, 전날 같은 시간과 전주 같은 시간 교통량을 사용하는 단순 기준
- `ridge.py`: 표준화·범주형 인코딩·비음수 보정을 포함한 Ridge 모델
- `train_ridge.py`: 학습 구간으로 Ridge를 학습하고 검증 구간에서 단순 기준과 비교
- `hist_gradient_boosting.py`: 범주형 인코딩과 비음수 보정을 포함한 부스팅 모델
- `train_hist_gradient_boosting.py`: 고정 부스팅을 Ridge·단순 기준과 비교
- `random_forest.py`: 범주형 정수 인코딩과 비음수 보정을 포함한 Random Forest
- `train_random_forest.py`: 고정 Random Forest를 기존 후보와 비교
- `finalize_random_forest.py`: 선정 모델을 학습+검증으로 재학습하고 시험 구간을 1회 평가

```bash
python src/models/train_ridge.py
python src/models/train_hist_gradient_boosting.py
python src/models/train_random_forest.py
python src/models/finalize_random_forest.py
```

다른 트리 기반 후보도 동일한 학습·검증 분할에 추가한다.
