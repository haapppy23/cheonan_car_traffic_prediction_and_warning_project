# models

단순 기준, Ridge와 트리 기반 후보 모델의 코드를 둔다.

- `baselines.py`: 1시간 전, 전날 같은 시간과 전주 같은 시간 교통량을 사용하는 단순 기준
- `ridge.py`: 표준화·범주형 인코딩·비음수 보정을 포함한 Ridge 모델
- `train_ridge.py`: 학습 구간으로 Ridge를 학습하고 검증 구간에서 단순 기준과 비교

```bash
python src/models/train_ridge.py
```

트리 기반 후보 모델은 Ridge 검증이 끝난 뒤 같은 분할에 추가한다.
