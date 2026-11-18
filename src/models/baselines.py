"""학습 없이 과거 관측값을 사용하는 단순 기준 모델."""

from __future__ import annotations

import pandas as pd


BASELINES = {
    "persistence": "lag_1",
    "daily_seasonal": "lag_24",
    "weekly_seasonal": "lag_168",
}

BASELINE_LABELS = {
    "persistence": "지속성",
    "daily_seasonal": "전날 같은 시간",
    "weekly_seasonal": "전주 같은 시간",
}


class BaselineValidationError(ValueError):
    """단순 기준 입력 자료가 평가 조건을 만족하지 않을 때 발생한다."""


def get_baseline_predictions(frame: pd.DataFrame) -> dict[str, pd.Series]:
    missing = sorted(set(BASELINES.values()) - set(frame.columns))
    if missing:
        raise BaselineValidationError(f"단순 기준에 필요한 열이 없습니다: {missing}")
    return {name: frame[column] for name, column in BASELINES.items()}
