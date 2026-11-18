"""표준화·범주형 인코딩과 음수 보정을 포함한 Ridge 교통량 모델."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "target_month",
    "target_is_weekend",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_24",
    "lag_168",
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_std_6",
]
CATEGORICAL_FEATURES = ["intersection", "approach"]
MODEL_FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]


class RidgeValidationError(ValueError):
    """Ridge 입력 데이터가 학습 또는 예측 조건을 만족하지 않을 때 발생한다."""


def validate_model_features(frame: pd.DataFrame) -> None:
    missing = sorted(set(MODEL_FEATURES) - set(frame.columns))
    if missing:
        raise RidgeValidationError(f"Ridge 입력에 필요한 열이 없습니다: {missing}")
    if frame[MODEL_FEATURES].isna().any().any():
        raise RidgeValidationError("Ridge 입력 변수에 결측값이 있습니다.")


def make_ridge_pipeline(alpha: float = 1.0) -> Pipeline:
    if alpha < 0:
        raise RidgeValidationError("Ridge alpha는 0 이상이어야 합니다.")
    preprocessing = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore", sparse_output=True, dtype=np.float32
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        sparse_threshold=1.0,
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("ridge", Ridge(alpha=alpha, solver="lsqr")),
        ]
    )


class RidgeTrafficModel:
    """학습 파이프라인과 비음수 교통량 예측 규칙을 함께 저장한다."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.pipeline = make_ridge_pipeline(alpha)

    def fit(self, features: pd.DataFrame, target: pd.Series) -> RidgeTrafficModel:
        validate_model_features(features)
        target_values = pd.to_numeric(target, errors="coerce")
        if target_values.isna().any():
            raise RidgeValidationError("Ridge 정답에 결측값이 있습니다.")
        if len(features) != len(target_values):
            raise RidgeValidationError("Ridge 입력과 정답의 행 수가 다릅니다.")
        self.pipeline.fit(features[MODEL_FEATURES], target_values)
        return self

    def predict_raw(self, features: pd.DataFrame) -> np.ndarray:
        validate_model_features(features)
        return np.asarray(
            self.pipeline.predict(features[MODEL_FEATURES]), dtype="float64"
        )

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.maximum(self.predict_raw(features), 0.0)
