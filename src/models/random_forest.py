"""원-핫 인코딩과 비음수 보정을 포함한 Random Forest 교통량 모델."""

from __future__ import annotations

import os
import warnings

os.environ.setdefault(
    "LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) - 1))
)

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.models.ridge import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES


class RandomForestValidationError(ValueError):
    """Random Forest 입력 자료나 설정이 유효하지 않을 때 발생한다."""


def validate_model_features(frame: pd.DataFrame) -> None:
    missing = sorted(set(MODEL_FEATURES) - set(frame.columns))
    if missing:
        raise RandomForestValidationError(
            f"Random Forest 입력에 필요한 열이 없습니다: {missing}"
        )
    if frame[MODEL_FEATURES].isna().any().any():
        raise RandomForestValidationError("Random Forest 입력 변수에 결측값이 있습니다.")


def make_random_forest_pipeline(
    *,
    n_estimators: int = 12,
    max_depth: int = 18,
    min_samples_leaf: int = 10,
) -> Pipeline:
    if n_estimators <= 0 or max_depth <= 0 or min_samples_leaf <= 0:
        raise RandomForestValidationError("Random Forest 설정값이 유효하지 않습니다.")
    preprocessing = ColumnTransformer(
        [
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype=np.float32,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        sparse_threshold=0.0,
    )
    regressor = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=0.7,
        n_jobs=1,
        random_state=42,
    )
    return Pipeline(
        [("preprocessing", preprocessing), ("random_forest", regressor)]
    )


class RandomForestTrafficModel:
    """고정 Random Forest 파이프라인과 비음수 예측 규칙을 저장한다."""

    def __init__(
        self,
        *,
        n_estimators: int = 12,
        max_depth: int = 18,
        min_samples_leaf: int = 10,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.pipeline = make_random_forest_pipeline(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )

    def fit(self, features: pd.DataFrame, target: pd.Series) -> RandomForestTrafficModel:
        validate_model_features(features)
        target_values = pd.to_numeric(target, errors="coerce")
        if target_values.isna().any():
            raise RandomForestValidationError("Random Forest 정답에 결측값이 있습니다.")
        if len(features) != len(target_values):
            raise RandomForestValidationError(
                "Random Forest 입력과 정답의 행 수가 다릅니다."
            )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Could not find the number of physical cores",
                category=UserWarning,
                module=r"joblib\.externals\.loky\.backend\.context",
            )
            self.pipeline.fit(features[MODEL_FEATURES], target_values)
        return self

    def predict_raw(self, features: pd.DataFrame) -> np.ndarray:
        validate_model_features(features)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Could not find the number of physical cores",
                category=UserWarning,
                module=r"joblib\.externals\.loky\.backend\.context",
            )
            prediction = self.pipeline.predict(features[MODEL_FEATURES])
        return np.asarray(prediction, dtype="float64")

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.maximum(self.predict_raw(features), 0.0)
