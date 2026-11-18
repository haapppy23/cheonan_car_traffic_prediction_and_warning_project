"""범주형 인코딩과 비음수 보정을 포함한 HistGradientBoosting 모델."""

from __future__ import annotations

import os
import warnings

os.environ.setdefault(
    "LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) - 1))
)

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.models.ridge import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES


class HistGradientBoostingValidationError(ValueError):
    """부스팅 입력 데이터나 설정이 올바르지 않을 때 발생한다."""


def validate_model_features(frame: pd.DataFrame) -> None:
    missing = sorted(set(MODEL_FEATURES) - set(frame.columns))
    if missing:
        raise HistGradientBoostingValidationError(
            f"HistGradientBoosting 입력에 필요한 열이 없습니다: {missing}"
        )
    if frame[MODEL_FEATURES].isna().any().any():
        raise HistGradientBoostingValidationError(
            "HistGradientBoosting 입력 변수에 결측값이 있습니다."
        )


def make_hist_gradient_boosting_pipeline(
    *,
    max_iter: int = 100,
    learning_rate: float = 0.1,
    max_leaf_nodes: int = 31,
) -> Pipeline:
    if max_iter <= 0 or learning_rate <= 0 or max_leaf_nodes < 2:
        raise HistGradientBoostingValidationError("부스팅 설정값이 유효하지 않습니다.")
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
    categorical_mask = [False] * len(NUMERIC_FEATURES) + [True] * len(
        CATEGORICAL_FEATURES
    )
    regressor = HistGradientBoostingRegressor(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        l2_regularization=1.0,
        categorical_features=categorical_mask,
        early_stopping=False,
        random_state=42,
    )
    return Pipeline(
        [("preprocessing", preprocessing), ("hist_gradient_boosting", regressor)]
    )


class HistGradientBoostingTrafficModel:
    """고정 설정 부스팅 파이프라인과 비음수 예측 규칙을 함께 저장한다."""

    def __init__(
        self,
        *,
        max_iter: int = 100,
        learning_rate: float = 0.1,
        max_leaf_nodes: int = 31,
    ) -> None:
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.pipeline = make_hist_gradient_boosting_pipeline(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
        )

    def fit(
        self, features: pd.DataFrame, target: pd.Series
    ) -> HistGradientBoostingTrafficModel:
        validate_model_features(features)
        target_values = pd.to_numeric(target, errors="coerce")
        if target_values.isna().any():
            raise HistGradientBoostingValidationError("부스팅 정답에 결측값이 있습니다.")
        if len(features) != len(target_values):
            raise HistGradientBoostingValidationError(
                "부스팅 입력과 정답의 행 수가 다릅니다."
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
