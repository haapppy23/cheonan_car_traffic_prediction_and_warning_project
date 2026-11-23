"""공휴일·기상 변수를 추가한 보충 실험용 Random Forest 모델."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.features.build_external_features import WEATHER_COLUMNS
from src.models.ridge import CATEGORICAL_FEATURES, NUMERIC_FEATURES


EXTERNAL_NUMERIC_FEATURES = [
    *NUMERIC_FEATURES,
    "target_is_holiday",
    *WEATHER_COLUMNS,
]
EXTERNAL_MODEL_FEATURES = [*EXTERNAL_NUMERIC_FEATURES, *CATEGORICAL_FEATURES]


class ExternalRandomForestError(ValueError):
    """외부 변수 Random Forest의 입력 또는 설정이 유효하지 않을 때 발생한다."""


def prepare_external_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(EXTERNAL_MODEL_FEATURES) - set(frame.columns))
    if missing:
        raise ExternalRandomForestError(f"외부 모델 입력에 필요한 열이 없습니다: {missing}")
    result = frame[EXTERNAL_MODEL_FEATURES].copy()
    result["target_is_holiday"] = result["target_is_holiday"].astype("int8")
    result[EXTERNAL_NUMERIC_FEATURES] = result[EXTERNAL_NUMERIC_FEATURES].astype(
        "float32"
    )
    if result.isna().any().any():
        raise ExternalRandomForestError("외부 모델 입력 변수에 결측값이 있습니다.")
    return result


def make_external_pipeline() -> Pipeline:
    preprocessing = ColumnTransformer(
        [
            ("numeric", "passthrough", EXTERNAL_NUMERIC_FEATURES),
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
        n_estimators=12,
        max_depth=18,
        min_samples_leaf=10,
        max_features=0.7,
        n_jobs=1,
        random_state=42,
    )
    return Pipeline(
        [("preprocessing", preprocessing), ("random_forest", regressor)]
    )


class ExternalRandomForestModel:
    """기존 Random Forest 설정을 고정하고 외부 변수만 추가한다."""

    def __init__(self) -> None:
        self.pipeline = make_external_pipeline()

    def fit(
        self, features: pd.DataFrame, target: pd.Series
    ) -> ExternalRandomForestModel:
        prepared = prepare_external_features(features)
        target_values = pd.to_numeric(target, errors="coerce")
        if target_values.isna().any() or len(prepared) != len(target_values):
            raise ExternalRandomForestError("외부 모델 정답이 유효하지 않습니다.")
        self.pipeline.fit(prepared, target_values)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        prediction = self.pipeline.predict(prepare_external_features(features))
        return np.maximum(np.asarray(prediction, dtype="float64"), 0.0)
