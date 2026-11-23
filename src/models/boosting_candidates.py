"""XGBoost와 CatBoost 교통량 후보 모델."""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor

from src.models.ridge import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES


class BoostingCandidateValidationError(ValueError):
    """추가 부스팅 후보의 입력이나 설정이 유효하지 않을 때 발생한다."""


def validate_model_features(frame: pd.DataFrame, *, label: str) -> None:
    missing = sorted(set(MODEL_FEATURES) - set(frame.columns))
    if missing:
        raise BoostingCandidateValidationError(
            f"{label} 입력에 필요한 열이 없습니다: {missing}"
        )
    if frame[MODEL_FEATURES].isna().any().any():
        raise BoostingCandidateValidationError(f"{label} 입력 변수에 결측값이 있습니다.")


def validate_target(features: pd.DataFrame, target: pd.Series, *, label: str) -> pd.Series:
    target_values = pd.to_numeric(target, errors="coerce")
    if target_values.isna().any():
        raise BoostingCandidateValidationError(f"{label} 정답에 결측값이 있습니다.")
    if len(features) != len(target_values):
        raise BoostingCandidateValidationError(
            f"{label} 입력과 정답의 행 수가 다릅니다."
        )
    return target_values


class XGBoostTrafficModel:
    """정수 범주 인코딩과 비음수 보정을 포함한 XGBoost 모델."""

    def __init__(
        self,
        *,
        n_estimators: int = 240,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        min_child_weight: float = 10.0,
        subsample: float = 0.9,
        colsample_bytree: float = 0.8,
        n_jobs: int = 4,
    ) -> None:
        if (
            n_estimators <= 0
            or max_depth <= 0
            or learning_rate <= 0
            or min_child_weight < 0
            or not 0 < subsample <= 1
            or not 0 < colsample_bytree <= 1
        ):
            raise BoostingCandidateValidationError("XGBoost 설정값이 유효하지 않습니다.")
        self.params = {
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth),
            "learning_rate": float(learning_rate),
            "min_child_weight": float(min_child_weight),
            "subsample": float(subsample),
            "colsample_bytree": float(colsample_bytree),
        }
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
        regressor = XGBRegressor(
            **self.params,
            objective="reg:squarederror",
            tree_method="hist",
            reg_lambda=1.0,
            random_state=42,
            n_jobs=n_jobs,
        )
        self.pipeline = Pipeline(
            [("preprocessing", preprocessing), ("xgboost", regressor)]
        )

    def fit(self, features: pd.DataFrame, target: pd.Series) -> XGBoostTrafficModel:
        validate_model_features(features, label="XGBoost")
        target_values = validate_target(features, target, label="XGBoost")
        self.pipeline.fit(features[MODEL_FEATURES], target_values)
        return self

    def predict_raw(self, features: pd.DataFrame) -> np.ndarray:
        validate_model_features(features, label="XGBoost")
        return np.asarray(
            self.pipeline.predict(features[MODEL_FEATURES]), dtype="float64"
        )

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.maximum(self.predict_raw(features), 0.0)


class CatBoostTrafficModel:
    """문자 범주를 CatBoost에 직접 전달하고 비음수 보정을 적용한다."""

    def __init__(
        self,
        *,
        iterations: int = 300,
        depth: int = 8,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 5.0,
        random_strength: float = 1.0,
        thread_count: int = 4,
    ) -> None:
        if (
            iterations <= 0
            or depth <= 0
            or learning_rate <= 0
            or l2_leaf_reg < 0
            or random_strength < 0
        ):
            raise BoostingCandidateValidationError("CatBoost 설정값이 유효하지 않습니다.")
        self.params = {
            "iterations": int(iterations),
            "depth": int(depth),
            "learning_rate": float(learning_rate),
            "l2_leaf_reg": float(l2_leaf_reg),
            "random_strength": float(random_strength),
        }
        self.model = CatBoostRegressor(
            **self.params,
            loss_function="RMSE",
            random_seed=42,
            thread_count=thread_count,
            allow_writing_files=False,
            verbose=False,
        )

    @staticmethod
    def _prepare(features: pd.DataFrame) -> pd.DataFrame:
        prepared = features[MODEL_FEATURES].copy()
        prepared[NUMERIC_FEATURES] = prepared[NUMERIC_FEATURES].astype("float32")
        for column in CATEGORICAL_FEATURES:
            prepared[column] = prepared[column].astype("string").astype(str)
        return prepared

    def fit(self, features: pd.DataFrame, target: pd.Series) -> CatBoostTrafficModel:
        validate_model_features(features, label="CatBoost")
        target_values = validate_target(features, target, label="CatBoost")
        self.model.fit(
            self._prepare(features),
            target_values,
            cat_features=CATEGORICAL_FEATURES,
        )
        return self

    def predict_raw(self, features: pd.DataFrame) -> np.ndarray:
        validate_model_features(features, label="CatBoost")
        return np.asarray(self.model.predict(self._prepare(features)), dtype="float64")

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.maximum(self.predict_raw(features), 0.0)
