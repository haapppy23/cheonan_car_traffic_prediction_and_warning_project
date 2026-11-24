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


FEATURE_LABELS = {
    "target_month": "예측 월",
    "target_is_weekend": "주말 여부",
    "hour_sin": "예측 시각(sin)",
    "hour_cos": "예측 시각(cos)",
    "weekday_sin": "요일(sin)",
    "weekday_cos": "요일(cos)",
    "lag_1": "1시간 전 교통량",
    "lag_2": "2시간 전 교통량",
    "lag_3": "3시간 전 교통량",
    "lag_24": "전날 같은 시각 교통량",
    "lag_168": "전주 같은 시각 교통량",
    "rolling_mean_3": "최근 3시간 평균",
    "rolling_mean_6": "최근 6시간 평균",
    "rolling_std_6": "최근 6시간 변동성",
    "intersection": "교차로",
    "approach": "접근로",
}


def clean_feature_names(raw_names: list[str] | np.ndarray) -> list[str]:
    """ColumnTransformer 접두사를 원래 입력 열 이름으로 바꾼다."""
    cleaned: list[str] = []
    for raw_name in raw_names:
        name = str(raw_name)
        if "__" in name:
            name = name.split("__", maxsplit=1)[1]
        cleaned.append(name)
    return cleaned


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

    def explain(
        self,
        features: pd.DataFrame,
        *,
        top_k: int = 3,
    ) -> list[dict[str, str | float]]:
        """한 행의 예측에 대한 상위 Tree SHAP 기여 변수를 반환한다."""
        if top_k <= 0:
            raise RandomForestValidationError("SHAP 표시 변수 수는 1 이상이어야 합니다.")
        validate_model_features(features)
        if len(features) != 1:
            raise RandomForestValidationError("서비스 SHAP 설명은 한 행씩 계산합니다.")

        try:
            preprocessing = self.pipeline.named_steps["preprocessing"]
            regressor = self.pipeline.named_steps["random_forest"]
        except KeyError as exc:
            raise RandomForestValidationError(
                "Random Forest 파이프라인 구성이 예상과 다릅니다."
            ) from exc

        import shap

        explainer = getattr(self, "_shap_explainer", None)
        if explainer is None:
            explainer = shap.TreeExplainer(regressor)
            self._shap_explainer = explainer

        transformed = preprocessing.transform(features[MODEL_FEATURES])
        explanation = explainer(transformed, check_additivity=False)
        values = np.asarray(explanation.values, dtype="float64")
        if values.ndim != 2 or values.shape[0] != 1:
            raise RandomForestValidationError("SHAP 설명 결과의 크기가 올바르지 않습니다.")

        feature_names = clean_feature_names(preprocessing.get_feature_names_out())
        if values.shape[1] != len(feature_names):
            raise RandomForestValidationError(
                "SHAP 값과 입력 변수 이름의 크기가 맞지 않습니다."
            )

        ranked = sorted(
            zip(feature_names, values[0], strict=True),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )[:top_k]
        return [
            {
                "feature": name,
                "label": FEATURE_LABELS.get(name, name),
                "contribution": float(value),
                "effect": "increase" if value >= 0 else "decrease",
            }
            for name, value in ranked
        ]
