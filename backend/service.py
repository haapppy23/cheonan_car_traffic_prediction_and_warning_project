"""저장된 교통량 이력으로 모델 입력을 만들고 예측 결과를 계산한다."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import pandas as pd


REQUIRED_HISTORY_HOURS = 168
HISTORY_COLUMNS = ["timestamp", "intersection", "approach", "volume"]
THRESHOLD_COLUMNS = ["intersection", "approach", "high_traffic_threshold"]


class TrafficPredictor(Protocol):
    def predict(self, features: pd.DataFrame) -> np.ndarray: ...


class PredictionServiceError(ValueError):
    """예측 요청을 처리할 수 없을 때 발생한다."""


class LocationNotFoundError(PredictionServiceError):
    """교차로·접근로 조합이 저장된 자료에 없을 때 발생한다."""


class HistoryUnavailableError(PredictionServiceError):
    """예측 직전 168시간 이력이 완전하지 않을 때 발생한다."""


@dataclass(frozen=True)
class PredictionResult:
    intersection: str
    approach: str
    target_timestamp: pd.Timestamp
    predicted_volume: float
    high_traffic_threshold: float
    high_traffic_alert: bool
    risk_score: float


def _validate_required_columns(
    frame: pd.DataFrame, required: list[str], label: str
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise PredictionServiceError(f"{label}에 필요한 열이 없습니다: {missing}")


def prepare_history(frame: pd.DataFrame) -> pd.DataFrame:
    _validate_required_columns(frame, HISTORY_COLUMNS, "교통량 이력")
    result = frame[HISTORY_COLUMNS].copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    result["volume"] = pd.to_numeric(result["volume"], errors="coerce")
    if result.isna().any().any():
        raise PredictionServiceError("교통량 이력에 결측값이나 잘못된 값이 있습니다.")
    if result["volume"].lt(0).any():
        raise PredictionServiceError("교통량 이력에 음수 값이 있습니다.")
    if result.duplicated(["timestamp", "intersection", "approach"]).any():
        raise PredictionServiceError("교통량 이력에 중복 시각이 있습니다.")
    return result.sort_values(
        ["intersection", "approach", "timestamp"], kind="stable"
    ).reset_index(drop=True)


def prepare_thresholds(frame: pd.DataFrame) -> pd.DataFrame:
    _validate_required_columns(frame, THRESHOLD_COLUMNS, "고교통량 기준")
    result = frame[THRESHOLD_COLUMNS].copy()
    result["high_traffic_threshold"] = pd.to_numeric(
        result["high_traffic_threshold"], errors="coerce"
    )
    if result.isna().any().any():
        raise PredictionServiceError("고교통량 기준에 결측값이 있습니다.")
    if result["high_traffic_threshold"].le(0).any():
        raise PredictionServiceError("고교통량 기준은 0보다 커야 합니다.")
    if result.duplicated(["intersection", "approach"]).any():
        raise PredictionServiceError("교차로·접근로별 고교통량 기준이 중복됩니다.")
    return result


class PredictionService:
    """모델·이력·기준값을 한 번 적재하여 1시간 후 예측을 제공한다."""

    def __init__(
        self,
        model: TrafficPredictor,
        history: pd.DataFrame,
        thresholds: pd.DataFrame,
    ) -> None:
        prepared_history = prepare_history(history)
        prepared_thresholds = prepare_thresholds(thresholds)
        self.model = model
        self._history_by_location = {
            (str(intersection), str(approach)): pd.Series(
                group["volume"].to_numpy(dtype="float64"),
                index=pd.DatetimeIndex(group["timestamp"]),
                dtype="float64",
            )
            for (intersection, approach), group in prepared_history.groupby(
                ["intersection", "approach"], sort=False, observed=True
            )
        }
        self._threshold_by_location = {
            (str(row.intersection), str(row.approach)): float(
                row.high_traffic_threshold
            )
            for row in prepared_thresholds.itertuples(index=False)
        }

    @classmethod
    def from_files(
        cls,
        *,
        model_path: Path,
        history_path: Path,
        thresholds_path: Path,
    ) -> PredictionService:
        for path, label in (
            (model_path, "최종 모델"),
            (history_path, "교통량 이력"),
            (thresholds_path, "고교통량 기준"),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} 파일이 없습니다: {path}")
        model = joblib.load(model_path)
        history = pd.read_csv(
            history_path,
            usecols=HISTORY_COLUMNS,
            encoding="utf-8-sig",
        )
        thresholds = pd.read_csv(
            thresholds_path,
            usecols=THRESHOLD_COLUMNS,
            encoding="utf-8-sig",
        )
        return cls(model, history, thresholds)

    def _resolve_target_timestamp(
        self,
        series: pd.Series,
        target_timestamp: pd.Timestamp | None,
    ) -> pd.Timestamp:
        if target_timestamp is None:
            return pd.Timestamp(series.index.max()) + pd.Timedelta(hours=1)
        target = pd.Timestamp(target_timestamp)
        if target.tzinfo is not None:
            raise HistoryUnavailableError("예측 시각은 한국 현지시각으로 입력해야 합니다.")
        if target != target.floor("h"):
            raise HistoryUnavailableError("예측 시각은 정각이어야 합니다.")
        return target

    def build_features(
        self,
        intersection: str,
        approach: str,
        target_timestamp: pd.Timestamp | None = None,
    ) -> tuple[pd.DataFrame, pd.Timestamp, float]:
        key = (intersection, approach)
        series = self._history_by_location.get(key)
        threshold = self._threshold_by_location.get(key)
        if series is None or threshold is None:
            raise LocationNotFoundError(
                f"저장된 교차로·접근로가 아닙니다: {intersection} / {approach}"
            )

        target = self._resolve_target_timestamp(series, target_timestamp)
        prior = series.loc[series.index < target].tail(REQUIRED_HISTORY_HOURS)
        expected_index = pd.date_range(
            end=target - pd.Timedelta(hours=1),
            periods=REQUIRED_HISTORY_HOURS,
            freq="h",
        )
        if len(prior) != REQUIRED_HISTORY_HOURS or not prior.index.equals(
            expected_index
        ):
            raise HistoryUnavailableError(
                "예측 직전 168시간의 연속된 교통량 이력이 필요합니다."
            )

        hour = target.hour
        weekday = target.dayofweek
        last_six = prior.iloc[-6:].to_numpy(dtype="float64")
        feature_row = pd.DataFrame(
            [
                {
                    "target_month": target.month,
                    "target_is_weekend": weekday >= 5,
                    "hour_sin": np.sin(2 * np.pi * hour / 24),
                    "hour_cos": np.cos(2 * np.pi * hour / 24),
                    "weekday_sin": np.sin(2 * np.pi * weekday / 7),
                    "weekday_cos": np.cos(2 * np.pi * weekday / 7),
                    "lag_1": float(prior.iloc[-1]),
                    "lag_2": float(prior.iloc[-2]),
                    "lag_3": float(prior.iloc[-3]),
                    "lag_24": float(prior.iloc[-24]),
                    "lag_168": float(prior.iloc[-168]),
                    "rolling_mean_3": float(prior.iloc[-3:].mean()),
                    "rolling_mean_6": float(last_six.mean()),
                    "rolling_std_6": float(last_six.std(ddof=0)),
                    "intersection": intersection,
                    "approach": approach,
                }
            ]
        )
        return feature_row, target, threshold

    def predict(
        self,
        intersection: str,
        approach: str,
        target_timestamp: pd.Timestamp | None = None,
    ) -> PredictionResult:
        features, target, threshold = self.build_features(
            intersection, approach, target_timestamp
        )
        predicted_volume = float(self.model.predict(features)[0])
        if not np.isfinite(predicted_volume) or predicted_volume < 0:
            raise PredictionServiceError("모델이 유효하지 않은 교통량을 반환했습니다.")
        return PredictionResult(
            intersection=intersection,
            approach=approach,
            target_timestamp=target,
            predicted_volume=predicted_volume,
            high_traffic_threshold=threshold,
            high_traffic_alert=predicted_volume > threshold,
            risk_score=predicted_volume / threshold,
        )
    
    def get_locations(self) -> dict:
        """
        사용 가능한 교차로와 접근로 목록을 중복 없이 정렬하여 반환합니다.
        """
        if not hasattr(self, "_history_by_location") or not self._history_by_location:
            return {"locations": []}

        # 딕셔너리의 (intersection, approach) 키에서 추출
        locations_dict = {}
        for inter, app in self._history_by_location.keys():
            if inter not in locations_dict:
                locations_dict[inter] = set()
            locations_dict[inter].add(app)

        result = [
            {"intersection": inter, "approaches": sorted(list(apps))}
            for inter, apps in sorted(locations_dict.items())
        ]

        return {"locations": result}