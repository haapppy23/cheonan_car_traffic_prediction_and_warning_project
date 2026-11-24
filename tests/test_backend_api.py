from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI, HTTPException

from backend.app import PredictionRequest, create_app
from backend.service import HistoryUnavailableError, PredictionService


class FixedModel:
    def __init__(self, prediction: float) -> None:
        self.prediction = prediction
        self.last_features: pd.DataFrame | None = None

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        self.last_features = features.copy()
        return np.array([self.prediction], dtype="float64")

    def explain(self, features: pd.DataFrame, *, top_k: int = 3):
        return [
            {
                "feature": "lag_1",
                "label": "1시간 전 교통량",
                "contribution": 24.5,
                "effect": "increase",
            }
        ][:top_k]


def make_service(hours: int = 168, prediction: float = 200.0) -> PredictionService:
    history = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=hours, freq="h"),
            "intersection": "테스트4",
            "approach": "테스트4_동",
            "volume": np.arange(1, hours + 1, dtype="float64"),
        }
    )
    thresholds = pd.DataFrame(
        {
            "intersection": ["테스트4"],
            "approach": ["테스트4_동"],
            "high_traffic_threshold": [150.0],
        }
    )
    return PredictionService(FixedModel(prediction), history, thresholds)


def get_endpoint(app: FastAPI, path: str, method: str):
    return next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == path
        and method in getattr(route, "methods", set())
    )


def test_build_features_uses_only_prior_168_hours() -> None:
    service = make_service()

    features, target, threshold = service.build_features(
        "테스트4", "테스트4_동"
    )

    assert target == pd.Timestamp("2026-01-08 00:00:00")
    assert threshold == 150.0
    assert features.loc[0, "lag_1"] == 168.0
    assert features.loc[0, "lag_2"] == 167.0
    assert features.loc[0, "lag_3"] == 166.0
    assert features.loc[0, "lag_24"] == 145.0
    assert features.loc[0, "lag_168"] == 1.0
    assert features.loc[0, "rolling_mean_3"] == 167.0
    assert features.loc[0, "rolling_mean_6"] == 165.5
    assert features.loc[0, "rolling_std_6"] == pytest.approx(
        np.std([163, 164, 165, 166, 167, 168], ddof=0)
    )


def test_predict_endpoint_returns_volume_and_alert() -> None:
    app = create_app(make_service())
    endpoint = get_endpoint(app, "/predict", "POST")

    response = endpoint(
        PredictionRequest(intersection="테스트4", approach="테스트4_동"),
        SimpleNamespace(app=app),
    )

    assert response.model_dump(mode="json") == {
        "intersection": "테스트4",
        "approach": "테스트4_동",
        "target_timestamp": "2026-01-08T00:00:00",
        "predicted_volume": 200.0,
        "high_traffic_threshold": 150.0,
        "high_traffic_alert": True,
        "risk_score": pytest.approx(4 / 3),
        "major_factors": [
            {
                "feature": "lag_1",
                "label": "1시간 전 교통량",
                "contribution": 24.5,
                "effect": "increase",
            }
        ],
    }


def test_predict_endpoint_rejects_unknown_location() -> None:
    app = create_app(make_service())
    endpoint = get_endpoint(app, "/predict", "POST")

    with pytest.raises(HTTPException) as error:
        endpoint(
            PredictionRequest(intersection="없는4", approach="없는4_동"),
            SimpleNamespace(app=app),
        )

    assert error.value.status_code == 404
    assert "저장된 교차로" in error.value.detail


def test_prediction_requires_continuous_168_hour_history() -> None:
    service = make_service(hours=167)

    with pytest.raises(HistoryUnavailableError, match="168시간"):
        service.predict("테스트4", "테스트4_동")


def test_health_endpoint_reports_loaded_model() -> None:
    app = create_app(make_service())
    endpoint = get_endpoint(app, "/health", "GET")
    response = endpoint()

    assert response.model_dump() == {"status": "ok", "model": "random_forest"}

def test_get_locations_service():
    service = make_service()
    locations_data = service.get_locations()

    assert "locations" in locations_data
    assert isinstance(locations_data["locations"], list)
    assert len(locations_data["locations"]) > 0
    assert locations_data["locations"][0]["intersection"] == "테스트4"
    assert locations_data["locations"][0]["approaches"] == ["테스트4_동"]


def test_get_locations_endpoint():
    app = create_app(make_service())
    endpoint = get_endpoint(app, "/locations", "GET")

    response = endpoint(SimpleNamespace(app=app))

    assert "locations" in response
    assert isinstance(response["locations"], list)
    assert response == {
        "locations": [
            {
                "intersection": "테스트4",
                "approaches": ["테스트4_동"],
            }
        ]
    }
