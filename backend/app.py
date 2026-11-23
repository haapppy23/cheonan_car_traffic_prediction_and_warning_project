"""천안시 스마트교차로 교통량 예측 FastAPI 애플리케이션."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.service import (
    HistoryUnavailableError,
    LocationNotFoundError,
    PredictionService,
    PredictionServiceError,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "final_random_forest.joblib"
DEFAULT_HISTORY_PATH = (
    PROJECT_ROOT / "data" / "processed" / "selected_traffic_long.csv"
)
DEFAULT_THRESHOLDS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "high_traffic_thresholds.csv"
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    intersection: str = Field(min_length=1, description="교차로명")
    approach: str = Field(min_length=1, description="접근로명")
    target_timestamp: datetime | None = Field(
        default=None,
        description="예측 대상 정각. 생략하면 저장된 마지막 관측의 1시간 후",
        examples=["2026-08-31T23:00:00"],
    )

    @field_validator("target_timestamp")
    @classmethod
    def require_local_naive_time(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is not None:
            raise ValueError("예측 시각은 시간대 표기 없는 한국 현지시각이어야 합니다.")
        return value


class PredictionResponse(BaseModel):
    intersection: str
    approach: str
    target_timestamp: datetime
    predicted_volume: float
    high_traffic_threshold: float
    high_traffic_alert: bool
    risk_score: float


class HealthResponse(BaseModel):
    status: str
    model: str


def create_app(service: PredictionService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.prediction_service = service or PredictionService.from_files(
            model_path=DEFAULT_MODEL_PATH,
            history_path=DEFAULT_HISTORY_PATH,
            thresholds_path=DEFAULT_THRESHOLDS_PATH,
        )
        yield

    application = FastAPI(
        title="천안시 스마트교차로 교통량 예측 API",
        version="1.0.0",
        lifespan=lifespan,
    )
    if service is not None:
        application.state.prediction_service = service

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", model="random_forest")

    @application.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
        prediction_service: PredictionService = request.app.state.prediction_service
        try:
            result = prediction_service.predict(
                payload.intersection,
                payload.approach,
                payload.target_timestamp,
            )
        except LocationNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (HistoryUnavailableError, PredictionServiceError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return PredictionResponse.model_validate(result, from_attributes=True)

    return application


app = create_app()

app = create_app()


@app.get("/locations")
def get_locations():
    """ 지원하는 교차로 및 접근로 목록을 반환 """
    return service.get_locations()