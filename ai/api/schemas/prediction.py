# api/schemas/prediction.py

"""Schemas de request/response para os endpoints de predição."""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ── Requests ──────────────────────────────────────────────────


class PredictRequest(BaseModel):
    """Request para predição (com ou sem filtro de veículos)."""
    target: str = Field(pattern=r"^(km|h)$", description="Métrica alvo: 'km' ou 'h'")
    vehicle_ids: Optional[List[int]] = Field(
        None, description="IDs dos veículos a predizer. Se None, prediz todos.",
    )


class PredictAndPersistRequest(BaseModel):
    """Request para predição e persistência de todos os veículos."""
    target: str = Field(pattern=r"^(km|h)$", description="Métrica alvo: 'km' ou 'h'")


# ── Response items ────────────────────────────────────────────


class DailyPredictionItem(BaseModel):
    veiculo_id: int
    data: date
    prediction: float


class HeadPredictionItem(BaseModel):
    veiculo_id: int
    head: int
    dt_inicio: date
    dt_fim: date
    prediction: float


class VehicleTypeProbabilities(BaseModel):
    veiculo_id: int
    probabilities: Dict[str, float]


class VehicleNotFound(BaseModel):
    veiculo_id: int
    reason: str


# ── Responses ─────────────────────────────────────────────────


class PredictResponse(BaseModel):
    """Resposta do endpoint predict (sem persistência)."""
    target: str
    predictions_daily: List[DailyPredictionItem]
    predictions_heads: List[HeadPredictionItem]
    type_probabilities: List[VehicleTypeProbabilities]
    not_found: List[VehicleNotFound] = []


class PredictAndPersistResponse(BaseModel):
    """Resultado interno do predict_and_persist (logado, não retornado ao cliente)."""
    target: str
    total_vehicles: int
    daily_rows_persisted: int
    head_rows_persisted: int


# ── Backtest ───────────────────────────────────────────────────────


class BacktestDailyItem(BaseModel):
    data: date
    actual: float
    predicted: float


class BacktestHeadItem(BaseModel):
    head: int
    dt_inicio: date
    dt_fim: date
    actual: float
    predicted: float


class BacktestResponse(BaseModel):
    """Comparação entre valores reais e preditos para um veículo."""
    veiculo_id: int
    target: str
    daily: List[BacktestDailyItem]
    heads: List[BacktestHeadItem]
