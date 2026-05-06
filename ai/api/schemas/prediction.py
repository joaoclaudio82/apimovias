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


# ── Response items ────────────────────────────────────────────


class DailyPredictionItem(BaseModel):
    veiculo_id: int
    data: date
    prediction: float


class HeadPredictionItem(BaseModel):
    veiculo_id: int
    dt_inicio: date
    dt_fim: date
    prediction: float


class VehicleTypeProbabilities(BaseModel):
    veiculo_id: int
    probabilities: Dict[str, float]


class VehicleQuality(BaseModel):
    veiculo_id: int
    quality: str
    quality_reason: Optional[str] = None


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
    vehicle_quality: List[VehicleQuality] = []
    not_found: List[VehicleNotFound] = []


# ── Backtest ───────────────────────────────────────────────────────


class BacktestDailyItem(BaseModel):
    data: date
    actual: float
    predicted: float


class BacktestHeadItem(BaseModel):
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
