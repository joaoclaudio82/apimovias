# api/routers/prediction.py

"""Endpoints de predição de produção."""

from __future__ import annotations

import logging
from datetime import date
from http import HTTPStatus
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.schemas.prediction import (
    PredictRequest,
    PredictResponse,
    BacktestResponse,
)
from api.services import prediction_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/prediction", tags=["prediction"])


@router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=HTTPStatus.OK,
    summary="Predição para veículos específicos ou todos",
)
async def predict(request: PredictRequest):
    """
    Executa predição sem persistência.

    Se ``vehicle_ids`` for fornecido, filtra apenas esses veículos.
    Retorna predições daily, heads, probabilidades de tipo e veículos não encontrados.
    """
    result = await prediction_service.predict(
        target=request.target,
        vehicle_ids=request.vehicle_ids,
    )
    return PredictResponse(**result)


@router.get(
    "/backtest/{veiculo_id}",
    response_model=BacktestResponse,
    status_code=HTTPStatus.OK,
    summary="Backtest: comparar predições com valores reais",
)
async def backtest(
    veiculo_id: int,
    target: str = Query(pattern=r"^(km|h)$", description="Métrica: 'km' ou 'h'"),
    date_from: Optional[date] = Query(None, description="Data inicial do gráfico diário"),
    date_to: Optional[date] = Query(None, description="Data final do gráfico diário"),
    max_heads: Optional[int] = Query(None, ge=1, description="Quantidade máxima de blocos (mais recentes)"),
):
    """Compara predições persistidas com valores reais de daily_activity."""
    result = await prediction_service.backtest(
        veiculo_id, target,
        date_from=date_from, date_to=date_to, max_heads=max_heads,
    )
    if result is None:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"Sem predições persistidas para veículo {veiculo_id} (target='{target}').",
        )
    return BacktestResponse(**result)
