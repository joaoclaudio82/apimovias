# api/routers/prediction.py

"""Endpoints de predição de produção."""

from __future__ import annotations

import logging
from http import HTTPStatus

from fastapi import APIRouter, HTTPException, Query

from api.schemas import Message
from api.schemas.prediction import (
    PredictAndPersistRequest,
    PredictRequest,
    PredictResponse,
    BacktestResponse,
)
from api.services import prediction_service, task_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/prediction", tags=["prediction"])

_TASK_STEP = "prediction"  # chave fixa no task_manager


def _check_not_running(target: str):
    """Levanta 409 se já houver predição em execução para o target."""
    if task_manager.is_running(_TASK_STEP, target):
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT,
            detail=f"Predição para '{target}' já está em execução.",
        )


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


@router.post(
    "/predict_and_persist",
    response_model=Message,
    status_code=HTTPStatus.ACCEPTED,
    summary="Predição para todos os veículos com persistência (background)",
)
async def predict_and_persist(request: PredictAndPersistRequest):
    """
    Submete predição para todos os veículos em background.

    Apenas uma execução por target é permitida em simultâneo.
    """
    _check_not_running(request.target)

    task_manager.submit(
        _TASK_STEP,
        request.target,
        prediction_service.predict_and_persist(target=request.target),
    )

    return Message(
        message=f"Predição para '{request.target}' submetida em background.",
    )


@router.get(
    "/backtest/{veiculo_id}",
    response_model=BacktestResponse,
    status_code=HTTPStatus.OK,
    summary="Backtest: comparar predições com valores reais",
)
async def backtest(
    veiculo_id: int,
    target: str = Query(pattern=r"^(km|h)$", description="Métrica: 'km' ou 'h'"),
):
    """Compara predições persistidas com valores reais de daily_activity."""
    result = await prediction_service.backtest(veiculo_id, target)
    if result is None:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"Sem predições persistidas para veículo {veiculo_id} (target='{target}').",
        )
    return BacktestResponse(**result)
