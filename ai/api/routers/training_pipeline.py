# api/routers/training_pipeline.py

"""Endpoints para execução e monitoramento do pipeline de treinamento."""

from __future__ import annotations

from http import HTTPStatus
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_session
from api.models import PipelineRun, PipelineStatus, PipelineStep, ModelType
from api.schemas.training_pipeline import (
    FullPipelineRequest,
    OptimizationRequest,
    PipelineRequest,
    PipelineRunResponse,
    SegmentationRequest,
)
from api.services import task_manager
from api.services.training_pipeline_service import (
    _create_run,
    run_dataset,
    run_full_pipeline,
    run_optimization,
    run_profiles,
    run_segmentation,
)

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _check_not_running(step: PipelineStep, target: str):
    """Levanta 409 se já houver tarefa rodando para (step, target)."""
    if task_manager.is_running(step.value, target):
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT,
            detail=f"{step.value}/{target} já está em execução.",
        )


# ------------------------------------------------------------------
# POST – iniciar etapas
# ------------------------------------------------------------------


@router.post(
    "/segmentation",
    response_model=PipelineRunResponse,
    status_code=HTTPStatus.ACCEPTED,
    summary="Iniciar segmentação",
)
async def start_segmentation(
    body: SegmentationRequest,
    session: AsyncSession = Depends(get_session),
):
    step = PipelineStep.SEGMENTATION
    _check_not_running(step, body.target)

    run_id = await _create_run(step, body.target)
    task_manager.submit(step.value, body.target, run_segmentation(run_id, body.target, body.data_path))

    run = await session.get(PipelineRun, run_id)
    return PipelineRunResponse.from_orm(run)


@router.post(
    "/profiles",
    response_model=PipelineRunResponse,
    status_code=HTTPStatus.ACCEPTED,
    summary="Gerar perfis de veículos",
)
async def start_profiles(
    body: PipelineRequest,
    session: AsyncSession = Depends(get_session),
):
    step = PipelineStep.PROFILES
    _check_not_running(step, body.target)

    run_id = await _create_run(step, body.target)
    task_manager.submit(step.value, body.target, run_profiles(run_id, body.target))

    run = await session.get(PipelineRun, run_id)
    return PipelineRunResponse.from_orm(run)


@router.post(
    "/datasets",
    response_model=PipelineRunResponse,
    status_code=HTTPStatus.ACCEPTED,
    summary="Gerar datasets de treinamento",
)
async def start_dataset(
    body: PipelineRequest,
    session: AsyncSession = Depends(get_session),
):
    step = PipelineStep.DATASET
    _check_not_running(step, body.target)

    run_id = await _create_run(step, body.target)
    task_manager.submit(step.value, body.target, run_dataset(run_id, body.target))

    run = await session.get(PipelineRun, run_id)
    return PipelineRunResponse.from_orm(run)


@router.post(
    "/optimization",
    response_model=PipelineRunResponse,
    status_code=HTTPStatus.ACCEPTED,
    summary="Otimizar hiperparâmetros e treinar modelo final",
)
async def start_optimization(
    body: OptimizationRequest,
    session: AsyncSession = Depends(get_session),
):
    step = PipelineStep.OPTIMIZATION
    _check_not_running(step, body.target)

    run_id = await _create_run(step, body.target, model_type=body.model_type.value)
    task_manager.submit(
        step.value,
        body.target,
        run_optimization(run_id, body.target, body.model_type.value),
    )

    run = await session.get(PipelineRun, run_id)
    return PipelineRunResponse.from_orm(run)


# ------------------------------------------------------------------
# POST – pipeline completo
# ------------------------------------------------------------------


@router.post(
    "/full",
    response_model=list[PipelineRunResponse],
    status_code=HTTPStatus.ACCEPTED,
    summary="Executar todas as etapas em sequência",
)
async def start_full_pipeline(
    body: FullPipelineRequest,
    session: AsyncSession = Depends(get_session),
):
    if task_manager.any_running():
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT,
            detail="Há etapas em execução. Aguarde a conclusão antes de iniciar o pipeline completo.",
        )

    run_ids = await run_full_pipeline(
        target=body.target,
        data_path=body.data_path,
        model_type=body.model_type.value,
    )

    runs = []
    for rid in run_ids:
        run = await session.get(PipelineRun, rid)
        runs.append(PipelineRunResponse.from_orm(run))
    return runs


# ------------------------------------------------------------------
# GET – consultar execuções
# ------------------------------------------------------------------


@router.get(
    "/runs/{run_id}",
    response_model=PipelineRunResponse,
    summary="Consultar uma execução",
)
async def get_run(
    run_id: int,
    session: AsyncSession = Depends(get_session),
):
    run = await session.get(PipelineRun, run_id)
    if not run:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Run não encontrado.")
    return PipelineRunResponse.from_orm(run)


@router.get(
    "/runs",
    response_model=list[PipelineRunResponse],
    summary="Listar execuções com filtros",
)
async def list_runs(
    step: Optional[PipelineStep] = Query(None),
    target: Optional[str] = Query(None, pattern=r"^(km|h)$"),
    status: Optional[PipelineStatus] = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
):
    query = select(PipelineRun).order_by(PipelineRun.created_at.desc())

    if step:
        query = query.where(PipelineRun.step == step)
    if target:
        query = query.where(PipelineRun.target == target)
    if status:
        query = query.where(PipelineRun.status == status)

    query = query.offset(offset).limit(limit)
    result = await session.execute(query)
    runs = result.scalars().all()
    return [PipelineRunResponse.from_orm(r) for r in runs]
