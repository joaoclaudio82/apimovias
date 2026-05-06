# api/routers/training_pipeline.py

"""Endpoints para execução e monitoramento do pipeline de treino."""

from __future__ import annotations

from http import HTTPStatus
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_session
from api.models import PipelineRun, PipelineStatus, PipelineStep, ModelType, ActiveModel
from api.schemas.training_pipeline import (
    ActiveModelResponse,
    BundleVersionResponse,
    DatasetRequest,
    FullPipelineRequest,
    OptimizationRequest,
    PipelineRequest,
    PipelineRunResponse,
    SegmentationRequest,
    TrainingRequest,
    UpdateActiveModelRequest,
)
from api.services import task_manager
from api.services.training_pipeline_service import (
    _create_run,
    run_dataset,
    run_full_pipeline,
    run_optimization,
    run_profiles,
    run_segmentation,
    run_training,
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
    summary="Iniciar segmentação de veículos",
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
    summary="Gerar perfis dos veículos",
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
    body: DatasetRequest,
    session: AsyncSession = Depends(get_session),
):
    step = PipelineStep.DATASET
    _check_not_running(step, body.target)

    run_id = await _create_run(step, body.target)
    task_manager.submit(step.value, body.target, run_dataset(run_id, body.target, use_cache=body.use_cache))

    run = await session.get(PipelineRun, run_id)
    return PipelineRunResponse.from_orm(run)


@router.post(
    "/optimization",
    response_model=PipelineRunResponse,
    status_code=HTTPStatus.ACCEPTED,
    summary="Otimizar hiperparâmetros e treinar o modelo final",
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


@router.post(
    "/training",
    response_model=PipelineRunResponse,
    status_code=HTTPStatus.ACCEPTED,
    summary="Treinar modelo diretamente com hiperparâmetros da configuração",
)
async def start_training(
    body: TrainingRequest,
    session: AsyncSession = Depends(get_session),
):
    step = PipelineStep.TRAINING
    _check_not_running(step, body.target)

    run_id = await _create_run(step, body.target, model_type=body.model_type.value)
    task_manager.submit(
        step.value,
        body.target,
        run_training(run_id, body.target, body.model_type.value),
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
        final_step=body.final_step.value,
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
    summary="Consultar execução por ID",
)
async def get_run(
    run_id: int,
    session: AsyncSession = Depends(get_session),
):
    run = await session.get(PipelineRun, run_id)
    if not run:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Execução não encontrada.")
    return PipelineRunResponse.from_orm(run)


@router.get(
    "/runs",
    response_model=list[PipelineRunResponse],
    summary="Listar execuções com filtros",
)
async def list_runs(
    step: Optional[PipelineStep] = Query(None),
    target: Optional[str] = Query(None, pattern=r"^(km|h|global)$"),
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


# ------------------------------------------------------------------
# GET – listar modelos vigentes
# ------------------------------------------------------------------


def _active_model_response(m: ActiveModel) -> ActiveModelResponse:
    from api.schemas.training_pipeline import _to_local_str

    return ActiveModelResponse(
        id=m.id,
        target=m.target,
        filename=m.filename,
        model_type=m.model_type,
        version_id=m.version_id,
        trained_at=str(m.trained_at),
        activated_at=_to_local_str(m.activated_at),
    )


def _resolve_output_cfg():
    from pathlib import Path as _Path
    from api.config.output_config import OutputConfig

    return OutputConfig.from_yaml(
        str(_Path(__file__).parent.parent.parent / "config" / "output_config.yaml")
    )


async def _ensure_active_models(session: AsyncSession) -> None:
    """
    Para cada target:
    1. Se o registo vigente aponta para um bundle inexistente → apaga o registo.
    2. Se não existe registo vigente → activa o último bundle disponível.
    """
    from pathlib import Path as _Path
    from datetime import date as _date, datetime as _datetime
    from moviasai.bundle import ModelBundle

    output_cfg = _resolve_output_cfg()
    forecasting_dir = _Path(output_cfg.models.forecasting)

    for target in ("km", "h"):
        result = await session.execute(
            select(ActiveModel).where(ActiveModel.target == target)
        )
        active = result.scalar_one_or_none()

        # Verificar se bundle do registo vigente existe
        if active is not None and active.version_id:
            bundle_dir = forecasting_dir / target / active.version_id
            if not (bundle_dir / ModelBundle.MODEL_FILENAME).exists():
                await session.delete(active)
                await session.flush()
                active = None

        # Se sem registo, activar o último bundle
        if active is None:
            bundles = ModelBundle.list_versions(forecasting_dir, target)
            if bundles:
                latest = bundles[0]
                manifest = latest.read_manifest()
                active = ActiveModel(
                    target=target,
                    filename=ModelBundle.MODEL_FILENAME,
                    model_type=manifest.get("model_type", "moe"),
                    trained_at=_date.today(),
                    version_id=latest.version_id,
                )
                session.add(active)
                await session.flush()


@router.get(
    "/models/active",
    response_model=list[ActiveModelResponse],
    summary="Listar modelos ONNX vigentes por target",
)
async def list_active_models(
    session: AsyncSession = Depends(get_session),
):
    """
    Retorna os modelos vigentes para cada target (km, h).

    Se o bundle vigente não existir em disco, o registo é removido
    e o último bundle disponível é activado automaticamente.
    """
    await _ensure_active_models(session)
    await session.commit()

    result = await session.execute(
        select(ActiveModel).order_by(ActiveModel.target)
    )
    rows = result.scalars().all()
    return [_active_model_response(m) for m in rows]


# ------------------------------------------------------------------
# PUT – trocar modelo vigente
# ------------------------------------------------------------------


@router.put(
    "/models/active",
    response_model=ActiveModelResponse,
    summary="Definir modelo ONNX vigente para um target",
)
async def update_active_model(
    body: UpdateActiveModelRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    Define o modelo vigente para predição de um target.

    O ``version_id`` identifica o bundle versionado a activar.
    Valida que o bundle existe em disco antes de activar.
    """
    from pathlib import Path as _Path
    from datetime import date as _date, datetime as _datetime
    from moviasai.bundle import ModelBundle

    output_cfg = _resolve_output_cfg()
    forecasting_dir = _Path(output_cfg.models.forecasting)

    # Validar que o bundle existe
    bundle_dir = forecasting_dir / body.target / body.version_id
    if not (bundle_dir / ModelBundle.MODEL_FILENAME).exists():
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"Bundle não encontrado: {body.target}/{body.version_id}",
        )

    bundle = ModelBundle(bundle_dir)
    manifest = bundle.read_manifest()
    model_type = manifest.get("model_type", "moe")

    # Upsert
    result = await session.execute(
        select(ActiveModel).where(ActiveModel.target == body.target)
    )
    active = result.scalar_one_or_none()

    if active is None:
        active = ActiveModel(
            target=body.target,
            filename=ModelBundle.MODEL_FILENAME,
            model_type=model_type,
            trained_at=_date.today(),
            version_id=body.version_id,
        )
        session.add(active)
    else:
        active.filename = ModelBundle.MODEL_FILENAME
        active.model_type = model_type
        active.version_id = body.version_id
        active.trained_at = _date.today()
        active.activated_at = _datetime.utcnow()

    await session.commit()
    await session.refresh(active)

    return _active_model_response(active)


# ------------------------------------------------------------------
# GET – listar bundles versionados em disco
# ------------------------------------------------------------------


@router.get(
    "/models/available",
    response_model=list[BundleVersionResponse],
    summary="Listar bundles versionados disponíveis para um target",
)
async def list_available_models(
    target: str = Query(pattern=r"^(km|h)$", description="Métrica: 'km' ou 'h'"),
):
    """
    Varre ``models/forecasting/{target}/`` e retorna todos os bundles
    versionados encontrados (mais recente primeiro).
    """
    from pathlib import Path as _Path
    from moviasai.bundle import ModelBundle

    output_cfg = _resolve_output_cfg()
    forecasting_dir = _Path(output_cfg.models.forecasting)

    bundles = ModelBundle.list_versions(forecasting_dir, target)
    results = []
    for b in bundles:
        manifest = b.read_manifest()
        # Converter created_at ISO UTC para hora local
        raw_created = manifest.get("created_at", "")
        try:
            from datetime import datetime as _dt
            from api.schemas.training_pipeline import _to_local_str
            dt_utc = _dt.fromisoformat(raw_created)
            created_local = _to_local_str(dt_utc) or raw_created
        except (ValueError, TypeError):
            created_local = raw_created
        results.append(BundleVersionResponse(
            version_id=manifest.get("version_id", b.version_id),
            target=manifest.get("target", target),
            model_type=manifest.get("model_type", "?"),
            created_at=created_local,
            model_hash=manifest.get("model_hash"),
        ))
    return results
