# api/routers/profiling.py

"""Endpoints para ingestão, profiling e predição combinada."""

from __future__ import annotations

import io
from http import HTTPStatus

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from sqlalchemy import select

from api.database import session_context
from api.models import VehicleMetadataH, VehicleMetadataKm, VehicleProfileFeature
from api.schemas import Message
from api.schemas.profiling import VehicleInfoResponse
from api.services import task_manager
from api.services.profiling_service import create_ingestion_run, ingest_and_predict

router = APIRouter(prefix="/profiling", tags=["profiling"])

_REQUIRED_COLUMNS = {"veiculo_id", "data", "h_dia_clean", "km_dia_clean"}
_TASK_STEP = "ingest_predict"


def _validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Valida e normaliza colunas do DataFrame de upload."""
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            detail=f"Colunas obrigatórias faltando: {sorted(missing)}",
        )
    df = df[list(_REQUIRED_COLUMNS)].copy()
    df["data"] = pd.to_datetime(df["data"]).dt.date
    df["veiculo_id"] = df["veiculo_id"].astype(int)
    df["h_dia_clean"] = pd.to_numeric(df["h_dia_clean"], errors="coerce").fillna(0.0)
    df["km_dia_clean"] = pd.to_numeric(df["km_dia_clean"], errors="coerce").fillna(0.0)
    return df


def _check_not_running():
    if task_manager.is_running(_TASK_STEP, "global"):
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT,
            detail="Já há uma ingestão+predição em execução.",
        )


# ------------------------------------------------------------------
# POST — ingestão + predição (CSV)
# ------------------------------------------------------------------


@router.post(
    "/ingest",
    status_code=HTTPStatus.ACCEPTED,
    summary="Ingestão de dados e predição (background)",
)
async def ingest(file: UploadFile = File(...)):
    """
    Recebe CSV com dados diários, atualiza perfis e executa predição
    para ambos os targets (h e km) numa única transação atómica.

    Se a ingestão ou predição falhar, nada é persistido.
    Devolve o ``run_id`` para consulta posterior em ``/pipeline/runs/{id}``.
    """
    _check_not_running()

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            detail=f"Erro ao ler CSV: {e}",
        )

    df = _validate_dataframe(df)

    run_id = await create_ingestion_run()

    task_manager.submit(
        _TASK_STEP, "global",
        ingest_and_predict(df, run_id),
    )

    return {"run_id": run_id, "message": "Ingestão e predição submetidas em background."}


# ------------------------------------------------------------------
# GET — informações de um veículo
# ------------------------------------------------------------------


@router.get(
    "/vehicle/{veiculo_id}",
    response_model=VehicleInfoResponse,
    status_code=HTTPStatus.OK,
    summary="Informações de perfil e metadados de um veículo",
)
async def vehicle_info(
    veiculo_id: int,
    target: str = Query(pattern=r"^(km|h)$", description="Métrica: 'km' ou 'h'"),
):
    """Retorna perfil e metadados de um veículo para o target indicado."""
    async with session_context() as session:
        # Perfil: features com feature_class == target ou 'type'
        result = await session.execute(
            select(VehicleProfileFeature.feature, VehicleProfileFeature.valor)
            .where(
                VehicleProfileFeature.veiculo_id == veiculo_id,
                VehicleProfileFeature.feature_class.in_([target, "type"]),
            )
        )
        features = {r[0]: r[1] for r in result.all()}

        # Metadados
        meta_cls = VehicleMetadataKm if target == "km" else VehicleMetadataH
        result = await session.execute(
            select(meta_cls).where(meta_cls.veiculo_id == veiculo_id)
        )
        meta_row = result.scalar_one_or_none()

    if not features and meta_row is None:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"Veículo {veiculo_id} não existe ou não teve atividade recente.",
        )

    metadata = None
    if meta_row is not None:
        metadata = {
            "veiculo_id": meta_row.veiculo_id,
            "dt_inicio": meta_row.dt_inicio,
            "dt_fim": meta_row.dt_fim,
            "upper": meta_row.upper,
            "quality": meta_row.quality,
            "quality_reason": meta_row.quality_reason,
        }

    return VehicleInfoResponse(
        veiculo_id=veiculo_id,
        target=target,
        profile=features or None,
        metadata=metadata,
    )
