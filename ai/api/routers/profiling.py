# api/routers/profiling.py

"""Endpoints para ingestão, profiling e predição combinada."""

from __future__ import annotations

import io
from http import HTTPStatus

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from sqlalchemy import select, union_all

from api.database import session_context
from api.models import (
    DailyActivity,
    DailyActivityRemoved,
    ProfileMetadata,
    VehicleMetadataH,
    VehicleMetadataKm,
    VehicleProfileFeature,
)
from api.schemas import Message
from api.schemas.profiling import ProfileMetadataResponse, VehicleInfoResponse, VehicleSummary
from api.schemas.training_pipeline import _to_local_str
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
# GET — listar veículos
# ------------------------------------------------------------------


@router.get(
    "/vehicles",
    response_model=list[VehicleSummary],
    status_code=HTTPStatus.OK,
    summary="Listar todos os veículos com metadados de ambos os targets",
)
async def list_vehicles(
    quality: int | None = Query(None, description="Filtrar por qualidade (0=válida,1=outlier,2=não modelável,3=vazia)"),
):
    """Retorna resumo de todos os veículos com metadados H e KM."""
    async with session_context() as session:
        result_km = await session.execute(select(VehicleMetadataKm))
        rows_km = {r.veiculo_id: r for r in result_km.scalars().all()}

        result_h = await session.execute(select(VehicleMetadataH))
        rows_h = {r.veiculo_id: r for r in result_h.scalars().all()}

    all_vids = sorted(set(rows_km) | set(rows_h))

    vehicles = []
    for vid in all_vids:
        km = rows_km.get(vid)
        h = rows_h.get(vid)

        # Qualidade é por veículo — pegar de qualquer target disponível
        q = km.quality if km else (h.quality if h else None)
        qr = km.quality_reason if km else (h.quality_reason if h else None)

        if quality is not None and q != quality:
            continue

        vehicles.append(VehicleSummary(
            veiculo_id=vid,
            quality=q,
            quality_reason=qr,
            dt_inicio_km=km.dt_inicio if km else None,
            dt_fim_km=km.dt_fim if km else None,
            upper_km=km.upper if km else None,
            dt_inicio_h=h.dt_inicio if h else None,
            dt_fim_h=h.dt_fim if h else None,
            upper_h=h.upper if h else None,
        ))

    return vehicles


# ------------------------------------------------------------------
# GET — profile metadata
# ------------------------------------------------------------------


@router.get(
    "/metadata",
    response_model=list[ProfileMetadataResponse],
    status_code=HTTPStatus.OK,
    summary="Histórico de metadados do perfil",
)
async def list_profile_metadata(
    last: bool = Query(False, description="Se True, retorna apenas o mais recente"),
):
    """Retorna o histórico de profile_metadata (ou só o último com ?last=true)."""
    async with session_context() as session:
        stmt = select(ProfileMetadata).order_by(ProfileMetadata.id.desc())
        if last:
            stmt = stmt.limit(1)
        result = await session.execute(stmt)
        rows = result.scalars().all()

    if not rows:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail="Nenhum profile_metadata encontrado. Execute a ingestão primeiro.",
        )

    return [
        ProfileMetadataResponse(
            id=r.id,
            n_veiculos=r.n_veiculos,
            dt_inicio=r.dt_inicio,
            dt_fim=r.dt_fim,
            sample_size=r.sample_size,
            created_at=_to_local_str(r.created_at),
        )
        for r in rows
    ]


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


# ------------------------------------------------------------------
# GET — série histórica diária de um veículo
# ------------------------------------------------------------------


@router.get(
    "/vehicle/{veiculo_id}/history",
    status_code=HTTPStatus.OK,
    summary="Série histórica diária de um veículo",
)
async def vehicle_history(
    veiculo_id: int,
    target: str = Query(pattern=r"^(km|h)$", description="Métrica: 'km' ou 'h'"),
):
    """Retorna a série temporal diária (DailyActivity + DailyActivityRemoved)."""
    col = "km" if target == "km" else "h"

    # União das duas tabelas
    stmt = union_all(
        select(
            DailyActivity.data,
            getattr(DailyActivity, col).label("valor"),
        ).where(DailyActivity.veiculo_id == veiculo_id),
        select(
            DailyActivityRemoved.data,
            getattr(DailyActivityRemoved, col).label("valor"),
        ).where(DailyActivityRemoved.veiculo_id == veiculo_id),
    ).order_by("data")

    async with session_context() as session:
        result = await session.execute(stmt)
        rows = result.all()

    # Filtrar dias com valor > 0
    series = [
        {"data": str(r.data), "valor": r.valor}
        for r in rows if r.valor and r.valor > 0
    ]

    if not series:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"Sem dados de {target.upper()} para o veículo {veiculo_id}.",
        )

    return series
