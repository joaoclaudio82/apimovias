# api/services/profiling_service.py

"""
Serviço de ingestão de dados: daily_activity, vehicle_profile, vehicle_metadata.

Fluxo:
1. Recebe dados diários brutos (CSV ou JSON)
2. Filtra apenas registros novos (data > dt_fim existente por veículo+target)
3. Insere em daily_activity (apenas linhas com h > 0 ou km > 0)
4. Reconstrói DataFrame alinhado (datas completas para todos os veículos)
5. Executa VehicleProfile.run_pipeline
6. Atualiza vehicle_profile (upsert por veículo)
7. Atualiza vehicle_metadata_h / vehicle_metadata_km
8. Aplica sample_size: remove dados antigos de daily_activity
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sqlalchemy import delete, insert, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import session_context
from api.models import (
    DailyActivity,
    PipelineRun,
    PipelineStatus,
    PipelineStep,
    VehicleMetadataH,
    VehicleMetadataKm,
    VehicleProfileFeature,
)

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


def _config_path(name: str) -> str:
    return str(_CONFIG_DIR / name)


# ------------------------------------------------------------------
# 1. Carregar metadados existentes do banco (dt_fim por veículo+target)
# ------------------------------------------------------------------


async def _load_metadata_cutoffs(
    session: AsyncSession,
) -> dict[str, dict[int, date]]:
    """
    Retorna {target: {veiculo_id: dt_fim}} para H e KM.
    Usado para filtrar registros antigos do upload.
    """
    cutoffs: dict[str, dict[int, date]] = {"h": {}, "km": {}}

    for model_cls, target in [(VehicleMetadataH, "h"), (VehicleMetadataKm, "km")]:
        result = await session.execute(
            select(model_cls.veiculo_id, model_cls.dt_fim)
        )
        for vid, dt_fim in result.all():
            cutoffs[target][vid] = dt_fim

    return cutoffs


# ------------------------------------------------------------------
# 2. Filtrar e inserir daily_activity
# ------------------------------------------------------------------


async def _insert_new_daily_activity(
    session: AsyncSession,
    df_upload: pd.DataFrame,
    cutoffs: dict[str, dict[int, date]],
) -> int:
    """
    Filtra registros novos e com target > 0, insere em daily_activity (bulk).
    Retorna a contagem de registros inseridos.
    """
    # Construir DataFrame de cutoffs para filtragem vetorizada
    h_cutoffs = pd.Series(cutoffs["h"], name="h_cutoff")
    km_cutoffs = pd.Series(cutoffs["km"], name="km_cutoff")

    df = df_upload.copy()
    df["h_cutoff"] = df["veiculo_id"].map(h_cutoffs)
    df["km_cutoff"] = df["veiculo_id"].map(km_cutoffs)

    # Verificar se é novo para pelo menos um target
    h_is_new = (df["h_dia_clean"] > 0) & (df["h_cutoff"].isna() | (df["data"] > df["h_cutoff"]))
    km_is_new = (df["km_dia_clean"] > 0) & (df["km_cutoff"].isna() | (df["data"] > df["km_cutoff"]))
    df_new = df[h_is_new | km_is_new].copy()

    if df_new.empty:
        return 0

    # Bulk insert via core
    records = [
        {
            "veiculo_id": int(row["veiculo_id"]),
            "data": row["data"],
            "h": float(row["h_dia_clean"]) if row["h_dia_clean"] > 0 else 0.0,
            "km": float(row["km_dia_clean"]) if row["km_dia_clean"] > 0 else 0.0,
        }
        for _, row in df_new.iterrows()
    ]
    await session.execute(insert(DailyActivity), records)
    await session.flush()

    return len(records)


# ------------------------------------------------------------------
# 3. Reconstruir DataFrame alinhado a partir do daily_activity
# ------------------------------------------------------------------


async def _build_aligned_dataframe(session: AsyncSession) -> pl.DataFrame:
    """
    Lê todo o daily_activity e constrói um DataFrame alinhado:
    todas as datas (min..max) para todos os veículos, preenchendo com 0.
    """
    result = await session.execute(
        select(
            DailyActivity.veiculo_id,
            DailyActivity.data,
            DailyActivity.h,
            DailyActivity.km,
        )
    )
    rows = result.all()

    if not rows:
        return pl.DataFrame(
            schema={"veiculo_id": pl.Int64, "data": pl.Date,
                    "h_dia_clean": pl.Float64, "km_dia_clean": pl.Float64}
        )

    df = pl.DataFrame(
        {
            "veiculo_id": [r[0] for r in rows],
            "data": [r[1] for r in rows],
            "h_dia_clean": [r[2] for r in rows],
            "km_dia_clean": [r[3] for r in rows],
        }
    )

    # Alinhar datas: todas as datas de min a max para todos os veículos
    all_veiculos = df.select("veiculo_id").unique()
    min_date = df["data"].min()
    max_date = df["data"].max()

    all_dates = pl.DataFrame(
        {"data": pl.date_range(min_date, max_date, eager=True)}
    )
    grid = all_veiculos.join(all_dates, how="cross")

    df_aligned = grid.join(df, on=["veiculo_id", "data"], how="left").with_columns(
        pl.col("h_dia_clean").fill_null(0.0),
        pl.col("km_dia_clean").fill_null(0.0),
    )

    return df_aligned.sort(["veiculo_id", "data"])


# ------------------------------------------------------------------
# 4. Executar VehicleProfile.run_pipeline
# ------------------------------------------------------------------


def _run_profile_pipeline(df_aligned: pl.DataFrame) -> tuple:
    """
    Executa o pipeline de perfil de veículo.
    Retorna (df_long, df_metadata_h, df_metadata_km, sample_size).
    """
    from api.config.data_quality_config import DataQualityConfig
    from api.config.output_config import OutputConfig
    from api.config.segmentation_config import SegmentationConfig
    from api.config.vehicle_profile_config import VehicleProfileConfig
    from moviasai.profiling.classification import SegmentationClassifier, TypeClassifier
    from moviasai.profiling.profile import VehicleProfile

    profile_cfg = VehicleProfileConfig.from_yaml(_config_path("vehicle_profile_config.yaml"))
    seg_cfg = SegmentationConfig.from_yaml(_config_path("segmentation_config.yaml"))
    output_cfg = OutputConfig.from_yaml(_config_path("output_config.yaml"))
    dq_cfg = DataQualityConfig.from_yaml(_config_path("data_quality_config.yaml"))

    cls_dir = Path(output_cfg.models.classification)
    type_clf = TypeClassifier.load_model(
        str(cls_dir / "stage1" / "stage1_type_BEST.onnx")
    )
    seg_clf_km = SegmentationClassifier.load_model(
        str(cls_dir / "stage2" / "stage2_km_BEST.onnx")
    )
    seg_clf_h = SegmentationClassifier.load_model(
        str(cls_dir / "stage2" / "stage2_h_BEST.onnx")
    )

    profile = VehicleProfile.from_config(
        segmentation_config=seg_cfg,
        profile_config=profile_cfg,
        data_quality_config=dq_cfg,
        override_features=True,
        extract_metadata=True,
    )

    df_long = profile.run_pipeline(
        df=df_aligned,
        type_classifier=type_clf,
        segment_classifier_km=seg_clf_km,
        segment_classifier_h=seg_clf_h,
    )

    return (
        df_long,
        profile.df_metadata_h,
        profile.df_metadata_km,
        profile_cfg.profile.sample_size,
    )


# ------------------------------------------------------------------
# 5. Upsert vehicle_profile (features long)
# ------------------------------------------------------------------


async def _upsert_vehicle_profile(
    session: AsyncSession,
    df_long: pd.DataFrame,
) -> int:
    """
    Atualiza features de perfil: apaga toda a tabela e insere o perfil novo.
    Veículos que saíram do pipeline não servem para predição.
    """
    if df_long.empty:
        return 0

    # Valor nulo no perfil é erro no pipeline — interromper
    null_mask = df_long["valor"].isnull()
    if null_mask.any():
        bad = df_long[null_mask][["veiculo_id", "feature", "feature_class"]].head(20)
        logger.error("Valores nulos no perfil:\n%s", bad.to_string(index=False))
        raise ValueError(
            f"Pipeline gerou {null_mask.sum()} features com valor nulo. "
            f"Veículos afetados: {df_long.loc[null_mask, 'veiculo_id'].unique().tolist()[:20]}"
        )

    # Apagar TODA a tabela — perfil antigo não serve para predição
    await session.execute(delete(VehicleProfileFeature))

    # Bulk insert via core
    records = [
        {
            "veiculo_id": int(row["veiculo_id"]),
            "feature": str(row["feature"]),
            "feature_class": str(row["feature_class"]),
            "valor": float(row["valor"]),
        }
        for _, row in df_long.iterrows()
    ]
    await session.execute(insert(VehicleProfileFeature), records)
    await session.flush()
    return len(records)


# ------------------------------------------------------------------
# 6. Upsert vehicle_metadata (H e KM)
# ------------------------------------------------------------------


async def _upsert_metadata(
    session: AsyncSession,
    df_meta: pd.DataFrame,
    model_cls: type,
) -> int:
    """
    Para cada veículo em df_meta:
    - Se existe: atualiza dt_fim, upper, quality e quality_reason (dt_inicio permanece)
    - Se não existe: insere registro completo
    """
    if df_meta is None or df_meta.empty:
        return 0

    has_quality = "quality" in df_meta.columns

    # Preparar dados em memória
    meta_records = []
    for _, row in df_meta.iterrows():
        dt_inicio = row["dt_inicio"]
        dt_fim = row["dt_fim"]
        if isinstance(dt_inicio, pd.Timestamp):
            dt_inicio = dt_inicio.date()
        if isinstance(dt_fim, pd.Timestamp):
            dt_fim = dt_fim.date()
        meta_records.append({
            "veiculo_id": int(row["veiculo_id"]),
            "dt_inicio": dt_inicio,
            "dt_fim": dt_fim,
            "upper": row["upper"] if pd.notna(row["upper"]) else None,
            "quality": int(row["quality"]) if has_quality and pd.notna(row.get("quality")) else None,
            "quality_reason": str(row["quality_reason"]) if has_quality and pd.notna(row.get("quality_reason")) else None,
        })

    new_vids = [r["veiculo_id"] for r in meta_records]

    # Descobrir quais já existem (uma única query)
    result = await session.execute(
        select(model_cls.veiculo_id).where(model_cls.veiculo_id.in_(new_vids))
    )
    existing_vids = {r[0] for r in result.all()}

    # Bulk update existentes
    to_update = [r for r in meta_records if r["veiculo_id"] in existing_vids]
    if to_update:
        for rec in to_update:
            await session.execute(
                update(model_cls)
                .where(model_cls.veiculo_id == rec["veiculo_id"])
                .values(
                    dt_fim=rec["dt_fim"],
                    upper=rec["upper"],
                    quality=rec["quality"],
                    quality_reason=rec["quality_reason"],
                )
            )

    # Bulk insert novos
    to_insert = [r for r in meta_records if r["veiculo_id"] not in existing_vids]
    if to_insert:
        await session.execute(insert(model_cls), to_insert)

    await session.flush()
    return len(meta_records)


# ------------------------------------------------------------------
# 7. Aplicar sample_size: remover dados antigos de daily_activity
# ------------------------------------------------------------------


async def _trim_daily_activity(
    session: AsyncSession,
    sample_size: int,
) -> tuple[int, int]:
    """
    Para cada veículo, mantém apenas os últimos `sample_size` dias com
    atividade (h > 0 ou km > 0). Remove registros mais antigos.
    Remove veículos inteiros se não restar nenhuma atividade.

    Retorna (registros_removidos, veiculos_removidos).
    """
    # Obter todos os veículos distintos
    result = await session.execute(
        select(DailyActivity.veiculo_id).distinct()
    )
    all_vids = [r[0] for r in result.all()]

    total_trimmed = 0
    vehicles_removed = 0

    for vid in all_vids:
        # Obter datas com atividade, ordenadas desc
        result = await session.execute(
            select(DailyActivity.data)
            .where(
                DailyActivity.veiculo_id == vid,
                (DailyActivity.h > 0) | (DailyActivity.km > 0),
            )
            .order_by(DailyActivity.data.desc())
        )
        active_dates = [r[0] for r in result.all()]

        if not active_dates:
            # Veículo sem atividade: remover tudo
            del_result = await session.execute(
                delete(DailyActivity).where(DailyActivity.veiculo_id == vid)
            )
            total_trimmed += del_result.rowcount
            vehicles_removed += 1
            continue

        if len(active_dates) <= sample_size:
            # Manter tudo, mas remover registros anteriores à data mais antiga
            cutoff_date = active_dates[-1]
        else:
            cutoff_date = active_dates[sample_size - 1]

        # Remover registros anteriores ao cutoff
        del_result = await session.execute(
            delete(DailyActivity).where(
                DailyActivity.veiculo_id == vid,
                DailyActivity.data < cutoff_date,
            )
        )
        total_trimmed += del_result.rowcount

    await session.flush()
    return total_trimmed, vehicles_removed


# ------------------------------------------------------------------
# ORQUESTRADOR PRINCIPAL
# ------------------------------------------------------------------


async def _run_ingestion_steps(
    session: AsyncSession,
    df_upload: pd.DataFrame,
) -> dict:
    """
    Executa os passos de ingestão numa sessão existente (sem commit).

    O chamador é responsável por fazer commit ou rollback.
    """
    import asyncio

    # 1. Carregar cutoffs existentes
    cutoffs = await _load_metadata_cutoffs(session)

    # 2. Inserir novos registros de daily_activity
    n_inserted = await _insert_new_daily_activity(session, df_upload, cutoffs)
    logger.info("daily_activity: %d registros inseridos", n_inserted)

    if n_inserted == 0:
        logger.info("Nenhum registro novo — ignorando pipeline")
        return {
            "daily_activity_inserted": 0,
            "daily_activity_trimmed": 0,
            "vehicles_removed": 0,
            "profile_features_upserted": 0,
            "metadata_h_upserted": 0,
            "metadata_km_upserted": 0,
        }

    # 3. Reconstruir DataFrame alinhado a partir do banco
    df_aligned = await _build_aligned_dataframe(session)
    logger.info("DataFrame alinhado: %d linhas", len(df_aligned))

    # 4. Executar pipeline (CPU-bound)
    loop = asyncio.get_running_loop()
    df_long, df_meta_h, df_meta_km, sample_size = await loop.run_in_executor(
        None, _run_profile_pipeline, df_aligned
    )
    logger.info("Pipeline concluído: %d features em df_long", len(df_long))

    # 5. Upsert vehicle_profile
    n_features = await _upsert_vehicle_profile(session, df_long)
    logger.info("vehicle_profile: %d features upserted", n_features)

    # 6. Upsert metadata H e KM
    n_meta_h = await _upsert_metadata(session, df_meta_h, VehicleMetadataH)
    n_meta_km = await _upsert_metadata(session, df_meta_km, VehicleMetadataKm)
    logger.info("metadata: H=%d, KM=%d upserted", n_meta_h, n_meta_km)

    # 7. Aplicar sample_size
    n_trimmed, n_vehicles_removed = await _trim_daily_activity(session, sample_size)
    logger.info("daily_activity trimmed: %d registros, %d veículos removidos",
                n_trimmed, n_vehicles_removed)

    return {
        "daily_activity_inserted": n_inserted,
        "daily_activity_trimmed": n_trimmed,
        "vehicles_removed": n_vehicles_removed,
        "profile_features_upserted": n_features,
        "metadata_h_upserted": n_meta_h,
        "metadata_km_upserted": n_meta_km,
    }


async def run_ingestion(df_upload: pd.DataFrame) -> dict:
    """
    Orquestra todo o processo de ingestão (standalone, com commit próprio).
    """
    async with session_context() as session:
        result = await _run_ingestion_steps(session, df_upload)
        await session.commit()
    return result


async def create_ingestion_run() -> int:
    """Cria um PipelineRun PENDING para ingestão e devolve o run_id."""
    async with session_context() as s:
        run = PipelineRun(
            step=PipelineStep.INGESTION,
            target="global",
            status=PipelineStatus.PENDING,
        )
        s.add(run)
        await s.commit()
        await s.refresh(run)
        return run.id


async def ingest_and_predict(df_upload: pd.DataFrame, run_id: int) -> dict:
    """
    Ingestão de dados + predição para ambos os targets numa única transação.

    Se a ingestão ou a predição falhar, tudo é revertido (rollback).
    Só faz predição se houver registros novos inseridos.
    Atualiza o PipelineRun identificado por *run_id*.
    """
    import json
    import traceback
    from datetime import datetime

    from api.services.prediction_service import predict_and_persist_with_session

    # Marcar RUNNING
    async with session_context() as s:
        run = await s.get(PipelineRun, run_id)
        run.status = PipelineStatus.RUNNING
        run.started_at = datetime.utcnow()
        await s.commit()

    try:
        async with session_context() as session:
            # 1. Ingestão
            ingestion = await _run_ingestion_steps(session, df_upload)

            prediction_h = None
            prediction_km = None

            if ingestion["daily_activity_inserted"] > 0:
                # 2. Predição para ambos os targets
                prediction_h = await predict_and_persist_with_session(session, "h")
                prediction_km = await predict_and_persist_with_session(session, "km")

            # 3. Commit atómico — tudo ou nada
            await session.commit()

        result = {
            "ingestion": ingestion,
            "prediction_h": prediction_h,
            "prediction_km": prediction_km,
        }

        # Marcar COMPLETED
        async with session_context() as s:
            run = await s.get(PipelineRun, run_id)
            run.status = PipelineStatus.COMPLETED
            run.finished_at = datetime.utcnow()
            run.metrics = json.dumps(result, default=str)
            await s.commit()

        return {"run_id": run_id, **result}

    except Exception:
        # Marcar FAILED
        async with session_context() as s:
            run = await s.get(PipelineRun, run_id)
            run.status = PipelineStatus.FAILED
            run.finished_at = datetime.utcnow()
            run.error_message = traceback.format_exc()[:4000]
            await s.commit()
        raise
