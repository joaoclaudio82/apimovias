# api/services/profiling_service.py

"""
Serviço de ingestão de dados: daily_activity, vehicle_profile, vehicle_metadata.

Fluxo:
1. Recebe dados diários brutos (CSV ou JSON)
2. Filtra apenas registros com data > dt_fim global (ProfileMetadata vigente)
3. Insere em daily_activity (apenas linhas com h > 0 ou km > 0)
4. Reconstrói DataFrame alinhado (datas completas para todos os veículos)
5. Executa VehicleProfile.run_pipeline
6. Atualiza vehicle_profile (upsert por veículo)
7. Atualiza vehicle_metadata_h / vehicle_metadata_km
8. Aplica sample_size: remove dados antigos de daily_activity
9. Calcula e persiste profile_metadata (valida monotonia temporal)
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
    DailyActivityRemoved,
    PipelineRun,
    PipelineStatus,
    PipelineStep,
    ProfileMetadata,
    VehicleMetadataH,
    VehicleMetadataKm,
    VehicleProfileFeature,
)

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


def _config_path(name: str) -> str:
    return str(_CONFIG_DIR / name)


def _resolve_classifiers_dir(output_cfg, active_version_id: str | None = None) -> Path:
    """
    Resolve o directório de classificadores a usar em produção.

    Prioridade:
    1. Bundle do modelo vigente (active_version_id da DB)
    2. Bundle mais recente em disco (fallback se não há ActiveModel)
    3. Lança erro se nenhum bundle encontrado (pipeline de treino deve ser executado primeiro)
    """
    from moviasai.bundle import ModelBundle

    base = Path(output_cfg.models.forecasting)

    # 1. Tentar bundle do modelo vigente
    if active_version_id:
        for target in ("km", "h"):
            try:
                bundle = ModelBundle.get_by_version(base, target, active_version_id)
                if bundle.classifiers_dir.exists():
                    return bundle.classifiers_dir
            except FileNotFoundError:
                continue

    # 2. Fallback: bundle mais recente em disco
    for target in ("km", "h"):
        versions = ModelBundle.list_versions(base, target)
        if versions:
            latest = versions[0]
            if latest.classifiers_dir.exists():
                return latest.classifiers_dir

    raise FileNotFoundError(
        "Nenhum bundle com classificadores encontrado em "
        f"{base}. Execute o pipeline de treino primeiro."
    )


# ------------------------------------------------------------------
# 1. Carregar cutoff global (dt_fim do perfil vigente)
# ------------------------------------------------------------------


async def _load_global_cutoff(session: AsyncSession) -> date | None:
    """
    Retorna dt_fim do ProfileMetadata mais recente, ou None na primeira ingestão.
    """
    result = await session.execute(
        select(ProfileMetadata.dt_fim)
        .order_by(ProfileMetadata.id.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


# ------------------------------------------------------------------
# 2. Filtrar e inserir daily_activity
# ------------------------------------------------------------------


async def _insert_new_daily_activity(
    session: AsyncSession,
    df_upload: pd.DataFrame,
    global_cutoff: date | None,
) -> int:
    """
    Filtra registros com data > cutoff global e pelo menos um target > 0.
    Insere em daily_activity (bulk). Retorna a contagem de registros inseridos.

    Na primeira ingestão (global_cutoff=None) aceita todos os registros.
    """
    df = df_upload.copy()

    # Filtrar por cutoff global (crescimento uniforme)
    if global_cutoff is not None:
        df = df[df["data"] > global_cutoff]

    # Manter apenas linhas com pelo menos um target > 0
    df = df[(df["h_dia_clean"] > 0) | (df["km_dia_clean"] > 0)]

    if df.empty:
        return 0

    # Bulk insert via core
    records = [
        {
            "veiculo_id": int(row["veiculo_id"]),
            "data": row["data"],
            "h": float(row["h_dia_clean"]) if row["h_dia_clean"] > 0 else 0.0,
            "km": float(row["km_dia_clean"]) if row["km_dia_clean"] > 0 else 0.0,
        }
        for _, row in df.iterrows()
    ]
    await session.execute(insert(DailyActivity), records)
    await session.flush()

    return len(records)


# ------------------------------------------------------------------
# 3. Reconstruir DataFrame alinhado a partir do daily_activity
# ------------------------------------------------------------------


async def _build_aligned_dataframe(
    session: AsyncSession, up_to: date | None = None,
) -> pl.DataFrame:
    """
    Lê daily_activity (opcionalmente até ``up_to``) e constrói DataFrame
    alinhado: todas as datas (min..max) para todos os veículos, com 0.
    """
    stmt = select(
        DailyActivity.veiculo_id,
        DailyActivity.data,
        DailyActivity.h,
        DailyActivity.km,
    )
    if up_to is not None:
        stmt = stmt.where(DailyActivity.data <= up_to)
    result = await session.execute(stmt)
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


def _run_profile_pipeline(df_aligned: pl.DataFrame, active_version_id: str | None = None) -> tuple:
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

    # Carregar classificadores do bundle do modelo vigente
    cls_dir = _resolve_classifiers_dir(output_cfg, active_version_id)

    type_clf = TypeClassifier.load_model(
        str(cls_dir / "stage1_type_BEST.onnx")
    )
    seg_clf_km = SegmentationClassifier.load_model(
        str(cls_dir / "stage2_km_BEST.onnx")
    )
    seg_clf_h = SegmentationClassifier.load_model(
        str(cls_dir / "stage2_h_BEST.onnx")
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

    # Incluir veículos inválidos (features sem NaN/Inf)
    df_long_invalid = profile.get_invalid_vehicles_long()
    if not df_long_invalid.empty:
        df_long = pd.concat([df_long, df_long_invalid], ignore_index=True)

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


async def _remove_stale_metadata(
    session: AsyncSession,
    df_meta: pd.DataFrame,
    model_cls: type,
) -> int:
    """
    Remove registos de metadata para veículos que já não estão no DataFrame.
    Isso garante que veículos reclassificados como single-target
    não mantêm registos no target inactivo.
    """
    if df_meta is None or df_meta.empty:
        # Se não há metadata nova, remover TODOS os registos existentes
        result = await session.execute(delete(model_cls))
        await session.flush()
        return result.rowcount

    current_vids = set(df_meta["veiculo_id"].astype(int).tolist())

    # Buscar todos os veiculo_id existentes na tabela
    result = await session.execute(select(model_cls.veiculo_id))
    existing_vids = {r[0] for r in result.all()}

    stale_vids = existing_vids - current_vids
    if not stale_vids:
        return 0

    await session.execute(
        delete(model_cls).where(model_cls.veiculo_id.in_(list(stale_vids)))
    )
    await session.flush()
    return len(stale_vids)


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
# 7. Aplicar sample_size: trimming global + metadata do perfil
# ------------------------------------------------------------------


def _first_monday_gte(d: date) -> date:
    """Primeira segunda-feira >= d."""
    offset = (7 - d.weekday()) % 7  # weekday(): 0=seg
    return d + timedelta(days=offset)


def _last_sunday_lte(d: date) -> date:
    """Último domingo <= d."""
    offset = (d.weekday() - 6) % 7  # 6=domingo
    return d - timedelta(days=offset)


async def _trim_daily_activity(
    session: AsyncSession,
    sample_size: int,
) -> tuple[int, int]:
    """
    Trimming global: mantém as últimas ``sample_size`` datas-calendário
    até ``dt_fim`` (último domingo <= max(data)) para todos os veículos.

    Armazenamento compacto: só existem registros onde h > 0 ou km > 0.
    Registros fora da janela [dt_inicio_sample, dt_fim] são removidos.
    Veículos sem nenhum registro remanescente são contados.

    Retorna (registros_removidos, veiculos_removidos).
    """
    from sqlalchemy import func as sa_func

    # Data máxima no banco
    result = await session.execute(select(sa_func.max(DailyActivity.data)))
    max_date = result.scalar_one_or_none()
    if max_date is None:
        return 0, 0

    dt_fim = _last_sunday_lte(max_date)
    dt_inicio_sample = dt_fim - timedelta(days=sample_size - 1)

    # Remover registros fora da janela
    del_result = await session.execute(
        delete(DailyActivity).where(
            (DailyActivity.data < dt_inicio_sample) | (DailyActivity.data > dt_fim)
        )
    )
    total_trimmed = del_result.rowcount

    # Contar veículos que ficaram sem registros
    # (veículos que existiam antes mas cujos registros estavam todos fora da janela)
    # Identificar pela ausência na tabela após trim
    # Não há como saber exactamente quantos foram removidos sem consulta prévia,
    # mas podemos contar os que restam
    result = await session.execute(
        select(sa_func.count(DailyActivity.veiculo_id.distinct()))
    )
    n_remaining = result.scalar_one()

    await session.flush()
    return total_trimmed, 0  # vehicles_removed reportado via perfil


async def _trim_daily_activity_removed(
    session: AsyncSession,
    sample_size: int,
) -> int:
    """
    Aplica a mesma janela de trimming de daily_activity à tabela
    daily_activity_removed: mantém até ``sample_size`` dias até dt_fim.
    """
    from sqlalchemy import func as sa_func

    result = await session.execute(select(sa_func.max(DailyActivityRemoved.data)))
    max_date = result.scalar_one_or_none()
    if max_date is None:
        return 0

    dt_fim = _last_sunday_lte(max_date)
    dt_inicio_sample = dt_fim - timedelta(days=sample_size - 1)

    del_result = await session.execute(
        delete(DailyActivityRemoved).where(
            (DailyActivityRemoved.data < dt_inicio_sample)
            | (DailyActivityRemoved.data > dt_fim)
        )
    )
    await session.flush()
    return del_result.rowcount


async def _compute_and_store_profile_metadata(
    session: AsyncSession,
    sample_size: int,
) -> dict:
    """
    Calcula metadados globais do perfil a partir do daily_activity
    e persiste em profile_metadata (mantendo histórico).

    Valida que o novo perfil é estritamente crescente no tempo:
    - dt_fim novo > dt_fim vigente
    - dt_inicio novo >= dt_inicio vigente

    Retorna dict com n_veiculos, dt_inicio, dt_fim.
    """
    from sqlalchemy import func as sa_func

    result = await session.execute(
        select(
            sa_func.min(DailyActivity.data),
            sa_func.max(DailyActivity.data),
            sa_func.count(DailyActivity.veiculo_id.distinct()),
        )
    )
    min_date, max_date, n_veiculos = result.one()

    if min_date is None:
        return {"n_veiculos": 0, "dt_inicio": None, "dt_fim": None}

    dt_inicio = _first_monday_gte(min_date)
    dt_fim = _last_sunday_lte(max_date)

    # Validar monotonia em relação ao perfil vigente
    prev_result = await session.execute(
        select(ProfileMetadata)
        .order_by(ProfileMetadata.id.desc())
        .limit(1)
    )
    prev = prev_result.scalar_one_or_none()

    if prev is not None:
        if dt_fim <= prev.dt_fim:
            raise ValueError(
                f"dt_fim do novo perfil ({dt_fim}) deve ser maior que o "
                f"vigente ({prev.dt_fim}). Os dados enviados não avançam "
                f"a janela temporal."
            )
        if dt_inicio < prev.dt_inicio:
            raise ValueError(
                f"dt_inicio do novo perfil ({dt_inicio}) não pode ser menor "
                f"que o vigente ({prev.dt_inicio}). "
                f"Verifique o sample_size ou os dados enviados."
            )

    meta = ProfileMetadata(
        n_veiculos=n_veiculos,
        dt_inicio=dt_inicio,
        dt_fim=dt_fim,
        sample_size=sample_size,
    )
    session.add(meta)
    await session.flush()

    logger.info(
        "profile_metadata: id=%d, n_veiculos=%d, dt_inicio=%s, dt_fim=%s",
        meta.id, n_veiculos, dt_inicio, dt_fim,
    )

    return {
        "n_veiculos": n_veiculos,
        "dt_inicio": str(dt_inicio),
        "dt_fim": str(dt_fim),
    }


# ------------------------------------------------------------------
# 8. Recuperar veículos removidos que tenham dados novos
# ------------------------------------------------------------------


async def _restore_removed_vehicles(
    session: AsyncSession,
    df_upload: pd.DataFrame,
) -> int:
    """
    Verifica se veículos em daily_activity_removed receberam dados novos.
    Se sim, move os dados de daily_activity_removed para daily_activity
    para que participem do pipeline de profiling.

    Retorna a quantidade de registos restaurados.
    """
    # IDs de veículos no upload
    upload_vids = df_upload["veiculo_id"].unique().tolist()
    if not upload_vids:
        return 0

    # Quais desses existem em daily_activity_removed?
    result = await session.execute(
        select(DailyActivityRemoved.veiculo_id.distinct()).where(
            DailyActivityRemoved.veiculo_id.in_(upload_vids)
        )
    )
    removed_vids = [r[0] for r in result.all()]
    if not removed_vids:
        return 0

    # Carregar dados removidos
    result = await session.execute(
        select(
            DailyActivityRemoved.veiculo_id,
            DailyActivityRemoved.data,
            DailyActivityRemoved.h,
            DailyActivityRemoved.km,
        ).where(DailyActivityRemoved.veiculo_id.in_(removed_vids))
    )
    removed_rows = result.all()

    if not removed_rows:
        return 0

    # Inserir em daily_activity (ignorar duplicatas)
    # Primeiro verificar quais datas já existem
    existing_result = await session.execute(
        select(DailyActivity.veiculo_id, DailyActivity.data).where(
            DailyActivity.veiculo_id.in_(removed_vids)
        )
    )
    existing_keys = {(r[0], r[1]) for r in existing_result.all()}

    records = []
    for vid, dt, h, km in removed_rows:
        if (vid, dt) not in existing_keys:
            records.append({
                "veiculo_id": vid,
                "data": dt,
                "h": h,
                "km": km,
            })

    if records:
        await session.execute(insert(DailyActivity), records)

    # Apagar de daily_activity_removed
    await session.execute(
        delete(DailyActivityRemoved).where(
            DailyActivityRemoved.veiculo_id.in_(removed_vids)
        )
    )
    await session.flush()

    logger.info(
        "Veículos restaurados de daily_activity_removed: %d veículos, %d registos",
        len(removed_vids), len(records),
    )
    return len(records)


# ------------------------------------------------------------------
# 9. Mover veículos não-VALID para daily_activity_removed
# ------------------------------------------------------------------


async def _archive_removed_vehicles(
    session: AsyncSession,
    df_meta_h: pd.DataFrame,
    df_meta_km: pd.DataFrame,
) -> int:
    """
    Após o pipeline de profiling, move dados de daily_activity para
    daily_activity_removed para veículos com quality inválida
    em AMBOS os targets.

    Critério: veículo é removido se quality não é VALID (0) nem
    SINGLE_TARGET (4) em h E em km.
    (Se for VALID ou SINGLE_TARGET em pelo menos um target, mantém-se.)

    Só arquiva veículos que têm pelo menos alguma atividade (h > 0 ou km > 0).

    Retorna o número de registos arquivados.
    """
    # Qualidades que mantêm o veículo activo
    _ACTIVE_QUALITIES = {0, 4}  # VALID, SINGLE_TARGET

    # Determinar veículos não activos em ambos os targets
    invalid_h = set()
    invalid_km = set()

    if df_meta_h is not None and not df_meta_h.empty and "quality" in df_meta_h.columns:
        invalid_h = set(
            df_meta_h.loc[~df_meta_h["quality"].isin(_ACTIVE_QUALITIES), "veiculo_id"].astype(int).tolist()
        )
    if df_meta_km is not None and not df_meta_km.empty and "quality" in df_meta_km.columns:
        invalid_km = set(
            df_meta_km.loc[~df_meta_km["quality"].isin(_ACTIVE_QUALITIES), "veiculo_id"].astype(int).tolist()
        )

    # Removidos = inválidos em AMBOS os targets
    to_archive = invalid_h & invalid_km
    if not to_archive:
        return 0

    # Carregar dados desses veículos do daily_activity
    result = await session.execute(
        select(
            DailyActivity.veiculo_id,
            DailyActivity.data,
            DailyActivity.h,
            DailyActivity.km,
        ).where(DailyActivity.veiculo_id.in_(list(to_archive)))
    )
    rows = result.all()

    if not rows:
        return 0

    # Filtrar: só arquivar veículos com pelo menos alguma actividade
    vids_with_activity = set()
    records = []
    for vid, dt, h, km in rows:
        if h > 0 or km > 0:
            vids_with_activity.add(vid)
        records.append({
            "veiculo_id": vid,
            "data": dt,
            "h": h,
            "km": km,
        })

    # Só arquivar veículos que têm actividade
    archive_vids = to_archive & vids_with_activity
    if not archive_vids:
        return 0

    archive_records = [r for r in records if r["veiculo_id"] in archive_vids]

    # Verificar quais já existem em daily_activity_removed (evitar duplicatas)
    existing_result = await session.execute(
        select(DailyActivityRemoved.veiculo_id, DailyActivityRemoved.data).where(
            DailyActivityRemoved.veiculo_id.in_(list(archive_vids))
        )
    )
    existing_keys = {(r[0], r[1]) for r in existing_result.all()}

    new_records = [
        r for r in archive_records
        if (r["veiculo_id"], r["data"]) not in existing_keys
    ]

    if new_records:
        await session.execute(insert(DailyActivityRemoved), new_records)

    # Apagar de daily_activity
    await session.execute(
        delete(DailyActivity).where(
            DailyActivity.veiculo_id.in_(list(archive_vids))
        )
    )
    await session.flush()

    logger.info(
        "Veículos arquivados em daily_activity_removed: %d veículos, %d registos",
        len(archive_vids), len(new_records),
    )
    return len(new_records)


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

    # 1. Carregar cutoff global (dt_fim do perfil vigente)
    global_cutoff = await _load_global_cutoff(session)

    # 1b. Restaurar veículos removidos que receberam dados novos
    n_restored = await _restore_removed_vehicles(session, df_upload)

    # 2. Inserir novos registros de daily_activity (data > cutoff global)
    n_inserted = await _insert_new_daily_activity(session, df_upload, global_cutoff)
    logger.info("daily_activity: %d inseridos, %d restaurados", n_inserted, n_restored)

    if n_inserted == 0 and n_restored == 0:
        logger.info("Nenhum registro novo — ignorando pipeline")
        return {
            "daily_activity_inserted": 0,
            "daily_activity_restored": 0,
            "daily_activity_trimmed": 0,
            "vehicles_archived": 0,
            "profile_features_upserted": 0,
            "metadata_h_upserted": 0,
            "metadata_km_upserted": 0,
        }

    # 3. Reconstruir DataFrame alinhado a partir do banco
    df_aligned = await _build_aligned_dataframe(session)
    logger.info("DataFrame alinhado: %d linhas", len(df_aligned))

    # 4. Executar pipeline (CPU-bound)
    # Buscar version_id do modelo vigente para resolver classificadores
    from api.models import ActiveModel
    result = await session.execute(
        select(ActiveModel.version_id).limit(1)
    )
    active_version_id = result.scalar_one_or_none()

    loop = asyncio.get_running_loop()
    df_long, df_meta_h, df_meta_km, sample_size = await loop.run_in_executor(
        None, _run_profile_pipeline, df_aligned, active_version_id
    )
    logger.info("Pipeline concluído: %d features em df_long", len(df_long))

    # 5. Upsert vehicle_profile
    n_features = await _upsert_vehicle_profile(session, df_long)
    logger.info("vehicle_profile: %d features upserted", n_features)

    # 6. Upsert metadata H e KM
    n_meta_h = await _upsert_metadata(session, df_meta_h, VehicleMetadataH)
    n_meta_km = await _upsert_metadata(session, df_meta_km, VehicleMetadataKm)
    logger.info("metadata: H=%d, KM=%d upserted", n_meta_h, n_meta_km)

    # 6b. Remover registos de metadata para veículos reclassificados como single-target
    n_removed_km = await _remove_stale_metadata(session, df_meta_km, VehicleMetadataKm)
    n_removed_h = await _remove_stale_metadata(session, df_meta_h, VehicleMetadataH)
    if n_removed_km or n_removed_h:
        logger.info("metadata stale removed: KM=%d, H=%d", n_removed_km, n_removed_h)

    # 7. Arquivar veículos não-VALID em daily_activity_removed
    n_archived = await _archive_removed_vehicles(session, df_meta_h, df_meta_km)

    # 8. Aplicar sample_size (trimming global)
    n_trimmed, _ = await _trim_daily_activity(session, sample_size)
    logger.info("daily_activity trimmed: %d registros removidos", n_trimmed)

    # 9. Aplicar trimming à tabela removed (mesma janela temporal)
    await _trim_daily_activity_removed(session, sample_size)

    # 10. Computar e persistir metadata do perfil
    profile_meta = await _compute_and_store_profile_metadata(session, sample_size)

    return {
        "daily_activity_inserted": n_inserted,
        "daily_activity_restored": n_restored,
        "daily_activity_trimmed": n_trimmed,
        "vehicles_archived": n_archived,
        "profile_features_upserted": n_features,
        "metadata_h_upserted": n_meta_h,
        "metadata_km_upserted": n_meta_km,
        "profile_metadata": profile_meta,
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


async def _first_ingestion(
    session: AsyncSession,
    loop,
    dataset_cfg,
    active_version_id: str | None,
    dt_fim: date,
    onnx_h, version_h,
    onnx_km, version_km,
) -> dict:
    """
    Cenário 1 — DB vazio (sem ProfileMetadata anterior).

    Perfil calculado uma única vez com todos os dados disponíveis.
    Predições são apenas inserções (não há registros anteriores).
    """
    from api.services.prediction_service import (
        extract_upper,
        pivot_profile_for_target,
        predict_single_iteration,
    )

    block_days = dataset_cfg.block_days
    n_heads = dataset_cfg.n_heads
    num_days = dataset_cfg.num_weeks_recent * 7

    df_aligned = await _build_aligned_dataframe(session, up_to=dt_fim)
    df_long, df_meta_h, df_meta_km, sample_size = await loop.run_in_executor(
        None, _run_profile_pipeline, df_aligned, active_version_id,
    )

    totals = {"heads_h": 0, "daily_h": 0, "heads_km": 0, "daily_km": 0}

    for target, onnx_fn, ver_id, df_meta in [
        ("h", onnx_h, version_h, df_meta_h),
        ("km", onnx_km, version_km, df_meta_km),
    ]:
        df_profile = pivot_profile_for_target(df_long, target)
        if df_profile.empty:
            continue

        vehicle_ids = df_profile["veiculo_id"].values
        upper = extract_upper(df_meta, vehicle_ids)

        counts = await predict_single_iteration(
            session=session, target=target, df_profile=df_profile,
            upper=upper, vehicle_ids=vehicle_ids, iter_dt_fim=dt_fim,
            onnx_filename=onnx_fn, version_id=ver_id, num_days=num_days,
            block_days=block_days, n_heads=n_heads,
            daily_horizon=dataset_cfg.daily_horizon,
        )

        if target == "h":
            totals["heads_h"] += counts["head_count"]
            totals["daily_h"] += counts["daily_count"]
        else:
            totals["heads_km"] += counts["head_count"]
            totals["daily_km"] += counts["daily_count"]

    logger.info("Primeira ingestão: dt_fim=%s", dt_fim)
    return {
        "df_long": df_long,
        "df_meta_h": df_meta_h,
        "df_meta_km": df_meta_km,
        "sample_size": sample_size,
        "totals": totals,
        "iterations": 1,
    }


async def _rolling_ingestion(
    session: AsyncSession,
    loop,
    dataset_cfg,
    active_version_id: str | None,
    dt_fim_full: date,
    prev_meta: ProfileMetadata,
    onnx_h, version_h,
    onnx_km, version_km,
) -> dict:
    """
    Cenário 2 — Ingestão subsequente com m blocos.

    Simula m ingestões isoladas: em cada iteração i o perfil é
    recalculado com dados até ``iter_dt_fim`` (sem vazamento temporal).
    ``fill_actual`` + upsert de predições a cada iteração.
    Devolve perfil/metadata apenas da última iteração (i == m-1).
    """
    from api.services.prediction_service import (
        extract_upper,
        fill_actual_from_daily_activity,
        pivot_profile_for_target,
        predict_single_iteration,
    )

    block_days = dataset_cfg.block_days
    n_heads = dataset_cfg.n_heads
    num_days = dataset_cfg.num_weeks_recent * 7

    new_days = (dt_fim_full - prev_meta.dt_fim).days
    m = max(1, new_days // block_days)

    logger.info(
        "Ingestão rolling: %d iterações de %d dias, dt_fim_full=%s",
        m, block_days, dt_fim_full,
    )

    totals = {"heads_h": 0, "daily_h": 0, "heads_km": 0, "daily_km": 0}
    df_long = df_meta_h = df_meta_km = None
    sample_size = None

    for i in range(m):
        iter_dt_fim = prev_meta.dt_fim + timedelta(days=(i + 1) * block_days)
        if iter_dt_fim > dt_fim_full:
            iter_dt_fim = dt_fim_full

        # Perfil recalculado com dados até iter_dt_fim
        df_aligned = await _build_aligned_dataframe(session, up_to=iter_dt_fim)
        df_long, df_meta_h, df_meta_km, sample_size = await loop.run_in_executor(
            None, _run_profile_pipeline, df_aligned, active_version_id,
        )

        for target, onnx_fn, ver_id, df_meta in [
            ("h", onnx_h, version_h, df_meta_h),
            ("km", onnx_km, version_km, df_meta_km),
        ]:
            # Preencher actual para o bloco recém-coberto
            await fill_actual_from_daily_activity(
                session, target, coverage_dt_fim=iter_dt_fim,
            )

            df_profile = pivot_profile_for_target(df_long, target)
            if df_profile.empty:
                continue

            vehicle_ids = df_profile["veiculo_id"].values
            upper = extract_upper(df_meta, vehicle_ids)

            counts = await predict_single_iteration(
                session=session, target=target, df_profile=df_profile,
                upper=upper, vehicle_ids=vehicle_ids, iter_dt_fim=iter_dt_fim,
                onnx_filename=onnx_fn, version_id=ver_id, num_days=num_days,
                block_days=block_days, n_heads=n_heads,
                daily_horizon=dataset_cfg.daily_horizon,
            )

            if target == "h":
                totals["heads_h"] += counts["head_count"]
                totals["daily_h"] += counts["daily_count"]
            else:
                totals["heads_km"] += counts["head_count"]
                totals["daily_km"] += counts["daily_count"]

        logger.info("Iteração %d/%d: iter_dt_fim=%s", i + 1, m, iter_dt_fim)

    return {
        "df_long": df_long,
        "df_meta_h": df_meta_h,
        "df_meta_km": df_meta_km,
        "sample_size": sample_size,
        "totals": totals,
        "iterations": m,
    }


async def ingest_and_predict(df_upload: pd.DataFrame, run_id: int) -> dict:
    """
    Orquestrador de ingestão + predição rolling.

    Cenário 1 (DB vazio): perfil único + apenas inserções.
    Cenário 2 (subsequente): m iterações com perfil recalculado por bloco.
    """
    import asyncio
    import json
    import traceback
    from datetime import datetime

    from sqlalchemy import func as sa_func

    from api.config.dataset_config import DatasetConfig
    from api.models import ActiveModel
    from api.services.prediction_service import _resolve_onnx_filename

    dataset_cfg = DatasetConfig.from_yaml(_config_path("dataset_config.yaml"))

    # Marcar RUNNING
    async with session_context() as s:
        run = await s.get(PipelineRun, run_id)
        run.status = PipelineStatus.RUNNING
        run.started_at = datetime.utcnow()
        await s.commit()

    try:
        async with session_context() as session:
            # ── 1. Inserir dados brutos ──────────────────────────
            global_cutoff = await _load_global_cutoff(session)
            n_restored = await _restore_removed_vehicles(session, df_upload)
            n_inserted = await _insert_new_daily_activity(
                session, df_upload, global_cutoff,
            )
            logger.info(
                "daily_activity: %d inseridos, %d restaurados",
                n_inserted, n_restored,
            )

            if n_inserted == 0 and n_restored == 0:
                await session.commit()
                async with session_context() as s:
                    run = await s.get(PipelineRun, run_id)
                    run.status = PipelineStatus.COMPLETED
                    run.finished_at = datetime.utcnow()
                    run.metrics = json.dumps(
                        {"ingestion": "no_new_data"}, default=str,
                    )
                    await s.commit()
                return {"run_id": run_id, "ingestion": "no_new_data"}

            # ── 2. Contexto comum ────────────────────────────────
            prev_result = await session.execute(
                select(ProfileMetadata)
                .order_by(ProfileMetadata.id.desc())
                .limit(1)
            )
            prev_meta = prev_result.scalar_one_or_none()

            result = await session.execute(
                select(sa_func.max(DailyActivity.data))
            )
            dt_fim_full = _last_sunday_lte(result.scalar_one())

            onnx_h, version_h = await _resolve_onnx_filename(session, "h")
            onnx_km, version_km = await _resolve_onnx_filename(session, "km")

            result = await session.execute(
                select(ActiveModel.version_id).limit(1)
            )
            active_version_id = result.scalar_one_or_none()

            loop = asyncio.get_running_loop()

            # ── 3. Despachar cenário ─────────────────────────────
            if prev_meta is None:
                out = await _first_ingestion(
                    session, loop, dataset_cfg, active_version_id,
                    dt_fim_full,
                    onnx_h, version_h, onnx_km, version_km,
                )
            else:
                out = await _rolling_ingestion(
                    session, loop, dataset_cfg, active_version_id,
                    dt_fim_full, prev_meta,
                    onnx_h, version_h, onnx_km, version_km,
                )

            # ── 4. Persistir perfil da última iteração ───────────
            df_long = out["df_long"]
            df_meta_h = out["df_meta_h"]
            df_meta_km = out["df_meta_km"]
            sample_size = out["sample_size"]
            totals = out["totals"]
            m = out["iterations"]

            n_features = await _upsert_vehicle_profile(session, df_long)
            n_meta_h = await _upsert_metadata(
                session, df_meta_h, VehicleMetadataH,
            )
            n_meta_km = await _upsert_metadata(
                session, df_meta_km, VehicleMetadataKm,
            )
            await _remove_stale_metadata(session, df_meta_km, VehicleMetadataKm)
            await _remove_stale_metadata(session, df_meta_h, VehicleMetadataH)
            n_archived = await _archive_removed_vehicles(
                session, df_meta_h, df_meta_km,
            )
            n_trimmed, _ = await _trim_daily_activity(session, sample_size)
            await _trim_daily_activity_removed(session, sample_size)
            profile_meta = await _compute_and_store_profile_metadata(
                session, sample_size,
            )

            # ── 5. Commit atómico ────────────────────────────────
            await session.commit()

        result = {
            "ingestion": {
                "daily_activity_inserted": n_inserted,
                "daily_activity_restored": n_restored,
                "daily_activity_trimmed": n_trimmed,
                "vehicles_archived": n_archived,
                "profile_features_upserted": n_features,
                "metadata_h_upserted": n_meta_h,
                "metadata_km_upserted": n_meta_km,
                "profile_metadata": profile_meta,
                "iterations": m,
            },
            "prediction_h": {
                "target": "h",
                "head_rows_persisted": totals["heads_h"],
                "daily_rows_persisted": totals["daily_h"],
            },
            "prediction_km": {
                "target": "km",
                "head_rows_persisted": totals["heads_km"],
                "daily_rows_persisted": totals["daily_km"],
            },
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
