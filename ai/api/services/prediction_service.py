# api/services/prediction_service.py

"""
Serviço de predição: executa inferência ONNX e (opcionalmente) persiste resultados.

Fluxo:
1. Carrega perfis (vehicle_profile) e metadados do banco
2. Carrega histórico recente (daily_activity)
3. Alinha e valida séries recentes (trunca ao último domingo, exige num_days completos)
4. Gera DataInput e normaliza
5. Executa inferência ONNX via ForecastingPredictor
6. Calcula datas de predição (daily e heads)
7. Retorna ou persiste resultados
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
from sqlalchemy import insert, or_, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import session_context
from api.models import (
    ActiveModel,
    DailyActivity,
    PredictionDaily,
    PredictionHead,
    ProfileMetadata,
    VehicleMetadataH,
    VehicleMetadataKm,
    VehicleProfileFeature,
)

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


def _config_path(name: str) -> str:
    return str(_CONFIG_DIR / name)


# ------------------------------------------------------------------
# DB: carregar dados
# ------------------------------------------------------------------


async def _load_vehicle_profiles(
    session: AsyncSession,
    target: str,
    vehicle_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Carrega vehicle_profile (formato long) e pivota para wide.

    Retorna DataFrame com colunas: veiculo_id, feat_1, feat_2, ...
    """
    stmt = select(
        VehicleProfileFeature.veiculo_id,
        VehicleProfileFeature.feature,
        VehicleProfileFeature.valor,
    ).where(VehicleProfileFeature.feature_class == target)

    if vehicle_ids is not None:
        stmt = stmt.where(VehicleProfileFeature.veiculo_id.in_(vehicle_ids))

    result = await session.execute(stmt)
    rows = result.all()

    if not rows:
        return pd.DataFrame(columns=["veiculo_id"])

    df_long = pd.DataFrame(rows, columns=["veiculo_id", "feature", "valor"])
    df_profile = df_long.pivot(
        index="veiculo_id", columns="feature", values="valor",
    ).reset_index()
    df_profile.columns.name = None
    return df_profile


async def _load_type_probabilities(
    session: AsyncSession,
    vehicle_ids: List[int],
) -> Dict[int, Dict[str, float]]:
    """Carrega probabilidades de tipo e features de tipificação."""
    stmt = select(
        VehicleProfileFeature.veiculo_id,
        VehicleProfileFeature.feature,
        VehicleProfileFeature.valor,
    ).where(
        or_(
            VehicleProfileFeature.feature.like("type%"),
            VehicleProfileFeature.feature_class == "type",
        ),
        VehicleProfileFeature.veiculo_id.in_(vehicle_ids),
    )
    result = await session.execute(stmt)
    rows = result.all()

    probs: Dict[int, Dict[str, float]] = {}
    for vid, feat, val in rows:
        probs.setdefault(vid, {})[feat] = val
    return probs


async def _load_metadata(
    session: AsyncSession,
    target: str,
    vehicle_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Carrega metadados (upper, quality, quality_reason) para o target."""
    model_cls = VehicleMetadataKm if target == "km" else VehicleMetadataH
    stmt = select(
        model_cls.veiculo_id,
        model_cls.upper,
        model_cls.quality,
        model_cls.quality_reason,
    )
    if vehicle_ids is not None:
        stmt = stmt.where(model_cls.veiculo_id.in_(vehicle_ids))
    result = await session.execute(stmt)
    rows = result.all()
    return pd.DataFrame(rows, columns=["veiculo_id", "upper", "quality", "quality_reason"])


async def _load_profile_metadata(session: AsyncSession) -> ProfileMetadata:
    """Carrega o ProfileMetadata mais recente. Levanta erro se não existir."""
    result = await session.execute(
        select(ProfileMetadata).order_by(ProfileMetadata.id.desc()).limit(1)
    )
    meta = result.scalar_one_or_none()
    if meta is None:
        raise ValueError(
            "Nenhum profile_metadata encontrado. Execute a ingestão primeiro."
        )
    return meta


async def _resolve_onnx_filename(session: AsyncSession, target: str) -> Tuple[str, Optional[str]]:
    """
    Retorna (filename, version_id) do modelo ONNX vigente para o target.

    Se não existir registo em active_models, insere o default de
    predictor_config.yaml e retorna-o com version_id=None.
    """
    result = await session.execute(
        select(ActiveModel).where(ActiveModel.target == target)
    )
    active = result.scalar_one_or_none()

    if active is not None:
        return active.filename, active.version_id

    # Auto-registar default a partir do config
    from api.config.predictor_config import PredictorConfig

    predictor_cfg = PredictorConfig.from_yaml(_config_path("predictor_config.yaml"))
    default_filename = getattr(predictor_cfg.models, target)

    active = ActiveModel(
        target=target,
        filename=default_filename,
        model_type="moe" if "moe" in default_filename else "multihead",
        trained_at=date.today(),
    )
    session.add(active)
    await session.flush()
    logger.info("active_models: registado default para %s → %s", target, default_filename)
    return default_filename, None


async def _load_daily_activity(
    session: AsyncSession,
    target: str,
    vehicle_ids: List[int],
    dt_fim: date,
    num_days: int,
) -> pl.DataFrame:
    """
    Carrega atividade diária recente para os veículos indicados,
    na janela global [dt_fim - num_days + 1, dt_fim].
    """
    target_col = "km" if target == "km" else "h"
    dt_inicio = dt_fim - timedelta(days=num_days - 1)

    placeholders = ", ".join(f":id_{i}" for i in range(len(vehicle_ids)))
    params = {f"id_{i}": vid for i, vid in enumerate(vehicle_ids)}

    stmt = text(f"""
        SELECT veiculo_id, data, {target_col} AS value
        FROM daily_activity
        WHERE veiculo_id IN ({placeholders})
        AND data >= :dt_inicio
        AND data <= :dt_fim
        ORDER BY veiculo_id, data
    """).bindparams(**params, dt_inicio=dt_inicio, dt_fim=dt_fim)

    result = await session.execute(stmt)
    rows = result.all()

    if not rows:
        return pl.DataFrame(schema={"veiculo_id": pl.Int64, "data": pl.Date, "value": pl.Float64})

    df = pl.DataFrame(
        {"veiculo_id": [r[0] for r in rows],
         "data": [r[1] for r in rows],
         "value": [r[2] for r in rows]},
    ).with_columns(pl.col("data").cast(pl.Date))

    return df


# ------------------------------------------------------------------
# Processamento: alinhar séries recentes (global dt_fim)
# ------------------------------------------------------------------


def _align_recent_series(
    df_daily: pl.DataFrame,
    vehicle_ids: np.ndarray,
    dt_fim: date,
    num_days: int,
) -> Tuple[pl.DataFrame, np.ndarray]:
    """
    Descomprime e alinha séries recentes a partir de dt_fim global.

    O armazenamento é compacto (sem zeros). Reconstrói a grelha
    completa [dt_fim - num_days + 1, dt_fim] para cada veículo,
    preenchendo com 0.0 onde não há dados.

    Returns
    -------
    df_recent : pl.DataFrame
        Séries alinhadas (num_days registos por veículo): veiculo_id, data, value
    valid_ids : np.ndarray
        IDs dos veículos (todos os fornecidos — zeros preenchidos)
    """
    dt_inicio = dt_fim - timedelta(days=num_days - 1)

    # Grelha completa: todos os veículos × todas as datas
    all_dates = pl.DataFrame(
        {"data": pl.date_range(dt_inicio, dt_fim, eager=True)}
    )
    all_vids = pl.DataFrame(
        {"veiculo_id": vehicle_ids.tolist()}
    )
    grid = all_vids.join(all_dates, how="cross")

    # Join com dados existentes, preencher zeros
    df_recent = (
        grid
        .join(df_daily, on=["veiculo_id", "data"], how="left")
        .with_columns(pl.col("value").fill_null(0.0))
        .sort("veiculo_id", "data")
    )

    return df_recent, vehicle_ids


# ------------------------------------------------------------------
# Processamento: converter para DataInput e predizer
# ------------------------------------------------------------------


def _run_prediction_sync(
    target: str,
    df_profile: pd.DataFrame,
    df_recent_pd: pd.DataFrame,
    upper: np.ndarray,
    onnx_filename: Optional[str] = None,
    version_id: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Executa pipeline completo de predição (sync, para run_in_executor).

    Se ``version_id`` estiver disponível, carrega ONNX do bundle versionado.
    Caso contrário, se ``onnx_filename`` for fornecido, usa esse ficheiro.
    """
    from api.config.dataset_config import DatasetConfig
    from api.config.output_config import OutputConfig
    from api.config.predictor_config import PredictorConfig
    from api.config.training_config import TrainingConfig
    from moviasai.data.dataset import GenerateDataInput
    from moviasai.data.normalization import ProfileDatasetNormalizer
    from moviasai.forecasting.predictor import ForecastingPredictor

    dataset_cfg = DatasetConfig.from_yaml(_config_path("dataset_config.yaml"))
    training_cfg = TrainingConfig.from_yaml(_config_path("training_config.yaml"))
    output_cfg = OutputConfig.from_yaml(_config_path("output_config.yaml"))
    predictor_cfg = PredictorConfig.from_yaml(_config_path("predictor_config.yaml"))

    # Resolver path do modelo ONNX
    if version_id is not None:
        from moviasai.bundle import ModelBundle
        onnx_path = (
            Path(output_cfg.models.forecasting)
            / target
            / version_id
            / ModelBundle.MODEL_FILENAME
        )
    elif onnx_filename is not None:
        setattr(predictor_cfg.models, target, onnx_filename)
        onnx_path = predictor_cfg.model_path(target, output_cfg.models.forecasting)
    else:
        onnx_path = predictor_cfg.model_path(target, output_cfg.models.forecasting)

    # Gerar DataInput
    data_input = GenerateDataInput(
        target=target,
        df_profile=df_profile,
        df_recent=df_recent_pd,
        upper=upper,
    ).generate()

    # Normalizar
    normalizer = ProfileDatasetNormalizer(
        cv_max=training_cfg.normalization.cv_max,
        ratio_max=training_cfg.normalization.ratio_max,
    )
    data_input_norm = normalizer.fit_normalize_data_input(data_input)

    # Predição ONNX
    predictor = ForecastingPredictor(
        onnx_path=onnx_path,
        horizon_weeks=dataset_cfg.horizon_weeks,
    )
    return predictor.predict(data_input_norm)


async def fill_actual_from_daily_activity(
    session: AsyncSession,
    target: str,
    coverage_dt_fim: Optional[date] = None,
) -> Dict[str, int]:
    """
    Preenche a coluna ``actual`` das predições existentes (daily e heads)
    com valores reais do daily_activity.

    Critério: se a data (daily) ou dt_fim (heads) está dentro do período
    coberto por ``coverage_dt_fim``, preenche actual.
    Dias sem linha em daily_activity são tratados como 0.

    Se ``coverage_dt_fim`` não for fornecido, usa ``ProfileMetadata.dt_fim``.

    Returns: {"daily_updated": N, "heads_updated": N}
    """
    target_col = "km" if target == "km" else "h"

    # Obter dt_fim de cobertura
    if coverage_dt_fim is None:
        profile_meta = await _load_profile_metadata(session)
        coverage_dt_fim = profile_meta.dt_fim

    # --- Daily: preencher actual onde ainda é NULL e data <= coverage ---
    stmt_daily = (
        select(PredictionDaily.id, PredictionDaily.veiculo_id, PredictionDaily.data)
        .where(
            PredictionDaily.target == target,
            PredictionDaily.actual.is_(None),
            PredictionDaily.data <= coverage_dt_fim,
        )
    )
    result = await session.execute(stmt_daily)
    pending_daily = result.all()

    daily_updated = 0
    if pending_daily:
        vids = list({r[1] for r in pending_daily})
        dates = list({r[2] for r in pending_daily})
        min_date, max_date = min(dates), max(dates)

        result = await session.execute(
            select(
                DailyActivity.veiculo_id,
                DailyActivity.data,
                getattr(DailyActivity, target_col),
            )
            .where(
                DailyActivity.veiculo_id.in_(vids),
                DailyActivity.data >= min_date,
                DailyActivity.data <= max_date,
            )
        )
        actual_map = {(r[0], r[1]): r[2] for r in result.all()}

        for pred_id, vid, dt in pending_daily:
            # Dia coberto → actual = valor real ou 0 se sem registo
            val = actual_map.get((vid, dt), 0.0)
            await session.execute(
                update(PredictionDaily)
                .where(PredictionDaily.id == pred_id)
                .values(actual=float(val))
            )
            daily_updated += 1

    # --- Heads: preencher actual onde dt_fim <= coverage ---
    stmt_heads = (
        select(
            PredictionHead.id,
            PredictionHead.veiculo_id,
            PredictionHead.dt_inicio,
            PredictionHead.dt_fim,
        )
        .where(
            PredictionHead.target == target,
            PredictionHead.actual.is_(None),
            PredictionHead.dt_fim <= coverage_dt_fim,
        )
    )
    result = await session.execute(stmt_heads)
    pending_heads = result.all()

    heads_updated = 0
    if pending_heads:
        vids = list({r[1] for r in pending_heads})
        all_min = min(r[2] for r in pending_heads)
        all_max = max(r[3] for r in pending_heads)

        result = await session.execute(
            select(
                DailyActivity.veiculo_id,
                DailyActivity.data,
                getattr(DailyActivity, target_col),
            )
            .where(
                DailyActivity.veiculo_id.in_(vids),
                DailyActivity.data >= all_min,
                DailyActivity.data <= all_max,
            )
        )
        from collections import defaultdict
        daily_by_vid: Dict[int, Dict[date, float]] = defaultdict(dict)
        for vid, dt, val in result.all():
            daily_by_vid[vid][dt] = val

        for pred_id, vid, dt_inicio, dt_fim in pending_heads:
            vid_data = daily_by_vid.get(vid, {})
            expected_days = (dt_fim - dt_inicio).days + 1
            # Bloco coberto → somar valores reais (dias sem registo = 0)
            actual_sum = sum(
                vid_data.get(dt_inicio + timedelta(days=d), 0.0)
                for d in range(expected_days)
            )
            await session.execute(
                update(PredictionHead)
                .where(PredictionHead.id == pred_id)
                .values(actual=float(actual_sum))
            )
            heads_updated += 1

    await session.flush()
    logger.info(
        f"[{target}] Actual preenchido: {daily_updated} daily, {heads_updated} heads"
    )
    return {"daily_updated": daily_updated, "heads_updated": heads_updated}


def _compute_all_head_dates(
    iter_dt_fim: date,
    block_days: int,
    n_heads: int,
) -> List[Tuple[date, date]]:
    """
    Calcula (dt_inicio, dt_fim) para cada um dos n heads da saída ONNX.

    head[0] → [iter_dt_fim+1, iter_dt_fim+block_days]
    head[1] → [iter_dt_fim+block_days+1, iter_dt_fim+2*block_days]
    ...
    """
    result = []
    cursor = iter_dt_fim + timedelta(days=1)
    for _ in range(n_heads):
        bloco_inicio = cursor
        bloco_fim = cursor + timedelta(days=block_days - 1)
        result.append((bloco_inicio, bloco_fim))
        cursor = bloco_fim + timedelta(days=1)
    return result


def _compute_daily_dates(
    dt_fim: date,
    daily_horizon: int,
) -> List[date]:
    """Datas para predições daily a partir de dt_fim + 1."""
    start = dt_fim + timedelta(days=1)
    return [start + timedelta(days=i) for i in range(daily_horizon)]


# ------------------------------------------------------------------
# DB: persistir predições (rolling)
# ------------------------------------------------------------------


async def _persist_heads(
    session: AsyncSession,
    target: str,
    vehicle_ids: np.ndarray,
    y_heads: np.ndarray,
    head_dates: List[Tuple[date, date]],
) -> int:
    """
    Persiste heads via upsert sem nunca apagar registos.

    - ``actual IS NOT NULL`` → preserva (não toca)
    - ``actual IS NULL`` → atualiza ``prediction``
    - Não existe → insere novo registo
    """
    n_heads = len(head_dates)

    # Montar dict (veiculo_id, dt_inicio) → (dt_fim, prediction)
    new_data: Dict[Tuple[int, date], Tuple[date, float]] = {}
    for head_idx in range(n_heads):
        dt_inicio, dt_fim = head_dates[head_idx]
        for i, vid in enumerate(vehicle_ids):
            val = float(y_heads[i, head_idx])
            if val != val:  # NaN
                continue
            new_data[(int(vid), dt_inicio)] = (dt_fim, val)

    if not new_data:
        return 0

    all_vids = list({k[0] for k in new_data})
    all_dt_inicios = list({k[1] for k in new_data})

    # Buscar registos existentes para estas chaves
    result = await session.execute(
        select(
            PredictionHead.veiculo_id,
            PredictionHead.dt_inicio,
            PredictionHead.actual,
        ).where(
            PredictionHead.target == target,
            PredictionHead.veiculo_id.in_(all_vids),
            PredictionHead.dt_inicio.in_(all_dt_inicios),
        )
    )
    existing = {(r[0], r[1]): r[2] for r in result.all()}

    # Separar em: skip (actual preenchido), update (actual NULL), insert (novo)
    to_update = []
    to_insert = []
    for (vid, dt_inicio), (dt_fim, pred) in new_data.items():
        if (vid, dt_inicio) in existing:
            if existing[(vid, dt_inicio)] is not None:
                continue  # actual preenchido → não toca
            to_update.append((vid, dt_inicio, dt_fim, pred))
        else:
            to_insert.append({
                "veiculo_id": vid,
                "target": target,
                "dt_inicio": dt_inicio,
                "dt_fim": dt_fim,
                "prediction": pred,
            })

    # Batch update: actual IS NULL → atualizar prediction
    for vid, dt_inicio, dt_fim, pred in to_update:
        await session.execute(
            update(PredictionHead)
            .where(
                PredictionHead.veiculo_id == vid,
                PredictionHead.target == target,
                PredictionHead.dt_inicio == dt_inicio,
            )
            .values(prediction=pred, dt_fim=dt_fim)
        )

    # Bulk insert novos
    if to_insert:
        await session.execute(insert(PredictionHead), to_insert)

    await session.flush()
    return len(to_update) + len(to_insert)


async def _persist_daily_predictions(
    session: AsyncSession,
    target: str,
    vehicle_ids: np.ndarray,
    y_daily: np.ndarray,
    daily_dates: List[date],
) -> int:
    """
    Persiste predições daily via upsert sem nunca apagar registos.

    - ``actual IS NOT NULL`` → preserva (não toca)
    - ``actual IS NULL`` → atualiza ``prediction``
    - Não existe → insere novo registo
    """
    # Montar dict (veiculo_id, data) → prediction
    new_data: Dict[Tuple[int, date], float] = {}
    for i, vid in enumerate(vehicle_ids):
        for j, dt in enumerate(daily_dates):
            val = float(y_daily[i, j])
            if val != val:  # NaN
                continue
            new_data[(int(vid), dt)] = val

    if not new_data:
        return 0

    all_vids = list({k[0] for k in new_data})
    all_dates = list({k[1] for k in new_data})

    # Buscar registos existentes para estas chaves
    result = await session.execute(
        select(
            PredictionDaily.veiculo_id,
            PredictionDaily.data,
            PredictionDaily.actual,
        ).where(
            PredictionDaily.target == target,
            PredictionDaily.veiculo_id.in_(all_vids),
            PredictionDaily.data.in_(all_dates),
        )
    )
    existing = {(r[0], r[1]): r[2] for r in result.all()}

    # Separar: skip (actual preenchido), update (actual NULL), insert (novo)
    to_update = []
    to_insert = []
    for (vid, dt), pred in new_data.items():
        if (vid, dt) in existing:
            if existing[(vid, dt)] is not None:
                continue  # actual preenchido → não toca
            to_update.append((vid, dt, pred))
        else:
            to_insert.append({
                "veiculo_id": vid,
                "target": target,
                "data": dt,
                "prediction": pred,
            })

    # Batch update: actual IS NULL → atualizar prediction
    for vid, dt, pred in to_update:
        await session.execute(
            update(PredictionDaily)
            .where(
                PredictionDaily.veiculo_id == vid,
                PredictionDaily.target == target,
                PredictionDaily.data == dt,
            )
            .values(prediction=pred)
        )

    # Bulk insert novos
    if to_insert:
        await session.execute(insert(PredictionDaily), to_insert)

    await session.flush()
    return len(to_update) + len(to_insert)


# ------------------------------------------------------------------
# Helpers: pivot de perfil para predição
# ------------------------------------------------------------------


def pivot_profile_for_target(df_long: pd.DataFrame, target: str) -> pd.DataFrame:
    """Pivota perfil long → wide para um target (input do ONNX)."""
    df_t = df_long[df_long["feature_class"] == target].copy()
    if df_t.empty:
        return pd.DataFrame(columns=["veiculo_id"])
    df_wide = df_t.pivot(
        index="veiculo_id", columns="feature", values="valor",
    ).reset_index()
    df_wide.columns.name = None
    return df_wide


def extract_upper(
    df_meta: pd.DataFrame, vehicle_ids: np.ndarray,
) -> np.ndarray:
    """Extrai upper bounds de metadata alinhados a vehicle_ids."""
    if df_meta is None or df_meta.empty:
        return np.full(len(vehicle_ids), np.nan)
    upper_map = df_meta.set_index("veiculo_id")["upper"]
    return upper_map.reindex(vehicle_ids).values.astype(np.float64)


async def predict_single_iteration(
    session: AsyncSession,
    target: str,
    df_profile: pd.DataFrame,
    upper: np.ndarray,
    vehicle_ids: np.ndarray,
    iter_dt_fim: date,
    onnx_filename: Optional[str],
    version_id: Optional[str],
    num_days: int,
    block_days: int,
    n_heads: int,
    daily_horizon: int,
) -> Dict:
    """
    Uma iteração de predição para um target: carrega série recente,
    executa ONNX e persiste heads + daily (upsert sem delete).
    """
    df_daily = await _load_daily_activity(
        session, target, vehicle_ids.tolist(), iter_dt_fim, num_days,
    )
    df_recent, _ = _align_recent_series(df_daily, vehicle_ids, iter_dt_fim, num_days)
    df_recent_pd = df_recent.to_pandas()
    df_recent_pd["data"] = pd.to_datetime(df_recent_pd["data"])
    df_recent_pd = df_recent_pd.rename(columns={"value": f"{target}_dia_clean"})

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        _run_prediction_sync,
        target, df_profile, df_recent_pd, upper, onnx_filename, version_id,
    )

    head_dates = _compute_all_head_dates(iter_dt_fim, block_days, n_heads)
    head_count = await _persist_heads(
        session, target, vehicle_ids, results["y_heads"], head_dates,
    )

    daily_dates = _compute_daily_dates(iter_dt_fim, daily_horizon)
    daily_count = await _persist_daily_predictions(
        session, target, vehicle_ids, results["y_daily"], daily_dates,
    )

    return {"head_count": head_count, "daily_count": daily_count}


# ------------------------------------------------------------------
# Helpers: verificar veículos não encontrados
# ------------------------------------------------------------------


async def _check_missing_vehicles(
    session: AsyncSession,
    requested_ids: List[int],
    found_ids: set,
    target: str,
) -> List[Dict]:
    """
    Para cada veículo pedido mas não encontrado no perfil,
    verifica nos metadados se foi excluído (e porquê).
    """
    quality_labels = {
        0: "VALID",
        1: "OUTLIER",
        2: "NOT_MODELABLE",
        3: "EMPTY",
    }

    missing_ids = [vid for vid in requested_ids if vid not in found_ids]
    if not missing_ids:
        return []

    model_cls = VehicleMetadataKm if target == "km" else VehicleMetadataH
    stmt = select(
        model_cls.veiculo_id,
        model_cls.quality,
        model_cls.quality_reason,
    ).where(model_cls.veiculo_id.in_(missing_ids))
    result = await session.execute(stmt)
    metadata_map = {r[0]: (r[1], r[2]) for r in result.all()}

    not_found = []
    for vid in missing_ids:
        if vid in metadata_map:
            quality, reason = metadata_map[vid]
            label = quality_labels.get(quality, f"quality={quality}")
            msg = f"Veículo excluído do perfil ({label})"
            if reason:
                msg += f": {reason}"
            not_found.append({"veiculo_id": vid, "reason": msg})
        else:
            not_found.append({
                "veiculo_id": vid,
                "reason": "Veículo não existe ou não tem atividade registrada",
            })

    return not_found


# ------------------------------------------------------------------
# API pública
# ------------------------------------------------------------------


async def predict(
    target: str,
    vehicle_ids: Optional[List[int]] = None,
) -> Dict:
    """
    Lê predições persistidas (geradas durante a ingestão).

    Se ``vehicle_ids`` for None, retorna todos os veículos.
    Retorna dict com daily, heads, type_probabilities e not_found.
    """
    async with session_context() as session:
        # 1. Consultar predições daily (só futuras: actual IS NULL)
        stmt_daily = (
            select(
                PredictionDaily.veiculo_id,
                PredictionDaily.data,
                PredictionDaily.prediction,
            )
            .where(
                PredictionDaily.target == target,
                PredictionDaily.actual.is_(None),
            )
            .order_by(PredictionDaily.veiculo_id, PredictionDaily.data)
        )
        if vehicle_ids:
            stmt_daily = stmt_daily.where(PredictionDaily.veiculo_id.in_(vehicle_ids))

        result = await session.execute(stmt_daily)
        daily_rows = result.all()

        pred_daily = [
            {"veiculo_id": r[0], "data": r[1], "prediction": r[2]}
            for r in daily_rows
        ]

        # 2. Consultar predições heads (só futuras: actual IS NULL)
        stmt_heads = (
            select(
                PredictionHead.veiculo_id,
                PredictionHead.dt_inicio,
                PredictionHead.dt_fim,
                PredictionHead.prediction,
            )
            .where(
                PredictionHead.target == target,
                PredictionHead.actual.is_(None),
            )
            .order_by(PredictionHead.veiculo_id, PredictionHead.dt_inicio)
        )
        if vehicle_ids:
            stmt_heads = stmt_heads.where(PredictionHead.veiculo_id.in_(vehicle_ids))

        result = await session.execute(stmt_heads)
        head_rows = result.all()

        pred_heads = [
            {
                "veiculo_id": r[0],
                "dt_inicio": r[1],
                "dt_fim": r[2],
                "prediction": r[3],
            }
            for r in head_rows
        ]

        # 3. Veículos encontrados
        found_vids = set(r[0] for r in daily_rows) | set(r[0] for r in head_rows)

        # 4. Probabilidades de tipo
        type_probs = await _load_type_probabilities(
            session, list(found_vids),
        )
        type_probs_list = [
            {"veiculo_id": int(vid), "probabilities": type_probs.get(int(vid), {})}
            for vid in sorted(found_vids)
        ]

        # 5. Rótulos de qualidade
        meta = await _load_metadata(session, target, list(found_vids))
        quality_labels = {
            0: "Válida",
            1: "Outlier",
            2: "Não modelável",
            3: "Vazia",
        }
        vehicle_quality = []
        for _, row in meta.iterrows():
            vid = int(row["veiculo_id"])
            if vid not in found_vids:
                continue
            q = row["quality"]
            label = quality_labels.get(q, f"Desconhecida ({q})") if pd.notna(q) else "Desconhecida"
            reason = str(row["quality_reason"]) if pd.notna(row["quality_reason"]) else None
            vehicle_quality.append({
                "veiculo_id": vid,
                "quality": label,
                "quality_reason": reason,
            })

        # 6. Veículos não encontrados
        not_found = []
        if vehicle_ids:
            not_found = await _check_missing_vehicles(
                session, vehicle_ids, found_vids, target,
            )

    return {
        "target": target,
        "predictions_daily": pred_daily,
        "predictions_heads": pred_heads,
        "type_probabilities": type_probs_list,
        "vehicle_quality": vehicle_quality,
        "not_found": not_found,
    }


async def predict_and_persist_with_session(
    session: AsyncSession,
    target: str,
) -> Dict:
    """
    Executa predição rolling para todos os veículos do perfil.

    Dados novos (após dt_fim do perfil anterior) são divididos em blocos
    de ``block_days`` dias. Para cada bloco (iteração):
    1. Preenche ``actual`` das predições anteriores cobertas por iter_dt_fim
    2. A janela recente avança ``block_days`` dias
    3. O ONNX é re-executado
    4. Persiste n heads + daily_horizon dias (upsert sem delete)

    Usa a sessão fornecida (sem commit) — o chamador controla a transação.
    """
    from api.config.dataset_config import DatasetConfig

    dataset_cfg = DatasetConfig.from_yaml(_config_path("dataset_config.yaml"))
    num_days = dataset_cfg.num_weeks_recent * 7
    block_days = dataset_cfg.block_days
    n_heads = dataset_cfg.n_heads

    # 1. Carregar todos os perfis
    df_profile = await _load_vehicle_profiles(session, target)
    if df_profile.empty:
        return {
            "target": target,
            "total_vehicles": 0,
            "daily_rows_persisted": 0,
            "head_rows_persisted": 0,
        }

    profile_vehicle_ids = df_profile["veiculo_id"].values

    # 2. Metadados (upper)
    meta = await _load_metadata(session, target, profile_vehicle_ids.tolist())
    upper_map = meta.set_index("veiculo_id")["upper"]
    upper = upper_map.reindex(profile_vehicle_ids).values.astype(np.float64)

    # 3. dt_fim global + modelo vigente
    profile_meta = await _load_profile_metadata(session)
    dt_fim = profile_meta.dt_fim
    onnx_filename, version_id = await _resolve_onnx_filename(session, target)

    # 4. Calcular número de iterações rolling (m)
    prev_result = await session.execute(
        select(ProfileMetadata)
        .where(ProfileMetadata.id < profile_meta.id)
        .order_by(ProfileMetadata.id.desc())
        .limit(1)
    )
    prev_meta = prev_result.scalar_one_or_none()

    if prev_meta is not None:
        prev_dt_fim = prev_meta.dt_fim
        new_days = (dt_fim - prev_dt_fim).days
        m = max(1, new_days // block_days)
    else:
        # Primeira ingestão: 1 iteração
        m = 1

    logger.info(
        f"[{target}] Rolling prediction: {m} iterações de {block_days} dias, "
        f"n_heads={n_heads}"
    )

    total_head_count = 0
    total_daily_count = 0

    # 5. Loop rolling
    for block_idx in range(m):
        # iter_dt_fim: fim da janela recente para esta iteração
        if prev_meta is not None and m > 1:
            iter_dt_fim = prev_meta.dt_fim + timedelta(days=(block_idx + 1) * block_days)
            if iter_dt_fim > dt_fim:
                iter_dt_fim = dt_fim
        else:
            iter_dt_fim = dt_fim

        # 5a. Preencher actual das predições cobertas até iter_dt_fim
        await fill_actual_from_daily_activity(
            session, target, coverage_dt_fim=iter_dt_fim,
        )

        # 5b. Carregar atividade diária com janela até iter_dt_fim
        df_daily = await _load_daily_activity(
            session, target, profile_vehicle_ids.tolist(), iter_dt_fim, num_days,
        )

        # 5c. Alinhar séries recentes
        df_recent, valid_ids = _align_recent_series(
            df_daily, profile_vehicle_ids, iter_dt_fim, num_days,
        )

        # 5d. Converter para pandas
        df_recent_pd = df_recent.to_pandas()
        df_recent_pd["data"] = pd.to_datetime(df_recent_pd["data"])
        df_recent_pd = df_recent_pd.rename(columns={"value": f"{target}_dia_clean"})

        # 5e. Executar ONNX
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            _run_prediction_sync,
            target, df_profile, df_recent_pd, upper, onnx_filename, version_id,
        )

        # 5f. Persistir heads (upsert)
        y_heads = results["y_heads"]
        head_dates = _compute_all_head_dates(iter_dt_fim, block_days, n_heads)
        head_count = await _persist_heads(
            session, target, profile_vehicle_ids,
            y_heads, head_dates,
        )
        total_head_count += head_count

        # 5g. Persistir daily (upsert)
        y_daily = results["y_daily"]
        daily_dates = _compute_daily_dates(iter_dt_fim, dataset_cfg.daily_horizon)
        daily_count = await _persist_daily_predictions(
            session, target, profile_vehicle_ids, y_daily, daily_dates,
        )
        total_daily_count += daily_count

        logger.info(
            f"[{target}] Iteração {block_idx + 1}/{m}: "
            f"iter_dt_fim={iter_dt_fim}, heads={head_count}, daily={daily_count}"
        )

    logger.info(
        f"[{target}] Predições persistidas: "
        f"{total_daily_count} daily, {total_head_count} heads "
        f"para {len(profile_vehicle_ids)} veículos ({m} iterações)"
    )

    return {
        "target": target,
        "total_vehicles": int(len(profile_vehicle_ids)),
        "daily_rows_persisted": total_daily_count,
        "head_rows_persisted": total_head_count,
    }


async def backtest(
    veiculo_id: int,
    target: str,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    max_heads: Optional[int] = None,
) -> Optional[Dict]:
    """
    Retorna predições com ``actual`` preenchido (comparação já feita na ingestão).

    Filtros opcionais:
    - ``date_from`` / ``date_to``: limitar período do gráfico diário.
    - ``max_heads``: retornar apenas os K blocos mais recentes.
    """
    async with session_context() as session:
        # Daily: só onde actual foi preenchido
        daily_q = (
            select(
                PredictionDaily.data,
                PredictionDaily.actual,
                PredictionDaily.prediction,
            )
            .where(
                PredictionDaily.veiculo_id == veiculo_id,
                PredictionDaily.target == target,
                PredictionDaily.actual.is_not(None),
            )
        )
        if date_from is not None:
            daily_q = daily_q.where(PredictionDaily.data >= date_from)
        if date_to is not None:
            daily_q = daily_q.where(PredictionDaily.data <= date_to)
        daily_q = daily_q.order_by(PredictionDaily.data)

        result = await session.execute(daily_q)
        daily_rows = result.all()

        # Heads: só onde actual foi preenchido, limitar por max_heads (mais recentes)
        heads_q = (
            select(
                PredictionHead.dt_inicio,
                PredictionHead.dt_fim,
                PredictionHead.actual,
                PredictionHead.prediction,
            )
            .where(
                PredictionHead.veiculo_id == veiculo_id,
                PredictionHead.target == target,
                PredictionHead.actual.is_not(None),
            )
            .order_by(PredictionHead.dt_inicio.desc())
        )
        if max_heads is not None:
            heads_q = heads_q.limit(max_heads)

        result = await session.execute(heads_q)
        head_rows = list(reversed(result.all()))  # reverter para ordem cronológica

        if not daily_rows and not head_rows:
            return None

        daily_comparison = [
            {"data": r[0], "actual": r[1], "predicted": r[2]}
            for r in daily_rows
        ]
        heads_comparison = [
            {"dt_inicio": r[0], "dt_fim": r[1], "actual": r[2], "predicted": r[3]}
            for r in head_rows
        ]

    return {
        "veiculo_id": veiculo_id,
        "target": target,
        "daily": daily_comparison,
        "heads": heads_comparison,
    }
