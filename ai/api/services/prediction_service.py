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
from sqlalchemy import delete, insert, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import session_context
from api.models import (
    DailyActivity,
    PredictionDaily,
    PredictionHead,
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
    """Carrega probabilidades de tipo (feature_class='type') para veículos."""
    stmt = select(
        VehicleProfileFeature.veiculo_id,
        VehicleProfileFeature.feature,
        VehicleProfileFeature.valor,
    ).where(
        VehicleProfileFeature.feature_class == "type",
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


async def _load_daily_activity(
    session: AsyncSession,
    target: str,
    vehicle_ids: List[int],
    num_days: int,
) -> pl.DataFrame:
    """
    Carrega atividade diária recente para os veículos indicados.

    Carrega mais dias do que o necessário (num_days + 7) para compensar
    o truncamento ao último domingo.
    """
    target_col = "km" if target == "km" else "h"
    fetch_days = num_days + 7  # margem para truncamento ao domingo

    # Gerar placeholders para IN clause (SQLite não suporta IN com tuple)
    placeholders = ", ".join(f":id_{i}" for i in range(len(vehicle_ids)))
    params = {f"id_{i}": vid for i, vid in enumerate(vehicle_ids)}

    stmt = text(f"""
        SELECT veiculo_id, data, {target_col} AS value
        FROM daily_activity
        WHERE veiculo_id IN ({placeholders})
        AND data >= (
            SELECT DATE(MAX(da2.data), '-{fetch_days} days')
            FROM daily_activity da2
            WHERE da2.veiculo_id = daily_activity.veiculo_id
        )
        ORDER BY veiculo_id, data
    """).bindparams(**params)

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
# Processamento: alinhar séries recentes
# ------------------------------------------------------------------


def _align_recent_series(
    df_daily: pl.DataFrame,
    vehicle_ids: np.ndarray,
    num_days: int,
) -> Tuple[pl.DataFrame, np.ndarray, Dict[int, date]]:
    """
    Trunca ao último domingo e filtra veículos com série completa.

    Returns
    -------
    df_recent : pl.DataFrame
        Séries alinhadas (num_days registos por veículo)
    valid_ids : np.ndarray
        IDs dos veículos com série completa
    last_dates : dict
        {veiculo_id: última data (domingo)} por veículo válido
    """
    df = df_daily.filter(
        pl.col("veiculo_id").is_in(vehicle_ids.tolist())
    ).sort("veiculo_id", "data")

    # Adicionar dia da semana (Polars: 1=seg, 7=dom)
    df = df.with_columns(pl.col("data").dt.weekday().alias("_dow"))

    # Truncar ao último domingo e pegar os últimos num_days
    df_recent = (
        df
        .group_by("veiculo_id")
        .map_groups(lambda g: (
            g
            .filter(pl.col("data") <= g["data"].filter(g["_dow"] == 7).last())
            .tail(num_days)
        ))
        .drop("_dow")
        .sort("veiculo_id", "data")
    )

    # Validar séries completas
    counts = df_recent.group_by("veiculo_id").len()
    valid_series = counts.filter(pl.col("len") == num_days)
    valid_ids = valid_series["veiculo_id"].to_numpy()

    # Filtrar para veículos válidos
    df_recent = df_recent.filter(pl.col("veiculo_id").is_in(valid_ids.tolist()))

    # Última data (domingo) por veículo
    last_dates = dict(
        df_recent
        .group_by("veiculo_id")
        .agg(pl.col("data").max().alias("last_date"))
        .iter_rows()
    )

    return df_recent, valid_ids, last_dates


# ------------------------------------------------------------------
# Processamento: converter para DataInput e predizer
# ------------------------------------------------------------------


def _run_prediction_sync(
    target: str,
    df_profile: pd.DataFrame,
    df_recent_pd: pd.DataFrame,
    upper: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Executa pipeline completo de predição (sync, para run_in_executor)."""
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
    predictor = ForecastingPredictor.from_config(
        target=target,
        predictor_cfg=predictor_cfg,
        output_cfg=output_cfg,
        horizon_weeks=dataset_cfg.horizon_weeks,
    )
    return predictor.predict(data_input_norm)


def _compute_prediction_dates(
    last_dates: Dict[int, date],
    vehicle_ids: np.ndarray,
    daily_horizon: int,
    horizon_weeks,
) -> Tuple[Dict[int, List[date]], Dict[int, List[Tuple[int, date, date]]]]:
    """
    Calcula datas para predições daily e heads a partir do último domingo.

    Returns
    -------
    daily_dates : {veiculo_id: [d1, d2, ...]}
    head_ranges : {veiculo_id: [(head_num, dt_inicio, dt_fim), ...]}
    """
    # Normalizar horizon_weeks
    if isinstance(horizon_weeks, int):
        head_weeks = [1] * horizon_weeks
    else:
        head_weeks = list(horizon_weeks)

    daily_dates: Dict[int, List[date]] = {}
    head_ranges: Dict[int, List[Tuple[int, date, date]]] = {}

    for vid in vehicle_ids:
        last_sunday = last_dates[vid]
        start = last_sunday + timedelta(days=1)  # segunda-feira seguinte

        # Daily: daily_horizon dias a partir de start
        daily_dates[vid] = [start + timedelta(days=i) for i in range(daily_horizon)]

        # Heads: cada head cobre N semanas consecutivas
        heads = []
        cursor = start
        for h_idx, weeks in enumerate(head_weeks, start=1):
            dt_inicio = cursor
            dt_fim = cursor + timedelta(days=weeks * 7 - 1)
            heads.append((h_idx, dt_inicio, dt_fim))
            cursor = dt_fim + timedelta(days=1)
        head_ranges[vid] = heads

    return daily_dates, head_ranges


# ------------------------------------------------------------------
# DB: persistir predições
# ------------------------------------------------------------------


async def _persist_predictions(
    session: AsyncSession,
    target: str,
    vehicle_ids: np.ndarray,
    y_daily: np.ndarray,
    y_heads: np.ndarray,
    daily_dates: Dict[int, List[date]],
    head_ranges: Dict[int, List[Tuple[int, date, date]]],
) -> Tuple[int, int]:
    """
    Apaga predições futuras (a partir da data mínima das novas) e insere as novas.

    Predições anteriores (cobertas por dados reais) são mantidas para backtest.

    Returns (daily_count, head_count).
    """
    # Determinar a data de corte: menor data das novas predições daily
    cutoff = min(dt for dates in daily_dates.values() for dt in dates)

    # Apagar apenas predições a partir do corte (futuro)
    await session.execute(
        delete(PredictionDaily).where(
            PredictionDaily.target == target,
            PredictionDaily.data >= cutoff,
        )
    )
    await session.execute(
        delete(PredictionHead).where(
            PredictionHead.target == target,
            PredictionHead.dt_inicio >= cutoff,
        )
    )

    # Bulk insert daily via core (skip NaN predictions)
    daily_records = []
    for i, vid in enumerate(vehicle_ids):
        dates = daily_dates[vid]
        for j, dt in enumerate(dates):
            val = float(y_daily[i, j])
            if val != val:  # NaN check
                continue
            daily_records.append({
                "veiculo_id": int(vid),
                "target": target,
                "data": dt,
                "prediction": val,
            })

    # Bulk insert heads via core (skip NaN predictions)
    head_records = []
    for i, vid in enumerate(vehicle_ids):
        ranges = head_ranges[vid]
        for j, (head_num, dt_inicio, dt_fim) in enumerate(ranges):
            val = float(y_heads[i, j])
            if val != val:  # NaN check
                continue
            head_records.append({
                "veiculo_id": int(vid),
                "target": target,
                "head": head_num,
                "dt_inicio": dt_inicio,
                "dt_fim": dt_fim,
                "prediction": val,
            })

    if daily_records:
        await session.execute(insert(PredictionDaily), daily_records)
    if head_records:
        await session.execute(insert(PredictionHead), head_records)
    await session.flush()

    return len(daily_records), len(head_records)


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
    Executa predição para veículos do perfil.

    Se ``vehicle_ids`` for None, prediz todos os veículos.
    Retorna dict com daily, heads, type_probabilities e not_found.
    """
    from api.config.dataset_config import DatasetConfig

    dataset_cfg = DatasetConfig.from_yaml(_config_path("dataset_config.yaml"))
    num_days = dataset_cfg.num_weeks_recent * 7

    async with session_context() as session:
        # 1. Carregar perfis
        df_profile = await _load_vehicle_profiles(session, target, vehicle_ids)
        if df_profile.empty:
            not_found = []
            if vehicle_ids:
                not_found = await _check_missing_vehicles(
                    session, vehicle_ids, set(), target,
                )
            return {
                "target": target,
                "predictions_daily": [],
                "predictions_heads": [],
                "type_probabilities": [],
                "not_found": not_found,
            }

        profile_vehicle_ids = df_profile["veiculo_id"].values

        # 2. Carregar metadados (upper)
        meta = await _load_metadata(session, target, profile_vehicle_ids.tolist())
        upper_map = meta.set_index("veiculo_id")["upper"]
        upper = upper_map.reindex(profile_vehicle_ids).values.astype(np.float64)

        # 3. Carregar atividade diária
        df_daily = await _load_daily_activity(
            session, target, profile_vehicle_ids.tolist(), num_days,
        )

        # 4. Alinhar séries recentes
        df_recent, valid_ids, last_dates = _align_recent_series(
            df_daily, profile_vehicle_ids, num_days,
        )

        n_dropped = len(profile_vehicle_ids) - len(valid_ids)
        if n_dropped > 0:
            logger.warning(f"[{target}] {n_dropped} veículos descartados (série < {num_days} dias)")

        # Filtrar profile e upper para veículos válidos
        mask = np.isin(profile_vehicle_ids, valid_ids)
        df_profile = df_profile[mask].reset_index(drop=True)
        upper = upper[mask]
        profile_vehicle_ids = df_profile["veiculo_id"].values

        # Converter df_recent para pandas (renomear value → {target}_dia_clean)
        df_recent_pd = df_recent.to_pandas()
        df_recent_pd["data"] = pd.to_datetime(df_recent_pd["data"])
        df_recent_pd = df_recent_pd.rename(columns={"value": f"{target}_dia_clean"})

        # 5. Predição (CPU-bound → executor)
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            _run_prediction_sync,
            target, df_profile, df_recent_pd, upper,
        )

        y_daily = results["y_daily"]
        y_heads = results["y_heads"]

        # 6. Calcular datas
        daily_dates, head_ranges = _compute_prediction_dates(
            last_dates, profile_vehicle_ids,
            dataset_cfg.daily_horizon, dataset_cfg.horizon_weeks,
        )

        # 7. Montar resposta daily
        pred_daily = []
        for i, vid in enumerate(profile_vehicle_ids):
            dates = daily_dates[int(vid)]
            for j, dt in enumerate(dates):
                pred_daily.append({
                    "veiculo_id": int(vid),
                    "data": dt,
                    "prediction": float(y_daily[i, j]),
                })

        # 8. Montar resposta heads
        pred_heads = []
        for i, vid in enumerate(profile_vehicle_ids):
            ranges = head_ranges[int(vid)]
            for j, (head_num, dt_inicio, dt_fim) in enumerate(ranges):
                pred_heads.append({
                    "veiculo_id": int(vid),
                    "head": head_num,
                    "dt_inicio": dt_inicio,
                    "dt_fim": dt_fim,
                    "prediction": float(y_heads[i, j]),
                })

        # 9. Probabilidades de tipo
        type_probs = await _load_type_probabilities(
            session, profile_vehicle_ids.tolist(),
        )
        type_probs_list = [
            {"veiculo_id": int(vid), "probabilities": type_probs.get(int(vid), {})}
            for vid in profile_vehicle_ids
        ]

        # 10. Veículos não encontrados
        not_found = []
        if vehicle_ids:
            not_found = await _check_missing_vehicles(
                session, vehicle_ids, set(profile_vehicle_ids.tolist()), target,
            )

    return {
        "target": target,
        "predictions_daily": pred_daily,
        "predictions_heads": pred_heads,
        "type_probabilities": type_probs_list,
        "not_found": not_found,
    }


async def predict_and_persist_with_session(
    session: AsyncSession,
    target: str,
) -> Dict:
    """
    Executa predição para todos os veículos do perfil e persiste resultados.

    Usa a sessão fornecida (sem commit) — o chamador controla a transação.
    """
    from api.config.dataset_config import DatasetConfig

    dataset_cfg = DatasetConfig.from_yaml(_config_path("dataset_config.yaml"))
    num_days = dataset_cfg.num_weeks_recent * 7

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

    # 2. Metadados
    meta = await _load_metadata(session, target, profile_vehicle_ids.tolist())
    upper_map = meta.set_index("veiculo_id")["upper"]
    upper = upper_map.reindex(profile_vehicle_ids).values.astype(np.float64)

    # 3. Atividade diária
    df_daily = await _load_daily_activity(
        session, target, profile_vehicle_ids.tolist(), num_days,
    )

    # 4. Alinhar séries recentes
    df_recent, valid_ids, last_dates = _align_recent_series(
        df_daily, profile_vehicle_ids, num_days,
    )

    n_dropped = len(profile_vehicle_ids) - len(valid_ids)
    if n_dropped > 0:
        logger.warning(f"[{target}] {n_dropped} veículos descartados (série < {num_days} dias)")

    # Filtrar
    mask = np.isin(profile_vehicle_ids, valid_ids)
    df_profile = df_profile[mask].reset_index(drop=True)
    upper = upper[mask]
    profile_vehicle_ids = df_profile["veiculo_id"].values

    # Converter df_recent para pandas
    df_recent_pd = df_recent.to_pandas()
    df_recent_pd["data"] = pd.to_datetime(df_recent_pd["data"])
    df_recent_pd = df_recent_pd.rename(columns={"value": f"{target}_dia_clean"})

    # 5. Predição
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        _run_prediction_sync,
        target, df_profile, df_recent_pd, upper,
    )

    y_daily = results["y_daily"]
    y_heads = results["y_heads"]

    # 6. Calcular datas
    daily_dates, head_ranges = _compute_prediction_dates(
        last_dates, profile_vehicle_ids,
        dataset_cfg.daily_horizon, dataset_cfg.horizon_weeks,
    )

    # 7. Persistir (flush, sem commit)
    daily_count, head_count = await _persist_predictions(
        session, target,
        profile_vehicle_ids, y_daily, y_heads,
        daily_dates, head_ranges,
    )

    logger.info(
        f"[{target}] Predições persistidas: "
        f"{daily_count} daily, {head_count} heads "
        f"para {len(profile_vehicle_ids)} veículos"
    )

    return {
        "target": target,
        "total_vehicles": int(len(profile_vehicle_ids)),
        "daily_rows_persisted": daily_count,
        "head_rows_persisted": head_count,
    }


async def predict_and_persist(target: str) -> Dict:
    """
    Executa predição para todos os veículos do perfil e persiste resultados.

    Atualiza as tabelas predictions_daily e predictions_heads.
    """
    async with session_context() as session:
        result = await predict_and_persist_with_session(session, target)
        await session.commit()
    return result


async def backtest(veiculo_id: int, target: str) -> Optional[Dict]:
    """
    Compara predições persistidas com valores reais de daily_activity.

    Returns None se não houver predições para o veículo.
    """
    target_col = "km" if target == "km" else "h"

    async with session_context() as session:
        # 1. Carregar predições daily
        result = await session.execute(
            select(PredictionDaily.data, PredictionDaily.prediction)
            .where(
                PredictionDaily.veiculo_id == veiculo_id,
                PredictionDaily.target == target,
            )
            .order_by(PredictionDaily.data)
        )
        pred_daily_rows = result.all()

        # 2. Carregar predições heads
        result = await session.execute(
            select(
                PredictionHead.head,
                PredictionHead.dt_inicio,
                PredictionHead.dt_fim,
                PredictionHead.prediction,
            )
            .where(
                PredictionHead.veiculo_id == veiculo_id,
                PredictionHead.target == target,
            )
            .order_by(PredictionHead.head)
        )
        pred_head_rows = result.all()

        if not pred_daily_rows and not pred_head_rows:
            return None

        # 3. Determinar range de datas necessário
        all_dates = [r[0] for r in pred_daily_rows]
        for _, dt_inicio, dt_fim, _ in pred_head_rows:
            all_dates.extend([dt_inicio, dt_fim])

        if not all_dates:
            return None

        min_date = min(all_dates)
        max_date = max(all_dates)

        # 4. Carregar valores reais do daily_activity
        result = await session.execute(
            select(DailyActivity.data, getattr(DailyActivity, target_col))
            .where(
                DailyActivity.veiculo_id == veiculo_id,
                DailyActivity.data >= min_date,
                DailyActivity.data <= max_date,
            )
            .order_by(DailyActivity.data)
        )
        actual_map = {r[0]: r[1] for r in result.all()}

        # 5. Montar comparação daily
        daily_comparison = []
        for dt, predicted in pred_daily_rows:
            daily_comparison.append({
                "data": dt,
                "actual": actual_map.get(dt, 0.0),
                "predicted": predicted,
            })

        # 6. Montar comparação heads (soma dos valores reais no intervalo)
        heads_comparison = []
        for head_num, dt_inicio, dt_fim, predicted in pred_head_rows:
            actual_sum = sum(
                val for d, val in actual_map.items()
                if dt_inicio <= d <= dt_fim
            )
            heads_comparison.append({
                "head": head_num,
                "dt_inicio": dt_inicio,
                "dt_fim": dt_fim,
                "actual": actual_sum,
                "predicted": predicted,
            })

    return {
        "veiculo_id": veiculo_id,
        "target": target,
        "daily": daily_comparison,
        "heads": heads_comparison,
    }
