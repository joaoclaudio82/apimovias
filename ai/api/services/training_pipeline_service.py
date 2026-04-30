# api/services/training_pipeline_service.py

"""
Serviço de execução das etapas do pipeline de treinamento.

Cada etapa é executada em background via asyncio (run_in_executor).
O estado é persistido na tabela ``pipeline_runs``.
"""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime
from pathlib import Path

from sqlalchemy import select

from api.database import session_context
from api.models import PipelineRun, PipelineStatus, PipelineStep, ModelType

logger = logging.getLogger(__name__)

# Diretório base dos configs (relativo a api/)
_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


def _config_path(name: str) -> str:
    return str(_CONFIG_DIR / name)


def _load_configs():
    """Carrega todas as configs comuns."""
    from api.config.output_config import OutputConfig

    return {
        "output": OutputConfig.from_yaml(_config_path("output_config.yaml")),
    }


def _resolve_pipeline_cls(model_type: str):
    """Resolve a classe do pipeline a partir do tipo de modelo."""
    from moviasai.forecasting.training_pipeline import (
        MultiHeadTrainingPipeline,
        MoETrainingPipeline,
    )

    mapping = {
        ModelType.MULTIHEAD: MultiHeadTrainingPipeline,
        ModelType.MOE: MoETrainingPipeline,
    }
    return mapping[model_type]


# ------------------------------------------------------------------
# Helpers de DB
# ------------------------------------------------------------------


async def _create_run(
    step: PipelineStep,
    target: str,
    model_type: str | None = None,
) -> int:
    """Cria um PipelineRun com status PENDING e retorna o id."""
    async with session_context() as session:
        run = PipelineRun(
            step=step,
            target=target,
            model_type=model_type,
            status=PipelineStatus.PENDING,
        )
        session.add(run)
        await session.commit()
        await session.refresh(run)
        return run.id


async def _mark_running(run_id: int):
    async with session_context() as session:
        run = await session.get(PipelineRun, run_id)
        run.status = PipelineStatus.RUNNING
        run.started_at = datetime.utcnow()
        await session.commit()


async def _mark_completed(
    run_id: int,
    metrics: dict | None = None,
    artifacts: dict | None = None,
):
    async with session_context() as session:
        run = await session.get(PipelineRun, run_id)
        run.status = PipelineStatus.COMPLETED
        run.finished_at = datetime.utcnow()
        if metrics:
            run.metrics = json.dumps(metrics, default=str)
        if artifacts:
            run.artifacts = json.dumps(artifacts, default=str)
        await session.commit()


async def _mark_failed(run_id: int, error: str):
    async with session_context() as session:
        run = await session.get(PipelineRun, run_id)
        run.status = PipelineStatus.FAILED
        run.finished_at = datetime.utcnow()
        run.error_message = error[:4000]
        await session.commit()


# ------------------------------------------------------------------
# 1. SEGMENTAÇÃO
# ------------------------------------------------------------------


async def run_segmentation(run_id: int, target: str, data_path: str):
    """Executa segmentação completa (stage1 + stage2)."""
    import asyncio

    await _mark_running(run_id)
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _segmentation_sync, target, data_path)
        await _mark_completed(run_id, artifacts=result)
    except Exception:
        await _mark_failed(run_id, traceback.format_exc())
        raise


def _segmentation_sync(target: str, data_path: str) -> dict:
    import polars as pl
    from api.config.segmentation_config import SegmentationConfig
    from api.config.data_quality_config import DataQualityConfig
    from api.config.output_config import OutputConfig
    from moviasai.profiling.segmentation_pipeline import VehicleSegmentationPipeline
    from moviasai.data.utils import load_raw_data

    cfg = SegmentationConfig.from_yaml(_config_path("segmentation_config.yaml"))
    dq_cfg = DataQualityConfig.from_yaml(_config_path("data_quality_config.yaml"))
    output_cfg = OutputConfig.from_yaml(_config_path("output_config.yaml"))

    df_daily = load_raw_data(data_path)

    pipeline = VehicleSegmentationPipeline.from_config(df_daily, cfg, output_cfg, dq_cfg)
    results = pipeline.run()

    return {
        "segmentation_dir": output_cfg.logs.segmentation,
        "classification_dir": output_cfg.models.classification,
        "train_data_dir": output_cfg.data.train_dataset,
    }


# ------------------------------------------------------------------
# 2. PERFIS
# ------------------------------------------------------------------


async def run_profiles(run_id: int, target: str):
    """Gera perfis versionados para o target."""
    import asyncio

    await _mark_running(run_id)
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _profiles_sync, target)
        await _mark_completed(run_id, artifacts=result)
    except Exception:
        await _mark_failed(run_id, traceback.format_exc())
        raise


def _profiles_sync(target: str) -> dict:
    from api.config.vehicle_profile_config import VehicleProfileConfig
    from api.config.segmentation_config import SegmentationConfig
    from api.config.output_config import OutputConfig
    from moviasai.data.utils import load_raw_data
    from moviasai.profiling.profile import VersionedVehicleProfile

    profile_cfg = VehicleProfileConfig.from_yaml(_config_path("vehicle_profile_config.yaml"))
    segmentation_cfg = SegmentationConfig.from_yaml(_config_path("segmentation_config.yaml"))
    output_cfg = OutputConfig.from_yaml(_config_path("output_config.yaml"))

    df = load_raw_data(output_cfg.train_data_path(target), target=target)

    profile = VersionedVehicleProfile.from_config(
        config=profile_cfg,
        segmentation_config=segmentation_cfg,
        output_config=output_cfg,
        metric=target,
        last=False,
        drop_duplicates=True,
    )
    profile.fit(df)
    profile.save(str(output_cfg.profile_dir(target)))

    return {
        "profile_dir": str(output_cfg.profile_dir(target)),
    }


# ------------------------------------------------------------------
# 3. DATASET
# ------------------------------------------------------------------


async def run_dataset(run_id: int, target: str):
    """Gera datasets de treinamento para o target."""
    import asyncio

    await _mark_running(run_id)
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _dataset_sync, target)
        await _mark_completed(run_id, artifacts=result)
    except Exception:
        await _mark_failed(run_id, traceback.format_exc())
        raise


def _dataset_sync(target: str) -> dict:
    from api.config.dataset_config import DatasetConfig
    from api.config.output_config import OutputConfig
    from moviasai.data.utils import load_raw_data
    from moviasai.data.dataset import ProfileDatasetGenerator
    from moviasai.profiling.profile import VersionedVehicleProfile

    dataset_cfg = DatasetConfig.from_yaml(_config_path("dataset_config.yaml"))
    output_cfg = OutputConfig.from_yaml(_config_path("output_config.yaml"))

    profile = VersionedVehicleProfile.load(
        profile_dir=str(output_cfg.profile_dir(target)),
        classifier_path=str(output_cfg.classifier_path(target)),
    )

    df_daily = load_raw_data(output_cfg.train_data_path(target), target=target).to_pandas()

    generator = ProfileDatasetGenerator(
        vehicle_profile=profile,
        df_daily=df_daily,
        min_weeks_general=dataset_cfg.min_weeks_general,
        num_weeks_recent=dataset_cfg.num_weeks_recent,
        horizon_weeks=dataset_cfg.horizon_weeks,
        daily_horizon=dataset_cfg.daily_horizon,
        min_recent_active_days=dataset_cfg.min_recent_active_days,
        cache_dir=output_cfg.dataset_cache_dir(target),
        cluster_features=dataset_cfg.cluster_features,
    )

    dataset = generator.generate()

    return {
        "cache_dir": str(output_cfg.dataset_cache_dir(target)),
        "n_samples": len(dataset),
        "n_general_features": dataset.X_general.shape[1],
        "n_recent_features": dataset.X_recent.shape[1],
    }


# ------------------------------------------------------------------
# 4. OTIMIZAÇÃO (+ retrain)
# ------------------------------------------------------------------


async def run_optimization(run_id: int, target: str, model_type: str):
    """Otimiza hiperparâmetros e retreina o modelo final."""
    import asyncio

    await _mark_running(run_id)
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, _optimization_sync, target, model_type,
        )
        await _mark_completed(run_id, metrics=result["metrics"], artifacts=result["artifacts"])
    except Exception:
        await _mark_failed(run_id, traceback.format_exc())
        raise


def _optimization_sync(target: str, model_type: str) -> dict:
    from api.config.dataset_config import DatasetConfig
    from api.config.training_config import TrainingConfig
    from api.config.model_config import ModelConfig
    from api.config.output_config import OutputConfig
    from api.config.optimization_config import OptimizationConfig
    from moviasai.data.utils import load_raw_data
    from moviasai.forecasting.optimization import HyperparameterOptimizer

    dataset_cfg = DatasetConfig.from_yaml(_config_path("dataset_config.yaml"))
    training_cfg = TrainingConfig.from_yaml(_config_path("training_config.yaml"))
    model_cfg = ModelConfig.from_yaml(_config_path("model_config.yaml"))
    output_cfg = OutputConfig.from_yaml(_config_path("output_config.yaml"))
    optuna_cfg = OptimizationConfig.from_yaml(_config_path("optimization_config.yaml"))

    pipeline_cls = _resolve_pipeline_cls(model_type)

    df_daily = load_raw_data(output_cfg.train_data_path(target), target=target)

    optimizer = HyperparameterOptimizer.from_config(
        df_daily=df_daily,
        dataset_cfg=dataset_cfg,
        training_cfg=training_cfg,
        model_cfg=model_cfg,
        output_config=output_cfg,
        optuna_cfg=optuna_cfg,
        target=target,
        pipeline_cls=pipeline_cls,
    )

    study = optimizer.optimize()
    pipeline = optimizer.retrain_best(max_epochs=training_cfg.trainer.max_epochs)

    # Extrair métricas serializáveis
    metrics_out = {}
    if pipeline.metrics:
        for split_name in ("val", "test"):
            if split_name not in pipeline.metrics:
                continue
            maint = pipeline.metrics[split_name]["maintenance"]
            metrics_out[split_name] = {}
            for k_label in ("k2", "k3", "k4"):
                if k_label not in maint:
                    continue
                m = maint[k_label]
                metrics_out[split_name][k_label] = {
                    "mean_error": float(m["mean_error"]),
                    "mae_days": float(m["mae_days"]),
                    "p90_error": float(m["p90_error"]),
                    "pct_late": float(m["pct_late"]),
                    "pct_early": float(m["pct_early"]),
                }

    metrics_out["best_params"] = study.best_params
    metrics_out["best_objective"] = float(study.best_value)
    metrics_out["n_trials"] = len(study.trials)

    return {
        "metrics": metrics_out,
        "artifacts": {
            "training_dir": str(output_cfg.training_dir(target)),
            "model_dir": output_cfg.models.forecasting,
            "model_type": model_type,
        },
    }


# ------------------------------------------------------------------
# 5. PIPELINE COMPLETO (sequencial)
# ------------------------------------------------------------------

_FULL_PIPELINE_KEY = ("full", "__global__")


async def run_full_pipeline(
    target: str,
    data_path: str,
    model_type: str,
) -> list[int]:
    """
    Cria os 4 PipelineRun e submete uma task que os executa em sequência.

    Retorna a lista de run_ids criados.
    """
    from api.services import task_manager

    steps = [
        PipelineStep.SEGMENTATION,
        PipelineStep.PROFILES,
        PipelineStep.DATASET,
        PipelineStep.OPTIMIZATION,
    ]
    run_ids = []
    for step in steps:
        mt = model_type if step == PipelineStep.OPTIMIZATION else None
        run_ids.append(await _create_run(step, target, model_type=mt))

    async def _sequential():
        seg_id, prof_id, ds_id, opt_id = run_ids
        try:
            await run_segmentation(seg_id, target, data_path)
            await run_profiles(prof_id, target)
            await run_dataset(ds_id, target)
            await run_optimization(opt_id, target, model_type)
        except Exception:
            logger.exception("Pipeline completo falhou para %s", target)

    task_manager.submit("full", "__global__", _sequential())
    return run_ids
