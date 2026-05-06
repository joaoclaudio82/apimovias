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
from api.models import PipelineRun, PipelineStatus, PipelineStep, ModelType, FinalStep

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
# Verificação de consistência da cadeia de artefactos
# ------------------------------------------------------------------


def _verify_artifact_chain(output_cfg, target: str) -> None:
    """Verifica que segmentação → perfis → dataset estão consistentes.

    Raises
    ------
    moviasai.versioning.StaleArtifactError
        Se alguma ligação na cadeia estiver desatualizada.
    FileNotFoundError
        Se algum manifest não existir.
    """
    from moviasai.versioning import PipelineManifest

    seg = PipelineManifest("segmentation", output_cfg.logs.segmentation)
    prof = PipelineManifest("profiles", output_cfg.profile_dir(target))
    ds = PipelineManifest("dataset", output_cfg.dataset_cache_dir(target))

    PipelineManifest.verify_chain(seg, prof, ds)
    logger.info("Cadeia de artefactos verificada para target=%s", target)


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

    # Manifest é escrito pelo próprio pipeline (segmentation_pipeline.py)

    return {
        "segmentation_dir": output_cfg.logs.segmentation,
        "models_dir": str(Path(output_cfg.logs.segmentation) / "models"),
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
    from moviasai.versioning import PipelineManifest

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

    profile_dir = output_cfg.profile_dir(target)
    profile.save(str(profile_dir))

    # Manifest: hash dos parquets gerados, com segmentação como parent
    seg_manifest = PipelineManifest("segmentation", output_cfg.logs.segmentation)
    profiles_manifest = PipelineManifest("profiles", profile_dir)
    profiles_manifest.write(
        output_hash=PipelineManifest.hash_files(
            profile_dir / "vehicle_profiles.parquet",
            profile_dir / "effective_periods.parquet",
            profile_dir / "metadata.json",
        ),
        parent=seg_manifest,
    )

    return {
        "profile_dir": str(profile_dir),
    }


# ------------------------------------------------------------------
# 3. DATASET
# ------------------------------------------------------------------


async def run_dataset(run_id: int, target: str, use_cache: bool = True):
    """Gera datasets de treinamento para o target."""
    import asyncio

    await _mark_running(run_id)
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _dataset_sync, target, use_cache)
        await _mark_completed(run_id, artifacts=result)
    except Exception:
        await _mark_failed(run_id, traceback.format_exc())
        raise


def _dataset_sync(target: str, use_cache: bool = True) -> dict:
    from api.config.dataset_config import DatasetConfig
    from api.config.output_config import OutputConfig
    from moviasai.data.utils import load_raw_data
    from moviasai.data.dataset import ProfileDatasetGenerator
    from moviasai.profiling.profile import VersionedVehicleProfile
    from moviasai.versioning import PipelineManifest

    dataset_cfg = DatasetConfig.from_yaml(_config_path("dataset_config.yaml"))
    output_cfg = OutputConfig.from_yaml(_config_path("output_config.yaml"))

    profile = VersionedVehicleProfile.load(
        profile_dir=str(output_cfg.profile_dir(target)),
        classifier_path=str(output_cfg.classifier_path(target)),
    )

    df_daily = load_raw_data(output_cfg.train_data_path(target), target=target).to_pandas()

    cache_dir = output_cfg.dataset_cache_dir(target) if use_cache else None
    generator = ProfileDatasetGenerator(
        vehicle_profile=profile,
        df_daily=df_daily,
        min_weeks_general=dataset_cfg.min_weeks_general,
        num_weeks_recent=dataset_cfg.num_weeks_recent,
        horizon_weeks=dataset_cfg.horizon_weeks,
        daily_horizon=dataset_cfg.daily_horizon,
        min_recent_active_days=dataset_cfg.min_recent_active_days,
        cache_dir=cache_dir,
        cluster_features=dataset_cfg.cluster_features,
    )

    dataset = generator.generate()

    # Manifest: usa o hash de dados já calculado pelo generator, com perfis como parent
    prof_manifest = PipelineManifest("profiles", output_cfg.profile_dir(target))
    ds_output_dir = cache_dir or output_cfg.dataset_cache_dir(target)
    ds_manifest = PipelineManifest("dataset", ds_output_dir)
    ds_manifest.write(
        output_hash=generator._compute_data_hash(),
        config_hash=PipelineManifest.hash_config(generator.config),
        parent=prof_manifest,
    )

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
    from datetime import date as _date, datetime as _datetime

    from sqlalchemy import select as sa_select

    from api.database import session_context
    from api.models import ActiveModel

    await _mark_running(run_id)
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, _optimization_sync, target, model_type,
        )

        # Registar/atualizar modelo vigente na DB
        artifacts = result["artifacts"]
        async with session_context() as session:
            res = await session.execute(
                sa_select(ActiveModel).where(ActiveModel.target == target)
            )
            active = res.scalar_one_or_none()
            trained_at = _date.today()

            if active is None:
                active = ActiveModel(
                    target=target,
                    filename=artifacts["dated_filename"],
                    model_type=model_type,
                    version_id=artifacts.get("version_id"),
                    trained_at=trained_at,
                )
                session.add(active)
            else:
                active.filename = artifacts["dated_filename"]
                active.model_type = model_type
                active.version_id = artifacts.get("version_id")
                active.trained_at = trained_at
                active.activated_at = _datetime.utcnow()
            await session.commit()

        await _mark_completed(run_id, metrics=result["metrics"], artifacts=artifacts)
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
    from moviasai.versioning import PipelineManifest

    dataset_cfg = DatasetConfig.from_yaml(_config_path("dataset_config.yaml"))
    training_cfg = TrainingConfig.from_yaml(_config_path("training_config.yaml"))
    model_cfg = ModelConfig.from_yaml(_config_path("model_config.yaml"))
    output_cfg = OutputConfig.from_yaml(_config_path("output_config.yaml"))

    # Verificar consistência da cadeia de artefactos
    _verify_artifact_chain(output_cfg, target)
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

    # Criar bundle versionado
    from moviasai.bundle import ModelBundle
    from pathlib import Path as _Path

    onnx_source = pipeline.exported_onnx_path
    seg_models_dir = _Path(output_cfg.logs.segmentation) / "models"

    bundle = ModelBundle.create(
        base_dir=_Path(output_cfg.models.forecasting),
        target=target,
        model_type=model_type,
        onnx_source=onnx_source,
        segmentation_models_dir=seg_models_dir,
        config_dir=_Path(_CONFIG_DIR),
    )

    # Extrair métricas serializáveis
    metrics_out = {}
    if pipeline.metrics:
        for split_name in ("val", "test"):
            if split_name not in pipeline.metrics:
                continue
            maint = pipeline.metrics[split_name]["maintenance"]
            metrics_out[split_name] = {}
            for k_label in sorted(maint.keys()):
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
            "version_id": bundle.version_id,
            "dated_filename": onnx_source.name,
            "trained_at": bundle.version_id.split("_")[0],
        },
    }


# ------------------------------------------------------------------
# 4b. TREINAMENTO DIRETO (sem otimização)
# ------------------------------------------------------------------


async def run_training(run_id: int, target: str, model_type: str):
    """Treina o modelo diretamente com hiperparâmetros dos ficheiros de configuração."""
    import asyncio
    from datetime import date as _date, datetime as _datetime

    from sqlalchemy import select as sa_select

    from api.database import session_context
    from api.models import ActiveModel

    await _mark_running(run_id)
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, _training_sync, target, model_type,
        )

        # Registar/atualizar modelo vigente na DB
        artifacts = result["artifacts"]
        async with session_context() as session:
            res = await session.execute(
                sa_select(ActiveModel).where(ActiveModel.target == target)
            )
            active = res.scalar_one_or_none()
            trained_at = _date.today()

            if active is None:
                active = ActiveModel(
                    target=target,
                    filename=artifacts["dated_filename"],
                    model_type=model_type,
                    version_id=artifacts.get("version_id"),
                    trained_at=trained_at,
                )
                session.add(active)
            else:
                active.filename = artifacts["dated_filename"]
                active.model_type = model_type
                active.version_id = artifacts.get("version_id")
                active.trained_at = trained_at
                active.activated_at = _datetime.utcnow()
            await session.commit()

        await _mark_completed(run_id, metrics=result["metrics"], artifacts=artifacts)
    except Exception:
        await _mark_failed(run_id, traceback.format_exc())
        raise


def _training_sync(target: str, model_type: str) -> dict:
    from api.config.dataset_config import DatasetConfig
    from api.config.training_config import TrainingConfig
    from api.config.model_config import ModelConfig
    from api.config.output_config import OutputConfig
    from moviasai.data.utils import load_raw_data
    from moviasai.versioning import PipelineManifest

    dataset_cfg = DatasetConfig.from_yaml(_config_path("dataset_config.yaml"))
    training_cfg = TrainingConfig.from_yaml(_config_path("training_config.yaml"))
    model_cfg = ModelConfig.from_yaml(_config_path("model_config.yaml"))
    output_cfg = OutputConfig.from_yaml(_config_path("output_config.yaml"))

    # Verificar consistência da cadeia de artefactos
    _verify_artifact_chain(output_cfg, target)

    pipeline_cls = _resolve_pipeline_cls(model_type)

    df_daily = load_raw_data(output_cfg.train_data_path(target), target=target)

    pipeline = pipeline_cls.from_config(
        df_daily=df_daily.to_pandas() if hasattr(df_daily, "to_pandas") else df_daily,
        dataset_cfg=dataset_cfg,
        training_cfg=training_cfg,
        model_cfg=model_cfg,
        output_config=output_cfg,
        target=target,
    )

    metrics = pipeline.run()

    # Criar bundle versionado
    from moviasai.bundle import ModelBundle

    onnx_source = pipeline.exported_onnx_path
    seg_models_dir = Path(output_cfg.logs.segmentation) / "models"

    bundle = ModelBundle.create(
        base_dir=Path(output_cfg.models.forecasting),
        target=target,
        model_type=model_type,
        onnx_source=onnx_source,
        segmentation_models_dir=seg_models_dir,
        config_dir=Path(_CONFIG_DIR),
    )

    # Extrair métricas serializáveis
    metrics_out = {}
    if metrics:
        for split_name in ("val", "test"):
            if split_name not in metrics:
                continue
            maint = metrics[split_name]["maintenance"]
            metrics_out[split_name] = {}
            for k_label in sorted(maint.keys()):
                m = maint[k_label]
                metrics_out[split_name][k_label] = {
                    "mean_error": float(m["mean_error"]),
                    "mae_days": float(m["mae_days"]),
                    "p90_error": float(m["p90_error"]),
                    "pct_late": float(m["pct_late"]),
                    "pct_early": float(m["pct_early"]),
                }

    return {
        "metrics": metrics_out,
        "artifacts": {
            "training_dir": str(output_cfg.training_dir(target)),
            "model_dir": output_cfg.models.forecasting,
            "model_type": model_type,
            "version_id": bundle.version_id,
            "dated_filename": onnx_source.name,
            "trained_at": bundle.version_id.split("_")[0],
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
    final_step: str = "optimization",
) -> list[int]:
    """
    Cria os PipelineRun e submete uma task que os executa em sequência.

    Parameters
    ----------
    final_step : str
        ``'optimization'`` ou ``'training'``. Determina a etapa final.

    Retorna a lista de run_ids criados.
    """
    from api.services import task_manager

    if final_step == FinalStep.TRAINING:
        final_pipeline_step = PipelineStep.TRAINING
    else:
        final_pipeline_step = PipelineStep.OPTIMIZATION

    steps = [
        PipelineStep.SEGMENTATION,
        PipelineStep.PROFILES,
        PipelineStep.DATASET,
        final_pipeline_step,
    ]
    run_ids = []
    for step in steps:
        mt = model_type if step in (PipelineStep.OPTIMIZATION, PipelineStep.TRAINING) else None
        run_ids.append(await _create_run(step, target, model_type=mt))

    async def _sequential():
        seg_id, prof_id, ds_id, final_id = run_ids
        try:
            await run_segmentation(seg_id, target, data_path)
            await run_profiles(prof_id, target)
            await run_dataset(ds_id, target)
            if final_step == FinalStep.TRAINING:
                await run_training(final_id, target, model_type)
            else:
                await run_optimization(final_id, target, model_type)
        except Exception:
            logger.exception("Pipeline completo falhou para %s", target)

    task_manager.submit("full", "__global__", _sequential())
    return run_ids
