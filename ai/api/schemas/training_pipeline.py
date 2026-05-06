# api/schemas/training_pipeline.py

"""Schemas de request/response para os endpoints do pipeline de treinamento."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Optional

from pydantic import BaseModel, Field

from api.models import FinalStep, ModelType, PipelineStatus, PipelineStep

# Fuso horário de apresentação (UTC-3 — São Paulo / Lisboa inverno)
_TZ_LOCAL = timezone(timedelta(hours=-3))


def _to_local(dt: datetime | None) -> datetime | None:
    """Converte datetime UTC (naive ou aware) para hora local."""
    if dt is None:
        return None
    utc = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    return utc.astimezone(_TZ_LOCAL)


def _to_local_str(dt: datetime | None) -> str | None:
    """Converte datetime UTC para string local formatada."""
    local = _to_local(dt)
    return local.strftime("%Y-%m-%d %H:%M:%S") if local else None


class PipelineRequest(BaseModel):
    """Request para iniciar uma etapa do pipeline."""
    target: str = Field(pattern=r"^(km|h)$", description="Métrica alvo: 'km' ou 'h'")


class DatasetRequest(PipelineRequest):
    """Request para gerar dataset (opcionalmente com cache)."""
    use_cache: bool = Field(default=True, description="Utilizar cache de dataset existente")


class SegmentationRequest(PipelineRequest):
    """Request para iniciar segmentação (requer caminho dos dados brutos)."""
    data_path: str = Field(description="Caminho do CSV de dados brutos de telemetria")


class OptimizationRequest(PipelineRequest):
    """Request para iniciar otimização (requer model_type)."""
    model_type: ModelType


class TrainingRequest(PipelineRequest):
    """Request para iniciar treinamento direto (requer model_type)."""
    model_type: ModelType


class FullPipelineRequest(SegmentationRequest):
    """Request para executar todas as etapas em sequência."""
    model_type: ModelType
    final_step: FinalStep = FinalStep.OPTIMIZATION


class PipelineRunResponse(BaseModel):
    """Representação de uma execução do pipeline."""
    id: int
    step: PipelineStep
    target: str
    model_type: Optional[str] = None
    status: PipelineStatus
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metrics: Optional[dict] = None
    artifacts: Optional[dict] = None
    created_at: datetime

    @classmethod
    def from_orm(cls, run) -> PipelineRunResponse:
        return cls(
            id=run.id,
            step=run.step,
            target=run.target,
            model_type=run.model_type,
            status=run.status,
            started_at=_to_local(run.started_at),
            finished_at=_to_local(run.finished_at),
            error_message=run.error_message,
            metrics=json.loads(run.metrics) if run.metrics else None,
            artifacts=json.loads(run.artifacts) if run.artifacts else None,
            created_at=_to_local(run.created_at),
        )


class PipelineRunsFilter(BaseModel):
    """Filtros para listar execuções."""
    step: Optional[PipelineStep] = None
    target: Optional[str] = Field(None, pattern=r"^(km|h)$")
    status: Optional[PipelineStatus] = None
    offset: int = Field(0, ge=0)
    limit: int = Field(50, ge=1, le=200)


class UpdateActiveModelRequest(BaseModel):
    """Request para trocar o modelo vigente de um target."""
    target: str = Field(pattern=r"^(km|h)$", description="Métrica alvo: 'km' ou 'h'")
    version_id: str = Field(description="ID do bundle versionado a activar")


class ActiveModelResponse(BaseModel):
    """Representação do modelo vigente para um target."""
    id: int
    target: str
    filename: str
    model_type: str
    version_id: Optional[str] = None
    trained_at: str
    activated_at: str


class BundleVersionResponse(BaseModel):
    """Representação de um bundle versionado disponível."""
    version_id: str
    target: str
    model_type: str
    created_at: str
    model_hash: Optional[str] = None
