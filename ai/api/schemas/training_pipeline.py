# api/schemas/training_pipeline.py

"""Schemas de request/response para os endpoints do pipeline de treinamento."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from api.models import ModelType, PipelineStatus, PipelineStep


class PipelineRequest(BaseModel):
    """Request para iniciar uma etapa do pipeline."""
    target: str = Field(pattern=r"^(km|h)$", description="Métrica alvo: 'km' ou 'h'")


class SegmentationRequest(PipelineRequest):
    """Request para iniciar segmentação (requer caminho dos dados brutos)."""
    data_path: str = Field(description="Caminho do CSV de dados brutos de telemetria")


class OptimizationRequest(PipelineRequest):
    """Request para iniciar otimização (requer model_type)."""
    model_type: ModelType


class FullPipelineRequest(SegmentationRequest):
    """Request para executar todas as etapas em sequência."""
    model_type: ModelType


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
            started_at=run.started_at,
            finished_at=run.finished_at,
            error_message=run.error_message,
            metrics=json.loads(run.metrics) if run.metrics else None,
            artifacts=json.loads(run.artifacts) if run.artifacts else None,
            created_at=run.created_at,
        )


class PipelineRunsFilter(BaseModel):
    """Filtros para listar execuções."""
    step: Optional[PipelineStep] = None
    target: Optional[str] = Field(None, pattern=r"^(km|h)$")
    status: Optional[PipelineStatus] = None
    offset: int = Field(0, ge=0)
    limit: int = Field(50, ge=1, le=200)
