# api/schemas/profiling.py

"""Schemas de request/response para ingestão e profiling."""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# Request — registros de atividade diária (formato JSON)
# ------------------------------------------------------------------


class DailyActivityRecord(BaseModel):
    """Um registro de atividade diária."""
    veiculo_id: int
    data: date
    h_dia_clean: float = Field(ge=0)
    km_dia_clean: float = Field(ge=0)


class DailyActivityUpload(BaseModel):
    """Payload JSON com lista de registros diários."""
    records: list[DailyActivityRecord]


# ------------------------------------------------------------------
# Response
# ------------------------------------------------------------------


class IngestionSummary(BaseModel):
    """Resumo do resultado da ingestão."""
    daily_activity_inserted: int = Field(description="Registros novos inseridos em daily_activity")
    daily_activity_trimmed: int = Field(description="Registros antigos removidos por sample_size")
    vehicles_removed: int = Field(description="Veículos removidos por não terem atividade residual")
    profile_features_upserted: int = Field(description="Features de perfil inseridas/atualizadas")
    metadata_h_upserted: int = Field(description="Registros de metadados H inseridos/atualizados")
    metadata_km_upserted: int = Field(description="Registros de metadados KM inseridos/atualizados")


class VehicleProfileFeatureResponse(BaseModel):
    veiculo_id: int
    feature: str
    feature_class: str
    valor: float


class VehicleMetadataResponse(BaseModel):
    veiculo_id: int
    dt_inicio: date
    dt_fim: date
    upper: Optional[float] = None
    quality: Optional[int] = None
    quality_reason: Optional[str] = None


class ProfileMetadataResponse(BaseModel):
    """Metadados globais de um perfil."""
    id: int
    n_veiculos: int
    dt_inicio: date
    dt_fim: date
    sample_size: int
    created_at: str


class VehicleInfoResponse(BaseModel):
    """Informações completas de um veículo para um target."""
    veiculo_id: int
    target: str
    profile: Optional[Dict[str, float]] = None
    metadata: Optional[VehicleMetadataResponse] = None
    message: Optional[str] = None


class VehicleSummary(BaseModel):
    """Resumo de um veículo na listagem."""
    veiculo_id: int
    quality: Optional[int] = None
    quality_reason: Optional[str] = None
    dt_inicio_km: Optional[date] = None
    dt_fim_km: Optional[date] = None
    upper_km: Optional[float] = None
    dt_inicio_h: Optional[date] = None
    dt_fim_h: Optional[date] = None
    upper_h: Optional[float] = None
