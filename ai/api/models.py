# api/models.py

from datetime import date, datetime
from enum import Enum
from typing import Optional

from sqlalchemy import Float, String, UniqueConstraint, func, Index, Text
from sqlalchemy.orm import Mapped, mapped_column, registry

table_registry = registry()


# ============================================================================
# DATA INGESTION — Atividade diária, perfil e metadados
# ============================================================================


@table_registry.mapped_as_dataclass
class DailyActivity:
    """Atividade diária de um veículo (apenas dias com target > 0)."""
    __tablename__ = "daily_activity"

    id: Mapped[int] = mapped_column(primary_key=True, init=False, autoincrement=True)
    veiculo_id: Mapped[int] = mapped_column(index=True)
    data: Mapped[date] = mapped_column()
    h: Mapped[float] = mapped_column(Float, default=0.0)
    km: Mapped[float] = mapped_column(Float, default=0.0)

    __table_args__ = (
        UniqueConstraint("veiculo_id", "data", name="uq_daily_activity_veiculo_data"),
        Index("idx_daily_veiculo_data", "veiculo_id", "data"),
    )


@table_registry.mapped_as_dataclass
class DailyActivityRemoved:
    """Atividade diária de veículos removidos por critérios de qualidade.

    Mesma estrutura de DailyActivity. Os dados são mantidos aqui para
    permitir a reintegração do veículo quando dados novos chegarem e a
    série acumulada passar nos critérios de qualidade.
    """
    __tablename__ = "daily_activity_removed"

    id: Mapped[int] = mapped_column(primary_key=True, init=False, autoincrement=True)
    veiculo_id: Mapped[int] = mapped_column(index=True)
    data: Mapped[date] = mapped_column()
    h: Mapped[float] = mapped_column(Float, default=0.0)
    km: Mapped[float] = mapped_column(Float, default=0.0)

    __table_args__ = (
        UniqueConstraint("veiculo_id", "data", name="uq_daily_removed_veiculo_data"),
        Index("idx_daily_removed_veiculo_data", "veiculo_id", "data"),
    )


@table_registry.mapped_as_dataclass
class VehicleProfileFeature:
    """Feature do perfil de um veículo (formato long: 1 linha = 1 feature)."""
    __tablename__ = "vehicle_profile"

    id: Mapped[int] = mapped_column(primary_key=True, init=False, autoincrement=True)
    veiculo_id: Mapped[int] = mapped_column(index=True)
    feature: Mapped[str] = mapped_column(String(120))
    feature_class: Mapped[str] = mapped_column(String(10), comment="'km', 'h' ou 'type'")
    valor: Mapped[float] = mapped_column(Float)

    __table_args__ = (
        UniqueConstraint("veiculo_id", "feature", name="uq_profile_veiculo_feature"),
        Index("idx_profile_veiculo_class", "veiculo_id", "feature_class"),
    )


@table_registry.mapped_as_dataclass
class VehicleMetadataH:
    """Metadados do veículo para a métrica H."""
    __tablename__ = "vehicle_metadata_h"

    id: Mapped[int] = mapped_column(primary_key=True, init=False, autoincrement=True)
    veiculo_id: Mapped[int] = mapped_column(unique=True, index=True)
    dt_inicio: Mapped[date] = mapped_column()
    dt_fim: Mapped[date] = mapped_column()
    upper: Mapped[Optional[float]] = mapped_column(Float, default=None)
    quality: Mapped[Optional[int]] = mapped_column(default=None, comment="0=VALID,1=OUTLIER,2=NOT_MODELABLE,3=EMPTY")
    quality_reason: Mapped[Optional[str]] = mapped_column(String(200), default=None)


@table_registry.mapped_as_dataclass
class VehicleMetadataKm:
    """Metadados do veículo para a métrica KM."""
    __tablename__ = "vehicle_metadata_km"

    id: Mapped[int] = mapped_column(primary_key=True, init=False, autoincrement=True)
    veiculo_id: Mapped[int] = mapped_column(unique=True, index=True)
    dt_inicio: Mapped[date] = mapped_column()
    dt_fim: Mapped[date] = mapped_column()
    upper: Mapped[Optional[float]] = mapped_column(Float, default=None)
    quality: Mapped[Optional[int]] = mapped_column(default=None, comment="0=VALID,1=OUTLIER,2=NOT_MODELABLE,3=EMPTY")
    quality_reason: Mapped[Optional[str]] = mapped_column(String(200), default=None)


@table_registry.mapped_as_dataclass
class ProfileMetadata:
    """Metadados globais do perfil carregado (atualizado a cada ingestão)."""
    __tablename__ = "profile_metadata"

    id: Mapped[int] = mapped_column(primary_key=True, init=False, autoincrement=True)
    n_veiculos: Mapped[int] = mapped_column(comment="Quantidade de veículos no perfil")
    dt_inicio: Mapped[date] = mapped_column(comment="Primeira segunda-feira >= min(data)")
    dt_fim: Mapped[date] = mapped_column(comment="Último domingo <= max(data)")
    sample_size: Mapped[int] = mapped_column(comment="sample_size utilizado no trimming")
    created_at: Mapped[datetime] = mapped_column(init=False, server_default=func.now())


# ============================================================================
# ACTIVE MODEL — modelo ONNX vigente por target
# ============================================================================


@table_registry.mapped_as_dataclass
class ActiveModel:
    """Modelo ONNX vigente para predição, por target (km/h)."""
    __tablename__ = "active_models"

    id: Mapped[int] = mapped_column(primary_key=True, init=False, autoincrement=True)
    target: Mapped[str] = mapped_column(String(2), unique=True, comment="'km' ou 'h'")
    filename: Mapped[str] = mapped_column(String(256), comment="Nome do ficheiro ONNX vigente")
    model_type: Mapped[str] = mapped_column(String(20), comment="'multihead' ou 'moe'")
    trained_at: Mapped[date] = mapped_column(comment="Data em que o modelo foi gerado")
    version_id: Mapped[Optional[str]] = mapped_column(String(64), default=None, comment="ID do bundle versionado")
    activated_at: Mapped[datetime] = mapped_column(init=False, server_default=func.now())


# ============================================================================
# PIPELINE RUNS
# ============================================================================


class PipelineStep(str, Enum):
    """Etapas do pipeline de treinamento."""
    SEGMENTATION = "segmentation"
    PROFILES = "profiles"
    DATASET = "dataset"
    OPTIMIZATION = "optimization"
    TRAINING = "training"
    INGESTION = "ingestion"


class FinalStep(str, Enum):
    """Etapa final do pipeline completo."""
    OPTIMIZATION = "optimization"
    TRAINING = "training"


class PipelineStatus(str, Enum):
    """Estado de uma execução do pipeline."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelType(str, Enum):
    """Tipo de modelo de forecasting."""
    MULTIHEAD = "multihead"
    MOE = "moe"


@table_registry.mapped_as_dataclass
class PipelineRun:
    """Registro de execução de uma etapa do pipeline."""
    __tablename__ = "pipeline_runs"

    id: Mapped[int] = mapped_column(primary_key=True, init=False, autoincrement=True)
    step: Mapped[PipelineStep] = mapped_column(index=True)
    target: Mapped[str] = mapped_column(index=True, comment="'km' ou 'h'")
    model_type: Mapped[Optional[str]] = mapped_column(
        default=None,
        comment="'multihead' ou 'moe' (apenas para optimization)",
    )
    status: Mapped[PipelineStatus] = mapped_column(default=PipelineStatus.PENDING)
    started_at: Mapped[Optional[datetime]] = mapped_column(default=None)
    finished_at: Mapped[Optional[datetime]] = mapped_column(default=None)
    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)
    metrics: Mapped[Optional[str]] = mapped_column(
        Text, default=None, comment="JSON com métricas de avaliação",
    )
    artifacts: Mapped[Optional[str]] = mapped_column(
        Text, default=None, comment="JSON com paths dos artefactos gerados",
    )
    created_at: Mapped[datetime] = mapped_column(init=False, server_default=func.now())

    __table_args__ = (
        Index("idx_step_target_status", "step", "target", "status"),
    )


# ============================================================================
# PREDICTIONS — Resultados de predição
# ============================================================================


@table_registry.mapped_as_dataclass
class PredictionDaily:
    """Predição diária de produção por veículo."""
    __tablename__ = "predictions_daily"

    id: Mapped[int] = mapped_column(primary_key=True, init=False, autoincrement=True)
    veiculo_id: Mapped[int] = mapped_column(index=True)
    target: Mapped[str] = mapped_column(String(10), index=True, comment="'km' ou 'h'")
    data: Mapped[date] = mapped_column()
    prediction: Mapped[float] = mapped_column(Float)
    actual: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=None)

    __table_args__ = (
        UniqueConstraint("veiculo_id", "target", "data", name="uq_pred_daily_veiculo_target_data"),
        Index("idx_pred_daily_target_veiculo", "target", "veiculo_id"),
    )


@table_registry.mapped_as_dataclass
class PredictionHead:
    """Predição agregada por bloco temporal (rolling) por veículo."""
    __tablename__ = "predictions_heads"

    id: Mapped[int] = mapped_column(primary_key=True, init=False, autoincrement=True)
    veiculo_id: Mapped[int] = mapped_column(index=True)
    target: Mapped[str] = mapped_column(String(10), index=True, comment="'km' ou 'h'")
    dt_inicio: Mapped[date] = mapped_column()
    dt_fim: Mapped[date] = mapped_column()
    prediction: Mapped[float] = mapped_column(Float)
    actual: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=None)

    __table_args__ = (
        UniqueConstraint("veiculo_id", "target", "dt_inicio", name="uq_pred_head_veiculo_target_dt"),
        Index("idx_pred_head_target_veiculo", "target", "veiculo_id"),
    )