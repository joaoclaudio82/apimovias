# api/models.py

from datetime import datetime, date, timedelta
from enum import Enum
import json

from sqlalchemy import func, Index, CheckConstraint, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, registry
from sqlalchemy.types import TypeDecorator

table_registry = registry()


class FloatArrayType(TypeDecorator):
    """Armazena lista de floats como JSON em TEXT"""
    impl = Text
    cache_ok = True
    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value)
        return None
    def process_result_value(self, value, dialect):
        if value is not None:
            return json.loads(value)
        return None


# ============================================================================
# VEHICLE PROFILE
# ============================================================================

class VehicleCategory(str, Enum):
    """Categoria do veículo"""
    KM = 'km'
    H = 'h'


@table_registry.mapped_as_dataclass
class Vehicle:
    """
    Veículo e seu perfil
    Samples:
    - samples: lista com últimas N amostras (valores raw, pode conter zeros)
    - samples_start_date: data da primeira posição do array samples
    - samples_end_date: data da última posição do array samples
    Período efetivo (com atividade > 0):
    - Calculado dinamicamente a partir de samples
    - first_effective_date = samples_start_date + índice do primeiro valor > 0
    - last_effective_date = samples_start_date + índice do último valor > 0
    Exemplo:
    - samples = [0, 0, 10.5, 12.3, ..., 8.7, 0, 0]  # 364 valores
    - samples_start_date = 2024-01-01
    - samples_end_date = 2024-12-30
    - first_effective_date = 2024-01-03 (calculado: índice 2)
    - last_effective_date = 2024-12-28 (calculado: índice 362)
    """
    __tablename__ = 'vehicles'
    id: Mapped[int] = mapped_column(primary_key=True)
    # Categoria e segmento
    category: Mapped[VehicleCategory] = mapped_column(index=True)
    segment: Mapped[int] = mapped_column(index=True)
    # Delimitação temporal dos samples (início e fim do array)
    samples_start_date: Mapped[date] = mapped_column(
        comment="Data da primeira posição do array samples"
    )
    samples_end_date: Mapped[date] = mapped_column(
        comment="Data da última posição do array samples"
    )
    # Normalização
    upper: Mapped[float] = mapped_column(
        comment="Percentil superior calculado sobre samples"
    )
    # Últimas N amostras (valores raw, pode conter zeros)
    samples: Mapped[list] = mapped_column(
        FloatArrayType,
        comment="Últimas N amostras (valores raw, pode conter zeros)"
    )
    created_at: Mapped[datetime] = mapped_column(init=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(init=False, server_default=func.now(), onupdate=func.now())
    __table_args__ = (
        Index('idx_category_segment', 'category', 'segment'),
        CheckConstraint('samples_start_date <= samples_end_date', name='check_samples_dates'),
    )
    def get_effective_period(self) -> tuple[date, date]:
        """
        Calcula período efetivo (datas com atividade > 0)
        Returns
        -------
        tuple
            (first_effective_date, last_effective_date)
        """
        if not self.samples:
            return self.samples_start_date, self.samples_end_date
        # Encontrar primeiro valor > 0
        try:
            first_idx = next(i for i, v in enumerate(self.samples) if v > 0)
        except StopIteration:
            # Todos zeros
            return self.samples_start_date, self.samples_start_date
        # Encontrar último valor > 0
        try:
            last_idx = len(self.samples) - 1 - next(
                i for i, v in enumerate(reversed(self.samples)) if v > 0
            )
        except StopIteration:
            last_idx = first_idx
        first_effective = self.samples_start_date + timedelta(days=first_idx)
        last_effective = self.samples_start_date + timedelta(days=last_idx)
        return first_effective, last_effective


@table_registry.mapped_as_dataclass
class VehicleSummaryFeatures:
    """
    Features de resumo do veículo (15 métricas)
    Nomes genéricos para reutilizar estrutura entre km e h.
    """
    __tablename__ = 'vehicle_summary_features'
    vehicle_id: Mapped[int] = mapped_column(
        ForeignKey('vehicles.id', ondelete='CASCADE'),
        primary_key=True
    )
  # 15 features
    per_day: Mapped[float]
    mean: Mapped[float]
    max: Mapped[float]
    median: Mapped[float]
    std: Mapped[float]
    continuity_score: Mapped[float]
    active_weeks_rate: Mapped[float]
    active_days_rate: Mapped[float]
    gaps_cv: Mapped[float]
    mean_gap: Mapped[float]
    max_gap: Mapped[float]
    cv: Mapped[float]
    p25: Mapped[float]
    p75: Mapped[float]
    iqr: Mapped[float]
    created_at: Mapped[datetime] = mapped_column(init=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(init=False, server_default=func.now(), onupdate=func.now())


@table_registry.mapped_as_dataclass
class VehicleWeekdayFeatures:
    """
    Features por dia da semana (70 registros por veículo)
    7 dias × 10 features = 70 registros por veículo.
    """
    __tablename__ = 'vehicle_weekday_features'
    vehicle_id: Mapped[int] = mapped_column(
        ForeignKey('vehicles.id', ondelete='CASCADE'),
        primary_key=True
    )
    day: Mapped[int] = mapped_column(
        primary_key=True,
        comment="Dia da semana: 1=seg, 2=ter, 3=qua, 4=qui, 5=sex, 6=sab, 7=dom"
    )
  # 10 features
    mean: Mapped[float]
    std: Mapped[float]
    median: Mapped[float]
    max: Mapped[float]
    min: Mapped[float]
    p25: Mapped[float]
    p75: Mapped[float]
    iqr: Mapped[float]
    prob_active: Mapped[float]
    cv: Mapped[float]
    created_at: Mapped[datetime] = mapped_column(init=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(init=False, server_default=func.now(), onupdate=func.now())
    __table_args__ = (
        CheckConstraint('day >= 1 AND day <= 7', name='check_valid_weekday'),
        Index('idx_vehicle_day', 'vehicle_id', 'day'),
    )