# api/schemas/prediction_schemas.py

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class DateToReachRequest(BaseModel):
    """Requisição para encontrar data de atingir valor alvo"""
    vehicle_ids: List[int] = Field(..., description="Lista de IDs de veículos")
    target_values: List[float] = Field(..., description="Valores alvo (acumulado) para cada veículo")
    n_jobs: int = Field(1, ge=-1, description="Paralelização (-1 = todos CPUs, 1 = sequencial)")
    
    @field_validator('target_values')
    @classmethod
    def validate_lengths(cls, v, info):
        vehicle_ids = info.data.get('vehicle_ids')
        if vehicle_ids and len(v) != len(vehicle_ids):
            raise ValueError("target_values deve ter mesmo tamanho que vehicle_ids")
        return v
    
    @field_validator('target_values')
    @classmethod
    def validate_positive(cls, v):
        if any(val <= 0 for val in v):
            raise ValueError("Todos os target_values devem ser positivos")
        return v


class DateToReachResponse(BaseModel):
    """Resposta da predição de data"""
    vehicle_id: int
    category: str
    segment: int
    target_value: float
    predicted_date: str = Field(..., description="Data prevista (ISO format)")
    n_steps: int = Field(..., description="Número de dias até atingir o alvo")
    accumulated_value: float = Field(..., description="Valor acumulado nas predições")
    
    class Config:
        json_schema_extra = {
            "example": {
                "vehicle_id": 1316,
                "category": "km",
                "segment": 2,
                "target_value": 10000.0,
                "predicted_date": "2024-06-15",
                "n_steps": 67,
                "accumulated_value": 10050.5
            }
        }


class AccumulatedAtStepRequest(BaseModel):
    """Requisição para predizer acumulado em step/data específico"""
    vehicle_ids: List[int] = Field(..., description="Lista de IDs de veículos")
    n_steps: Optional[List[int]] = Field(
        None,
        description="Número de steps para cada veículo (opcional se reference_dates fornecido)"
    )
    reference_dates: Optional[List[str]] = Field(
        None,
        description="Datas de referência ISO para cada veículo (opcional se n_steps fornecido)"
    )
    n_jobs: int = Field(1, ge=-1, description="Paralelização")
    
    @field_validator('n_steps', 'reference_dates')
    @classmethod
    def validate_at_least_one(cls, v, info):
        field_name = info.field_name
        
        if field_name == 'n_steps':
            if v is None and info.data.get('reference_dates') is None:
                raise ValueError("Deve fornecer n_steps ou reference_dates")
        
        return v
    
    @field_validator('n_steps')
    @classmethod
    def validate_n_steps_length(cls, v, info):
        if v is None:
            return v
        
        vehicle_ids = info.data.get('vehicle_ids')
        if vehicle_ids and len(v) != len(vehicle_ids):
            raise ValueError("n_steps deve ter mesmo tamanho que vehicle_ids")
        
        if any(n <= 0 for n in v):
            raise ValueError("Todos os n_steps devem ser positivos")
        
        return v
    
    @field_validator('reference_dates')
    @classmethod
    def validate_reference_dates_length(cls, v, info):
        if v is None:
            return v
        
        vehicle_ids = info.data.get('vehicle_ids')
        if vehicle_ids and len(v) != len(vehicle_ids):
            raise ValueError("reference_dates deve ter mesmo tamanho que vehicle_ids")
        
        return v


class AccumulatedAtStepResponse(BaseModel):
    """Resposta da predição de acumulado"""
    vehicle_id: int
    category: str
    segment: int
    n_steps: int
    reference_date: str = Field(..., description="Data de referência (ISO format)")
    accumulated_value: float = Field(..., description="Valor acumulado previsto")
    
    class Config:
        json_schema_extra = {
            "example": {
                "vehicle_id": 1316,
                "category": "km",
                "segment": 2,
                "n_steps": 90,
                "reference_date": "2024-06-30",
                "accumulated_value": 13500.75
            }
        }

