# api/routers/vehicle_profiles.py

from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends, Form
from http import HTTPStatus
from typing import Annotated
import logging

from api.database import get_session
from api.services.vehicle_profile_service import VehicleProfileService
from api.schemas.vehicle_profile_schemas import (
    VehicleInfoResponse,
    StatisticsResponse,
    ProfileUpdateResponse,
)
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/vehicle-profiles', tags=['vehicle-profiles'])

Session = Annotated[AsyncSession, Depends(get_session)]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOCKER_SHARED_CSV_PATH = Path('/shared/movias.csv')
LOCAL_SHARED_CSV_FALLBACK = PROJECT_ROOT / 'extractor' / 'data' / 'movias.csv'

def get_service(session: Session) -> VehicleProfileService:
    """Dependency para obter service"""
    return VehicleProfileService(session)


def _resolve_shared_csv_path() -> Path:
    if DOCKER_SHARED_CSV_PATH.exists():
        return DOCKER_SHARED_CSV_PATH
    if LOCAL_SHARED_CSV_FALLBACK.exists():
        return LOCAL_SHARED_CSV_FALLBACK
    return DOCKER_SHARED_CSV_PATH


@router.post(
    '/update',
    response_model=ProfileUpdateResponse,
    summary='Atualiza perfis via CSV compartilhado'
)
async def update_profiles_csv(
    service: VehicleProfileService = Depends(get_service),
    target: str = Form(..., description="Target: 'km_dia_clean' ou 'h_dia_clean'"),
):
    """
    Atualiza perfis de veículos a partir do CSV compartilhado.
    """
    logger.info('Atualização de perfis via CSV solicitada')

    try:
        resolved_csv = _resolve_shared_csv_path()
        result = await service.update_from_csv_path(target=target, csv_path=str(resolved_csv))

        return ProfileUpdateResponse(
            target=target,
            n_vehicles=result['n_vehicles'],
            message='Atualização concluída com sucesso',
        )

    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.error(f'Arquivo não encontrado: {e}')
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f'Arquivo não encontrado: {str(e)}',
        )
    except ValueError as e:
        logger.error(f'Erro de validação: {e}')
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception('Erro ao atualizar perfis via CSV')
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f'Erro ao atualizar perfis: {str(e)}',
        )


@router.get(
    '/profile/{vehicle_id}',
    summary='Busca perfil completo de um veículo'
)
async def get_profile(
    vehicle_id: int,
    service: VehicleProfileService = Depends(get_service)
):
    """
    Busca perfil completo de um veículo
    
    Retorna todas as features calculadas (85 features + metadata).
    """
    try:
        profile = await service.get_vehicle_profile(vehicle_id)
        if not profile:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail=f"Perfil não encontrado para veículo {vehicle_id}"
            )
        return profile
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Erro ao buscar perfil do veículo {vehicle_id}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Erro ao buscar perfil: {str(e)}"
        )


@router.get(
    '/vehicle/{vehicle_id}/info',
    response_model=VehicleInfoResponse,
    summary='Busca informações básicas do veículo'
)
async def get_vehicle_info(
    vehicle_id: int,
    service: VehicleProfileService = Depends(get_service)
):
    """Busca informações básicas do veículo"""
    try:
        info = await service.get_vehicle_info(vehicle_id)
        if not info:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail=f"Veículo {vehicle_id} não encontrado"
            )
        return info
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Erro ao buscar informações do veículo {vehicle_id}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Erro ao buscar informações: {str(e)}"
        )


@router.get(
    '/statistics',
    response_model=StatisticsResponse,
    status_code=HTTPStatus.OK,
    summary='Estatísticas do serviço'
)
async def get_statistics(
    service: VehicleProfileService = Depends(get_service)
):
    """Estatísticas agregadas por categoria"""
    try:
        stats = await service.get_statistics()
        return StatisticsResponse(status='ok', statistics=stats)
    except Exception as e:
        logger.exception("Erro ao obter estatísticas")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Erro ao obter estatísticas: {str(e)}"
        )
