# api/routers/vehicle_profiles.py

from fastapi import APIRouter, HTTPException, Depends
from http import HTTPStatus
from typing import Annotated
import logging

from api.database import get_session
from api.services.vehicle_profile_service import VehicleProfileService
from api.schemas.vehicle_profile_schemas import (
    ProfileUpdateResponse,
    ProfileUpdatePathRequest,
    VehicleInfoResponse,
    StatisticsResponse
)
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/vehicle-profiles', tags=['vehicle-profiles'])

Session = Annotated[AsyncSession, Depends(get_session)]


def get_service(session: Session) -> VehicleProfileService:
    """Dependency para obter service"""
    return VehicleProfileService(session)


@router.post(
    '/update-csv-path',
    response_model=ProfileUpdateResponse,
    summary='Atualiza perfis via caminho de CSV compartilhado'
)
async def update_profiles_csv_path(
    request: ProfileUpdatePathRequest,
    service: VehicleProfileService = Depends(get_service),
):
    logger.info(
        "Atualização via caminho CSV solicitada | target=%s | path=%s",
        request.target,
        request.csv_path,
    )

    try:
        result = await service.update_from_csv_path(
            target=request.target,
            csv_path=request.csv_path,
        )
        return ProfileUpdateResponse(
            target=request.target,
            n_vehicles=result['n_vehicles'],
            message='Atualização concluída com sucesso',
        )
    except FileNotFoundError as e:
        logger.error("Arquivo CSV não encontrado: %s", e)
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=str(e))
    except ValueError as e:
        logger.error("Erro de validação no CSV: %s", e)
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Erro ao atualizar perfis via CSV compartilhado")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Erro ao atualizar perfis: {str(e)}"
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

