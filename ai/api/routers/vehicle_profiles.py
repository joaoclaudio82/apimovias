# api/routers/vehicle_profiles.py

from fastapi import APIRouter, HTTPException, Depends, Request, Body
from fastapi.responses import StreamingResponse

from http import HTTPStatus
from typing import Annotated
import pandas as pd
import io
import logging

from api.database import get_session
from api.services.vehicle_profile_service import VehicleProfileService
from api.schemas.vehicle_profile_schemas import (
    ImportFromFileResponse,
    ImportFromFileRequest,
    VehicleInfoResponse,
    StatisticsResponse,
)

from api.config.vehicle_profile_config import VehicleProfileConfig
from api.config.data_ingestion_config import DataIngestionConfig
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/vehicle-profiles', tags=['vehicle-profiles'])

Session = Annotated[AsyncSession, Depends(get_session)]


def get_vehicle_profile_config(request: Request) -> VehicleProfileConfig:
    """Dependency para obter configuração de perfis"""
    config = request.app.state.vehicle_profile_config
    if config is None:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="Configuração de perfis não disponível"
        )
    return config


def get_data_ingestion_config(request: Request) -> DataIngestionConfig:
    """Dependency para obter configuração de ingestão"""
    config = request.app.state.data_ingestion_config
    if config is None:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="Configuração de ingestão não disponível"
        )
    return config


def get_service(
    session: Session,
    profile_config: VehicleProfileConfig = Depends(get_vehicle_profile_config),
    ingestion_config: DataIngestionConfig = Depends(get_data_ingestion_config)
) -> VehicleProfileService:
    """Dependency para obter service"""
    return VehicleProfileService(session, profile_config, ingestion_config)


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


@router.get(
    '/status',
    status_code=HTTPStatus.OK,
    summary='Status do serviço'
)
async def get_status():
    """Status do serviço de perfis de veículos"""
    return {
        'status': 'ok',
        'message': 'Serviço de perfis disponível',
        'endpoints': [
            '/import',
            '/update-csv',
            '/profile/{vehicle_id}',
            '/vehicle/{vehicle_id}/info',
            '/statistics',
            '/status'
        ]
    }


@router.get(
    '/vehicles/category',
    summary='Lista todos os veículos com category e segment'
)
async def get_all_vehicles_category(
    service: VehicleProfileService = Depends(get_service),
    format: str = 'json'
):
    """
    Retorna todos os veículos com category e segment
    Query Parameters:
    - format: 'json' (padrão) ou 'csv'
    Retorna DataFrame com colunas: veiculo_id, category, segment
    """
    try:
        df = await service.get_all_vehicles_category()
        if format == 'csv':
            stream = io.StringIO()
            df.to_csv(stream, index=False)
            return StreamingResponse(
                iter([stream.getvalue()]),
                media_type="text/csv",
                headers={
                    "Content-Disposition": "attachment; filename=vehicles_category_segment.csv"
                }
            )
        else:
            return df.to_dict(orient='records')
    except Exception as e:
        logger.exception("Erro ao listar veículos")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Erro ao listar veículos: {str(e)}"
        )


@router.get(
    '/vehicles/category/{category}',
    summary='Lista veículos de uma categoria'
)
async def get_vehicles_by_category(
    category: str,
    service: VehicleProfileService = Depends(get_service)
):
    """
    Lista veículos de uma categoria específica
    Path Parameters:
    - category: 'km' ou 'h'
    Retorna DataFrame com colunas: veiculo_id, category, segment
    """
    try:
        df = await service.get_vehicles_by_category(category)
        return df.to_dict(orient='records')
    except ValueError as e:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception(f"Erro ao listar veículos da categoria {category}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Erro ao listar veículos: {str(e)}"
        )


@router.get(
    '/vehicles/category/{category}/segment/{segment}',
    summary='Lista veículos de uma categoria e segmento'
)
async def get_vehicles_by_segment(
    category: str,
    segment: int,
    service: VehicleProfileService = Depends(get_service)
):
    """
    Lista veículos de uma categoria e segmento específicos
  Path Parameters:
    - category: 'km' ou 'h'
    - segment: número do segmento (0, 1, 2, ...)
  Retorna DataFrame com colunas: veiculo_id, category, segment
    """
    try:
        df = await service.get_vehicles_by_segment(category, segment)
        return df.to_dict(orient='records')
    except ValueError as e:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception(f"Erro ao listar veículos {category} segmento {segment}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Erro ao listar veículos: {str(e)}"
        )


@router.post('/import', response_model=ImportFromFileResponse)
async def import_from_file(
    request: ImportFromFileRequest = Body(default_factory=ImportFromFileRequest),
    service: VehicleProfileService = Depends(get_service)
):
    """
    Importa e atualiza perfis usando o CSV configurado automaticamente.
  Pipeline completo unificado:
    1. Carrega arquivo da configuração/fallback
    2. Separa existentes/novos
    3. Atualiza existentes (concatena com samples)
    4. Classifica novos
    5. Reclassifica segments
    6. Extrai features
    7. Salva no banco
    """
    try:
        result = await service.import_from_file(
            vehicle_ids=request.vehicle_ids
        )
        return ImportFromFileResponse(status='success', **result)
    except Exception as e:
        logger.exception("Erro ao importar perfis")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
