# api/routers/vehicle_profiles.py

from fastapi import APIRouter, HTTPException, Depends, Form
from http import HTTPStatus
from typing import Annotated
import pandas as pd
import logging

from api.database import get_session
from api.services.vehicle_profile_service import VehicleProfileService
from api.schemas.vehicle_profile_schemas import (
    ProfileImportResponse,
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
    '/import',
    response_model=ProfileImportResponse,
    status_code=HTTPStatus.CREATED,
    summary='Importa perfis de arquivos'
)
async def import_profiles(
    service: VehicleProfileService = Depends(get_service),
    target: str = Form(..., description="Target: 'km_dia_clean' ou 'h_dia_clean'"),
    profile_dir: str = Form(..., description="Diretório com arquivos do perfil"),
    segmentation_file_path: str = Form(..., description="Caminho do arquivo CSV com veiculo_id, target, segment")
):
    """
    Importa perfis de arquivos salvos pelo VehicleProfile
    
    Requer arquivo CSV de segmentação com colunas:
    - veiculo_id: ID do veículo
    - target: target do perfil (deve ser igual ao parâmetro target)
    - segment: número do segmento (0, 1, 2, ...)
    """
    logger.info('Importação de perfis solicitada')
    
    try:
        # Ler arquivo de segmentação do path fornecido
        try:
            df_segmentation = pd.read_csv(segmentation_file_path)
        except FileNotFoundError:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail=f"Arquivo de segmentação não encontrado: {segmentation_file_path}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail=f"Erro ao ler arquivo de segmentação: {str(e)}"
            )
        
        # Validar colunas
        required_cols = ['veiculo_id', 'target', 'segment']
        missing = [col for col in required_cols if col not in df_segmentation.columns]
        if missing:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail=f"Colunas faltando no arquivo de segmentação: {missing}"
            )
        
        # Importar
        result = await service.import_from_file(
            target=target,
            profile_dir=profile_dir,
            df_segmentation=df_segmentation
        )
        
        return ProfileImportResponse(
            target=result['target'],
            category=result['category'],
            n_vehicles=result['n_vehicles'],
            message='Importação concluída com sucesso'
        )
        
    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.error(f"Arquivo não encontrado: {e}")
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=f"Arquivo não encontrado: {str(e)}")
    except ValueError as e:
        logger.error(f"Erro de validação: {e}")
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Erro ao importar perfis")
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=f"Erro ao importar perfis: {str(e)}")

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
            '/update-csv-path',
            '/profile/{vehicle_id}',
            '/vehicle/{vehicle_id}/info',
            '/statistics',
            '/status'
        ]
    }
