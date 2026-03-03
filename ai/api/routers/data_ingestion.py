# api/routers/data_ingestion.py

from fastapi import APIRouter, HTTPException, Depends, Request
from http import HTTPStatus
from typing import Annotated
import logging

from api.database import get_session
from api.services.data_ingestion_service import DataIngestionService
from api.config.data_ingestion_config import DataIngestionConfig
from api.config.vehicle_profile_config import VehicleProfileConfig
from api.schemas.data_ingestion_schemas import (
    DataIngestionRequest,
    DataIngestionResponse,
)

from sqlalchemy.ext.asyncio import AsyncSession
from pathlib import Path

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/data-ingestion', tags=['data-ingestion'])

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


def get_data_ingestion_service(
    session: Session,
    config: DataIngestionConfig = Depends(get_data_ingestion_config),
    profile_config: VehicleProfileConfig = Depends(get_vehicle_profile_config),
) -> DataIngestionService:
    """Dependency para obter service"""
    return DataIngestionService(session, config, profile_config)


@router.post(
    '/run',
    response_model=DataIngestionResponse,
    summary='Executa pipeline de ingestão de dados'
)
async def run_data_ingestion(
    request: DataIngestionRequest,
    service: DataIngestionService = Depends(get_data_ingestion_service)
):
    """
    Executa pipeline completo de ingestão de dados
    
    Fluxo:
    1. Formata dataset (DatasetFormatter)
    2. Obtém segmentação do banco
    3. Divide por segmento (ClassSplitter)
    4. Gera janelas (DatasetGenerator)
    5. Limpa intermediários (opcional)
    
    **Parâmetros:**
    - clean_intermediate: Se True, remove arquivos intermediários
    - verbose: Se True, exibe logs detalhados
    
    **Exemplo:**
    ```json
    {
        "clean_intermediate": true,
        "verbose": true
    }
    ```
    """
    logger.info('Ingestão de dados solicitada')
    
    try:
        result = await service.run_ingestion(
            clean_intermediate=request.clean_intermediate,
            verbose=request.verbose
        )
        
        return DataIngestionResponse(
            status='success',
            categories_processed=result['categories_processed'],
            total_windows_generated=result['total_windows_generated'],
            output_dir=result['output_dir'],
            profile_dir=result['profile_dir'],
            intermediate_cleaned=result['intermediate_cleaned'],
            details={k: v for k, v in result.items() if k not in [
                'categories_processed', 'total_windows_generated', 
                'output_dir', 'profile_dir', 'intermediate_cleaned'
            ]}
        )
    
    except FileNotFoundError as e:
        logger.error(f"Arquivo não encontrado: {e}")
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=str(e))
    except ValueError as e:
        logger.error(f"Erro de validação: {e}")
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Erro ao executar ingestão")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Erro ao executar ingestão: {str(e)}"
        )


@router.get(
    '/status',
    summary='Status do serviço de ingestão'
)
async def get_ingestion_status(
    config: DataIngestionConfig = Depends(get_data_ingestion_config)
):
    """Status do serviço de ingestão"""
    return {
        'status': 'ok',
        'dataset_path': config.dataset_path,
        'output_dir': config.output_dir,
        'categories': config.categories,
        'formatter': {
            'min_days': config.formatter.min_days,
            'min_weeks': config.formatter.min_weeks,
            'max_gap': config.formatter.max_gap
        },
        'generator': {
            'history_size': config.generator.history_size,
            'forecast_horizon': config.generator.forecast_horizon
        }
    }