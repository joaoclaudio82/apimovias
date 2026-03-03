# api/routers/training.py

from fastapi import (
    APIRouter, HTTPException, Depends, BackgroundTasks, Request
)
from http import HTTPStatus
from typing import Annotated
import logging

from api.database import get_session
from api.services.training_service import TrainingService
from api.config.training_config import TrainingConfig
from api.schemas.training_schemas import (
    TrainingRequest,
    TrainingResponse
)
from sqlalchemy.ext.asyncio import AsyncSession
from pathlib import Path

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/training', tags=['training'])

Session = Annotated[AsyncSession, Depends(get_session)]


def get_training_config(request: Request) -> TrainingConfig:
    """Dependency para obter configuração de treinamento"""
    config = request.app.state.training_config
    if config is None:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="Configuração de treinamento não disponível"
        )
    return config


def get_training_service(
    session: Session,
    config: TrainingConfig = Depends(get_training_config)
) -> TrainingService:
    """Dependency para obter service"""
    return TrainingService(session, config)


@router.post(
    '/run',
    response_model=TrainingResponse,
    summary='Executa treinamento de modelos'
)
async def run_training(
    request: TrainingRequest,
    service: TrainingService = Depends(get_training_service)
):
    """
    Executa treinamento de modelos
    **Fluxo:**
    1. Carrega datasets do diretório windows/
    2. Carrega perfis de veículos
    3. Treina modelos PyTorch (opcional)
    4. Treina modelos Globais (opcional)
    5. Salva resultados em pickle
    **Parâmetros:**
    - train_pytorch: Se True, treina modelos PyTorch
    - train_global: Se True, treina modelos Globais
    - verbose: Se True, exibe logs detalhados
    **Exemplo:**
    ```json
    {
      "train_pytorch": true,
      "train_global": true,
      "verbose": true
    }
    ```
    **Nota:** Este endpoint pode demorar bastante (minutos a horas).
    Considere usar background tasks para treinamentos longos.
    """
    logger.info('Treinamento solicitado')
    try:
        result = await service.run_training(
            train_pytorch=request.train_pytorch,
            train_global=request.train_global,
            verbose=request.verbose
        )
        # Verificar erros
        errors = {}
        if 'pytorch_error' in result:
            errors['pytorch'] = result['pytorch_error']
        if 'global_error' in result:
            errors['global'] = result['global_error']
        return TrainingResponse(
            status='success' if not errors else 'partial_success',
            models_trained=result['models_trained'],
            work_dir=result['work_dir'],
            pytorch_trained=result['pytorch_results'] is not None,
            global_trained=result['global_results'] is not None,
            errors=errors if errors else None
        )
    except FileNotFoundError as e:
        logger.error(f"Arquivo não encontrado: {e}")
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=str(e))
    except ValueError as e:
        logger.error(f"Erro de validação: {e}")
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Erro ao executar treinamento")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Erro ao executar treinamento: {str(e)}"
        )


@router.post(
    '/run-background',
    summary='Executa treinamento em background'
)
async def run_training_background(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    service: TrainingService = Depends(get_training_service)
):
    """
    Executa treinamento em background (não bloqueia a resposta)
    Útil para treinamentos longos. Retorna imediatamente e executa
    o treinamento em background.
    **Nota:** Não há como consultar o status do treinamento após iniciar.
    Considere implementar um sistema de jobs para isso.
    """
    logger.info('Treinamento em background solicitado')
    # Adicionar task em background
    background_tasks.add_task(
        service.run_training,
        train_pytorch=request.train_pytorch,
        train_global=request.train_global,
        verbose=request.verbose
    )
    return {
        'status': 'started',
        'message': 'Treinamento iniciado em background'
    }


@router.get(
    '/status',
    summary='Status do serviço de treinamento'
)
async def get_training_status(
    config: TrainingConfig = Depends(get_training_config)
):
    """Status do serviço de treinamento"""
    models_config = config.get_models_config()
    return {
        'status': 'ok',
        'windows_dir': config.windows_dir,
        'profile_dir': config.profile_dir,
        'work_dir': config.work_dir,
        'seed': config.seed,
        'pytorch_models': config.pytorch.models if config.pytorch.enabled else [],
        'global_models': config.global_.models if config.global_.enabled else [],
        'pytorch_enabled': models_config.pytorch.enabled,
        'global_enabled': models_config.global_.enabled
    }