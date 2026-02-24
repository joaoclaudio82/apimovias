# api/routers/predictions.py
from fastapi import APIRouter, HTTPException, Depends, Request
from http import HTTPStatus
from typing import Annotated, List
import logging

from api.database import get_session
from api.services.prediction_service import PredictorService
from api.config.prediction_config import PredictorConfig
from api.schemas.prediction_schemas import (
    DateToReachRequest,
    DateToReachResponse,
    AccumulatedAtStepRequest,
    AccumulatedAtStepResponse,
)
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/predictions', tags=['predictions'])

Session = Annotated[AsyncSession, Depends(get_session)]


# ✅ Dependency usando Request para acessar app.state
def get_predictor_config(request: Request) -> PredictorConfig:
    """Dependency para obter configuração de predição"""
    config = request.app.state.predictor_config
    
    if config is None:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="Configuração de predição não disponível"
        )
    
    return config


def get_predictor_service(
    session: Session,
    config: PredictorConfig = Depends(get_predictor_config)
) -> PredictorService:
    """Dependency para obter service"""
    return PredictorService(session, config)


@router.post(
    '/date-to-reach',
    response_model=List[DateToReachResponse],
    summary='Prediz data para atingir valor acumulado'
)
async def predict_date_to_reach(
    request: DateToReachRequest,
    service: PredictorService = Depends(get_predictor_service)
):
    """
    Encontra a data em que o valor acumulado DAS PREDIÇÕES ultrapassa o alvo
    
    O acumulado é calculado APENAS sobre as predições futuras, começando do zero.
    Não inclui o valor acumulado do histórico.
    
    **Exemplo de uso:**
    
    "Quando o veículo 1316 vai acumular 10.000 km a partir de hoje?"
    
    ```json
    {
      "vehicle_ids": [1316],
      "target_values": [10000.0],
      "n_jobs": 1
    }
    ```
    
    **Batch com paralelização:**
    
    ```json
    {
      "vehicle_ids": [1316, 18230, 10495],
      "target_values": [10000.0, 15000.0, 8000.0],
      "n_jobs": -1
    }
    ```
    """
    logger.info(
        'Predição date-to-reach solicitada: '
        f'{len(request.vehicle_ids)} veículos'
    )
    
    try:
        results = await service.predict_date_to_reach(
            vehicle_ids=request.vehicle_ids,
            target_values=request.target_values,
            n_jobs=request.n_jobs
        )
        
        logger.info(f'Predições concluídas: {len(results)} resultados')
        
        return results
        
    except ValueError as e:
        logger.error(f"Erro de validação: {e}")
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"Modelo não encontrado: {e}")
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception("Erro ao predizer date-to-reach")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Erro ao predizer: {str(e)}"
        )


@router.post(
    '/accumulated-at-step',
    response_model=List[AccumulatedAtStepResponse],
    summary='Prediz valor acumulado em step/data específico'
)
async def predict_accumulated_at_step(
    request: AccumulatedAtStepRequest,
    service: PredictorService = Depends(get_predictor_service)
):
    """
    Prediz o valor acumulado DAS PREDIÇÕES em uma data de referência ou após n_steps
    
    O acumulado é calculado APENAS sobre as predições futuras, começando do zero.
    Não inclui o valor acumulado do histórico.
    
    **Exemplo 1: Usar n_steps**
    
    "Quanto o veículo 1316 vai acumular nos próximos 90 dias?"
    
    ```json
    {
      "vehicle_ids": [1316],
      "n_steps": [90],
      "n_jobs": 1
    }
    ```
    
    **Exemplo 2: Usar reference_dates**
    
    "Quanto o veículo vai acumular até 31/12/2024?"
    
    ```json
    {
      "vehicle_ids": [1316],
      "reference_dates": ["2024-12-31"],
      "n_jobs": 1
    }
    ```
    
    **Exemplo 3: Batch com paralelização**
    
    ```json
    {
      "vehicle_ids": [1316, 18230, 10495],
      "n_steps": [90, 120, 60],
      "n_jobs": -1
    }
    ```
    """
    logger.info(
        'Predição accumulated-at-step solicitada: '
        f'{len(request.vehicle_ids)} veículos'
    )
    
    try:
        results = await service.predict_accumulated_at_step(
            vehicle_ids=request.vehicle_ids,
            n_steps=request.n_steps,
            reference_dates=request.reference_dates,
            n_jobs=request.n_jobs
        )
        
        logger.info(f'Predições concluídas: {len(results)} resultados')
        
        return results
        
    except ValueError as e:
        logger.error(f"Erro de validação: {e}")
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"Modelo não encontrado: {e}")
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception("Erro ao predizer accumulated-at-step")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Erro ao predizer: {str(e)}"
        )

