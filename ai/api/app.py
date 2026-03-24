# api/app.py
from http import HTTPStatus
from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
from pathlib import Path

from api.database import engine
from api.models import table_registry
from api.routers import vehicle_profiles, predictions, data_ingestion, training, evaluation
from api.schemas import Message
from api.config.prediction_config import PredictorConfig
from api.config.vehicle_profile_config import VehicleProfileConfig
from api.config.data_ingestion_config import DataIngestionConfig
from api.config.training_config import TrainingConfig

logger = logging.getLogger("uvicorn")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle do app"""
    logger.info("="*60)
    logger.info("INICIANDO APLICAÇÃO")
    logger.info("="*60)
    
    # Criar tabelas
    async with engine.begin() as conn:
        await conn.run_sync(table_registry.metadata.create_all)
    logger.info("✅ Tabelas criadas/verificadas")
   
    # Carregar configurações
    logger.info("\n" + "="*60)
    logger.info("CARREGANDO CONFIGURAÇÕES")
    logger.info("="*60)
    config_base_path = Path(__file__).parent.parent / "config"

    # 1. Predictor Config
    try:
        predictor_config_path = config_base_path / "predictor_config.yaml"
        if predictor_config_path.exists():
            app.state.predictor_config = PredictorConfig.from_yaml(str(predictor_config_path))
            # Validar modelos
            validation = app.state.predictor_config.validate_all_models_exist()
            logger.info(f"✅ Predictor config: {len(validation['found'])} modelos encontrados")
            if validation['missing']:
                logger.warning(f"⚠️  {len(validation['missing'])} modelos faltando")
        else:
            app.state.predictor_config = None
            logger.warning(f"⚠️  Predictor config não encontrada: {predictor_config_path}")
    except Exception as e:
        logger.error(f"❌ Erro ao carregar predictor config: {e}")
        app.state.predictor_config = None
    
    # 2. Vehicle Profile Config
    try:
        profile_config_path = config_base_path / "vehicle_profile_config.yaml"
        if profile_config_path.exists():
            app.state.vehicle_profile_config = VehicleProfileConfig.from_yaml(str(profile_config_path))
            logger.info("✅ Vehicle profile config carregada")
        else:
            app.state.vehicle_profile_config = None
            logger.warning(f"⚠️  Vehicle profile config não encontrada: {profile_config_path}")
    except Exception as e:
        logger.error(f"❌ Erro ao carregar vehicle profile config: {e}")
        app.state.vehicle_profile_config = None
    
    # 3. Data Ingestion Config
    try:
        ingestion_config_path = config_base_path / "data_ingestion_config.yaml"
        if ingestion_config_path.exists():
            app.state.data_ingestion_config = DataIngestionConfig.from_yaml(str(ingestion_config_path))
            logger.info("✅ Data ingestion config carregada")
        else:
            app.state.data_ingestion_config = None
            logger.warning(f"⚠️  Data ingestion config não encontrada: {ingestion_config_path}")
    except Exception as e:
        logger.error(f"❌ Erro ao carregar data ingestion config: {e}")
        app.state.data_ingestion_config = None
    
    # 4. Training Config
    try:
        training_config_path = config_base_path / "training_config.yaml"
        if training_config_path.exists():
            app.state.training_config = TrainingConfig.from_yaml(str(training_config_path))
            logger.info("✅ Training config carregada")
        else:
            app.state.training_config = None
            logger.warning(f"⚠️  Training config não encontrada: {training_config_path}")
    except Exception as e:
        logger.error(f"❌ Erro ao carregar training config: {e}")
        app.state.training_config = None
    
    logger.info("="*60)
    logger.info("✅ APLICAÇÃO INICIADA COM SUCESSO")
    logger.info("="*60)
    yield
    
    logger.info("Finalizando aplicação...")
    await engine.dispose()
    logger.info("✅ Conexões fechadas")


app = FastAPI(
    title="Movias AI API",
    description="""
    API para gerenciamento de perfis de veículos e predições de manutenção.
    ## Funcionalidades
    * **Perfis de Veículos** - Importação, atualização e consulta de perfis
    * **Predições** - Predição de datas e valores acumulados
    * **Ingestão de Dados** - Pipeline de processamento de dados
    * **Treinamento** - Treinamento de modelos de ML
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)


@app.get(
    '/',
    status_code=HTTPStatus.OK,
    response_model=Message,
    tags=['health'],
    summary='Health check'
)
async def read_root():
    """Endpoint de health check"""
    return {'message': 'Movias AI API - Online'}

app.include_router(vehicle_profiles.router)
app.include_router(predictions.router)
app.include_router(data_ingestion.router)
app.include_router(training.router)
app.include_router(evaluation.router)