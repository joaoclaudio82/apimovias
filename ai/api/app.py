# api/app.py

from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
from pathlib import Path

from api.database import engine
from api.models import table_registry
from api.routers.vehicle_profiles import router as vehicle_profiles_router
from api.config.prediction_config import PredictorConfig

logger = logging.getLogger("uvicorn")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle do app"""
    
    logger.info("Iniciando aplicação...")
    
    # Criar tabelas
    async with engine.begin() as conn:
        await conn.run_sync(table_registry.metadata.create_all)
    logger.info("Tabelas criadas/verificadas")
    
    # Carregar configuração de predição
    try:
        config_path = Path(__file__).parent.parent / "config" / "predictor_config.yaml"
        
        if config_path.exists():
            predictor_config = PredictorConfig.from_yaml(str(config_path))
            
            # ✅ Armazenar no app.state em vez de variável global
            app.state.predictor_config = predictor_config
            
            validation = predictor_config.validate_all_models_exist()
            logger.info(f"✅ Modelos encontrados: {len(validation['found'])}")
            
            if validation['missing']:
                logger.warning(f"⚠️  Modelos faltando: {len(validation['missing'])}")
        else:
            logger.warning(f"⚠️  Arquivo não encontrado: {config_path}")
            app.state.predictor_config = None
    
    except Exception as e:
        logger.error(f"❌ Erro ao carregar configuração: {e}")
        app.state.predictor_config = None

    logger.info(f"🔍 Config carregado? {app.state.predictor_config is not None}")
    logger.info("✅ Aplicação iniciada com sucesso!")
    
    yield
    
    logger.info("Finalizando aplicação...")
    await engine.dispose()
    logger.info("✅ Conexões fechadas.")


app = FastAPI(
    title="Movias AI API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)


@app.get('/health', tags=['health'])
async def health():
    return {'status': 'ok'}


# Incluir routers
app.include_router(vehicle_profiles_router)

try:
    from api.routers.predictions import router as predictions_router
    app.include_router(predictions_router)
except Exception as exc:
    logger.warning(
        'Router de predições desabilitado no startup (%s)',
        str(exc),
    )
