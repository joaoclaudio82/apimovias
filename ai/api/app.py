# api/app.py

from http import HTTPStatus
from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
from pathlib import Path

from api.database import engine, session_context
from api.models import table_registry, User
from api.routers import auth, users, vehicle_profiles, predictions
from api.schemas import Message
from api.schemas.user_schemas import UserType
from api.security import get_password_hash
from api.config.prediction_config import PredictorConfig
from sqlalchemy import select

logger = logging.getLogger("uvicorn")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle do app"""
    
    logger.info("Iniciando aplicação...")
    
    # Criar tabelas
    async with engine.begin() as conn:
        await conn.run_sync(table_registry.metadata.create_all)
    logger.info("Tabelas criadas/verificadas")
    
    # Criar admin
    async with session_context() as session:
        existing_admin = await session.scalar(
            select(User).where(User.username == "admin")
        )
        if not existing_admin:
            admin = User(
                name="Administrador do Sistema",
                username="admin",
                type=UserType.admin,
                password=get_password_hash("admin123"),
            )
            session.add(admin)
            await session.commit()
            logger.info("Usuário admin criado")
        else:
            logger.info("Usuário admin já existe")
    
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
    description="""API para gerenciamento de perfis de veículos e predições.""",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)


@app.get('/', status_code=HTTPStatus.OK, response_model=Message, tags=['health'])
async def read_root():
    return {'message': 'Movias AI API - Online'}


# Incluir routers
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(vehicle_profiles.router)
app.include_router(predictions.router)