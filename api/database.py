from sqlalchemy import create_engine, Engine
from api.settings import settings

engine: Engine = create_engine(
    f'postgresql+psycopg2://{settings.DATABASE_USER}:{settings.DATABASE_PASSWORD}@{settings.DATABASE_HOST}:{settings.DATABASE_PORT}/{settings.DATABASE_DATABASE}',
    connect_args={'sslmode': settings.DATABASE_SSL_MODE},
    pool_pre_ping=True,
)
