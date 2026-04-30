# database.py
from typing import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from api.settings import Settings

settings = Settings()

_is_sqlite = 'sqlite' in settings.DATABASE_URL

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=True,
    connect_args={'check_same_thread': False}
    if _is_sqlite
    else {},
)

# Activar WAL mode no SQLite para permitir leituras concorrentes durante escritas
if _is_sqlite:
    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

AsyncSessionLocal = sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


# Para uso manual (com decorator)
@asynccontextmanager
async def session_context():
    async with AsyncSessionLocal() as session:
        yield session
