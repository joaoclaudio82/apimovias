# database.py
from typing import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from api.settings import Settings

settings = Settings()

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=True,
    connect_args={'check_same_thread': False}
    if 'sqlite' in settings.DATABASE_URL
    else {},
)

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