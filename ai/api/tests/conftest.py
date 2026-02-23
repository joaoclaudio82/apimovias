from contextlib import contextmanager
from datetime import datetime
from http import HTTPStatus

import factory
import factory.fuzzy
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import StaticPool

from api.app import app
from api.database import get_session
from api.models import User, UserType, table_registry
from api.security import create_access_token, get_password_hash


# -------------------------------------------------------------------------
# FIXTURE: Sessão assíncrona (banco em memória isolado)
# -------------------------------------------------------------------------
@pytest_asyncio.fixture
async def session():
    engine = create_async_engine(
        'sqlite+aiosqlite:///:memory:',
        connect_args={'check_same_thread': False},
        poolclass=StaticPool,
    )

    async with engine.begin() as conn:
        await conn.run_sync(table_registry.metadata.create_all)

    async with AsyncSession(engine, expire_on_commit=False) as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(table_registry.metadata.drop_all)


# -------------------------------------------------------------------------
# FIXTURE: Cliente de teste com override de sessão
# -------------------------------------------------------------------------
@pytest_asyncio.fixture
def client(session):
    def get_session_override():
        return session

    app.dependency_overrides[get_session] = get_session_override

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


# -------------------------------------------------------------------------
# USER FACTORY
# -------------------------------------------------------------------------
class UserFactory(factory.Factory):
    class Meta:
        model = User

    name = factory.Sequence(lambda n: f'Test {n}')
    username = factory.LazyAttribute(
        lambda obj: f'test{obj.name.replace(" ", "").lower()}'
    )
    type = UserType.user
    password = factory.LazyAttribute(lambda obj: f'{obj.username}@test.com')


@pytest_asyncio.fixture
async def user_factory(session):
    async def factory(**kwargs):
        password = 'secret'
        user = UserFactory(password=get_password_hash(password), **kwargs)
        session.add(user)
        await session.commit()
        await session.refresh(user)
        user.clean_password = password
        return user

    return factory


@pytest_asyncio.fixture
async def user(user_factory):
    """Usuário padrão para testes"""
    return await user_factory()


@pytest_asyncio.fixture
async def other_user(user_factory):
    """Outro usuário auxiliar"""
    return await user_factory()


@pytest_asyncio.fixture
async def admin_user(session):
    admin = User(
        name="Administrador",
        username="admin",
        type=UserType.admin,
        password=get_password_hash("admin123"),
    )
    session.add(admin)
    await session.commit()
    await session.refresh(admin)
    return admin

@pytest_asyncio.fixture
def admin_token(admin_user):
    token_data = {"sub": admin_user.username}
    return create_access_token(data=token_data)


# -------------------------------------------------------------------------
# FIXTURE: Token JWT autenticado
# -------------------------------------------------------------------------
@pytest_asyncio.fixture
def token(client, user):
    """Obtém o token JWT real via /auth/token"""
    response = client.post(
        '/auth/token',
        data={'username': user.username, 'password': user.clean_password},
        headers={'Content-Type': 'application/x-www-form-urlencoded'},
    )

    msg = f'Falha na autenticação: {response.text}'
    assert response.status_code == HTTPStatus.OK, msg

    return response.json()['access_token']


@pytest_asyncio.fixture
def token_auth_header(token):
    """Retorna o cabeçalho Authorization para endpoints protegidos"""
    return {'Authorization': f'Bearer {token}'}


# -------------------------------------------------------------------------
# MOCK TIME: para fixar timestamps de criação/atualização
# -------------------------------------------------------------------------
@contextmanager
def _mock_db_time(*, model, time=datetime(2025, 1, 1)):
    def fake_time_hook(mapper, connection, target):
        if hasattr(target, 'created_at'):
            target.created_at = time
        if hasattr(target, 'updated_at'):
            target.updated_at = time

    event.listen(model, 'before_insert', fake_time_hook)
    yield time
    event.remove(model, 'before_insert', fake_time_hook)


@pytest_asyncio.fixture
def mock_db_time():
    return _mock_db_time
