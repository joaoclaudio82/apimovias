from dataclasses import asdict

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_session
from api.models import User, UserType


@pytest.mark.asyncio
async def test_get_session():
    session_generator = get_session()
    session = await anext(session_generator)
    assert isinstance(session, AsyncSession)
    await session_generator.aclose()


@pytest.mark.asyncio
async def test_create_user(session, mock_db_time):
    with mock_db_time(model=User) as time:
        new_user = User(
            name='Test da Silva',
            username='test',
            type=UserType.admin,
            password='secret',
        )

        session.add(new_user)
        await session.commit()

    user = await session.scalar(select(User).where(User.username == 'test'))

    assert asdict(user) == {
        'id': 1,
        'name': 'Test da Silva',
        'username': 'test',
        'type': UserType.admin.value,
        'password': 'secret',
        'created_at': time,
        'updated_at': time,
    }
