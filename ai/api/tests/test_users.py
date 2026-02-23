from http import HTTPStatus

import pytest

from api.models import UserType
from api.schemas import UserPublic, UserSchema


# ----------------------------------------------
# Criar usuário — apenas admin pode criar
# ----------------------------------------------
@pytest.mark.asyncio
async def test_admin_can_create_user(client, admin_token):
    response = client.post(
        '/users/',
        headers={'Authorization': f'Bearer {admin_token}'},
        json={
            'name': 'Test da Silva',
            'username': 'test',
            'type': UserType.user.value,
            'password': 'secret',
        },
    )

    assert response.status_code == HTTPStatus.CREATED
    assert response.json() == {
        'name': 'Test da Silva',
        'username': 'test',
        'type': UserType.user.value,
    }


@pytest.mark.asyncio
async def test_non_admin_cannot_create_user(client, token):
    response = client.post(
        '/users/',
        headers={'Authorization': f'Bearer {token}'},
        json={
            'name': 'Test da Silva',
            'username': 'testuser',
            'type': UserType.user.value,
            'password': 'secret',
        },
    )

    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.json() == {
        'detail': 'Somente administradores podem criar usuários.'
    }


# ----------------------------------------------
# Criar com conflito
# ----------------------------------------------
@pytest.mark.asyncio
async def test_create_user_with_conflict(client, admin_token, user):
    response = client.post(
        '/users/',
        headers={'Authorization': f'Bearer {admin_token}'},
        json={
            'name': 'Test Pereira',
            'username': user.username,
            'type': UserType.user.value,
            'password': 'secret',
        },
    )

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json() == {'detail': f'Usuário "{user.username}" já existe.'}


# ----------------------------------------------
# Leitura (todos os usuários)
# ----------------------------------------------
@pytest.mark.asyncio
async def test_read_users(client):
    response = client.get('/users/')
    assert response.status_code == HTTPStatus.OK
    assert response.json() == {'users': []}


@pytest.mark.asyncio
async def test_read_users_with_users(client, user):
    user_schema = UserPublic.model_validate(user).model_dump()
    response = client.get('/users/')
    assert response.status_code == HTTPStatus.OK
    assert response.json() == {'users': [user_schema]}


@pytest.mark.asyncio
async def test_read_user_by_id(client, user):
    user_schema = UserPublic.model_validate(user).model_dump()

    response = client.get('/users/1')
    assert response.status_code == HTTPStatus.OK
    assert response.json() == user_schema

    response = client.get('/users/999')
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json() == {'detail': 'Usuário não encontrado'}


# ----------------------------------------------
# Atualização — apenas admin pode
# ----------------------------------------------
@pytest.mark.asyncio
async def test_admin_update_user(client, admin_token, user):
    user_schema = UserSchema.model_validate(user).model_dump()
    user_schema['name'] = 'Test Pereira'

    user_public = UserPublic(**user_schema)

    response = client.put(
        f'/users/{user.id}',
        headers={'Authorization': f'Bearer {admin_token}'},
        json=user_schema,
    )
    assert response.status_code == HTTPStatus.OK
    assert response.json() == user_public.model_dump()


@pytest.mark.asyncio
async def test_update_user_integrity_error(client, admin_token, user, other_user):
    response = client.put(
        f'/users/{user.id}',
        headers={'Authorization': f'Bearer {admin_token}'},
        json={
            'name': 'Teste da Silva',
            'username': other_user.username,  # causa conflito
            'type': UserType.admin.value,
            'password': 'secret',
        },
    )

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json() == {'detail': 'Usuário ou matrícula já existem'}


# ----------------------------------------------
# Exclusão — apenas admin pode
# ----------------------------------------------
@pytest.mark.asyncio
async def test_admin_can_delete_user(client, admin_token, user):
    response = client.delete(
        f'/users/{user.id}',
        headers={'Authorization': f'Bearer {admin_token}'},
    )

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {'message': 'Usuário removido.'}


@pytest.mark.asyncio
async def test_non_admin_cannot_delete_user(client, token, user):
    response = client.delete(
        f'/users/{user.id}',
        headers={'Authorization': f'Bearer {token}'},
    )

    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.json() == {
        'detail': 'Somente administradores podem deletar usuários.'
    }
