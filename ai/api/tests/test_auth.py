from http import HTTPStatus

import pytest
from freezegun import freeze_time


@pytest.mark.asyncio
async def test_get_token(client, user):
    response = client.post(
        '/auth/token',
        data={'username': user.username, 'password': user.clean_password},
    )

    token = response.json()
    assert response.status_code == HTTPStatus.OK
    assert 'access_token' in token
    assert 'token_type' in token


@pytest.mark.asyncio
async def test_token_expired_after_time(client, user):
    with freeze_time('2025-07-15 12:00:00'):
        response = client.post(
            '/auth/token',
            data={'username': user.username, 'password': user.clean_password},
        )
        assert response.status_code == HTTPStatus.OK
        token = response.json()['access_token']

    with freeze_time('2025-07-15 12:31:00'):
        response = client.put(
            f'/users/{user.id}',
            headers={'Authorization': f'Bearer {token}'},
            json={
                'name': 'Wrong',
                'username': 'wrongwrong',
                'registration_number': 10,
                'password': 'wrong',
            },
        )

        assert response.status_code == HTTPStatus.UNAUTHORIZED
        msg = 'token expirado'
        assert response.json() == {
            'detail': f'Não foi possível validar as credenciais: {msg}'
        }


@pytest.mark.asyncio
async def test_token_inexistent_user(client):
    response = client.post(
        '/auth/token', data={'username': 'test', 'password': 'test_pass'}
    )

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    assert response.json() == {'detail': 'Usuário ou senha incorretos'}


@pytest.mark.asyncio
async def test_token_wrong_password(client, user):
    response = client.post(
        '/auth/token',
        data={'username': user.username, 'password': 'wrong_password'},
    )

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    assert response.json() == {'detail': 'Usuário ou senha incorretos'}


@pytest.mark.asyncio
async def test_token_refresh(client, token):
    response = client.post(
        '/auth/refresh_token', headers={'Authorization': f'Bearer {token}'}
    )

    data = response.json()

    assert response.status_code == HTTPStatus.OK
    assert 'access_token' in data
    assert 'token_type' in data
    assert data['token_type'] == 'bearer'


@pytest.mark.asyncio
async def test_token_expired_dont_refresh(client, user):
    with freeze_time('2025-08-01 12:00:00'):
        response = client.post(
            '/auth/token',
            data={'username': user.username, 'password': user.clean_password},
        )

        assert response.status_code == HTTPStatus.OK
        token = response.json()['access_token']

    with freeze_time('2025-08-01 12:31:00'):
        response = client.post(
            '/auth/refresh_token', headers={'Authorization': f'Bearer {token}'}
        )

        assert response.status_code == HTTPStatus.UNAUTHORIZED
        msg = 'token expirado'
        assert response.json() == {
            'detail': f'Não foi possível validar as credenciais: {msg}'
        }


@pytest.mark.asyncio
async def test_service_token_access(client):
    payload = {
        'client_id': 'viscosidade_batch',
        'client_secret': 'supersecretservicekey123',
    }
    response = client.post('/auth/service_token', json=payload)
    token = response.json()['access_token']
    headers = {'Authorization': f'Bearer {token}'}
    resp = client.get('/predictions/status', headers=headers)
    assert resp.status_code == HTTPStatus.OK
