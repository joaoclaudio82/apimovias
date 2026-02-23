from http import HTTPStatus

from jwt import decode

from api.security import create_access_token, settings


def test_jwt():
    data = {'test': 'test'}
    token = create_access_token(data)

    decoded = decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    assert decoded['test'] == data['test']
    assert 'exp' in decoded


def test_jwt_invalid_token(client):
    response = client.delete(
        '/users/1', headers={'Authorization': 'Bearer token-invalido'}
    )

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    msg = 'token inválido'
    assert response.json() == {
        'detail': f'Não foi possível validar as credenciais: {msg}',
    }


def test_get_current_user_not_found(client):
    data = {'no-username': 'test'}
    token = create_access_token(data)

    response = client.delete(
        '/users/1',
        headers={'Authorization': f'Bearer {token}'},
    )

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    msg = "campo 'sub' ausente no token"
    assert response.json() == {
        'detail': f'Não foi possível validar as credenciais: {msg}',
    }


def test_get_current_user_does_not_exists(client):
    data = {'sub': 'tesp'}
    token = create_access_token(data)

    response = client.delete(
        '/users/1',
        headers={'Authorization': f'Bearer {token}'},
    )

    msg = 'usuário não existe no banco'
    assert response.status_code == HTTPStatus.UNAUTHORIZED
    assert response.json() == {
        'detail': f'Não foi possível validar as credenciais: {msg}',
    }
