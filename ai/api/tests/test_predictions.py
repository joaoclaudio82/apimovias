from http import HTTPStatus

import pytest


@pytest.mark.asyncio
async def test_predict_viscosity_valid(client, token_auth_header):
    """Teste feliz - predição válida"""
    response = client.post(
        '/predictions/viscosity',
        headers=token_auth_header,
        json={
            'smile_1': 'CCO',
            'smile_2': 'O=C=O',
            'fraction': 0.5,
            'temperature': 300.0,
        },
    )

    assert response.status_code == HTTPStatus.OK
    data = response.json()
    assert 'viscosity' in data
    assert isinstance(data['viscosity'], float)


@pytest.mark.asyncio
async def test_predict_viscosity_missing_smile1(client, token_auth_header):
    """Falta smile_1"""
    response = client.post(
        '/predictions/viscosity',
        headers=token_auth_header,
        json={'temperature': 300.0},
    )
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
async def test_predict_viscosity_fraction_without_smile2(client, token_auth_header):
    """fraction sem smile_2"""
    response = client.post(
        '/predictions/viscosity',
        headers=token_auth_header,
        json={'smile_1': 'CCO', 'fraction': 0.5, 'temperature': 300.0},
    )
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
async def test_predict_viscosity_smile2_without_fraction(client, token_auth_header):
    """smile_2 sem fraction"""
    response = client.post(
        '/predictions/viscosity',
        headers=token_auth_header,
        json={'smile_1': 'CCO', 'smile_2': 'O=C=O', 'temperature': 300.0},
    )
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
async def test_predict_viscosity_batch_valid(client, token_auth_header):
    """Teste feliz - Previsão de viscosidade em lote"""
    payload = {
        'inputs': [
            {'smile_1': 'CCO', 'temperature': 300.0},
            {
                'smile_1': 'CCO',
                'smile_2': 'O=C=O',
                'fraction': 0.5,
                'temperature': 320.0,
            },
        ]
    }

    response = client.post(
        '/predictions/viscosity/batch',
        headers=token_auth_header,
        json=payload,
    )

    assert response.status_code == HTTPStatus.OK, response.text

    data = response.json()
    assert 'predictions' in data
    assert len(data['predictions']) == len(payload['inputs'])

    # Cada resultado deve conter o campo viscosity (float)
    for result in data['predictions']:
        assert 'viscosity' in result
        assert isinstance(result['viscosity'], float)


@pytest.mark.asyncio
async def test_predict_viscosity_batch_missing_required(client, token_auth_header):
    """Deve retornar erro pois falta smile_1"""
    payload = {
        'inputs': [
            {'temperature': 298.15},
        ]
    }

    response = client.post(
        '/predictions/viscosity/batch',
        headers=token_auth_header,
        json=payload,
    )

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
async def test_predict_viscosity_batch_fraction_without_smile2(
    client, token_auth_header
):
    """fraction informado sem smile_2 — deve falhar"""
    payload = {
        'inputs': [
            {'smile_1': 'CCO', 'fraction': 0.5, 'temperature': 298.15},
        ]
    }

    response = client.post(
        '/predictions/viscosity/batch',
        headers=token_auth_header,
        json=payload,
    )

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
async def test_predict_viscosity_batch_smile2_without_fraction(
    client, token_auth_header
):
    """smile_2 informado sem fraction — deve falhar"""
    payload = {
        'inputs': [
            {'smile_1': 'CCO', 'smile_2': 'O=C=O', 'temperature': 298.15},
        ]
    }

    response = client.post(
        '/predictions/viscosity/batch',
        headers=token_auth_header,
        json=payload,
    )

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
async def test_predict_viscosity_batch_unauthorized(client):
    """Sem token — deve retornar 401"""
    payload = {'inputs': [{'smile_1': 'CCO', 'temperature': 298.15}]}

    response = client.post(
        '/predictions/viscosity/batch',
        json=payload,
    )

    assert response.status_code == HTTPStatus.UNAUTHORIZED
