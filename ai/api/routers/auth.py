from datetime import timedelta
from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_session
from api.models import User

from api.schemas.auth_schemas import ServiceTokenRequest, Token
from api.security import (
    create_access_token,
    get_current_user,
    verify_password,
)

router = APIRouter(prefix='/auth', tags=['auth'])

OAuth2Form = Annotated[OAuth2PasswordRequestForm, Depends()]
Session = Annotated[AsyncSession, Depends(get_session)]
CurrentUser = Annotated[User, Depends(get_current_user)]


@router.post('/token', response_model=Token)
async def login_for_access_token(form_data: OAuth2Form, session: Session):
    user = await session.scalar(select(User).where(User.username == form_data.username))

    if not user:
        raise HTTPException(
            status_code=HTTPStatus.UNAUTHORIZED,
            detail='Usuário ou senha incorretos',
        )

    if not verify_password(form_data.password, user.password):
        raise HTTPException(
            status_code=HTTPStatus.UNAUTHORIZED,
            detail='Usuário ou senha incorretos',
        )

    access_token = create_access_token(data={'sub': user.username})
    return {'access_token': access_token, 'token_type': 'bearer'}


@router.post('/refresh_token', response_model=Token)
async def refresh_access_token(user: CurrentUser):
    new_access_token = create_access_token(data={'sub': user.username})
    return {'access_token': new_access_token, 'token_type': 'bearer'}


SERVICE_ACCOUNTS = {
    'movias': 'moviaskey',
}


@router.post('/service_token', summary='Gera token JWT de serviço')
async def login_service_account(payload: ServiceTokenRequest):
    client_secret_expected = SERVICE_ACCOUNTS.get(payload.client_id)
    if not client_secret_expected or payload.client_secret != client_secret_expected:
        raise HTTPException(
            status_code=HTTPStatus.UNAUTHORIZED,
            detail='Credenciais de serviço inválidas',
        )
    access_token_expires = timedelta(hours=8)
    service_sub = f'service:{payload.client_id}'
    access_token = create_access_token(
        data={'sub': service_sub}, expires_delta=access_token_expires
    )
    return {'access_token': access_token, 'token_type': 'bearer'}
