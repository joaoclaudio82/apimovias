from datetime import datetime, timedelta
from http import HTTPStatus
from typing import TypeAlias, TypedDict, Union
from zoneinfo import ZoneInfo

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import jwt
from jose.exceptions import ExpiredSignatureError, JWTError
from pwdlib import PasswordHash
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_session
from api.models import User
from api.settings import Settings

settings = Settings()
pwd_context = PasswordHash.recommended()
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl='auth/token', refreshUrl='auth/refresh_token'
)


class ServiceIdentity(TypedDict):
    service_name: str
    is_service: bool


AuthSubject: TypeAlias = Union[User, ServiceIdentity]


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Gera um Access Token JWT assinado com tempo de expiração opcional."""
    to_encode = data.copy()
    expire = datetime.now(tz=ZoneInfo('UTC')) + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({
        'exp': expire,
        'iat': datetime.now(tz=ZoneInfo('UTC')),
    })
    encoded_jwt = jwt.encode(
        claims=to_encode,
        key=settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )
    return encoded_jwt


def get_password_hash(password: str):
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)


async def get_current_user(
    session: AsyncSession = Depends(get_session),
    token: str = Depends(oauth2_scheme),
) -> AuthSubject:
    """Valida token JWT de usuários humanos e tokens de serviço."""

    def credentials_exception(msg: str):
        return HTTPException(
            status_code=HTTPStatus.UNAUTHORIZED,
            detail=f'Não foi possível validar as credenciais: {msg}',
            headers={'WWW-Authenticate': 'Bearer'},
        )

    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        subject = payload.get('sub')

        if not subject:
            raise credentials_exception("campo 'sub' ausente no token")

    except ExpiredSignatureError:
        raise credentials_exception('token expirado')

    except JWTError:
        raise credentials_exception('token inválido')

    # -------------------------------------------------------------------------
    # 💡 NOVO: aceitar tokens de serviço (prefixo "service:")
    # -------------------------------------------------------------------------
    if subject.startswith('service:'):
        service_name = subject.split('service:')[1]
        # Retorna um dict simples identificando o consumidor
        return {'service_name': service_name, 'is_service': True}

    # -------------------------------------------------------------------------
    # 🧍 TOKENS DE USUÁRIO NORMAL
    # -------------------------------------------------------------------------
    user = await session.scalar(select(User).where(User.username == subject))
    if not user:
        raise credentials_exception('usuário não existe no banco')

    return user
