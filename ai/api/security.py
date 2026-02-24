from datetime import datetime, timedelta
from http import HTTPStatus
from typing import TypeAlias, TypedDict, Union
from zoneinfo import ZoneInfo
import base64
import hashlib
import hmac
import logging
import os

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
logger = logging.getLogger(__name__)
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl='auth/token', refreshUrl='auth/refresh_token'
)


class PBKDF2PasswordHash:
    """Fallback de hash de senha usando somente bibliotecas padrão."""

    algorithm = 'pbkdf2_sha256'
    iterations = 600_000

    @staticmethod
    def _to_str(value: str | bytes) -> str:
        return value.decode('utf-8') if isinstance(value, bytes) else value

    def hash(self, password: str | bytes, *, salt: bytes | None = None) -> str:
        password_str = self._to_str(password)
        salt_bytes = salt or os.urandom(16)
        digest = hashlib.pbkdf2_hmac(
            'sha256',
            password_str.encode('utf-8'),
            salt_bytes,
            self.iterations,
        )
        salt_b64 = base64.b64encode(salt_bytes).decode('ascii')
        digest_b64 = base64.b64encode(digest).decode('ascii')
        return f'{self.algorithm}${self.iterations}${salt_b64}${digest_b64}'

    def verify(self, password: str | bytes, hashed_password: str | bytes) -> bool:
        password_str = self._to_str(password)
        hashed = self._to_str(hashed_password)
        try:
            algorithm, iterations, salt_b64, digest_b64 = hashed.split('$', 3)
            if algorithm != self.algorithm:
                return False
            salt = base64.b64decode(salt_b64.encode('ascii'))
            expected = base64.b64decode(digest_b64.encode('ascii'))
            computed = hashlib.pbkdf2_hmac(
                'sha256',
                password_str.encode('utf-8'),
                salt,
                int(iterations),
            )
            return hmac.compare_digest(computed, expected)
        except Exception:
            return False


def _build_password_context():
    try:
        return PasswordHash.recommended()
    except Exception as exc:
        logger.warning(
            "Argon2 indisponível (%s). Usando fallback PBKDF2.",
            str(exc),
        )
        return PBKDF2PasswordHash()


pwd_context = _build_password_context()


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
