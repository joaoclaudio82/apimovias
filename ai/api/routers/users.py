from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_session
from api.models import User
from api.schemas import FilterPage, Message
from api.schemas.user_schemas import (
    UserList, 
    UserPublic, 
    UserSchema, 
    UserType
)
    
from api.security import (
    AuthSubject,
    get_current_user,
    get_password_hash,
)

router = APIRouter(prefix='/users', tags=['users'])
Session = Annotated[AsyncSession, Depends(get_session)]
CurrentUser = Annotated[AuthSubject, Depends(get_current_user)]


@router.post('/', status_code=HTTPStatus.CREATED, response_model=UserPublic)
async def create_user(user: UserSchema, session: Session, current_user: CurrentUser):
    if current_user.type != UserType.admin:
        raise HTTPException(
            status_code=HTTPStatus.FORBIDDEN,
            detail='Somente administradores podem criar usuários.',
        )
    existing_user = await session.scalar(
        select(User).where(User.username == user.username)
    )
    if existing_user:
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT,
            detail=f'Usuário "{user.username}" já existe.',
        )
    db_user = User(
        name=user.name,
        username=user.username,
        type=user.type,
        password=get_password_hash(user.password),
    )
    session.add(db_user)
    await session.commit()
    await session.refresh(db_user)
    return db_user

@router.get('/me', response_model=UserPublic)
async def read_current_user(current_user: CurrentUser):
    """Retorna o perfil do usuário autenticado."""
    return current_user


@router.get('/', response_model=UserList)
async def read_users(session: Session, filter_users: Annotated[FilterPage, Query()]):
    query = await session.scalars(
        select(User).offset(filter_users.offset).limit(filter_users.limit)
    )

    users = query.all()
    return {'users': users}


@router.get('/{user_id}', response_model=UserPublic)
async def read_user_by_id(user_id: int, session: Session):
    db_user = await session.scalar(select(User).where(User.id == user_id))
    if not db_user:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND, detail='Usuário não encontrado'
        )

    return db_user


@router.put('/{user_id}', response_model=UserPublic)
async def update_user(
    user_id: int,
    user: UserSchema,
    session: Session,
    current_user: CurrentUser,
):
    if current_user.type != UserType.admin:
        raise HTTPException(
            status_code=HTTPStatus.FORBIDDEN,
            detail='Somente administradores podem atualizar usuários.',
        )

    db_user = await session.scalar(select(User).where(User.id == user_id))
    if not db_user:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail='Usuário não encontrado.',
        )

    try:
        db_user.name = user.name
        db_user.username = user.username
        db_user.type = user.type
        db_user.password = get_password_hash(user.password)
        await session.commit()
        await session.refresh(db_user)
        return db_user
    except IntegrityError:
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT,
            detail='Usuário  já existe.',
        )


@router.delete('/{user_id}', response_model=Message)
async def delete_user(
    user_id: int,
    session: Session,
    current_user: CurrentUser,
):
    if current_user.type != UserType.admin:
        raise HTTPException(
            status_code=HTTPStatus.FORBIDDEN,
            detail='Somente administradores podem deletar usuários.',
        )

    db_user = await session.scalar(select(User).where(User.id == user_id))
    if not db_user:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail='Usuário não encontrado.',
        )

    await session.delete(db_user)
    await session.commit()

    return {'message': 'Usuário removido.'}
