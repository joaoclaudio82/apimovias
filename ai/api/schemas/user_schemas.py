from pydantic import BaseModel, ConfigDict

from api.models import (
    UserType,
)


class UserNew(BaseModel):
    name: str
    username: str
    type: UserType
    model_config = ConfigDict(from_attributes=True)


class UserPublic(UserNew):
    id: int


class UserSchema(UserNew):
    password: str


class UserList(BaseModel):
    users: list[UserPublic]

