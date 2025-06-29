from datetime import datetime
from typing import Optional, Protocol, runtime_checkable, TypeAlias
from pydantic import BaseModel, Field

class TimestampMixin(BaseModel):
    created_at: datetime = Field(...)
    updated_at: Optional[datetime] = Field(default=None)

    def touch(self) -> None: ...

class AcolyteBaseModel(BaseModel): ...

@runtime_checkable
class Identifiable(Protocol):
    @property
    def primary_key(self) -> str: ...
    @property
    def primary_key_field(self) -> str: ...

class StandardIdMixin(BaseModel):
    id: str = Field(...)

    @property
    def primary_key(self) -> str: ...
    @property
    def primary_key_field(self) -> str: ...

class SessionIdMixin(BaseModel):
    session_id: str = Field(...)

    @property
    def primary_key(self) -> str: ...
    @property
    def primary_key_field(self) -> str: ...

IdentifiableMixin: TypeAlias = StandardIdMixin

def get_model_primary_key(model: Identifiable) -> str: ...
def get_model_primary_key_field(model: Identifiable) -> str: ...
