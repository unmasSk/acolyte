from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
from pydantic import BaseModel

class AcolyteError(Exception):
    id: str  # Generated internally, not passed to __init__
    timestamp: datetime
    message: str
    code: str
    context: Dict[str, Any]
    cause: Optional[Exception]
    suggestions: List[str]

    def __init__(
        self,
        message: str,
        code: Optional[str] = ...,
        context: Optional[Dict[str, Any]] = ...,
        cause: Optional[Exception] = ...,
    ) -> None: ...
    def to_dict(self) -> Dict[str, Any]: ...
    def add_suggestion(self, suggestion: str) -> None: ...
    def is_retryable(self) -> bool: ...

class DatabaseError(AcolyteError): ...
class SQLiteBusyError(DatabaseError): ...
class SQLiteCorruptError(DatabaseError): ...
class SQLiteConstraintError(DatabaseError): ...
class VectorStaleError(AcolyteError): ...
class ConfigurationError(AcolyteError): ...
class ValidationError(AcolyteError): ...
class NotFoundError(AcolyteError): ...
class ExternalServiceError(AcolyteError): ...

class ErrorType(str, Enum):
    VALIDATION = "validation_error"
    NOT_FOUND = "not_found"
    INTERNAL = "internal_error"
    EXTERNAL_SERVICE = "external_service_error"
    CONFIGURATION = "configuration_error"
    AUTHENTICATION = "authentication_error"
    RATE_LIMIT = "rate_limit_error"

class ErrorDetail(BaseModel):
    field: str
    value: Any
    reason: str
    message: str = ...

class ErrorResponse(BaseModel):
    error_type: ErrorType
    message: str
    error_id: Optional[str] = ...
    details: Optional[List[ErrorDetail]] = ...
    context: Optional[Dict[str, Any]] = ...
    suggestions: Optional[List[str]] = ...
    code: Optional[str] = ...
    param: Optional[str] = ...
    type: Optional[str] = ...

def validation_error(
    field: str, value: Any, reason: str, message: Optional[str] = ...
) -> ErrorResponse: ...
def not_found_error(
    resource: str, identifier: str, suggestions: Optional[List[str]] = ...
) -> ErrorResponse: ...
def internal_error(
    message: str = ..., error_id: Optional[str] = ..., context: Optional[Dict[str, Any]] = ...
) -> ErrorResponse: ...
def external_service_error(
    service: str, message: Optional[str] = ..., retry_after: Optional[int] = ...
) -> ErrorResponse: ...
def configuration_error(
    setting: str, current_value: Any = ..., expected: Optional[str] = ...
) -> ErrorResponse: ...
def from_exception(exc: AcolyteError) -> ErrorResponse: ...
