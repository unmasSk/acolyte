"""
Jerarquía de excepciones unificada para ACOLYTE.
ÚNICA FUENTE de excepciones y respuestas de error para todo el sistema.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from acolyte.core.id_generator import generate_id
from acolyte.core.utils.datetime_utils import utc_now, format_iso


# ============================================================================
# PARTE 1: EXCEPCIONES PYTHON (Para raise/catch)
# ============================================================================


class AcolyteError(Exception):
    """
    Error base del sistema ACOLYTE.

    Características:
    1. Serialización estructurada
    2. Contexto rico
    3. Sugerencias de resolución
    4. ID único para tracking
    """

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        self.id: str = generate_id()
        self.timestamp: datetime = utc_now()
        self.message: str = message
        self.code: str = code or self.__class__.__name__
        self.context: Dict[str, Any] = context or {}
        self.cause: Optional[Exception] = cause
        self.suggestions: List[str] = []  # Lista de sugerencias de resolución
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializa a diccionario para API.

        Returns:
            {
                "error_id": "hex32chars",
                "code": "DatabaseError",
                "message": "Cannot connect to SQLite",
                "timestamp": "2024-01-20T10:30:00Z",
                "context": {...}
            }
        """
        result: Dict[str, Any] = {
            "error_id": self.id,
            "code": self.code,
            "message": self.message,
            "timestamp": format_iso(self.timestamp),
            "context": self.context,
        }

        if self.cause:
            result["cause"] = {"type": type(self.cause).__name__, "message": str(self.cause)}

        if self.suggestions:
            result["suggestions"] = self.suggestions

        return result

    def add_suggestion(self, suggestion: str) -> None:
        """
        Añade sugerencia de resolución al error.

        Las sugerencias se acumulan en una lista para proporcionar
        múltiples opciones de resolución al usuario.

        Args:
            suggestion: Texto con la sugerencia de resolución

        Example:
            error = DatabaseError("Cannot connect to SQLite")
            error.add_suggestion("Verificar que el archivo .acolyte.db existe")
            error.add_suggestion("Comprobar permisos de escritura en el directorio")
        """
        if not suggestion or not isinstance(suggestion, str):
            return

        # Evitar duplicados
        if suggestion not in self.suggestions:
            self.suggestions.append(suggestion)

    def is_retryable(self) -> bool:
        """Indica si el error es reintentable."""
        # Por defecto, los errores no son reintentables
        # Las subclases pueden override este comportamiento
        return False


class DatabaseError(AcolyteError):
    """Error relacionado con base de datos SQLite o Weaviate."""

    def is_retryable(self) -> bool:
        """Los errores de BD a veces son reintentables (locks, timeouts)."""
        return True


class SQLiteBusyError(DatabaseError):
    """Error SQLite BUSY - base de datos bloqueada temporalmente.

    Características:
    - ES REINTENTABLE: Causado por locks temporales
    - Estrategia: Exponential backoff
    - Común en escrituras concurrentes
    """

    def is_retryable(self) -> bool:
        """BUSY errors son siempre reintentables."""
        return True


class SQLiteCorruptError(DatabaseError):
    """Error SQLite CORRUPT - base de datos corrupta.

    Características:
    - NO ES REINTENTABLE: Error permanente
    - Requiere intervención manual
    - Backup y restauración necesarios
    """

    def is_retryable(self) -> bool:
        """CORRUPT errors nunca son reintentables."""
        return False


class SQLiteConstraintError(DatabaseError):
    """Error SQLite CONSTRAINT - violación de restricciones.

    Características:
    - NO ES REINTENTABLE: Error de lógica/datos
    - Indica problema en el código o datos
    - Requiere corrección de la query o datos
    """

    def is_retryable(self) -> bool:
        """CONSTRAINT errors no son reintentables."""
        return False


class VectorStaleError(AcolyteError):
    """Error cuando los embeddings están desactualizados."""

    pass


class ConfigurationError(AcolyteError):
    """Error de configuración del sistema."""

    pass


class ValidationError(AcolyteError):
    """
    Error de validación con detalles de campos.

    Estructura:
    {
        "errors": [
            {
                "field": "session_id",
                "value": "invalid",
                "reason": "not_hex_format",
                "message": "Must be 32 char hex"
            }
        ]
    }
    """

    pass


class NotFoundError(AcolyteError):
    """Error cuando un recurso no se encuentra."""

    pass


class ExternalServiceError(AcolyteError):
    """
    Error de servicio externo (Ollama, Weaviate).

    Tracking:
    - Servicio que falló
    - Tiempo de respuesta
    - Intentos realizados
    """

    def is_retryable(self) -> bool:
        """Los errores de servicios externos generalmente son reintentables."""
        return True


# ============================================================================
# PARTE 2: MODELOS DE RESPUESTA HTTP (Para API responses)
# ============================================================================


class ErrorType(str, Enum):
    """Tipos de error que puede devolver la API."""

    VALIDATION = "validation_error"
    NOT_FOUND = "not_found"
    INTERNAL = "internal_error"
    EXTERNAL_SERVICE = "external_service_error"
    CONFIGURATION = "configuration_error"
    AUTHENTICATION = "authentication_error"
    RATE_LIMIT = "rate_limit_error"


class ErrorDetail(BaseModel):
    """Detalle específico de un error de validación."""

    field: str = Field(..., description="Campo que falló la validación")
    value: Any = Field(..., description="Valor inválido recibido")
    reason: str = Field(..., description="Razón del fallo")
    message: str = Field(..., description="Mensaje explicativo")


class ErrorResponse(BaseModel):
    """
    Respuesta de error estructurada para la API.
    Compatible con el formato esperado por clientes OpenAI.
    """

    error_type: ErrorType = Field(..., description="Tipo de error")
    message: str = Field(..., description="Mensaje principal del error")
    error_id: Optional[str] = Field(default=None, description="ID único para tracking")

    # Detalles adicionales
    details: Optional[List[ErrorDetail]] = Field(
        default=None, description="Detalles específicos (para errores de validación)"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None, description="Contexto adicional del error"
    )
    suggestions: Optional[List[str]] = Field(
        default=None, description="Sugerencias para resolver el error"
    )

    # Para compatibilidad con formato OpenAI
    code: Optional[str] = Field(default=None, description="Código del error")
    param: Optional[str] = Field(default=None, description="Parámetro que causó el error")
    type: Optional[str] = Field(default=None, description="Tipo de error OpenAI")


# ============================================================================
# PARTE 3: FUNCIONES HELPER (Puente entre excepciones y respuestas)
# ============================================================================


def validation_error(
    field: str, value: Any, reason: str, message: Optional[str] = None
) -> ErrorResponse:
    """
    Crea un error de validación con detalles del campo.

    Args:
        field: Campo que falló
        value: Valor recibido
        reason: Razón del fallo (e.g., "not_hex_format", "too_long")
        message: Mensaje personalizado (opcional)

    Returns:
        ErrorResponse configurado para validación
    """
    default_message = f"Validation failed for field '{field}'"

    return ErrorResponse(
        error_type=ErrorType.VALIDATION,
        message=message or default_message,
        details=[
            ErrorDetail(
                field=field,
                value=value,
                reason=reason,
                message=message or f"Invalid value for {field}: {reason}",
            )
        ],
        code="validation_error",
        param=field,
    )


def not_found_error(
    resource: str, identifier: str, suggestions: Optional[List[str]] = None
) -> ErrorResponse:
    """
    Crea un error para recurso no encontrado.

    Args:
        resource: Tipo de recurso (e.g., "session", "task", "file")
        identifier: ID o nombre del recurso
        suggestions: Sugerencias opcionales

    Returns:
        ErrorResponse configurado para not found
    """
    return ErrorResponse(
        error_type=ErrorType.NOT_FOUND,
        message=f"{resource.capitalize()} not found: {identifier}",
        context={"resource": resource, "identifier": identifier},
        suggestions=suggestions
        or [f"Verificar que el {resource} existe", f"Comprobar el ID del {resource}"],
        code="not_found",
        type="invalid_request_error",
    )


def internal_error(
    message: str = "An internal error occurred",
    error_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> ErrorResponse:
    """
    Crea un error interno del servidor.

    Args:
        message: Mensaje del error
        error_id: ID para tracking (se genera si no se proporciona)
        context: Contexto adicional

    Returns:
        ErrorResponse configurado para error interno
    """
    return ErrorResponse(
        error_type=ErrorType.INTERNAL,
        message=message,
        error_id=error_id or generate_id(),
        context=context,
        suggestions=[
            "Si el error persiste, revisar los logs",
            "Intentar de nuevo en unos momentos",
        ],
        code="internal_server_error",
        type="api_error",
    )


def external_service_error(
    service: str, message: Optional[str] = None, retry_after: Optional[int] = None
) -> ErrorResponse:
    """
    Crea un error de servicio externo (Ollama, Weaviate).

    Args:
        service: Nombre del servicio que falló
        message: Mensaje personalizado
        retry_after: Segundos para reintentar (opcional)

    Returns:
        ErrorResponse configurado para servicio externo
    """
    default_message = f"External service '{service}' is unavailable"

    context: Dict[str, Any] = {"service": service}
    if retry_after:
        context["retry_after"] = retry_after

    return ErrorResponse(
        error_type=ErrorType.EXTERNAL_SERVICE,
        message=message or default_message,
        context=context,
        suggestions=[
            s
            for s in [
                f"Verificar que {service} está ejecutándose",
                f"Comprobar conectividad con {service}",
                "Reintentar en unos momentos" if retry_after else None,
            ]
            if s is not None
        ],
        code="external_service_error",
        type="api_error",
    )


def configuration_error(
    setting: str, current_value: Any = None, expected: Optional[str] = None
) -> ErrorResponse:
    """
    Crea un error de configuración.

    Args:
        setting: Nombre de la configuración problemática
        current_value: Valor actual (si se conoce)
        expected: Descripción del valor esperado

    Returns:
        ErrorResponse configurado para error de configuración
    """
    message = f"Invalid configuration for '{setting}'"
    if expected:
        message += f". Expected: {expected}"

    context: Dict[str, Any] = {"setting": setting}
    if current_value is not None:
        context["current_value"] = current_value

    suggestions = [
        f"Revisar la configuración '{setting}' en .acolyte",
        "Verificar el formato del archivo de configuración",
    ]
    if expected:
        suggestions.insert(0, f"El valor debe ser: {expected}")

    return ErrorResponse(
        error_type=ErrorType.CONFIGURATION,
        message=message,
        context=context,
        suggestions=suggestions,
        code="configuration_error",
        param=setting,
    )


def from_exception(exc: AcolyteError) -> ErrorResponse:
    """
    Convierte una excepción AcolyteError en ErrorResponse para API.

    Args:
        exc: Excepción a convertir

    Returns:
        ErrorResponse listo para serializar
    """
    # Mapear tipo de excepción a ErrorType
    error_type_map = {
        "ValidationError": ErrorType.VALIDATION,
        "NotFoundError": ErrorType.NOT_FOUND,
        "ConfigurationError": ErrorType.CONFIGURATION,
        "ExternalServiceError": ErrorType.EXTERNAL_SERVICE,
        "DatabaseError": ErrorType.INTERNAL,
        "SQLiteBusyError": ErrorType.INTERNAL,
        "SQLiteCorruptError": ErrorType.INTERNAL,
        "SQLiteConstraintError": ErrorType.VALIDATION,  # Constraint = validation error
        "VectorStaleError": ErrorType.INTERNAL,
    }

    error_type = error_type_map.get(exc.code, ErrorType.INTERNAL)

    return ErrorResponse(
        error_type=error_type,
        message=exc.message,
        error_id=exc.id,
        context=exc.context,
        suggestions=exc.suggestions or None,
        code=exc.code.lower().replace("error", "_error"),
        type="api_error" if error_type == ErrorType.INTERNAL else "invalid_request_error",
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Excepciones Python
    "AcolyteError",
    "DatabaseError",
    "SQLiteBusyError",
    "SQLiteCorruptError",
    "SQLiteConstraintError",
    "VectorStaleError",
    "ConfigurationError",
    "ValidationError",
    "NotFoundError",
    "ExternalServiceError",
    # Modelos de respuesta
    "ErrorType",
    "ErrorDetail",
    "ErrorResponse",
    # Funciones helper
    "validation_error",
    "not_found_error",
    "internal_error",
    "external_service_error",
    "configuration_error",
    "from_exception",
]
