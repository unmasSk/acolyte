"""
Acolyte Core module.

Exports the fundamental system components.
"""

# Configuration
from acolyte.core.secure_config import Settings, ConfigValidator

# Database - import directly from database.py using relative import
from .database import (
    DatabaseManager,
    InsightStore,  # Dream specialized store - CORRECTLY IN CORE (it's infrastructure)
    FetchType,
    QueryResult,
    StoreResult,
    get_db_manager,
)

# Exceptions and errors
from acolyte.core.exceptions import (
    # Python exceptions
    AcolyteError,
    DatabaseError,
    VectorStaleError,
    ConfigurationError,
    ValidationError,
    NotFoundError,
    ExternalServiceError,
    # HTTP response models
    ErrorType,
    ErrorDetail,
    ErrorResponse,
    # Helper functions
    validation_error,
    not_found_error,
    internal_error,
    external_service_error,
    configuration_error,
    from_exception,
)

# Logging
from acolyte.core.logging import (
    AsyncLogger,
    SensitiveDataMasker,
    PerformanceLogger,
    logger,  # Pre-configured global logger
)

# Events
from acolyte.core.events import EventType, Event, EventBus, WebSocketManager

# LLM
from acolyte.core.ollama import OllamaClient

# Chunking
from acolyte.core.chunking_config import (
    ChunkingStrategy,
    ChunkingConfig,
    StrategyConfig,
    ValidationResult,
)

# Tokens
from acolyte.core.token_counter import (
    TokenEncoder,
    OllamaEncoder,
    SmartTokenCounter,
    TokenBudgetManager,
    TokenCount,
    ContextSplit,
    TruncateStrategy,
)

# Tracing and metrics
from acolyte.core.tracing import tracer, metrics, LocalTracer, MetricsCollector

# ID generator
from acolyte.core.id_generator import IDGenerator, generate_id, is_valid_id

# Important constants
OLLAMA_MODEL = "acolyte:latest"
DEFAULT_BIND_HOST = "127.0.0.1"
DEFAULT_BIND_PORT = 8000
ID_LENGTH = 32  # Hex ID length

# Module version
__version__ = "0.1.0"

# Public exports list
__all__ = [
    # Configuration
    "Settings",
    "ConfigValidator",
    # Database
    "DatabaseManager",
    "InsightStore",
    "FetchType",
    "QueryResult",
    "StoreResult",
    "get_db_manager",
    # Exceptions
    "AcolyteError",
    "DatabaseError",
    "VectorStaleError",
    "ConfigurationError",
    "ValidationError",
    "NotFoundError",
    "ExternalServiceError",
    # HTTP error models
    "ErrorType",
    "ErrorDetail",
    "ErrorResponse",
    # Error helper functions
    "validation_error",
    "not_found_error",
    "internal_error",
    "external_service_error",
    "configuration_error",
    "from_exception",
    # Logging
    "AsyncLogger",
    "SensitiveDataMasker",
    "PerformanceLogger",
    "logger",
    # Eventos
    "EventType",
    "Event",
    "EventBus",
    "WebSocketManager",
    # LLM
    "OllamaClient",
    # Chunking
    "ChunkingStrategy",
    "ChunkingConfig",
    "StrategyConfig",
    "ValidationResult",
    # Tokens
    "TokenEncoder",
    "OllamaEncoder",
    "SmartTokenCounter",
    "TokenBudgetManager",
    "TokenCount",
    "ContextSplit",
    "TruncateStrategy",
    # Tracing
    "tracer",
    "metrics",
    "LocalTracer",
    "MetricsCollector",
    # Generador de IDs
    "IDGenerator",
    "generate_id",
    "is_valid_id",
    # Constants
    "OLLAMA_MODEL",
    "DEFAULT_BIND_HOST",
    "DEFAULT_BIND_PORT",
    "ID_LENGTH",
]
