"""
Módulo Models de ACOLYTE.
Exporta todos los modelos principales para uso en otros módulos.
"""

# Base
from .base import (
    AcolyteBaseModel,
    TimestampMixin,
    IdentifiableMixin,  # Keep for compatibility
    # NEW:
    Identifiable,
    StandardIdMixin,
    SessionIdMixin,
    get_model_primary_key,
    get_model_primary_key_field,
)

# Chat (OpenAI compatible)
from acolyte.models.chat import Role, Message, ChatRequest, ChatResponse, Choice, Usage

# Chunks
from acolyte.models.chunk import ChunkType, ChunkMetadata, Chunk

# Conversaciones
from acolyte.models.conversation import (
    ConversationStatus,
    Conversation,
    ConversationSearchRequest,
    ConversationSearchResult,
)

# Tareas
from acolyte.models.task_checkpoint import TaskType, TaskStatus, TaskCheckpoint

# Decisiones técnicas
from acolyte.models.technical_decision import DecisionType, TechnicalDecision

# Documentos
from acolyte.models.document import DocumentType, Document, IndexingBatch, IndexingProgress

# Dream/Optimización
from acolyte.models.dream import (
    OptimizationStatus,
    InsightType,
    DreamInsight,
    OptimizationMetrics,
    OptimizationRequest,
    OptimizationResult,
    DreamState,
)

# Tipos semánticos
from acolyte.models.semantic_types import (
    TokenDistribution,
    TaskDetection,
    SummaryResult,
    SessionReference,
    DetectedDecision,
)

# Common metadata
from acolyte.models.common import FileMetadata, GitMetadata, LanguageInfo

# Errores ahora vienen de Core
from acolyte.core.exceptions import (
    ErrorType,
    ErrorResponse,
    ErrorDetail,
    validation_error,
    not_found_error,
    internal_error,
    external_service_error,
    configuration_error,
)

__all__ = [
    # Base
    "AcolyteBaseModel",
    "TimestampMixin",
    "IdentifiableMixin",  # Keep for compatibility
    # NEW:
    "Identifiable",
    "StandardIdMixin",
    "SessionIdMixin",
    "get_model_primary_key",
    "get_model_primary_key_field",
    # Chat
    "Role",
    "Message",
    "ChatRequest",
    "ChatResponse",
    "Choice",
    "Usage",
    # Chunks
    "ChunkType",
    "ChunkMetadata",
    "Chunk",
    # Conversaciones
    "ConversationStatus",
    "Conversation",
    "ConversationSearchRequest",
    "ConversationSearchResult",
    # Tareas
    "TaskType",
    "TaskStatus",
    "TaskCheckpoint",
    # Decisiones técnicas
    "DecisionType",
    "TechnicalDecision",
    # Documentos
    "DocumentType",
    "Document",
    "IndexingBatch",
    "IndexingProgress",
    # Dream
    "OptimizationStatus",
    "InsightType",
    "DreamInsight",
    "OptimizationMetrics",
    "OptimizationRequest",
    "OptimizationResult",
    "DreamState",
    # Tipos semánticos
    "TokenDistribution",
    "TaskDetection",
    "SummaryResult",
    "SessionReference",
    "DetectedDecision",
    # Metadata
    "FileMetadata",
    "GitMetadata",
    "LanguageInfo",
    # Errors (desde Core)
    "ErrorType",
    "ErrorResponse",
    "ErrorDetail",
    "validation_error",
    "not_found_error",
    "internal_error",
    "external_service_error",
    "configuration_error",
]
