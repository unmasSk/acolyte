"""
ACOLYTE - Your Private AI Programming Assistant with Infinite Memory.

Local and private AI copilot powered by Qwen-2.5-Coder through Ollama.
"""

from acolyte._version import __version__, __version_info__

__author__ = "Bextia"
__license__ = "BSL"

# Core components
from acolyte.core import (
    logger,
    Settings,
    get_db_manager,
    generate_id,
    AcolyteError,
    ValidationError,
    ConfigurationError,
    NotFoundError,
    ExternalServiceError,
)

# API application - Lazy import to avoid initialization logs during CLI help
# from acolyte.api import app  # Removed to prevent premature API initialization

# Main services
from acolyte.services import (
    ChatService,
    ConversationService,
    TaskService,
    GitService,
    IndexingService,
    ReindexService,
)

# Models - OpenAI compatible
from acolyte.models import (
    # Chat models
    Role,
    Message,
    ChatRequest,
    ChatResponse,
    Choice,
    Usage,
    # Core models
    Conversation,
    TaskCheckpoint,
    TechnicalDecision,
    # Chunk models
    ChunkType,
    ChunkMetadata,
    Chunk,
    # Dream models
    DreamState,
    DreamInsight,
    OptimizationStatus,
)

# RAG components
from acolyte.rag import HybridSearch
from acolyte.rag.chunking import ChunkerFactory

# Semantic components
from acolyte.semantic import (
    Summarizer,
    PromptBuilder,
    QueryAnalyzer,
    TaskDetector,
    DecisionDetector,
    ReferenceResolver,
)

# Dream system - simplified imports
from acolyte.dream import create_dream_orchestrator


# Lazy import function for API
def get_app():
    """
    Get the ACOLYTE FastAPI application (lazy import).

    This function imports the API only when needed, preventing
    initialization logs from appearing during CLI help commands.
    """
    from acolyte.api import app

    return app


# Package metadata
__all__ = [
    # Version info
    "__version__",
    "__version_info__",
    "__author__",
    "__license__",
    # Core
    "logger",
    "Settings",
    "get_db_manager",
    "generate_id",
    # Exceptions
    "AcolyteError",
    "ValidationError",
    "ConfigurationError",
    "NotFoundError",
    "ExternalServiceError",
    # API - Use get_app() instead of direct import
    "get_app",
    # Services
    "ChatService",
    "ConversationService",
    "TaskService",
    "GitService",
    "IndexingService",
    "ReindexService",
    # Models
    "Role",
    "Message",
    "ChatRequest",
    "ChatResponse",
    "Choice",
    "Usage",
    "Conversation",
    "TaskCheckpoint",
    "TechnicalDecision",
    "ChunkType",
    "ChunkMetadata",
    "Chunk",
    "DreamState",
    "DreamInsight",
    "OptimizationStatus",
    # RAG
    "HybridSearch",
    "ChunkerFactory",
    # Semantic
    "Summarizer",
    "PromptBuilder",
    "QueryAnalyzer",
    "TaskDetector",
    "DecisionDetector",
    "ReferenceResolver",
    # Dream
    "create_dream_orchestrator",
]


# Convenience function for quick setup
def create_app():
    """
    Create and return the ACOLYTE FastAPI application.

    Example:
        >>> app = create_app()
        >>> # Use with uvicorn
        >>> import uvicorn
        >>> uvicorn.run(app, host="127.0.0.1", port=8000)

    Returns:
        FastAPI: Configured application instance
    """
    return get_app()


# Quick access to configuration
def get_config():
    """
    Get the current ACOLYTE configuration.

    Example:
        >>> config = get_config()
        >>> print(config.get("model.name"))
        'qwen2.5-coder:3b'

    Returns:
        Settings: Configuration instance
    """
    return Settings()


# Version check
def check_version():
    """
    Check ACOLYTE version and dependencies.

    Example:
        >>> info = check_version()
        >>> print(f"ACOLYTE {info['acolyte']} on Python {info['python']}")

    Returns:
        dict: Version information including:
            - acolyte: ACOLYTE version
            - python: Python version
            - platform: OS platform
            - torch: PyTorch version (if installed)
            - weaviate: Weaviate client version (if installed)
    """
    import sys
    import platform

    try:
        import torch

        torch_version = torch.__version__
    except ImportError:
        torch_version = "Not installed"

    try:
        import weaviate

        weaviate_version = weaviate.__version__
    except ImportError:
        weaviate_version = "Not installed"

    return {
        "acolyte": __version__,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(),
        "torch": torch_version,
        "weaviate": weaviate_version,
    }


def is_ready():
    """
    Check if ACOLYTE is ready to use.

    Example:
        >>> if is_ready():
        >>>     print("ACOLYTE is ready!")
        >>> else:
        >>>     print("ACOLYTE needs configuration")

    Returns:
        bool: True if ACOLYTE is properly configured and ready
    """
    try:
        # Check database
        db = get_db_manager()
        # Verify database is accessible by checking the connection path exists
        _ = db.db_path

        # Check configuration
        config = Settings()
        # Verify configuration is valid by accessing a required setting
        model_name = config.get("model.name")
        return model_name is not None
    except Exception:
        return False
