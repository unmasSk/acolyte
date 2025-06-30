"""
Módulo API - Interfaz HTTP de ACOLYTE.
Configurado para localhost only (127.0.0.1) por seguridad.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
import time
import uuid

# Core imports
from acolyte.core import get_db_manager, logger
from acolyte.core.secure_config import Settings
from acolyte.core.exceptions import (
    AcolyteError,
    ErrorResponse,
    from_exception,
    internal_error,
    validation_error,
    ErrorType,
)

# Routers import
from .openai import router as openai_router
from .health import router as health_router
from .dream import router as dream_router
from .index import router as index_router
from .websockets import router as ws_router


# Create configuration instance
config = Settings()
logger.info("API module initializing...", module="api")

# Determine if we are in debug mode
DEBUG = config.get("api.debug", True)


# Lifespan context manager para startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages the application lifecycle."""
    # Startup
    logger.info("ACOLYTE API starting...")

    try:
        port = config.get("ports.backend", 8000)
        logger.info("API configured", host="localhost", port=port)

        # Verificar que la BD está accesible
        get_db_manager()
        logger.info("SQLite database connected", status="success")

        # Verificar otros servicios críticos
        weaviate_port = config.get("ports.weaviate", 8080)
        ollama_port = config.get("ports.ollama", 11434)

        logger.info("Weaviate expected", host="localhost", port=weaviate_port)
        logger.info("Ollama expected", host="localhost", port=ollama_port)

        # Modo debug
        if DEBUG:
            logger.info("Debug mode active", docs_available=True, path="/api/docs")
        else:
            logger.info("Production mode active", docs_available=False)

        logger.info("ACOLYTE API started successfully")

    except Exception as e:
        logger.error("ACOLYTE API startup failed", error=str(e))
        raise

    # Yield control to FastAPI
    yield

    # Shutdown
    logger.info("ACOLYTE API shutting down", action="cleanup_started")

    # Close database connection
    try:
        # DatabaseManager doesn't require explicit close - SQLite closes automatically
        logger.info("SQLite cleanup checked", status="success")
    except Exception as e:
        logger.error("Database close failed", error=str(e))

    # Note: Ollama and WebSocketManager clean up automatically
    # Services handle their own instances

    logger.info("Resource cleanup completed", status="success")


# Create FastAPI application with lifespan
app = FastAPI(
    title="ACOLYTE Local Assistant",
    version="1.0.0",
    description="Asistente de programación 100% local con memoria infinita",
    docs_url="/api/docs" if DEBUG else None,  # Solo en desarrollo
    redoc_url=None,  # Deshabilitado siempre por seguridad
    openapi_url="/api/openapi.json" if DEBUG else None,  # Solo en desarrollo
    lifespan=lifespan,  # Usar el nuevo sistema de lifespan
)

# Compression middleware for responses > 1KB
app.add_middleware(GZipMiddleware, minimum_size=1024)

# CORS only for localhost (future dashboard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Dashboard web
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False,  # We don't need cookies
)


# Middleware for Request ID and timing
@app.middleware("http")
async def add_request_metadata(request: Request, call_next) -> Response:
    """
    Añade metadata a cada request:
    - X-Request-ID único
    - X-Processing-Time (solo con debug)
    - Logging de requests
    """
    # Generate unique Request ID
    request_id = f"req_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    start_time = time.time()

    # Add to request context
    request.state.request_id = request_id
    request.state.start_time = start_time

    # Log incoming request
    logger.info("API request", method=request.method, path=request.url.path, request_id=request_id)

    # Process request
    response = await call_next(request)

    # Calculate processing time
    processing_time = int((time.time() - start_time) * 1000)  # ms

    # Headers always present
    response.headers["X-Request-ID"] = request_id

    # Debug headers only if enabled
    if DEBUG and request.url.path.startswith("/v1/"):
        response.headers["X-Processing-Time"] = str(processing_time)
        # Other useful headers for debugging
        response.headers["X-API-Version"] = "1.0.0"

    # Log response
    logger.info(
        "API response",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        processing_time_ms=processing_time,
        request_id=request_id,
    )

    return response


# Global exception handlers
@app.exception_handler(AcolyteError)
async def acolyte_exception_handler(request: Request, exc: AcolyteError) -> Response:
    """
    Maneja excepciones internas de ACOLYTE convirtiendo a ErrorResponse.
    """
    error_response = from_exception(exc)

    # Log error with context
    logger.error(
        "ACOLYTE error handled",
        code=exc.code,
        error_message=exc.message,
        error_id=error_response.error_id,
        request_id=getattr(request.state, 'request_id', 'unknown'),
    )

    # Determine HTTP status code based on error type
    status_code_map = {
        "ValidationError": 400,
        "NotFoundError": 404,
        "ConfigurationError": 500,
        "ExternalServiceError": 503,
        "DatabaseError": 500,
        "SQLiteBusyError": 503,  # Service temporarily unavailable
        "SQLiteCorruptError": 500,
        "SQLiteConstraintError": 400,  # Bad request
        "VectorStaleError": 409,  # Conflict
    }

    status_code = status_code_map.get(exc.code, 500)

    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=status_code, content=error_response.model_dump(exclude_none=True)
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> Response:
    """
    Maneja errores de validación de Pydantic convirtiendo a formato estándar.
    """
    # Get the first validation error
    first_error = exc.errors()[0] if exc.errors() else {}
    field = ".".join(str(loc) for loc in first_error.get("loc", []))
    msg = first_error.get("msg", "Validation failed")

    error_response = validation_error(
        field=field or "request",
        value=first_error.get("input", "unknown"),
        reason="validation_failed",
        message=msg,
    )

    logger.warning(
        "Validation error",
        field=field,
        validation_message=msg,
        request_id=getattr(request.state, 'request_id', 'unknown'),
    )

    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=400, content=error_response.model_dump(exclude_none=True))


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> Response:
    """
    Maneja excepciones HTTP estándar.
    """
    error_response = ErrorResponse(
        error_type=ErrorType.INTERNAL,
        message=exc.detail or "HTTP error occurred",
        code=f"http_{exc.status_code}",
    )

    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=exc.status_code, content=error_response.model_dump(exclude_none=True)
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> Response:
    """
    Maneja cualquier excepción no capturada.
    """
    request_id = getattr(request.state, 'request_id', 'unknown')

    logger.error(
        "Unhandled exception",
        exception_type=type(exc).__name__,
        error=str(exc),
        request_id=request_id,
        exc_info=True,
    )

    error_response = internal_error(
        message="An unexpected error occurred",
        error_id=request_id,
        context={"exception_type": type(exc).__name__},
    )

    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=500, content=error_response.model_dump(exclude_none=True))


# Register routers with correct routes

# Public OpenAI-compatible routes (MANDATORY)
app.include_router(openai_router, prefix="/v1", tags=["OpenAI Compatible"])

# ACOLYTE system routes
app.include_router(health_router, prefix="/api", tags=["System"])

app.include_router(dream_router, prefix="/api/dream", tags=["Optimization"])

app.include_router(index_router, prefix="/api/index", tags=["Indexing"])

# WebSockets
app.include_router(ws_router, prefix="/api/ws", tags=["WebSocket"])


# Export app for use in main
__all__ = ["app"]
