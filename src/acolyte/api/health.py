"""
Health check system ACOLYTE.
Monitors the status of all critical components.
"""

from fastapi import APIRouter, Response
from typing import Dict, Any, Optional, List, cast
import psutil
import time
import asyncio
from acolyte.core.utils.datetime_utils import utc_now_iso

# Core imports
from acolyte.core import get_db_manager, logger, FetchType
from acolyte.core.ollama import OllamaClient
from acolyte.core.secure_config import Settings
from acolyte.core.exceptions import (
    internal_error,
)

# Services imports
# Note: We don't need ConversationService - we use SQL directly for stats

# Embeddings import
from acolyte.embeddings import get_embeddings

router = APIRouter()

# Track service start time
SERVICE_START_TIME = time.time()

# Configuration
config = Settings()
logger.info("Health API initializing...", module="health")

# Singleton instance of IndexingService (lazy loading)
_indexing_service = None


def get_indexing_service() -> Optional[Any]:
    """Get singleton instance of IndexingService with robust error handling."""
    global _indexing_service

    if _indexing_service is None:
        try:
            from acolyte.services import IndexingService

            _indexing_service = IndexingService()
            logger.info("IndexingService initialized", status="success")
        except ImportError as e:
            logger.warning("IndexingService not available", error=str(e))
            logger.info("[UNTESTED PATH] IndexingService import failed")
            _indexing_service = "not_available"  # Mark as not available
        except Exception as e:
            # Capture any other error during creation
            logger.error("IndexingService initialization failed", error=str(e), exc_info=True)
            logger.info("[UNTESTED PATH] IndexingService initialization error")
            _indexing_service = "error"  # Mark as error

    # Return None if not available or there was an error
    if _indexing_service in ["not_available", "error"]:
        return None

    return _indexing_service


@router.get("/health")
async def health_check(response: Response) -> Dict[str, Any]:
    """
    Full health check of the system.

    Checks:
    - Ollama (acolyte:latest)
    - Weaviate (vector database)
    - SQLite (metadata)
    - UniXcoder (embeddings)
    - System (CPU, memory, disk)

    Returns:
        Dict with full state and metrics
    """
    health_status: Dict[str, Any] = {
        "status": "healthy",  # healthy | degraded | unhealthy
        "timestamp": utc_now_iso(),
        "version": "1.0.0",
        "uptime_seconds": int(time.time() - SERVICE_START_TIME),
        "services": {},
        "system": {},
    }

    # List of checks to perform
    checks = [
        ("ollama", _check_ollama),
        ("database", _check_database),
        ("embeddings", _check_embeddings),
        ("weaviate", _check_weaviate),
    ]

    # Execute checks in parallel for faster execution
    tasks = [check_func() for _, check_func in checks]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    for i, (service_name, _) in enumerate(checks):
        result = results[i]

        if isinstance(result, Exception):
            health_status["services"][service_name] = {
                "status": "unhealthy",
                "error": str(result),
                "error_type": type(result).__name__,
            }
            health_status["status"] = "unhealthy"
        elif isinstance(result, dict):
            health_status["services"][service_name] = result

            # Adjust general status based on service status
            if result["status"] == "unhealthy":
                health_status["status"] = "unhealthy"
            elif result["status"] == "degraded" and health_status["status"] == "healthy":
                health_status["status"] = "degraded"

    # Check system (always last)
    try:
        health_status["system"] = await _check_system()
    except Exception as e:
        logger.error("System check failed", error=str(e))
        health_status["system"] = {"error": str(e)}
        if health_status["status"] == "healthy":
            health_status["status"] = "degraded"

    # Set HTTP status code based on health
    if health_status["status"] == "unhealthy":
        response.status_code = 503  # Service Unavailable
    elif health_status["status"] == "degraded":
        response.status_code = 200  # OK with warnings

    logger.info(
        f"Health Check | Status: {health_status['status']} | "
        f"Services: {len([s for s in health_status['services'].values() if s.get('status') == 'healthy'])}/"
        f"{len(health_status['services'])}"
    )

    return health_status


@router.get("/stats")
async def system_stats() -> Dict[str, Any]:
    """
    General system statistics for ACOLYTE.
    Useful for dashboard and monitoring.
    """
    try:
        # Get services
        db_manager = get_db_manager()

        # Conversation statistics
        try:
            conversation_stats: Dict[str, Any] = {
                "total": 0,
                "today": 0,
                "this_week": 0,
                "average_messages_per_session": 0.0,
            }

            # Direct SQL query to get basic stats
            query = """
            SELECT 
                COUNT(*) as total_conversations,
                COUNT(CASE WHEN DATE(created_at) = DATE('now') THEN 1 END) as today,
                COUNT(CASE WHEN DATE(created_at) >= DATE('now', '-7 days') THEN 1 END) as this_week
            FROM conversations
            """

            result = await db_manager.execute_async(query, fetch=FetchType.ONE)
            if result.data and isinstance(result.data, dict):
                conversation_stats["total"] = result.data.get("total_conversations", 0)
                conversation_stats["today"] = result.data.get("today", 0)
                conversation_stats["this_week"] = result.data.get("this_week", 0)

        except Exception as e:
            logger.warning("Failed to get conversation stats", error=str(e))
            conversation_stats = {"error": "Unable to fetch stats"}

        # Indexing statistics
        indexing_service = get_indexing_service()

        if indexing_service is None:
            # Service not available or error during initialization
            indexing_stats = {
                "total_files": 0,
                "total_chunks": 0,
                "languages": {},
                "chunk_types": {},
                "last_indexed": None,
                "error": "IndexingService not available",
            }
        else:
            try:
                # Get real stats from singleton service
                indexing_stats = await indexing_service.get_stats()

            except Exception as e:
                logger.warning("Failed to get indexing stats", error=str(e))
                indexing_stats = {"error": "Unable to fetch stats", "details": str(e)}

        # Storage statistics
        try:
            # Size of SQLite database
            import os

            db_path = config.get("database.path", ".acolyte.db")
            db_size_mb = 0.0

            if os.path.exists(db_path):
                db_size_mb = os.path.getsize(db_path) / (1024 * 1024)

            storage_stats = {
                "database": {"size_mb": round(db_size_mb, 2)},
                "vectors": {"count": 0, "size_mb": 0.0},  # TODO: Stats de Weaviate
                "total_size_mb": round(db_size_mb, 2),
            }

        except Exception as e:
            logger.warning("Failed to get storage stats", error=str(e))
            storage_stats = {"error": "Unable to fetch stats"}

        return {
            "conversations": conversation_stats,
            "indexing": indexing_stats,
            "performance": {
                "uptime_seconds": int(time.time() - SERVICE_START_TIME),
                "average_response_time_ms": 0,  # TODO: Implement tracking
                "p95_response_time_ms": 0,
                "cache_hit_rate": 0.0,
            },
            "optimization": {
                "fatigue_level": 0.0,  # TODO: Implement when Dream is ready
                "last_optimization": None,
                "next_recommended": None,
                "optimizations_completed": 0,
            },
            "storage": storage_stats,
        }

    except Exception as e:
        logger.error("Stats retrieval failed", error=str(e), exc_info=True)
        error_response = internal_error(
            message="Failed to retrieve system statistics", context={"error_type": type(e).__name__}
        )
        return {"error": error_response.model_dump()}


@router.get("/websocket-stats")
async def websocket_stats() -> Dict[str, Any]:
    """
    Active WebSocket connection statistics.
    Useful for debugging and monitoring progress connections.
    """
    try:
        from acolyte.api.websockets.progress import get_connection_stats, get_active_tasks

        # Get connection statistics
        connection_stats = get_connection_stats()
        active_tasks = get_active_tasks()

        return {
            "active_connections": len(active_tasks),
            "active_task_ids": active_tasks,
            "connection_details": connection_stats,
            "timestamp": utc_now_iso(),
        }

    except ImportError:
        logger.warning("WebSocket progress module not available", module="websocket")
        return {
            "error": "WebSocket stats not available",
            "reason": "Progress module not imported",
        }
    except Exception as e:
        logger.error("WebSocket stats failed", error=str(e), exc_info=True)
        error_response = internal_error(
            message="Failed to retrieve WebSocket statistics",
            context={"error_type": type(e).__name__},
        )
        return {"error": error_response.model_dump()}


# ============================================================================
# INDIVIDUAL CHECK FUNCTIONS
# ============================================================================


async def _check_ollama() -> Dict[str, Any]:
    """
    Check the status of Ollama and the acolyte:latest model.
    """
    try:
        ollama_start = time.time()

        # Create Ollama client
        ollama_port = config.get("ports.ollama", 11434)
        ollama_client = OllamaClient(base_url=f"http://localhost:{ollama_port}")

        # ACOLYTE always uses acolyte:latest, no needs to list models
        # Try a simple generation to verify it works
        try:
            test_response = await ollama_client.generate(prompt="Hello", max_tokens=5)
            generation_works = bool(test_response)
        except Exception as e:
            logger.warning("Ollama test generation failed", error=str(e))
            generation_works = False

        response_time = int((time.time() - ollama_start) * 1000)

        return {
            "status": "healthy" if generation_works else "unhealthy",
            "model": "acolyte:latest",
            "generation_test": "passed" if generation_works else "failed",
            "response_time_ms": response_time,
        }

    except Exception as e:
        logger.error("Ollama check failed", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e),
            "suggestion": "Verify Ollama is running on localhost:11434",
        }


async def _check_database() -> Dict[str, Any]:
    """
    Check the status of the SQLite database.
    """
    try:
        db_start = time.time()
        db_manager = get_db_manager()

        # Verify that the main tables exist
        tables_result = await db_manager.execute_async(
            "SELECT name FROM sqlite_master WHERE type='table'", fetch=FetchType.ALL
        )
        # Type assertion for Pylance
        table_rows = cast(List[Dict[str, Any]], tables_result.data or [])
        tables = [row["name"] for row in table_rows]

        required_tables = {
            "conversations",
            "tasks",
            "task_checkpoints",
            "technical_decisions",
            "dream_state",
        }
        missing_tables = required_tables - set(tables)

        # Count records in main tables
        counts = {}
        for table in ["conversations", "tasks", "technical_decisions"]:
            if table in tables:
                count_result = await db_manager.execute_async(
                    f"SELECT COUNT(*) as count FROM {table}", fetch=FetchType.ONE
                )
                if count_result.data and isinstance(count_result.data, dict):
                    counts[table] = count_result.data.get("count", 0)

        response_time = int((time.time() - db_start) * 1000)

        # Calculate database size
        import os

        db_path = config.get("database.path", ".acolyte.db")
        size_mb = 0.0
        if os.path.exists(db_path):
            size_mb = os.path.getsize(db_path) / (1024 * 1024)

        status = "healthy"
        if missing_tables:
            status = "degraded"

        return {
            "status": status,
            "response_time_ms": response_time,
            "size_mb": round(size_mb, 2),
            "tables": {
                "total": len(tables),
                "required_present": len(required_tables - missing_tables),
                "missing": list(missing_tables) if missing_tables else None,
            },
            "record_counts": counts,
        }

    except Exception as e:
        logger.error("Database check failed", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e),
            "suggestion": "Check database file permissions and integrity",
        }


async def _check_embeddings() -> Dict[str, Any]:
    """
    Check the status of the UniXcoder embeddings service.
    """
    try:
        embeddings_start = time.time()

        # Get embeddings service (singleton)
        embeddings_service = get_embeddings()

        # Test simple embedding
        test_text = "Hello world"
        embedding = embeddings_service.encode(test_text)

        response_time = int((time.time() - embeddings_start) * 1000)

        # Verify that the embedding is valid
        is_valid = (
            embedding is not None
            and hasattr(embedding, 'numpy')
            and embedding.numpy.shape[0] == 768  # UniXcoder dimensions
        )

        # Get metrics if available
        cache_stats: Dict[str, Any] = {"enabled": False}
        try:
            if hasattr(embeddings_service, 'cache'):
                cache_stats = {
                    "enabled": True,
                    "size": int(getattr(embeddings_service.cache, 'size', 0)),
                    "hit_rate": float(getattr(embeddings_service.cache, 'hit_rate', 0.0)),
                }
        except Exception:
            pass

        return {
            "status": "healthy" if is_valid else "degraded",
            "model": "microsoft/unixcoder-base",
            "dimensions": 768,
            "test_embedding": "passed" if is_valid else "failed",
            "response_time_ms": response_time,
            "cache": cache_stats,
            "device": getattr(embeddings_service, 'device', 'unknown'),
        }

    except Exception as e:
        logger.error("Embeddings check failed", error=str(e))
        logger.info("[UNTESTED PATH] Embeddings service check failed")

        # Detect error type
        error_type = "unknown"
        if "CUDA" in str(e) or "device" in str(e).lower():
            error_type = "gpu_unavailable"
        elif "memory" in str(e).lower() or "OOM" in str(e):
            error_type = "out_of_memory"
        elif "model" in str(e).lower() or "load" in str(e).lower():
            error_type = "model_load_failed"

        return {
            "status": "unhealthy",
            "error": str(e),
            "error_type": error_type,
            "suggestion": _get_embeddings_suggestion(error_type),
        }


async def _check_weaviate() -> Dict[str, Any]:
    """
    Check the status of Weaviate and collections.
    Simplified error handling for mono-user.
    """
    weaviate_start = time.time()

    # 1. Verify if the library is installed
    try:
        import weaviate

        # Try modern import first (weaviate-client 3.x/4.x)
        try:
            from weaviate.exceptions import WeaviateConnectionError  # type: ignore
        except (ImportError, AttributeError):
            # Fallback for older versions or mocking
            WeaviateConnectionError = Exception
    except ImportError:
        return {
            "status": "degraded",
            "error": "weaviate_client_not_installed",
            "suggestion": "Install weaviate-client: pip install weaviate-client",
            "response_time_ms": int((time.time() - weaviate_start) * 1000),
        }

    # 2. Try connecting to Weaviate
    weaviate_url = f"http://localhost:{config.get('ports.weaviate', 8080)}"

    try:
        client = weaviate.Client(weaviate_url)
    except Exception as e:
        logger.info("[UNTESTED PATH] Weaviate client creation failed")
        return {
            "status": "unhealthy",
            "error": "client_creation_failed",
            "error_type": type(e).__name__,
            "url": weaviate_url,
            "suggestion": "Check Weaviate URL configuration",
            "response_time_ms": int((time.time() - weaviate_start) * 1000),
        }

    # 3. Verify connection
    try:
        is_ready = client.is_ready()
        if not is_ready:
            return {
                "status": "unhealthy",
                "error": "weaviate_not_ready",
                "url": weaviate_url,
                "suggestion": "Ensure Weaviate service is running",
                "response_time_ms": int((time.time() - weaviate_start) * 1000),
            }
    except WeaviateConnectionError as e:
        return {
            "status": "unhealthy",
            "error": "connection_failed",
            "error_details": str(e),
            "url": weaviate_url,
            "suggestion": f"Cannot reach Weaviate at {weaviate_url}",
            "response_time_ms": int((time.time() - weaviate_start) * 1000),
        }

    # 4. Get collection statistics
    try:
        schema = client.schema.get()
        schema_dict = cast(Dict[str, Any], schema)
        collections = schema_dict.get("classes", [])
        collection_stats = {}

        # Expected collections in ACOLYTE
        expected_collections = {"CodeChunk", "Document", "Conversation", "Task", "DreamInsight"}
        found_collections = {col["class"] for col in collections}
        missing_collections = expected_collections - found_collections

        # Get count of each collection
        for collection in collections:
            collection_name = collection["class"]
            try:
                count_result = client.query.aggregate(collection_name).with_meta_count().do()
                aggregate_data = (
                    count_result.get("data", {}).get("Aggregate", {}).get(collection_name, [])
                )
                count = aggregate_data[0].get("meta", {}).get("count", 0) if aggregate_data else 0
                collection_stats[collection_name] = count
            except Exception as e:
                logger.warning(
                    "Failed to get collection count", collection=collection_name, error=str(e)
                )
                collection_stats[collection_name] = 0

        # Add missing collections
        for missing in missing_collections:
            collection_stats[missing] = 0

        # Calculate total vectors
        total_vectors = sum(collection_stats.values())

        # Determine status
        status = "healthy"
        if missing_collections:
            status = "degraded"
        if len(missing_collections) == len(expected_collections):
            status = "unhealthy"

        result = {
            "status": status,
            "url": weaviate_url,
            "collections": collection_stats,
            "total_vectors": total_vectors,
            "response_time_ms": int((time.time() - weaviate_start) * 1000),
        }

        if missing_collections:
            result["missing_collections"] = list(missing_collections)
            result["suggestion"] = "Initialize missing collections using database setup scripts"

        return result

    except Exception as e:
        logger.error("Weaviate stats failed", error=str(e))
        logger.info("[UNTESTED PATH] Weaviate collection stats retrieval failed")
        return {
            "status": "degraded",
            "url": weaviate_url,
            "error": "stats_collection_failed",
            "error_type": type(e).__name__,
            "suggestion": "Weaviate is running but stats unavailable",
            "response_time_ms": int((time.time() - weaviate_start) * 1000),
        }


async def _check_system() -> Dict[str, Any]:
    """
    Check system resources (CPU, memory, disk).
    """
    try:
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()

        # Memory
        memory = psutil.virtual_memory()

        # Disk (current directory)
        disk = psutil.disk_usage(".")

        # Determine status based on resource usage
        status = "healthy"
        warnings = []

        if cpu_percent > 90:
            status = "degraded"
            warnings.append("High CPU usage")

        if memory.percent > 90:
            status = "degraded"
            warnings.append("High memory usage")

        if disk.percent > 90:
            status = "degraded"
            warnings.append("Low disk space")

        return {
            "status": status,
            "warnings": warnings if warnings else None,
            "cpu": {
                "percent": round(cpu_percent, 1),
                "cores": cpu_count,
            },
            "memory": {
                "percent": round(memory.percent, 1),
                "used_gb": round(memory.used / (1024**3), 2),
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
            },
            "disk": {
                "percent": round(disk.percent, 1),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "total_gb": round(disk.total / (1024**3), 2),
            },
        }

    except Exception as e:
        logger.error("System resource check failed", error=str(e))
        return {
            "status": "degraded",
            "error": str(e),
        }


def _get_embeddings_suggestion(error_type: str) -> str:
    """
    Get specific suggestion based on the type of embedding error.
    """
    suggestions = {
        "gpu_unavailable": "GPU not available, will fallback to CPU",
        "out_of_memory": "Insufficient memory, try reducing batch size",
        "model_load_failed": "Check if UniXcoder model files are available",
        "unknown": "Check embeddings service configuration",
    }
    return suggestions.get(error_type, suggestions["unknown"])
