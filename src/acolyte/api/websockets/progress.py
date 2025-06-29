"""
WebSocket for long-running operations progress.
Primarily used for initial indexing from dashboard.

Implemented features:
- Real-time progress every 500ms
- Automatic heartbeat every 30s when no activity
- Automatic detection of lost connections
- Automatic cleanup of inactive connections
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, TypedDict, Any
from datetime import datetime
import asyncio
import time
from acolyte.core.utils.datetime_utils import utc_now

# Core imports
from acolyte.core.secure_config import Settings
from acolyte.core.logging import logger
from acolyte.core.events import event_bus, EventType, ProgressEvent, Event

router = APIRouter()

# Configuration
config = Settings()

# Configuration of heartbeat and connections from .acolyte with validation
# Apply safe limits to prevent memory/performance problems
max_connections_raw = config.get("websockets.max_connections", 100)
heartbeat_interval_raw = config.get("websockets.heartbeat_interval", 30)
connection_timeout_raw = config.get("websockets.connection_timeout", 60)

# Validate and apply safe limits
MAX_CONNECTIONS = max(1, min(max_connections_raw, 1000))  # Between 1 and 1000
HEARTBEAT_INTERVAL = max(10, min(heartbeat_interval_raw, 300))  # Between 10s and 5min
CONNECTION_TIMEOUT = max(30, min(connection_timeout_raw, 3600))  # Between 30s and 1h


# TypedDict for connection information
class ConnectionInfo(TypedDict):
    """Information about an active WebSocket connection."""

    websocket: WebSocket
    last_ping: float
    last_activity: float
    connected_at: datetime
    event_queue: asyncio.Queue[ProgressEvent]


# Log if limits
if max_connections_raw != MAX_CONNECTIONS:
    logger.warning(
        "WebSocket max_connections limited",
        original=max_connections_raw,
        limited_to=MAX_CONNECTIONS,
    )
if heartbeat_interval_raw != HEARTBEAT_INTERVAL:
    logger.warning(
        "WebSocket heartbeat_interval limited",
        original=heartbeat_interval_raw,
        limited_to=HEARTBEAT_INTERVAL,
    )
if connection_timeout_raw != CONNECTION_TIMEOUT:
    logger.warning(
        "WebSocket connection_timeout limited",
        original=connection_timeout_raw,
        limited_to=CONNECTION_TIMEOUT,
    )

# Memory storage of active connections with timestamps and queues
active_connections: Dict[str, ConnectionInfo] = {}

# Log final configuration
logger.info(
    "WebSocket configuration validated",
    max_connections=MAX_CONNECTIONS,
    heartbeat_interval=HEARTBEAT_INTERVAL,
    connection_timeout=CONNECTION_TIMEOUT,
)


@router.websocket("/progress/{task_id}")
async def websocket_progress(websocket: WebSocket, task_id: str):
    """
    WebSocket for real-time progress updates.

    Used primarily for:
    - Initial project indexing (can take minutes)
    - Full re-indexing (emergency cases)

    Message format sent:
    ```json
    {
        "type": "progress",
        "task_id": "idx_1234567_abc123",
        "status": "indexing",
        "progress_percent": 45.5,
        "current_operation": "Indexing src/services/auth.py",
        "stats": {
            "files_processed": 456,
            "files_total": 1000,
            "files_skipped": 23,
            "chunks_created": 2341,
            "embeddings_generated": 2341,
            "errors": 2
        },
        "timing": {
            "elapsed_seconds": 120,
            "estimated_remaining": 147,
            "files_per_second": 3.8
        },
        "timestamp": "2024-01-15T10:30:45Z"
    }
    ```

    New heartbeat message types:
    ```json
    {
        "type": "heartbeat",
        "task_id": "idx_1234567_abc123",
        "message": "Connection alive",
        "timestamp": "2024-01-15T10:30:45Z",
        "connection_duration": 120
    }
    ```
    """
    # Verify connection limit
    if len(active_connections) >= MAX_CONNECTIONS:
        await websocket.close(code=1008, reason="Too many connections")
        logger.warning(
            "WebSocket connection rejected",
            task_id=task_id,
            reason="max_connections_reached",
            max_connections=MAX_CONNECTIONS,
        )
        return

    await websocket.accept()

    # Create queue for events of this task
    event_queue: asyncio.Queue[ProgressEvent] = asyncio.Queue()

    # Register connection with timestamp and queue
    connection_info: ConnectionInfo = {
        "websocket": websocket,
        "last_ping": time.time(),
        "last_activity": time.time(),
        "connected_at": utc_now(),
        "event_queue": event_queue,
    }
    active_connections[task_id] = connection_info

    logger.info("WebSocket connected", task_id=task_id)

    # Subscribe to progress events for this task
    def handle_progress_event(event: Event):
        """Handles progress events from EventBus with robust task_id filtering."""
        # Verify that it is a ProgressEvent and has task_id
        if isinstance(event, ProgressEvent) and hasattr(event, 'task_id'):
            # Use dedicated task_id field for precise filtering
            if event.task_id == task_id:
                # Put event in queue for asynchronous processing
                try:
                    event_queue.put_nowait(event)
                except asyncio.QueueFull:
                    logger.warning("Event queue full", task_id=task_id)

    # Subscribe to progress events
    unsubscribe = event_bus.subscribe(
        EventType.PROGRESS,
        handle_progress_event,
        filter=lambda e: hasattr(e, 'source') and e.source == "indexing_service",
    )

    # Start heartbeat task in background
    heartbeat_task = asyncio.create_task(_heartbeat_handler(task_id))

    try:
        # Send initial message
        await websocket.send_json(
            {
                "type": "connected",
                "task_id": task_id,
                "message": "Connected to progress stream",
                "timestamp": utc_now().isoformat(),
            }
        )

        # Main loop listening for events
        start_time = time.time()
        try:
            while True:
                try:
                    # Wait for events with timeout to allow heartbeat
                    event = await asyncio.wait_for(
                        event_queue.get(), timeout=0.5  # Check every 500ms
                    )

                    # Process progress event
                    if isinstance(event, ProgressEvent):
                        # Calculate percentage
                        progress_percent = (
                            (event.current / event.total * 100) if event.total > 0 else 0
                        )

                        # Build progress message with real event data
                        message = {
                            "type": "progress",
                            "task_id": task_id,
                            "status": "indexing",
                            "progress_percent": round(progress_percent, 1),
                            "current_operation": event.message,
                            "stats": {
                                "files_processed": event.current,
                                "files_total": event.total,
                                "files_skipped": event.files_skipped,
                                "chunks_created": event.chunks_created,
                                "embeddings_generated": event.embeddings_generated,
                                "errors": event.errors,
                            },
                            "timing": {
                                "elapsed_seconds": int(time.time() - start_time),
                                "estimated_remaining": (
                                    int((event.total - event.current) * 0.1)
                                    if event.total > event.current
                                    else 0
                                ),
                                "files_per_second": round(
                                    event.current / max(time.time() - start_time, 1), 2
                                ),
                            },
                            "timestamp": utc_now().isoformat(),
                        }

                        await websocket.send_json(message)

                        # Verify if completed (when current == total)
                        if event.current >= event.total and event.total > 0:
                            await websocket.send_json(
                                {
                                    "type": "complete",
                                    "task_id": task_id,
                                    "message": "Indexing completed successfully",
                                    "final_stats": {
                                        "total_files": event.total,
                                        "total_chunks": event.chunks_created,
                                        "total_embeddings": event.embeddings_generated,
                                        "duration_seconds": int(time.time() - start_time),
                                        "errors": event.errors,
                                        "files_skipped": event.files_skipped,
                                    },
                                    "timestamp": utc_now().isoformat(),
                                }
                            )
                            break

                        # Update activity timestamp
                        if task_id in active_connections:
                            active_connections[task_id]["last_activity"] = time.time()

                except asyncio.TimeoutError:
                    # Normal timeout, continue the loop
                    # This allows the heartbeat to work
                    pass

        except Exception as e:
            logger.error("Progress loop error", task_id=task_id, error=str(e))
            await _send_error_message(task_id, f"Progress error: {str(e)}")

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected", task_id=task_id)
        pass
    except Exception as e:
        logger.error("WebSocket error", task_id=task_id, error=str(e))
        await _send_error_message(task_id, f"WebSocket error: {str(e)}")
    finally:
        # Unsubscribe from EventBus
        if 'unsubscribe' in locals() and unsubscribe:
            unsubscribe()

        # Cancel heartbeat task
        if 'heartbeat_task' in locals():
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

        # Clean up connection
        if task_id in active_connections:
            del active_connections[task_id]
            logger.info("WebSocket connection cleaned up", task_id=task_id)


def get_active_tasks() -> list[str]:
    """Get list of tasks with connected clients."""
    return list(active_connections.keys())


def get_connection_stats() -> Dict[str, Dict[str, Any]]:
    """Get active connection statistics."""
    stats = {}
    current_time = time.time()

    for task_id, connection_info in active_connections.items():
        stats[task_id] = {
            "connected_duration": current_time - connection_info["last_activity"],
            "last_ping_ago": current_time - connection_info["last_ping"],
            "last_activity_ago": current_time - connection_info["last_activity"],
            "connected_at": connection_info["connected_at"].isoformat(),
        }

    return stats


async def _heartbeat_handler(task_id: str):
    """Handles the heartbeat for a specific connection.

    Features:
    - Sends ping every 30 seconds when no activity
    - Automatically detects lost connections
    - Cleans up connections that don't respond
    """
    while task_id in active_connections:
        try:
            current_time = time.time()
            connection_info = active_connections[task_id]
            websocket = connection_info["websocket"]

            time_since_activity = current_time - connection_info["last_activity"]
            time_since_ping = current_time - connection_info["last_ping"]

            # If there is no recent activity, send heartbeat
            if time_since_activity > HEARTBEAT_INTERVAL and time_since_ping > HEARTBEAT_INTERVAL:
                try:
                    await websocket.send_json(
                        {
                            "type": "heartbeat",
                            "task_id": task_id,
                            "message": "Connection alive",
                            "timestamp": utc_now().isoformat(),
                            "connection_duration": int(time_since_activity),
                        }
                    )

                    connection_info["last_ping"] = current_time
                    logger.debug("Heartbeat sent", task_id=task_id)

                except Exception as e:
                    logger.warning("Heartbeat failed", task_id=task_id, error=str(e))
                    # Connection lost, mark for cleanup
                    break

            # Verify connection timeout
            if time_since_activity > CONNECTION_TIMEOUT:
                logger.info(
                    "Connection timeout", task_id=task_id, timeout_seconds=CONNECTION_TIMEOUT
                )
                await _send_error_message(task_id, "Connection timeout")
                await websocket.close(code=1011, reason="timeout")
                break

            # Wait before the next check
            await asyncio.sleep(10)  # Check every 10 seconds

        except asyncio.CancelledError:
            logger.debug("Heartbeat cancelled", task_id=task_id)
            break
        except Exception as e:
            logger.error("Heartbeat error", task_id=task_id, error=str(e))
            break

    # Clean up connection if we exit the loop
    if task_id in active_connections:
        del active_connections[task_id]
        logger.info("Heartbeat cleanup completed", task_id=task_id)


async def _send_error_message(task_id: str, error_message: str):
    """Sends an error message to a specific client if connected."""
    if task_id not in active_connections:
        return

    connection_info = active_connections[task_id]
    websocket = connection_info["websocket"]

    try:
        await websocket.send_json(
            {
                "type": "error",
                "task_id": task_id,
                "message": error_message,
                "timestamp": utc_now().isoformat(),
            }
        )
    except Exception:
        # If the send fails, the connection is lost
        pass
