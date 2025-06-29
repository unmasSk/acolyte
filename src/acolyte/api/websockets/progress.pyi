from fastapi import APIRouter, WebSocket
from typing import Dict, TypedDict, Any
from datetime import datetime
import asyncio
from acolyte.core.events import ProgressEvent

router: APIRouter

MAX_CONNECTIONS: int
HEARTBEAT_INTERVAL: int
CONNECTION_TIMEOUT: int

class ConnectionInfo(TypedDict):
    websocket: WebSocket
    last_ping: float
    last_activity: float
    connected_at: datetime
    event_queue: asyncio.Queue[ProgressEvent]

active_connections: Dict[str, ConnectionInfo]

async def websocket_progress(websocket: WebSocket, task_id: str) -> None: ...
def get_active_tasks() -> list[str]: ...
def get_connection_stats() -> Dict[str, Dict[str, Any]]: ...
async def _heartbeat_handler(task_id: str) -> None: ...
async def _send_error_message(task_id: str, error_message: str) -> None: ...
