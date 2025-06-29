"""
WebSocket module - Only for long-running operations progress.
NO includes streaming of logs (over-engineering for mono-user).
"""

# Export directly the progress router
from acolyte.api.websockets.progress import router

# Also export useful functions
from acolyte.api.websockets.progress import get_active_tasks, get_connection_stats

__all__ = ["router", "get_active_tasks", "get_connection_stats"]
