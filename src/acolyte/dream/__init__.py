"""
Dream System - Deep Search for your code.

Like Deep Search in modern AIs but specialized for your project.
Analyzes patterns, detects bugs, and optimizes memory during "sleep" cycles.
"""

from typing import TYPE_CHECKING, Optional, Any

# Version info
__version__ = "0.1.0"
__all__ = [
    "DreamOrchestrator",
    "DreamState",
    "FatigueLevel",
    "DreamTrigger",
    "create_dream_orchestrator",
]

# Lazy imports to avoid circular dependencies
from .orchestrator import DreamTrigger  # Import from orchestrator where it's defined

if TYPE_CHECKING:
    from .orchestrator import DreamOrchestrator
    from .state_manager import DreamState
    from .fatigue_monitor import FatigueLevel
else:
    # Runtime imports
    def __getattr__(name):
        if name == "DreamOrchestrator":
            from .orchestrator import DreamOrchestrator

            return DreamOrchestrator
        elif name == "DreamState":
            from .state_manager import DreamState

            return DreamState
        elif name == "FatigueLevel":
            from .fatigue_monitor import FatigueLevel

            return FatigueLevel
        elif name == "DreamTrigger":
            # Already imported above
            return DreamTrigger
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def create_dream_orchestrator(weaviate_client: Optional[Any] = None) -> 'DreamOrchestrator':
    """
    Factory function to create a DreamOrchestrator instance.

    Args:
        weaviate_client: Optional Weaviate client for search functionality.
                       If not provided, search features will be limited.

    Returns:
        Configured DreamOrchestrator instance
    """
    from .orchestrator import DreamOrchestrator

    return DreamOrchestrator(weaviate_client=weaviate_client)
