"""
Fatigue Monitor - Calculates system fatigue based on real Git metrics.

Uses GitMetadata fields to compute a fatigue score that indicates
when the system needs deep analysis and optimization.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from acolyte.core.database import DatabaseManager
from acolyte.core.secure_config import Settings
from acolyte.models.common.metadata import GitMetadata
from acolyte.rag.enrichment import EnrichmentService
from acolyte.rag.retrieval.hybrid_search import HybridSearch

class FatigueLevel:
    """Fatigue level constants and helpers."""

    HEALTHY: float
    MODERATE: float
    HIGH: float
    CRITICAL: float

    @staticmethod
    def get_description(level: float) -> str: ...

class FatigueMonitor:
    """
    Monitors system fatigue based on real Git activity metrics.

    Fatigue represents:
    - Code churn and instability
    - Recent activity levels
    - Architectural changes
    - Time since last optimization
    """

    db: DatabaseManager
    config: Settings
    enrichment_service: EnrichmentService
    search: HybridSearch
    threshold: float
    emergency_threshold: float

    def __init__(self, weaviate_client: Any) -> None: ...
    async def calculate_fatigue(self) -> Dict[str, Any]: ...
    async def _calculate_fatigue_components(self) -> Dict[str, float]: ...
    async def _calculate_time_factor(self) -> float: ...
    async def _calculate_file_instability(self) -> float: ...
    async def _calculate_recent_activity(self) -> float: ...
    async def _calculate_code_volatility(self) -> float: ...
    async def _calculate_architectural_changes(self) -> float: ...
    async def _get_file_git_metadata(self, file_path: str) -> Optional[GitMetadata]: ...
    def _generate_fatigue_explanation(self, components: Dict[str, float], total: float) -> str: ...
    async def _check_fatigue_triggers(self) -> List[Dict[str, Any]]: ...
    async def reduce_fatigue(self, factor: float = 0.3) -> None: ...
    async def get_fatigue_history(self, days: int = 7) -> List[Tuple[datetime, float]]: ...
