"""
Dream State Manager - Type stubs for state management.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from enum import Enum

from acolyte.core.database import DatabaseManager
from acolyte.core.secure_config import Settings

class DreamState(str, Enum):
    MONITORING: str
    DROWSY: str
    DREAMING: str
    REM: str
    DEEP_SLEEP: str
    WAKING: str

class DreamStateManager:
    VALID_TRANSITIONS: Dict[DreamState, List[DreamState]]

    db: DatabaseManager
    settings: Settings
    cycle_duration_minutes: float
    state_proportions: Dict[DreamState, float]
    state_durations: Dict[DreamState, float]

    def __init__(self) -> None: ...
    async def get_current_state(self) -> DreamState: ...
    async def transition_to(self, new_state: DreamState) -> None: ...
    async def set_session_id(self, session_id: str) -> None: ...
    def get_session_id(self) -> Optional[str]: ...
    async def get_state_duration(self) -> timedelta: ...
    async def get_estimated_completion(self) -> Optional[datetime]: ...
    async def get_last_optimization_time(self) -> Optional[datetime]: ...
    async def record_phase_metrics(self, phase: str, phase_metrics: Dict[str, Any]) -> None: ...
    async def abort_analysis(self, reason: str) -> None: ...
    async def get_state_info(self) -> Dict[str, Any]: ...
