"""
Dream State Manager - Manages Dream system states and transitions.

Handles the state machine for Dream analysis cycles and persists
state information to the database.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from enum import Enum
import json
import asyncio

from acolyte.core.logging import logger
from acolyte.core.database import get_db_manager, FetchType, DatabaseManager
from acolyte.core.exceptions import ValidationError, DatabaseError
from acolyte.core.secure_config import Settings
from acolyte.core.utils.datetime_utils import utc_now, utc_now_iso, parse_iso_datetime


class DreamState(str, Enum):
    """
    Dream system states.

    Represents different phases of the analysis cycle.
    """

    MONITORING = "MONITORING"  # Normal operation, watching for triggers
    DROWSY = "DROWSY"  # Preparing for analysis, saving state
    DREAMING = "DREAMING"  # Initial exploration with extended context
    REM = "REM"  # Deep analysis of specific areas
    DEEP_SLEEP = "DEEP_SLEEP"  # Consolidation and pattern synthesis
    WAKING = "WAKING"  # Preparing results and insights


class DreamStateManager:
    """
    Manages Dream system state transitions and persistence.

    Ensures valid state transitions and tracks analysis progress.
    """

    # Valid state transitions
    VALID_TRANSITIONS: Dict[DreamState, List[DreamState]] = {
        DreamState.MONITORING: [DreamState.DROWSY],
        DreamState.DROWSY: [DreamState.DREAMING, DreamState.MONITORING],  # Can abort
        DreamState.DREAMING: [DreamState.REM, DreamState.MONITORING],  # Can abort
        DreamState.REM: [DreamState.DEEP_SLEEP, DreamState.MONITORING],  # Can abort
        DreamState.DEEP_SLEEP: [DreamState.WAKING, DreamState.MONITORING],  # Can abort
        DreamState.WAKING: [DreamState.MONITORING],  # Must complete
    }

    # Type annotations for instance attributes
    db: DatabaseManager
    _current_state: Optional[DreamState]
    _state_start_time: Optional[datetime]
    _session_id: Optional[str]
    settings: Settings
    cycle_duration_minutes: float
    state_proportions: Dict[DreamState, float]
    state_durations: Dict[DreamState, float]
    _state_lock: asyncio.Lock

    def __init__(self) -> None:
        """Initialize state manager."""
        self.db = get_db_manager()
        self._current_state = None
        self._state_start_time = None
        self._session_id = None
        self._state_lock = asyncio.Lock()

        # Load configuration
        self.settings = Settings()
        self.cycle_duration_minutes: float = self.settings.get("dream.cycle_duration_minutes", 5)

        # Define state duration proportions (must sum to 1.0)
        self.state_proportions: Dict[DreamState, float] = {
            DreamState.DROWSY: 0.1,  # 10% of cycle
            DreamState.DREAMING: 0.3,  # 30% of cycle
            DreamState.REM: 0.4,  # 40% of cycle
            DreamState.DEEP_SLEEP: 0.1,  # 10% of cycle
            DreamState.WAKING: 0.1,  # 10% of cycle
        }

        # Calculate actual durations based on configured cycle duration
        self.state_durations: Dict[DreamState, float] = {}
        self._calculate_state_durations()

        logger.info(
            "Dream state manager initialized",
            module="dream",
            cycle_duration_minutes=self.cycle_duration_minutes,
        )

    async def get_current_state(self) -> DreamState:
        """
        Get current Dream state from database.

        Returns:
            Current DreamState enum value
        """
        try:
            async with self._state_lock:
                if self._current_state is None:
                    # Load from database
                    result = await self.db.execute_async(
                        "SELECT metrics FROM dream_state WHERE id = 1", (), FetchType.ONE
                    )

                    if result.data and isinstance(result.data, dict) and "metrics" in result.data:
                        metrics: Dict[str, Any] = json.loads(result.data["metrics"])
                        state_str: str = metrics.get("current_state", DreamState.MONITORING.value)
                        self._current_state = DreamState(state_str)
                    else:
                        self._current_state = DreamState.MONITORING

                return self._current_state

        except Exception as e:
            logger.error("Failed to get current state", error=str(e))
            return DreamState.MONITORING

    async def transition_to(self, new_state: DreamState) -> None:
        """
        Transition to a new state.

        Args:
            new_state: Target state to transition to

        Raises:
            ValidationError: If transition is invalid
            DatabaseError: If database update fails
        """
        async with self._state_lock:
            # Get current state without acquiring lock again
            current = self._current_state
            if current is None:
                # If not cached, load it (lock already held)
                result = await self.db.execute_async(
                    "SELECT metrics FROM dream_state WHERE id = 1", (), FetchType.ONE
                )
                if result.data and isinstance(result.data, dict) and "metrics" in result.data:
                    metrics = json.loads(result.data["metrics"])
                    current = DreamState(metrics.get("current_state", DreamState.MONITORING.value))
                else:
                    current = DreamState.MONITORING
                self._current_state = current

            # Check if transition is valid
            if new_state not in self.VALID_TRANSITIONS.get(current, []):
                raise ValidationError(
                    "Invalid state transition: {} -> {}".format(current.value, new_state.value)
                )

            try:
                # Update state in memory
                old_state = current
                self._current_state = new_state
                self._state_start_time = utc_now()

                # Persist to database
                await self._persist_state_change(old_state, new_state)

                # Log transition
                logger.info(
                    "Dream state transition",
                    from_state=old_state.value,
                    to_state=new_state.value,
                    session_id=self._session_id,
                )

            except Exception as e:
                # Rollback on error
                self._current_state = current
                logger.error("Failed to transition state", error=str(e))
                raise DatabaseError("State transition failed: {}".format(str(e)))

    async def _persist_state_change(self, old_state: DreamState, new_state: DreamState) -> None:
        """
        Persist state change to database.

        Args:
            old_state: Previous state
            new_state: New state
        """
        # Get current metrics
        result = await self.db.execute_async(
            "SELECT metrics FROM dream_state WHERE id = 1", (), FetchType.ONE
        )

        metrics: Dict[str, Any] = {}
        if result.data and isinstance(result.data, dict) and "metrics" in result.data:
            metrics = json.loads(result.data["metrics"])

        # Update metrics
        metrics.update(
            {
                "current_state": new_state.value,
                "last_state": old_state.value,
                "state_changed_at": utc_now_iso(),
                "session_id": self._session_id,
                "transitions_history": metrics.get("transitions_history", []),
            }
        )

        # Add to history (keep last 10)
        metrics["transitions_history"].append(
            {
                "from": old_state.value,
                "to": new_state.value,
                "timestamp": utc_now_iso(),
            }
        )
        metrics["transitions_history"] = metrics["transitions_history"][-10:]

        # Update database
        await self.db.execute_async(
            """
            UPDATE dream_state 
            SET metrics = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
            """,
            (json.dumps(metrics),),
        )

    async def set_session_id(self, session_id: str) -> None:
        """
        Set current Dream session ID.

        Args:
            session_id: Dream analysis session ID
        """
        async with self._state_lock:
            self._session_id = session_id

            # Also persist to metrics
            result = await self.db.execute_async(
                "SELECT metrics FROM dream_state WHERE id = 1", (), FetchType.ONE
            )

            metrics: Dict[str, Any] = {}
            if result.data and isinstance(result.data, dict) and "metrics" in result.data:
                metrics = json.loads(result.data["metrics"])

            metrics["current_session_id"] = session_id

            await self.db.execute_async(
                "UPDATE dream_state SET metrics = ? WHERE id = 1", (json.dumps(metrics),)
            )

    def get_session_id(self) -> Optional[str]:
        """Get current Dream session ID."""
        # No need for lock on simple read of immutable string
        return self._session_id

    async def get_state_duration(self) -> timedelta:
        """
        Get duration in current state.

        Returns:
            Time spent in current state
        """
        if self._state_start_time:
            return utc_now() - self._state_start_time

        # Load from database
        result = await self.db.execute_async(
            "SELECT metrics FROM dream_state WHERE id = 1", (), FetchType.ONE
        )

        if result.data and isinstance(result.data, dict) and "metrics" in result.data:
            metrics = json.loads(result.data["metrics"])
            changed_at = metrics.get("state_changed_at")
            if changed_at:
                changed_time = parse_iso_datetime(changed_at)
                # utc_now() already returns UTC time
                return utc_now() - changed_time

        return timedelta(0)

    def _calculate_state_durations(self) -> None:
        """
        Calculate state durations based on configured cycle duration.

        Uses proportions to distribute time across states.
        """
        self.state_durations = {
            state: self.cycle_duration_minutes * proportion
            for state, proportion in self.state_proportions.items()
        }

        # Validate proportions sum to 1.0
        total_proportion = sum(self.state_proportions.values())
        if abs(total_proportion - 1.0) > 0.001:
            logger.warning(
                "State proportions do not sum to 1.0",
                total=total_proportion,
                proportions=self.state_proportions,
            )

    def _reload_configuration(self) -> None:
        """
        Reload configuration and recalculate durations.

        Useful if configuration changes during runtime.
        """
        new_duration: float = self.settings.get("dream.cycle_duration_minutes", 5)
        if new_duration != self.cycle_duration_minutes:
            self.cycle_duration_minutes = new_duration
            self._calculate_state_durations()
            logger.info("Dream cycle duration updated", new_duration_minutes=new_duration)

    async def get_estimated_completion(self) -> Optional[datetime]:
        """
        Get estimated completion time for current analysis.

        Returns:
            Estimated completion datetime or None
        """
        current = await self.get_current_state()

        if current == DreamState.MONITORING:
            return None

        # Calculate remaining time
        remaining_minutes = 0.0

        # Add remaining time for current state
        if current in self.state_durations:
            duration = await self.get_state_duration()
            elapsed_minutes = duration.total_seconds() / 60
            remaining_in_current = max(0, self.state_durations[current] - elapsed_minutes)
            remaining_minutes += remaining_in_current

        # Add time for remaining states
        remaining_states: List[DreamState] = []
        if current == DreamState.DROWSY:
            remaining_states = [
                DreamState.DREAMING,
                DreamState.REM,
                DreamState.DEEP_SLEEP,
                DreamState.WAKING,
            ]
        elif current == DreamState.DREAMING:
            remaining_states = [DreamState.REM, DreamState.DEEP_SLEEP, DreamState.WAKING]
        elif current == DreamState.REM:
            remaining_states = [DreamState.DEEP_SLEEP, DreamState.WAKING]
        elif current == DreamState.DEEP_SLEEP:
            remaining_states = [DreamState.WAKING]

        for state in remaining_states:
            remaining_minutes += self.state_durations.get(state, 0)

        # Calculate estimated completion
        if remaining_minutes > 0:
            return utc_now() + timedelta(minutes=remaining_minutes)

        return None

    async def get_last_optimization_time(self) -> Optional[datetime]:
        """
        Get timestamp of last completed optimization.

        Returns:
            Last optimization datetime or None
        """
        try:
            result = await self.db.execute_async(
                "SELECT last_optimization FROM dream_state WHERE id = 1", (), FetchType.ONE
            )

            if result.data and isinstance(result.data, dict) and "last_optimization" in result.data:
                last_opt = parse_iso_datetime(result.data["last_optimization"])
                # utc_now() already returns UTC time
                return last_opt

            return None

        except Exception as e:
            logger.error("Failed to get last optimization time", error=str(e))
            return None

    async def record_phase_metrics(self, phase: str, phase_metrics: Dict[str, Any]) -> None:
        """
        Record metrics for a specific phase.

        Args:
            phase: Phase name (e.g., "exploration", "analysis")
            phase_metrics: Metrics to record
        """
        try:
            # Get current metrics
            result = await self.db.execute_async(
                "SELECT metrics FROM dream_state WHERE id = 1", (), FetchType.ONE
            )

            current_metrics: Dict[str, Any] = {}
            if result.data and isinstance(result.data, dict) and "metrics" in result.data:
                current_metrics = json.loads(result.data["metrics"])

            # Update phase metrics
            if "phase_metrics" not in current_metrics:
                current_metrics["phase_metrics"] = {}

            current_metrics["phase_metrics"][phase] = {
                "timestamp": utc_now_iso(),
                "data": phase_metrics,
            }

            # Save back
            await self.db.execute_async(
                "UPDATE dream_state SET metrics = ? WHERE id = 1", (json.dumps(current_metrics),)
            )

        except Exception as e:
            logger.error("Failed to record phase metrics", error=str(e))

    async def abort_analysis(self, reason: str) -> None:
        """
        Abort current analysis and return to monitoring.

        Args:
            reason: Reason for aborting
        """
        async with self._state_lock:
            if self._current_state == DreamState.MONITORING:
                return  # Already monitoring

            current = self._current_state or DreamState.MONITORING

            logger.warning(
                "Aborting Dream analysis",
                current_state=current.value,
                reason=reason,
                session_id=self._session_id,
            )

            # Record abort in metrics (will use own lock)
            # Release lock temporarily to avoid nested locks

        # Record metrics outside lock to avoid potential deadlock
        await self.record_phase_metrics("abort", {"state": current.value, "reason": reason})

        async with self._state_lock:
            # Force transition to monitoring
            self._current_state = DreamState.MONITORING
            await self._persist_state_change(current, DreamState.MONITORING)

            # Clear session
            self._session_id = None
            self._state_start_time = None

    async def get_state_info(self) -> Dict[str, Any]:
        """
        Get comprehensive state information.

        Returns:
            Dict with current state, duration, session info, etc.
        """
        current = await self.get_current_state()
        duration = await self.get_state_duration()

        info: Dict[str, Any] = {
            "current_state": current.value,
            "state_duration_seconds": int(duration.total_seconds()),
            "session_id": self._session_id,
            "can_abort": current != DreamState.WAKING,  # Can't abort during waking
        }

        # Add completion estimate if analyzing
        if current != DreamState.MONITORING:
            completion = await self.get_estimated_completion()
            if completion:
                info["estimated_completion"] = completion.isoformat()

        # Add last optimization info
        last_opt = await self.get_last_optimization_time()
        if last_opt:
            info["last_optimization"] = last_opt.isoformat()
            info["hours_since_optimization"] = int((utc_now() - last_opt).total_seconds() / 3600)

        # Add configured cycle duration and state durations
        info["cycle_duration_minutes"] = self.cycle_duration_minutes
        info["state_durations"] = {
            state.value: round(duration, 2) for state, duration in self.state_durations.items()
        }

        return info
