"""
Task Service - Task and Decision Management.

Implements the Task > Session > Message hierarchy (Decision #6).
Tasks group multiple related sessions.
"""

from acolyte.core.logging import logger
from acolyte.core.tracing import MetricsCollector
from acolyte.core.database import get_db_manager, FetchType
from acolyte.core.exceptions import DatabaseError, NotFoundError, ValidationError
from acolyte.core.id_generator import generate_id
from acolyte.core.utils.file_types import FileTypeDetector
from acolyte.models.task_checkpoint import TaskCheckpoint, TaskType, TaskStatus
from acolyte.models.technical_decision import TechnicalDecision, DecisionType
from typing import List, Optional, Dict, Any, cast
from datetime import datetime
from acolyte.core.utils.datetime_utils import utc_now, utc_now_iso
import json
import re


class TaskService:
    """
    Manages tasks that group multiple sessions.

    IMPORTANT:
    - A task can last days/weeks (e.g., "refactor auth")
    - Multiple sessions belong to one task
    - Detects and records technical decisions (Decision #13)
    """

    def __init__(self):
        self.metrics = MetricsCollector()
        self.db = get_db_manager()  # Get singleton DatabaseManager instance
        logger.info("TaskService initialized")

    async def create_task(
        self, title: str, description: str, task_type: TaskType, initial_session_id: str
    ) -> str:
        """
        Create new task.

        NOTE: Semantic detects WHEN to create, this method creates it
        task_type must be converted to .upper() for DB

        Args:
            title: Task title
            description: Detailed description
            task_type: Task type (enum)
            initial_session_id: Session that initiates the task

        Returns:
            ID of created task

        Raises:
            DatabaseError: If DB creation fails
        """
        try:
            task_id = generate_id()

            # Create the task
            await self.db.execute_async(
                """
                    INSERT INTO tasks (
                        id, title, description, task_type, status,
                        user_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                (
                    task_id,
                    title,
                    description,
                    task_type.value.upper(),  # UPPERCASE for DB
                    TaskStatus.IN_PROGRESS.value.upper(),
                    "default",  # Mono-user
                    utc_now_iso(),
                    utc_now_iso(),
                ),
                FetchType.NONE,
            )

            # Associate the initial session
            await self.db.execute_async(
                """
                    INSERT INTO task_sessions (task_id, session_id)
                    VALUES (?, ?)
                    """,
                (task_id, initial_session_id),
                FetchType.NONE,
            )

            self.metrics.increment("services.task.tasks_created")
            self.metrics.increment(f"services.task.task_type.{task_type.value}")

            logger.info("Created new task", task_id=task_id, title=title, type=task_type.value)

            return task_id

        except Exception as e:
            logger.error("Failed to create task", error=str(e))
            raise DatabaseError(f"Failed to create task: {str(e)}") from e

    async def associate_session_to_task(self, task_id: str, session_id: str):
        """
        Associate session to existing task.

        IMPORTANT: Many-to-many relationship in task_sessions

        Args:
            task_id: Task ID
            session_id: Session ID to associate

        Raises:
            NotFoundError: If task doesn't exist
            DatabaseError: If association fails
        """
        try:
            # Verify task exists
            result = await self.db.execute_async(
                "SELECT id FROM tasks WHERE id = ?", (task_id,), FetchType.ONE
            )

            if not result.data:
                raise NotFoundError(f"Task {task_id} not found")

            # Verify association doesn't exist already
            result = await self.db.execute_async(
                """
                    SELECT 1 FROM task_sessions 
                    WHERE task_id = ? AND session_id = ?
                    """,
                (task_id, session_id),
                FetchType.ONE,
            )

            if not result.data:
                # Create association
                await self.db.execute_async(
                    """
                    INSERT INTO task_sessions (task_id, session_id)
                    VALUES (?, ?)
                    """,
                    (task_id, session_id),
                    FetchType.NONE,
                )

                # Update task timestamp
                await self.db.execute_async(
                    """
                    UPDATE tasks 
                    SET updated_at = ?
                    WHERE id = ?
                    """,
                    (utc_now_iso(), task_id),
                    FetchType.NONE,
                )

                self.metrics.increment("services.task.sessions_associated")
                logger.info("Associated session to task", task_id=task_id, session_id=session_id)
            else:
                # Association already exists, normal operation
                pass

        except NotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to associate session", error=str(e))
            raise DatabaseError(f"Failed to associate session: {str(e)}") from e

    async def get_task_full_context(self, task_id: str) -> Dict[str, Any]:
        """
        Retrieve COMPLETE task context.

        Includes:
        - All sessions
        - All technical decisions
        - Related code (via RAG)
        - Activity timeline

        Args:
            task_id: Task ID

        Returns:
            Dict with task, sessions, decisions, statistics, timeline

        Raises:
            NotFoundError: If task doesn't exist
            DatabaseError: If retrieval fails
        """
        try:
            # Get the task
            result = await self.db.execute_async(
                "SELECT * FROM tasks WHERE id = ?", (task_id,), FetchType.ONE
            )

            if not result.data:
                raise NotFoundError(f"Task {task_id} not found")

            task = cast(Dict[str, Any], result.data)

            # Get all associated sessions
            result = await self.db.execute_async(
                """
                    SELECT c.* FROM conversations c
                    JOIN task_sessions ts ON c.id = ts.session_id
                    WHERE ts.task_id = ?
                    ORDER BY c.created_at ASC
                    """,
                (task_id,),
                FetchType.ALL,
            )

            sessions = cast(List[Dict[str, Any]], result.data if result.data else [])

            # Get technical decisions
            result = await self.db.execute_async(
                """
                    SELECT * FROM technical_decisions
                    WHERE task_id = ?
                    ORDER BY created_at ASC
                    """,
                (task_id,),
                FetchType.ALL,
            )

            decisions = cast(List[Dict[str, Any]], result.data if result.data else [])

            # Get activity summary
            result = await self.db.execute_async(
                """
                    SELECT 
                        COUNT(DISTINCT ts.session_id) as session_count,
                        COUNT(DISTINCT td.id) as decision_count,
                        MIN(c.created_at) as first_activity,
                        MAX(c.updated_at) as last_activity,
                        SUM(c.total_tokens) as total_tokens_used
                    FROM tasks t
                    LEFT JOIN task_sessions ts ON t.id = ts.task_id
                    LEFT JOIN conversations c ON ts.session_id = c.id
                    LEFT JOIN technical_decisions td ON t.id = td.task_id
                    WHERE t.id = ?
                    """,
                (task_id,),
                FetchType.ONE,
            )

            stats = cast(Dict[str, Any], result.data if result.data else {})

            # Build complete context
            context = {
                "task": task,
                "sessions": sessions,
                "decisions": decisions,
                "statistics": stats,
                "timeline": self._build_timeline(sessions, decisions),
                "key_files": self._extract_key_files(sessions),
                "summary": self._generate_task_summary(task, sessions, decisions),
            }

            self.metrics.gauge("services.task.task_context_size", len(sessions))
            self.metrics.gauge("services.task.task_decisions", len(decisions))

            return context

        except NotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to get task context", error=str(e))
            raise DatabaseError(f"Failed to get task context: {str(e)}") from e

    async def save_technical_decision(self, decision: TechnicalDecision):
        """
        Save detected technical decision.

        The decision must have session_id and task_id already assigned.

        Args:
            decision: Technical decision with all fields completed

        Raises:
            DatabaseError: If DB save fails
            ValidationError: If required fields are missing
        """
        try:
            # Validate required IDs
            if not decision.session_id:
                raise ValidationError("Technical decision must have session_id")
            if not decision.task_id:
                raise ValidationError("Technical decision must have task_id")

            decision_id = generate_id()

            await self.db.execute_async(
                """
                    INSERT INTO technical_decisions (
                        id, task_id, session_id, decision_type,
                        title, description, rationale,
                        alternatives_considered, impact_level,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                (
                    decision_id,
                    decision.task_id,  # Use from object
                    decision.session_id,  # Use from object
                    decision.decision_type.upper(),  # UPPERCASE - decision_type is already string
                    decision.title,
                    decision.description,
                    decision.rationale,
                    (
                        json.dumps(decision.alternatives_considered)
                        if decision.alternatives_considered
                        else None
                    ),
                    decision.impact_level,
                    utc_now_iso(),
                ),
                FetchType.NONE,
            )

            self.metrics.increment("services.task.decisions_saved")
            self.metrics.increment(f"services.task.decision_type.{decision.decision_type}")
            self.metrics.gauge("services.task.decision_impact", decision.impact_level)

            logger.info(
                "Saved technical decision",
                decision_id=decision_id,
                type=decision.decision_type,
                impact=decision.impact_level,
            )

        except Exception as e:
            logger.error("Failed to save decision", error=str(e))
            raise DatabaseError(f"Failed to save decision: {str(e)}") from e

    async def find_active_task(self, user_id: str = "default") -> Optional[TaskCheckpoint]:
        """
        Find user's active task.

        NOTE: user_id always "default" (mono-user)
        Status must be IN_PROGRESS

        Args:
            user_id: User ID (always "default")

        Returns:
            Active TaskCheckpoint or None if none

        Note:
            Doesn't throw exceptions, returns None if fails
        """
        try:
            result = await self.db.execute_async(
                """
                    SELECT * FROM tasks
                    WHERE user_id = ? AND status = ?
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                (user_id, TaskStatus.IN_PROGRESS.value.upper()),
                FetchType.ONE,
            )

            if not result.data:
                return None

            row = cast(Dict[str, Any], result.data)

            # Convert to TaskCheckpoint
            task = TaskCheckpoint(
                id=row["id"],
                title=row["title"],
                description=row["description"],
                task_type=TaskType(row["task_type"].lower()),  # Convert from DB uppercase
                status=TaskStatus(row["status"].lower()),  # Convert from DB uppercase
                initial_context=row.get("description", ""),  # Use description as initial context
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
            logger.info("[UNTESTED PATH] Created TaskCheckpoint from DB row")

            self.metrics.increment("services.task.active_task_found")
            return task

        except Exception as e:
            logger.error("Failed to find active task", error=str(e))
            logger.info("[UNTESTED PATH] Exception in find_active_task")
            return None

    async def complete_task(self, task_id: str):
        """
        Mark a task as completed.

        Args:
            task_id: Task ID to complete

        Raises:
            DatabaseError: If update fails
        """
        try:
            await self.db.execute_async(
                """
                    UPDATE tasks
                    SET status = ?, updated_at = ?
                    WHERE id = ?
                    """,
                (TaskStatus.COMPLETED.value.upper(), utc_now_iso(), task_id),
                FetchType.NONE,
            )

            self.metrics.increment("services.task.tasks_completed")
            logger.info("Task completed", task_id=task_id)

        except Exception as e:
            logger.error("Failed to complete task", error=str(e))
            raise DatabaseError(f"Failed to complete task: {str(e)}") from e

    async def get_recent_decisions(self, task_id: str, limit: int = 5) -> List[TechnicalDecision]:
        """
        Get recent technical decisions from a task.

        IMPLEMENTS: Getting TechnicalDecision objects to use get_summary().

        Args:
            task_id: Task ID
            limit: Maximum number of decisions to return

        Returns:
            List of TechnicalDecision objects ordered by date (most recent first)

        Note:
            Doesn't throw exceptions, returns empty list if fails
        """
        try:
            result = await self.db.execute_async(
                """
                SELECT * FROM technical_decisions
                WHERE task_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (task_id, limit),
                FetchType.ALL,
            )

            if not result.data:
                return []

            # Convert DB rows to TechnicalDecision objects
            decisions = []
            typed_rows = cast(List[Dict[str, Any]], result.data)
            for row in typed_rows:
                try:
                    # Parse alternatives_considered if exists
                    alternatives = []
                    if row["alternatives_considered"]:
                        alternatives = json.loads(row["alternatives_considered"])

                    decision = TechnicalDecision(
                        id=row["id"],
                        task_id=row["task_id"],
                        session_id=row["session_id"],
                        decision_type=DecisionType(
                            row["decision_type"].lower()
                        ),  # Convert from DB uppercase
                        title=row["title"],
                        description=row["description"],
                        rationale=row["rationale"],
                        alternatives_considered=alternatives,
                        impact_level=int(row["impact_level"]),
                        created_at=(
                            datetime.fromisoformat(row["created_at"])
                            if row["created_at"]
                            else utc_now()
                        ),
                    )
                    decisions.append(decision)

                except Exception as e:
                    logger.warning(
                        "Could not parse decision",
                        decision_id=row.get('id', 'unknown'),
                        error=str(e),
                    )
                    continue

            return decisions

        except Exception as e:
            logger.error("Failed to get recent decisions", error=str(e))
            return []  # Don't fail, return empty

    def _build_timeline(self, sessions: List[Dict], decisions: List[Dict]) -> List[Dict[str, Any]]:
        """Build activity timeline."""
        timeline = []

        # Add sessions
        for session in sessions:
            timeline.append(
                {
                    "type": "session",
                    "timestamp": session["created_at"],
                    "title": f"Session: {session['id'][:8]}",
                    "summary": session.get("summary", "")[:100],
                }
            )

        # Add decisions
        for decision in decisions:
            timeline.append(
                {
                    "type": "decision",
                    "timestamp": decision["created_at"],
                    "title": decision["title"],
                    "impact": decision["impact_level"],
                }
            )

        # Sort by timestamp
        timeline.sort(key=lambda x: x["timestamp"])

        return timeline

    def _extract_key_files(self, sessions: List[Dict]) -> List[str]:
        """Extract key files mentioned in sessions."""
        files = set()

        # Get all supported extensions dynamically
        supported_extensions = FileTypeDetector.get_all_supported_extensions()
        # Remove the dot prefix and join with |
        extensions_pattern = "|".join(ext[1:] for ext in supported_extensions)

        for session in sessions:
            summary = session.get("summary", "")
            # Search for file patterns in summary with all extensions
            file_pattern = rf"\b[\w\-\.]+\.(?:{extensions_pattern})\b"
            found_files = re.findall(file_pattern, summary, re.IGNORECASE)
            files.update(found_files)

        return sorted(list(files))[:10]  # Top 10 files

    def _generate_task_summary(
        self, task: Dict, sessions: List[Dict], decisions: List[Dict]
    ) -> str:
        """Generate executive summary of the task."""
        decision_types = [d["decision_type"] for d in decisions]
        unique_types = set(decision_types)

        summary = (
            f"Task '{task['title']}' with {len(sessions)} sessions "
            f"and {len(decisions)} technical decisions. "
        )

        if unique_types:
            summary += f"Decision types: {', '.join(unique_types)}. "

        if task["status"] == "COMPLETED":
            summary += "Status: Completed."
        else:
            summary += "Status: In progress."

        return summary
