from typing import Optional, Dict, List
from acolyte.models.semantic_types import TaskDetection
from acolyte.models.task_checkpoint import TaskCheckpoint
from acolyte.core.tracing import MetricsCollector

class TaskDetector:
    language: str
    patterns: Dict[str, Dict[str, List[str]]]
    confidence_threshold: float
    metrics: MetricsCollector

    def __init__(self) -> None: ...
    async def detect_task_context(
        self,
        message: str,
        current_task: Optional[TaskCheckpoint] = ...,
        recent_messages: Optional[List[str]] = ...,
    ) -> TaskDetection: ...
    def _check_new_task_patterns(self, message: str, lang: str) -> Optional[TaskDetection]: ...
    def _check_continuation_patterns(
        self, message: str, lang: str, current_task: TaskCheckpoint
    ) -> Optional[TaskDetection]: ...
    def _calculate_context_similarity(
        self, message: str, current_task: TaskCheckpoint, recent_messages: List[str]
    ) -> float: ...
    def _get_default_patterns(self) -> Dict[str, Dict[str, List[str]]]: ...
