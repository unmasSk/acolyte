"""
Task start and continuation detector.

Automatically detects when the user starts a new task or continues with an existing one. Supports Spanish and English.
"""

import re
from typing import Optional, Dict, List

from acolyte.core.logging import logger
from acolyte.core.utils.datetime_utils import utc_now
from acolyte.core.secure_config import Settings
from acolyte.core.tracing import MetricsCollector
from acolyte.models.semantic_types import TaskDetection
from acolyte.models.task_checkpoint import TaskCheckpoint
from acolyte.semantic.utils import detect_language


class TaskDetector:
    """Detects task context in conversations."""

    def __init__(self):
        # Load language configuration
        settings = Settings()
        self.language = settings.get("semantic.language", "es")
        self.patterns = settings.get("semantic.task_detection.patterns", {})
        self.confidence_threshold = settings.get(
            "semantic.task_detection.confidence_threshold", 0.6
        )

        # Default patterns if not in config
        if not self.patterns:
            self.patterns = self._get_default_patterns()

        # Metrics
        self.metrics = MetricsCollector()
        logger.info(
            "TaskDetector initialized",
            language=self.language,
            confidence_threshold=self.confidence_threshold,
        )

    async def detect_task_context(
        self,
        message: str,
        current_task: Optional[TaskCheckpoint] = None,
        recent_messages: Optional[List[str]] = None,
    ) -> TaskDetection:
        """
        Detects if it is a new task or a continuation.

        Args:
            message: Current user message
            current_task: Active task if exists
            recent_messages: Recent messages for context

        Returns:
            TaskDetection with the result of the analysis
        """
        start_time = utc_now()

        message_lower = message.lower()

        # 1. Detect message language
        detected_lang = detect_language(message, self.language)

        # 2. Search for new task patterns
        new_task_detection = self._check_new_task_patterns(message_lower, detected_lang)
        if new_task_detection:
            self.metrics.increment("semantic.task_detector.new_tasks_detected")
            self.metrics.record("semantic.task_detector.confidence", new_task_detection.confidence)
            result = new_task_detection

        # 3. Search for explicit continuation
        elif current_task:
            continuation = self._check_continuation_patterns(
                message_lower, detected_lang, current_task
            )
            if continuation:
                self.metrics.increment("semantic.task_detector.continuations_detected")
                self.metrics.record("semantic.task_detector.confidence", continuation.confidence)
                result = continuation
            else:
                # 4. Analyze similar context
                if recent_messages:
                    similarity_score = self._calculate_context_similarity(
                        message, current_task, recent_messages
                    )
                    if similarity_score > self.confidence_threshold:
                        self.metrics.increment("semantic.task_detector.similarity_continuations")
                        result = TaskDetection(
                            is_new_task=False,
                            task_title=None,
                            continues_task=current_task.id,
                            confidence=similarity_score,
                        )
                    else:
                        result = TaskDetection(is_new_task=False, task_title=None, confidence=0.0)
                else:
                    result = TaskDetection(is_new_task=False, task_title=None, confidence=0.0)
        else:
            # 5. No clear context detected
            result = TaskDetection(is_new_task=False, task_title=None, confidence=0.0)

        # Record metrics
        elapsed_ms = (utc_now() - start_time).total_seconds() * 1000
        self.metrics.record("semantic.task_detector.detection_time_ms", elapsed_ms)
        self.metrics.increment("semantic.task_detector.detections_total")

        return result

    def _check_new_task_patterns(self, message: str, lang: str) -> Optional[TaskDetection]:
        """Looks for patterns indicating a new task."""
        # Get language patterns
        lang_patterns = self.patterns.get(lang, {})
        new_task_patterns = lang_patterns.get("new_task", [])

        for pattern in new_task_patterns:
            # Convert simple pattern to regex
            if not pattern.startswith("^"):
                # Add boundary and capture group
                regex = rf"\b{pattern}\s+(.+?)(?:\.|$)"
            else:
                regex = pattern

            match = re.search(regex, message)
            if match:
                # Extract task title
                if match.groups():
                    task_title = match.group(1).strip()
                    # Clean title
                    task_title = re.sub(r"^(el|la|los|las|un|una|the|a|an)\s+", "", task_title)
                    task_title = task_title[:100]  # Limit length
                else:
                    # Use part of message as title
                    task_title = message[:50] + "..." if len(message) > 50 else message

                logger.info("Detected new task", task_title=task_title)

                return TaskDetection(is_new_task=True, task_title=task_title, confidence=0.9)

        return None

    def _check_continuation_patterns(
        self, message: str, lang: str, current_task: TaskCheckpoint
    ) -> Optional[TaskDetection]:
        """Looks for continuation patterns."""
        lang_patterns = self.patterns.get(lang, {})
        continuation_patterns = lang_patterns.get("continuation", [])

        for pattern in continuation_patterns:
            if re.search(pattern, message):
                logger.info(
                    "Detected task continuation",
                    task_id=current_task.id,
                    task_title=current_task.title,
                )

                return TaskDetection(
                    is_new_task=False,
                    task_title=None,
                    continues_task=current_task.id,
                    confidence=0.8,
                )

        return None

    def _calculate_context_similarity(
        self, message: str, current_task: TaskCheckpoint, recent_messages: List[str]
    ) -> float:
        """
        Calculates context similarity with the current task.

        Uses simple heuristics without ML:
        - Shared keywords
        - Mentioned entities (files, functions)
        - Similar message length
        """
        # Extract keywords from current message
        current_words = set(re.findall(r"\b\w{4,}\b", message.lower()))

        # Extract words from task context
        task_context = current_task.description or current_task.title
        task_words = set(re.findall(r"\b\w{4,}\b", task_context.lower()))

        # Add words from recent messages
        for msg in recent_messages[-5:]:  # Last 5 messages
            task_words.update(re.findall(r"\b\w{4,}\b", msg.lower()))

        # Calculate overlap
        if not current_words or not task_words:
            return 0.0

        intersection = len(current_words & task_words)
        union = len(current_words | task_words)

        # Jaccard similarity
        similarity = intersection / union if union > 0 else 0.0

        # Bonus for mentioning specific files/functions
        file_pattern = r"\b[\w\-]+\.(?:py|js|ts|java|go)\b"
        current_files = set(re.findall(file_pattern, message))
        task_files = set(re.findall(file_pattern, task_context))

        if current_files & task_files:
            similarity += 0.2

        return min(similarity, 1.0)

    def _get_default_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Returns default patterns if not present in config."""
        return {
            "es": {
                "new_task": [
                    "vamos a implementar",
                    "necesito crear",
                    "empecemos con",
                    "quiero desarrollar",
                    "hay que hacer",
                    "implementemos",
                    "agreguemos",
                ],
                "continuation": [
                    "sigamos con",
                    "continuemos",
                    "donde quedamos",
                    "lo que estábamos haciendo",
                    r"sobre el (.+) que",
                ],
            },
            "en": {
                "new_task": [
                    "let's implement",
                    "I need to create",
                    "let's start with",
                    "I want to develop",
                    "we need to make",
                    "let's add",
                ],
                "continuation": [
                    "let's continue",
                    "where were we",
                    "back to",
                    "what we were doing",
                    r"about the (.+) that",
                ],
            },
        }
