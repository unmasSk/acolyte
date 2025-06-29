"""
Dynamic System Prompt builder.

Builds the dynamic prompt layer based on the current context.
The base layer comes from the Modelfile during installation.
"""

from typing import Optional, List, Dict, Any

from acolyte.core.logging import logger
from acolyte.core.utils.datetime_utils import utc_now
from acolyte.core.token_counter import get_token_counter
from acolyte.core.tracing import MetricsCollector
from acolyte.models.conversation import Conversation
from acolyte.models.task_checkpoint import TaskCheckpoint


class PromptBuilder:
    """Builds optimized dynamic prompts."""

    def __init__(self):
        self.token_counter = get_token_counter()
        # Metrics
        self.metrics = MetricsCollector()
        logger.info("PromptBuilder initialized")

    def build_dynamic_context(
        self,
        project: Dict[str, Any],
        session: Conversation,
        task: Optional[TaskCheckpoint] = None,
        recent_files: Optional[List[str]] = None,
        recent_decisions: Optional[List] = None,
        available_tokens: int = 2000,
    ) -> str:
        """
        Builds the dynamic System Prompt layer.

        IMPLEMENTS: get_summary() for rich technical decision context.

        Args:
            project: Project info (name, stack, branch, etc)
            session: Current session
            task: Active task if exists
            recent_files: Recently modified files
            recent_decisions: List of recent TechnicalDecision objects
            available_tokens: Token limit for the prompt

        Returns:
            Formatted dynamic system prompt
        """
        start_time = utc_now()

        # Build prompt sections
        sections = []

        # 1. Project context
        project_section = self._build_project_section(project, recent_files)
        sections.append(project_section)

        # 2. Session context
        session_section = self._build_session_section(session, task)
        sections.append(session_section)

        # 3. Continuity context
        if task:
            continuity_section = self._build_continuity_section(task, session)
            sections.append(continuity_section)

            # 3.1. Recent technical decisions (using get_summary())
            if recent_decisions:
                decisions_section = self._build_decisions_section(recent_decisions)
                sections.append(decisions_section)

        # 4. Available capabilities
        capabilities_section = self._build_capabilities_section()
        sections.append(capabilities_section)

        # Combine all
        full_context = "\n\n".join(sections)

        # Check token limit
        token_count = self.token_counter.count_tokens(full_context)
        if token_count > available_tokens:
            logger.warning(
                "Dynamic prompt exceeds limit", current_tokens=token_count, limit=available_tokens
            )
            # Prioritize recent information
            full_context = self._prioritize_context(sections, available_tokens)
            token_count = self.token_counter.count_tokens(full_context)
            self.metrics.increment("semantic.prompt_builder.prompts_truncated")

        # Register metrics
        elapsed_ms = (utc_now() - start_time).total_seconds() * 1000
        self.metrics.record("semantic.prompt_builder.build_time_ms", elapsed_ms)
        self.metrics.record("semantic.prompt_builder.prompt_tokens", token_count)
        self.metrics.increment("semantic.prompt_builder.prompts_built")

        return full_context

    def _build_project_section(
        self, project: Dict[str, Any], recent_files: Optional[List[str]] = None
    ) -> str:
        """Builds project context section."""
        lines = ["# CONTEXTO ACTUAL DEL PROYECTO"]

        # Basic info
        lines.append(f"## Proyecto: {project.get('name', 'Unknown')}")

        # Technology stack
        stack = project.get("stack", {})
        if stack:
            techs = []
            for category, items in stack.items():
                if isinstance(items, list) and items:
                    techs.extend(items)
            if techs:
                lines.append(f"- Stack detectado: {', '.join(techs[:10])}")

        # Recent files
        if recent_files:
            files_to_show = recent_files[-5:]  # Last 5
            lines.append(f"- Archivos recientes: {', '.join(files_to_show)}")

        # Current branch
        if project.get("current_branch"):
            lines.append(f"- Branch actual: {project['current_branch']}")

        return "\n".join(lines)

    def _build_session_section(
        self, session: Conversation, task: Optional[TaskCheckpoint] = None
    ) -> str:
        """Builds session context section."""
        lines = ["## Sesión"]

        lines.append(f"- ID: {session.session_id}")
        lines.append(f"- Mensajes previos: {len(session.messages)}")

        if task:
            lines.append(f"- Tarea activa: {task.title}")
        else:
            lines.append("- Tarea activa: Exploración general")

        return "\n".join(lines)

    def _build_continuity_section(self, task: TaskCheckpoint, session: Conversation) -> str:
        """
        Builds continuity context section.

        IMPLEMENTS: get_summary() for rich task context.
        """
        lines = ["## Contexto de continuidad"]

        # Task summary using get_summary()
        task_summary = task.get_summary()
        lines.append(f"- Tarea activa: {task_summary}")

        # Last session activity
        if session.messages:
            last_msg = session.messages[-1]
            if last_msg.role == "user":
                content = last_msg.content[:100]
                lines.append(f'- Última actividad: "{content}..."')

        # NOTE: Technical decisions are obtained separately from TaskService
        # since task.key_decisions are simple strings, not TechnicalDecision objects
        # Complete integration will be done in build_dynamic_context()

        return "\n".join(lines)

    def _build_decisions_section(self, recent_decisions: List) -> str:
        """
        Builds recent technical decisions section.

        IMPLEMENTS: get_summary() for rich decision context.

        Args:
            recent_decisions: List of TechnicalDecision objects

        Returns:
            Formatted section with decisions
        """
        lines = ["## Decisiones técnicas recientes"]

        if not recent_decisions:
            lines.append("- Ninguna decisión registrada")
        else:
            lines.append(f"- Total: {len(recent_decisions)} decisiones")

            # Show last 3 decisions using get_summary()
            for i, decision in enumerate(recent_decisions[-3:], 1):
                try:
                    # Use get_summary() from TechnicalDecision object
                    decision_summary = decision.get_summary()
                    lines.append(f"  {i}. {decision_summary}")
                except AttributeError:
                    # Fallback if object doesn't have get_summary()
                    lines.append(f"  {i}. {decision.title} - {decision.decision_type}")

        return "\n".join(lines)

    def _build_capabilities_section(self) -> str:
        """Builds available capabilities section."""
        return """## Capacidades disponibles
- Puedo ver y analizar tu código
- Puedo sugerir cambios (que Cursor puede aplicar automáticamente)
- Mantengo contexto de conversaciones anteriores
- Detecto y registro decisiones técnicas importantes"""

    # Method _format_recent_decisions removed because key_decisions are strings

    def _prioritize_context(self, sections: List[str], available_tokens: int) -> str:
        """
        Prioritizes context when exceeding the limit.

        Priority:
        1. Capabilities (always)
        2. Current session
        3. Basic project
        4. Continuity (if fits)
        """
        prioritized = []
        remaining_tokens = available_tokens

        # Always include capabilities
        capabilities = sections[-1]  # Last section
        cap_tokens = self.token_counter.count_tokens(capabilities)
        prioritized.append(capabilities)
        remaining_tokens -= cap_tokens

        # Include session if it fits
        if len(sections) > 1:
            session = sections[1]
            session_tokens = self.token_counter.count_tokens(session)
            if session_tokens <= remaining_tokens:
                prioritized.insert(0, session)
                remaining_tokens -= session_tokens

        # Include basic project if it fits
        if len(sections) > 0 and remaining_tokens > 100:
            project = sections[0]
            # Shorten if necessary
            project_lines = project.split("\n")[:4]  # Only first lines
            project_short = "\n".join(project_lines)
            project_tokens = self.token_counter.count_tokens(project_short)
            if project_tokens <= remaining_tokens:
                prioritized.insert(0, project_short)

        return "\n\n".join(prioritized)
