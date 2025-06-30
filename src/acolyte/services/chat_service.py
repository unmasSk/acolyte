"""
Chat Service - MAIN ORCHESTRATOR.

Coordinates the entire chat flow by integrating all components.
"""

from acolyte.core.logging import logger
from acolyte.core.tracing import MetricsCollector
from acolyte.core.token_counter import TokenBudgetManager, SmartTokenCounter
from acolyte.core.ollama import OllamaClient
from acolyte.core.exceptions import AcolyteError, ExternalServiceError
from acolyte.core.secure_config import Settings
from acolyte.core.id_generator import generate_id
from acolyte.core.utils.retry import retry_async
from acolyte.semantic import (
    Summarizer,
    TaskDetector,
    PromptBuilder,
    QueryAnalyzer,
    DecisionDetector,
)
from acolyte.rag.retrieval.hybrid_search import HybridSearch
from acolyte.rag.compression import ContextualCompressor  # Decision #14
from acolyte.models.task_checkpoint import TaskCheckpoint, TaskType
from acolyte.models.technical_decision import TechnicalDecision, DecisionType
from acolyte.models.semantic_types import DetectedDecision
from acolyte.services.conversation_service import ConversationService
from acolyte.services.task_service import TaskService
from acolyte.services.git_service import GitService
from acolyte.dream import create_dream_orchestrator
from acolyte.dream.orchestrator import DreamTrigger
from typing import Dict, Any, Optional, cast
from datetime import datetime, timedelta
from acolyte.core.utils.datetime_utils import utc_now


class ChatService:
    """
    Orchestrates the entire chat flow.

    CRITICAL FLOW:
    1. Loads previous context automatically (Decision #7)
    2. Analyzes query with Semantic
    3. Searches with hybrid RAG
    4. Generates with Ollama
    5. Resumes with Semantic
    6. Persists with Services
    7. Detects decisions
    """

    def __init__(
        self,
        context_size: int,
        conversation_service=None,
        task_service=None,
        debug_mode: bool = False,
    ):
        """
        Initializes ChatService with optional dependency injection.

        Args:
            context_size: Size of the model context
            conversation_service: ConversationService instance (optional)
            task_service: TaskService instance (optional)
            debug_mode: Whether to include debug information in responses
        """
        self.metrics = MetricsCollector()
        self.token_manager = TokenBudgetManager(context_size)
        self.token_counter = SmartTokenCounter()  # Para conteo preciso
        self.ollama = OllamaClient()
        self.config = Settings()
        self.debug_mode = debug_mode

        # Initialize Weaviate client for RAG and Dream
        try:
            import weaviate  # type: ignore

            weaviate_port = self.config.get("ports.weaviate", 42080)
            weaviate_url = f"http://localhost:{weaviate_port}"
            self.weaviate_client = weaviate.Client(weaviate_url)

            if not self.weaviate_client.is_ready():
                logger.warning("Weaviate not ready")
                self.weaviate_client = None
        except Exception as e:
            logger.warning("Weaviate not available", error=str(e))
            self.weaviate_client = None

        # Semantic components
        self.query_analyzer = QueryAnalyzer()
        self.task_detector = TaskDetector()
        self.prompt_builder = PromptBuilder()
        self.summarizer = Summarizer()
        self.decision_detector = DecisionDetector()

        # Services - use injected or create new ones
        if conversation_service is None:
            self.conversation_service = ConversationService()
        else:
            self.conversation_service = conversation_service

        if task_service is None:
            self.task_service = TaskService()
        else:
            self.task_service = task_service

        # RAG
        try:
            if self.weaviate_client:
                self.hybrid_search = HybridSearch(weaviate_client=self.weaviate_client)
                self.compressor = ContextualCompressor(self.token_counter)  # For specific queries
            else:
                logger.warning("RAG components not available - Weaviate client missing")
                self.hybrid_search = None
                self.compressor = None
        except Exception as e:
            logger.warning("RAG components not available", error=str(e))
            self.hybrid_search = None
            self.compressor = None

        # Active session cache (mono-user)
        self._active_session_id: Optional[str] = None
        self._active_task: Optional[TaskCheckpoint] = None
        self._last_user_message: Optional[str] = None
        self.git_service: Optional[GitService] = None

        # Dream system integration
        try:
            if self.weaviate_client:
                # Use factory with weaviate client for full functionality
                self.dream_orchestrator = create_dream_orchestrator(
                    weaviate_client=self.weaviate_client
                )
            else:
                logger.warning("Dream system limited - no Weaviate client available")
                self.dream_orchestrator = None
        except Exception as e:
            logger.warning("Dream system not available", error=str(e))
            self.dream_orchestrator = None

        logger.info("ChatService initialized", context_size=context_size, debug_mode=debug_mode)

    async def process_message(
        self, message: str, session_id: Optional[str] = None, debug: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Processes a complete message.

        DETAILED FLOW:
        1. If no session_id, detects if it's a new chat
        2. Loads previous context (complete task or last session)
        3. Analyzes intent for token distribution
        4. Detects if it's a new task or continuation
        5. Searches for relevant code with RAG
        6. Builds dynamic prompt
        7. Generates response with Ollama
        8. Generates summary WITH chunks for context
        9. Detects technical decisions
        10. Persists everything

        Args:
            message: User message
            session_id: Session ID (optional, created if not exists)
            debug: Override debug mode (optional)

        Returns:
            Dict with response, session_id, task_id, tokens_used, processing_time
            If debug=True, includes additional debug_info

        Raises:
            AcolyteError: General system error
            ExternalServiceError: If Ollama or external services fail
            DatabaseError: If persistence fails
        """
        start_time = utc_now()

        # Override debug mode if specified
        include_debug = debug if debug is not None else self.debug_mode

        # Store last user message for Dream analysis check
        self._last_user_message = message

        try:
            # STEP 1: Session management
            if not session_id:
                session_id = await self._handle_new_chat()

            self._active_session_id = session_id

            # Load session context
            session_context = await self.conversation_service.get_session_context(
                session_id, include_related=True
            )

            # STEP 2: Query analysis
            # IMPORTANT: TokenDistribution is an object, use .type for string
            distribution = self.query_analyzer.analyze_query_intent(message)
            self.token_manager.allocate_for_query_type(distribution.type)  # ← .type

            logger.info(
                "Query analysis complete",
                query_type=distribution.type,
                response_ratio=distribution.response_ratio,
            )

            # STEP 3: Task detection
            task_detection = await self.task_detector.detect_task_context(
                message, self._active_task
            )

            # If new task is detected, create it
            if task_detection.is_new_task and task_detection.task_title:
                await self.task_service.create_task(
                    title=task_detection.task_title,
                    description=message,
                    task_type=self._infer_task_type(message),
                    initial_session_id=session_id,
                )

                # Load the newly created task
                self._active_task = await self.task_service.find_active_task()
                logger.info("Created new task", title=task_detection.task_title)

            elif task_detection.continues_task and self._active_task:
                # Associate current session to existing task
                await self.task_service.associate_session_to_task(self._active_task.id, session_id)

            # STEP 4: RAG search
            chunks = []
            if self.hybrid_search:
                try:
                    # If query is specific, use compression
                    if distribution.type in ["simple", "generation"] and self.compressor:
                        available_tokens = self.token_manager.get_remaining("rag")
                        chunks = await self.hybrid_search.search_with_compression(
                            query=message, max_chunks=10, token_budget=available_tokens
                        )
                        self.metrics.increment("chat.compressed_searches")
                    else:
                        chunks = await self.hybrid_search.search(query=message, max_chunks=10)
                        # Convert ScoredChunk to Chunk
                        chunks = [scored_chunk.chunk for scored_chunk in chunks]

                    self.metrics.gauge("chat.chunks_retrieved", len(chunks))

                except Exception as e:
                    logger.error("RAG search failed", error=str(e))
                    chunks = []

            # STEP 5: Build prompt
            available_tokens = self.token_manager.get_remaining("system")

            # Get project information from configuration
            project_info = await self._get_project_info()

            # Get recent technical decisions if there is an active task
            recent_decisions = []
            if self._active_task:
                try:
                    # Get last 3 decisions from the task
                    recent_decisions = await self.task_service.get_recent_decisions(
                        self._active_task.id, limit=3
                    )
                except Exception as e:
                    logger.warning("Could not get recent decisions", error=str(e))

            dynamic_prompt = self.prompt_builder.build_dynamic_context(
                project=project_info,
                session=session_context["session"],
                task=self._active_task,
                recent_decisions=recent_decisions,
                available_tokens=available_tokens,
            )

            # STEP 6: Generate response WITH RETRY
            response_tokens = self.token_manager.get_remaining("response")

            try:
                response = await self._generate_with_retry(
                    system_prompt=dynamic_prompt,  # Dinámico de Semantic
                    user_message=message,
                    context_chunks=chunks,
                    max_tokens=response_tokens,
                )

                # Count tokens used with SmartTokenCounter
                tokens_used = self.token_counter.count_tokens(response)
                self.token_manager.use("response", tokens_used)

            except Exception as e:
                logger.error("Ollama generation failed after retries", error=str(e))
                raise ExternalServiceError(f"Failed to generate response: {str(e)}") from e

            # STEP 7: Generate summary WITH CHUNKS
            summary_result = await self.summarizer.generate_summary(
                user_msg=message,
                assistant_msg=response,
                context_chunks=chunks,  # ← NECESSARY for entities
            )

            # STEP 8: Detect decisions
            detected_decision: Optional[DetectedDecision] = (
                self.decision_detector.detect_technical_decision(response)
            )

            # STEP 9: Persist
            await self.conversation_service.save_conversation_turn(
                session_id=session_id,
                user_message=message,
                assistant_response=response,
                summary=summary_result.summary,
                tokens_used=tokens_used,
                task_checkpoint_id=self._active_task.id if self._active_task else None,
            )

            # If a decision is detected, save it
            if detected_decision and self._active_task:
                # Validate type before using
                if not hasattr(detected_decision, "decision_type"):
                    logger.warning("Invalid decision object detected")
                else:
                    # Convert DetectedDecision to complete TechnicalDecision
                    technical_decision = TechnicalDecision(
                        id=generate_id(),
                        created_at=utc_now(),
                        decision_type=DecisionType(detected_decision.decision_type),
                        title=detected_decision.title,
                        description=detected_decision.description,
                        rationale=detected_decision.rationale,
                        alternatives_considered=detected_decision.alternatives_considered,
                        impact_level=detected_decision.impact_level,
                        session_id=session_id,  # ChatService completes the ID
                        task_id=self._active_task.id,
                    )

                    await self.task_service.save_technical_decision(
                        technical_decision  # It already has task_id and session_id
                    )

                    logger.info(
                        "Technical decision saved",
                        type=detected_decision.decision_type,
                        impact=detected_decision.impact_level,
                    )

            # Calculate metrics
            processing_time = (utc_now() - start_time).total_seconds()
            self.metrics.record("chat.processing_time_seconds", processing_time)
            self.metrics.increment("chat.messages_processed")

            # Build response
            result = {
                "response": response,
                "session_id": session_id,
                "task_id": self._active_task.id if self._active_task else None,
                "tokens_used": {
                    "prompt": self.token_manager.used.get("system", 0),
                    "context": self.token_manager.used.get("rag", 0),
                    "response": tokens_used,
                    "total": sum(self.token_manager.used.values()),
                },
                "processing_time": processing_time,
            }

            # Add debug info if requested
            if include_debug:
                # BUGFIX: The method is get_distribution(), not get_total_used()
                distribution_map = self.token_manager.get_distribution()
                # Sum tokens from all categories and subcategories
                tokens_used = sum(val for cat in distribution_map.values() for val in cat.values())
                processing_time = (utc_now() - start_time).total_seconds()
                debug_info = {
                    "session_id": session_id,
                    "query_type": distribution.type,
                    "task_detected": task_detection.is_new_task,
                    "task_title": task_detection.task_title,
                    "chunks_retrieved": len(chunks),
                    "prompt": {
                        "system": dynamic_prompt,
                        "user": message,
                    },
                    "timing": {
                        "total_processing_seconds": round(processing_time, 2),
                    },
                    "tokens": self.token_manager.get_distribution(),
                    "summary_compression": summary_result.tokens_saved / max(1, tokens_used),
                }
                result["debug_info"] = debug_info

            # STEP 10: Check Dream fatigue and suggest if needed
            if not include_debug and self.dream_orchestrator:
                dream_suggestion = await self._check_dream_suggestion()
                if dream_suggestion:
                    result["suggestion"] = dream_suggestion

            return result

        except Exception as e:
            logger.error("Error processing message", error=str(e))
            self.metrics.increment("chat.processing_errors")
            raise AcolyteError(f"Failed to process message: {str(e)}") from e

    async def _handle_new_chat(self) -> str:
        """Handles the logic for starting a new chat session."""
        self._active_task = await self.task_service.find_active_task()

        # If there's an active task with an initial session, use that
        if self._active_task and self._active_task.initial_session_id:
            logger.info(
                "Continuing with active task",
                task_title=self._active_task.title,
            )
            return self._active_task.initial_session_id

        # If there's no active task, try to load the last session as context
        last_session = await self.conversation_service.get_last_session()
        if last_session:
            logger.info("Continuing from last session", session_id=last_session["id"])
            return last_session["id"]

        # Otherwise, create a new session
        logger.info("Starting new conversation")
        new_session_id = await self.conversation_service.create_session("New conversation")
        return new_session_id

    def _infer_task_type(self, message: str) -> TaskType:
        """
        Infers the task type based on the message.

        Args:
            message: User message

        Returns:
            Inferred TaskType (default: RESEARCH)
        """
        message_lower = message.lower()

        # Check in order of specificity
        if any(kw in message_lower for kw in ["error", "bug", "fix", "arreglar", "problema"]):
            return TaskType.DEBUGGING
        if any(kw in message_lower for kw in ["refactor", "mejorar", "optimizar", "clean"]):
            return TaskType.REFACTORING
        if any(kw in message_lower for kw in ["document", "explicar", "readme", "docs"]):
            return TaskType.DOCUMENTATION
        if any(kw in message_lower for kw in ["review", "revisar", "check", "validar"]):
            return TaskType.REVIEW
        if any(
            kw in message_lower
            for kw in ["research", "investigar", "analizar", "explorar", "study"]
        ):
            return TaskType.RESEARCH
        if any(
            kw in message_lower
            for kw in ["implement", "crear", "añadir", "develop", "build", "feature"]
        ):
            return TaskType.IMPLEMENTATION

        # Default for generic queries
        return TaskType.RESEARCH

    async def _generate_with_retry(
        self,
        system_prompt: str,
        user_message: str,
        context_chunks: list,
        max_tokens: int,
        max_attempts: int = 3,
    ) -> str:
        """
        Generates response with Ollama using retry logic.

        IMPLEMENTS: is_retryable() for robustness.

        Args:
            system_prompt: Dynamic system prompt
            user_message: User message
            context_chunks: Chunks of code for context
            max_tokens: Response token limit
            max_attempts: Maximum number of attempts

        Returns:
            Response generated by Ollama

        Raises:
            ExternalServiceError: If all attempts fail
        """
        # Track attempts for metrics
        attempt_count = 0

        # Custom exception to signal non-retryable errors
        class NonRetryableError(Exception):
            def __init__(self, original_error: Exception):
                self.original_error = original_error
                super().__init__(str(original_error))

        # Create wrapper that preserves is_retryable() logic
        async def retryable_ollama_operation():
            nonlocal attempt_count
            attempt_count += 1

            # Track retry attempts after the first one
            if attempt_count > 1:
                self.metrics.increment("chat.ollama_retries_attempted")

            try:
                # Combine user message with context chunks into a single prompt
                # OllamaClient doesn't handle chunks separately
                full_prompt = user_message
                if context_chunks:
                    # Add context from chunks
                    context_text = "\n\nRelevant code context:\n"
                    for chunk in context_chunks[:5]:  # Limit to first 5 chunks
                        context_text += f"\n--- {chunk.metadata.file_path} ---\n"
                        context_text += chunk.content[:500]  # Limit each chunk
                        context_text += "\n"
                    full_prompt = context_text + "\n\nUser query: " + user_message

                response = await self.ollama.generate(
                    prompt=full_prompt,
                    system=system_prompt,
                    max_tokens=max_tokens,
                    stream=False,  # Ensure we get string, not AsyncIterator
                )

                # Type narrowing: when stream=False, response is always str
                return cast(str, response)

            except (AcolyteError, ExternalServiceError) as e:
                # Check if the error is retryable
                if hasattr(e, 'is_retryable') and not e.is_retryable():
                    logger.error("Ollama generation failed permanently", error=str(e))
                    # Wrap in NonRetryableError to stop retries
                    raise NonRetryableError(e)
                # Re-raise to trigger retry
                raise
            except Exception as e:
                # Convert uncontrolled errors to ExternalServiceError (retryable by default)
                raise ExternalServiceError(f"Ollama error: {str(e)}", cause=e)

        try:
            # Use retry_async with exponential backoff
            response = await retry_async(
                retryable_ollama_operation,
                max_attempts=max_attempts,
                backoff="exponential",
                initial_delay=1.0,  # 2^0 = 1s for first retry
                retry_on=(AcolyteError, ExternalServiceError),  # NOT NonRetryableError
                logger=logger,
            )

            # Success - handle metrics
            if attempt_count > 1:
                logger.info("Ollama generation succeeded after retry", attempts=attempt_count)
                self.metrics.increment("chat.ollama_retries_successful")

            return response

        except NonRetryableError as e:
            # Non-retryable error, re-raise the original
            raise e.original_error
        except Exception as e:
            # All attempts failed
            self.metrics.increment("chat.ollama_retries_exhausted")
            if isinstance(e, (AcolyteError, ExternalServiceError)):
                raise
            # Wrap any other exception
            raise ExternalServiceError("Ollama generation failed after all retries") from e

    async def _get_project_info(self) -> Dict[str, Any]:
        """Gets project info using an instance of GitService."""
        info = {
            "project_name": self.config.get("project.name", "Unknown Project"),
            "current_branch": "main",  # Default value
            "recent_files": [],  # Default value
        }
        try:
            # Use the instance-level git_service, creating it only if it doesn't exist.
            # This makes the service testable by allowing injection of a mock service.
            if not self.git_service:
                self.git_service = GitService()

            git = self.git_service
            info["current_branch"] = git.repo.active_branch.name
            info["recent_files"] = git.get_most_recent_files()

        except Exception as e:
            logger.warning("Failed to get project info from Git", error=str(e))
            # Return defaults on any failure
        return info

    async def _check_dream_suggestion(self) -> Optional[Dict[str, Any]]:
        """
        Checks if a dream analysis should be suggested based on fatigue.
        """
        if not self.dream_orchestrator:
            return None

        try:
            # First and foremost, do not suggest if an analysis is already running
            if await self.dream_orchestrator.is_analysis_in_progress():
                logger.info("Dream suggestion skipped: analysis already in progress.")
                return None

            fatigue_info = await self.dream_orchestrator.check_fatigue_level()
            is_high = fatigue_info.get("is_high", False)
            is_emergency = fatigue_info.get("is_emergency", False)

            # If fatigue is emergency level, suggest immediately, ignoring other checks
            if is_emergency:
                logger.info("Emergency fatigue detected, forcing dream suggestion.")
                # Fall through to request analysis
                pass
            # Otherwise, perform standard checks
            elif not is_high:
                return None
            elif not self._last_user_message or not self._is_code_related_query(
                self._last_user_message
            ):
                logger.info("Dream suggestion skipped: not a code-related query.")
                return None
            else:
                last_optimization_iso = fatigue_info.get("last_optimization")
                if last_optimization_iso:
                    last_opt_time = datetime.fromisoformat(last_optimization_iso)
                    if utc_now() - last_opt_time < timedelta(hours=2):
                        logger.info("Dream suggestion skipped: recent optimization.")
                        return None

            # If all checks passed or were bypassed by emergency, request analysis
            analysis_request = await self.dream_orchestrator.request_analysis(
                trigger=DreamTrigger("FATIGUE_SUGGESTION"),
                context={
                    "session_id": self._active_session_id,
                    "task_id": self._active_task.id if self._active_task else None,
                    "fatigue_level": fatigue_info.get("fatigue_level"),
                    "is_emergency": is_emergency,
                },
            )

            # Only return a suggestion if the orchestrator requires user permission
            if analysis_request.get("status") == "permission_required":
                return {
                    "type": "dream_analysis",
                    "request_id": analysis_request["request_id"],
                    "message": analysis_request["message"],
                    "fatigue_level": fatigue_info.get("fatigue_level"),
                    "benefits": analysis_request.get("benefits"),
                    "estimated_duration": analysis_request.get("estimated_duration_minutes"),
                }

        except Exception as e:
            logger.error("Failed to check dream suggestion", error=str(e))

        return None

    def _is_code_related_query(self, message: str) -> bool:
        """
        Check if user message is related to code/implementation.

        Args:
            message: User message to check

        Returns:
            True if message seems code-related
        """
        if not message:
            return False

        # Keywords that indicate code-related queries
        code_keywords = [
            # English
            "implement",
            "code",
            "function",
            "class",
            "method",
            "bug",
            "error",
            "fix",
            "test",
            "refactor",
            "optimize",
            "debug",
            "compile",
            "build",
            "import",
            "module",
            "package",
            "api",
            "endpoint",
            "query",
            "database",
            # Spanish
            "implementar",
            "código",
            "función",
            "clase",
            "método",
            "arreglar",
            "probar",
            "optimizar",
            "depurar",
            "compilar",
            "construir",
        ]

        message_lower = message.lower()
        return any(keyword in message_lower for keyword in code_keywords)

    async def request_dream_analysis(
        self, user_query: str, focus_areas: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Requests an analysis from the Dream system.
        """
        if not self.dream_orchestrator:
            raise AcolyteError("Dream system is not available")

        # The context is important for the analysis
        context = {
            "session_id": self._active_session_id,
            "task_id": self._active_task.id if self._active_task else None,
        }

        analysis_result = await self.dream_orchestrator.request_analysis(
            trigger=DreamTrigger("USER_REQUEST"),
            user_query=user_query,
            focus_areas=focus_areas,
            context=context,
        )
        return analysis_result

    async def get_active_session_info(self) -> Dict[str, Any]:
        """
        Gets active session information.

        Useful for dashboard or status checks.

        Returns:
            Dict with session_id, task info, token_usage, debug_mode
        """
        info = {
            "session_id": self._active_session_id,
            "task": {
                "id": self._active_task.id if self._active_task else None,
                "title": self._active_task.title if self._active_task else None,
            },
            "token_usage": self.token_manager.get_distribution(),
            "debug_mode": self.debug_mode,
        }

        # Add Dream status if available
        if self.dream_orchestrator:
            try:
                fatigue_data = await self.dream_orchestrator.check_fatigue_level()
                info["dream_status"] = {
                    "fatigue_level": fatigue_data["fatigue_level"],
                    "is_high": fatigue_data["is_high"],
                    "threshold": fatigue_data["threshold"],
                    "last_optimization": fatigue_data.get("last_optimization"),
                }
            except Exception as e:
                info["dream_status"] = None
                logger.debug("Could not get Dream status", error=str(e))
        else:
            info["dream_status"] = None

        return info
