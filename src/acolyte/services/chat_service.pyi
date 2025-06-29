from typing import Dict, Any, Optional, List
from acolyte.services.conversation_service import ConversationService
from acolyte.services.task_service import TaskService
from acolyte.services.git_service import GitService
from acolyte.core.tracing import MetricsCollector
from acolyte.core.token_counter import TokenBudgetManager, SmartTokenCounter
from acolyte.core.ollama import OllamaClient
from acolyte.core.secure_config import Settings
from acolyte.semantic import (
    Summarizer,
    TaskDetector,
    PromptBuilder,
    QueryAnalyzer,
    DecisionDetector,
)
from acolyte.rag.retrieval.hybrid_search import HybridSearch
from acolyte.rag.compression import ContextualCompressor
from acolyte.models.task_checkpoint import TaskCheckpoint, TaskType
from acolyte.dream import DreamOrchestrator

class ChatService:
    metrics: MetricsCollector
    token_manager: TokenBudgetManager
    token_counter: SmartTokenCounter
    ollama: OllamaClient
    config: Settings
    debug_mode: bool
    weaviate_client: Optional[Any]
    query_analyzer: QueryAnalyzer
    task_detector: TaskDetector
    prompt_builder: PromptBuilder
    summarizer: Summarizer
    decision_detector: DecisionDetector
    conversation_service: ConversationService
    task_service: TaskService
    hybrid_search: Optional[HybridSearch]
    compressor: Optional[ContextualCompressor]
    git_service: Optional[GitService]
    dream_orchestrator: Optional[DreamOrchestrator]
    _active_session_id: Optional[str]
    _active_task: Optional[TaskCheckpoint]
    _last_user_message: Optional[str]

    def __init__(
        self,
        context_size: int,
        conversation_service: Optional[ConversationService] = None,
        task_service: Optional[TaskService] = None,
        debug_mode: bool = False,
    ) -> None: ...
    async def process_message(
        self, message: str, session_id: Optional[str] = None, debug: Optional[bool] = None
    ) -> Dict[str, Any]: ...
    async def _handle_new_chat(self) -> str: ...
    def _infer_task_type(self, message: str) -> TaskType: ...
    async def _generate_with_retry(
        self,
        system_prompt: str,
        user_message: str,
        context_chunks: list,
        max_tokens: int,
        max_attempts: int = 3,
    ) -> str: ...
    async def _get_project_info(self) -> Dict[str, Any]: ...
    async def _check_dream_suggestion(self) -> Optional[Dict[str, Any]]: ...
    def _is_code_related_query(self, message: str) -> bool: ...
    async def request_dream_analysis(
        self, user_query: str, focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]: ...
    async def get_active_session_info(self) -> Dict[str, Any]: ...
