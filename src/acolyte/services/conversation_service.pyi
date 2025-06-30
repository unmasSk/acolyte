from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from acolyte.core.tracing import MetricsCollector
from acolyte.core.token_counter import SmartTokenCounter
from acolyte.core.secure_config import Settings
from acolyte.core.database import DatabaseManager
from acolyte.core.events import Event
from acolyte.models.conversation import ConversationSearchRequest, ConversationSearchResult

class ConversationService:
    metrics: MetricsCollector
    token_counter: SmartTokenCounter
    config: Settings
    db: DatabaseManager
    _session_cache: Optional[Any]
    _cache_subscription: Any

    def __init__(self) -> None: ...
    async def create_session(self, initial_message: str) -> str: ...
    async def save_conversation_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        summary: str,
        tokens_used: int,
        task_checkpoint_id: Optional[str] = None,
    ) -> None: ...
    async def find_related_sessions(
        self, query: str, current_session_id: str, limit: int = 5
    ) -> List[Dict[str, Any]]: ...
    async def get_session_context(
        self, session_id: str, include_related: bool = True
    ) -> Dict[str, Any]: ...
    async def search_conversations(
        self, request: ConversationSearchRequest
    ) -> List[ConversationSearchResult]: ...
    async def get_last_session(self) -> Optional[Dict[str, Any]]: ...
    async def complete_session(self, session_id: str) -> None: ...
    async def _search_by_keywords(
        self,
        query: str,
        exclude_session_id: Optional[str],
        limit: int,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        include_completed: bool = True,
        task_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]: ...
    async def _search_by_keywords_typed(
        self,
        query: str,
        exclude_session_id: Optional[str],
        limit: int,
        time_range: Optional[Tuple[datetime, datetime]],
        include_completed: bool = True,
        task_id: Optional[str] = None,
    ) -> List[ConversationSearchResult]: ...
    def _extract_keywords(self, text: str) -> List[str]: ...
    async def _handle_cache_invalidation(self, event: Event) -> None: ...
    async def _execute_with_retry(
        self,
        operation_name: str,
        db_operation: Any,
        *args: Any,
        max_attempts: int = 3,
        **kwargs: Any,
    ) -> Any: ...
