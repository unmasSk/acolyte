from typing import Optional, List, Dict, Any, Union, Iterator
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager
import sqlite3

class FetchType(Enum):
    ONE = "one"
    ALL = "all"
    NONE = "none"

@dataclass
class QueryResult:
    data: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]
    rows_affected: int
    last_row_id: Optional[int]

@dataclass
class StoreResult:
    success: bool
    id: str
    message: str
    stats: Dict[str, Any]

class DatabaseManager:
    db_path: str

    def __init__(self, db_path: Optional[str] = ...) -> None: ...
    def _get_connection(self) -> sqlite3.Connection: ...
    @contextmanager
    def transaction(self, isolation_level: str = ...) -> Iterator[sqlite3.Connection]: ...
    async def execute_async(
        self,
        query: str,
        params: Union[tuple[Any, ...], list[Any]] = ...,
        fetch: Optional[FetchType] = ...,
    ) -> QueryResult: ...
    def migrate_schema(self, target_version: int) -> None: ...

class InsightStore:
    db: DatabaseManager

    def __init__(self, db_manager: DatabaseManager) -> None: ...
    async def store_insights(
        self, session_id: str, insights: List[Dict[str, Any]], compression_level: int = ...
    ) -> StoreResult: ...

def get_db_manager() -> DatabaseManager: ...
