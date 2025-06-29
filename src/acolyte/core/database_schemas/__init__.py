"""
Database module for ACOLYTE.

Manages dual persistence: SQLite + Weaviate.

Note: Database initialization script has been moved to /scripts/init_database.py
"""

# Import from database.py file in parent directory (..)
from ..database import (
    DatabaseManager,
    get_db_manager,
    InsightStore,
    FetchType,
    QueryResult,
    StoreResult,
)

__all__ = [
    "DatabaseManager",
    "get_db_manager",
    "InsightStore",
    "FetchType",
    "QueryResult",
    "StoreResult",
]
