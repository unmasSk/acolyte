"""
Collections module - Weaviate schema management for ACOLYTE.

Defines and manages the 5 main collections:
- Conversation: Conversation summaries
- CodeChunk: Code fragments with 18 ChunkTypes
- Document: Complete documents
- Task: Session grouping
- DreamInsight: Optimizer insights
"""

from acolyte.rag.collections.manager import CollectionManager, get_collection_manager
from acolyte.rag.collections.collection_names import CollectionName

__all__ = ["CollectionManager", "get_collection_manager", "CollectionName"]
