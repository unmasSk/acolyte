from enum import Enum
from typing import List

class CollectionName(str, Enum):
    CONVERSATION = "Conversation"
    CODE_CHUNK = "CodeChunk"
    DOCUMENT = "Document"
    TASK = "Task"
    DREAM_INSIGHT = "DreamInsight"

    def __str__(self) -> str: ...
    @classmethod
    def list_all(cls) -> List[str]: ...
