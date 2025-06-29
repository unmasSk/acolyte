"""
Collection names enum for collections module.
Defines ACOLYTE's 5 mandatory collections.
"""

from enum import Enum
from typing import List


class CollectionName(str, Enum):
    """
    Weaviate collection names.
    These are the system's 5 mandatory collections.
    """

    CONVERSATION = "Conversation"
    CODE_CHUNK = "CodeChunk"
    DOCUMENT = "Document"
    TASK = "Task"
    DREAM_INSIGHT = "DreamInsight"

    def __str__(self):
        """Return the value directly like a str enum should."""
        return self.value

    @classmethod
    def list_all(cls) -> List[str]:
        """
        List all collection names.

        Returns:
            List with enum values (not names)
        """
        return [member.value for member in cls]
