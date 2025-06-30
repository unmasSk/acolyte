from pathlib import Path
from typing import Set, Dict, ClassVar
from enum import Enum

class FileCategory(Enum):
    CODE = "code"
    DOCUMENTATION = "documentation"
    CONFIGURATION = "configuration"
    DATA = "data"
    OTHER = "other"

class FileTypeDetector:
    LANGUAGE_MAP: ClassVar[Dict[str, str]]
    CATEGORY_MAP: ClassVar[Dict[FileCategory, Set[str]]]
    SPECIAL_FILES: ClassVar[Dict[str, str]]
    SPECIAL_FILES_CATEGORIES: ClassVar[Dict[str, FileCategory]]

    @classmethod
    def get_language(cls, path: Path) -> str: ...
    @classmethod
    def get_category(cls, path: Path) -> FileCategory: ...
    @classmethod
    def is_supported(cls, path: Path) -> bool: ...
    @classmethod
    def get_all_supported_extensions(cls) -> Set[str]: ...
