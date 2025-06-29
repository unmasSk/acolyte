from pathlib import Path
from typing import Set, Dict
from enum import Enum


class FileCategory(Enum):
    CODE = "code"
    DOCUMENTATION = "documentation"
    CONFIGURATION = "configuration"
    DATA = "data"
    OTHER = "other"


class FileTypeDetector:
    """Centralized file type detection and classification."""

    # Master mapping of extensions to languages
    LANGUAGE_MAP: Dict[str, str] = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".cs": "csharp",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".r": "r",
        ".m": "objective-c",
        ".mm": "objective-cpp",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "zsh",
        ".fish": "bash",  # Shell variants use bash
        ".lua": "lua",
        ".perl": "perl",
        ".pl": "perl",
        ".vim": "vimscript",
        ".el": "elisp",
        ".sql": "sql",
        ".md": "markdown",
        ".markdown": "markdown",
        ".rst": "restructuredtext",
        ".html": "html",
        ".htm": "html",
        ".xml": "xml",
        ".css": "css",
        ".scss": "scss",
        ".sass": "sass",
        ".less": "less",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".ini": "ini",
        ".cfg": "ini",
        ".conf": "ini",
        ".properties": "ini",
        ".dockerfile": "dockerfile",
        ".makefile": "makefile",
        ".cmake": "cmake",
    }

    # Category mappings
    CATEGORY_MAP: Dict[FileCategory, Set[str]] = {
        FileCategory.CODE: {
            ".py",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".java",
            ".go",
            ".rs",
            ".c",
            ".cpp",
            ".h",
            ".hpp",
            ".cc",
            ".cxx",
            ".cs",
            ".rb",
            ".php",
            ".swift",
            ".kt",
            ".scala",
            ".r",
            ".m",
            ".mm",
            ".sh",
            ".bash",
            ".zsh",
            ".fish",
            ".lua",
            ".perl",
            ".pl",
            ".vim",
            ".el",
            ".html",
            ".htm",
            ".xml",
            ".css",
            ".scss",
            ".sass",
            ".less",
        },
        FileCategory.DOCUMENTATION: {".md", ".rst", ".txt", ".adoc", ".markdown"},
        FileCategory.CONFIGURATION: {
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".ini",
            ".cfg",
            ".env",
            ".properties",
            ".xml",
            ".conf",
            ".config",
            ".gitignore",
            ".dockerignore",
            ".editorconfig",
            ".npmignore",
            ".eslintignore",
            ".prettierignore",
            ".stylelintignore",
            ".log",  # Log files are often used for configuration
        },
        FileCategory.DATA: {".csv", ".sql"},
    }

    # Special files without extensions
    SPECIAL_FILES: Dict[str, str] = {
        "dockerfile": "dockerfile",
        "makefile": "makefile",
        "rakefile": "ruby",
        "gemfile": "ruby",
        "gemfile.lock": "ruby",
        "pipfile": "toml",
        "pipfile.lock": "toml",
        "cargo.toml": "toml",
        "cargo.lock": "toml",
        "package.json": "json",
        "tsconfig.json": "json",
        "composer.json": "json",
        "composer.lock": "json",
        "requirements.txt": "text",
        "setup.py": "python",
        "go.mod": "go",
        "go.sum": "go",
        "pom.xml": "xml",
        "build.gradle": "groovy",
        "settings.gradle": "groovy",
        "docker-compose.yml": "yaml",
        "docker-compose.yaml": "yaml",
        "cmakelists.txt": "cmake",
        "jest.config.js": "javascript",
        "webpack.config.js": "javascript",
        "vite.config.js": "javascript",
        "podfile": "ruby",
        "package.swift": "swift",
        ".eslintrc": "json",
        ".prettierrc": "json",
        ".babelrc": "json",
        ".gitignore": "gitignore",
        ".dockerignore": "dockerignore",
        ".gitattributes": "gitattributes",
        ".editorconfig": "editorconfig",
        ".npmignore": "npmignore",
        ".eslintignore": "eslintignore",
        ".prettierignore": "prettierignore",
        ".stylelintignore": "stylelintignore",
    }

    # Special files categories mapping
    SPECIAL_FILES_CATEGORIES: Dict[str, FileCategory] = {
        ".env": FileCategory.CONFIGURATION,
        ".gitignore": FileCategory.CONFIGURATION,
        ".dockerignore": FileCategory.CONFIGURATION,
        "Dockerfile": FileCategory.CONFIGURATION,
        "dockerfile": FileCategory.CONFIGURATION,
        "Makefile": FileCategory.CONFIGURATION,
        "makefile": FileCategory.CONFIGURATION,
        "README": FileCategory.DOCUMENTATION,
        "LICENSE": FileCategory.DOCUMENTATION,
        ".eslintrc": FileCategory.CONFIGURATION,
        ".prettierrc": FileCategory.CONFIGURATION,
        ".babelrc": FileCategory.CONFIGURATION,
        ".editorconfig": FileCategory.CONFIGURATION,
        ".npmignore": FileCategory.CONFIGURATION,
        ".eslintignore": FileCategory.CONFIGURATION,
        ".prettierignore": FileCategory.CONFIGURATION,
        ".stylelintignore": FileCategory.CONFIGURATION,
        ".gitattributes": FileCategory.CONFIGURATION,
        "requirements.txt": FileCategory.CONFIGURATION,
        "package.json": FileCategory.CONFIGURATION,
        "tsconfig.json": FileCategory.CONFIGURATION,
        "composer.json": FileCategory.CONFIGURATION,
        "cargo.toml": FileCategory.CONFIGURATION,
        "go.mod": FileCategory.CONFIGURATION,
        "pom.xml": FileCategory.CONFIGURATION,
        "docker-compose.yml": FileCategory.CONFIGURATION,
        "docker-compose.yaml": FileCategory.CONFIGURATION,
        "podfile": FileCategory.CONFIGURATION,
        "Podfile": FileCategory.CONFIGURATION,
    }

    @classmethod
    def get_language(cls, path: Path) -> str:
        """Get programming language for a file."""
        # Check special files first (case-insensitive)
        filename_lower = path.name.lower()
        if filename_lower in cls.SPECIAL_FILES:
            return cls.SPECIAL_FILES[filename_lower]

        # Then check by extension
        return cls.LANGUAGE_MAP.get(path.suffix.lower(), "unknown")

    @classmethod
    def get_category(cls, path: Path) -> FileCategory:
        """Get category for a file."""
        # Check special files first (case-insensitive)
        filename_lower = path.name.lower()
        if filename_lower in cls.SPECIAL_FILES_CATEGORIES:
            return cls.SPECIAL_FILES_CATEGORIES[filename_lower]

        # Also check original case for compatibility
        if path.name in cls.SPECIAL_FILES_CATEGORIES:
            return cls.SPECIAL_FILES_CATEGORIES[path.name]

        # Check by extension
        suffix = path.suffix.lower()

        # Special handling for files starting with . but no extension
        if not suffix and path.name.startswith("."):
            # These are typically config files
            if path.name.lower() in [
                ".gitignore",
                ".dockerignore",
                ".editorconfig",
                ".npmignore",
                ".eslintignore",
                ".prettierignore",
                ".stylelintignore",
                ".gitattributes",
            ]:
                return FileCategory.CONFIGURATION

        for category, extensions in cls.CATEGORY_MAP.items():
            if suffix in extensions:
                return category

        return FileCategory.OTHER

    @classmethod
    def is_supported(cls, path: Path) -> bool:
        """Check if file type is supported for indexing."""
        return cls.get_category(path) != FileCategory.OTHER

    @classmethod
    def get_all_supported_extensions(cls) -> Set[str]:
        """Get all supported file extensions."""
        all_extensions = set()
        for extensions in cls.CATEGORY_MAP.values():
            all_extensions.update(extensions)
        return all_extensions
