"""
Chunker factory for ACOLYTE.
Creates appropriate chunkers based on file type and language detection.
"""

from pathlib import Path
from typing import Optional, Dict
import re

from acolyte.core.logging import logger
from acolyte.core.utils.file_types import FileTypeDetector
from .base import BaseChunker

# Language configuration separated for maintainability
from .language_config import LANGUAGE_CONFIG


class ChunkerFactory:
    """
    Factory for creating language-specific chunkers.

    Features:
    - Automatic language detection by extension
    - Fallback to adaptive chunker
    - Extensible language registry
    """

    # Extension to language mapping
    EXTENSION_MAP: Dict[str, str] = LANGUAGE_CONFIG['extensions']

    # Shebang patterns for language detection
    SHEBANG_PATTERNS = {
        re.compile(pattern, re.IGNORECASE): lang
        for pattern, lang in LANGUAGE_CONFIG['shebang_patterns'].items()
    }

    @classmethod
    def create(cls, file_path: str, content: Optional[str] = None) -> BaseChunker:
        """
        Create appropriate chunker for the given file.

        Args:
            file_path: Path to the file
            content: Optional content for language detection

        Returns:
            Language-specific chunker or DefaultChunker as fallback
        """
        # Validate binary files
        if content and '\x00' in content[:1000]:
            raise ValueError(f"Binary file not supported: {file_path}")

        language = cls.detect_language(file_path, content)
        logger.debug(f"Detected language '{language}' for {file_path}")

        # Try to get language-specific chunker first
        chunker = cls._get_language_chunker(language)

        if chunker:
            logger.info(f"Using {chunker.__class__.__name__} for {file_path}")
            return chunker
        else:
            # Fallback to DefaultChunker for unknown languages
            logger.info(f"Using DefaultChunker (fallback) for {file_path}")
            from .languages.default import DefaultChunker

            return DefaultChunker(language)

    @classmethod
    def detect_language(cls, file_path: str, content: Optional[str] = None) -> str:
        """
        Detect language from file path and optionally content.

        Detection order:
        1. File extension (using FileTypeDetector)
        2. Shebang line (if content provided)
        3. Content patterns (if content provided)
        4. 'unknown' as fallback

        Args:
            file_path: Path to the file
            content: Optional file content

        Returns:
            Detected language identifier
        """
        path = Path(file_path)

        # 1. Try FileTypeDetector first
        language = FileTypeDetector.get_language(path)
        if language != "unknown":
            return language

        # 2. Try shebang if content available
        if content:
            first_line = content.split('\n')[0] if content else ''
            if first_line.startswith('#!'):
                for pattern, lang in cls.SHEBANG_PATTERNS.items():
                    if pattern.match(first_line):
                        return lang

            # 3. Try content patterns
            language = cls._detect_from_content(content)
            if language:
                return language

        # 4. Special cases by filename
        filename = path.name.lower()
        return LANGUAGE_CONFIG['special_files'].get(filename, 'unknown')

    @classmethod
    def _detect_from_content(cls, content: str) -> Optional[str]:
        """
        Detect language from content patterns.

        Args:
            content: File content

        Returns:
            Detected language or None
        """
        # Content patterns from config
        patterns = {}
        for lang, pattern_list in LANGUAGE_CONFIG['content_patterns'].items():
            patterns[tuple(pattern_list)] = lang

        # Count matches for each language
        scores = {}

        for pattern_group, language in patterns.items():
            score = 0
            for pattern in pattern_group:
                matches = len(re.findall(pattern, content, re.MULTILINE))
                score += matches

            if score > 0:
                scores[language] = score

        # Return language with highest score
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]

        return None

    # Class-level chunker mapping
    _chunker_map = {
        'python': ('python', 'PythonChunker'),
        'default': ('default', 'DefaultChunker'),
        'makefile': ('makefile', 'MakefileChunker'),
        'dockerfile': ('dockerfile', 'DockerfileChunker'),
        'bash': ('bash', 'BashChunker'),
        'shell': ('bash', 'BashChunker'),
        'sh': ('bash', 'BashChunker'),
        'zsh': ('bash', 'BashChunker'),
        'fish': ('bash', 'BashChunker'),
        'javascript': ('typescript', 'TypeScriptChunker'),  # TS chunker handles JS too
        'typescript': ('typescript', 'TypeScriptChunker'),
        'java': ('java', 'JavaChunker'),
        'go': ('go', 'GoChunker'),
        'rust': ('rust', 'RustChunker'),
        'c': ('c', 'CChunker'),
        'cpp': ('cpp', 'CppChunker'),
        'csharp': ('csharp', 'CSharpChunker'),
        'ruby': ('ruby', 'RubyChunker'),
        'php': ('php', 'PhpChunker'),
        'perl': ('perl', 'PerlChunker'),
        'swift': ('swift', 'SwiftChunker'),
        'kotlin': ('kotlin', 'KotlinChunker'),
        'vimscript': ('vim', 'VimChunker'),
        'elisp': ('elisp', 'ElispChunker'),
        'r': ('r', 'RChunker'),
        'lua': ('lua', 'LuaChunker'),
        'sql': ('sql', 'SQLChunker'),
        'markdown': ('markdown', 'MarkdownChunker'),
        'html': ('html', 'HtmlChunker'),
        'xml': ('xml', 'XmlChunker'),
        'css': ('css', 'CssChunker'),
        'scss': ('css', 'CssChunker'),
        'sass': ('css', 'CssChunker'),
        'less': ('css', 'CssChunker'),
        'json': ('json', 'JSONChunker'),
        'yaml': ('yaml', 'YamlChunker'),
        'toml': ('toml', 'TomlChunker'),
        'ini': ('ini', 'IniChunker'),
    }

    @classmethod
    def _get_language_chunker(cls, language: str) -> Optional[BaseChunker]:
        """
        Get chunker instance for the specified language with full validation.

        Args:
            language: Language identifier

        Returns:
            Validated chunker instance or None if not available/invalid
        """
        if language not in cls._chunker_map:
            return None

        module_name, class_name = cls._chunker_map[language]

        try:
            # Dynamic import
            module = __import__(
                f'acolyte.rag.chunking.languages.{module_name}', fromlist=[class_name]
            )
            chunker_class = getattr(module, class_name)

            # Validate it's a BaseChunker subclass
            if not issubclass(chunker_class, BaseChunker):
                logger.error(f"{class_name} is not a BaseChunker subclass")
                return None

            # Try to instantiate
            chunker = chunker_class()

            # Validate required methods exist and are callable
            required_methods = ['chunk', '_get_language_name', '_get_import_node_types']
            for method in required_methods:
                if not hasattr(chunker, method) or not callable(getattr(chunker, method)):
                    logger.error(f"{class_name} missing required method: {method}")
                    return None

            return chunker

        except Exception as e:
            # If import fails due to Set or other issues, use default
            if 'Set' in str(e) and language == 'python':
                logger.warning("Python chunker failed to load due to typing issue, using default")
                logger.info("[UNTESTED PATH] Python chunker typing issue fallback")
                # Import directly, not through __init__
                from acolyte.rag.chunking.languages.default import DefaultChunker

                return DefaultChunker('python')
            logger.error(f"Failed to load chunker for {language}: {type(e).__name__}: {e}")
            return None

    @classmethod
    def get_supported_languages(cls) -> Dict[str, str]:
        """
        Get mapping of supported languages to their extensions.

        Returns:
            Dict mapping language names to example extensions
        """
        # Get all supported extensions from FileTypeDetector
        extensions = FileTypeDetector.get_all_supported_extensions()

        # Build language to extensions mapping
        languages = {}
        for ext in extensions:
            lang = FileTypeDetector.get_language(Path(f"test{ext}"))
            if lang != "unknown":
                if lang not in languages:
                    languages[lang] = []
                languages[lang].append(ext)

        return languages

    @classmethod
    def is_supported_file(cls, file_path: str) -> bool:
        """
        Check if file type is supported for chunking.

        Args:
            file_path: Path to check

        Returns:
            True if file can be chunked
        """
        path = Path(file_path)

        # First check FileTypeDetector
        if FileTypeDetector.is_supported(path):
            return True

        # Check special filenames from language config
        if path.name.lower() in LANGUAGE_CONFIG.get('special_files', {}):
            return True

        return False


# Convenience function
def get_chunker(file_path: str, content: Optional[str] = None) -> BaseChunker:
    """
    Get appropriate chunker for a file.

    Convenience wrapper around ChunkerFactory.create().

    Args:
        file_path: Path to the file
        content: Optional content for better detection

    Returns:
        Appropriate chunker instance
    """
    return ChunkerFactory.create(file_path, content)
