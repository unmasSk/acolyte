"""
Language-specific chunkers for ACOLYTE.
Each chunker understands the syntax and semantics of its target language.
"""

from typing import Dict, Type, Optional
from acolyte.rag.chunking.base import BaseChunker

# Registry will be populated as chunkers are implemented
LANGUAGE_CHUNKERS: Dict[str, Type[BaseChunker]] = {}

# Import implemented chunkers
try:
    from .python import PythonChunker

    LANGUAGE_CHUNKERS['python'] = PythonChunker
except ImportError:  # pragma: no cover
    pass

try:
    from .makefile import MakefileChunker

    LANGUAGE_CHUNKERS['makefile'] = MakefileChunker
except ImportError:  # pragma: no cover
    pass

try:
    from .dockerfile import DockerfileChunker

    LANGUAGE_CHUNKERS['dockerfile'] = DockerfileChunker
except ImportError:  # pragma: no cover
    pass

try:
    from .perl import PerlChunker

    LANGUAGE_CHUNKERS['perl'] = PerlChunker
except ImportError:  # pragma: no cover
    pass

try:
    from .elisp import ElispChunker

    LANGUAGE_CHUNKERS['elisp'] = ElispChunker
except ImportError:  # pragma: no cover
    pass

try:
    from .vim import VimChunker

    LANGUAGE_CHUNKERS['vimscript'] = VimChunker
except ImportError:  # pragma: no cover
    pass

try:
    from .lua import LuaChunker

    LANGUAGE_CHUNKERS['lua'] = LuaChunker
except ImportError:  # pragma: no cover
    pass

try:
    from .r import RChunker

    LANGUAGE_CHUNKERS['r'] = RChunker
except ImportError:  # pragma: no cover
    pass

try:
    from .xml import XmlChunker

    LANGUAGE_CHUNKERS['xml'] = XmlChunker
except ImportError:  # pragma: no cover
    pass

try:
    from .bash import BashChunker

    LANGUAGE_CHUNKERS['bash'] = BashChunker
    # Shell variants all use BashChunker
    LANGUAGE_CHUNKERS['shell'] = BashChunker
    LANGUAGE_CHUNKERS['sh'] = BashChunker
    LANGUAGE_CHUNKERS['zsh'] = BashChunker
    LANGUAGE_CHUNKERS['fish'] = BashChunker
except ImportError:  # pragma: no cover
    pass

try:
    from .typescript import TypeScriptChunker

    LANGUAGE_CHUNKERS['typescript'] = TypeScriptChunker
    LANGUAGE_CHUNKERS['javascript'] = TypeScriptChunker  # TS handles JS too
except ImportError:  # pragma: no cover
    pass

try:
    from .java import JavaChunker

    LANGUAGE_CHUNKERS['java'] = JavaChunker
except ImportError:  # pragma: no cover
    pass

try:
    from .kotlin import KotlinChunker

    LANGUAGE_CHUNKERS['kotlin'] = KotlinChunker
except ImportError:  # pragma: no cover
    pass

try:
    from .markdown import MarkdownChunker

    LANGUAGE_CHUNKERS['markdown'] = MarkdownChunker
except ImportError:  # pragma: no cover
    pass

try:
    from .go import GoChunker

    LANGUAGE_CHUNKERS['go'] = GoChunker
except ImportError:  # pragma: no cover
    pass

try:
    from .rust import RustChunker

    LANGUAGE_CHUNKERS['rust'] = RustChunker
except ImportError:  # pragma: no cover
    pass

try:
    from .c import CChunker

    LANGUAGE_CHUNKERS['c'] = CChunker
except ImportError:  # pragma: no cover
    pass

try:
    from .cpp import CppChunker

    LANGUAGE_CHUNKERS['cpp'] = CppChunker
except ImportError:  # pragma: no cover
    pass

try:
    from .csharp import CSharpChunker

    LANGUAGE_CHUNKERS['csharp'] = CSharpChunker
except ImportError:  # pragma: no cover
    pass

try:
    from .ruby import RubyChunker

    LANGUAGE_CHUNKERS['ruby'] = RubyChunker
except ImportError:  # pragma: no cover
    pass

try:
    from .config_base import ConfigChunkerBase

    LANGUAGE_CHUNKERS['config'] = ConfigChunkerBase
except ImportError:  # pragma: no cover
    pass

try:
    from .default import DefaultChunker

    LANGUAGE_CHUNKERS['default'] = DefaultChunker
except ImportError:  # pragma: no cover
    pass

try:
    from .json import JSONChunker

    LANGUAGE_CHUNKERS['json'] = JSONChunker
except ImportError:  # pragma: no cover
    pass

try:
    from .yaml import YamlChunker

    LANGUAGE_CHUNKERS['yaml'] = YamlChunker
except ImportError:  # pragma: no cover
    pass

try:
    from .toml import TomlChunker

    LANGUAGE_CHUNKERS['toml'] = TomlChunker
except ImportError:  # pragma: no cover
    pass

try:
    from .ini import IniChunker

    LANGUAGE_CHUNKERS['ini'] = IniChunker
except ImportError:  # pragma: no cover
    pass

try:
    from .html import HtmlChunker

    LANGUAGE_CHUNKERS['html'] = HtmlChunker
except ImportError:  # pragma: no cover
    pass

try:
    from .css import CssChunker

    LANGUAGE_CHUNKERS['css'] = CssChunker
except ImportError:  # pragma: no cover
    pass

try:
    from .php import PhpChunker

    LANGUAGE_CHUNKERS['php'] = PhpChunker
except ImportError:  # pragma: no cover
    pass

try:
    from .swift import SwiftChunker

    LANGUAGE_CHUNKERS['swift'] = SwiftChunker
except ImportError:  # pragma: no cover
    pass

try:
    from .sql import SQLChunker

    LANGUAGE_CHUNKERS['sql'] = SQLChunker
except ImportError:  # pragma: no cover
    pass


def get_language_chunker(language: str) -> Optional[Type[BaseChunker]]:
    """
    Get chunker class for a specific language.

    Args:
        language: Language identifier (e.g., 'python', 'javascript')

    Returns:
        Chunker class or None if not implemented
    """
    return LANGUAGE_CHUNKERS.get(language.lower())


def register_chunker(language: str, chunker_class: Type[BaseChunker]) -> None:
    """
    Register a chunker for a language.

    Args:
        language: Language identifier
        chunker_class: Chunker class to register

    Raises:
        TypeError: If chunker_class is not a subclass of BaseChunker
    """
    if not issubclass(chunker_class, BaseChunker):
        raise TypeError(
            f"chunker_class must be a subclass of BaseChunker, " f"got {chunker_class.__name__}"
        )
    LANGUAGE_CHUNKERS[language.lower()] = chunker_class


__all__ = [
    'LANGUAGE_CHUNKERS',
    'get_language_chunker',
    'register_chunker',
]
