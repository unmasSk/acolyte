"""
Language name mappings for tree-sitter-languages.

Maps our internal language names to the names expected by tree-sitter-languages.
Also tracks which languages are actually supported.
"""

# Map our names to tree-sitter-languages names
TREE_SITTER_LANGUAGE_MAP = {
    # Direct matches (no change needed)
    'python': 'python',
    'javascript': 'javascript',
    'typescript': 'typescript',
    'java': 'java',
    'ruby': 'ruby',
    'rust': 'rust',
    'c': 'c',
    'cpp': 'cpp',
    'go': 'go',
    'php': 'php',
    'kotlin': 'kotlin',
    'css': 'css',
    'html': 'html',
    'json': 'json',
    'yaml': 'yaml',
    'toml': 'toml',
    'markdown': 'markdown',
    'bash': 'bash',
    'lua': 'lua',
    'r': 'r',
    'haskell': 'haskell',
    'ocaml': 'ocaml',
    'scala': 'scala',
    'julia': 'julia',
    'erlang': 'erlang',
    'elixir': 'elixir',
    'elm': 'elm',
    # Name differences
    'csharp': 'c-sharp',
    'c_sharp': 'c-sharp',
    'cs': 'c-sharp',
    'makefile': 'make',
    'make': 'make',
    'elisp': 'elisp',
    'emacs-lisp': 'elisp',
    'sql': 'sql',
    'sqlite': 'sqlite',
    'perl': 'perl',
    'dockerfile': 'dockerfile',
    'shell': 'bash',
    'sh': 'bash',
    'zsh': 'bash',
    'fish': 'bash',
    # TSX/JSX handled by typescript/javascript
    'tsx': 'tsx',
    'jsx': 'javascript',  # JSX uses JS parser
    # SCSS/SASS/LESS - not directly supported, fallback to CSS
    'scss': 'css',
    'sass': 'css',
    'less': 'css',
}

# Languages that are NOT supported by tree-sitter-languages
# These will need to use fallback chunking
UNSUPPORTED_LANGUAGES = {
    'vim',
    'vimscript',
    'ini',
    'cfg',
    'conf',
    'config',
    'xml',
    'swift',
    'csharp',  # c-sharp doesn't work despite being listed
    'c_sharp',
    'cs',
    'objc',
    'objective-c',
    'properties',
    'gradle',
    'groovy',
    'cmake',
    'zig',
    'dart',
    'vue',
    'svelte',
}

# Languages we have chunkers for
IMPLEMENTED_CHUNKERS = {
    'python',
    'javascript',
    'typescript',
    'java',
    'ruby',
    'rust',
    'c',
    'cpp',
    'csharp',
    'go',
    'php',
    'kotlin',
    'css',
    'html',
    'json',
    'yaml',
    'toml',
    'markdown',
    'bash',
    'lua',
    'r',
    'sql',
    'perl',
    'dockerfile',
    'makefile',
    'elisp',
    'vim',
    'ini',
    'xml',
    'swift',
}


def get_tree_sitter_language_name(language: str) -> str:
    """
    Get the tree-sitter-languages name for a language.

    Args:
        language: Our internal language name

    Returns:
        The name expected by tree-sitter-languages

    Raises:
        ValueError: If the language is not supported
    """
    language_lower = language.lower()

    if language_lower in UNSUPPORTED_LANGUAGES:
        raise ValueError(f"Language '{language}' is not supported by tree-sitter-languages")

    if language_lower not in TREE_SITTER_LANGUAGE_MAP:
        raise ValueError(f"Unknown language: '{language}'")

    return TREE_SITTER_LANGUAGE_MAP[language_lower]


def is_language_supported(language: str) -> bool:
    """
    Check if a language is supported by tree-sitter-languages.

    Args:
        language: Language name to check

    Returns:
        True if supported, False otherwise
    """
    language_lower = language.lower()
    return (
        language_lower in TREE_SITTER_LANGUAGE_MAP and language_lower not in UNSUPPORTED_LANGUAGES
    )


def get_supported_languages() -> list[str]:
    """
    Get list of all languages supported by tree-sitter-languages.

    Returns:
        List of supported language names
    """
    return [lang for lang in TREE_SITTER_LANGUAGE_MAP.keys() if lang not in UNSUPPORTED_LANGUAGES]
