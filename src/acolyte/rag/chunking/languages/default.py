"""
Default chunker for files when language-specific chunkers are not available.
Uses heuristic pattern matching and structure detection for intelligent chunking.
"""

import re
from typing import Dict, List, Optional, Any, Tuple, Pattern, cast

from acolyte.models.chunk import Chunk, ChunkType, ChunkMetadata
from acolyte.core.logging import logger
from acolyte.core.secure_config import Settings
from acolyte.core.id_generator import generate_id
from acolyte.rag.chunking.base import BaseChunker


class DefaultChunker(BaseChunker):
    """
    Enhanced default chunker with heuristic pattern detection.

    Features:
    - Pattern-based function/class detection
    - Basic metadata extraction (TODOs, complexity hints)
    - Structure-aware chunking by indentation
    - Language-agnostic patterns for common constructs

    This is used when:
    1. Tree-sitter doesn't support the language
    2. Language-specific chunker doesn't exist
    3. As a fallback when parsing fails
    """

    # Common function patterns across languages
    FUNCTION_PATTERNS = [
        # Traditional function declarations
        (r'^\s*(def|function|func|fn|fun|sub|method)\s+(\w+)', 'function_name'),
        # C-style functions
        (
            r'^\s*(public|private|protected|static|virtual|override|async)?\s*\w+\s+(\w+)\s*\(',
            'c_style',
        ),
        # Arrow functions and lambdas
        (r'^\s*(?:const|let|var)?\s*(\w+)\s*=\s*(?:async\s*)?\(.*?\)\s*=>', 'arrow'),
        # Method-like patterns
        (r'^\s*(\w+)\s*:\s*(?:async\s*)?function\s*\(', 'method'),
    ]

    # Class/type patterns
    CLASS_PATTERNS = [
        (r'^\s*(class|struct|interface|trait|type|record)\s+(\w+)', 'class_name'),
        (r'^\s*(public|private|protected)?\s*class\s+(\w+)', 'java_style'),
        (r'^\s*type\s+(\w+)\s*=', 'type_alias'),
    ]

    # Import patterns
    IMPORT_PATTERNS = [
        r'^\s*import\s+',
        r'^\s*from\s+\S+\s+import',
        r'^\s*require\s*\(',
        r'^\s*include\s+',
        r'^\s*use\s+',
        r'^\s*#include\s*[<"]',
        r'^\s*using\s+',
        r'^\s*@import\s+',
        r'^\s*load\s*\(',
        r'^\s*source\s+',
    ]

    # Comment patterns for TODO extraction
    COMMENT_PATTERNS = [
        (r'^\s*#', '#'),  # Python, Ruby, Shell
        (r'^\s*//', '//'),  # C++, Java, JavaScript
        (r'^\s*/\*', '/*'),  # C-style block
        (r'^\s*--', '--'),  # SQL, Haskell, Lua
        (r'^\s*%', '%'),  # LaTeX, Matlab
        (r'^\s*;', ';'),  # Lisp, Assembly
        (r'^\s*"', '"'),  # Vim
        (r'^\s*\'', "'"),  # Some configs
    ]

    # TODO patterns
    TODO_KEYWORDS = ['TODO', 'FIXME', 'HACK', 'BUG', 'XXX', 'NOTE', 'OPTIMIZE', 'REFACTOR']

    def __init__(self, language: str = 'unknown'):
        """Initialize with specified language name."""
        self._language = language
        super().__init__()
        self.config = Settings()
        self.chunk_size = self._get_chunk_size()
        self.overlap = self.config.get('indexing.overlap', 0.2)

        # Mark that we don't support tree-sitter
        self._tree_sitter_supported = False
        self.parser = None
        self.language = None
        self.chunk_node_types = {}

        # Compile patterns for efficiency
        self._compiled_patterns: Dict[str, Any] = {
            'functions': [(re.compile(p, re.MULTILINE), n) for p, n in self.FUNCTION_PATTERNS],
            'classes': [(re.compile(p, re.MULTILINE), n) for p, n in self.CLASS_PATTERNS],
            'imports': [re.compile(p, re.MULTILINE) for p in self.IMPORT_PATTERNS],
            'comments': [(re.compile(p), marker) for p, marker in self.COMMENT_PATTERNS],
        }

    def _get_language_name(self) -> str:
        """Return the language name."""
        return self._language

    def _get_chunk_size(self) -> int:
        """Get chunk size for this language from config."""
        # Try language-specific size first
        size = self.config.get(f'indexing.chunk_sizes.{self._language}', None)

        # Fallback to default
        if size is None:
            size = self.config.get('indexing.chunk_sizes.default', 100)

        return int(size)

    def _get_tree_sitter_language(self) -> Any:
        """No tree-sitter support."""
        logger.info("[UNTESTED PATH] default._get_tree_sitter_language called")
        return None

    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """No node types for line-based chunking."""
        return {}

    def _get_import_node_types(self) -> List[str]:
        """No import node types."""
        logger.info("[UNTESTED PATH] default._get_import_node_types called")
        return []

    async def chunk(self, content: str, file_path: str) -> List[Chunk]:
        """Enhanced chunking with structure detection and metadata extraction."""
        logger.info(f"Using enhanced default chunker for {file_path} (language: {self._language})")

        chunks: List[Chunk] = []
        lines = content.split('\n')

        # Extract imports at the top
        import_end = self._find_import_section_end(lines)
        if import_end > 0:
            import_chunk = self._create_chunk(
                content='\n'.join(lines[:import_end]),
                chunk_type=ChunkType.IMPORTS,
                file_path=file_path,
                start_line=1,
                end_line=import_end,
                name='imports',
            )
            if import_chunk:
                import_chunk.metadata.language_specific = {
                    'import_count': len([line for line in lines[:import_end] if line.strip()])
                }
                chunks.append(import_chunk)

        # Detect and chunk structural blocks
        structural_chunks = self._extract_structural_chunks(
            lines[import_end:], file_path, start_offset=import_end
        )
        chunks.extend(structural_chunks)

        # Chunk any remaining content
        if structural_chunks:
            last_chunk_end = structural_chunks[-1].metadata.end_line
        else:
            last_chunk_end = import_end

        if last_chunk_end < len(lines):
            logger.info("[UNTESTED PATH] default chunking remaining content")
            remaining_chunks = self._chunk_remaining_content(
                lines[last_chunk_end:], file_path, start_offset=last_chunk_end
            )
            chunks.extend(remaining_chunks)

        return chunks

    def _find_import_section_end(self, lines: List[str]) -> int:
        """Find where imports end using heuristic patterns."""
        import_end = 0
        found_non_import = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Skip empty lines and comments
            if not stripped or self._is_comment_line(stripped):
                continue

            # Check if it's an import
            if self._is_import_line(stripped):
                import_end = i + 1
                if found_non_import:
                    # Imports after code - unusual but possible
                    break
            else:
                # Non-import, non-comment line
                if import_end > 0:
                    # We've seen imports and now hit code
                    break
                found_non_import = True

        return import_end

    def _is_import_line(self, line: str) -> bool:
        """Check if line matches import patterns."""
        import_patterns = cast(List[Pattern[str]], self._compiled_patterns['imports'])
        return any(pattern.match(line) for pattern in import_patterns)

    def _is_comment_line(self, line: str) -> bool:
        """Check if line is likely a comment."""
        stripped = line.strip()
        if not stripped:
            return False
        comment_patterns = cast(List[Tuple[Pattern[str], str]], self._compiled_patterns['comments'])
        return any(pattern.match(stripped) for pattern, _ in comment_patterns)

    def _extract_structural_chunks(
        self, lines: List[str], file_path: str, start_offset: int = 0
    ) -> List[Chunk]:
        """Extract chunks based on structural patterns (functions, classes)."""
        chunks: List[Chunk] = []
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Skip empty lines and comments
            if not stripped or self._is_comment_line(stripped):
                i += 1
                continue

            # Check for function pattern
            func_match = self._match_function_pattern(line)
            if func_match:
                logger.info("[UNTESTED PATH] default function pattern matched")
                chunk = self._extract_function_chunk(lines, i, file_path, start_offset, func_match)
                if chunk:
                    chunks.append(chunk)
                    i = chunk.metadata.end_line - start_offset
                    continue

            # Check for class pattern
            class_match = self._match_class_pattern(line)
            if class_match:
                chunk = self._extract_class_chunk(lines, i, file_path, start_offset, class_match)
                if chunk:
                    chunks.append(chunk)
                    i = chunk.metadata.end_line - start_offset
                    continue

            i += 1

        return chunks

    def _match_function_pattern(self, line: str) -> Optional[Tuple[str, str]]:
        """Match function patterns and return (name, pattern_type)."""
        function_patterns = cast(
            List[Tuple[Pattern[str], str]], self._compiled_patterns['functions']
        )
        for pattern, pattern_type in function_patterns:
            match = pattern.match(line)
            if match:
                # Try to extract function name from groups
                name = None
                for group in match.groups():
                    if group and group not in [
                        'def',
                        'function',
                        'func',
                        'fn',
                        'fun',
                        'sub',
                        'method',
                        'public',
                        'private',
                        'protected',
                        'static',
                        'virtual',
                        'override',
                        'async',
                    ]:
                        name = group
                        break
                return (name or 'anonymous', pattern_type)
        return None

    def _match_class_pattern(self, line: str) -> Optional[Tuple[str, str]]:
        """Match class patterns and return (name, pattern_type)."""
        class_patterns = cast(List[Tuple[Pattern[str], str]], self._compiled_patterns['classes'])
        for pattern, pattern_type in class_patterns:
            match = pattern.match(line)
            if match:
                # Extract class name from groups
                name = None
                for group in match.groups():
                    if group and group not in [
                        'class',
                        'struct',
                        'interface',
                        'trait',
                        'type',
                        'record',
                        'public',
                        'private',
                        'protected',
                    ]:
                        name = group
                        break
                return (name or 'anonymous', pattern_type)
        return None

    def _extract_function_chunk(
        self,
        lines: List[str],
        start_idx: int,
        file_path: str,
        start_offset: int,
        match_info: Tuple[str, str],
    ) -> Optional[Chunk]:
        """Extract a function chunk using indentation detection."""
        func_name, pattern_type = match_info

        # Find function boundaries using indentation
        end_idx = self._find_block_end(lines, start_idx)

        # Include any preceding comments/decorators
        actual_start = self._find_block_start_with_comments(lines, start_idx)

        content = '\n'.join(lines[actual_start : end_idx + 1])

        chunk = self._create_chunk(
            content=content,
            chunk_type=ChunkType.FUNCTION,
            file_path=file_path,
            start_line=actual_start + start_offset + 1,
            end_line=end_idx + start_offset + 1,
            name=func_name,
        )

        # Add metadata
        if chunk:
            metadata = self._extract_basic_metadata(content)
            metadata['detected_as'] = pattern_type
            metadata['has_parameters'] = '(' in lines[start_idx]
            chunk.metadata.language_specific = metadata

        return chunk

    def _extract_class_chunk(
        self,
        lines: List[str],
        start_idx: int,
        file_path: str,
        start_offset: int,
        match_info: Tuple[str, str],
    ) -> Optional[Chunk]:
        """Extract a class chunk using indentation detection."""
        class_name, pattern_type = match_info

        # Find class boundaries
        end_idx = self._find_block_end(lines, start_idx)

        # Include any preceding comments/decorators
        actual_start = self._find_block_start_with_comments(lines, start_idx)

        content = '\n'.join(lines[actual_start : end_idx + 1])

        chunk = self._create_chunk(
            content=content,
            chunk_type=ChunkType.CLASS,
            file_path=file_path,
            start_line=actual_start + start_offset + 1,
            end_line=end_idx + start_offset + 1,
            name=class_name,
        )

        # Add metadata
        if chunk:
            metadata = self._extract_basic_metadata(content)
            metadata['detected_as'] = pattern_type
            metadata['method_count'] = self._count_methods_heuristic(content)
            chunk.metadata.language_specific = metadata

        return chunk

    def _find_block_end(self, lines: List[str], start_idx: int) -> int:
        """Find end of code block using indentation heuristics."""
        if start_idx >= len(lines):
            return start_idx

        # Get base indentation
        base_indent = self._get_indentation(lines[start_idx])

        # Handle single-line definitions
        if start_idx + 1 < len(lines):
            next_line = lines[start_idx + 1]
            next_indent = self._get_indentation(next_line)
            if next_indent <= base_indent and next_line.strip():
                # Next line is not indented - probably single line
                return start_idx

        # Find where indentation returns to base level or less
        end_idx = start_idx
        inside_block = False

        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                continue

            current_indent = self._get_indentation(line)

            # Check if we're inside the block
            if current_indent > base_indent:
                inside_block = True
                end_idx = i
            elif inside_block and current_indent <= base_indent:
                # Block has ended
                break

        return end_idx

    def _find_block_start_with_comments(self, lines: List[str], def_idx: int) -> int:
        """Find start including preceding comments and decorators."""
        start_idx = def_idx

        # Look backwards for comments and decorators
        for i in range(def_idx - 1, -1, -1):
            line = lines[i].strip()

            # Stop at empty line unless it's between decorators
            if not line:
                # Check if we should continue looking
                found_decorator_above = False
                for j in range(i - 1, -1, -1):
                    if lines[j].strip():
                        if self._is_decorator_or_comment(lines[j].strip()):
                            found_decorator_above = True
                            logger.info("[UNTESTED PATH] default decorator found above")
                        break
                if not found_decorator_above:
                    logger.info("[UNTESTED PATH] default no decorator above")
                    break
                continue

            # Include comments and decorators
            if self._is_decorator_or_comment(line):
                logger.info("[UNTESTED PATH] default decorator/comment included")
                start_idx = i
            else:
                break

        return start_idx

    def _is_decorator_or_comment(self, line: str) -> bool:
        """Check if line is a decorator or comment."""
        if not line:
            return False

        # Common decorator patterns
        decorator_patterns = ['@', '#[', '<<', '[[']

        # Check comments
        if self._is_comment_line(line):
            return True

        # Check decorators
        return any(line.startswith(p) for p in decorator_patterns)

    def _get_indentation(self, line: str) -> int:
        """Get indentation level of a line."""
        return len(line) - len(line.lstrip())

    def _count_methods_heuristic(self, content: str) -> int:
        """Count probable methods in content."""
        count = 0
        function_patterns = cast(
            List[Tuple[Pattern[str], str]], self._compiled_patterns['functions']
        )
        for pattern, _ in function_patterns:
            count += len(pattern.findall(content))
        return count

    def _extract_basic_metadata(self, content: str) -> Dict[str, Any]:
        """Extract basic metadata without tree-sitter."""
        metadata = {
            'lines_of_code': len([line for line in content.split('\n') if line.strip()]),
            'todos': self._extract_todos_heuristic(content),
            'complexity_hint': self._calculate_complexity_hint(content),
            'has_comments': self._has_comments(content),
        }

        # Add pattern detection results
        patterns = self._detect_patterns_heuristic(content)
        if patterns:
            metadata['patterns'] = patterns

        return metadata

    def _extract_todos_heuristic(self, content: str) -> List[Dict[str, Any]]:
        """Extract TODOs from content using patterns."""
        todos: List[Dict[str, Any]] = []
        lines = content.split('\n')

        for i, line in enumerate(lines):
            # Check each TODO keyword
            for keyword in self.TODO_KEYWORDS:
                if keyword in line.upper():
                    # Extract the TODO text
                    pattern = rf'{keyword}[:\s]*(.*?)(?:\n|$)'
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        todo_text = match.group(1).strip()
                        # Clean comment markers
                        todo_text = re.sub(r'^[/*#\-\s]+', '', todo_text)
                        todo_text = re.sub(r'[*/\s\-]+$', '', todo_text)

                        todos.append(
                            {
                                'type': keyword,
                                'text': todo_text,
                                'line': i + 1,  # 1-indexed
                            }
                        )
                        break

        return todos

    def _calculate_complexity_hint(self, content: str) -> int:
        """Calculate rough complexity based on control structures."""
        complexity = 1  # Base complexity

        # Keywords that increase complexity
        control_keywords = [
            r'\bif\b',
            r'\belse\b',
            r'\belif\b',
            r'\belsif\b',
            r'\bfor\b',
            r'\bwhile\b',
            r'\bforeach\b',
            r'\bswitch\b',
            r'\bcase\b',
            r'\bwhen\b',
            r'\btry\b',
            r'\bcatch\b',
            r'\bexcept\b',
            r'\bfinally\b',
            r'\b\?\s*:',
            r'&&',
            r'\|\|',  # Ternary and logical operators
        ]

        for keyword in control_keywords:
            complexity += len(re.findall(keyword, content, re.IGNORECASE))

        return complexity

    def _has_comments(self, content: str) -> bool:
        """Check if content has comments."""
        for line in content.split('\n'):
            if self._is_comment_line(line):
                return True
        return False

    def _detect_patterns_heuristic(self, content: str) -> Dict[str, List[str]]:
        """Detect code patterns using heuristics."""
        patterns: Dict[str, List[str]] = {'anti': [], 'good': []}

        lines = content.split('\n')
        loc = len([line for line in lines if line.strip()])

        # Anti-patterns
        if loc > 100:
            logger.info("[UNTESTED PATH] default long block anti-pattern")
            patterns['anti'].append('long_block')

        # Deep nesting
        max_indent = max((self._get_indentation(line) for line in lines if line.strip()), default=0)
        if max_indent > 20:  # 5 levels with 4-space indent
            logger.info("[UNTESTED PATH] default deep nesting anti-pattern")
            patterns['anti'].append('deep_nesting')

        # Good patterns
        if self._has_comments(content):
            patterns['good'].append('documented')

        # Check for error handling
        if any(kw in content.lower() for kw in ['try', 'catch', 'except', 'error', 'err']):
            patterns['good'].append('error_handling')

        return {k: v for k, v in patterns.items() if v}

    def _chunk_remaining_content(
        self, lines: List[str], file_path: str, start_offset: int = 0
    ) -> List[Chunk]:
        """Chunk any remaining content that wasn't captured by patterns."""
        chunks: List[Chunk] = []

        # Filter out empty lines at start/end
        while lines and not lines[0].strip():
            logger.info("[UNTESTED PATH] default removing empty line at start")
            lines.pop(0)
            start_offset += 1
        while lines and not lines[-1].strip():
            logger.info("[UNTESTED PATH] default removing empty line at end")
            lines.pop()

        if not lines:
            return chunks

        # Use smart line chunking
        overlap_lines = int(self.chunk_size * self.overlap)
        step = max(1, self.chunk_size - overlap_lines)

        for i in range(0, len(lines), step):
            chunk_lines = lines[i : i + self.chunk_size]
            if not chunk_lines or not ''.join(chunk_lines).strip():
                continue

            content = '\n'.join(chunk_lines)

            # Try to determine chunk type from content
            chunk_type = self._guess_chunk_type(content)

            chunk = self._create_chunk(
                content=content,
                chunk_type=chunk_type,
                file_path=file_path,
                start_line=i + start_offset + 1,
                end_line=min(i + len(chunk_lines), len(lines)) + start_offset,
                name=f"{chunk_type.value}_chunk",
            )

            if chunk:
                chunk.metadata.language_specific = self._extract_basic_metadata(content)
                chunks.append(chunk)

        return chunks

    def _guess_chunk_type(self, content: str) -> ChunkType:
        """Guess chunk type from content patterns."""
        # Check for test patterns
        if any(pattern in content.lower() for pattern in ['test', 'spec', 'assert', 'expect']):
            return ChunkType.TESTS

        # Check for documentation
        comment_lines = sum(
            1 for line in content.split('\n') if self._is_comment_line(line.strip())
        )
        total_lines = len([line for line in content.split('\n') if line.strip()])
        if total_lines > 0 and comment_lines / total_lines > 0.5:
            return ChunkType.DOCSTRING

        # Check for configuration
        if any(
            pattern in content.lower()
            for pattern in ['config', 'settings', 'options', 'debug', 'api_key']
        ):
            config_score = content.count('=') + content.count(':')
            if config_score > 2:  # Lowered threshold
                return ChunkType.CONSTANTS

        # Default to module
        return ChunkType.MODULE

    def _create_chunk(
        self,
        content: str,
        chunk_type: ChunkType,
        file_path: str,
        start_line: int,
        end_line: int,
        name: Optional[str] = None,
    ) -> Chunk:
        """
        Helper to create chunks with consistent metadata.

        Args:
            content: Chunk content
            chunk_type: Type from ChunkType enum
            file_path: Source file path
            start_line: Starting line number (1-indexed)
            end_line: Ending line number (1-indexed)
            name: Optional name

        Returns:
            Chunk with complete metadata
        """
        # Validate line numbers
        if start_line < 1:
            logger.warning(f"Invalid start_line {start_line}, setting to 1")
            logger.info("[UNTESTED PATH] default invalid start_line")
            start_line = 1
        if end_line < start_line:
            logger.warning(f"Invalid end_line {end_line} < start_line {start_line}, adjusting")
            logger.info("[UNTESTED PATH] default invalid end_line")
            end_line = start_line

        return Chunk(
            id=generate_id(),
            content=content,
            metadata=ChunkMetadata(
                chunk_type=chunk_type,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                language=self._language,
                name=name,
            ),
        )

    def _extract_dependencies_from_imports(self, import_nodes: Any) -> List[str]:
        """No tree-sitter, so no imports to extract."""
        return []

    def _is_comment_node(self, node: Any) -> bool:
        """No tree-sitter nodes."""
        logger.info("[UNTESTED PATH] default._is_comment_node called")
        return False


# Convenience function
def get_default_chunker(language: str = 'unknown') -> DefaultChunker:
    """Get a default chunker for the specified language."""
    return DefaultChunker(language)
