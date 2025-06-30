"""
Base chunker using tree-sitter for ACOLYTE.
Provides uniform AST parsing across multiple languages.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Set, Any
from tree_sitter_languages import get_parser, get_language  # type: ignore

from acolyte.models.chunk import Chunk, ChunkType, ChunkMetadata
from acolyte.core.secure_config import Settings
from acolyte.core.logging import logger
from acolyte.core.id_generator import generate_id
from acolyte.core.token_counter import get_token_counter
from .mixins import ComplexityMixin, TodoExtractionMixin
from .language_mappings import get_tree_sitter_language_name, is_language_supported


class BaseChunker(ABC, ComplexityMixin, TodoExtractionMixin):
    """
    Base chunker using tree-sitter for consistent AST parsing.

    Advantages:
    - Uniform API across all languages
    - Concrete syntax tree (preserves all details)
    - Error recovery (parses even invalid code)
    - Incremental parsing support
    - Wide language support
    """

    def __init__(self):
        """Initialize with tree-sitter parser."""
        # Load config first
        self.config = Settings()
        self.chunk_size = self._get_chunk_size()
        self.overlap = self.config.get('indexing.overlap', 0.2)

        # Initialize tree-sitter using tree-sitter-languages
        language_name = self._get_language_name()
        self._tree_sitter_supported = False

        try:
            # Check if language is supported and get correct name
            if is_language_supported(language_name):
                ts_language_name = get_tree_sitter_language_name(language_name)
                self.parser = get_parser(ts_language_name)
                self.language = get_language(ts_language_name)
                self._tree_sitter_supported = True
            else:
                logger.warning(
                    f"Language '{language_name}' not supported by tree-sitter-languages, using fallback"
                )
                self.parser = None
                self.language = None
        except Exception as e:
            logger.error(f"Failed to load tree-sitter parser for {language_name}: {e}")
            self.parser = None
            self.language = None

        # Language-specific node types to chunk
        self.chunk_node_types = self._get_chunk_node_types()

        # Initialize token counter
        self.token_counter = get_token_counter()

    @abstractmethod
    def _get_language_name(self) -> str:
        """
        Get the language name for configuration lookup.

        Returns:
            Language identifier (e.g., 'python', 'javascript')
        """
        pass

    @abstractmethod
    def _get_tree_sitter_language(self) -> Any:
        """
        Get the tree-sitter language object.

        This method can return None if tree-sitter is not supported.
        BaseChunker will handle the fallback to line-based chunking.

        Returns:
            Tree-sitter Language object for this language or None
        """
        pass

    @abstractmethod
    def _get_import_node_types(self) -> List[str]:
        """
        Get node types that represent imports/includes for this language.

        Must be implemented by language-specific chunkers.

        Returns:
            List of node type names that represent imports
        """
        pass

    def _get_chunk_size(self) -> int:
        """Get chunk size for this language from config."""
        language = self._get_language_name()

        # Try language-specific size first
        size = self.config.get(f'indexing.chunk_sizes.{language}', None)

        # Fallback to default
        if size is None:
            size = self.config.get('indexing.chunk_sizes.default', 100)

        return size

    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """
        Get node types that should become chunks.

        Override in language-specific implementations for customization.

        Returns:
            Mapping from tree-sitter node types to ChunkTypes
        """
        # Common defaults that work for many languages
        return {
            'function_definition': ChunkType.FUNCTION,
            'function_declaration': ChunkType.FUNCTION,
            'method_definition': ChunkType.METHOD,
            'method_declaration': ChunkType.METHOD,
            'class_definition': ChunkType.CLASS,
            'class_declaration': ChunkType.CLASS,
            'interface_declaration': ChunkType.INTERFACE,
            'module': ChunkType.MODULE,
            'import_statement': ChunkType.IMPORTS,
            'import_declaration': ChunkType.IMPORTS,
        }

    async def chunk(self, content: str, file_path: str) -> List[Chunk]:
        """
        Chunk code using tree-sitter parsing.

        Strategy:
        1. Parse with tree-sitter
        2. Walk AST to find chunkable nodes
        3. Extract chunks with proper boundaries
        4. Handle imports specially
        5. Apply smart overlap
        """
        # If tree-sitter is not supported, use line-based fallback
        if not self._tree_sitter_supported or self.parser is None:
            logger.info(f"Using line-based chunking for {file_path} (tree-sitter not available)")
            return self._chunk_by_lines(content, file_path)

        # Parse the content
        tree = self.parser.parse(bytes(content, 'utf8'))

        if tree.root_node.has_error:
            logger.warning(f"Parse errors in {file_path}, but continuing with partial AST")

        chunks = []
        lines = content.split('\n')
        # Track processed ranges by node type to allow nested chunks
        processed_ranges: Dict[str, Set[Tuple[int, int]]] = {}

        # Extract imports first
        import_chunks = self._extract_imports(tree.root_node, lines, file_path, processed_ranges)
        chunks.extend(import_chunks)

        # Walk tree to find chunkable nodes
        def walk_tree(node, depth=0):
            """Recursively walk the AST."""
            # Check if this node type should be chunked
            node_type = node.type

            if node_type in self.chunk_node_types:
                chunk = self._create_chunk_from_node(
                    node, lines, file_path, self.chunk_node_types[node_type], processed_ranges
                )
                if chunk:
                    chunks.append(chunk)

            # Recurse into children
            for child in node.children:
                walk_tree(child, depth + 1)

        walk_tree(tree.root_node)

        # Handle any remaining code
        remaining_chunks = self._extract_remaining_code(
            tree.root_node, lines, file_path, processed_ranges
        )
        chunks.extend(remaining_chunks)

        # Sort by start line
        chunks.sort(key=lambda c: c.metadata.start_line)

        # Validate and add overlap
        chunks = self._validate_chunks(chunks)
        chunks = self._add_smart_overlap(chunks, preserve_imports=True)

        return chunks

    def _extract_imports(
        self,
        root_node,
        lines: List[str],
        file_path: str,
        processed_ranges: Dict[str, Set[Tuple[int, int]]],
    ) -> List[Chunk]:
        """Extract all import statements as chunks."""
        import_chunks = []
        import_nodes = []

        # Get language-specific import node types
        import_types = self._get_import_node_types()

        # Find all import nodes
        def find_imports(node):
            if node.type in import_types:
                import_nodes.append(node)
            for child in node.children:
                find_imports(child)

        find_imports(root_node)

        if not import_nodes:
            return []

        # Mark individual imports as processed to avoid duplication
        for node in import_nodes:
            start_line = int(node.start_point[0])
            end_line = int(node.end_point[0])
            if node.type not in processed_ranges:
                processed_ranges[node.type] = set()
            processed_ranges[node.type].add((start_line, end_line))

        # Group consecutive imports
        if import_nodes:
            first_import = import_nodes[0]
            last_import = import_nodes[-1]

            start_line = int(first_import.start_point[0])
            end_line = int(last_import.end_point[0])

            # Include any leading comments
            # Note: start_line is 0-based from tree-sitter
            while start_line > 0 and (
                lines[start_line - 1].strip().startswith('#')
                or lines[start_line - 1].strip().startswith('//')
                or not lines[start_line - 1].strip()
            ):
                start_line -= 1

            content = '\n'.join(lines[start_line : end_line + 1])

            # Extract dependencies from import nodes
            dependencies = []
            if hasattr(self, '_extract_dependencies_from_imports'):
                try:
                    dependencies = self._extract_dependencies_from_imports(import_nodes)
                except Exception as e:
                    logger.warning(f"Failed to extract dependencies: {e}")
                    dependencies = []

            chunk = self._create_chunk(
                content=content,
                chunk_type=ChunkType.IMPORTS,
                file_path=file_path,
                start_line=start_line + 1,
                end_line=end_line + 1,
                name='imports',
            )

            # Assign dependencies metadata if found
            if dependencies or True:  # Always set language_specific even if empty
                chunk.metadata.language_specific = {'dependencies': dependencies}

            # Mark as processed for imports
            if 'imports' not in processed_ranges:
                processed_ranges['imports'] = set()
            for line in range(start_line, end_line + 1):
                # Store as 0-based for consistency with tree-sitter
                processed_ranges['imports'].add((line, line))

            import_chunks.append(chunk)

        return import_chunks

    def _create_chunk_from_node(
        self,
        node,
        lines: List[str],
        file_path: str,
        chunk_type: ChunkType,
        processed_ranges: Dict[str, Set[Tuple[int, int]]],
    ) -> Optional[Chunk]:
        """Create a chunk from a tree-sitter node."""
        # Tree-sitter uses 0-based line numbers
        start_line = int(node.start_point[0])
        end_line = int(node.end_point[0])

        # Permitir chunks anidados de métodos en Ruby
        ruby_method_types = {'method', 'singleton_method'}
        if node.type in ruby_method_types and self._get_language_name() == 'ruby':
            # No marcar el rango como procesado para métodos Ruby
            pass
        else:
            # Check if this exact range was already processed for this node type
            node_type_ranges = processed_ranges.get(node.type, set())
            if (start_line, end_line) in node_type_ranges:
                logger.info("[UNTESTED PATH] Skipping already processed node range")
                return None

        # Include leading comments/decorators
        while start_line > 0:
            prev_line = lines[start_line - 1].strip()
            if (
                prev_line.startswith('#')
                or prev_line.startswith('//')
                or prev_line.startswith('/*')
                or prev_line.startswith('@')
                or prev_line.startswith('/**')
                or not prev_line
            ):
                start_line -= 1
            else:
                break

        # Extract content
        content = '\n'.join(lines[start_line : end_line + 1])

        # Get node name
        name = self._extract_node_name(node)

        # Marcar como procesado solo si no es método Ruby
        if not (node.type in ruby_method_types and self._get_language_name() == 'ruby'):
            if node.type not in processed_ranges:
                processed_ranges[node.type] = set()
            processed_ranges[node.type].add((start_line, end_line))

        return self._create_chunk(
            content=content,
            chunk_type=chunk_type,
            file_path=file_path,
            start_line=start_line + 1,
            end_line=end_line + 1,
            name=name,
        )

    def _extract_node_name(self, node) -> Optional[str]:
        """
        Extract name from various node types.

        Tree-sitter nodes often have an 'identifier' or 'name' child.
        """
        # Look for common name patterns
        for child in node.children:
            if child.type == 'identifier' or child.type == 'property_identifier':
                return child.text.decode('utf8')
            elif child.type == 'name':
                return child.text.decode('utf8')

        # For some nodes, the name might be nested
        if node.type in ['function_declaration', 'function_definition']:
            for child in node.children:
                if child.type == 'identifier':
                    return child.text.decode('utf8')
                # Check nested (e.g., in declarator)
                for grandchild in child.children:
                    if grandchild.type == 'identifier':
                        return grandchild.text.decode('utf8')

        return None

    def _extract_dependencies_from_imports(self, import_nodes) -> List[str]:
        """Extract dependency names from import nodes."""
        deps = set()

        for node in import_nodes:
            # Different languages have different import structures
            # This is a generic attempt - override in specific implementations
            text = node.text.decode('utf8')

            # Common patterns
            if 'from' in text and 'import' in text:
                # Python style: from X import Y
                parts = text.split('from')[1].split('import')[0].strip()
                deps.add(parts.split('.')[0])
            elif 'import' in text:
                # Simple import
                parts = text.replace('import', '').strip()
                # Handle various formats
                if ' as ' in parts:
                    parts = parts.split(' as ')[0]
                if ' from ' in parts:
                    parts = parts.split(' from ')[1].strip('"\'')
                deps.add(parts.split('.')[0].split('/')[0].strip('"\''))

        return list(deps)

    def _extract_remaining_code(
        self,
        root_node,
        lines: List[str],
        file_path: str,
        processed_ranges: Dict[str, Set[Tuple[int, int]]],
    ) -> List[Chunk]:
        """Extract any significant unprocessed code."""
        chunks = []
        unprocessed_lines = []

        for i, line in enumerate(lines):
            # Check if line is processed by any node type
            is_processed = False
            for node_ranges in processed_ranges.values():
                if any(s <= i <= e for (s, e) in node_ranges):
                    is_processed = True
                    break

            if not is_processed and line.strip():
                unprocessed_lines.append((i, line))

        if not unprocessed_lines:
            logger.info("[UNTESTED PATH] No unprocessed lines found for remaining code extraction")
            return chunks

        # Group consecutive unprocessed lines
        groups = []
        current_group = [unprocessed_lines[0]]

        for i in range(1, len(unprocessed_lines)):
            if unprocessed_lines[i][0] - unprocessed_lines[i - 1][0] <= 2:
                current_group.append(unprocessed_lines[i])
            else:
                if len(current_group) >= 5:  # Only keep significant groups
                    groups.append(current_group)
                current_group = [unprocessed_lines[i]]

        if len(current_group) >= 5:
            groups.append(current_group)

        # Create chunks from groups
        for group in groups:
            start_line = group[0][0]
            end_line = group[-1][0]

            content = '\n'.join(lines[start_line : end_line + 1])

            chunk = self._create_chunk(
                content=content,
                chunk_type=ChunkType.MODULE,
                file_path=file_path,
                start_line=start_line + 1,
                end_line=end_line + 1,
                name='module_code',
            )
            chunks.append(chunk)

        return chunks

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
            name: Optional name (function/class name)

        Returns:
            Chunk with complete metadata
        """
        # Line numbers are already 1-based from caller

        # Validate line numbers
        if start_line < 1:
            logger.warning(f"Invalid start_line {start_line}, setting to 1")
            logger.info("[UNTESTED PATH] Invalid start_line validation triggered")
            start_line = 1
        if end_line < start_line:
            logger.warning(f"Invalid end_line {end_line} < start_line {start_line}, adjusting")
            logger.info("[UNTESTED PATH] Invalid end_line validation triggered")
            end_line = start_line
        return Chunk(
            id=generate_id(),
            content=content,
            metadata=ChunkMetadata(
                chunk_type=chunk_type,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                language=self._get_language_name(),
                name=name,
            ),
        )

    def _chunk_by_lines(
        self, content: str, file_path: str, chunk_type: ChunkType = ChunkType.UNKNOWN
    ) -> List[Chunk]:
        """
        Fallback: simple line-based chunking with overlap.

        Used when tree-sitter parsing fails completely.

        Args:
            content: Content to chunk
            file_path: Source file path
            chunk_type: Type to assign to chunks

        Returns:
            List of line-based chunks
        """
        lines = content.split('\n')
        chunks = []

        overlap_lines = int(self.chunk_size * self.overlap)
        step = max(1, self.chunk_size - overlap_lines)

        for i in range(0, len(lines), step):
            chunk_lines = lines[i : i + self.chunk_size]
            if not chunk_lines or not ''.join(chunk_lines).strip():
                continue

            chunk_content = '\n'.join(chunk_lines)

            chunk = self._create_chunk(
                content=chunk_content,
                chunk_type=chunk_type,
                file_path=file_path,
                start_line=i + 1,
                end_line=min(i + len(chunk_lines), len(lines)),
            )
            chunks.append(chunk)

        return chunks

    def _add_smart_overlap(self, chunks: List[Chunk], preserve_imports: bool = True) -> List[Chunk]:
        """
        Add intelligent overlap between chunks.

        Args:
            chunks: List of chunks to enhance
            preserve_imports: Whether to preserve import context

        Returns:
            Chunks with smart overlap added
        """
        if len(chunks) <= 1:
            return chunks

        enhanced_chunks = [chunks[0]]

        for i in range(1, len(chunks)):
            current = chunks[i]
            previous = chunks[i - 1]

            overlap_content = self._generate_overlap_content(
                previous,
                current,
                preserve_imports and i >= 1,  # Include imports from second chunk onwards
            )

            if overlap_content:
                # Create new chunk with overlap prepended
                enhanced_content = f"{overlap_content}\n\n{current.content}"
                enhanced = Chunk(id=current.id, content=enhanced_content, metadata=current.metadata)
                enhanced_chunks.append(enhanced)
            else:
                enhanced_chunks.append(current)

        return enhanced_chunks

    def _generate_overlap_content(
        self, previous: Chunk, current: Chunk, include_imports: bool
    ) -> str:
        """Generate tree-sitter aware smart overlap."""
        overlap_parts = []

        # Include imports for non-import chunks
        if (
            include_imports
            and previous.metadata.chunk_type == ChunkType.IMPORTS
            and current.metadata.chunk_type != ChunkType.IMPORTS
        ):
            # Extract key imports
            import_lines = previous.content.split('\n')
            key_imports = []

            # Prioritize local/relative imports
            for line in import_lines:
                if any(pattern in line for pattern in ['./', '../', 'from .']):
                    key_imports.append(line)
                    if len(key_imports) >= 3:
                        break

            if key_imports:
                overlap_parts.append("# Key imports:")
                overlap_parts.extend(key_imports)
                overlap_parts.append("")

        # For methods, include class context
        if current.metadata.chunk_type == ChunkType.METHOD:
            lang_spec = current.metadata.language_specific or {}
            if 'class_name' in lang_spec:
                logger.info("[UNTESTED PATH] Generating overlap for method with class context")
                class_name = lang_spec['class_name']
                overlap_parts.append(f"# Method of class {class_name}")

        return '\n'.join(overlap_parts) if overlap_parts else ""

    def _validate_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Validate and filter chunks.

        Removes:
        - Empty chunks
        - Duplicate chunks
        - Too small chunks (configurable)

        Splits:
        - Chunks that exceed token limits

        Args:
            chunks: List of chunks to validate

        Returns:
            Validated chunks
        """
        validated = []
        seen_content = set()

        min_chunk_lines = self.config.get('indexing.min_chunk_lines', 5)
        max_chunk_tokens = self.config.get('indexing.max_chunk_tokens', 8000)

        for chunk in chunks:
            # Skip empty
            if not chunk.content.strip():
                logger.debug(f"Skipping empty chunk from {chunk.metadata.file_path}")
                continue

            # Skip too small - pero preserva tipos importantes
            lines = chunk.content.count('\n') + 1
            if lines < min_chunk_lines:
                # Always keep these chunk types regardless of size
                important_types = {
                    ChunkType.IMPORTS,
                    ChunkType.FUNCTION,
                    ChunkType.METHOD,
                    ChunkType.CONSTRUCTOR,
                    ChunkType.PROPERTY,
                    ChunkType.CLASS,
                    ChunkType.INTERFACE,
                    ChunkType.CONSTANTS,
                    ChunkType.TYPES,
                    ChunkType.MODULE,  # For config files
                    ChunkType.TESTS,  # Nunca filtrar TESTS por tamaño
                }

                # Also keep small chunks from config files
                config_extensions = {
                    '.yaml',
                    '.yml',
                    '.json',
                    '.toml',
                    '.ini',
                    '.env',
                    '.xml',
                    '.html',
                    '.htm',
                    '.xhtml',
                    '.svg',
                }
                if chunk.metadata.file_path.lower().endswith(tuple(config_extensions)):
                    # Config files often have important single-line entries
                    pass  # Don't skip
                elif chunk.metadata.chunk_type not in important_types:
                    logger.debug(
                        f"Skipping small chunk ({lines} lines) from {chunk.metadata.file_path}"
                    )
                    continue

            # Skip duplicates
            content_hash = hash(chunk.content.strip())
            if content_hash in seen_content:
                logger.debug(f"Skipping duplicate chunk from {chunk.metadata.file_path}")
                continue

            seen_content.add(content_hash)

            # Check token size
            token_count = self.token_counter.count_tokens(chunk.content)

            if token_count > max_chunk_tokens:
                # Split large chunks
                logger.warning(
                    f"Chunk too large: {token_count} tokens, splitting",
                    file_path=chunk.metadata.file_path,
                    chunk_type=chunk.metadata.chunk_type,
                    start_line=chunk.metadata.start_line,
                )

                # Split the chunk into smaller pieces
                split_chunks = self._split_large_chunk(chunk, max_chunk_tokens)
                validated.extend(split_chunks)
            else:
                validated.append(chunk)

        return validated

    def _split_large_chunk(self, chunk: Chunk, max_tokens: int) -> List[Chunk]:
        """
        Split a large chunk into smaller pieces while preserving structure.

        Args:
            chunk: Large chunk to split
            max_tokens: Maximum tokens per chunk

        Returns:
            List of smaller chunks
        """
        split_chunks = []
        content = chunk.content
        lines = content.split('\n')

        # Try to split at natural boundaries (empty lines, function boundaries)
        current_content = []
        current_tokens = 0
        part_number = 1

        for i, line in enumerate(lines):
            line_tokens = self.token_counter.count_tokens(line + '\n')

            # Check if adding this line would exceed limit
            if current_tokens + line_tokens > max_tokens and current_content:
                # Create a chunk from current content
                chunk_content = '\n'.join(current_content)

                # Calculate actual line numbers
                start_line = chunk.metadata.start_line + (i - len(current_content))
                end_line = chunk.metadata.start_line + i - 1

                split_chunk = self._create_chunk(
                    content=chunk_content,
                    chunk_type=chunk.metadata.chunk_type,
                    file_path=chunk.metadata.file_path,
                    start_line=start_line,
                    end_line=end_line,
                    name=(
                        f"{chunk.metadata.name or 'chunk'}_part{part_number}"
                        if chunk.metadata.name
                        else None
                    ),
                )
                split_chunks.append(split_chunk)

                # Reset for next chunk
                current_content = [line]
                current_tokens = line_tokens
                part_number += 1
            else:
                current_content.append(line)
                current_tokens += line_tokens

        # Handle remaining content
        if current_content:
            logger.info("[UNTESTED PATH] Handling remaining content in large chunk split")
            chunk_content = '\n'.join(current_content)

            # Calculate line numbers for last chunk
            start_line = chunk.metadata.start_line + (len(lines) - len(current_content))
            end_line = chunk.metadata.end_line

            split_chunk = self._create_chunk(
                content=chunk_content,
                chunk_type=chunk.metadata.chunk_type,
                file_path=chunk.metadata.file_path,
                start_line=start_line,
                end_line=end_line,
                name=(
                    f"{chunk.metadata.name or 'chunk'}_part{part_number}"
                    if chunk.metadata.name
                    else None
                ),
            )
            split_chunks.append(split_chunk)

        logger.info(
            f"Split large chunk into {len(split_chunks)} parts",
            original_tokens=self.token_counter.count_tokens(content),
            max_tokens=max_tokens,
        )

        return split_chunks


class LanguageChunker(BaseChunker):
    """
    Base class for language-specific chunkers with required method implementations.

    This intermediate class ensures all language chunkers implement the necessary
    methods for proper functionality.
    """

    @abstractmethod
    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """
        Get node types that should become chunks for this language.

        Each language must define its specific node type mappings.

        Returns:
            Mapping from tree-sitter node types to ChunkTypes
        """
        pass

    @abstractmethod
    def _extract_dependencies_from_imports(self, import_nodes) -> List[str]:
        """
        Extract dependency names from import nodes for this language.

        Each language has different import syntax and must implement
        proper parsing.

        Args:
            import_nodes: List of tree-sitter nodes representing imports

        Returns:
            List of dependency names extracted from imports
        """
        pass

    @abstractmethod
    def _is_comment_node(self, node) -> bool:
        """
        Check if a node represents a comment in this language.

        Each language has different comment node types.

        Args:
            node: Tree-sitter node to check

        Returns:
            True if the node is a comment
        """
        pass

    # Optional hooks for language-specific behavior
    def _should_include_node(self, node) -> bool:
        """
        Override to filter nodes before chunking.

        Default implementation includes all nodes.

        Args:
            node: Tree-sitter node to check

        Returns:
            True if the node should be processed for chunking
        """
        return True

    def _post_process_chunk(self, chunk: Chunk) -> Optional[Chunk]:
        """
        Override for language-specific post-processing.

        Can be used to add metadata, filter chunks, or transform content.

        Args:
            chunk: Chunk to post-process

        Returns:
            Processed chunk or None to exclude it
        """
        return chunk

    def _enhance_chunk_metadata(self, chunk: Chunk, node) -> None:
        """
        Override to add language-specific metadata to chunks.

        This is called after basic chunk creation but before validation.

        Args:
            chunk: Chunk to enhance
            node: Tree-sitter node that generated this chunk
        """
        pass
