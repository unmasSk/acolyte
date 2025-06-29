"""
YAML chunker using tree-sitter-languages.
Handles configuration files with semantic understanding.
"""

from typing import Dict, List, Any, Optional
from tree_sitter_languages import get_language  # type: ignore
from acolyte.core.logging import logger
from acolyte.models.chunk import Chunk, ChunkType
from acolyte.rag.chunking.languages.config_base import ConfigChunkerBase


class YamlChunker(ConfigChunkerBase):
    """
    YAML-specific chunker using tree-sitter.

    YAML is different from programming languages - it's configuration/data.
    We chunk by semantic sections and extract configuration metadata.
    """

    def _get_language_name(self) -> str:
        """Return language identifier."""
        return 'yaml'

    def _get_tree_sitter_language(self) -> Any:
        """Get YAML language for tree-sitter."""
        return get_language('yaml')

    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """
        YAML-specific node types to chunk.

        YAML doesn't have functions/classes, but has:
        - document: Full YAML document
        - block_mapping_pair: Key-value pairs at root level
        - block_sequence: Arrays/lists

        We'll chunk by top-level sections for meaningful divisions.
        """
        return {
            # YAML doesn't map cleanly to our code-oriented ChunkTypes
            # We'll override _chunk() to handle YAML semantically
        }

    async def chunk(self, content: str, file_path: str) -> List[Chunk]:
        """
        Override to handle YAML's unique structure.

        Strategy:
        1. Parse with tree-sitter
        2. Identify top-level sections
        3. Create chunks for each major configuration block
        4. Extract metadata about configuration
        """
        # Parse the content
        tree = self.parser.parse(bytes(content, 'utf8'))

        if tree.root_node.has_error:
            logger.warning(f"Parse errors in {file_path}, falling back to line-based chunking")
            return self._chunk_by_lines(content, file_path)

        chunks: List[Chunk] = []
        lines = content.split('\n')

        # For YAML, we'll find top-level keys and chunk by those
        chunks = self._extract_yaml_sections(tree.root_node, lines, file_path)

        # Sort by start line
        chunks.sort(key=lambda c: c.metadata.start_line)

        # Validate chunks
        chunks = self._validate_chunks(chunks)

        return chunks

    def _extract_yaml_sections(
        self, root_node: Any, lines: List[str], file_path: str
    ) -> List[Chunk]:
        """Extract semantic sections from YAML."""
        chunks: List[Chunk] = []

        # The root node IS the stream, not a child of it
        if root_node.type == 'stream':
            # Process all documents in the stream
            for doc_node in root_node.children:
                if doc_node.type == 'document':
                    chunks.extend(self._process_document(doc_node, lines, file_path))
        elif root_node.type == 'document':
            # Direct document node
            chunks.extend(self._process_document(root_node, lines, file_path))

        return chunks

    def _process_document(self, doc_node: Any, lines: List[str], file_path: str) -> List[Chunk]:
        """Process a YAML document node."""
        chunks: List[Chunk] = []

        # Find the content node (usually block_node)
        content_node = None
        for child in doc_node.children:
            if child.type in ['block_node', 'block_mapping', 'block_sequence']:
                content_node = child
                break

        if not content_node:
            return chunks

        # Process based on content type
        if content_node.type == 'block_mapping':
            # Key-value pairs - chunk by top-level keys
            chunks.extend(self._chunk_block_mapping(content_node, lines, file_path))
        elif content_node.type == 'block_sequence':
            # Array at root - chunk the whole array
            chunk = self._create_chunk_from_range(
                content_node, lines, file_path, name="root_array", chunk_type=ChunkType.CONSTANTS
            )
            if chunk:
                chunks.append(chunk)
        elif content_node.type == 'block_node':
            # Look inside block_node for the actual content
            for child in content_node.children:
                if child.type == 'block_mapping':
                    chunks.extend(self._chunk_block_mapping(child, lines, file_path))
                elif child.type == 'block_sequence':
                    chunk = self._create_chunk_from_range(
                        child, lines, file_path, name="root_array", chunk_type=ChunkType.CONSTANTS
                    )
                    if chunk:
                        chunks.append(chunk)

        return chunks

    def _chunk_block_mapping(self, mapping_node, lines: List[str], file_path: str) -> List[Chunk]:
        """Chunk a block mapping by top-level keys."""
        chunks = []

        for child in mapping_node.children:
            if child.type == 'block_mapping_pair':
                # Extract the key and create a chunk for this section
                key_node = self._find_key_node(child)
                if key_node:
                    key_name = key_node.text.decode('utf8').strip(':').strip()

                    # Determine chunk type based on key name
                    chunk_type = self._determine_chunk_type(key_name)

                    chunk = self._create_chunk_from_range(
                        child, lines, file_path, name=key_name, chunk_type=chunk_type
                    )

                    if chunk:
                        # Add YAML-specific metadata
                        chunk.metadata.language_specific = self._extract_yaml_metadata(
                            child, lines, key_name
                        )
                        chunks.append(chunk)

        return chunks

    def _find_key_node(self, mapping_pair_node):
        """Find the key node in a mapping pair."""
        for child in mapping_pair_node.children:
            if child.type in [
                'flow_node',
                'plain_scalar',
                'single_quoted_scalar',
                'double_quoted_scalar',
                'block_scalar',
            ]:
                return child
        return None

    def _determine_chunk_type(self, key_name: str) -> ChunkType:
        """Determine appropriate ChunkType based on YAML key."""
        key_lower = key_name.lower()

        # Common configuration patterns
        if any(pattern in key_lower for pattern in ['import', 'include', 'require']):
            return ChunkType.IMPORTS
        elif any(pattern in key_lower for pattern in ['const', 'env', 'config', 'settings']):
            return ChunkType.CONSTANTS
        elif any(pattern in key_lower for pattern in ['type', 'schema', 'model']):
            return ChunkType.TYPES
        elif any(pattern in key_lower for pattern in ['test', 'spec']):
            return ChunkType.TESTS
        else:
            # Most YAML content is configuration
            return ChunkType.MODULE

    def _create_chunk_from_range(
        self, node, lines: List[str], file_path: str, name: str, chunk_type: ChunkType
    ) -> Optional[Chunk]:
        """Create a chunk from a node range."""
        start_line = node.start_point[0]
        end_line = node.end_point[0]

        # Include any leading comments
        while start_line > 0:
            prev_line = lines[start_line - 1].strip()
            if prev_line.startswith('#') or not prev_line:
                start_line -= 1
            else:
                break

        # Extract content
        content = '\n'.join(lines[start_line : end_line + 1])

        return self._create_chunk(
            content=content,
            chunk_type=chunk_type,
            file_path=file_path,
            start_line=start_line + 1,
            end_line=end_line + 1,
            name=name,
        )

    def _extract_yaml_metadata(self, node, lines: List[str], key_name: str) -> Dict[str, Any]:
        """Extract YAML-specific metadata."""
        metadata = {
            'key_name': key_name,
            'structure_type': 'unknown',  # mapping, sequence, scalar
            'depth': 0,
            'has_comments': False,
            'todos': [],
            'references': [],  # Anchors and aliases
            'complexity': {
                'nesting_depth': 0,
                'child_count': 0,
                'total_lines': node.end_point[0] - node.start_point[0] + 1,
            },
        }

        # Analyze structure
        metadata.update(self._analyze_structure(node))

        # Extract TODOs from comments
        start_line = node.start_point[0]
        end_line = node.end_point[0]

        for i in range(max(0, start_line - 5), min(len(lines), end_line + 1)):
            line = lines[i]
            if '#' in line:
                metadata['has_comments'] = True
                comment = line[line.index('#') :].strip('#').strip()

                # Check for TODOs
                for marker in ['TODO', 'FIXME', 'HACK', 'NOTE', 'WARNING']:
                    if marker in comment.upper():
                        # Get todos list safely
                        todos_value = metadata.get('todos')
                        if isinstance(todos_value, list):
                            todos_value.append({'type': marker, 'text': comment, 'line': i + 1})

        # Check for common patterns
        metadata['patterns'] = self._detect_yaml_patterns(node, lines, key_name)

        return metadata

    def _analyze_structure(self, node: Any, depth: int = 0) -> Dict[str, Any]:
        """Analyze the structure of a YAML node."""
        result = {
            'structure_type': 'scalar',
            'complexity': {'nesting_depth': depth, 'child_count': 0},
        }

        # Find the value node
        value_node = None
        for child in node.children:
            if child.type == ':':
                # After colon is the value
                idx = node.children.index(child)
                if idx + 1 < len(node.children):
                    value_node = node.children[idx + 1]
                    break

        if not value_node:
            return result

        # Determine structure type
        if value_node.type == 'block_mapping':
            result['structure_type'] = 'mapping'
            complexity_dict = result.get('complexity', {})
            if isinstance(complexity_dict, dict):
                complexity_dict['child_count'] = len(
                    [c for c in value_node.children if c.type == 'block_mapping_pair']
                )
                # Recurse for depth
                max_child_depth = depth
                for child in value_node.children:
                    if child.type == 'block_mapping_pair':
                        child_result = self._analyze_structure(child, depth + 1)
                        child_complexity = child_result.get('complexity', {})
                        if isinstance(child_complexity, dict):
                            max_child_depth = max(
                                max_child_depth, child_complexity.get('nesting_depth', 0)
                            )
                complexity_dict['nesting_depth'] = max_child_depth

        elif value_node.type == 'block_sequence':
            result['structure_type'] = 'sequence'
            complexity_dict = result.get('complexity', {})
            if isinstance(complexity_dict, dict):
                complexity_dict['child_count'] = len(
                    [c for c in value_node.children if c.type == 'block_sequence_item']
                )

        elif value_node.type in [
            'plain_scalar',
            'single_quoted_scalar',
            'double_quoted_scalar',
            'block_scalar',
        ]:
            result['structure_type'] = 'scalar'

        return result

    def _detect_yaml_patterns(self, node, lines: List[str], key_name: str) -> Dict[str, List[str]]:
        """Detect common YAML patterns and potential issues."""
        patterns: Dict[str, List[str]] = {'configuration': [], 'security': [], 'quality': []}

        content = node.text.decode('utf8')

        # Configuration patterns
        config_list = patterns.get('configuration', [])
        if isinstance(config_list, list):
            if any(
                pattern in key_name.lower()
                for pattern in ['database', 'db', 'connection', 'datasource']
            ):
                config_list.append('database_config')

            if any(
                pattern in key_name.lower() for pattern in ['api', 'endpoint', 'service', 'url']
            ):
                config_list.append('service_config')

            if any(
                pattern in key_name.lower()
                for pattern in ['docker', 'container', 'kubernetes', 'k8s']
            ):
                config_list.append('container_config')

        # Security issues
        security_list = patterns.get('security', [])
        if isinstance(security_list, list):
            if any(
                pattern in content.lower()
                for pattern in ['password:', 'secret:', 'token:', 'key:', 'api_key:']
            ):
                if not any(var in content for var in ['${', '{{', 'env.', 'ENV[']):
                    security_list.append('potential_hardcoded_secret')

            if 'localhost' in content or '127.0.0.1' in content:
                security_list.append('hardcoded_localhost')

        # Quality indicators
        quality_list = patterns.get('quality', [])
        if isinstance(quality_list, list):
            if '${' in content or '{{' in content:
                quality_list.append('uses_variables')

            if any(ref in content for ref in ['&', '*']):
                quality_list.append('uses_yaml_references')

            if content.count('\n') > 50:
                logger.info("[UNTESTED PATH] yaml large configuration block")
                quality_list.append('large_configuration_block')

        return patterns
