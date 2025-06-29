"""
TOML chunker using tree-sitter-languages.
Extracts configuration structure and metadata for search.
"""

from typing import Dict, List, Any, Optional
from tree_sitter_languages import get_language  # type: ignore

from acolyte.models.chunk import Chunk, ChunkType
from acolyte.rag.chunking.languages.config_base import ConfigChunkerBase


class TomlChunker(ConfigChunkerBase):
    """
    TOML-specific chunker using tree-sitter.

    TOML is configuration, not code, so we extract:
    - Tables/sections as main chunks
    - Important config values
    - TO-DOs and FIXMEs in comments
    """

    def _get_language_name(self) -> str:
        """Return language identifier."""
        return 'toml'

    def _get_tree_sitter_language(self) -> Any:
        """Get TOML language for tree-sitter."""
        return get_language('toml')

    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """
        TOML node types mapped to ChunkTypes.

        Since TOML is config, we use different ChunkTypes:
        - Tables → MODULE (logical sections)
        - Important pairs → CONSTANTS (configuration values)
        """
        return {
            # Tables are like modules/namespaces in TOML
            'table': ChunkType.MODULE,
            'table_array_element': ChunkType.MODULE,
            # We don't chunk individual pairs, but track them in metadata
            # BaseChunker will handle document-level chunking
        }

    def _create_chunk_from_node(
        self, node, lines: List[str], file_path: str, chunk_type: ChunkType, processed_ranges
    ) -> Optional["Chunk"]:
        """Override to extract TOML-specific metadata."""
        chunk = super()._create_chunk_from_node(
            node, lines, file_path, chunk_type, processed_ranges
        )

        if chunk:
            # Extract metadata for this table/section
            chunk.metadata.language_specific = self._extract_table_metadata(node, lines, chunk)

            # For TOML, the "name" should be the table path
            if node.type == 'table':
                table_name = self._extract_table_name(node)
                if table_name:
                    chunk.metadata.name = table_name

        return chunk

    def _extract_table_name(self, table_node) -> Optional[str]:
        """Extract the full table name like [tool.poetry.dependencies]."""
        for child in table_node.children:
            if child.type == 'table_header':
                # Get the dotted path
                parts = []
                for subchild in child.children:
                    if subchild.type == 'bare_key' or subchild.type == 'quoted_key':
                        parts.append(subchild.text.decode('utf8').strip('"\''))

                return '.'.join(parts) if parts else None
        return None

    def _extract_table_metadata(self, node, lines: List[str], chunk: "Chunk") -> Dict[str, Any]:
        """Extract metadata from a TOML table section."""
        metadata: Dict[str, Any] = {
            'table_type': 'standard' if node.type == 'table' else 'array',
            'keys': [],
            'has_subtables': False,
            'todos': [],
            'patterns': {'config': []},
            'complexity': {
                'key_count': 0,
                'max_nesting': 0,
                'array_count': 0,
                'inline_table_count': 0,
            },
        }

        # Extract all key-value pairs in this section
        self._analyze_table_content(
            node, lines, metadata, chunk.metadata.start_line - 1, chunk.metadata.end_line
        )

        # Detect common configuration patterns
        patterns_list = self._detect_config_patterns_toml(metadata['keys'])
        if isinstance(metadata['patterns'], dict) and 'config' in metadata['patterns']:
            metadata['patterns']['config'] = patterns_list

        # Extract TODOs from comments
        for i in range(chunk.metadata.start_line - 1, chunk.metadata.end_line):
            if i < len(lines):
                line = lines[i]
                if '#' in line:
                    comment = line[line.index('#') :].strip('#').strip()
                    for marker in ['TODO:', 'FIXME:', 'HACK:', 'NOTE:']:
                        if marker in comment.upper():
                            todos_list = metadata.get('todos', [])
                            if isinstance(todos_list, list):
                                todos_list.append(
                                    {'type': marker.rstrip(':'), 'text': comment, 'line': i + 1}
                                )

        return metadata

    def _analyze_table_content(
        self,
        node,
        lines: List[str],
        metadata: Dict[str, Any],
        start_line: int,
        end_line: int,
        depth: int = 0,
    ):
        """Recursively analyze table content for metadata."""
        metadata['complexity']['max_nesting'] = max(metadata['complexity']['max_nesting'], depth)

        for child in node.children:
            # Key-value pairs
            if child.type == 'pair':
                key_node = None
                value_node = None

                for subchild in child.children:
                    if subchild.type in ['bare_key', 'quoted_key']:
                        key_node = subchild
                    elif subchild.type != '=':
                        value_node = subchild

                if key_node:
                    key = key_node.text.decode('utf8').strip('"\'')
                    value_type = value_node.type if value_node else 'unknown'

                    keys_list = metadata.get('keys', [])
                    if isinstance(keys_list, list):
                        keys_list.append(
                            {'name': key, 'type': value_type, 'line': child.start_point[0] + 1}
                        )
                    metadata['complexity']['key_count'] += 1

                    # Count arrays and inline tables
                    if value_type == 'array':
                        metadata['complexity']['array_count'] += 1
                    elif value_type == 'inline_table':
                        metadata['complexity']['inline_table_count'] += 1

            # Nested tables
            elif child.type == 'table':
                metadata['has_subtables'] = True
                # Don't recurse into nested tables - they'll be separate chunks

            # Continue analyzing within current bounds
            elif child.start_point[0] >= start_line and child.end_point[0] < end_line:
                self._analyze_table_content(child, lines, metadata, start_line, end_line, depth + 1)

    def _detect_config_patterns_toml(self, keys: List[Dict[str, Any]]) -> List[str]:
        """Detect common configuration patterns in TOML."""
        patterns = []
        key_names = [k['name'].lower() for k in keys]

        # Database config
        if any(k in key_names for k in ['host', 'port', 'database', 'username']):
            patterns.append('database_config')

        # API/Service config
        if any(k in key_names for k in ['api_key', 'secret', 'token', 'endpoint']):
            patterns.append('api_config')

        # Server config
        if any(k in key_names for k in ['listen', 'bind', 'workers', 'timeout']):
            patterns.append('server_config')

        # Dependencies (poetry, cargo, etc)
        if any('version' in k for k in key_names):
            patterns.append('dependencies')

        # Build/CI config
        if any(k in key_names for k in ['script', 'command', 'build', 'test']):
            patterns.append('build_config')

        # Feature flags
        if any(k in key_names for k in ['enabled', 'feature', 'flag']):
            patterns.append('feature_flags')

        return patterns

    def _get_chunk_size(self) -> int:
        """TOML files should have smaller chunks since they're usually config."""
        # Try toml-specific size first
        size = self.config.get('indexing.chunk_sizes.toml', None)

        # Try config size as fallback
        if size is None:
            size = self.config.get('indexing.chunk_sizes.config', 50)

        # Ultimate fallback
        if size is None:
            size = 50  # Small chunks for config files

        return size
