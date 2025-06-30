"""
JSON chunker using tree-sitter-languages.
Intelligent chunking for configuration and data files.
"""

from typing import Dict, List, Any, Optional
from tree_sitter_languages import get_language  # type: ignore

from acolyte.models.chunk import Chunk, ChunkType
from acolyte.core.logging import logger
from acolyte.rag.chunking.languages.config_base import ConfigChunkerBase


class JSONChunker(ConfigChunkerBase):
    """
    JSON-specific chunker using tree-sitter.

    Focuses on pragmatic extraction for config files and API schemas.
    """

    def _get_language_name(self) -> str:
        """Return language identifier."""
        return 'json'

    def _get_tree_sitter_language(self) -> Any:
        """Get JSON language for tree-sitter."""
        return get_language('json')

    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """
        JSON node types to chunk.

        For JSON, we focus on top-level objects and arrays that represent
        logical configuration sections.
        """
        return {
            # JSON doesn't have functions/classes, so we map to CONSTANTS
            'object': ChunkType.CONSTANTS,
            'array': ChunkType.CONSTANTS,
        }

    async def chunk(self, content: str, file_path: str) -> List[Chunk]:
        """
        Override to provide JSON-specific chunking strategy.

        For JSON files, we chunk based on:
        1. Known config file patterns (package.json, tsconfig.json, etc.)
        2. Top-level keys in objects
        3. Large arrays that might be data
        """
        # Detect config file type
        config_type = self._detect_config_type(file_path)

        # Parse with tree-sitter
        tree = self.parser.parse(bytes(content, 'utf8'))
        root = tree.root_node

        chunks = []

        # For known config files, chunk by logical sections
        if config_type:
            chunks = self._chunk_by_config_type(root, content, file_path, config_type)
        else:
            # For generic JSON, chunk by top-level structure
            chunks = self._chunk_generic_json(root, content, file_path)

        # Validate and enhance chunks
        chunks = self._validate_chunks(chunks)

        # Add metadata to all chunks
        for chunk in chunks:
            chunk.metadata.language_specific = self._extract_json_metadata(
                chunk.content, config_type
            )

        return chunks

    def _detect_config_type(self, file_path: str) -> Optional[str]:
        """Detect known config file types."""
        file_name = file_path.lower().split('/')[-1]

        config_types = {
            'package.json': 'npm_package',
            'tsconfig.json': 'typescript_config',
            'composer.json': 'composer_package',
            'cargo.toml': 'cargo_manifest',  # Actually TOML but similar structure
            '.eslintrc.json': 'eslint_config',
            'appsettings.json': 'dotnet_config',
            'launch.json': 'vscode_launch',
            'settings.json': 'vscode_settings',
            '.prettierrc.json': 'prettier_config',
            'manifest.json': 'manifest',
        }

        return config_types.get(file_name)

    def _chunk_by_config_type(
        self, root, content: str, file_path: str, config_type: str
    ) -> List[Chunk]:
        """Chunk known config files by logical sections."""
        chunks = []
        lines = content.split('\n')

        # Define sections for each config type
        sections = {
            'npm_package': ['scripts', 'dependencies', 'devDependencies', 'peerDependencies'],
            'typescript_config': ['compilerOptions', 'include', 'exclude', 'files'],
            'eslint_config': ['rules', 'extends', 'plugins', 'overrides'],
            # Add more as needed
        }

        # Get relevant sections for this config type
        target_sections = sections.get(config_type, [])

        # Parse and find sections
        if root.type == 'document' and root.children:
            json_root = root.children[0]  # First child is the actual JSON

            if json_root.type == 'object':
                # Create one chunk for the main metadata
                main_chunk = self._create_main_config_chunk(
                    json_root, lines, file_path, config_type
                )
                if main_chunk:
                    chunks.append(main_chunk)

                # Create chunks for each important section
                for pair in json_root.children:
                    if pair.type == 'pair':
                        key_node = self._get_pair_key(pair)
                        if key_node and key_node.text.decode('utf8').strip('"') in target_sections:
                            chunk = self._create_section_chunk(
                                pair, lines, file_path, key_node.text.decode('utf8').strip('"')
                            )
                            if chunk:
                                chunks.append(chunk)

        # If no specific sections found, fall back to generic
        if not chunks:
            chunks = self._chunk_generic_json(root, content, file_path)

        return chunks

    def _chunk_generic_json(self, root, content: str, file_path: str) -> List[Chunk]:
        """Generic JSON chunking for unknown files."""
        lines = content.split('\n')

        # For small files, return as single chunk
        if len(lines) < self.chunk_size:
            return [
                self._create_chunk(
                    content=content,
                    chunk_type=ChunkType.CONSTANTS,
                    file_path=file_path,
                    start_line=1,
                    end_line=len(lines),
                    name='json_data',
                )
            ]

        # For larger files, try to chunk by top-level keys
        chunks = []
        if root.type == 'document' and root.children:
            json_root = root.children[0]

            if json_root.type == 'object':
                # Chunk by top-level keys
                for pair in json_root.children:
                    if pair.type == 'pair':
                        start_line = pair.start_point[0]
                        end_line = pair.end_point[0]

                        # Only create chunk if it's substantial
                        if end_line - start_line >= 5:
                            key_node = self._get_pair_key(pair)
                            key_name = (
                                key_node.text.decode('utf8').strip('"') if key_node else 'unknown'
                            )

                            chunk_content = '\n'.join(lines[start_line : end_line + 1])
                            chunk = self._create_chunk(
                                content=chunk_content,
                                chunk_type=ChunkType.CONSTANTS,
                                file_path=file_path,
                                start_line=start_line + 1,
                                end_line=end_line + 1,
                                name=key_name,
                            )
                            chunks.append(chunk)

        # If still no chunks or very few, fall back to line-based
        if len(chunks) < 2:
            return self._chunk_by_lines(content, file_path, ChunkType.CONSTANTS)

        return chunks

    def _create_main_config_chunk(
        self, json_root, lines: List[str], file_path: str, config_type: str
    ):
        """Create chunk for main config metadata."""
        import json

        # Extract key metadata fields based on config type
        metadata_keys = {
            'npm_package': ['name', 'version', 'description', 'main', 'type', 'engines'],
            'typescript_config': ['extends', 'references'],
            'composer_package': ['name', 'type', 'description', 'require'],
        }

        relevant_keys = metadata_keys.get(config_type, [])

        # Parse the full JSON first
        full_content = '\n'.join(lines)
        try:
            full_data = json.loads(full_content)
        except json.JSONDecodeError:
            # If can't parse, fall back to returning None
            logger.info("[UNTESTED PATH] json parse error in main config chunk")
            return None

        # Extract only the relevant metadata
        metadata_obj = {}
        for key in relevant_keys:
            if key in full_data:
                metadata_obj[key] = full_data[key]

        if metadata_obj:
            # Serialize back to valid JSON
            content = json.dumps(metadata_obj, indent=2)

            return self._create_chunk(
                content=content,
                chunk_type=ChunkType.CONSTANTS,
                file_path=file_path,
                start_line=1,
                end_line=len(lines),
                name=f'{config_type}_metadata',
            )

        return None

    def _create_section_chunk(self, pair_node, lines: List[str], file_path: str, section_name: str):
        """Create chunk for a specific config section."""
        start_line = pair_node.start_point[0]
        end_line = pair_node.end_point[0]

        content = '\n'.join(lines[start_line : end_line + 1])

        return self._create_chunk(
            content=content,
            chunk_type=ChunkType.CONSTANTS,
            file_path=file_path,
            start_line=start_line + 1,
            end_line=end_line + 1,
            name=section_name,
        )

    def _get_pair_key(self, pair_node):
        """Extract key from a JSON pair node."""
        for child in pair_node.children:
            if child.type == 'string':
                return child
        logger.info("[UNTESTED PATH] json no string key found in pair")
        return None

    def _extract_json_metadata(self, content: str, config_type: Optional[str]) -> Dict[str, Any]:
        """Extract pragmatic metadata from JSON content."""
        import json

        metadata: Dict[str, Any] = {
            'config_type': config_type,
            'structure': {'type': 'unknown', 'depth': 0},
            'size_metrics': {
                'lines': content.count('\n') + 1,
                'keys': 0,
                'arrays': 0,
            },
            'patterns': [],
        }

        try:
            data = json.loads(content)

            # Analyze structure
            structure = self._analyze_structure(data)
            metadata['structure'] = structure

            # Count elements
            metadata['size_metrics']['keys'] = self._count_keys(data)
            metadata['size_metrics']['arrays'] = self._count_arrays(data)

            # Detect patterns
            metadata['patterns'] = self._detect_patterns(data, config_type)

            # Add common config metadata using inherited methods
            metadata['env_vars'] = self._extract_env_vars(content)
            metadata['secrets'] = self._detect_secrets(content)
            metadata['urls'] = self._extract_urls(content)
            metadata['paths'] = self._extract_paths(content)
            # TO-DOs would need to be extracted from original AST comments, not JSON content
            metadata['todos'] = []

            # Extract special fields for known configs
            if config_type == 'npm_package':
                metadata['package_info'] = {
                    'name': data.get('name', 'unknown'),
                    'version': data.get('version', 'unknown'),
                    'has_scripts': 'scripts' in data,
                    'has_dependencies': 'dependencies' in data,
                    'is_private': data.get('private', False),
                }
            elif config_type == 'typescript_config':
                compiler_opts = data.get('compilerOptions', {})
                metadata['ts_config'] = {
                    'target': compiler_opts.get('target', 'unknown'),
                    'module': compiler_opts.get('module', 'unknown'),
                    'strict': compiler_opts.get('strict', False),
                    'lib': compiler_opts.get('lib', []),
                }

        except json.JSONDecodeError:
            logger.info("[UNTESTED PATH] json decode error in metadata extraction")
            metadata['parse_error'] = True

        return metadata

    def _analyze_structure(self, data: Any, depth: int = 0) -> Dict[str, Any]:
        """Analyze JSON structure recursively."""
        if isinstance(data, dict):
            max_depth = depth
            for value in data.values():
                child_structure = self._analyze_structure(value, depth + 1)
                if isinstance(child_structure, dict):
                    max_depth = max(max_depth, child_structure.get('depth', depth))

            return {
                'type': 'object',
                'depth': max_depth,
                'key_count': len(data),
            }
        elif isinstance(data, list):
            max_depth = depth
            for item in data[:10]:  # Sample first 10 items
                child_structure = self._analyze_structure(item, depth + 1)
                if isinstance(child_structure, dict):
                    max_depth = max(max_depth, child_structure.get('depth', depth))

            return {
                'type': 'array',
                'depth': max_depth,
                'length': len(data),
            }
        else:
            return {
                'type': type(data).__name__,
                'depth': depth,
            }

    def _count_keys(self, data: Any) -> int:
        """Count total keys in nested structure."""
        if isinstance(data, dict):
            count = len(data)
            for value in data.values():
                count += self._count_keys(value)
            return count
        elif isinstance(data, list):
            return sum(self._count_keys(item) for item in data)
        return 0

    def _count_arrays(self, data: Any) -> int:
        """Count total arrays in nested structure."""
        if isinstance(data, dict):
            return sum(self._count_arrays(value) for value in data.values())
        elif isinstance(data, list):
            logger.info("[UNTESTED PATH] json counting nested arrays")
            return 1 + sum(self._count_arrays(item) for item in data)
        return 0

    def _detect_patterns(self, data: Any, config_type: Optional[str]) -> List[str]:
        """Detect common patterns in JSON data."""
        patterns = []

        if isinstance(data, dict):
            keys = set(data.keys())

            # API/Schema patterns
            if {'type', 'properties', 'required'}.issubset(keys):
                patterns.append('json_schema')
            if {'openapi', 'paths', 'components'}.intersection(keys):
                patterns.append('openapi_spec')
            if '$schema' in keys:
                patterns.append('schema_validated')

            # Config patterns
            if {'extends', 'rules', 'plugins'}.intersection(keys):
                patterns.append('linter_config')
            if {'scripts', 'dependencies', 'devDependencies'}.intersection(keys):
                patterns.append('package_manifest')
            if {'compilerOptions', 'include', 'exclude'}.intersection(keys):
                patterns.append('typescript_config')

            # Data patterns
            if all(isinstance(v, (dict, list)) for v in data.values()):
                patterns.append('nested_structure')
            if any('url' in str(k).lower() or 'uri' in str(k).lower() for k in keys):
                patterns.append('contains_urls')
            if any(
                'key' in str(k).lower() or 'token' in str(k).lower() or 'secret' in str(k).lower()
                for k in keys
            ):
                patterns.append('possible_secrets')

        return patterns
