"""
Dockerfile chunker using tree-sitter-languages.
Handles Docker-specific constructs and multi-stage builds.
"""

from typing import Dict, List, Optional, Any
from tree_sitter_languages import get_language

from acolyte.models.chunk import ChunkType
from acolyte.core.logging import logger
from acolyte.rag.chunking.base import BaseChunker
from acolyte.rag.chunking.mixins import PatternDetectionMixin


class DockerfileChunker(BaseChunker, PatternDetectionMixin):
    """
    Dockerfile-specific chunker using tree-sitter.

    Handles:
    - Multi-stage builds (multiple FROM statements)
    - Build arguments and environment variables
    - Layer optimization patterns
    - Comments and documentation
    """

    def _get_language_name(self) -> str:
        """Return language identifier."""
        return 'dockerfile'

    def _get_import_node_types(self) -> List[str]:
        """Get node types that represent imports for dockerfile."""
        return ['from_instruction']

    def _is_comment_node(self, node: Any) -> bool:
        """Check if node is a comment."""
        return node.type in ['comment', 'line_comment', 'block_comment']

    def _get_tree_sitter_language(self) -> Any:
        """Get Dockerfile language for tree-sitter."""
        logger.info("[UNTESTED PATH] dockerfile._get_tree_sitter_language called")
        return get_language('dockerfile')

    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """
        Dockerfile-specific node types to chunk.

        Tree-sitter Dockerfile nodes mapped to ChunkTypes.
        """
        return {
            # Each FROM starts a new build stage
            'from_instruction': ChunkType.MODULE,
            # Commands that modify the image
            'run_instruction': ChunkType.FUNCTION,
            'copy_instruction': ChunkType.FUNCTION,
            'add_instruction': ChunkType.FUNCTION,
            # Configuration
            'env_instruction': ChunkType.CONSTANTS,
            'arg_instruction': ChunkType.CONSTANTS,
            # Metadata
            'label_instruction': ChunkType.DOCSTRING,
            'maintainer_instruction': ChunkType.DOCSTRING,
            # Entry points
            'cmd_instruction': ChunkType.FUNCTION,
            'entrypoint_instruction': ChunkType.FUNCTION,
            # Other important instructions
            'expose_instruction': ChunkType.INTERFACE,
            'volume_instruction': ChunkType.INTERFACE,
            'workdir_instruction': ChunkType.MODULE,
            'user_instruction': ChunkType.MODULE,
        }

    def _create_chunk_from_node(
        self,
        node: Any,
        lines: List[str],
        file_path: str,
        chunk_type: ChunkType,
        processed_ranges: Dict[str, Any],
    ) -> Optional[Any]:
        """Override to handle Dockerfile-specific cases."""
        # For FROM instructions, try to include the entire stage
        if node.type == 'from_instruction':
            chunk = self._create_stage_chunk(node, lines, file_path, processed_ranges)
            if chunk:
                return chunk

        # For RUN instructions, group consecutive RUN commands
        elif node.type == 'run_instruction':
            chunk = self._create_run_chunk(node, lines, file_path, processed_ranges)
            if chunk:
                return chunk

        # Standard processing for other nodes
        chunk = super()._create_chunk_from_node(
            node, lines, file_path, chunk_type, processed_ranges
        )

        # Add Dockerfile-specific metadata
        if chunk:
            chunk.metadata.language_specific = self._extract_instruction_metadata(node)

        return chunk

    def _create_stage_chunk(
        self, from_node: Any, lines: List[str], file_path: str, processed_ranges: Dict[str, Any]
    ) -> Optional[Any]:
        """Create chunk for entire build stage starting with FROM."""
        start_line = from_node.start_point[0]

        # Find the next FROM instruction or end of file
        end_line = len(lines) - 1

        # Use tree-sitter to find next FROM instruction
        sibling = from_node.next_named_sibling
        while sibling:
            if sibling.type == 'from_instruction':
                end_line = sibling.start_point[0] - 1
                break
            sibling = sibling.next_named_sibling

        # Check if already processed
        if any(
            start_line <= s <= end_line or start_line <= e <= end_line
            for ranges in processed_ranges.values()
            for (s, e) in ranges
        ):
            return None

        # Include leading comments
        while start_line > 0 and lines[start_line - 1].strip().startswith('#'):
            logger.info("[UNTESTED PATH] dockerfile including leading comments in stage")
            start_line -= 1

        content = '\n'.join(lines[start_line : end_line + 1])

        # Extract stage name if present
        stage_name = self._extract_stage_name(from_node)

        # Mark entire range as processed
        if 'processed' not in processed_ranges:
            processed_ranges['processed'] = set()
        processed_ranges['processed'].add((start_line, end_line))

        return self._create_chunk(
            content=content,
            chunk_type=ChunkType.MODULE,
            file_path=file_path,
            start_line=start_line + 1,
            end_line=end_line + 1,
            name=stage_name or "build_stage",
        )

    def _create_run_chunk(
        self, run_node: Any, lines: List[str], file_path: str, processed_ranges: Dict[str, Any]
    ) -> Optional[Any]:
        """Group consecutive RUN instructions into single chunk."""
        start_line = run_node.start_point[0]
        end_line = run_node.end_point[0]

        # Check if already processed
        if any(
            start_line <= s <= end_line or start_line <= e <= end_line
            for ranges in processed_ranges.values()
            for (s, e) in ranges
        ):
            return None

        # Look for consecutive RUN instructions using tree-sitter
        sibling = run_node.next_named_sibling
        while sibling:
            if sibling.type == 'run_instruction':
                # Include this RUN in our chunk
                end_line = sibling.end_point[0]
                sibling = sibling.next_named_sibling
            else:
                # Stop at first non-RUN instruction
                break

        # Include leading comments
        while start_line > 0 and lines[start_line - 1].strip().startswith('#'):
            start_line -= 1

        content = '\n'.join(lines[start_line : end_line + 1])

        # Mark as processed
        if 'processed' not in processed_ranges:
            processed_ranges['processed'] = set()
        processed_ranges['processed'].add((start_line, end_line))

        return self._create_chunk(
            content=content,
            chunk_type=ChunkType.FUNCTION,
            file_path=file_path,
            start_line=start_line + 1,
            end_line=end_line + 1,
            name="run_commands",
        )

    def _extract_stage_name(self, from_node: Any) -> Optional[str]:
        """Extract stage name from FROM instruction if present."""
        # Look for AS in the AST
        for child in from_node.children:
            if child.type == 'as':
                # Next sibling should be the stage name
                idx = from_node.children.index(child)
                if idx + 1 < len(from_node.children):
                    stage_name_node = from_node.children[idx + 1]
                    if stage_name_node.type == 'image_alias':
                        return str(stage_name_node.text.decode('utf8').strip())

        # Fallback to text parsing if AST doesn't have it
        text = from_node.text.decode('utf8')
        if ' AS ' in text.upper():
            parts = text.split(' AS ', 1)
            if len(parts) > 1:
                logger.info("[UNTESTED PATH] dockerfile stage name from text parsing")
                return str(parts[1].strip())
        return None

    def _extract_instruction_metadata(self, node: Any) -> Dict[str, Any]:
        """Extract Dockerfile-specific metadata from instruction."""
        metadata = {
            'instruction': node.type.replace('_instruction', '').upper(),
            'is_multi_line': '\n' in node.text.decode('utf8'),
        }

        text = node.text.decode('utf8')

        # Extract specific metadata based on instruction type
        if node.type == 'from_instruction':
            # Extract base image
            parts = text.split()
            if len(parts) > 1:
                metadata['base_image'] = parts[1]
                # Check if it has a stage name
                if ' AS ' in text.upper():
                    metadata['stage_name'] = text.split(' AS ', 1)[1].strip()

        elif node.type == 'expose_instruction':
            # Extract ports
            ports = []
            parts = text.split()[1:]  # Skip EXPOSE
            for part in parts:
                if '/' in part:
                    ports.append(part)  # port/protocol
                else:
                    ports.append(f"{part}/tcp")  # default to tcp
            metadata['ports'] = ports

        elif node.type == 'env_instruction':
            # Extract environment variables
            env_vars = {}
            # Simple parsing, could be improved
            parts = text.split(None, 1)[1] if ' ' in text else ''
            if '=' in parts:
                key, value = parts.split('=', 1)
                env_vars[key.strip()] = value.strip().strip('"\'')
            metadata['env_vars'] = env_vars

        elif node.type in ['cmd_instruction', 'entrypoint_instruction']:
            # Check format (exec vs shell)
            if '[' in text and ']' in text:
                metadata['format'] = 'exec'
            else:
                metadata['format'] = 'shell'

        elif node.type == 'copy_instruction' or node.type == 'add_instruction':
            # Check if it has --from flag (multi-stage copy)
            if '--from=' in text:
                metadata['multi_stage_copy'] = True
                # Extract source stage
                import re

                match = re.search(r'--from=(\S+)', text)
                if match:
                    logger.info("[UNTESTED PATH] dockerfile multi-stage copy source")
                    metadata['source_stage'] = match.group(1)

        return metadata

    def _extract_dependencies_from_imports(self, import_nodes: List[Any]) -> List[str]:
        """
        Dockerfiles don't have imports, but we can extract base images as dependencies.
        """
        deps = set()

        for node in import_nodes:
            if node.type == 'from_instruction':
                text = node.text.decode('utf8')
                parts = text.split()
                if len(parts) > 1:
                    base_image = parts[1]
                    # Remove tag/digest to get base image name
                    if ':' in base_image:
                        base_image = base_image.split(':')[0]
                    if '@' in base_image:
                        base_image = base_image.split('@')[0]
                    deps.add(base_image)

        return sorted(list(deps))
