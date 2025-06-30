"""
Bash/Shell chunker using tree-sitter-languages.
Handles sh, bash, zsh, fish shell scripts.
"""

from typing import Dict, List, Optional, Any, Set, Tuple, cast
from tree_sitter_languages import get_language

from acolyte.models.chunk import ChunkType, Chunk
from acolyte.core.logging import logger
from acolyte.rag.chunking.base import BaseChunker


class BashChunker(BaseChunker):
    """
    Shell script chunker using tree-sitter.

    Handles:
    - Bash (.sh, .bash)
    - Zsh (.zsh)
    - Fish (.fish)
    - Generic shell scripts
    """

    def _get_language_name(self) -> str:
        """Return language identifier."""
        return 'bash'

    def _get_import_node_types(self) -> List[str]:
        """Get node types that represent imports for bash."""
        return ['source_command', 'command']

    def _is_comment_node(self, node: Any) -> bool:
        """Check if node is a comment."""
        logger.info("[UNTESTED PATH] bash._is_comment_node called")
        return node.type in ['comment', 'line_comment', 'block_comment']

    def _get_tree_sitter_language(self) -> Any:
        """Get Bash language for tree-sitter."""
        logger.info("[UNTESTED PATH] bash._get_tree_sitter_language called")
        return get_language('bash')

    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """Bash-specific node types to chunk."""
        types = super()._get_chunk_node_types()

        # Shell-specific mappings
        types.update(
            {
                'function_definition': ChunkType.FUNCTION,
                'variable_assignment': ChunkType.CONSTANTS,
                'case_statement': ChunkType.FUNCTION,
                'heredoc_redirect': ChunkType.DOCSTRING,
                'compound_statement': ChunkType.MODULE,
                'subshell': ChunkType.MODULE,
                'command_substitution': ChunkType.MODULE,
                'shebang': ChunkType.IMPORTS,
            }
        )

        return types

    def _create_chunk_from_node(
        self,
        node: Any,
        lines: List[str],
        file_path: str,
        chunk_type: ChunkType,
        processed_ranges: Dict[str, Set[Tuple[int, int]]],
    ) -> Optional[Chunk]:
        """Override to handle Bash-specific cases."""
        # Special handling for functions
        if node.type == 'function_definition':
            chunk_type = self._determine_function_type(node)

        # For variable assignments, check if they're exports/constants
        elif node.type == 'variable_assignment':
            if not self._is_constant_variable(node, lines):
                return None  # Skip non-constant variables

        # Standard processing
        chunk = super()._create_chunk_from_node(
            node, lines, file_path, chunk_type, processed_ranges
        )

        # Add shell-specific metadata
        if chunk:
            if chunk.metadata.chunk_type == ChunkType.FUNCTION:
                chunk.metadata.language_specific = self._extract_function_metadata(node)
            elif chunk.metadata.chunk_type in [ChunkType.CONSTANTS, ChunkType.MODULE]:
                chunk.metadata.language_specific = self._extract_script_metadata(node)

        return chunk

    def _determine_function_type(self, func_node: Any) -> ChunkType:
        """Determine function type based on naming patterns."""
        name = self._extract_node_name(func_node)

        if name and ('test' in name.lower() or 'check' in name.lower()):
            return ChunkType.TESTS

        if name in ['main', 'run', 'start', 'init']:
            return ChunkType.CONSTRUCTOR

        return ChunkType.FUNCTION

    def _is_constant_variable(self, var_node: Any, lines: List[str]) -> bool:
        """Check if variable should be chunked as constant."""
        node_text = cast(str, var_node.text.decode('utf8'))

        # Export statements are constants
        if 'export' in node_text:
            return True

        # UPPER_CASE convention
        for child in var_node.children:
            if child.type == 'variable_name':
                var_name = cast(str, child.text.decode('utf8'))
                if var_name.isupper() and '_' in var_name:
                    logger.info("[UNTESTED PATH] bash uppercase variable detected")
                    return True

        # Check if preceded by config comment
        start_line = var_node.start_point[0]
        if start_line > 0:
            prev_line = lines[start_line - 1].strip()
            if prev_line.startswith('#') and any(
                word in prev_line.lower() for word in ['config', 'constant', 'setting', 'default']
            ):
                return True

        return False

    def _extract_node_name(self, node: Any) -> Optional[str]:
        """Extract name from function nodes."""
        if node.type == 'function_definition':
            for child in node.children:
                if child.type == 'word':
                    return cast(str, child.text.decode('utf8'))

        # Call parent method and handle the return value explicitly
        parent_result = super()._extract_node_name(node)
        logger.info("[UNTESTED PATH] bash parent node name extraction")
        return parent_result if isinstance(parent_result, str) else None

    def _extract_function_metadata(self, func_node: Any) -> Dict[str, Any]:
        """Extract shell function metadata."""
        metadata: Dict[str, Any] = {
            'name': self._extract_node_name(func_node),
            'has_local_vars': False,
            'calls_functions': [],
            'uses_conditionals': False,
            'uses_loops': False,
            'uses_error_handling': False,
            'uses_strict_mode': False,
        }

        # Analyze function body
        def analyze_node(node: Any) -> None:
            if node.type == 'local_variable_declaration':
                metadata['has_local_vars'] = True
            elif node.type == 'command' and len(node.children) > 0:
                # Check for function calls
                first_child = node.children[0]
                if first_child.type == 'command_name':
                    cmd = cast(str, first_child.text.decode('utf8'))
                    # Skip basic shell builtins
                    if cmd not in [
                        'echo',
                        'printf',
                        'cd',
                        'ls',
                        'cp',
                        'mv',
                        'rm',
                        'exit',
                        'return',
                    ]:
                        calls_list = metadata['calls_functions']
                        if isinstance(calls_list, list):
                            calls_list.append(cmd)
            elif node.type in ['if_statement', 'elif_clause']:
                metadata['uses_conditionals'] = True
            elif node.type in ['for_statement', 'while_statement']:
                logger.info("[UNTESTED PATH] bash loop detected")
                metadata['uses_loops'] = True

            # Recurse
            for child in node.children:
                analyze_node(child)

        analyze_node(func_node)

        # Ensure calls_functions is a list before using set
        calls_functions = metadata.get('calls_functions', [])
        if isinstance(calls_functions, list):
            metadata['calls_functions'] = list(set(calls_functions))

        # Check for shell-specific patterns
        func_text = cast(str, func_node.text.decode('utf8'))
        if any(pattern in func_text for pattern in ['set -e', 'set -o errexit', 'trap']):
            metadata['uses_error_handling'] = True
        if any(pattern in func_text for pattern in ['set -u', 'set -o nounset']):
            metadata['uses_strict_mode'] = True

        return metadata

    def _extract_dependencies_from_imports(self, import_nodes: List[Any]) -> List[str]:
        """Extract dependencies from source/. commands."""
        deps: Set[str] = set()

        # Shell doesn't have traditional imports
        # Look for source patterns in the content
        for node in import_nodes:
            if node.type == 'command':
                cmd_text = cast(str, node.text.decode('utf8'))
                # Check for source or . commands
                if cmd_text.startswith('source ') or cmd_text.startswith('. '):
                    # Extract the file being sourced
                    parts = cmd_text.split()
                    if len(parts) > 1:
                        file_path = parts[1].strip('"\'')
                        logger.info("[UNTESTED PATH] bash dependency found")
                        deps.add(file_path)

        return sorted(list(deps))

    def _extract_script_metadata(self, node: Any) -> Dict[str, Any]:
        """Extract metadata for script sections."""
        metadata = {
            'type': node.type,
            'has_error_handling': False,
            'uses_strict_mode': False,
        }

        # Check for error handling patterns
        node_text = cast(str, node.text.decode('utf8'))
        if any(pattern in node_text for pattern in ['set -e', 'set -o errexit', 'trap']):
            metadata['has_error_handling'] = True

        if any(pattern in node_text for pattern in ['set -u', 'set -o nounset']):
            metadata['uses_strict_mode'] = True

        return metadata
