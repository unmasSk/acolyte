"""
Makefile chunker using tree-sitter-languages.
Handles targets, variables, conditionals and includes.
"""

from typing import Dict, List, Any, Optional
from tree_sitter_languages import get_language  # type: ignore

from acolyte.models.chunk import ChunkType
from acolyte.rag.chunking.base import BaseChunker


class MakefileChunker(BaseChunker):
    """
    Makefile-specific chunker using tree-sitter.

    Handles:
    - Targets/rules with dependencies
    - Variable assignments
    - Conditionals (ifdef, ifndef, ifeq)
    - Include directives
    - Function calls
    """

    def _get_language_name(self) -> str:
        """Return language identifier."""
        return 'makefile'

    def _get_import_node_types(self) -> List[str]:
        """Get node types that represent imports for makefile."""
        return ['include_directive']

    def _is_comment_node(self, node) -> bool:
        """Check if node is a comment."""
        return node.type in ['comment', 'line_comment', 'block_comment']

    def _get_tree_sitter_language(self) -> Any:
        """Get Makefile language for tree-sitter."""
        return get_language('makefile')

    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """
        Makefile-specific node types to chunk.

        Tree-sitter Makefile node types mapped to ChunkTypes.
        """
        return {
            # Rules (targets)
            'rule': ChunkType.FUNCTION,  # Targets are like functions
            # Variables
            'variable_assignment': ChunkType.CONSTANTS,
            # Conditionals
            'conditional': ChunkType.MODULE,  # Conditional blocks
            # Includes
            'include': ChunkType.IMPORTS,
            'sinclude': ChunkType.IMPORTS,
            '-include': ChunkType.IMPORTS,
        }

    def _create_chunk_from_node(
        self, node, lines: List[str], file_path: str, chunk_type: ChunkType, processed_ranges
    ):
        """Override to handle Makefile-specific cases."""
        # For rules, determine if it's a special target
        if node.type == 'rule':
            chunk_type = self._determine_rule_type(node)

        # Standard processing
        chunk = super()._create_chunk_from_node(
            node, lines, file_path, chunk_type, processed_ranges
        )

        # Add Makefile-specific metadata
        if chunk:
            if chunk.metadata.chunk_type == ChunkType.FUNCTION:  # Rules
                chunk.metadata.language_specific = self._extract_rule_metadata(node)
            elif chunk.metadata.chunk_type == ChunkType.CONSTANTS:  # Variables
                chunk.metadata.language_specific = self._extract_variable_metadata(node)

        return chunk

    def _determine_rule_type(self, rule_node) -> ChunkType:
        """Determine if a rule is special (phony, test, etc)."""
        # Get target names
        targets = []
        for child in rule_node.children:
            if child.type == 'targets':
                for target in child.children:
                    if target.type == 'word':
                        targets.append(target.text.decode('utf8'))

        # Check for special targets
        for target in targets:
            # Test targets
            if any(test in target for test in ['test', 'check', 'verify']):
                return ChunkType.TESTS

            # Phony targets (build tasks)
            if target.startswith('.'):
                return ChunkType.MODULE

        return ChunkType.FUNCTION

    def _extract_rule_metadata(self, rule_node) -> Dict[str, Any]:
        """Extract metadata from a Makefile rule."""
        metadata: Dict[str, Any] = {
            'targets': [],
            'prerequisites': [],
            'is_phony': False,
            'has_recipe': False,
            'is_pattern_rule': False,
        }

        for child in rule_node.children:
            if child.type == 'targets':
                # Extract target names
                for target in child.children:
                    if target.type == 'word':
                        target_name = target.text.decode('utf8')
                        targets_list = metadata.get('targets', [])
                        if isinstance(targets_list, list):
                            targets_list.append(target_name)
                        # Check if phony
                        if target_name.startswith('.'):
                            metadata['is_phony'] = True
                        # Check if pattern rule
                        if '%' in target_name:
                            metadata['is_pattern_rule'] = True

            elif child.type == 'prerequisites':
                # Extract dependencies
                for prereq in child.children:
                    if prereq.type == 'word':
                        prereqs_list = metadata.get('prerequisites', [])
                        if isinstance(prereqs_list, list):
                            prereqs_list.append(prereq.text.decode('utf8'))

            elif child.type == 'recipe':
                # Check if has commands
                metadata['has_recipe'] = True

        return metadata

    def _extract_variable_metadata(self, var_node) -> Dict[str, Any]:
        """Extract metadata from variable assignment."""
        metadata = {
            'variable_name': None,
            'assignment_type': '=',  # =, :=, ?=, +=
            'is_export': False,
            'is_override': False,
        }

        # Check modifiers
        parent = var_node.parent
        if parent and parent.type == 'export_directive':
            metadata['is_export'] = True
        elif parent and parent.type == 'override_directive':
            metadata['is_override'] = True

        # Extract variable details
        for child in var_node.children:
            if child.type == 'variable':
                metadata['variable_name'] = child.text.decode('utf8')
            elif child.type in ['=', ':=', '?=', '+=']:
                metadata['assignment_type'] = child.text.decode('utf8')

        return metadata

    def _extract_dependencies_from_imports(self, import_nodes) -> List[str]:
        """
        Extract included Makefiles.

        Examples:
        - include common.mk -> ['common.mk']
        - -include config/*.mk -> ['config/*.mk']
        """
        deps = []

        for node in import_nodes:
            # Skip the include keyword, get the filenames
            for child in node.children:
                if child.type not in ['include', 'sinclude', '-include']:
                    text = child.text.decode('utf8').strip()
                    if text:
                        deps.append(text)

        return deps

    def _extract_node_name(self, node) -> Optional[str]:
        """Override to extract Makefile-specific names."""
        if node.type == 'rule':
            # Get first target as name
            for child in node.children:
                if child.type == 'targets':
                    for target in child.children:
                        if target.type == 'word':
                            return target.text.decode('utf8')

        elif node.type == 'variable_assignment':
            # Get variable name
            for child in node.children:
                if child.type == 'variable':
                    return child.text.decode('utf8')

        return super()._extract_node_name(node)
