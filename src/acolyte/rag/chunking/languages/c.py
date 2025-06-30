"""
C chunker using tree-sitter-languages.
Handles C code with full metadata extraction.
"""

from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from tree_sitter_languages import get_language
from tree_sitter import Node

from acolyte.models.chunk import ChunkType, Chunk
from acolyte.core.logging import logger
from acolyte.rag.chunking.base import BaseChunker
from acolyte.rag.chunking.mixins import SecurityAnalysisMixin, PatternDetectionMixin


class CChunker(BaseChunker, SecurityAnalysisMixin, PatternDetectionMixin):
    """
    C-specific chunker using tree-sitter.

    Extracts comprehensive metadata for C code including:
    - Function signatures with types
    - Struct/union/enum definitions
    - Preprocessor directives
    - Memory management patterns
    - Security vulnerabilities
    """

    def _get_language_name(self) -> str:
        """Return language identifier."""
        return 'c'

    def _get_import_node_types(self) -> List[str]:
        """Get node types that represent imports for c."""
        return ['preproc_include']

    def _is_comment_node(self, node: Node) -> bool:
        """Check if node is a comment."""
        return node.type in ['comment', 'line_comment', 'block_comment']

    def _get_tree_sitter_language(self) -> Any:
        """Get C language for tree-sitter."""
        logger.info("[UNTESTED PATH] c._get_tree_sitter_language called")
        return get_language('c')

    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """
        C-specific node types to chunk.

        Tree-sitter C node types:
        - function_definition: Complete function implementations
        - declaration: Function declarations/prototypes
        - struct_specifier: Struct definitions
        - union_specifier: Union definitions
        - enum_specifier: Enum definitions
        - preproc_include: #include directives
        - preproc_function_def: Macro functions
        """
        return {
            # Functions
            'function_definition': ChunkType.FUNCTION,
            'declaration': ChunkType.UNKNOWN,  # Will refine based on content
            # Types
            'struct_specifier': ChunkType.CLASS,  # Closest equivalent
            'union_specifier': ChunkType.CLASS,
            'enum_specifier': ChunkType.TYPES,
            # Preprocessor
            'preproc_include': ChunkType.IMPORTS,
            'preproc_function_def': ChunkType.FUNCTION,
            'preproc_def': ChunkType.CONSTANTS,
        }

    def _create_chunk_from_node(
        self,
        node: Node,
        lines: List[str],
        file_path: str,
        chunk_type: ChunkType,
        processed_ranges: Dict[str, Set[Tuple[int, int]]],
    ) -> Optional[Chunk]:
        """Override to handle C-specific cases."""
        # Refine declaration nodes
        if node.type == 'declaration':
            chunk_type = self._determine_declaration_type(node)
            if chunk_type == ChunkType.UNKNOWN:
                return None  # Skip non-function declarations for now

        # Standard processing
        chunk = super()._create_chunk_from_node(
            node, lines, file_path, chunk_type, processed_ranges
        )

        # Add C-specific metadata
        if chunk:
            chunk.metadata.language_specific = self._extract_c_metadata(node, lines)

        return chunk

    def _determine_declaration_type(self, node: Node) -> ChunkType:
        """Determine what kind of declaration this is."""
        # Look for function declarations (prototypes)
        for child in node.children:
            if child.type == 'function_declarator':
                return ChunkType.FUNCTION
            elif child.type == 'init_declarator':
                # Check if it's a function pointer
                logger.info("[UNTESTED PATH] c init_declarator check")
                for subchild in child.children:
                    if subchild.type == 'pointer_declarator':
                        for subsubchild in subchild.children:
                            if subsubchild.type == 'function_declarator':
                                return ChunkType.FUNCTION

        return ChunkType.UNKNOWN

    def _extract_c_metadata(self, node: Node, lines: List[str]) -> Dict[str, Any]:
        """Extract C-specific metadata."""
        metadata: Dict[str, Any] = {
            'modifiers': [],
            'return_type': None,
            'parameters': [],
            'is_static': False,
            'is_inline': False,
            'is_extern': False,
            'storage_class': None,
            'complexity': {},
            'patterns': {'anti': [], 'good': []},
            'todos': [],
            'security': [],
            'memory_ops': [],
            'preprocessor': [],
        }

        # Extract based on node type
        if node.type == 'function_definition':
            metadata.update(self._extract_function_metadata(node, lines))
        elif node.type == 'struct_specifier':
            metadata.update(self._extract_struct_metadata(node))
        elif node.type == 'enum_specifier':
            metadata.update(self._extract_enum_metadata(node))

        # Use mixins for common functionality
        metadata['todos'] = self._extract_todos(node)
        metadata['security'] = self._detect_security_issues(node)

        # Extract C-specific patterns
        metadata['memory_ops'] = self._extract_memory_ops(node)

        return metadata

    def _extract_function_metadata(self, func_node: Node, lines: List[str]) -> Dict[str, Any]:
        """Extract metadata specific to functions."""
        metadata: Dict[str, Any] = {
            'modifiers': [],
            'return_type': 'void',
            'parameters': [],
            'complexity': {},
        }

        # Extract modifiers and return type
        for child in func_node.children:
            if child.type == 'storage_class_specifier':
                storage = child.text.decode('utf8') if child.text is not None else ''
                metadata['storage_class'] = storage
                metadata['is_static'] = storage == 'static'
                metadata['is_extern'] = storage == 'extern'
                metadata['modifiers'].append(storage)

            elif child.type == 'type_qualifier':
                metadata['modifiers'].append(
                    child.text.decode('utf8') if child.text is not None else ''
                )

            elif child.type in ['primitive_type', 'type_identifier', 'sized_type_specifier']:
                metadata['return_type'] = (
                    child.text.decode('utf8') if child.text is not None else ''
                )

            elif child.type == 'pointer_declarator':
                logger.info("[UNTESTED PATH] c pointer return type")
                metadata['return_type'] += '*'

            elif child.type == 'function_declarator':
                # Extract function name and parameters
                for subchild in child.children:
                    if subchild.type == 'parameter_list':
                        metadata['parameters'] = self._extract_parameters(subchild)

            elif child.type == 'compound_statement':
                # Use mixin for complexity analysis
                metadata['complexity'] = self._calculate_complexity(child)

        # Use mixin for pattern detection
        patterns = self._detect_patterns(func_node, metadata)
        metadata['patterns'] = patterns

        return metadata

    def _extract_parameters(self, param_list_node: Node) -> List[Dict[str, Any]]:
        """Extract parameter information."""
        params: List[Dict[str, Any]] = []

        for child in param_list_node.children:
            if child.type == 'parameter_declaration':
                param = {'name': 'unnamed', 'type': '', 'is_pointer': False, 'is_const': False}

                # Extract type and name
                type_parts = []
                for subchild in child.children:
                    if subchild.type in [
                        'primitive_type',
                        'type_identifier',
                        'sized_type_specifier',
                    ]:
                        type_parts.append(
                            subchild.text.decode('utf8') if subchild.text is not None else ''
                        )
                    elif (
                        subchild.type == 'type_qualifier'
                        and subchild.text is not None
                        and subchild.text.decode('utf8') == 'const'
                    ):
                        param['is_const'] = True
                    elif subchild.type == 'pointer_declarator':
                        param['is_pointer'] = True
                        # Look for the identifier
                        for subsubchild in subchild.children:
                            if subsubchild.type == 'identifier':
                                param['name'] = (
                                    subsubchild.text.decode('utf8')
                                    if subsubchild.text is not None
                                    else ''
                                )
                    elif subchild.type == 'identifier':
                        param['name'] = (
                            subchild.text.decode('utf8') if subchild.text is not None else ''
                        )
                    elif subchild.type == 'array_declarator':
                        # Handle array parameters
                        for subsubchild in subchild.children:
                            if subsubchild.type == 'identifier':
                                param['name'] = (
                                    subsubchild.text.decode('utf8')
                                    if subsubchild.text is not None
                                    else ''
                                )
                        type_parts.append('[]')

                param['type'] = ' '.join(type_parts)
                params.append(param)

        return params

    # Override security patterns for C-specific vulnerabilities
    def _get_security_patterns(self) -> List[Callable[[Node, str], Optional[Dict[str, Any]]]]:
        """C-specific security checks."""
        return [
            self._check_buffer_overflow,
            self._check_format_string,
            self._check_null_pointer,
            self._check_sql_injection,
            self._check_hardcoded_credentials,
            self._check_weak_crypto,
        ]

    def _check_buffer_overflow(self, node: Node, text: str) -> Optional[Dict[str, Any]]:
        """Check for buffer overflow vulnerabilities."""
        if node.type == 'call_expression':
            func_name = None
            for child in node.children:
                if child.type == 'identifier':
                    func_name = child.text.decode('utf8') if child.text is not None else ''
                    break

            if func_name in ['strcpy', 'strcat', 'gets', 'sprintf']:
                return {
                    'type': 'buffer_overflow_risk',
                    'severity': 'high',
                    'function': func_name,
                    'description': f'Unsafe function {func_name} - use {func_name}n or safer alternative',
                }
        return None

    def _check_format_string(self, node: Node, text: str) -> Optional[Dict[str, Any]]:
        """Check for format string vulnerabilities."""
        if node.type == 'call_expression':
            func_name = None
            for child in node.children:
                if child.type == 'identifier':
                    func_name = child.text.decode('utf8') if child.text is not None else ''
                    break

            if func_name in ['printf', 'sprintf', 'fprintf']:
                # Check if first arg is not a literal
                args = [c for c in node.children if c.type == 'argument_list']
                if args:
                    first_arg = None
                    for arg in args[0].children:
                        if arg.type not in [',', '(', ')']:
                            first_arg = arg
                            break

                    if first_arg and first_arg.type != 'string_literal':
                        return {
                            'type': 'format_string_vulnerability',
                            'severity': 'high',
                            'function': func_name,
                            'description': 'Format string is not a literal',
                        }
        return None

    def _check_null_pointer(self, node: Node, text: str) -> Optional[Dict[str, Any]]:
        """Check for potential null pointer dereferences."""
        if node.type == 'pointer_expression':
            # Simple heuristic: check if pointer is checked before use
            parent = node.parent
            if parent and parent.type != 'binary_expression':
                return {
                    'type': 'potential_null_dereference',
                    'severity': 'medium',
                    'description': 'Pointer dereference without null check',
                }
        return None

    def _extract_memory_ops(self, node: Node) -> List[Dict[str, str]]:
        """Extract memory operations (malloc, free, etc.)."""
        memory_ops: List[Dict[str, str]] = []

        def walk_tree(node: Node) -> None:
            if node.type == 'call_expression':
                func_name = None
                for child in node.children:
                    if child.type == 'identifier':
                        func_name = child.text.decode('utf8') if child.text is not None else ''
                        break

                if func_name and func_name in ['malloc', 'calloc', 'realloc', 'free', 'alloca']:
                    memory_ops.append(
                        {'operation': func_name, 'line': str(node.start_point[0] + 1)}
                    )

            for child in node.children:
                walk_tree(child)

        walk_tree(node)
        return memory_ops

    def _extract_struct_metadata(self, struct_node: Node) -> Dict[str, Any]:
        """Extract metadata for struct definitions."""
        metadata: Dict[str, Any] = {
            'type': 'struct',
            'fields': [],
            'has_pointers': False,
            'has_unions': False,
            'size_hint': None,
        }

        # Extract struct fields
        for child in struct_node.children:
            if child.type == 'field_declaration_list':
                for field in child.children:
                    if field.type == 'field_declaration':
                        field_info = self._extract_field_info(field)
                        if field_info:
                            metadata['fields'].append(field_info)
                            if field_info['is_pointer']:
                                metadata['has_pointers'] = True

        return metadata

    def _extract_field_info(self, field_node: Node) -> Optional[Dict[str, Any]]:
        """Extract information about a struct field."""
        field: Dict[str, Any] = {
            'name': '',
            'type': '',
            'is_pointer': False,
            'is_array': False,
            'is_bitfield': False,
        }

        type_parts: List[str] = []

        for child in field_node.children:
            if child.type in ['primitive_type', 'type_identifier', 'sized_type_specifier']:
                type_parts.append(child.text.decode('utf8') if child.text is not None else '')
            elif child.type == 'pointer_declarator':
                field['is_pointer'] = True
                # Look for name
                for subchild in child.children:
                    if subchild.type == 'field_identifier':
                        field['name'] = (
                            subchild.text.decode('utf8') if subchild.text is not None else ''
                        )
            elif child.type == 'field_identifier':
                field['name'] = child.text.decode('utf8') if child.text is not None else ''
            elif child.type == 'array_declarator':
                field['is_array'] = True
                for subchild in child.children:
                    if subchild.type == 'field_identifier':
                        field['name'] = (
                            subchild.text.decode('utf8') if subchild.text is not None else ''
                        )
            elif child.type == 'bitfield_clause':
                field['is_bitfield'] = True

        field['type'] = ' '.join(type_parts)

        return field if field['name'] else None

    def _extract_enum_metadata(self, enum_node: Node) -> Dict[str, Any]:
        """Extract metadata for enum definitions."""
        metadata: Dict[str, Any] = {'type': 'enum', 'values': []}

        # Extract enum values
        for child in enum_node.children:
            if child.type == 'enumerator_list':
                for enumerator in child.children:
                    if enumerator.type == 'enumerator':
                        # Get the name
                        for subchild in enumerator.children:
                            if subchild.type == 'identifier':
                                metadata['values'].append(
                                    subchild.text.decode('utf8')
                                    if subchild.text is not None
                                    else ''
                                )

        return metadata

    def _extract_dependencies_from_imports(self, import_nodes: List[Node]) -> List[str]:
        """Extract C header dependencies."""
        deps: List[str] = []

        for node in import_nodes:
            if node.type == 'preproc_include':
                # Extract the header name
                for child in node.children:
                    if child.type in ['string_literal', 'system_lib_string']:
                        if child.text is not None:
                            header = child.text.decode('utf8').strip('"<>')
                            deps.append(header)

        return deps
