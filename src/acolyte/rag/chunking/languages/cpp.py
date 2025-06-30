"""
C++ chunker using tree-sitter-languages.
Handles C++ code with full metadata extraction including templates, classes, and modern features.
"""

from typing import Dict, List, Optional, Any, Set, Tuple, Callable, cast
from tree_sitter_languages import get_language
import tree_sitter

from acolyte.models.chunk import ChunkType, Chunk
from acolyte.core.logging import logger
from acolyte.rag.chunking.base import BaseChunker
from acolyte.rag.chunking.mixins import SecurityAnalysisMixin


class CppChunker(BaseChunker, SecurityAnalysisMixin):
    """
    C++-specific chunker using tree-sitter.

    Extracts comprehensive metadata for C++ code including:
    - Classes with inheritance and access specifiers
    - Templates (class, function, variable)
    - Namespaces and using declarations
    - Modern C++ features (auto, lambda, concepts)
    - RAII patterns and smart pointers
    - STL usage patterns
    """

    def _get_language_name(self) -> str:
        """Return language identifier."""
        return 'cpp'

    def _get_import_node_types(self) -> List[str]:
        """Get node types that represent imports for cpp."""
        return ['preproc_include', 'using_declaration', 'using_directive']

    def _extract_imports(
        self,
        root_node: tree_sitter.Node,
        lines: List[str],
        file_path: str,
        processed_ranges: Dict[str, Set[Tuple[int, int]]],
    ) -> List[Chunk]:
        """Extract import statements as individual chunks for C++."""
        import_chunks = []
        import_nodes = []

        # Get language-specific import node types
        import_types = self._get_import_node_types()

        # Find all import nodes
        def find_imports(node: tree_sitter.Node) -> None:
            if node.type in import_types:
                import_nodes.append(node)
            for child in node.children:
                find_imports(child)

        find_imports(root_node)

        if not import_nodes:
            return []

        # Create individual chunks for each import
        for node in import_nodes:
            start_line = node.start_point[0]
            end_line = node.end_point[0]

            # Mark as processed
            if node.type not in processed_ranges:
                processed_ranges[node.type] = set()
            processed_ranges[node.type].add((start_line, end_line))

            # Extract content
            content = '\n'.join(lines[start_line : end_line + 1])

            # Extract dependencies
            dependencies = self._extract_dependencies_from_imports([node])

            chunk = self._create_chunk(
                content=content,
                chunk_type=ChunkType.IMPORTS,
                file_path=file_path,
                start_line=start_line + 1,
                end_line=end_line + 1,
                name=f'import_{node.type}',
            )

            # Add dependencies metadata
            if dependencies:
                chunk.metadata.language_specific = {'dependencies': dependencies}

            import_chunks.append(chunk)

        return import_chunks

    def _is_comment_node(self, node: tree_sitter.Node) -> bool:
        """Check if node is a comment."""
        return node.type in ['comment', 'line_comment', 'block_comment']

    def _get_tree_sitter_language(self) -> Any:
        """Get C++ language for tree-sitter."""
        logger.info("[UNTESTED PATH] cpp._get_tree_sitter_language called")
        return get_language('cpp')

    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """
        C++-specific node types to chunk.

        Tree-sitter C++ node types:
        - function_definition: Function implementations
        - class_specifier: Class definitions
        - struct_specifier: Struct definitions (public by default)
        - namespace_definition: Namespace blocks
        - template_declaration: Template definitions
        - using_declaration: Using statements
        - preproc_include: Include directives
        """
        return {
            # Functions and methods
            'function_definition': ChunkType.FUNCTION,
            'template_declaration': ChunkType.UNKNOWN,  # Will refine
            # Classes and types
            'class_specifier': ChunkType.CLASS,
            'struct_specifier': ChunkType.CLASS,
            'union_specifier': ChunkType.CLASS,
            'enum_specifier': ChunkType.TYPES,
            'enum_class_specifier': ChunkType.TYPES,
            # Namespaces
            'namespace_definition': ChunkType.NAMESPACE,
            # Imports and using
            'preproc_include': ChunkType.IMPORTS,
            'using_declaration': ChunkType.IMPORTS,
            'using_directive': ChunkType.IMPORTS,
            'alias_declaration': ChunkType.TYPES,
        }

    def _get_decision_node_types(self) -> Set[str]:
        """Override to add C++-specific decision nodes."""
        base_types = super()._get_decision_node_types()
        # Add C++ specific
        base_types.add('for_range_loop')
        base_types.add('try_statement')
        base_types.add('catch_clause')
        return base_types

    def _get_nesting_node_types(self) -> Set[str]:
        """Override to add C++-specific nesting nodes."""
        base_types = super()._get_nesting_node_types()
        base_types.add('for_range_loop')
        return base_types

    def _get_security_patterns(
        self,
    ) -> List[Callable[[tree_sitter.Node, str], Optional[Dict[str, Any]]]]:
        """Override to add C++-specific security checks."""
        patterns = super()._get_security_patterns()
        patterns.extend(
            [
                self._check_raw_pointer_usage,
                self._check_c_style_cast,
                self._check_unsafe_c_functions,
            ]
        )
        return patterns

    def _create_chunk_from_node(
        self,
        node: tree_sitter.Node,
        lines: List[str],
        file_path: str,
        chunk_type: ChunkType,
        processed_ranges: Dict[str, Set[Tuple[int, int]]],
    ) -> Optional[Chunk]:
        """Override to handle C++-specific cases."""
        # Handle template declarations specially
        if node.type == 'template_declaration':
            chunk_type = self._determine_template_type(node)
            if chunk_type == ChunkType.UNKNOWN:
                return None

        # Standard processing
        chunk = super()._create_chunk_from_node(
            node, lines, file_path, chunk_type, processed_ranges
        )

        # Add C++-specific metadata
        if chunk:
            chunk.metadata.language_specific = self._extract_cpp_metadata(node, lines)

        return chunk

    def _determine_template_type(self, node: tree_sitter.Node) -> ChunkType:
        """Determine what kind of template this is."""
        # Look at the child to determine template type
        for child in node.children:
            if child.type == 'function_definition':
                return ChunkType.FUNCTION
            elif child.type in ['class_specifier', 'struct_specifier']:
                return ChunkType.CLASS
            elif child.type == 'alias_declaration':
                return ChunkType.TYPES

        return ChunkType.UNKNOWN

    def _extract_cpp_metadata(self, node: tree_sitter.Node, lines: List[str]) -> Dict[str, Any]:
        """Extract C++-specific metadata."""
        metadata = {
            'modifiers': [],
            'visibility': 'private',  # Default for class
            'is_virtual': False,
            'is_override': False,
            'is_final': False,
            'is_const': False,
            'is_noexcept': False,
            'is_template': False,
            'template_params': [],
            'base_classes': [],
            'return_type': None,
            'parameters': [],
            'complexity': {},
            'patterns': {'anti': [], 'good': []},
            'todos': [],
            'security': [],
            'modern_cpp': [],
            'stl_usage': [],
        }

        # Extract based on node type
        if node.type == 'function_definition':
            metadata.update(self._extract_function_metadata(node, lines))
        elif node.type in ['class_specifier', 'struct_specifier']:
            metadata.update(self._extract_class_metadata(node))
        elif node.type == 'namespace_definition':
            metadata.update(self._extract_namespace_metadata(node))
        elif node.type == 'template_declaration':
            metadata['is_template'] = True
            metadata['template_params'] = self._extract_template_params(node)
            # Process the actual declaration but preserve template flag
            for child in node.children:
                if child.type in ['function_definition', 'class_specifier', 'struct_specifier']:
                    child_metadata = self._extract_cpp_metadata(child, lines)
                    # Merge metadata, preserving template flags
                    metadata.update(child_metadata)
                    metadata['is_template'] = True
                    metadata['template_params'] = self._extract_template_params(node)

        # Extract TODOs/FIXMEs using mixin
        metadata['todos'] = self._extract_todos(node)

        # Analyze for security and patterns
        metadata['security'] = self._detect_security_issues(node)
        metadata['modern_cpp'] = self._analyze_modern_cpp(node)
        metadata['stl_usage'] = self._extract_stl_usage(node)

        return metadata

    def _extract_function_metadata(
        self, func_node: tree_sitter.Node, lines: List[str]
    ) -> Dict[str, Any]:
        """Extract metadata specific to C++ functions."""
        metadata: Dict[str, Any] = {
            'modifiers': [],
            'return_type': 'void',
            'parameters': [],
            'complexity': {},
            'throws': [],
            'is_noexcept': False,
            'is_virtual': False,
            'is_override': False,
            'is_final': False,
            'is_const': False,
        }

        # Ensure modifiers is a list
        modifiers_list = cast(List[str], metadata['modifiers'])

        # Ensure return_type is a string
        return_type = cast(str, metadata['return_type'])

        # Extract function components
        for child in func_node.children:
            # Storage class and qualifiers
            if child.type == 'storage_class_specifier':
                storage = child.text.decode('utf8') if child.text is not None else ''
                modifiers_list.append(storage)

            elif child.type == 'type_qualifier':
                qualifier = child.text.decode('utf8') if child.text is not None else ''
                modifiers_list.append(qualifier)
                if qualifier == 'const':
                    metadata['is_const'] = True

            elif child.type == 'virtual':
                metadata['is_virtual'] = True
                modifiers_list.append('virtual')

            # Return type
            elif child.type in [
                'primitive_type',
                'type_identifier',
                'qualified_identifier',
                'template_type',
                'auto',
            ]:
                return_type = child.text.decode('utf8') if child.text is not None else ''
                metadata['return_type'] = return_type

            elif child.type == 'reference_declarator':
                return_type += '&'
                metadata['return_type'] = return_type

            elif child.type == 'pointer_declarator':
                return_type += '*'
                metadata['return_type'] = return_type

            # Function declarator
            elif child.type == 'function_declarator':
                # Extract name and parameters
                for subchild in child.children:
                    if subchild.type == 'parameter_list':
                        metadata['parameters'] = self._extract_parameters(subchild)
                    elif subchild.type == 'noexcept':
                        metadata['is_noexcept'] = True
                        modifiers_list.append('noexcept')
                    elif subchild.type == 'trailing_return_type':
                        # C++11 trailing return type
                        return_type = (
                            subchild.text.decode('utf8') if subchild.text is not None else ''
                        )
                        metadata['return_type'] = return_type
                    elif (
                        subchild.type == 'type_qualifier'
                        and subchild.text.decode('utf8') == 'const'
                        if subchild.text is not None
                        else ''
                    ):
                        metadata['is_const'] = True
                        if 'const' not in modifiers_list:
                            modifiers_list.append('const')

            # Function body
            elif child.type == 'compound_statement':
                metadata['complexity'] = self._calculate_complexity(child)

            # Override/final specifiers
            elif child.type == 'virtual_specifier':
                spec = child.text.decode('utf8') if child.text is not None else ''
                if spec == 'override':
                    metadata['is_override'] = True
                elif spec == 'final':
                    metadata['is_final'] = True
                modifiers_list.append(spec)

        # Detect constructor/destructor
        func_name = self._extract_node_name(func_node)
        if func_name:
            parent_class = self._get_parent_class_name(func_node)
            if parent_class:
                if func_name == parent_class:
                    metadata['is_constructor'] = True
                elif func_name == f'~{parent_class}':
                    metadata['is_destructor'] = True

        # Check for override in more locations
        def find_virtual_specifiers(node: tree_sitter.Node) -> None:
            for child in node.children:
                if child.type == 'virtual_specifier':
                    spec = child.text.decode('utf8') if child.text is not None else ''
                    if spec == 'override':
                        metadata['is_override'] = True
                        if 'override' not in modifiers_list:
                            modifiers_list.append('override')
                    elif spec == 'final':
                        metadata['is_final'] = True
                        if 'final' not in modifiers_list:
                            modifiers_list.append('final')
                find_virtual_specifiers(child)

        find_virtual_specifiers(func_node)

        # Update metadata with the list
        metadata['modifiers'] = modifiers_list

        # Detect patterns
        if 'patterns' not in metadata:
            metadata['patterns'] = {'anti': [], 'good': []}
        patterns_dict = cast(Dict[str, List[str]], metadata['patterns'])
        anti_patterns = patterns_dict.get('anti', [])
        if isinstance(anti_patterns, list):
            anti_patterns.extend(self._detect_anti_patterns(func_node, metadata))
        good_patterns = patterns_dict.get('good', [])
        if isinstance(good_patterns, list):
            good_patterns.extend(self._detect_good_patterns(func_node, metadata))

        return metadata

    def _extract_class_metadata(self, class_node: tree_sitter.Node) -> Dict[str, Any]:
        """Extract metadata for class definitions."""
        metadata: Dict[str, Any] = {
            'type': 'class' if class_node.type == 'class_specifier' else 'struct',
            'base_classes': [],
            'members': {'public': [], 'protected': [], 'private': []},
            'methods': {'public': [], 'protected': [], 'private': []},
            'is_abstract': False,
            'has_virtual_destructor': False,
            'follows_raii': False,
            'uses_rule_of_five': False,
        }

        current_access = 'private' if class_node.type == 'class_specifier' else 'public'

        # Extract base classes - tree-sitter places them as direct children
        # The inheritance syntax ': public BaseClass' is parsed as base_class_clause
        for i, child in enumerate(class_node.children):
            if child.type == 'base_class_clause':
                metadata['base_classes'] = self._extract_base_classes(child)
                break
            # For debugging: check all children
            elif child.type == ':' and i + 1 < len(class_node.children):
                # The next child after ':' might contain base classes
                next_child = class_node.children[i + 1]
                if next_child.type == 'base_class_clause':
                    metadata['base_classes'] = self._extract_base_classes(next_child)
                    break

        # Extract members and methods
        for child in class_node.children:
            if child.type == 'field_declaration_list':
                for member in child.children:
                    if member.type == 'access_specifier':
                        # Update current access level
                        access_text = child.text.decode('utf8') if child.text is not None else ''
                        current_access = access_text.rstrip(':')

                    elif member.type == 'field_declaration':
                        # Member variable
                        field_info = self._extract_field_info(member)
                        if field_info:
                            members_dict = metadata.get('members', {})
                            if isinstance(members_dict, dict) and current_access in members_dict:
                                members_list = members_dict[current_access]
                                if isinstance(members_list, list):
                                    members_list.append(field_info)

                    elif member.type == 'function_definition':
                        # Method
                        method_name = self._extract_node_name(member)
                        if method_name:
                            method_info = {
                                'name': method_name,
                                'is_virtual': self._has_virtual_keyword(member),
                                'is_const': self._is_const_method(member),
                            }
                            methods_dict = metadata.get('methods', {})
                            if isinstance(methods_dict, dict) and current_access in methods_dict:
                                methods_list = methods_dict[current_access]
                                if isinstance(methods_list, list):
                                    methods_list.append(method_info)

                            # Check for virtual destructor
                            if method_name.startswith('~') and method_info['is_virtual']:
                                metadata['has_virtual_destructor'] = True

                    elif member.type == 'template_declaration':
                        # Template member
                        pass  # Handle if needed

        # Analyze patterns
        metadata['follows_raii'] = self._check_raii_pattern(metadata)
        metadata['uses_rule_of_five'] = self._check_rule_of_five(metadata)
        metadata['is_abstract'] = self._has_pure_virtual_methods(class_node)

        return metadata

    def _extract_template_params(self, template_node: tree_sitter.Node) -> List[Dict[str, str]]:
        """Extract template parameters."""
        params = []

        for child in template_node.children:
            if child.type == 'template_parameter_list':
                for param in child.children:
                    if param.type == 'type_parameter_declaration':
                        param_info = {'type': 'type', 'name': ''}
                        for subchild in param.children:
                            if subchild.type == 'type_identifier':
                                param_info['name'] = (
                                    subchild.text.decode('utf8')
                                    if subchild.text is not None
                                    else ''
                                )
                        params.append(param_info)

                    elif param.type == 'parameter_declaration':
                        # Non-type template parameter
                        param_info = {'type': 'value', 'name': ''}
                        for subchild in param.children:
                            if subchild.type == 'identifier':
                                param_info['name'] = (
                                    subchild.text.decode('utf8')
                                    if subchild.text is not None
                                    else ''
                                )
                        params.append(param_info)

        return params

    def _extract_parameters(self, param_list_node: tree_sitter.Node) -> List[Dict[str, Any]]:
        """Extract parameter information for C++ functions."""
        params = []

        for child in param_list_node.children:
            if child.type == 'parameter_declaration':
                param = {
                    'name': 'unnamed',
                    'type': '',
                    'is_reference': False,
                    'is_pointer': False,
                    'is_const': False,
                    'is_rvalue_ref': False,
                    'default_value': None,
                }

                # Extract parameter details
                type_parts = []
                for subchild in child.children:
                    if (
                        subchild.type == 'type_qualifier'
                        and subchild.text.decode('utf8') == 'const'
                        if subchild.text is not None
                        else ''
                    ):
                        param['is_const'] = True
                        type_parts.append('const')

                    elif subchild.type in [
                        'primitive_type',
                        'type_identifier',
                        'qualified_identifier',
                        'auto',
                    ]:
                        type_parts.append(
                            subchild.text.decode('utf8') if subchild.text is not None else ''
                        )

                    elif subchild.type == 'reference_declarator':
                        param['is_reference'] = True
                        # Check for rvalue reference
                        if (
                            '&&' in subchild.text.decode('utf8')
                            if subchild.text is not None
                            else ''
                        ):
                            param['is_rvalue_ref'] = True
                        # Extract name
                        for ref_child in subchild.children:
                            if ref_child.type == 'identifier':
                                param['name'] = (
                                    ref_child.text.decode('utf8')
                                    if ref_child.text is not None
                                    else ''
                                )

                    elif subchild.type == 'pointer_declarator':
                        param['is_pointer'] = True
                        for ptr_child in subchild.children:
                            if ptr_child.type == 'identifier':
                                param['name'] = (
                                    ptr_child.text.decode('utf8')
                                    if ptr_child.text is not None
                                    else ''
                                )

                    elif subchild.type == 'identifier':
                        param['name'] = (
                            subchild.text.decode('utf8') if subchild.text is not None else ''
                        )

                    elif subchild.type == 'default_value':
                        # Extract default parameter value
                        param['default_value'] = (
                            subchild.text.decode('utf8').lstrip('=').strip()
                            if subchild.text is not None
                            else ''
                        )

                param['type'] = ' '.join(type_parts)
                params.append(param)

        return params

    def _analyze_modern_cpp(self, node: tree_sitter.Node) -> List[Dict[str, str]]:
        """Detect modern C++ features usage."""
        features = []

        def walk_tree(node: tree_sitter.Node) -> None:
            # Auto keyword
            if node.type == 'auto':
                features.append(
                    {'feature': 'auto', 'line': str(node.start_point[0] + 1), 'c++_version': '11'}
                )

            # Lambda expressions
            elif node.type == 'lambda_expression':
                features.append(
                    {'feature': 'lambda', 'line': str(node.start_point[0] + 1), 'c++_version': '11'}
                )

            # Range-based for loops
            elif node.type == 'for_range_loop':
                features.append(
                    {
                        'feature': 'range_for',
                        'line': str(node.start_point[0] + 1),
                        'c++_version': '11',
                    }
                )

            # Smart pointers - check multiple node types and contexts
            elif node.type in [
                'qualified_identifier',
                'template_type',
                'type_identifier',
                'template_argument_list',
            ]:
                text = node.text.decode('utf8') if node.text is not None else ''
                # Check for smart pointers in various forms
                if 'unique_ptr' in text:
                    features.append(
                        {
                            'feature': 'smart_pointer_unique_ptr',
                            'line': str(node.start_point[0] + 1),
                            'c++_version': '11',
                        }
                    )
                elif 'shared_ptr' in text or 'make_shared' in text:
                    features.append(
                        {
                            'feature': 'smart_pointer_shared_ptr',
                            'line': str(node.start_point[0] + 1),
                            'c++_version': '11',
                        }
                    )
                elif 'weak_ptr' in text:
                    features.append(
                        {
                            'feature': 'smart_pointer_weak_ptr',
                            'line': str(node.start_point[0] + 1),
                            'c++_version': '11',
                        }
                    )

            # Move semantics
            elif node.type == 'call_expression':
                for child in node.children:
                    if child.type == 'qualified_identifier':
                        func_name = child.text.decode('utf8') if child.text is not None else ''
                        if func_name == 'std::move':
                            features.append(
                                {
                                    'feature': 'move_semantics',
                                    'line': str(node.start_point[0] + 1),
                                    'c++_version': '11',
                                }
                            )

            # Recurse
            for child in node.children:
                walk_tree(child)

        walk_tree(node)
        return features

    def _extract_stl_usage(self, node: tree_sitter.Node) -> List[Dict[str, str]]:
        """Extract STL container and algorithm usage."""
        stl_usage = []

        def walk_tree(node: tree_sitter.Node) -> None:
            if node.type in ['qualified_identifier', 'type_identifier']:
                text = node.text.decode('utf8') if node.text is not None else ''

                # STL containers
                stl_containers = [
                    'vector',
                    'list',
                    'deque',
                    'set',
                    'map',
                    'unordered_set',
                    'unordered_map',
                    'array',
                    'forward_list',
                    'queue',
                    'stack',
                ]

                for container in stl_containers:
                    if f'std::{container}' in text or text == container:
                        stl_usage.append(
                            {
                                'type': 'container',
                                'name': container,
                                'line': str(node.start_point[0] + 1),
                            }
                        )
                        break

                # STL algorithms
                stl_algorithms = [
                    'sort',
                    'find',
                    'copy',
                    'transform',
                    'accumulate',
                    'for_each',
                    'remove',
                    'unique',
                    'reverse',
                ]

                for algo in stl_algorithms:
                    if f'std::{algo}' in text:
                        stl_usage.append(
                            {
                                'type': 'algorithm',
                                'name': algo,
                                'line': str(node.start_point[0] + 1),
                            }
                        )
                        break

            for child in node.children:
                walk_tree(child)

        walk_tree(node)
        return stl_usage

    # C++-specific security checks
    def _check_raw_pointer_usage(
        self, node: tree_sitter.Node, text: str
    ) -> Optional[Dict[str, Any]]:
        """Check for raw pointer usage without smart pointers."""
        if node.type == 'new_expression':
            # Check for raw pointer usage
            return {
                'type': 'raw_pointer_new',
                'severity': 'medium',
                'description': 'Consider using std::unique_ptr or std::shared_ptr',
                'line': node.start_point[0] + 1,
            }
        return None

    def _check_c_style_cast(self, node: tree_sitter.Node, text: str) -> Optional[Dict[str, Any]]:
        """Check for C-style casts."""
        if node.type == 'cast_expression':
            if '(' in text and ')' in text:
                return {
                    'type': 'c_style_cast',
                    'severity': 'low',
                    'description': 'Use static_cast, dynamic_cast, const_cast, or reinterpret_cast',
                    'line': node.start_point[0] + 1,
                }
        return None

    def _check_unsafe_c_functions(
        self, node: tree_sitter.Node, text: str
    ) -> Optional[Dict[str, Any]]:
        """Check for unsafe C functions."""
        if node.type == 'call_expression':
            for child in node.children:
                if child.type == 'identifier':
                    func_name = child.text.decode('utf8') if child.text is not None else ''
                    if func_name in ['strcpy', 'strcat', 'sprintf', 'gets']:
                        return {
                            'type': 'unsafe_c_function',
                            'severity': 'high',
                            'function': func_name,
                            'description': 'Use std::string or safer alternatives',
                            'line': node.start_point[0] + 1,
                        }
        return None

    def _detect_anti_patterns(
        self, func_node: tree_sitter.Node, metadata: Dict[str, Any]
    ) -> List[str]:
        """Detect C++ anti-patterns."""
        patterns = []

        # God function
        if metadata['complexity'].get('lines_of_code', 0) > 100:
            patterns.append('god_function')

        # Deep nesting
        if metadata['complexity'].get('nesting_depth', 0) > 4:
            patterns.append('deep_nesting')

        # High complexity
        if metadata['complexity'].get('cyclomatic', 0) > 10:
            patterns.append('high_complexity')

        # Too many parameters
        params = metadata.get('parameters', [])
        if isinstance(params, list) and len(params) > 7:
            patterns.append('too_many_parameters')

        # Non-const reference parameters (can be confusing)
        if isinstance(params, list):
            for param in params:
                if (
                    isinstance(param, dict)
                    and param.get('is_reference')
                    and not param.get('is_const')
                    and not param.get('is_rvalue_ref')
                ):
                    patterns.append('non_const_ref_param')
                    break

        # Exception in destructor (if detectable)
        if metadata.get('is_destructor') and metadata['complexity'].get('exceptions', 0) > 0:
            patterns.append('exception_in_destructor')

        return patterns

    def _detect_good_patterns(
        self, func_node: tree_sitter.Node, metadata: Dict[str, Any]
    ) -> List[str]:
        """Detect good C++ patterns."""
        patterns = []

        # RAII usage
        if metadata.get('is_constructor') or metadata.get('is_destructor'):
            patterns.append('raii_pattern')

        # Const correctness
        if metadata.get('is_const', False) and metadata.get('is_method'):
            patterns.append('const_correct')

        # Noexcept usage
        if metadata.get('is_noexcept', False):
            patterns.append('noexcept_specified')

        # Override usage
        if metadata.get('is_override', False):
            patterns.append('explicit_override')

        return patterns

    def _extract_base_classes(self, base_clause_node: tree_sitter.Node) -> List[Dict[str, Any]]:
        """Extract base class information."""
        bases = []

        for child in base_clause_node.children:
            if child.type == 'base_class':
                base_info: Dict[str, Any] = {
                    'name': '',
                    'access': 'private',  # Default for class
                    'is_virtual': False,
                }

                for subchild in child.children:
                    if subchild.type == 'access_specifier':
                        base_info['access'] = (
                            subchild.text.decode('utf8') if subchild.text is not None else ''
                        )
                    elif subchild.type == 'virtual':
                        base_info['is_virtual'] = True
                    elif subchild.type in ['type_identifier', 'qualified_identifier']:
                        base_info['name'] = (
                            subchild.text.decode('utf8') if subchild.text is not None else ''
                        )

                if base_info['name']:
                    bases.append(base_info)

        return bases

    def _extract_field_info(self, field_node: tree_sitter.Node) -> Optional[Dict[str, Any]]:
        """Extract field/member variable information."""
        field = {
            'name': '',
            'type': '',
            'is_static': False,
            'is_const': False,
            'is_mutable': False,
            'is_reference': False,
            'is_pointer': False,
        }

        type_parts = []

        for child in field_node.children:
            if child.type == 'storage_class_specifier':
                if child.text.decode('utf8') == 'static' if child.text is not None else '':
                    field['is_static'] = True
                elif child.text.decode('utf8') == 'mutable' if child.text is not None else '':
                    field['is_mutable'] = True

            elif child.type == 'type_qualifier':
                if child.text.decode('utf8') == 'const' if child.text is not None else '':
                    field['is_const'] = True

            elif child.type in ['primitive_type', 'type_identifier', 'qualified_identifier']:
                type_parts.append(child.text.decode('utf8') if child.text is not None else '')

            elif child.type == 'reference_declarator':
                field['is_reference'] = True
                for subchild in child.children:
                    if subchild.type == 'field_identifier':
                        field['name'] = (
                            subchild.text.decode('utf8') if subchild.text is not None else ''
                        )

            elif child.type == 'pointer_declarator':
                field['is_pointer'] = True
                for subchild in child.children:
                    if subchild.type == 'field_identifier':
                        field['name'] = (
                            subchild.text.decode('utf8') if subchild.text is not None else ''
                        )

            elif child.type == 'field_identifier':
                field['name'] = child.text.decode('utf8') if child.text is not None else ''

        field['type'] = ' '.join(type_parts)

        return field if field['name'] else None

    def _extract_namespace_metadata(self, ns_node: tree_sitter.Node) -> Dict[str, Any]:
        """Extract namespace metadata."""
        metadata = {'type': 'namespace', 'name': '', 'is_inline': False, 'is_anonymous': False}

        for child in ns_node.children:
            if child.type == 'namespace_identifier':
                metadata['name'] = child.text.decode('utf8') if child.text is not None else ''
            elif child.type == 'inline':
                metadata['is_inline'] = True

        if not metadata['name']:
            metadata['is_anonymous'] = True

        return metadata

    # Helper methods
    def _is_inside_class(self, node: tree_sitter.Node) -> bool:
        """Check if node is inside a class definition."""
        parent = node.parent
        while parent:
            if parent.type in ['class_specifier', 'struct_specifier']:
                return True
            parent = parent.parent
        return False

    def _get_parent_class_name(self, node: tree_sitter.Node) -> Optional[str]:
        """Get the name of the parent class if inside one."""
        parent = node.parent
        while parent:
            if parent.type in ['class_specifier', 'struct_specifier']:
                for child in parent.children:
                    if child.type == 'type_identifier':
                        return cast(
                            str, child.text.decode('utf8') if child.text is not None else ''
                        )
            parent = parent.parent
        return None

    def _has_virtual_keyword(self, node: tree_sitter.Node) -> bool:
        """Check if function has virtual keyword."""
        for child in node.children:
            if child.type == 'virtual':
                return True
        return False

    def _is_const_method(self, node: tree_sitter.Node) -> bool:
        """Check if method is const."""
        for child in node.children:
            if child.type == 'function_declarator':
                for subchild in child.children:
                    if (
                        subchild.type == 'type_qualifier'
                        and subchild.text.decode('utf8') == 'const'
                        if subchild.text is not None
                        else ''
                    ):
                        return True
        return False

    def _has_pure_virtual_methods(self, class_node: tree_sitter.Node) -> bool:
        """Check if class has pure virtual methods (abstract)."""

        def walk_tree(node: tree_sitter.Node) -> bool:
            if node.type == 'abstract_function_declarator':
                return True
            # Look for = 0 pattern
            if node.type == 'function_definition':
                text = node.text.decode('utf8') if node.text is not None else ''
                if '= 0' in text:
                    return True

            for child in node.children:
                if walk_tree(child):
                    return True
            return False

        return walk_tree(class_node)

    def _check_raii_pattern(self, metadata: Dict[str, Any]) -> bool:
        """Check if class follows RAII pattern."""
        # Simple heuristic: has destructor and manages resources
        methods_dict = metadata.get('methods', {})
        has_destructor = False
        if isinstance(methods_dict, dict):
            has_destructor = any(
                method['name'].startswith('~')
                for methods in methods_dict.values()
                if isinstance(methods, list)
                for method in methods
                if isinstance(method, dict) and 'name' in method
            )

        # Check for resource management indicators
        members_dict = metadata.get('members', {})
        has_resources = False
        if isinstance(members_dict, dict):
            has_resources = any(
                (field.get('is_pointer', False) or 'handle' in field.get('type', '').lower())
                for fields in members_dict.values()
                if isinstance(fields, list)
                for field in fields
                if isinstance(field, dict)
            )

        return has_destructor and has_resources

    def _check_rule_of_five(self, metadata: Dict[str, Any]) -> bool:
        """Check if class implements rule of five."""
        required_methods = set()

        # Look for special member functions
        methods_dict = metadata.get('methods', {})
        if isinstance(methods_dict, dict):
            for methods in methods_dict.values():
                if isinstance(methods, list):
                    for method in methods:
                        if isinstance(method, dict) and 'name' in method:
                            name = method['name']
                            # Constructor
                            if name == metadata.get('class_name', ''):
                                required_methods.add('constructor')
                            # Destructor
                            elif name.startswith('~'):
                                required_methods.add('destructor')
                            # Copy constructor (would need more analysis)
                            # Move constructor (would need more analysis)
                            # Copy assignment (would need more analysis)
                            # Move assignment (would need more analysis)

        # Simplified check - just verify destructor exists for now
        return 'destructor' in required_methods

    def _extract_dependencies_from_imports(self, import_nodes: List[tree_sitter.Node]) -> List[str]:
        """Extract C++ header dependencies."""
        deps = []

        for node in import_nodes:
            if node.type == 'preproc_include':
                for child in node.children:
                    if child.type in ['string_literal', 'system_lib_string']:
                        header = (
                            child.text.decode('utf8').strip('"<>') if child.text is not None else ''
                        )
                        deps.append(header)

            elif node.type in ['using_declaration', 'using_directive']:
                # Extract namespace being used
                text = node.text.decode('utf8') if node.text is not None else ''
                if 'namespace' in text:
                    ns = text.split('namespace')[-1].strip(';').strip()
                    if ns and ns != 'std':  # Don't track std
                        deps.append(f'namespace:{ns}')

        return deps
