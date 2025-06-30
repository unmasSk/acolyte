"""
PHP chunker using tree-sitter-languages.
Comprehensive metadata extraction for PHP 5.6+ and PHP 8.x features.
"""

from typing import Dict, List, Any, Optional, Set
from tree_sitter_languages import get_language  # type: ignore

from acolyte.models.chunk import ChunkType
from acolyte.core.logging import logger
from acolyte.rag.chunking.base import BaseChunker
from acolyte.rag.chunking.mixins import SecurityAnalysisMixin, PatternDetectionMixin


class PhpChunker(BaseChunker, SecurityAnalysisMixin, PatternDetectionMixin):
    """
    PHP-specific chunker using tree-sitter.

    Handles:
    - PHP 5.6+ features (traits, generators, variadic)
    - PHP 7.x features (return types, null coalesce, spaceship)
    - PHP 8.x features (union types, attributes, named args, enums)
    - PSR standards detection
    - Security vulnerability patterns
    """

    def _get_language_name(self) -> str:
        """Return language identifier."""
        return 'php'

    def _get_import_node_types(self) -> List[str]:
        """Get node types that represent imports for php."""
        return [
            'use_declaration',
            'namespace_use_declaration',
            'require_expression',
            'include_expression',
        ]

    def _is_comment_node(self, node: Any) -> bool:
        """Check if node is a comment."""
        return node.type in ['comment', 'line_comment', 'block_comment']

    def _get_tree_sitter_language(self) -> Any:
        """Get PHP language for tree-sitter."""
        logger.info("[UNTESTED PATH] php._get_tree_sitter_language called")
        return get_language('php')

    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """
        PHP-specific node types to chunk.

        Tree-sitter PHP node types:
        - function_definition - Regular functions
        - method_declaration - Class methods
        - class_declaration - Classes
        - interface_declaration - Interfaces
        - trait_declaration - Traits
        - enum_declaration - Enums (PHP 8.1+)
        - namespace_definition - Namespaces
        """
        return {
            # Functions and methods
            'function_definition': ChunkType.FUNCTION,
            'method_declaration': ChunkType.METHOD,
            # Classes and similar
            'class_declaration': ChunkType.CLASS,
            'interface_declaration': ChunkType.INTERFACE,
            'trait_declaration': ChunkType.CLASS,  # Traits are class-like
            'enum_declaration': ChunkType.CLASS,  # Enums are class-like
            # Namespace (module-like)
            'namespace_definition': ChunkType.MODULE,
            # Imports
            'namespace_use_declaration': ChunkType.IMPORTS,
            'namespace_aliasing_clause': ChunkType.IMPORTS,
        }

    def _create_chunk_from_node(
        self,
        node: Any,
        lines: List[str],
        file_path: str,
        chunk_type: ChunkType,
        processed_ranges: Any,
    ) -> Any:
        """Override to handle PHP-specific cases and extract rich metadata."""
        # Special handling for constructors
        if node.type == 'method_declaration':
            method_name = self._get_method_name(node)
            if method_name == '__construct':
                chunk_type = ChunkType.CONSTRUCTOR
            elif method_name and method_name.startswith('test'):
                chunk_type = ChunkType.TESTS

        # Standard processing
        chunk = super()._create_chunk_from_node(
            node, lines, file_path, chunk_type, processed_ranges
        )

        if not chunk:
            return None

        # Extract comprehensive PHP metadata
        if chunk.metadata.chunk_type in [
            ChunkType.FUNCTION,
            ChunkType.METHOD,
            ChunkType.CONSTRUCTOR,
        ]:
            chunk.metadata.language_specific = self._extract_function_metadata(node, lines)
        elif chunk.metadata.chunk_type == ChunkType.CLASS:
            chunk.metadata.language_specific = self._extract_class_metadata(node, lines)
        elif chunk.metadata.chunk_type == ChunkType.INTERFACE:
            chunk.metadata.language_specific = self._extract_interface_metadata(node, lines)

        return chunk

    def _get_method_name(self, node: Any) -> Optional[str]:
        """Extract method name from method declaration node."""
        for child in node.children:
            if child.type == 'name':
                logger.info("[UNTESTED PATH] php._get_method_name found name")
                return child.text.decode('utf8')
        return None

    def _extract_function_metadata(self, func_node: Any, lines: List[str]) -> Dict[str, Any]:
        """Extract comprehensive PHP function/method metadata."""
        modifiers: List[str] = []
        parameters: List[Dict[str, Any]] = []
        throws: List[str] = []
        attributes: List[str] = []
        test_coverage_hints: List[str] = []
        internal_deps: List[str] = []
        external_deps: List[str] = []

        metadata: Dict[str, Any] = {
            # Basic info
            'visibility': 'public',  # Default for functions
            'modifiers': modifiers,
            'is_static': False,
            'is_abstract': False,
            'is_final': False,
            'parameters': parameters,
            'return_type': None,
            'throws': throws,
            # PHP 8 attributes
            'attributes': attributes,
            # Complexity metrics
            'complexity': self._calculate_complexity(func_node),
            # Code quality
            'quality': {
                'has_docblock': False,
                'uses_type_hints': False,
                'uses_return_type': False,
                'test_coverage_hints': test_coverage_hints,
            },
            # Dependencies
            'dependencies': {'internal': internal_deps, 'external': external_deps},
        }
        # Initialize patterns properly
        metadata['patterns'] = {'anti': [], 'framework': []}
        # Detect patterns and merge results
        detected_patterns = self._detect_patterns(func_node, {'complexity': metadata['complexity']})
        if isinstance(detected_patterns, dict):
            if 'anti' in detected_patterns:
                metadata['patterns']['anti'] = detected_patterns['anti']
            if 'framework' in detected_patterns:
                metadata['patterns']['framework'] = detected_patterns['framework']
        metadata['todos'] = (
            self._extract_todos(func_node) if hasattr(self, '_extract_todos') else []
        )
        metadata['security'] = []

        # Extract modifiers and visibility
        for child in func_node.children:
            if child.type == 'visibility_modifier':
                metadata['visibility'] = child.text.decode('utf8')
            elif child.type == 'static_modifier':
                metadata['is_static'] = True
                modifiers.append('static')
            elif child.type == 'abstract_modifier':
                metadata['is_abstract'] = True
                modifiers.append('abstract')
            elif child.type == 'final_modifier':
                metadata['is_final'] = True
                modifiers.append('final')

            # PHP 8 attributes
            elif child.type == 'attribute_list':
                extracted_attrs = self._extract_attributes(child)
                attributes.clear()
                attributes.extend(extracted_attrs)

            # Parameters
            elif child.type == 'formal_parameters':
                extracted_params = self._extract_parameters(child)
                parameters.clear()
                parameters.extend(extracted_params)
                metadata['quality']['uses_type_hints'] = any(p.get('type') for p in parameters)

            # Return type
            elif child.type == 'return_type':
                logger.info("[UNTESTED PATH] php return type detected")
                metadata['return_type'] = child.text.decode('utf8').strip(': ')
                metadata['quality']['uses_return_type'] = True

            # Function body for analysis
            elif child.type == 'compound_statement':
                self._analyze_function_body(child, metadata, lines)

        # Check for docblock
        start_line = func_node.start_point[0]
        if start_line > 0:
            prev_line = start_line - 1
            while prev_line >= 0 and lines[prev_line].strip().startswith('*'):
                if '/**' in lines[prev_line]:
                    metadata['quality']['has_docblock'] = True
                    # Extract @throws from docblock
                    for i in range(prev_line, start_line):
                        if '@throws' in lines[i]:
                            exception = lines[i].split('@throws')[1].strip().split()[0]
                            throws.append(exception)
                    break
                prev_line -= 1

        return metadata

    def _extract_class_metadata(self, class_node: Any, lines: List[str]) -> Dict[str, Any]:
        """Extract comprehensive PHP class metadata."""
        modifiers: List[str] = []
        implements_list: List[str] = []
        uses_list: List[str] = []  # Traits
        attributes: List[str] = []  # PHP 8 attributes
        methods: List[str] = []
        properties: List[str] = []
        constants: List[str] = []
        test_coverage_hints: List[str] = []
        anti_patterns: List[str] = []
        framework_patterns: List[str] = []
        internal_deps: List[str] = []
        external_deps: List[str] = []

        metadata: Dict[str, Any] = {
            'type': 'class',  # class, trait, enum
            'modifiers': modifiers,
            'is_abstract': False,
            'is_final': False,
            'extends': None,
            'implements': implements_list,
            'uses': uses_list,  # Traits
            'attributes': attributes,  # PHP 8 attributes
            'methods': methods,
            'properties': properties,
            'constants': constants,
            'quality': {
                'has_docblock': False,
                'follows_psr': False,
                'has_constructor': False,
                'test_coverage_hints': test_coverage_hints,
            },
            'patterns': {'anti': anti_patterns, 'framework': framework_patterns},
            'dependencies': {'internal': internal_deps, 'external': external_deps},
        }

        for child in class_node.children:
            if child.type == 'abstract_modifier':
                metadata['is_abstract'] = True
                modifiers.append('abstract')
            elif child.type == 'final_modifier':
                metadata['is_final'] = True
                modifiers.append('final')
            elif child.type == 'base_clause':
                # extends clause
                for subchild in child.children:
                    if subchild.type == 'name' or subchild.type == 'qualified_name':
                        metadata['extends'] = subchild.text.decode('utf8')
            elif child.type == 'class_interface_clause':
                # implements clause
                logger.info("[UNTESTED PATH] php class implements clause")
                for subchild in child.children:
                    if subchild.type == 'name' or subchild.type == 'qualified_name':
                        implements_list.append(subchild.text.decode('utf8'))
            elif child.type == 'attribute_list':
                extracted_attrs = self._extract_attributes(child)
                attributes.clear()
                attributes.extend(extracted_attrs)
            elif child.type == 'declaration_list':
                self._analyze_class_body(child, metadata)

        # Check for docblock
        start_line = class_node.start_point[0]
        if start_line > 0 and '/**' in lines[start_line - 1]:
            metadata['quality']['has_docblock'] = True

        # Detect patterns
        if metadata['extends'] and 'Controller' in metadata['extends']:
            framework_patterns.append('mvc_controller')
        if any('Repository' in impl for impl in implements_list):
            framework_patterns.append('repository_pattern')

        # PSR-4 check removed - would need file_path parameter

        return metadata

    def _extract_interface_metadata(self, interface_node: Any, lines: List[str]) -> Dict[str, Any]:
        """Extract PHP interface metadata."""
        extends_list: List[str] = []
        methods: List[str] = []
        constants: List[str] = []
        framework_patterns: List[str] = []

        metadata: Dict[str, Any] = {
            'type': 'interface',
            'extends': extends_list,
            'methods': methods,
            'constants': constants,
            'quality': {'has_docblock': False},
            'patterns': {'framework': framework_patterns},
        }

        for child in interface_node.children:
            if child.type == 'base_clause':
                for subchild in child.children:
                    if subchild.type in ['name', 'qualified_name']:
                        extends_list.append(subchild.text.decode('utf8'))
            elif child.type == 'declaration_list':
                self._analyze_interface_body(child, metadata)

        return metadata

    def _extract_parameters(self, params_node: Any) -> List[Dict[str, Any]]:
        """Extract parameter information including PHP 8 features."""
        parameters = []

        for child in params_node.children:
            if child.type in [
                'simple_parameter',
                'variadic_parameter',
                'property_promotion_parameter',
            ]:
                param = {
                    'name': None,
                    'type': None,
                    'default': None,
                    'optional': False,
                    'variadic': child.type == 'variadic_parameter',
                    'promoted': child.type == 'property_promotion_parameter',
                    'visibility': None,  # For promoted properties
                }

                for subchild in child.children:
                    if subchild.type == 'variable_name':
                        param['name'] = subchild.text.decode('utf8')
                    elif subchild.type in ['type_declaration', 'union_type', 'intersection_type']:
                        param['type'] = subchild.text.decode('utf8')
                    elif subchild.type == 'default_value':
                        logger.info("[UNTESTED PATH] php parameter default value")
                        param['default'] = subchild.text.decode('utf8').strip('= ')
                        param['optional'] = True
                    elif subchild.type == 'visibility_modifier':
                        param['visibility'] = subchild.text.decode('utf8')

                if param['name']:
                    parameters.append(param)

        return parameters

    def _extract_attributes(self, attr_list_node: Any) -> List[str]:
        """Extract PHP 8 attributes."""
        attributes = []
        for child in attr_list_node.children:
            if child.type == 'attribute':
                attr_text = child.text.decode('utf8')
                attributes.append(attr_text.strip('#[]'))
        return attributes

    def _analyze_function_body(
        self, body_node: Any, metadata: Dict[str, Any], lines: List[str]
    ) -> None:
        """Analyze function body for patterns and security issues."""
        # Security analysis
        metadata['security'] = self._detect_php_security_issues(body_node, 0)

        # Walk AST for dependency analysis
        self._analyze_dependencies(body_node, metadata)

        # Test hints
        if self._uses_test_framework(body_node):
            hints_list = metadata.get('quality', {}).get('test_coverage_hints', [])
            if isinstance(hints_list, list):
                hints_list.append('uses_test_framework')

    def _detect_php_security_issues(self, node: Any, start_line: int) -> List[Dict[str, Any]]:
        """PHP-specific security pattern detection."""
        issues = []

        def walk_for_security(n, depth=0):
            line = n.start_point[0] + 1

            # SQL injection risks
            if n.type == 'binary_expression' and any(
                sql_keyword in n.text.decode('utf8').lower()
                for sql_keyword in ['select', 'insert', 'update', 'delete', 'from', 'where']
            ):
                # Check if it's string concatenation with variables
                if any(op in n.text.decode('utf8') for op in ['.', '.=']):
                    for child in n.children:
                        if child.type == 'variable_name':
                            issues.append(
                                {
                                    'type': 'sql_injection_risk',
                                    'severity': 'high',
                                    'line': line,
                                    'description': 'SQL query with string concatenation',
                                }
                            )
                            break

            # eval() usage
            elif n.type == 'function_call_expression':
                func_name = ''
                for child in n.children:
                    if child.type == 'name':
                        func_name = child.text.decode('utf8')
                        break

                if func_name in ['eval', 'assert', 'create_function']:
                    issues.append(
                        {
                            'type': 'code_injection_risk',
                            'severity': 'critical',
                            'line': line,
                            'description': f'Use of dangerous function: {func_name}',
                        }
                    )
                elif func_name in ['exec', 'system', 'shell_exec', 'passthru']:
                    logger.info("[UNTESTED PATH] php command injection risk")
                    issues.append(
                        {
                            'type': 'command_injection_risk',
                            'severity': 'high',
                            'line': line,
                            'description': f'Shell command execution: {func_name}',
                        }
                    )
                elif func_name in ['file_get_contents', 'include', 'require'] and depth > 2:
                    # Check if using variable paths
                    for child in n.children:
                        if child.type == 'arguments':
                            for arg in child.children:
                                if arg.type == 'variable_name':
                                    logger.info("[UNTESTED PATH] php file inclusion risk")
                                    issues.append(
                                        {
                                            'type': 'file_inclusion_risk',
                                            'severity': 'medium',
                                            'line': line,
                                            'description': 'Dynamic file inclusion',
                                        }
                                    )
                                    break

            # Hardcoded credentials
            elif n.type == 'assignment_expression':
                text = n.text.decode('utf8').lower()
                if any(secret in text for secret in ['password', 'api_key', 'secret', 'token']):
                    # Check if it's a literal string assignment
                    for child in n.children:
                        if child.type == 'string' and len(child.text) > 10:
                            logger.info("[UNTESTED PATH] php hardcoded credential")
                            issues.append(
                                {
                                    'type': 'hardcoded_credential',
                                    'severity': 'critical',
                                    'line': line,
                                    'description': 'Possible hardcoded credential',
                                }
                            )

            # Weak crypto
            elif n.type == 'function_call_expression':
                func_name = self._get_function_name(n)
                if func_name in ['md5', 'sha1']:
                    issues.append(
                        {
                            'type': 'weak_crypto',
                            'severity': 'medium',
                            'line': line,
                            'description': f'Weak hashing algorithm: {func_name}',
                        }
                    )

            # Recurse
            for child in n.children:
                walk_for_security(child, depth + 1)

        walk_for_security(node)
        return issues

    def _analyze_dependencies(self, node: Any, metadata: Dict[str, Any]) -> None:
        """Extract internal and external dependencies from function body."""

        def walk_for_deps(n):
            if n.type == 'function_call_expression':
                func_name = self._get_function_name(n)
                if func_name:
                    # Check if it's a framework/library function
                    if any(fw in func_name.lower() for fw in ['symfony', 'laravel', 'doctrine']):
                        ext_deps = metadata.get('dependencies', {}).get('external', [])
                        if isinstance(ext_deps, list) and func_name not in ext_deps:
                            ext_deps.append(func_name)

            elif n.type == 'object_creation_expression':
                class_name = ''
                for child in n.children:
                    if child.type in ['name', 'qualified_name']:
                        class_name = child.text.decode('utf8')
                        break

                if class_name:
                    if '\\' in class_name:  # Namespaced
                        ext_deps = metadata.get('dependencies', {}).get('external', [])
                        if isinstance(ext_deps, list) and class_name not in ext_deps:
                            ext_deps.append(class_name)
                    else:
                        int_deps = metadata.get('dependencies', {}).get('internal', [])
                        if isinstance(int_deps, list) and class_name not in int_deps:
                            int_deps.append(class_name)

            elif n.type == 'member_call_expression':
                # Track method calls that might indicate framework usage
                method_text = n.text.decode('utf8')
                if any(
                    pattern in method_text for pattern in ['->render(', '->json(', '->redirect(']
                ):
                    patterns_dict = metadata.get('patterns', {})
                    if not isinstance(patterns_dict, dict):
                        metadata['patterns'] = {'anti': [], 'framework': []}
                        patterns_dict = metadata['patterns']
                    framework_list = patterns_dict.get('framework', [])
                    if isinstance(framework_list, list) and 'mvc_framework' not in framework_list:
                        framework_list.append('mvc_framework')

            # Recurse
            for child in n.children:
                walk_for_deps(child)

        walk_for_deps(node)

    def _analyze_class_body(self, body_node: Any, metadata: Dict[str, Any]) -> None:
        """Analyze class body for methods, properties, etc."""
        for child in body_node.children:
            if child.type == 'method_declaration':
                method_name = self._get_method_name(child)
                if method_name:
                    methods_list = metadata.get('methods', [])
                    if isinstance(methods_list, list):
                        methods_list.append(method_name)
                    if method_name == '__construct':
                        metadata['quality']['has_constructor'] = True
                    elif method_name.startswith('test'):
                        hints_list = metadata.get('quality', {}).get('test_coverage_hints', [])
                        if isinstance(hints_list, list):
                            hints_list.append('has_test_methods')

            elif child.type == 'property_declaration':
                for subchild in child.children:
                    if subchild.type == 'property_element':
                        prop_name = subchild.text.decode('utf8').strip('$;')
                        props_list = metadata.get('properties', [])
                        if isinstance(props_list, list):
                            props_list.append(prop_name)

            elif child.type == 'const_declaration':
                for subchild in child.children:
                    if subchild.type == 'const_element':
                        const_name = self._get_const_name(subchild)
                        if const_name:
                            const_list = metadata.get('constants', [])
                            if isinstance(const_list, list):
                                const_list.append(const_name)

            elif child.type == 'use_declaration':
                # Trait usage
                for subchild in child.children:
                    if subchild.type in ['name', 'qualified_name']:
                        uses_list = metadata.get('uses', [])
                        if isinstance(uses_list, list):
                            uses_list.append(subchild.text.decode('utf8'))

    def _analyze_interface_body(self, body_node: Any, metadata: Dict[str, Any]) -> None:
        """Analyze interface body."""
        for child in body_node.children:
            if child.type == 'method_declaration':
                method_name = self._get_method_name(child)
                if method_name:
                    methods_list = metadata.get('methods', [])
                    if isinstance(methods_list, list):
                        methods_list.append(method_name)
            elif child.type == 'const_declaration':
                for subchild in child.children:
                    if subchild.type == 'const_element':
                        const_name = self._get_const_name(subchild)
                        if const_name:
                            const_list = metadata.get('constants', [])
                            if isinstance(const_list, list):
                                const_list.append(const_name)

    def _get_function_name(self, node: Any) -> Optional[str]:
        """Extract function name from function call node."""
        for child in node.children:
            if child.type == 'name':
                return child.text.decode('utf8')
        return None

    def _get_const_name(self, const_element: Any) -> Optional[str]:
        """Extract constant name."""
        for child in const_element.children:
            if child.type == 'name':
                return child.text.decode('utf8')
        return None

    def _uses_test_framework(self, node: Any) -> bool:
        """Check if the function uses PHPUnit or other test frameworks."""
        test_indicators = ['assert', 'expect', 'mock', 'getMock', 'createMock', 'expectException']

        def check_node(n):
            if n.type == 'function_call_expression':
                func_name = self._get_function_name(n)
                if func_name and any(indicator in func_name for indicator in test_indicators):
                    return True

            for child in n.children:
                if check_node(child):
                    return True
            return False

        return check_node(node)

    def _extract_dependencies_from_imports(self, import_nodes: List[Any]) -> List[str]:
        """Extract PHP use statements and requires."""
        deps: Set[str] = set()

        for node in import_nodes:
            if node.type == 'namespace_use_declaration':
                # use statements
                for child in node.children:
                    if child.type == 'namespace_use_clause':
                        for subchild in child.children:
                            if subchild.type == 'qualified_name':
                                namespace = subchild.text.decode('utf8')
                                # Extract root namespace
                                root = namespace.split('\\')[0]
                                deps.add(root)

        return sorted(list(deps))
