"""
Go chunker using tree-sitter-languages.
Extracts comprehensive metadata for superior code search and context.
"""

from typing import Dict, List, Any, Optional, Callable
from tree_sitter_languages import get_language

from acolyte.models.chunk import ChunkType
from acolyte.core.logging import logger
from acolyte.rag.chunking.base import BaseChunker
from acolyte.rag.chunking.mixins import SecurityAnalysisMixin


class GoChunker(BaseChunker, SecurityAnalysisMixin):
    """
    Go-specific chunker using tree-sitter.

    Extracts rich metadata including:
    - Receivers for methods (pointer vs value)
    - Interface satisfaction
    - Goroutines and channels
    - Error handling patterns
    - Package visibility
    """

    def _get_language_name(self) -> str:
        """Return language identifier."""
        return 'go'

    def _get_tree_sitter_language(self) -> Any:
        """Get Go language for tree-sitter."""
        return get_language('go')

    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """
        Go-specific node types to chunk.

        Go tree-sitter node types:
        - function_declaration: Regular functions
        - method_declaration: Methods with receivers
        - type_declaration: Type definitions (struct, interface, alias)
        - const_declaration: Constants
        - var_declaration: Package-level variables
        - import_declaration: Import statements
        """
        return {
            # Functions and methods
            'function_declaration': ChunkType.FUNCTION,
            'method_declaration': ChunkType.METHOD,
            # Types
            'type_declaration': ChunkType.UNKNOWN,  # Will be refined
            # Constants (important in Go)
            'const_declaration': ChunkType.CONSTANTS,
            'var_declaration': ChunkType.UNKNOWN,  # Could be constants if package-level
            # Imports - NO LONGER A SEPARATE CHUNK
            # 'import_declaration': ChunkType.IMPORTS,
            # Special: init functions
            'func_literal': ChunkType.UNKNOWN,  # Anonymous functions
        }

    def _extract_node_name(self, node: Any) -> Optional[str]:
        """Extract name from Go AST nodes."""
        # For methods - look for field_identifier after receiver
        if node.type == 'method_declaration':
            # Children order: func, parameter_list (receiver), field_identifier
            for i, child in enumerate(node.children):
                if child.type == 'field_identifier':
                    return child.text.decode('utf8')

        # For functions - look for field_identifier
        elif node.type == 'function_declaration':
            for child in node.children:
                if child.type == 'field_identifier':
                    return child.text.decode('utf8')

        # For type declarations - look inside type_spec
        elif node.type == 'type_declaration':
            for child in node.children:
                if child.type == 'type_spec':
                    for spec_child in child.children:
                        if spec_child.type == 'type_identifier':
                            logger.info("[UNTESTED PATH] go type_identifier extraction")
                            return spec_child.text.decode('utf8')

        # For const/var declarations - look inside const_spec/var_spec
        elif node.type in ['const_declaration', 'var_declaration']:
            spec_type = 'const_spec' if node.type == 'const_declaration' else 'var_spec'
            for child in node.children:
                if child.type == spec_type:
                    for spec_child in child.children:
                        if spec_child.type == 'identifier':
                            return spec_child.text.decode('utf8')

        # Fallback to base implementation
        return super()._extract_node_name(node)

    def _create_chunk_from_node(
        self,
        node: Any,
        lines: List[str],
        file_path: str,
        chunk_type: ChunkType,
        processed_ranges: Dict[str, Any],
    ) -> Optional[Any]:
        """Override to handle Go-specific cases and extract rich metadata."""
        # Handle type declarations
        if node.type == 'type_declaration':
            chunk_type = self._determine_type_kind(node)

        # Check for special functions
        if node.type == 'function_declaration':
            name = self._extract_node_name(node)
            if name == 'init':
                logger.info("[UNTESTED PATH] go init function")
                chunk_type = ChunkType.CONSTRUCTOR  # init functions are special
            elif name and name.startswith('Test'):
                chunk_type = ChunkType.TESTS
            elif name and name.startswith('Benchmark'):
                chunk_type = ChunkType.TESTS
            elif name and name.startswith('Example'):
                chunk_type = ChunkType.DOCSTRING  # Example functions are documentation

        # Standard processing
        chunk = super()._create_chunk_from_node(
            node, lines, file_path, chunk_type, processed_ranges
        )

        if not chunk:
            return None

        # Extract Go-specific metadata
        if chunk.metadata.chunk_type == ChunkType.METHOD:
            chunk.metadata.language_specific = self._extract_method_metadata(node)
        elif chunk.metadata.chunk_type == ChunkType.FUNCTION:
            chunk.metadata.language_specific = self._extract_function_metadata(node)
        elif chunk.metadata.chunk_type in [ChunkType.CLASS, ChunkType.INTERFACE]:
            chunk.metadata.language_specific = self._extract_type_metadata(node)
        elif chunk.metadata.chunk_type == ChunkType.CONSTANTS:
            chunk.metadata.language_specific = self._extract_const_metadata(node)

        # Add common metadata
        self._add_common_metadata(chunk, node, lines)

        return chunk

    def _determine_type_kind(self, type_node: Any) -> ChunkType:
        """Determine if type declaration is struct, interface, or alias."""
        for child in type_node.children:
            if child.type == 'type_spec':
                for spec_child in child.children:
                    if spec_child.type == 'struct_type':
                        return ChunkType.CLASS  # Structs are like classes
                    elif spec_child.type == 'interface_type':
                        return ChunkType.INTERFACE
                # If no struct/interface, it's a type alias
                logger.info("[UNTESTED PATH] go type alias")
                return ChunkType.TYPES

        return ChunkType.TYPES

    def _extract_method_metadata(self, method_node: Any) -> Dict[str, Any]:
        """Extract Go method-specific metadata."""
        metadata: Dict[str, Any] = {
            'receiver': None,
            'receiver_type': None,
            'is_pointer_receiver': False,
            'parameters': [],
            'return_types': [],
            'is_exported': False,
            'has_error_return': False,
            'complexity': {'cyclomatic': 1, 'nesting_depth': 0},
            'patterns': {'anti': [], 'idiomatic': []},
            'uses_goroutines': False,
            'uses_channels': False,
            'defer_count': 0,
        }

        # Extract receiver - first parameter list
        for child in method_node.children:
            if child.type == 'parameter_list' and metadata['receiver'] is None:
                # This is the receiver
                for param in child.children:
                    if param.type == 'parameter_declaration':
                        receiver_text = param.text.decode('utf8')
                        metadata['receiver'] = receiver_text
                        metadata['is_pointer_receiver'] = '*' in receiver_text
                        # Extract receiver type
                        for p_child in param.children:
                            if p_child.type == 'pointer_type':
                                logger.info("[UNTESTED PATH] go pointer receiver type")
                                for pp_child in p_child.children:
                                    if pp_child.type == 'type_identifier':
                                        metadata['receiver_type'] = pp_child.text.decode('utf8')
                            elif p_child.type == 'type_identifier':
                                logger.info("[UNTESTED PATH] go value receiver type")
                                metadata['receiver_type'] = p_child.text.decode('utf8')
                        break

        # Get method name for export check
        name = self._extract_node_name(method_node)
        if name and name[0].isupper():
            metadata['is_exported'] = True

        # Extract parameters and returns
        self._extract_function_signature(method_node, metadata)

        # Analyze method body
        self._analyze_function_body(method_node, metadata)

        return metadata

    def _extract_function_metadata(self, func_node: Any) -> Dict[str, Any]:
        """Extract Go function-specific metadata."""
        metadata: Dict[str, Any] = {
            'parameters': [],
            'return_types': [],
            'is_exported': False,
            'has_error_return': False,
            'is_variadic': False,
            'complexity': {'cyclomatic': 1, 'nesting_depth': 0},
            'patterns': {'anti': [], 'idiomatic': []},
            'uses_goroutines': False,
            'uses_channels': False,
            'defer_count': 0,
            'panic_count': 0,
            'recover_used': False,
        }

        # Get function name
        name = self._extract_node_name(func_node)
        if name and name[0].isupper():
            metadata['is_exported'] = True

        # Special function detection
        if name == 'main':
            metadata['is_main'] = True
        elif name == 'init':
            metadata['is_init'] = True

        # Extract signature
        self._extract_function_signature(func_node, metadata)

        # Analyze body
        self._analyze_function_body(func_node, metadata)

        return metadata

    def _extract_function_signature(self, node: Any, metadata: Dict[str, Any]) -> None:
        """Extract parameters and return types from function/method.
        Mejorado: si no hay nodo 'result', revisa si el último 'parameter_list' después de los parámetros contiene solo tipos (sin identificadores) y trátalo como tipos de retorno.
        """
        is_method = node.type == 'method_declaration'
        param_count = 0
        parameter_lists = []
        result_found = False

        for child in node.children:
            if child.type == 'parameter_list':
                parameter_lists.append(child)
                # Skip first parameter_list only for methods (it's the receiver)
                if not (is_method and param_count == 0):
                    if '...' in child.text.decode('utf8'):
                        metadata['is_variadic'] = True
                    for param in child.children:
                        if param.type == 'parameter_declaration':
                            param_info = self._extract_parameter_info(param)
                            if param_info:
                                metadata['parameters'].append(param_info)
                param_count += 1
            elif child.type == 'result':
                result_found = True
                if child.text:
                    return_text = child.text.decode('utf8').strip()
                    if return_text.startswith('('):
                        metadata['return_types'] = self._parse_return_types(return_text)
                    else:
                        metadata['return_types'] = [return_text]
                    if any(rt.strip() == 'error' for rt in metadata['return_types']):
                        metadata['has_error_return'] = True
        # Si no se encontró nodo 'result', revisa si el último parameter_list es de tipos de retorno
        if not result_found and len(parameter_lists) > 0:
            logger.info("[UNTESTED PATH] go return types from parameter_list")
            last_param_list = parameter_lists[-1]
            # Si todos los hijos son type_identifier, pointer_type, etc. (sin identifier), es return
            return_types = []
            only_types = True
            for param in last_param_list.children:
                if param.type == 'parameter_declaration':
                    has_identifier = any(c.type == 'identifier' for c in param.children)
                    if has_identifier:
                        only_types = False
                        break
                    # Extrae el tipo
                    for c in param.children:
                        if c.type in [
                            'type_identifier',
                            'pointer_type',
                            'slice_type',
                            'map_type',
                            'qualified_type',
                        ]:
                            logger.info("[UNTESTED PATH] go return type extraction from param")
                            return_types.append(c.text.decode('utf8'))
            if only_types and return_types:
                logger.info("[UNTESTED PATH] go setting return types from param list")
                metadata['return_types'] = return_types
                if any(rt.strip() == 'error' for rt in return_types):
                    metadata['has_error_return'] = True

    def _extract_parameter_info(self, param_node: Any) -> Optional[Dict[str, Any]]:
        """Extract parameter information."""
        param_info = {'name': '', 'type': '', 'variadic': False}

        # Check if parameter contains ... for variadic
        param_text = param_node.text.decode('utf8') if param_node.text else ''
        if '...' in param_text:
            param_info['variadic'] = True

        for child in param_node.children:
            if child.type == 'identifier':
                param_info['name'] = child.text.decode('utf8')
            elif child.type == 'variadic_type':
                logger.info("[UNTESTED PATH] go variadic type")
                param_info['variadic'] = True
                # Get the full type including ...
                param_info['type'] = child.text.decode('utf8')
            elif child.type in [
                'type_identifier',
                'pointer_type',
                'slice_type',
                'array_type',
                'qualified_type',
            ]:
                param_info['type'] = child.text.decode('utf8')

        return param_info if param_info['type'] else None

    def _parse_return_types(self, return_text: str) -> List[str]:
        """Parse return types from result string."""
        # Simple parsing - could be improved
        return_text = return_text.strip('()')
        types = []
        for part in return_text.split(','):
            part = part.strip()
            if part:
                types.append(part)
        return types

    def _analyze_function_body(self, node: Any, metadata: Dict[str, Any]) -> None:
        """Analyze function body for patterns and complexity."""

        def walk_body(n: Any, depth: int = 0) -> None:
            # Update max nesting depth
            metadata['complexity']['nesting_depth'] = max(
                metadata['complexity']['nesting_depth'], depth
            )

            # Count complexity-increasing constructs
            if n.type in ['if_statement', 'for_statement', 'switch_statement', 'expression_case']:
                metadata['complexity']['cyclomatic'] += 1

            # Detect goroutines
            if n.type == 'go_statement':
                metadata['uses_goroutines'] = True

            # Detect channels
            if n.type in [
                'channel_type',
                'send_statement',
                'receive_statement',
                'receive_operator',
            ]:
                metadata['uses_channels'] = True

            # Count defer statements
            if n.type == 'defer_statement':
                metadata['defer_count'] += 1

            # Count panic/recover
            if n.type == 'call_expression':
                for child in n.children:
                    if child.type == 'identifier':
                        text = child.text.decode('utf8')
                        if text == 'panic':
                            logger.info("[UNTESTED PATH] go panic detected")
                            metadata['panic_count'] = metadata.get('panic_count', 0) + 1
                        elif text == 'recover':
                            metadata['recover_used'] = True

            # Recurse
            for child in n.children:
                new_depth = (
                    depth + 1 if n.type in ['if_statement', 'for_statement', 'block'] else depth
                )
                walk_body(child, new_depth)

        # Find body block
        for child in node.children:
            if child.type == 'block':
                walk_body(child)
                break

        # Detect anti-patterns
        if metadata['complexity']['cyclomatic'] > 10:
            metadata['patterns']['anti'].append('high_complexity')  # type: ignore[attr-defined]
        if metadata['complexity']['nesting_depth'] > 4:
            metadata['patterns']['anti'].append('deep_nesting')  # type: ignore[attr-defined]
        if metadata.get('panic_count', 0) > 0 and not metadata.get('recover_used'):
            metadata['patterns']['anti'].append('unrecovered_panic')  # type: ignore[attr-defined]
        if metadata['defer_count'] > 5:
            logger.info("[UNTESTED PATH] go excessive defer")
            metadata['patterns']['anti'].append('excessive_defer')  # type: ignore[attr-defined]

        # Detect idiomatic patterns
        if (
            metadata['has_error_return']
            and metadata['return_types']
            and metadata['return_types'][-1] == 'error'
        ):
            metadata['patterns']['idiomatic'].append('error_last_return')  # type: ignore[attr-defined]
        if metadata['uses_channels'] and metadata['uses_goroutines']:
            logger.info("[UNTESTED PATH] go concurrent pattern")
            metadata['patterns']['idiomatic'].append('concurrent_pattern')  # type: ignore[attr-defined]

    def _extract_type_metadata(self, type_node: Any) -> Dict[str, Any]:
        """Extract struct/interface metadata."""
        metadata: Dict[str, Any] = {
            'is_exported': False,
            'fields': [],
            'methods': [],  # For interfaces
            'embeds': [],  # Embedded types
            'tags': {},  # Struct tags
            'implements': [],  # Interfaces this might implement
        }

        # Get type name
        name = self._extract_node_name(type_node)
        if name and name[0].isupper():
            metadata['is_exported'] = True

        # Extract fields or methods
        for child in type_node.children:
            if child.type == 'type_spec':
                for spec_child in child.children:
                    if spec_child.type == 'struct_type':
                        self._extract_struct_fields(spec_child, metadata)
                    elif spec_child.type == 'interface_type':
                        self._extract_interface_methods(spec_child, metadata)

        return metadata

    def _extract_struct_fields(self, struct_node: Any, metadata: Dict[str, Any]) -> None:
        """Extract fields from struct."""
        for child in struct_node.children:
            if child.type == 'field_declaration_list':
                for field in child.children:
                    if field.type == 'field_declaration':
                        field_info = self._extract_field_info(field)
                        if field_info:
                            metadata['fields'].append(field_info)  # type: ignore[attr-defined]
                            # Check for embedded types
                            if field_info.get('embedded'):
                                metadata['embeds'].append(field_info['type'])  # type: ignore[attr-defined]

    def _extract_field_info(self, field_node: Any) -> Optional[Dict[str, Any]]:
        """Extract field information including tags."""
        field_info: Dict[str, Any] = {'name': '', 'type': '', 'tag': '', 'is_exported': False}

        for child in field_node.children:
            if child.type == 'field_identifier':
                field_info['name'] = child.text.decode('utf8')
                if (
                    field_info['name']
                    and len(str(field_info['name'])) > 0
                    and str(field_info['name'])[0].isupper()
                ):
                    field_info['is_exported'] = True
            elif child.type in [
                'type_identifier',
                'pointer_type',
                'slice_type',
                'array_type',
                'qualified_type',
            ]:
                field_info['type'] = child.text.decode('utf8')
            elif child.type == 'raw_string_literal':
                field_info['tag'] = child.text.decode('utf8').strip('`')

        # For embedded fields (no name)
        if not field_info['name'] and field_info['type']:
            logger.info("[UNTESTED PATH] go embedded field")
            return {'type': field_info['type'], 'embedded': True}

        return field_info if field_info['type'] else None

    def _extract_interface_methods(self, interface_node: Any, metadata: Dict[str, Any]) -> None:
        """Extract method signatures from interface."""
        for child in interface_node.children:
            if child.type == 'method_spec':
                method_info = self._extract_method_spec(child)
                if method_info:
                    metadata['methods'].append(method_info)
            elif child.type == '{':
                # Interface body starts
                continue
            elif child.type == '}':
                # Interface body ends
                continue
            elif child.type in ['\n', 'comment']:
                # Skip whitespace and comments
                continue
            else:
                # Direct method spec without wrapper
                logger.info("[UNTESTED PATH] go direct method spec in interface")
                method_info = self._extract_method_spec(child)
                if method_info:
                    metadata['methods'].append(method_info)

    def _extract_method_spec(self, method_node: Any) -> Optional[Dict[str, Any]]:
        """Extract method specification from interface."""
        method_info = {'name': '', 'parameters': [], 'returns': []}

        if method_node.type == 'method_spec':
            for child in method_node.children:
                if child.type == 'field_identifier':
                    method_info['name'] = child.text.decode('utf8')
                elif child.type == 'parameter_list':
                    logger.info("[UNTESTED PATH] go method spec parameters")
                    method_info['parameters'] = child.text.decode('utf8')
                elif child.type == 'result':
                    logger.info("[UNTESTED PATH] go method spec result")
                    method_info['returns'] = child.text.decode('utf8')
        elif method_node.type == 'field_identifier':
            # Simple method without parameters
            logger.info("[UNTESTED PATH] go simple method without params")
            method_info['name'] = method_node.text.decode('utf8')

        return method_info if method_info['name'] else None

    def _extract_const_metadata(self, const_node: Any) -> Dict[str, Any]:
        """Extract constant metadata."""
        metadata: Dict[str, Any] = {
            'constants': [],
            'is_exported': False,
            'has_iota': False,
        }

        # Check text for iota
        if const_node.text and b'iota' in const_node.text:
            metadata['has_iota'] = True

        # Extract all constants in the declaration
        for child in const_node.children:
            if child.type == 'const_spec':
                const_info: Dict[str, Any] = {
                    'name': '',
                    'type': '',
                    'value': '',
                    'is_exported': False,
                }
                for spec_child in child.children:
                    if spec_child.type == 'identifier':
                        const_info['name'] = spec_child.text.decode('utf8')
                        if (
                            const_info['name']
                            and len(str(const_info['name'])) > 0
                            and str(const_info['name'])[0].isupper()
                        ):
                            const_info['is_exported'] = True
                            metadata['is_exported'] = True
                    elif spec_child.type == 'type_identifier':
                        const_info['type'] = spec_child.text.decode('utf8')
                    elif spec_child.type in [
                        'interpreted_string_literal',
                        'raw_string_literal',
                        'int_literal',
                        'float_literal',
                        'true',
                        'false',
                        'identifier',  # For iota
                    ]:
                        value_text = spec_child.text.decode('utf8')
                        const_info['value'] = value_text
                        if value_text == 'iota':
                            metadata['has_iota'] = True

                if const_info['name']:
                    metadata['constants'].append(const_info)

        return metadata

    def _add_common_metadata(self, chunk: Any, node: Any, lines: List[str]) -> None:
        """Add metadata common to multiple chunk types (e.g., comments, security)."""
        if not chunk.metadata.language_specific:
            chunk.metadata.language_specific = {}

        # Extract TODOs/FIXMEs from comments using the mixin
        todos = self._extract_todos(node)
        if todos:
            chunk.metadata.language_specific['todos'] = todos

        # Extract security issues using the mixin
        security_issues = self._detect_security_issues(node)
        if security_issues:
            chunk.metadata.language_specific['security'] = security_issues

        # Add all file imports to each chunk for context
        root_node = node
        while root_node.parent:
            root_node = root_node.parent

        import_declarations = [n for n in root_node.children if n.type == 'import_declaration']
        if import_declarations:
            # Assume one import block for simplicity, as is common
            categorized_imports = self._categorize_imports(import_declarations[0])
            chunk.metadata.language_specific.update(categorized_imports)

    def _categorize_imports(self, import_node: Any) -> Dict[str, List[str]]:
        """Categorize imports as internal/external/stdlib."""
        # Standard library packages (partial list, usar startswith para subpaquetes)
        stdlib_packages = [
            'fmt',
            'io',
            'os',
            'strings',
            'bytes',
            'errors',
            'time',
            'math',
            'net',
            'http',
            'json',
            'encoding',
            'crypto',
            'sync',
            'context',
            'database',
            'sql',
            'log',
            'regexp',
            'sort',
            'strconv',
            'testing',
        ]

        import_paths: List[str] = []

        # Use a tree-sitter query for the most robust node finding.
        query_text = "(import_spec) @spec"
        query = self._get_tree_sitter_language().query(query_text)
        captures = query.captures(import_node)

        for node, _ in captures:
            # The path is a direct child of the import_spec node
            for child in node.children:
                if child.type in ['interpreted_string_literal', 'raw_string_literal']:
                    path = child.text.decode('utf8').strip('"')
                    import_paths.append(path)
                    break

        categorized: Dict[str, List[str]] = {'internal': [], 'external': [], 'stdlib': []}
        for import_path in set(import_paths):  # Use set to avoid duplicates
            if import_path.startswith('./') or import_path.startswith('../'):
                categorized['internal'].append(import_path)
            elif any(
                import_path == pkg or import_path.startswith(pkg + '/') for pkg in stdlib_packages
            ):
                categorized['stdlib'].append(import_path)
            else:
                categorized['external'].append(import_path)

        return categorized

    def _get_import_node_types(self) -> List[str]:
        """Get node types that represent imports for Go."""
        return [
            'import_declaration',
            'import_spec',  # Individual import within import block
        ]

    def _is_comment_node(self, node: Any) -> bool:
        """Check if node is a Go comment."""
        return node.type in ['comment', 'line_comment', 'block_comment']

    def _extract_dependencies_from_imports(self, import_nodes: List[Any]) -> List[str]:
        """Override to handle Go import patterns."""
        deps = set()

        for node in import_nodes:
            text = node.text.decode('utf8')

            # Go imports are always quoted
            import_lines = text.split('\n')
            for line in import_lines:
                if '"' in line:
                    # Extract import path safely
                    parts = line.split('"')
                    if len(parts) >= 2:
                        import_path = parts[1]
                    else:
                        continue
                    # Take the first part as the main dependency
                    if '/' in import_path:
                        deps.add(import_path.split('/')[0])
                    else:
                        deps.add(import_path)

        return sorted(list(deps))

    # Override security patterns for Go-specific checks
    def _get_security_patterns(self) -> List[Callable[[Any, str], Optional[Dict[str, Any]]]]:
        """Get Go-specific security check functions."""
        return [
            self._check_sql_injection,
            self._check_hardcoded_credentials,
            self._check_weak_crypto,
            self._check_go_specific_security,
        ]

    def _check_weak_crypto(self, node: Any, text: str) -> Optional[Dict[str, Any]]:
        """Check for weak cryptographic practices."""
        # Go-specific weak crypto check
        if 'crypto/md5' in text or 'md5.New()' in text or 'md5.Sum' in text:
            return {
                'type': 'weak_crypto',
                'severity': 'medium',
                'description': 'MD5 is cryptographically weak',
            }

        # Check parent implementation
        return super()._check_weak_crypto(node, text)

    def _check_hardcoded_credentials(self, node: Any, text: str) -> Optional[Dict[str, Any]]:
        """Override to return 'hardcoded_credentials' as type."""
        result = super()._check_hardcoded_credentials(node, text)
        if result and result['type'] == 'hardcoded_credential':
            result['type'] = 'hardcoded_credentials'
        return result

    def _check_go_specific_security(self, node: Any, text: str) -> Optional[Dict[str, Any]]:
        """Check for Go-specific security issues."""
        # SQL injection with fmt.Sprintf
        if 'fmt.Sprintf' in text and any(
            sql in text.upper() for sql in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']
        ):
            return {
                'type': 'sql_injection_risk',
                'severity': 'high',
                'description': 'SQL query built with fmt.Sprintf',
            }

        # Command injection with exec.Command
        if 'exec.Command' in text:
            logger.info("[UNTESTED PATH] go exec.Command security issue")
            return {
                'type': 'command_injection_risk',
                'severity': 'high',
                'description': 'exec.Command usage requires careful input validation',
            }

        return None
