"""
Kotlin chunker using tree-sitter-languages.
Comprehensive metadata extraction for Kotlin/Android code.
"""

from typing import Dict, List, Any, Callable, Optional
from tree_sitter_languages import get_language  # type: ignore

from acolyte.models.chunk import ChunkType
from acolyte.core.logging import logger
from acolyte.rag.chunking.base import BaseChunker
from acolyte.rag.chunking.mixins import SecurityAnalysisMixin, PatternDetectionMixin


class KotlinChunker(BaseChunker, SecurityAnalysisMixin, PatternDetectionMixin):
    """
    Kotlin-specific chunker using tree-sitter.

    Handles:
    - Regular classes, data classes, sealed classes, objects
    - Interfaces with default implementations
    - Extension functions and properties
    - Coroutines (suspend functions)
    - Null safety annotations
    - Android-specific patterns
    """

    def _get_language_name(self) -> str:
        """Return language identifier."""
        return 'kotlin'

    def _get_import_node_types(self) -> List[str]:
        """Get node types that represent imports for kotlin."""
        return ['import_list', 'import_header']

    def _is_comment_node(self, node) -> bool:
        """Check if node is a comment."""
        return node.type in ['comment', 'line_comment', 'block_comment']

    def _get_tree_sitter_language(self) -> Any:
        """Get Kotlin language for tree-sitter."""
        return get_language('kotlin')

    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """
        Kotlin-specific node types to chunk.

        Kotlin AST nodes:
        - class_declaration (regular, data, sealed, enum)
        - object_declaration (object, companion object)
        - interface_declaration
        - function_declaration
        - property_declaration
        - import_list
        """
        return {
            # Classes and objects
            'class_declaration': ChunkType.CLASS,
            'object_declaration': ChunkType.CLASS,  # Kotlin objects are like singleton classes
            # Interfaces
            'interface_declaration': ChunkType.INTERFACE,
            # Functions
            'function_declaration': ChunkType.FUNCTION,
            'anonymous_function': ChunkType.FUNCTION,
            'lambda_literal': ChunkType.FUNCTION,
            # Properties (top-level)
            'property_declaration': ChunkType.PROPERTY,
            # Imports
            'import_list': ChunkType.IMPORTS,
            # Type aliases
            'type_alias': ChunkType.TYPES,
        }

    def _create_chunk_from_node(
        self, node, lines: List[str], file_path: str, chunk_type: ChunkType, processed_ranges
    ):
        """Override to handle Kotlin-specific cases."""
        # Determine if it's a special type of class
        if node.type == 'class_declaration':
            chunk_type = self._determine_class_type(node)

        # Check if function is special (constructor, init block, etc.)
        elif node.type == 'function_declaration':
            chunk_type = self._determine_function_type(node)

        # Standard processing
        chunk = super()._create_chunk_from_node(
            node, lines, file_path, chunk_type, processed_ranges
        )

        # Add Kotlin-specific metadata
        if chunk:
            if chunk.metadata.chunk_type in [
                ChunkType.FUNCTION,
                ChunkType.METHOD,
                ChunkType.CONSTRUCTOR,
                ChunkType.TESTS,
            ]:
                chunk.metadata.language_specific = self._extract_function_metadata(node)
            elif chunk.metadata.chunk_type == ChunkType.CLASS:
                chunk.metadata.language_specific = self._extract_class_metadata(node)
            elif chunk.metadata.chunk_type == ChunkType.PROPERTY:
                chunk.metadata.language_specific = self._extract_property_metadata(node)
            elif chunk.metadata.chunk_type == ChunkType.IMPORTS:
                # Extract dependencies for import chunks
                logger.info("[UNTESTED PATH] kotlin imports chunk")
                chunk.metadata.language_specific = {
                    'dependencies': self._extract_dependencies_from_imports([chunk])
                }

        return chunk

    def _determine_class_type(self, class_node) -> ChunkType:
        """Determine specific type of Kotlin class."""
        # Check if it's inside another class (nested/inner)
        parent = class_node.parent
        while parent:
            if parent.type == 'class_declaration':
                return ChunkType.CLASS  # Nested class
            parent = parent.parent

        return ChunkType.CLASS

    def _determine_function_type(self, func_node) -> ChunkType:
        """Determine if function is method, constructor, etc."""
        # Get function name
        name = self._extract_node_name(func_node)

        # Check if it's a constructor
        if name == 'constructor':
            return ChunkType.CONSTRUCTOR

        # Check if it's an init block
        if name == 'init':
            return ChunkType.CONSTRUCTOR

        # Check if inside a class/object/interface
        parent = func_node.parent
        while parent:
            if parent.type in ['class_declaration', 'object_declaration', 'interface_declaration']:
                return ChunkType.METHOD
            parent = parent.parent

        # Check for test functions
        if name and (name.startswith('test') or 'Test' in name):
            # Check for @Test annotation
            for child in func_node.children:
                if child.type == 'modifiers':
                    if self._has_annotation(child, ['Test', 'ParameterizedTest']):
                        return ChunkType.TESTS

        return ChunkType.FUNCTION

    def _extract_function_metadata(self, func_node) -> Dict[str, Any]:
        """Extract Kotlin-specific function metadata."""
        metadata = {
            # Basic info
            'modifiers': [],
            'visibility': 'public',  # Default in Kotlin
            'is_abstract': False,
            'is_suspend': False,
            'is_inline': False,
            'is_operator': False,
            'is_infix': False,
            'is_extension': False,
            'parameters': [],
            'return_type': None,
            'receiver_type': None,  # For extension functions
            'annotations': [],
            # Complexity
            'complexity': {'cyclomatic': 1, 'nesting_depth': 0, 'lines_of_code': 0, 'branches': 0},
            # Patterns
            'patterns': {'anti': [], 'framework': []},
            # TO-DOs
            'todos': [],
            # Quality
            'quality': {'has_docstring': False, 'test_coverage_hints': []},
            # Security
            'security': [],
            # Dependencies
            'dependencies': {'internal': [], 'external': []},
        }

        # Extract function text to check for modifiers
        func_text = func_node.text.decode('utf8')
        first_line = func_text.split('\n')[0]

        # Check for modifiers in the function declaration text
        if 'suspend fun' in first_line or first_line.startswith('suspend '):
            logger.info("[UNTESTED PATH] kotlin suspend function")
            metadata['is_suspend'] = True
            metadata['modifiers'].append('suspend')
        if 'operator fun' in first_line or ' operator ' in first_line:
            logger.info("[UNTESTED PATH] kotlin operator function")
            metadata['is_operator'] = True
            metadata['modifiers'].append('operator')
        if 'inline fun' in first_line or first_line.startswith('inline '):
            logger.info("[UNTESTED PATH] kotlin inline function")
            metadata['is_inline'] = True
            metadata['modifiers'].append('inline')
        if 'infix fun' in first_line or ' infix ' in first_line:
            logger.info("[UNTESTED PATH] kotlin infix function")
            metadata['is_infix'] = True
            metadata['modifiers'].append('infix')
        if 'abstract fun' in first_line or first_line.startswith('abstract '):
            logger.info("[UNTESTED PATH] kotlin abstract function")
            metadata['is_abstract'] = True
            metadata['modifiers'].append('abstract')

        # Check for extension function by looking at the function name structure
        # Extension functions have the receiver type as part of the name
        # func_name_node = None  # Eliminado porque no se usa

        # Extract modifiers and annotations
        for child in func_node.children:
            # Modifiers can be direct children in Kotlin
            if child.type == 'suspend':
                logger.info("[UNTESTED PATH] kotlin suspend modifier node")
                metadata['is_suspend'] = True
                metadata['modifiers'].append('suspend')
            elif child.type == 'inline':
                logger.info("[UNTESTED PATH] kotlin inline modifier node")
                metadata['is_inline'] = True
                metadata['modifiers'].append('inline')
            elif child.type == 'operator':
                logger.info("[UNTESTED PATH] kotlin operator modifier node")
                metadata['is_operator'] = True
                metadata['modifiers'].append('operator')
            elif child.type == 'infix':
                logger.info("[UNTESTED PATH] kotlin infix modifier node")
                metadata['is_infix'] = True
                metadata['modifiers'].append('infix')
            elif child.type == 'abstract':
                logger.info("[UNTESTED PATH] kotlin abstract modifier node")
                metadata['is_abstract'] = True
                metadata['modifiers'].append('abstract')
            elif child.type == 'modifiers':
                modifiers = metadata['modifiers']
                extracted = self._extract_modifiers(child)
                if isinstance(extracted, list):
                    modifiers.extend(extracted)
                metadata['visibility'] = self._extract_visibility(child)
                metadata['annotations'] = self._extract_annotations(child)

            # Look for receiver type in function signature
            elif child.type == 'simple_identifier' and child.text:
                # For extension functions, check the parent structure
                # Extension functions in Kotlin have a specific pattern in the AST
                parent = func_node.parent
                if parent:
                    # Look through all children to find receiver type pattern
                    children = list(func_node.children)
                    for i, c in enumerate(children):
                        if c == child and i > 0:
                            # Search backwards to find the receiver_type node before the function identifier
                            j = i - 1
                            while j >= 0:
                                if children[j].type in [
                                    '.',
                                    'type',
                                    'user_type',
                                    'nullable_type',
                                    'type_reference',
                                ]:
                                    metadata['is_extension'] = True
                                    metadata['receiver_type'] = children[j].text.decode('utf8')
                                    break
                                elif (
                                    children[j].type not in ['\n', ' ', '\t', '\r']
                                    and children[j].text.decode('utf8').strip()
                                ):
                                    # If we find another type of relevant node, stop the search
                                    logger.info(
                                        "[UNTESTED PATH] kotlin extension function detection stop"
                                    )
                                    break
                                j -= 1
            # Extract parameters
            elif child.type == 'function_value_parameters':
                metadata['parameters'] = self._extract_parameters(child)

            # Extract return type
            elif child.type == 'function_type':
                # Return type is in function_type node
                for subchild in child.children:
                    if subchild.type == 'type' or subchild.type == 'nullable_type':
                        metadata['return_type'] = subchild.text.decode('utf8')

            # Extract function body for analysis
            elif child.type == 'function_body':
                # Calculate complexity using mixin
                metadata['complexity'] = self._calculate_complexity(child)
                # Extract TODOs using mixin
                metadata['todos'] = self._extract_todos(child)
                # Detect patterns using mixin with metadata - pass metadata directly
                detected_patterns = self._detect_patterns(child, metadata)
                # Merge with language-specific patterns
                lang_patterns = self._detect_language_patterns(child, detected_patterns, metadata)
                metadata['patterns'] = lang_patterns
                # Check for security issues using mixin
                metadata['security'] = self._detect_security_issues(child)

        return metadata

    def _extract_class_metadata(self, class_node) -> Dict[str, Any]:
        """Extract Kotlin-specific class metadata."""
        metadata = {
            'modifiers': [],
            'visibility': 'public',
            'is_data_class': False,
            'is_sealed_class': False,
            'is_enum_class': False,
            'is_inner': False,
            'is_companion_object': False,
            'is_object': class_node.type == 'object_declaration',
            'primary_constructor': None,
            'properties': [],
            'methods': [],
            'nested_classes': [],
            'super_types': [],
            'annotations': [],
            'has_docstring': False,
        }

        # Extract class text to check for modifiers
        class_text = class_node.text.decode('utf8')
        first_line = class_text.split('\n')[0]

        # Check for modifiers in the class declaration text
        if first_line.startswith('data class'):
            logger.info("[UNTESTED PATH] kotlin data class")
            metadata['is_data_class'] = True
            metadata['modifiers'].append('data')
        if first_line.startswith('sealed class'):
            metadata['is_sealed_class'] = True
            metadata['modifiers'].append('sealed')
        if first_line.startswith('enum class'):
            metadata['is_enum_class'] = True
            metadata['modifiers'].append('enum')
        if 'inner class' in first_line:
            metadata['is_inner'] = True
            metadata['modifiers'].append('inner')
        if 'companion object' in class_text[:100]:  # Check in first 100 chars
            metadata['is_companion_object'] = True
            metadata['modifiers'].append('companion')

        # In Kotlin, modifiers like 'data', 'sealed' are direct children
        for child in class_node.children:
            # Check for class modifiers as direct children
            if child.type == 'data':
                logger.info("[UNTESTED PATH] kotlin data modifier node")
                metadata['is_data_class'] = True
                metadata['modifiers'].append('data')
            elif child.type == 'sealed':
                logger.info("[UNTESTED PATH] kotlin sealed modifier node")
                metadata['is_sealed_class'] = True
                metadata['modifiers'].append('sealed')
            elif child.type == 'enum':
                metadata['is_enum_class'] = True
                metadata['modifiers'].append('enum')
            elif child.type == 'inner':
                metadata['is_inner'] = True
                metadata['modifiers'].append('inner')
            elif child.type == 'companion':
                metadata['is_companion_object'] = True
                metadata['modifiers'].append('companion')
            elif child.type == 'modifiers':
                modifiers = metadata['modifiers']
                extracted = self._extract_modifiers(child)
                if isinstance(extracted, list):
                    modifiers.extend(extracted)
                metadata['visibility'] = self._extract_visibility(child)
                metadata['annotations'] = self._extract_annotations(child)

            # Extract primary constructor
            elif child.type == 'primary_constructor':
                metadata['primary_constructor'] = self._extract_primary_constructor(child)

            # Sometimes constructor params are directly in class_declaration
            elif child.type == 'class_parameter' or child.type == 'function_value_parameters':
                # This is the primary constructor parameters
                if not metadata['primary_constructor']:
                    params = (
                        self._extract_parameters(child)
                        if child.type == 'function_value_parameters'
                        else []
                    )
                    metadata['primary_constructor'] = {'visibility': 'public', 'parameters': params}
                    # Also extract properties from constructor parameters
                    # In Kotlin, val/var in constructor params become properties
                    for param_child in child.children:
                        if param_child.type == 'parameter':
                            for p in param_child.children:
                                if p.type in ['val', 'var']:
                                    # This parameter is also a property
                                    prop_name = None
                                    for sibling in param_child.children:
                                        if sibling.type == 'simple_identifier':
                                            prop_name = sibling.text.decode('utf8')
                                            logger.info(
                                                "[UNTESTED PATH] kotlin property from constructor param"
                                            )
                                            break
                                    if prop_name and prop_name not in metadata['properties']:
                                        metadata['properties'].append(prop_name)

            # Extract supertype list
            elif child.type == 'delegation_specifiers':
                logger.info("[UNTESTED PATH] kotlin delegation specifiers")
                metadata['super_types'] = self._extract_super_types(child)

            # Extract class body
            elif child.type == 'class_body':
                for member in child.children:
                    if member.type == 'property_declaration':
                        prop_name = self._extract_node_name(member)
                        if prop_name:
                            metadata['properties'].append(prop_name)
                    elif member.type == 'function_declaration':
                        method_name = self._extract_node_name(member)
                        if method_name:
                            metadata['methods'].append(method_name)
                    elif member.type in ['class_declaration', 'object_declaration']:
                        nested_name = self._extract_node_name(member)
                        if nested_name:
                            metadata['nested_classes'].append(nested_name)

        # Check for KDoc
        metadata['has_docstring'] = self._has_kdoc(class_node)

        return metadata

    def _extract_property_metadata(self, prop_node) -> Dict[str, Any]:
        """Extract Kotlin property metadata."""
        metadata = {
            'modifiers': [],
            'visibility': 'public',
            'is_var': True,  # vs val
            'is_lateinit': False,
            'is_const': False,
            'is_override': False,
            'type': None,
            'initializer': None,
            'has_getter': False,
            'has_setter': False,
            'annotations': [],
        }

        for child in prop_node.children:
            if child.type == 'modifiers':
                metadata['modifiers'] = self._extract_modifiers(child)
                metadata['visibility'] = self._extract_visibility(child)
                modifiers_list = metadata['modifiers']
                if isinstance(modifiers_list, list):
                    metadata['is_lateinit'] = 'lateinit' in modifiers_list
                    metadata['is_const'] = 'const' in modifiers_list
                    metadata['is_override'] = 'override' in modifiers_list
                metadata['annotations'] = self._extract_annotations(child)

            elif child.type in ['val', 'var']:
                # Kotlin property declarations start with val/var
                for subchild in child.children:
                    if subchild.type == 'type':
                        metadata['type'] = subchild.text.decode('utf8')
                metadata['is_var'] = child.type == 'var'

            # Check for custom getter/setter
            elif child.type == 'getter':
                logger.info("[UNTESTED PATH] kotlin property getter")
                metadata['has_getter'] = True
            elif child.type == 'setter':
                logger.info("[UNTESTED PATH] kotlin property setter")
                metadata['has_setter'] = True

        return metadata

    def _extract_modifiers(self, modifiers_node) -> List[str]:
        """Extract all modifiers from a modifiers node."""
        modifiers = []
        for child in modifiers_node.children:
            if child.type in [
                'visibility_modifier',
                'member_modifier',
                'function_modifier',
                'property_modifier',
                'inheritance_modifier',
                'parameter_modifier',
                'platform_modifier',
            ]:
                modifiers.append(child.text.decode('utf8'))
        return modifiers

    def _extract_visibility(self, modifiers_node) -> str:
        """Extract visibility from modifiers."""
        for child in modifiers_node.children:
            if child.type == 'visibility_modifier':
                return child.text.decode('utf8')
        return 'public'  # Default in Kotlin

    def _extract_annotations(self, modifiers_node) -> List[str]:
        """Extract annotations."""
        annotations = []
        for child in modifiers_node.children:
            if child.type == 'annotation':
                annotations.append(child.text.decode('utf8'))
        return annotations

    def _has_annotation(self, modifiers_node, names: List[str]) -> bool:
        """Check if any of the given annotations exist."""
        annotations = self._extract_annotations(modifiers_node)
        return any(any(name in ann for ann in annotations) for name in names)

    def _extract_parameters(self, params_node) -> List[Dict[str, Any]]:
        """Extract function parameters with types and defaults."""
        parameters = []

        for child in params_node.children:
            if child.type == 'parameter':
                param: Dict[str, Any] = {
                    'name': None,
                    'type': None,
                    'optional': False,
                    'default': None,
                    'is_vararg': False,
                }

                for subchild in child.children:
                    if subchild.type == 'simple_identifier':
                        param['name'] = subchild.text.decode('utf8')
                    elif subchild.type == 'type':
                        param_type = subchild.text.decode('utf8')
                        param['type'] = param_type
                        # Check if nullable type
                        if param_type and isinstance(param_type, str) and '?' in param_type:
                            param['optional'] = True
                    elif subchild.type == 'parameter_modifiers':
                        modifier_text = subchild.text.decode('utf8')
                        if isinstance(modifier_text, str) and 'vararg' in modifier_text:
                            param['is_vararg'] = True
                    elif subchild.type == 'expression':
                        param['default'] = subchild.text.decode('utf8')
                        param['optional'] = True

                if param['name']:
                    parameters.append(param)

        return parameters

    def _extract_primary_constructor(self, constructor_node) -> Dict[str, Any]:
        """Extract primary constructor info."""
        visibility = 'public'  # Default visibility
        parameters = []

        # Find modifiers child if exists
        for child in constructor_node.children:
            if child.type == 'modifiers':
                visibility = self._extract_visibility(child)
            elif child.type == 'function_value_parameters':
                parameters = self._extract_parameters(child)

        return {
            'visibility': visibility,
            'parameters': parameters,
        }

    def _extract_super_types(self, delegation_node) -> List[str]:
        """Extract superclass and interfaces."""
        super_types = []
        for child in delegation_node.children:
            if child.type == 'delegation_specifier':
                for subchild in child.children:
                    if subchild.type == 'user_type':
                        super_types.append(subchild.text.decode('utf8'))
        return super_types

    def _extract_node_name(self, node) -> Optional[str]:
        """Extrae el nombre de nodos Kotlin, usando primero la lógica base y luego patrones específicos de Kotlin."""
        # Primero intenta con la lógica base
        name = super()._extract_node_name(node)
        if name:
            return name

        # Si no encuentra, busca específicamente en Kotlin
        for child in node.children:
            if child.type == 'simple_identifier':
                return child.text.decode('utf8')
            elif child.type == 'type_identifier':
                return child.text.decode('utf8')

        # Para algunos nodos, el nombre puede estar anidado más profundo
        for child in node.children:
            for grandchild in child.children:
                if grandchild.type == 'simple_identifier':
                    return grandchild.text.decode('utf8')
                elif grandchild.type == 'type_identifier':
                    return grandchild.text.decode('utf8')

        return None

    def _get_decision_node_types(self) -> set:
        """Override for Kotlin-specific decision nodes."""
        kotlin_nodes = {
            'if_expression',
            'when_expression',
            'when_entry',
            'while_statement',
            'do_while_statement',
            'for_statement',
            'catch_block',
            'elvis_expression',
            'conjunction',
            'disjunction',
        }
        return kotlin_nodes | super()._get_decision_node_types()

    def _detect_language_patterns(
        self, node, patterns: Dict[str, List[str]], metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[str]]:
        """Override to add Kotlin-specific patterns."""
        # Use provided metadata or empty dict
        if metadata is None:
            metadata = {}
        annotations = metadata.get('annotations', [])
        if not isinstance(annotations, list):
            annotations = []

        # Android patterns
        if isinstance(annotations, list):
            for ann in annotations:
                if isinstance(ann, str):
                    if '@Composable' in ann or '@Preview' in ann:
                        patterns['framework'].append('jetpack_compose')
                        break

            for ann in annotations:
                if isinstance(ann, str):
                    if '@AndroidEntryPoint' in ann or '@HiltViewModel' in ann:
                        patterns['framework'].append('hilt_injection')
                        break

        # Coroutines
        is_suspend = metadata.get('is_suspend', False)
        if is_suspend:
            patterns['framework'].append('coroutines')

        # Spring patterns
        if isinstance(annotations, list):
            for ann in annotations:
                if isinstance(ann, str) and '@' in ann:
                    if any(s in ann for s in ['Service', 'Repository', 'Controller', 'Component']):
                        patterns['framework'].append('spring_component')
                        break

        return patterns

    def _get_security_patterns(self) -> List[Callable[[Any, str], Optional[Dict[str, Any]]]]:
        """Override to add Kotlin-specific security checks."""
        base_patterns = super()._get_security_patterns()
        return base_patterns + [self._check_unsafe_null_assertion]

    def _check_unsafe_null_assertion(self, node, text: str) -> Optional[Dict[str, Any]]:
        """Check for unsafe !! null assertions."""
        # Check in the text directly for !! pattern
        if '!!' in text and node.type in [
            'postfix_expression',
            'postfix_unary_expression',
            'expression',
        ]:
            logger.info("[UNTESTED PATH] kotlin unsafe null assertion")
            return {
                'type': 'unsafe_null_assertion',
                'severity': 'medium',
                'description': 'Unsafe null assertion can cause runtime crashes',
            }
        return None

    def _has_kdoc(self, node) -> bool:
        """Check if node has KDoc documentation."""
        # Look for KDoc comment before the node
        parent = node.parent
        if parent:
            # Find the index of the current node
            node_index = None
            for i, child in enumerate(parent.children):
                if child == node:
                    node_index = i
                    break

            if node_index is not None and node_index > 0:
                # Only check the immediately preceding node (ignoring whitespace)
                # KDoc must be directly before the declaration
                for i in range(node_index - 1, -1, -1):
                    prev_child = parent.children[i]
                    # Ignore whitespace nodes
                    if (
                        prev_child.type in ['\n', ' ', '\t', '\r']
                        or not prev_child.text.decode('utf8').strip()
                    ):
                        continue
                    # Check if it's a multiline comment
                    if prev_child.type == 'multiline_comment':
                        # Check if it's KDoc (starts with /**)
                        comment_text = prev_child.text.decode('utf8').strip()
                        return comment_text.startswith('/**') and comment_text.endswith('*/')
                    else:
                        # Found something else, no KDoc
                        return False
        return False

    def _extract_dependencies_from_imports(self, import_nodes) -> List[str]:
        """Extract Kotlin import dependencies."""
        deps = set()

        for node in import_nodes:
            # Handle both import_list nodes and import chunks
            nodes_to_check = []
            if hasattr(node, 'type'):
                if node.type == 'import_list':
                    nodes_to_check = node.children
                else:
                    # It's probably a chunk, extract from content
                    if hasattr(node, 'content'):
                        # Parse import statements from content
                        lines = node.content.split('\n')
                        for line in lines:
                            line = line.strip()
                            if line.startswith('import '):
                                import_path = line.replace('import ', '').strip()
                                # Handle alias imports
                                if ' as ' in import_path:
                                    import_path = import_path.split(' as ')[0].strip()
                                # Take root package
                                if import_path:
                                    root_package = import_path.split('.')[0]
                                    # Common Kotlin/Android packages are external
                                    if root_package in [
                                        'kotlin',
                                        'java',
                                        'javax',
                                        'android',
                                        'androidx',
                                        'kotlinx',
                                        'org',
                                        'com',
                                        'io',
                                    ]:
                                        deps.add(root_package)
                                    else:
                                        # Likely internal
                                        logger.info("[UNTESTED PATH] kotlin internal import")
                                        deps.add(import_path)
                        continue

            # Process nodes from import_list
            for child in nodes_to_check:
                if child.type == 'import_header':
                    text = child.text.decode('utf8')
                    # Remove 'import' keyword
                    import_path = text.replace('import', '').strip()
                    # Handle alias imports
                    if ' as ' in import_path:
                        import_path = import_path.split(' as ')[0].strip()
                    # Take root package
                    if import_path:
                        root_package = import_path.split('.')[0]
                        # Common Kotlin/Android packages are external
                        if root_package in [
                            'kotlin',
                            'java',
                            'javax',
                            'android',
                            'androidx',
                            'kotlinx',
                            'org',
                            'com',
                            'io',
                        ]:
                            deps.add(root_package)
                        else:
                            # Likely internal
                            logger.info("[UNTESTED PATH] kotlin internal import from list")
                            deps.add(import_path)

        return sorted(list(deps))
