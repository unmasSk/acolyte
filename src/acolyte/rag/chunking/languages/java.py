"""
Java chunker using tree-sitter-languages.
Extracts comprehensive metadata for intelligent code search.
"""

from typing import Dict, List, Any, Optional
from tree_sitter_languages import get_language  # type: ignore

from acolyte.models.chunk import ChunkType
from acolyte.core.logging import logger
from acolyte.rag.chunking.base import BaseChunker
from acolyte.rag.chunking.mixins import PatternDetectionMixin


class JavaChunker(BaseChunker, PatternDetectionMixin):
    """
    Java-specific chunker using tree-sitter.

    Extracts rich metadata including:
    - Annotations (@Override, @Deprecated, @Test, etc.)
    - Access modifiers and visibility
    - Generics and type parameters
    - Exception handling
    - Complexity metrics
    - Common Java anti-patterns
    - Security vulnerabilities
    """

    def _get_language_name(self) -> str:
        """Return language identifier."""
        return 'java'

    def _get_tree_sitter_language(self) -> Any:
        """Get Java language for tree-sitter."""
        logger.info("[UNTESTED PATH] java._get_tree_sitter_language")
        return get_language('java')

    def _get_decision_node_types(self):
        """Override to add Java-specific decision nodes."""
        java_nodes = {
            'switch_expression',
            'conditional_expression',
            'do_statement',
            'enhanced_for_statement',
            'synchronized_statement',
            'assert_statement',
        }
        return super()._get_decision_node_types() | java_nodes

    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """
        Java-specific node types to chunk.

        Tree-sitter Java node types:
        - class_declaration: Regular classes
        - interface_declaration: Interfaces
        - enum_declaration: Enums
        - annotation_type_declaration: @interface definitions
        - method_declaration: Methods
        - constructor_declaration: Constructors
        - field_declaration: Class fields
        - import_declaration: Import statements
        - package_declaration: Package declaration
        """
        return {
            # Classes and types
            'class_declaration': ChunkType.CLASS,
            'interface_declaration': ChunkType.INTERFACE,
            'enum_declaration': ChunkType.CLASS,  # Treat enums as classes
            'annotation_type_declaration': ChunkType.INTERFACE,  # @interface
            'record_declaration': ChunkType.CLASS,  # Java 14+ records
            # Methods
            'method_declaration': ChunkType.METHOD,
            'constructor_declaration': ChunkType.CONSTRUCTOR,
            # Fields (for constants)
            'field_declaration': ChunkType.CONSTANTS,  # Will refine based on modifiers
            # Imports and package
            'import_declaration': ChunkType.IMPORTS,
            'package_declaration': ChunkType.MODULE,
        }

    def _create_chunk_from_node(
        self, node, lines: List[str], file_path: str, chunk_type: ChunkType, processed_ranges
    ):
        """Override to handle Java-specific cases and extract rich metadata."""
        # Special handling for fields - determine if constant
        if node.type == 'field_declaration':
            modifiers = self._extract_modifiers(node)
            if 'static' in modifiers and 'final' in modifiers:
                chunk_type = ChunkType.CONSTANTS
            else:
                # Regular fields don't need chunks
                return None

        # Create base chunk
        chunk = super()._create_chunk_from_node(
            node, lines, file_path, chunk_type, processed_ranges
        )

        if not chunk:
            return None

        # Extract comprehensive metadata
        metadata: Dict[str, Any] = {}

        # Basic metadata
        modifiers = self._extract_modifiers(node)
        metadata['modifiers'] = modifiers
        metadata['visibility'] = self._determine_visibility(modifiers)
        metadata['is_abstract'] = 'abstract' in modifiers
        metadata['is_static'] = 'static' in modifiers
        metadata['is_final'] = 'final' in modifiers

        # Annotations
        annotations = self._extract_annotations(node)
        if annotations:
            logger.info("[UNTESTED PATH] java annotations detected")
            metadata['annotations'] = annotations

        # Type-specific metadata
        if chunk_type == ChunkType.CLASS:
            metadata.update(self._extract_class_metadata(node))
        elif chunk_type in [ChunkType.METHOD, ChunkType.CONSTRUCTOR]:
            metadata.update(self._extract_method_metadata(node))
        elif chunk_type == ChunkType.INTERFACE:
            metadata.update(self._extract_interface_metadata(node))

        # Calculate complexity using mixin
        if chunk_type in [ChunkType.METHOD, ChunkType.CONSTRUCTOR]:
            metadata['complexity'] = self._calculate_complexity(node)

        # Detect patterns AND frameworks
        annotations = self._extract_annotations(node)
        patterns = self._detect_patterns(node, metadata)

        # Extract frameworks from annotations
        frameworks = set()
        for ann in annotations:
            if any(
                spring in ann
                for spring in [
                    '@SpringBoot',
                    '@Controller',
                    '@Service',
                    '@Repository',
                    '@Component',
                    '@RestController',
                    '@Autowired',
                ]
            ):
                frameworks.add('spring')
            if any(
                test in ann for test in ['@Test', '@ParameterizedTest', '@BeforeEach', '@AfterEach']
            ):
                frameworks.add('junit')
            if '@Entity' in ann or '@Table' in ann:
                frameworks.add('jpa')
            if any(
                lombok in ann
                for lombok in ['@Data', '@Getter', '@Setter', '@RequiredArgsConstructor']
            ):
                frameworks.add('lombok')

        if frameworks:
            metadata['frameworks'] = list(frameworks)

        if patterns['anti'] or patterns['framework']:
            metadata['patterns'] = patterns

        # Extract TODOs using mixin
        todos = self._extract_todos(node)
        if todos:
            metadata['todos'] = todos

        # Quality indicators
        metadata['quality'] = self._assess_quality(node, chunk_type)

        # Security vulnerabilities
        security_issues = self._detect_security_issues(node, lines)
        if security_issues:
            metadata['security'] = security_issues

        # Dependencies (for classes/interfaces)
        if chunk_type in [ChunkType.CLASS, ChunkType.INTERFACE]:
            dependencies = self._analyze_dependencies(node)
            metadata['dependencies'] = dependencies

        # Set the metadata
        chunk.metadata.language_specific = metadata

        return chunk

    def _extract_modifiers(self, node) -> List[str]:
        """Extract all modifiers from a declaration."""
        modifiers = []

        # Look for modifiers child
        for child in node.children:
            if child.type == 'modifiers':
                for modifier in child.children:
                    if modifier.type in [
                        'public',
                        'private',
                        'protected',
                        'static',
                        'final',
                        'abstract',
                        'synchronized',
                        'volatile',
                        'transient',
                        'native',
                        'strictfp',
                        'default',
                    ]:
                        modifiers.append(modifier.type)
                    elif modifier.type == 'annotation':
                        # Skip annotations here, handle separately
                        pass

        return modifiers

    def _determine_visibility(self, modifiers: List[str]) -> str:
        """Determine visibility from modifiers."""
        for visibility in ['public', 'protected', 'private']:
            if visibility in modifiers:
                return visibility
        return 'package'  # Default Java visibility

    def _extract_annotations(self, node) -> List[str]:
        """Extract all annotations from a node."""
        annotations = []

        # Look in modifiers
        for child in node.children:
            if child.type == 'modifiers':
                for modifier in child.children:
                    if modifier.type == 'annotation' or modifier.type == 'marker_annotation':
                        ann_text = modifier.text.decode('utf8').strip()
                        annotations.append(ann_text)

        return annotations

    def _extract_class_metadata(self, node) -> Dict[str, Any]:
        """Extract Java class-specific metadata."""
        metadata: Dict[str, Any] = {
            'implements': [],
            'extends': None,
            'type_parameters': [],
            'nested_classes': [],
            'methods': [],
            'fields': [],
            'is_inner_class': False,
            'generics': False,
        }

        # Check if inner class
        parent = node.parent
        while parent:
            if parent.type == 'class_body':
                logger.info("[UNTESTED PATH] java inner class detected")
                metadata['is_inner_class'] = True
                break
            parent = parent.parent

        for child in node.children:
            # Superclass
            if child.type == 'superclass':
                for subchild in child.children:
                    if subchild.type == 'type_identifier':
                        metadata['extends'] = subchild.text.decode('utf8')

            # Interfaces
            elif child.type == 'super_interfaces':
                for interface in child.children:
                    if interface.type == 'type_list':
                        for type_node in interface.children:
                            if type_node.type == 'type_identifier':
                                metadata['implements'].append(type_node.text.decode('utf8'))

            # Check for generics in class declaration
            if child.type == 'type_parameters':
                metadata['type_parameters'] = self._extract_type_parameters(child)
                if metadata['type_parameters']:
                    metadata['generics'] = True

            # Class body
            elif child.type == 'class_body':
                for member in child.children:
                    if member.type == 'method_declaration':
                        method_name = self._extract_node_name(member)
                        if method_name:
                            metadata['methods'].append(method_name)
                    elif member.type == 'field_declaration':
                        field_names = self._extract_field_names(member)
                        metadata['fields'].extend(field_names)
                    elif member.type in ['class_declaration', 'interface_declaration']:
                        nested_name = self._extract_node_name(member)
                        if nested_name:
                            metadata['nested_classes'].append(nested_name)

        return metadata

    def _extract_method_metadata(self, node) -> Dict[str, Any]:
        """Extract Java method-specific metadata."""
        metadata: Dict[str, Any] = {
            'parameters': [],
            'return_type': None,
            'throws': [],
            'type_parameters': [],
            'is_synchronized': False,
            'is_native': False,
            'generics': False,
        }

        modifiers = self._extract_modifiers(node)
        metadata['is_synchronized'] = 'synchronized' in modifiers
        metadata['is_native'] = 'native' in modifiers

        for child in node.children:
            # Return type
            if child.type in [
                'type_identifier',
                'void_type',
                'generic_type',
                'array_type',
                'integral_type',
                'floating_point_type',
            ]:
                metadata['return_type'] = child.text.decode('utf8')

            # Type parameters for methods
            elif child.type == 'type_parameters':
                metadata['type_parameters'] = self._extract_type_parameters(child)
                if metadata['type_parameters']:
                    metadata['generics'] = True

            # Parameters
            elif child.type == 'formal_parameters':
                metadata['parameters'] = self._extract_parameters(child)

            # Throws clause
            elif child.type == 'throws':
                for throw_child in child.children:
                    if throw_child.type == 'type_identifier':
                        metadata['throws'].append(throw_child.text.decode('utf8'))

        return metadata

    def _extract_interface_metadata(self, node) -> Dict[str, Any]:
        """Extract Java interface-specific metadata."""
        metadata: Dict[str, Any] = {
            'extends': [],  # Interfaces can extend multiple interfaces
            'type_parameters': [],
            'methods': [],
            'constants': [],
            'default_methods': [],
            'static_methods': [],
            'generics': False,
        }

        for child in node.children:
            # Extended interfaces
            if child.type == 'extends_interfaces':
                for interface in child.children:
                    if interface.type == 'type_list':
                        for type_node in interface.children:
                            if type_node.type == 'type_identifier':
                                extends_list = metadata.get('extends')
                                if isinstance(extends_list, list):
                                    extends_list.append(type_node.text.decode('utf8'))

            # Type parameters
            elif child.type == 'type_parameters':
                metadata['type_parameters'] = self._extract_type_parameters(child)
                if metadata['type_parameters']:
                    metadata['generics'] = True

            # Interface body
            elif child.type == 'interface_body':
                for member in child.children:
                    if member.type == 'method_declaration':
                        method_name = self._extract_node_name(member)
                        if method_name:
                            modifiers = self._extract_modifiers(member)
                            if 'default' in modifiers:
                                default_methods = metadata.get('default_methods')
                                if isinstance(default_methods, list):
                                    default_methods.append(method_name)
                            elif 'static' in modifiers:
                                static_methods = metadata.get('static_methods')
                                if isinstance(static_methods, list):
                                    static_methods.append(method_name)
                            else:
                                methods = metadata.get('methods')
                                if isinstance(methods, list):
                                    methods.append(method_name)
                    elif member.type == 'constant_declaration':
                        # Interface fields are implicitly public static final
                        field_names = self._extract_field_names(member)
                        constants = metadata.get('constants')
                        if isinstance(constants, list):
                            constants.extend(field_names)

        return metadata

    def _extract_type_parameters(self, node) -> List[str]:
        """Extract generic type parameters."""
        params = []
        for child in node.children:
            if child.type == 'type_parameter':
                param_text = child.text.decode('utf8')
                params.append(param_text)
        return params

    def _extract_parameters(self, node) -> List[Dict[str, Any]]:
        """Extract method parameters with types."""
        params = []

        for child in node.children:
            if child.type == 'formal_parameter' or child.type == 'spread_parameter':
                param: Dict[str, Any] = {
                    'name': None,
                    'type': None,
                    'is_varargs': child.type == 'spread_parameter',
                    'is_final': False,
                    'annotations': [],
                }

                for param_child in child.children:
                    if param_child.type == 'identifier':
                        param['name'] = param_child.text.decode('utf8')
                    elif param_child.type in [
                        'type_identifier',
                        'generic_type',
                        'array_type',
                        'integral_type',
                    ]:
                        param['type'] = param_child.text.decode('utf8')
                    elif param_child.type == 'modifiers':
                        for modifier in param_child.children:
                            if modifier.type == 'final':
                                param['is_final'] = True
                            elif modifier.type == 'annotation':
                                annotations_list = param.get('annotations')
                                if isinstance(annotations_list, list):
                                    annotations_list.append(modifier.text.decode('utf8'))

                if param['name']:
                    params.append(param)

        return params

    def _extract_field_names(self, field_node) -> List[str]:
        """Extract field names from field declaration."""
        names = []

        for child in field_node.children:
            if child.type == 'variable_declarator':
                for var_child in child.children:
                    if var_child.type == 'identifier':
                        names.append(var_child.text.decode('utf8'))

        return names

    def _detect_patterns(
        self, node, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[str]]:
        """Detect anti-patterns and framework patterns."""
        # Get base patterns from mixin
        patterns = super()._detect_patterns(node, metadata or {})

        # Framework patterns are now extracted directly in _create_chunk_from_node
        # Just add Java-specific patterns here

        # Check for Singleton pattern
        if metadata and metadata.get('chunk_type') == ChunkType.CLASS:
            # Simple heuristic: private constructor + static instance
            has_private_constructor = False
            has_static_instance = False

            for child in node.children:
                if child.type == 'class_body':
                    for member in child.children:
                        if member.type == 'constructor_declaration':
                            mods = self._extract_modifiers(member)
                            if 'private' in mods:
                                has_private_constructor = True
                        elif member.type == 'field_declaration':
                            mods = self._extract_modifiers(member)
                            if 'static' in mods:
                                # Check if field type matches class name
                                logger.info("[UNTESTED PATH] java static field for singleton")
                                has_static_instance = True  # Simplified

            if has_private_constructor and has_static_instance:
                logger.info("[UNTESTED PATH] java singleton pattern detected")
                framework_patterns = patterns.get('framework')
                if isinstance(framework_patterns, list):
                    framework_patterns.append('singleton_pattern')

        return patterns

    def _assess_quality(self, node, chunk_type: ChunkType) -> Dict[str, Any]:
        """Assess code quality indicators."""
        quality = {
            'has_javadoc': False,
            'test_coverage_hints': [],
            'follows_naming_convention': True,
        }

        # Check for Javadoc
        if node.start_point[0] > 0:
            # Look for /** */ comment before the node
            for child in node.parent.children if node.parent else []:
                if child.type == 'comment' and child.end_point[0] == node.start_point[0] - 1:
                    comment_text = child.text.decode('utf8')
                    if comment_text.startswith('/**'):
                        quality['has_javadoc'] = True
                        break

        # Test coverage hints
        annotations = self._extract_annotations(node)
        if any('@Test' in ann for ann in annotations):
            test_hints = quality.get('test_coverage_hints')
            if isinstance(test_hints, list):
                test_hints.append('is_test_method')

        # Naming conventions
        name = self._extract_node_name(node)
        if name:
            if chunk_type == ChunkType.CLASS:
                # Classes should start with uppercase
                if not name[0].isupper():
                    logger.info("[UNTESTED PATH] java class naming convention violation")
                    quality['follows_naming_convention'] = False
            elif chunk_type in [ChunkType.METHOD, ChunkType.FUNCTION]:
                # Methods should start with lowercase
                if name[0].isupper():
                    quality['follows_naming_convention'] = False
            elif chunk_type == ChunkType.CONSTANTS:
                # Constants should be UPPER_CASE
                if not name.isupper():
                    quality['follows_naming_convention'] = False

        return quality

    def _detect_security_issues(self, node, lines: List[str]) -> List[Dict[str, Any]]:
        """Detect common Java security issues."""
        issues = []

        node_text = node.text.decode('utf8')

        # SQL injection risk - look for string concatenation in SQL
        if 'PreparedStatement' not in node_text and any(
            sql in node_text
            for sql in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'executeQuery', 'executeUpdate']
        ):
            if '+' in node_text or 'concat' in node_text:
                logger.info("[UNTESTED PATH] java SQL injection risk")
                issues.append(
                    {
                        'type': 'sql_injection_risk',
                        'severity': 'high',
                        'line': node.start_point[0] + 1,
                    }
                )

        # Hardcoded credentials
        suspicious_vars = ['password', 'passwd', 'pwd', 'secret', 'api_key', 'apikey']
        for i, line in enumerate(lines[node.start_point[0] : node.end_point[0] + 1]):
            for var in suspicious_vars:
                if var in line.lower() and '=' in line and ('"' in line or "'" in line):
                    # Check if it's not reading from config
                    if not any(
                        cfg in line for cfg in ['getProperty', 'getenv', 'config', 'Config']
                    ):
                        logger.info("[UNTESTED PATH] java hardcoded credential")
                        issues.append(
                            {
                                'type': 'hardcoded_credential',
                                'severity': 'critical',
                                'line': node.start_point[0] + i + 1,
                            }
                        )
                        break

        # Weak random number generation
        if 'java.util.Random' in node_text and any(
            crypto in node_text for crypto in ['password', 'token', 'key', 'salt', 'crypto']
        ):
            logger.info("[UNTESTED PATH] java weak random")
            issues.append(
                {
                    'type': 'weak_random',
                    'severity': 'medium',
                    'line': node.start_point[0] + 1,
                    'suggestion': 'Use SecureRandom for cryptographic purposes',
                }
            )

        return issues

    def _analyze_dependencies(self, node) -> Dict[str, List[str]]:
        """Analyze internal and external dependencies."""
        deps: Dict[str, List[str]] = {'internal': [], 'external': []}

        # This is simplified - in real implementation would need access to imports
        # and would analyze field types, method parameters, etc.

        # For now, just mark if uses common frameworks based on annotations
        annotations = self._extract_annotations(node)
        frameworks = set()

        for ann in annotations:
            if 'springframework' in ann:
                logger.info("[UNTESTED PATH] java spring framework")
                frameworks.add('spring-framework')
            elif 'junit' in ann:
                logger.info("[UNTESTED PATH] java junit")
                frameworks.add('junit')
            elif 'lombok' in ann:
                logger.info("[UNTESTED PATH] java lombok")
                frameworks.add('lombok')
            elif 'jackson' in ann:
                logger.info("[UNTESTED PATH] java jackson")
                frameworks.add('jackson')

        deps['external'] = list(frameworks)

        return deps

    def _get_import_node_types(self) -> List[str]:
        """Get node types that represent imports for Java."""
        return ['import_declaration']

    def _is_comment_node(self, node) -> bool:
        """Check if node is a Java comment."""
        return node.type in ['comment', 'line_comment', 'block_comment']

    def _extract_dependencies_from_imports(self, import_nodes) -> List[str]:
        """Extract Java-specific import dependencies."""
        deps = set()

        for node in import_nodes:
            if node.type == 'import_declaration':
                # Extract the package/class being imported
                for child in node.children:
                    if child.type == 'scoped_identifier':
                        # Full import path
                        import_text = child.text.decode('utf8')
                        # Get base package (first part before .)
                        base_package = import_text.split('.')[0]
                        # Filter out java.* and javax.* as they're built-in
                        if base_package not in ['java', 'javax']:
                            deps.add(base_package)

        return sorted(list(deps))
