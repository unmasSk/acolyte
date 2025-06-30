"""
Rust chunker using tree-sitter-languages.
Extracts comprehensive metadata for Rust's unique features.
"""

from typing import Dict, List, Any, Optional, Set, Callable
from tree_sitter_languages import get_language  # type: ignore

from acolyte.models.chunk import ChunkType, Chunk
from acolyte.core.logging import logger
from acolyte.rag.chunking.base import LanguageChunker
from acolyte.rag.chunking.mixins import SecurityAnalysisMixin, PatternDetectionMixin


class RustChunker(LanguageChunker, SecurityAnalysisMixin, PatternDetectionMixin):
    """
    Rust-specific chunker using tree-sitter.

    Handles Rust's unique features:
    - Ownership and lifetimes
    - Traits and implementations
    - Macros and attributes
    - Unsafe blocks
    - Pattern matching
    - Error handling with Result/Option
    """

    def _get_language_name(self) -> str:
        """Return language identifier."""
        return 'rust'

    def _get_tree_sitter_language(self) -> Any:
        """Get Rust language for tree-sitter."""
        logger.info("[UNTESTED PATH] rust._get_tree_sitter_language called")
        return get_language('rust')

    def _get_import_node_types(self) -> List[str]:
        """Rust-specific import node types."""
        return [
            'use_declaration',
            'extern_crate_declaration',
        ]

    def _is_comment_node(self, node) -> bool:
        """Check if node is a Rust comment."""
        return node.type in ['line_comment', 'block_comment', 'doc_comment']

    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """
        Rust-specific node types to chunk.

        Tree-sitter Rust node types:
        - function_item: Regular functions
        - impl_item: Trait implementations
        - trait_item: Trait definitions
        - struct_item: Struct definitions
        - enum_item: Enum definitions
        - mod_item: Module definitions
        - const_item: Constants
        - static_item: Static variables
        - type_alias: Type aliases
        - use_declaration: Use statements
        """
        return {
            # Functions
            'function_item': ChunkType.FUNCTION,
            # Types and structures
            'struct_item': ChunkType.CLASS,  # Structs as classes
            'enum_item': ChunkType.CLASS,  # Enums as classes
            'trait_item': ChunkType.INTERFACE,
            'impl_item': ChunkType.METHOD,  # Will refine based on content
            # Modules and namespaces
            'mod_item': ChunkType.MODULE,
            # Constants and types
            'const_item': ChunkType.CONSTANTS,
            'static_item': ChunkType.CONSTANTS,
            'type_alias': ChunkType.TYPES,
            # Imports
            'use_declaration': ChunkType.IMPORTS,
            'extern_crate_declaration': ChunkType.IMPORTS,
            # Macros
            'macro_definition': ChunkType.FUNCTION,  # macro_rules! definitions
            # Trait method signatures
            'function_signature_item': ChunkType.FUNCTION,  # Methods in trait definitions
        }

    def _extract_node_name(self, node) -> Optional[str]:
        """
        Extract name from Rust AST nodes.

        Rust has specific patterns for where names appear in the AST.
        """
        # For functions, the pattern is: fn <identifier>
        if node.type == 'function_item':
            for i, child in enumerate(node.children):
                if child.type == 'identifier':
                    logger.info("[UNTESTED PATH] rust function identifier found")
                    return child.text.decode('utf8')
                # Sometimes the identifier is after 'fn' keyword
                if child.type == 'fn' and i + 1 < len(node.children):
                    next_child = node.children[i + 1]
                    if next_child.type == 'identifier':
                        logger.info("[UNTESTED PATH] rust function identifier after fn")
                        return next_child.text.decode('utf8')

        # For structs, enums, and traits: struct/enum/trait <type_identifier>
        elif node.type in ['struct_item', 'enum_item', 'trait_item']:
            for i, child in enumerate(node.children):
                if child.type == 'type_identifier':
                    logger.info("[UNTESTED PATH] rust type identifier found")
                    return child.text.decode('utf8')
                # Sometimes after the keyword
                if child.type in ['struct', 'enum', 'trait'] and i + 1 < len(node.children):
                    next_child = node.children[i + 1]
                    if next_child.type == 'type_identifier':
                        return next_child.text.decode('utf8')

        # For impl blocks, extract the type being implemented
        elif node.type == 'impl_item':
            # impl blocks can be "impl Type" or "impl Trait for Type"
            for child in node.children:
                if child.type == 'type_identifier':
                    return child.text.decode('utf8')

        # For constants and statics: const/static <identifier>
        elif node.type in ['const_item', 'static_item']:
            for i, child in enumerate(node.children):
                if child.type == 'identifier':
                    return child.text.decode('utf8')
                if child.type in ['const', 'static'] and i + 1 < len(node.children):
                    next_child = node.children[i + 1]
                    if next_child.type == 'identifier':
                        logger.info("[UNTESTED PATH] rust const/static identifier found")
                        return next_child.text.decode('utf8')

        # For type aliases: type <type_identifier>
        elif node.type == 'type_alias':
            for i, child in enumerate(node.children):
                if child.type == 'type_identifier':
                    return child.text.decode('utf8')
                if child.type == 'type' and i + 1 < len(node.children):
                    next_child = node.children[i + 1]
                    if next_child.type == 'type_identifier':
                        return next_child.text.decode('utf8')

        # For modules: mod <identifier>
        elif node.type == 'mod_item':
            for i, child in enumerate(node.children):
                if child.type == 'identifier':
                    return child.text.decode('utf8')
                if child.type == 'mod' and i + 1 < len(node.children):
                    next_child = node.children[i + 1]
                    if next_child.type == 'identifier':
                        return next_child.text.decode('utf8')

        # For macro rules: macro_rules! <identifier>
        elif node.type == 'macro_definition':
            # Extract macro name from text
            text = node.text.decode('utf8')
            if 'macro_rules!' in text:
                # Pattern: macro_rules! macro_name
                parts = text.split('macro_rules!')
                if len(parts) > 1:
                    name_part = parts[1].strip().split('{')[0].strip()
                    return name_part

        # For function signatures in traits
        elif node.type == 'function_signature_item':
            for i, child in enumerate(node.children):
                if child.type == 'identifier':
                    return child.text.decode('utf8')
                # Sometimes the identifier is after 'fn' keyword
                if child.type == 'fn' and i + 1 < len(node.children):
                    next_child = node.children[i + 1]
                    if next_child.type == 'identifier':
                        return next_child.text.decode('utf8')

        # Fallback to base implementation
        return super()._extract_node_name(node)

    def _create_chunk_from_node(
        self, node, lines: List[str], file_path: str, chunk_type: ChunkType, processed_ranges
    ):
        """Override to handle Rust-specific cases."""
        # Handle impl blocks specially
        if node.type == 'impl_item':
            chunk_type = self._determine_impl_type(node)

        # Handle test functions
        if node.type == 'function_item' and (
            self._is_test_function(node) or self._is_inside_test_module(node)
        ):
            chunk_type = ChunkType.TESTS

        # Handle async trait methods
        if node.type == 'function_signature_item':
            # These are trait method signatures, classify as FUNCTION initially
            chunk_type = ChunkType.FUNCTION

        # Check if function is inside an impl block (should be METHOD)
        if node.type == 'function_item' and self._is_inside_impl_block(node):
            chunk_type = ChunkType.METHOD

        # Standard processing
        chunk = super()._create_chunk_from_node(
            node, lines, file_path, chunk_type, processed_ranges
        )

        if not chunk:
            return None

        # Extract and apply Rust-specific metadata
        metadata = self._extract_rust_metadata(node, chunk_type)
        if metadata:
            chunk.metadata.language_specific = metadata

        return chunk

    def _is_inside_impl_block(self, node) -> bool:
        """Check if a function is inside an impl block."""
        parent = node.parent
        while parent:
            if parent.type == 'impl_item':
                return True
            parent = parent.parent
        return False

    def _determine_impl_type(self, impl_node) -> ChunkType:
        """Determine if impl is for a trait or inherent implementation."""
        # Check if it's a trait implementation by looking for 'for' keyword
        impl_text = impl_node.text.decode('utf8')
        # Pattern: impl TraitName for TypeName or unsafe impl TraitName for TypeName
        if ' for ' in impl_text.split('{')[0]:  # Only check before the opening brace
            return ChunkType.METHOD  # Trait impl

        # Otherwise it's inherent impl
        return ChunkType.METHOD

    def _is_test_function(self, func_node) -> bool:
        """Detect Rust unit–test functions."""
        # Attribute directly on the function
        for child in func_node.children:
            if child.type == 'attribute_item':
                attr = child.text.decode('utf8')
                if any(tag in attr for tag in ('#[test]', '#[tokio::test]')):
                    return True
        # Raw text fallback
        text = func_node.text.decode('utf8')
        if any(tag in text for tag in ('#[test]', '#[tokio::test]')):
            return True
        # Function inside cfg(test) module
        if self._is_inside_test_module(func_node):
            return True
        # Look at previous sibling attributes
        prev = func_node.prev_sibling if hasattr(func_node, 'prev_sibling') else None
        while prev and prev.type == 'attribute_item':
            if '#[test]' in prev.text.decode('utf8') or '#[tokio::test]' in prev.text.decode(
                'utf8'
            ):
                return True
            prev = prev.prev_sibling if hasattr(prev, 'prev_sibling') else None
        return False

    def _is_inside_test_module(self, node) -> bool:
        """Return True if the node is inside a `mod` annotated with `#[cfg(test)]`."""
        parent = node.parent
        while parent:
            if parent.type == 'mod_item':
                # Check attributes on the module
                for child in parent.children:
                    if child.type == 'attribute_item' and '#[cfg(test)]' in child.text.decode(
                        'utf8'
                    ):
                        return True
                # Also fallback to raw text
                if '#[cfg(test)]' in parent.text.decode('utf8'):
                    return True
            parent = parent.parent
        return False

    def _extract_rust_metadata(self, node, chunk_type: ChunkType) -> Dict[str, Any]:
        """Extract comprehensive Rust-specific metadata."""
        metadata = {
            'visibility': self._extract_visibility(node),
            'is_async': self._is_async(node),
            'is_unsafe': self._is_unsafe(node),
            'is_const': self._is_const(node),
            'attributes': self._extract_attributes(node),
            'lifetime_params': self._extract_lifetimes(node),
            'generic_params': self._extract_generics(node),
            'complexity': self._calculate_complexity(node),
            'patterns': self._detect_rust_patterns(node),
            'todos': self._extract_todos(node),
            'quality': self._analyze_quality(node),
            'security': self._detect_security_issues(node),
            'dependencies': self._extract_dependencies(node),
        }

        # Type-specific metadata
        if chunk_type == ChunkType.FUNCTION:
            metadata.update(self._extract_function_metadata(node))
        elif chunk_type in [ChunkType.CLASS, ChunkType.INTERFACE]:
            metadata.update(self._extract_type_metadata(node))
        elif chunk_type == ChunkType.METHOD:
            metadata.update(self._extract_impl_metadata(node))

        return metadata

    def _extract_visibility(self, node) -> str:
        """Extract visibility modifier (pub, pub(crate), etc)."""
        for child in node.children:
            if child.type == 'visibility_modifier':
                return child.text.decode('utf8')
        return 'private'

    def _is_async(self, node) -> bool:
        """Check if function is async."""
        if node.type == 'function_item':
            # Check the node text for 'async' keyword at the beginning
            node_text = node.text.decode('utf8').strip()
            # Look for async patterns - must be at the beginning
            if (
                node_text.startswith('async fn')
                or node_text.startswith('pub async fn')
                or node_text.startswith('async unsafe fn')
            ):
                return True
        # Also check for async in trait methods defined in #[async_trait]
        if node.type == 'function_signature_item':
            # Check if parent has async_trait attribute
            parent = node.parent
            while parent and parent.type != 'trait_item':
                parent = parent.parent
            if parent:
                for child in parent.children:
                    if child.type == 'attribute_item' and '#[async_trait]' in child.text.decode(
                        'utf8'
                    ):
                        return True
        return False

    def _is_unsafe(self, node) -> bool:
        """Detecta si un nodo es unsafe: funciones, métodos, traits, impls, bloques y métodos en traits/impls."""
        node_text = node.text.decode('utf8')

        # Funciones y métodos (incluyendo en traits/impls)
        if node.type in ('function_item', 'function_signature_item'):
            # Buscar 'unsafe fn' en cualquier parte del texto de la función
            if 'unsafe fn' in node_text:
                return True
            # También buscar 'unsafe' seguido de 'fn' con espacios opcionales
            if 'unsafe' in node_text and 'fn' in node_text:
                # Verificar que 'unsafe' viene antes de 'fn'
                unsafe_pos = node_text.find('unsafe')
                fn_pos = node_text.find('fn')
                if unsafe_pos < fn_pos:
                    return True
        # Traits
        if node.type == 'trait_item' and node_text.strip().startswith('unsafe trait'):
            return True
        # Impl
        if node.type == 'impl_item' and node_text.strip().startswith('unsafe impl'):
            return True
        # Bloques unsafe
        if node.type == 'unsafe_block':
            return True
        # Métodos dentro de impls o traits
        parent = node.parent
        while parent:
            parent_text = parent.text.decode('utf8')
            if parent.type in ('impl_item', 'trait_item') and 'unsafe' in parent_text.split('{')[0]:
                return True
            parent = parent.parent
        return False

    def _is_const(self, node) -> bool:
        """Check if function is const."""
        if node.type == 'function_item':
            # Check the node text for 'const fn' pattern at the beginning
            node_text = node.text.decode('utf8').strip()
            # Must check at the beginning of the function definition
            if node_text.startswith('const fn') or node_text.startswith('pub const fn'):
                return True
            # Also check after visibility modifiers
            if ' const fn ' in node_text[:50]:  # Check in first 50 chars
                return True
        return False

    def _extract_attributes(self, node) -> List[str]:
        """Extract outer/inner attributes (`#[...]`) and doc comments for a given AST node."""
        attributes: List[str] = []
        # 1. Direct attribute_item children captured by tree-sitter
        for child in node.children:
            if child.type == 'attribute_item':
                attributes.append(child.text.decode('utf8').strip())
            elif child.type in ('doc_comment', 'line_comment'):
                txt = child.text.decode('utf8').lstrip()
                if txt.startswith(('///', '//!')):
                    attributes.append(txt)
        # 2. Leading siblings (attribute items placed *before* this node)
        prev = node.prev_sibling if hasattr(node, 'prev_sibling') else None
        while prev and prev.type in ('attribute_item', 'line_comment', 'doc_comment'):
            if prev.type == 'attribute_item':
                attributes.append(prev.text.decode('utf8').strip())
            elif prev.type in ('doc_comment', 'line_comment'):
                txt = prev.text.decode('utf8').lstrip()
                if txt.startswith(('///', '//!')):
                    attributes.append(txt)
            prev = prev.prev_sibling if hasattr(prev, 'prev_sibling') else None
        # 3. Raw text inside node
        raw_text = node.text.decode('utf8', errors='ignore')
        for line in raw_text.split('\n'):
            line_str = line.strip()
            if line_str.startswith('#[') and line_str.endswith(']'):
                if line_str not in attributes:
                    attributes.append(line_str)
            elif line_str.startswith(('///', '//!')) and line_str not in attributes:
                attributes.append(line_str)
        return attributes

    def _extract_lifetimes(self, node) -> List[str]:
        """Extract lifetime parameters."""
        lifetimes: List[str] = []
        for child in node.children:
            if child.type == 'generic_parameters' or child.type == 'type_parameters':
                # Walk the parameters
                self._collect_lifetimes(child, lifetimes)
        return lifetimes

    def _collect_lifetimes(self, node, lifetimes: List[str]):
        """Recursively collect lifetime parameters."""
        if node.type == 'lifetime':
            lifetimes.append(node.text.decode('utf8'))
        for child in node.children:
            self._collect_lifetimes(child, lifetimes)

    def _extract_generics(self, node) -> List[str]:
        """Extract generic type parameters."""
        generics: List[str] = []
        for child in node.children:
            if child.type == 'generic_parameters' or child.type == 'type_parameters':
                # Walk the parameters
                self._collect_generics(child, generics)
        return generics

    def _collect_generics(self, node, generics: List[str]):
        """Recursively collect generic parameters."""
        # Check if this node is a type identifier that could be a generic parameter
        if node.type == 'type_identifier':
            # Check if it's in a generic context
            parent = node.parent
            while parent:
                if parent.type in [
                    'generic_parameters',
                    'type_parameters',
                    'generic_parameter',
                    'type_parameter',
                ]:
                    generics.append(node.text.decode('utf8'))
                    break
                parent = parent.parent

        # Also check for generic parameters in the text directly
        if node.type in ['generic_parameters', 'type_parameters']:
            node_text = node.text.decode('utf8')
            # Extract type names from generic parameters like <T, U, V>
            if '<' in node_text and '>' in node_text:
                generic_part = node_text[node_text.find('<') + 1 : node_text.rfind('>')]
                # Split by comma and extract type names
                for param in generic_part.split(','):
                    param = param.strip()
                    if param and not param.startswith(('&', '*', 'mut ', 'const ')):
                        # Extract the type name (before any bounds like : Send)
                        if ':' in param:
                            param = param.split(':')[0].strip()
                        if param and param not in generics:
                            generics.append(param)

        # Recurse into children
        for child in node.children:
            self._collect_generics(child, generics)

    # Override base complexity calculation to add Rust-specific metrics
    def _calculate_complexity(self, node) -> Dict[str, int]:
        """Calculate complexity metrics including Rust-specific ones."""
        # Get base complexity from mixin
        complexity = super()._calculate_complexity(node)

        # Add Rust-specific metrics
        complexity['unsafe_blocks'] = 0
        complexity['pattern_matches'] = 0

        self._analyze_rust_complexity(node, complexity)

        return complexity

    def _analyze_rust_complexity(self, node, complexity: Dict[str, int]):
        """Add Rust-specific complexity metrics."""
        # Count match arms
        if node.type == 'match_arm':
            complexity['cyclomatic'] += 1
            complexity['pattern_matches'] += 1

        # Count unsafe blocks
        if node.type == 'unsafe_block':
            complexity['unsafe_blocks'] += 1

        # Count error handling
        if node.type == 'try_expression':
            complexity['cyclomatic'] += 1

        # Track unwrap/expect invocations without touching cyclomatic complexity
        if node.type == 'call_expression':
            for child in node.children:
                if child.type == 'identifier' and child.text.decode('utf8') in ('unwrap', 'expect'):
                    complexity.setdefault('unwrap_expect_calls', 0)
                    complexity['unwrap_expect_calls'] += 1
                    break

        # Recurse
        for child in node.children:
            self._analyze_rust_complexity(child, complexity)

    # Override to add Rust-specific decision types
    def _get_decision_node_types(self):
        """Get Rust decision node types."""
        base_types = super()._get_decision_node_types()
        rust_types = {
            'if_expression',
            'match_expression',
            'while_expression',
            'for_expression',
            'loop_expression',
            'match_arm',
        }
        return base_types | rust_types

    def _detect_rust_patterns(self, node) -> Dict[str, List[str]]:
        """Detect Rust-specific patterns and anti-patterns."""
        patterns: Dict[str, List[str]] = {'anti': [], 'rust_specific': [], 'idioms': []}

        # Get base patterns from mixin
        base_patterns = self._detect_patterns(node)
        patterns['anti'].extend(base_patterns.get('anti', []))

        # Add Rust-specific pattern detection
        self._analyze_rust_patterns(node, patterns)

        return patterns

    def _analyze_rust_patterns(self, node, patterns: Dict[str, List[str]]) -> None:
        """Analyze for Rust-specific patterns."""
        # Get pattern lists once at the beginning
        patterns_rust = patterns.setdefault('rust_specific', [])
        patterns_anti = patterns.setdefault('anti', [])
        patterns_idioms = patterns.setdefault('idioms', [])

        # Check for unwrap/expect usage in the node text
        node_text = node.text.decode('utf8')
        if '.unwrap()' in node_text and 'unwrap_usage' not in patterns_rust:
            patterns_rust.append('unwrap_usage')
        if '.expect(' in node_text and 'expect_usage' not in patterns_rust:
            patterns_rust.append('expect_usage')

        # Rust-specific patterns
        if node.type == 'call_expression':
            # Identify called function name
            func_ident = None
            for child in node.children:
                if child.type == 'identifier':
                    func_ident = child.text.decode('utf8')
                    break
            if func_ident:
                if func_ident == 'clone':
                    if 'excessive_cloning' not in patterns_anti:
                        patterns_anti.append('excessive_cloning')
                elif func_ident == 'unwrap' and 'unwrap_usage' not in patterns_rust:
                    patterns_rust.append('unwrap_usage')
                elif func_ident == 'expect' and 'expect_usage' not in patterns_rust:
                    patterns_rust.append('expect_usage')
        # Heurística adicional para clone()
        if '.clone()' in node_text and 'excessive_cloning' not in patterns_anti:
            patterns_anti.append('excessive_cloning')

        # Check for macro invocations including panic!
        if node.type == 'macro_invocation':
            macro_text = node.text.decode('utf8')
            if macro_text.startswith('panic!'):
                patterns_rust.append('panic_usage')

        # Also recursively check for panic! in the text
        if 'panic!' in node_text and 'panic_usage' not in patterns_rust:
            patterns_rust.append('panic_usage')

        if node.type == 'unsafe_block':
            patterns_rust.append('unsafe_usage')

        # Good patterns
        if node.type == 'match_expression':
            if self._is_exhaustive_match(node):
                patterns_idioms.append('exhaustive_matching')

        if node.type == 'try_expression':
            patterns_idioms.append('error_propagation')

        # Recurse
        for child in node.children:
            self._analyze_rust_patterns(child, patterns)

    def _analyze_quality(self, node) -> Dict[str, Any]:
        """Analyze quality metrics including doc comments and test hints."""
        quality: Dict[str, Any] = {
            'has_docstring': False,
            'has_tests': False,
            'error_handling': 'none',
            'test_coverage_hints': [],
            'documentation_completeness': 0.0,
        }
        raw_text = node.text.decode('utf8', errors='ignore')
        # Doc comments detection (/// or //! anywhere in the node text)
        if '///' in raw_text or '//!' in raw_text:
            quality['has_docstring'] = True
        else:
            for child in node.children:
                if child.type in ('doc_comment', 'line_comment') and child.text.decode(
                    'utf8'
                ).lstrip().startswith(('///', '//!')):
                    quality['has_docstring'] = True
                    break
        # Error handling kind
        err_types = self._analyze_error_handling(node)
        if 'result' in err_types:
            quality['error_handling'] = 'result'
        elif 'option' in err_types:
            quality['error_handling'] = 'option'
        elif 'panic' in err_types:
            quality['error_handling'] = 'panic'
        # Test hints
        if any(
            tag in raw_text
            for tag in ('#[test]', '#[tokio::test]', '#[cfg(test)]', '#[should_panic]')
        ):
            quality['has_tests'] = True
            test_hints = quality['test_coverage_hints']
            assert isinstance(test_hints, list)
            test_hints.append('has_test_attributes')
        if any(assert_kw in raw_text for assert_kw in ('assert!', 'assert_eq!', 'assert_ne!')):
            test_hints = quality['test_coverage_hints']
            assert isinstance(test_hints, list)
            test_hints.append('uses_assertions')
        return quality

    # Override security patterns for Rust-specific vulnerabilities
    def _get_security_patterns(self) -> List[Callable[[Any, str], dict]]:
        """Get Rust-specific security check functions."""
        base_patterns = super()._get_security_patterns()
        rust_patterns = [
            self._check_unsafe_usage,
            self._check_panic_on_input,
            self._check_integer_overflow,
            self._check_transmute,
        ]
        return base_patterns + rust_patterns

    def _check_unsafe_usage(self, node, text: str) -> Dict[str, Any]:
        """Check for unsafe code blocks."""
        if node.type == 'unsafe_block':
            return {
                'type': 'unsafe_code',
                'severity': 'medium',
                'description': 'Unsafe code block requires careful review',
            }
        # Also check if it's an unsafe function/trait/impl
        elif self._is_unsafe(node):
            return {
                'type': 'unsafe_code',
                'severity': 'medium',
                'description': f'Unsafe {node.type.replace("_item", "")} requires careful review',
            }
        return {}

    def _check_panic_on_input(self, node, text: str) -> Dict[str, Any]:
        """Check for unwrap/expect on external input."""
        # Check if this node or its parent contains external input patterns
        is_input_context = self._is_external_input(node)
        if not is_input_context and node.parent:
            is_input_context = self._is_external_input(node.parent)

        # If we're in an input context, check for panic-inducing patterns
        if is_input_context:
            # Check for unwrap/expect in the node text
            if any(pattern in text for pattern in ['.unwrap()', '.expect(']):
                return {
                    'type': 'panic_on_input',
                    'severity': 'high',
                    'description': 'Potential panic on external input',
                }
        return {}

    def _check_integer_overflow(self, node, text: str) -> Dict[str, Any]:
        """Check for potential integer overflow."""
        if any(op in text for op in ['as u8', 'as u16', 'as u32', 'as i8', 'as i16', 'as i32']):
            return {
                'type': 'integer_cast',
                'severity': 'low',
                'description': 'Integer cast may overflow',
            }
        return {}

    def _check_transmute(self, node, text: str) -> Dict[str, Any]:
        """Check for unsafe transmute usage."""
        # transmute is a function call, not a node type
        # Accept fully-qualified or bare calls (e.g. `transmute::<T,U>(v)`)
        if 'transmute' in text:
            return {
                'type': 'unsafe_transmute',
                'severity': 'critical',
                'description': 'Transmute can violate memory safety',
            }
        return {}

    def _extract_dependencies(self, node) -> Dict[str, List[str]]:
        """Extract internal and external dependencies from node and its children."""
        deps: Dict[str, Set[str]] = {
            'internal': set(),
            'external': set(),
        }  # Use sets to avoid duplicates

        # Recursively find all use declarations within this node
        self._collect_use_declarations(node, deps)

        # Convert sets back to sorted lists
        return {
            'internal': sorted(list(deps['internal'])),
            'external': sorted(list(deps['external'])),
        }

    def _collect_use_declarations(self, node, deps: Dict[str, Set[str]]):
        """Recursively collect all use declarations."""
        if node.type == 'use_declaration':
            use_text = node.text.decode('utf8')
            # Remove 'use' and ';'
            import_path = use_text.replace('use', '').replace(';', '').strip()

            # Determine if internal or external
            if (
                import_path.startswith('crate::')
                or import_path.startswith('super::')
                or import_path.startswith('self::')
            ):
                deps['internal'].add(import_path)
            else:
                # External crate
                crate_name = import_path.split('::')[0]
                if crate_name not in ['std', 'core', 'alloc']:  # Skip standard library
                    deps['external'].add(crate_name)

        # Recurse into children
        for child in node.children:
            self._collect_use_declarations(child, deps)

    def _extract_dependencies_from_imports(self, import_nodes) -> List[str]:
        """Extract Rust-specific import dependencies."""
        deps = set()

        for node in import_nodes:
            text = node.text.decode('utf8')

            if node.type == 'use_declaration':
                # Extract crate name
                import_path = text.replace('use', '').replace(';', '').strip()
                crate_name = import_path.split('::')[0]
                if crate_name not in ['std', 'core', 'alloc', 'crate', 'super', 'self']:
                    deps.add(crate_name)

        return sorted(list(deps))

    def _extract_function_metadata(self, func_node) -> Dict[str, Any]:
        """Extract function-specific metadata."""
        metadata = {
            'parameters': self._extract_parameters(func_node),
            'return_type': self._extract_return_type(func_node),
            'is_generic': bool(self._extract_generics(func_node)),
            'is_recursive': self._is_recursive(func_node),
            'calls_unsafe': self._calls_unsafe(func_node),
        }

        return metadata

    def _extract_parameters(self, func_node) -> List[Dict[str, Any]]:
        """Extract function parameters with types."""
        params = []

        for child in func_node.children:
            if child.type == 'parameters':
                for param in child.children:
                    if param.type == 'parameter':
                        param_info = self._parse_parameter(param)
                        if param_info:
                            params.append(param_info)

        return params

    def _parse_parameter(self, param_node) -> Optional[Dict[str, Any]]:
        """Parse a single parameter."""
        param_info = {
            'name': '',
            'type': '',
            'mutable': False,
            'reference': False,
            'lifetime': None,
        }

        # Extract parameter details
        for child in param_node.children:
            if child.type == 'identifier':
                param_info['name'] = child.text.decode('utf8')
            elif child.type == 'type_identifier':
                param_info['type'] = child.text.decode('utf8')
            elif child.type == 'mut':
                param_info['mutable'] = True
            elif child.type == 'reference_type':
                param_info['reference'] = True
                # Check for lifetime
                for ref_child in child.children:
                    if ref_child.type == 'lifetime':
                        param_info['lifetime'] = ref_child.text.decode('utf8')

        return param_info if param_info['name'] else None

    def _extract_return_type(self, func_node) -> Optional[str]:
        """Extract function return type."""
        for child in func_node.children:
            if child.type == 'return_type':
                # Skip the '->' and get the actual type
                type_text = child.text.decode('utf8').replace('->', '').strip()
                return type_text
        return None

    def _extract_type_metadata(self, type_node) -> Dict[str, Any]:
        """Extract struct/enum/trait metadata."""
        metadata = {
            'fields': [],
            'methods': [],
            'derives': [],
            'trait_bounds': [],
            'is_generic': bool(self._extract_generics(type_node)),
        }

        # Extract derives
        for attr in self._extract_attributes(type_node):
            if '#[derive' in attr:
                # Parse derive attributes
                derives_text = attr[attr.find('(') + 1 : attr.find(')')].strip()
                metadata['derives'] = [d.strip() for d in derives_text.split(',')]

        # For structs, extract fields
        if type_node.type == 'struct_item':
            metadata['fields'] = self._extract_struct_fields(type_node)

        # For enums, extract variants
        elif type_node.type == 'enum_item':
            metadata['variants'] = self._extract_enum_variants(type_node)

        # For traits, extract required methods
        elif type_node.type == 'trait_item':
            metadata['required_methods'] = self._extract_trait_methods(type_node)

        return metadata

    def _extract_impl_metadata(self, impl_node) -> Dict[str, Any]:
        """Extract impl block metadata."""
        metadata: Dict[str, Any] = {
            'target_type': '',
            'trait_name': None,
            'methods': [],
            'associated_types': [],
            'is_generic': bool(self._extract_generics(impl_node)),
        }

        # Extract what we're implementing for
        impl_text = impl_node.text.decode('utf8')

        # Check if it's a trait implementation
        if ' for ' in impl_text:
            # Pattern: impl TraitName for TypeName or unsafe impl TraitName for TypeName
            parts = impl_text.split(' for ')
            if len(parts) >= 2:
                trait_part = parts[0].replace('impl', '').replace('unsafe', '').strip()
                # Extract trait name (may have generics)
                trait_name = trait_part.split('<')[0].strip()
                if trait_name:  # Ensure we have a valid trait name
                    metadata['trait_name'] = trait_name

                # Extract target type
                type_part = parts[1].split('{')[0].strip()
                type_name = type_part.split('<')[0].strip()
                metadata['target_type'] = type_name
        else:
            # Inherent implementation
            for child in impl_node.children:
                if child.type == 'type_identifier':
                    metadata['target_type'] = child.text.decode('utf8')
                    break

        return metadata

    # Helper methods specific to Rust
    def _is_in_hot_path(self, node) -> bool:
        """Check if code is in a performance-critical path."""
        # Look for loop or iterator context
        parent = node.parent
        while parent:
            if parent.type in ['for_expression', 'while_expression', 'loop_expression']:
                return True
            parent = parent.parent
        return False

    def _is_exhaustive_match(self, match_node) -> bool:
        """Check if match is exhaustive."""
        # Look for wildcard pattern
        for child in match_node.children:
            if child.type == 'match_arm':
                for arm_child in child.children:
                    if arm_child.type == 'wildcard_pattern' or arm_child.text == b'_':
                        return True
        return False

    def _analyze_error_handling(self, node):
        """Analyze error handling patterns."""
        patterns = set()

        def analyze(n):
            if n.type == 'type_identifier' and n.text.decode('utf8') in ['Result', 'Option']:
                patterns.add(n.text.decode('utf8').lower())
            elif n.type == 'panic_macro':
                patterns.add('panic')

            for child in n.children:
                analyze(child)

        analyze(node)
        return patterns

    def _is_external_input(self, node) -> bool:
        """Check if value comes from external input."""
        # Get broader context - check node, parent, and grandparent
        contexts = [node]
        if node.parent:
            contexts.append(node.parent)
            if node.parent.parent:
                contexts.append(node.parent.parent)

        # Check all contexts for input indicators
        for context in contexts:
            context_text = context.text.decode('utf8', errors='ignore')

            # Common input sources
            input_indicators = [
                'args()',
                'stdin()',
                'read_line',
                'from_str',
                'parse()',
                'env::',
                'env.args',
                'io::stdin',
                'File::open',
                'read_to_string',
            ]

            if any(indicator in context_text for indicator in input_indicators):
                return True

        return False

    def _is_recursive(self, func_node) -> bool:
        """Check if function is recursive."""
        func_name = self._extract_node_name(func_node)
        if not func_name:
            return False

        # Look for self-calls
        def check_recursion(node):
            if node.type == 'call_expression':
                for child in node.children:
                    if child.type == 'identifier' and child.text.decode('utf8') == func_name:
                        return True

            for child in node.children:
                if check_recursion(child):
                    return True
            return False

        return check_recursion(func_node)

    def _calls_unsafe(self, func_node) -> bool:
        """Check if function calls unsafe code."""

        def check_unsafe(node):
            if node.type == 'unsafe_block':
                return True

            for child in node.children:
                if check_unsafe(child):
                    return True
            return False

        return check_unsafe(func_node)

    def _extract_struct_fields(self, struct_node) -> List[Dict[str, Any]]:
        """Extract struct fields."""
        fields = []

        for child in struct_node.children:
            if child.type == 'field_declaration_list':
                for field in child.children:
                    if field.type == 'field_declaration':
                        field_info = {'name': '', 'type': '', 'visibility': 'private'}

                        for field_child in field.children:
                            if field_child.type == 'field_identifier':
                                field_info['name'] = field_child.text.decode('utf8')
                            elif field_child.type == 'type_identifier':
                                field_info['type'] = field_child.text.decode('utf8')
                            elif field_child.type == 'visibility_modifier':
                                field_info['visibility'] = field_child.text.decode('utf8')

                        if field_info['name']:
                            fields.append(field_info)

        return fields

    def _extract_enum_variants(self, enum_node) -> List[Dict[str, Any]]:
        """Extract enum variants."""
        variants = []

        for child in enum_node.children:
            if child.type == 'enum_variant_list':
                for variant in child.children:
                    if variant.type == 'enum_variant':
                        variant_info = {'name': '', 'fields': []}

                        for variant_child in variant.children:
                            if variant_child.type == 'identifier':
                                variant_info['name'] = variant_child.text.decode('utf8')

                        if variant_info['name']:
                            variants.append(variant_info)

        return variants

    def _extract_trait_methods(self, trait_node) -> List[str]:
        """Extract required trait methods."""
        methods = []

        for child in trait_node.children:
            if child.type == 'declaration_list':
                for item in child.children:
                    if item.type == 'function_signature_item':
                        method_name = self._extract_node_name(item)
                        if method_name:
                            methods.append(method_name)

        return methods

    async def chunk(self, content: str, file_path: str) -> List[Chunk]:
        """Override to add smart overlap after chunking."""
        # Call parent chunk method
        chunks = await super().chunk(content, file_path)
        # Add smart overlap and return the enhanced chunks
        return self._create_chunks_with_smart_overlap(chunks)

    def _create_chunks_with_smart_overlap(
        self, chunks: List[Chunk], preserve_imports: bool = True
    ) -> List[Chunk]:
        """Return a new list of chunks with smart overlap added for better context. Does not modify the input chunks in-place."""
        if len(chunks) <= 1:
            return chunks

        enhanced_chunks = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]

            # Extract imports from previous chunk
            prev_lines = prev_chunk.content.split('\n')
            imports = []
            for line in prev_lines:
                if line.strip().startswith('use ') and line.strip().endswith(';'):
                    imports.append(line)

            overlap_parts = []
            if imports and preserve_imports:
                import_context = "\n".join(["# Key imports:"] + imports[:5])  # Limit to 5 imports
                overlap_parts.append(import_context)

            # You can add more overlap logic here if desired

            if overlap_parts:
                enhanced_content = "\n\n".join(overlap_parts + [curr_chunk.content])
                enhanced = Chunk(
                    id=curr_chunk.id, content=enhanced_content, metadata=curr_chunk.metadata
                )
                enhanced_chunks.append(enhanced)
            else:
                enhanced_chunks.append(curr_chunk)

        return enhanced_chunks

    def _extract_imports(self, root_node, lines, file_path, processed_ranges):
        """Extract all use/extern_crate as an import chunk with well-formatted dependencies."""
        import_chunks = []
        import_types = self._get_import_node_types()
        import_nodes = []

        # Buscar todos los nodos de importación
        def find_imports(node):
            if node.type in import_types:
                import_nodes.append(node)
            for child in node.children:
                find_imports(child)

        find_imports(root_node)

        if not import_nodes:
            return []

        # Marcar como procesados
        for node in import_nodes:
            start_line = node.start_point[0]
            end_line = node.end_point[0]
            if node.type not in processed_ranges:
                processed_ranges[node.type] = set()
            processed_ranges[node.type].add((start_line, end_line))

        # Agrupar imports consecutivos
        first_import = import_nodes[0]
        last_import = import_nodes[-1]
        start_line = first_import.start_point[0]
        end_line = last_import.end_point[0]
        while start_line > 0 and (
            lines[start_line - 1].strip().startswith('//')
            or lines[start_line - 1].strip().startswith('#')
            or not lines[start_line - 1].strip()
        ):
            start_line -= 1
        content = '\n'.join(lines[start_line : end_line + 1])

        chunk = self._create_chunk(
            content=content,
            chunk_type=ChunkType.IMPORTS,
            file_path=file_path,
            start_line=start_line + 1,
            end_line=end_line + 1,
            name='imports',
        )

        # Asignar dependencias bien formateadas
        chunk.metadata.language_specific = {'dependencies': self._extract_dependencies(root_node)}

        # Marcar líneas como procesadas
        if 'imports' not in processed_ranges:
            processed_ranges['imports'] = set()
        for line in range(start_line, end_line + 1):
            processed_ranges['imports'].add((line, line))

        import_chunks.append(chunk)
        return import_chunks
