"""
Ruby chunker using tree-sitter-languages.
Extracts comprehensive metadata for Ruby code search and context.
"""

import re
from typing import Dict, List, Any, Optional, Set
from tree_sitter_languages import get_language  # type: ignore
import logging

from acolyte.models.chunk import ChunkType
from acolyte.rag.chunking.base import LanguageChunker
from acolyte.rag.chunking.mixins import (
    PatternDetectionMixin,
    SecurityAnalysisMixin,
    DependencyAnalysisMixin,
)

logger = logging.getLogger(__name__)


class RubyChunker(
    LanguageChunker, PatternDetectionMixin, SecurityAnalysisMixin, DependencyAnalysisMixin
):
    """
    Ruby-specific chunker using tree-sitter.

    Handles Ruby's unique features:
    - Modules and nested classes
    - Blocks, procs, and lambdas
    - attr_accessor, attr_reader, attr_writer
    - Mix-ins and inheritance
    - Ruby's metaprogramming patterns
    """

    def _get_language_name(self) -> str:
        """Return language identifier."""
        return 'ruby'

    def _get_tree_sitter_language(self) -> Any:
        """Get Ruby language for tree-sitter."""
        return get_language('ruby')

    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """
        Ruby-specific node types to chunk.

        Tree-sitter Ruby node types:
        - method: Regular methods
        - singleton_method: Class methods (def self.method)
        - class: Class definitions
        - module: Module definitions
        - assignment: For constants
        - call: For imports and test blocks
        """
        return {
            # Methods
            'method': ChunkType.FUNCTION,
            'singleton_method': ChunkType.FUNCTION,
            # Classes and modules
            'class': ChunkType.CLASS,
            'module': ChunkType.MODULE,
            # Special Ruby constructs
            'singleton_class': ChunkType.CLASS,  # class << self
            # Constants (we'll filter for uppercase)
            'assignment': ChunkType.CONSTANTS,
            # Imports and special calls
            'call': ChunkType.IMPORTS,  # Will filter for require/it/describe/context
        }

    def _create_chunk_from_node(
        self, node, lines: List[str], file_path: str, chunk_type: ChunkType, processed_ranges
    ):
        """Create chunk from tree-sitter node with Ruby-specific metadata."""
        # Handle method nodes
        if node.type in ['method', 'singleton_method']:
            return self._create_method_chunk(node, lines, file_path, chunk_type, processed_ranges)

        # Handle call nodes (imports, tests, etc.)
        if node.type == 'call':
            return self._create_call_chunk(node, lines, file_path, chunk_type, processed_ranges)

        # Handle assignment nodes (constants)
        if node.type == 'assignment':
            if not self._is_constant_assignment(node):
                return None

        # Create chunk using base implementation
        chunk = super()._create_chunk_from_node(
            node, lines, file_path, chunk_type, processed_ranges
        )

        # Add language-specific metadata based on chunk type
        if chunk:
            self._add_metadata_to_chunk(chunk, node)

        return chunk

    def _create_method_chunk(
        self, node, lines: List[str], file_path: str, chunk_type: ChunkType, processed_ranges
    ):
        """Create chunk for method nodes with proper type detection."""
        # Determine if it's constructor, test or regular function
        chunk_type = self._determine_method_type(node)
        name = self._extract_node_name(node)
        visibility = self._get_method_visibility(node)

        chunk = super()._create_chunk_from_node(
            node, lines, file_path, chunk_type, processed_ranges
        )

        if chunk:
            chunk.metadata.name = name
            chunk.metadata.language_specific = self._extract_method_metadata(node)
            if chunk.metadata.language_specific is not None:
                chunk.metadata.language_specific['visibility'] = visibility

        return chunk

    def _create_call_chunk(
        self, node, lines: List[str], file_path: str, chunk_type: ChunkType, processed_ranges
    ):
        """Create chunk for call nodes (imports, tests, etc.)."""
        method_name = self._get_call_method(node)
        patterns = self._detect_language_patterns(node, {'framework': []})
        dependencies = self._extract_dependencies_from_node(node)

        # Determine chunk type based on call method
        is_rspec = 'rspec_test' in patterns.get('framework', [])
        is_test_method = method_name in ['it', 'describe', 'context', 'test']
        is_import = method_name in ['require', 'require_relative', 'load', 'autoload']

        if is_rspec or is_test_method:
            chunk_type = ChunkType.TESTS
        elif is_import or patterns['framework']:
            chunk_type = ChunkType.IMPORTS
        else:
            return None

        chunk = super()._create_chunk_from_node(
            node, lines, file_path, chunk_type, processed_ranges
        )

        # Handle RSpec test blocks that base chunker might ignore
        if not chunk and is_rspec:
            chunk = self._create_minimal_test_chunk(
                node, lines, file_path, method_name or "", patterns, dependencies
            )

        if chunk:
            self._add_call_metadata_to_chunk(
                chunk, chunk_type, method_name or "", patterns, dependencies, node
            )

        return chunk

    def _create_minimal_test_chunk(
        self,
        node,
        lines: List[str],
        file_path: str,
        method_name: str,
        patterns: Dict,
        dependencies: List,
    ):
        """Create minimal test chunk when base chunker ignores RSpec blocks."""
        from acolyte.models.chunk import Chunk, ChunkMetadata

        desc = self._extract_test_description(node)
        name = desc[:50] + '...' if desc and len(desc) > 50 else (desc or method_name)

        return Chunk(
            file_path=file_path,
            start_line=getattr(node, 'start_point', (0,))[0] + 1,
            end_line=getattr(node, 'end_point', (0,))[0] + 1,
            code='\n'.join(
                lines[
                    getattr(node, 'start_point', (0,))[0] : getattr(node, 'end_point', (0,))[0] + 1
                ]
            ),
            metadata=ChunkMetadata(
                chunk_type=ChunkType.TESTS,
                name=name,
                language_specific={
                    'patterns': patterns,
                    'dependencies': {'internal': [], 'external': dependencies or []},
                },
            ),
        )

    def _extract_test_description(self, node) -> Optional[str]:
        """Extract test description from RSpec/test blocks."""
        for child in node.children:
            if child.type == 'argument_list':
                for arg in child.children:
                    if arg.type == 'string':
                        return arg.text.decode('utf8').strip("'\"")
        return None

    def _add_call_metadata_to_chunk(
        self,
        chunk,
        chunk_type: ChunkType,
        method_name: str,
        patterns: Dict,
        dependencies: List,
        node,
    ):
        """Add metadata to call chunks (imports, tests)."""
        if (
            not hasattr(chunk.metadata, 'language_specific')
            or chunk.metadata.language_specific is None
        ):
            chunk.metadata.language_specific = {}

        # Always propagate detected patterns
        if 'patterns' not in chunk.metadata.language_specific:
            chunk.metadata.language_specific['patterns'] = patterns
        else:
            # Merge detected frameworks
            for k, v in patterns.items():
                if k in chunk.metadata.language_specific['patterns']:
                    chunk.metadata.language_specific['patterns'][k] = list(
                        set(chunk.metadata.language_specific['patterns'][k] + v)
                    )
                else:
                    chunk.metadata.language_specific['patterns'][k] = v

        # Always include dependencies even if empty
        if 'dependencies' not in chunk.metadata.language_specific:
            chunk.metadata.language_specific['dependencies'] = {'internal': [], 'external': []}
        if dependencies:
            chunk.metadata.language_specific['dependencies']['external'].extend(dependencies)

        # For TESTS, use description as name if available
        if chunk_type == ChunkType.TESTS:
            desc = self._extract_test_description(node)
            if desc:
                chunk.metadata.name = desc[:50] + '...' if len(desc) > 50 else desc
            else:
                chunk.metadata.name = method_name

    def _add_metadata_to_chunk(self, chunk, node):
        """Add language-specific metadata based on chunk type."""
        if chunk.metadata.chunk_type == ChunkType.FUNCTION:
            chunk.metadata.language_specific = self._extract_method_metadata(node)
        elif chunk.metadata.chunk_type == ChunkType.CLASS:
            chunk.metadata.language_specific = self._extract_class_metadata(node)
        elif chunk.metadata.chunk_type == ChunkType.MODULE:
            chunk.metadata.language_specific = self._extract_module_metadata(node)
        elif chunk.metadata.chunk_type == ChunkType.IMPORTS:
            deps = self._extract_dependencies_from_node(node)
            patterns = self._detect_language_patterns(node, {'framework': []})
            chunk.metadata.language_specific = {
                'dependencies': {'internal': [], 'external': deps if deps else []},
                'patterns': patterns,
            }
        elif chunk.metadata.chunk_type == ChunkType.TESTS:
            chunk.metadata.language_specific = self._extract_test_metadata(node)
        elif chunk.metadata.chunk_type == ChunkType.CONSTANTS:
            chunk.metadata.language_specific = {
                'security': self._detect_security_issues(node),
                'todos': self._extract_todos(node),
            }

    def _get_call_method(self, node) -> Optional[str]:
        """Extract method name from a call node."""
        for child in node.children:
            if child.type == 'identifier':
                return child.text.decode('utf8')
        return None

    def _is_constant_assignment(self, node) -> bool:
        """Check if assignment is a constant (uppercase)."""
        for child in node.children:
            if child.type == 'constant':
                return True
        return False

    def _determine_method_type(self, method_node) -> ChunkType:
        """Determine if a method is a test, constructor, etc."""
        name = self._extract_node_name(method_node)

        # Check if it's initialize (constructor)
        if name == 'initialize':
            logger.info("[UNTESTED PATH] ruby constructor detected")
            return ChunkType.CONSTRUCTOR

        # Check for test methods (RSpec/Minitest patterns)
        if name and (name.startswith('test_') or name.startswith('it_')):
            logger.info("[UNTESTED PATH] ruby test method detected")
            return ChunkType.TESTS

        # For RSpec/Minitest: only if the immediate parent is a call describe/context/it/test
        parent = getattr(method_node, 'parent', None)
        if parent and parent.type == 'call':
            call_method = self._get_call_method(parent)
            if call_method in ['describe', 'context', 'it', 'test']:
                return ChunkType.TESTS
        # Otherwise, it's a regular FUNCTION
        return ChunkType.FUNCTION

    def _extract_method_metadata(self, method_node) -> Dict[str, Any]:
        """Extract Ruby-specific method metadata."""
        metadata = {
            'visibility': 'public',  # Default in Ruby
            'is_singleton': method_node.type == 'singleton_method',
            'parameters': [],
            'has_yield': False,
            'has_block_param': False,
            'aliases': [],
            'complexity': self._calculate_complexity(method_node),
        }

        # Extract parameters
        for child in getattr(method_node, 'children', []):
            if child.type == 'method_parameters':
                params = self._extract_parameters(child)
                metadata['parameters'] = params
                # Check if any parameter is a block
                for param in params:
                    if param.get('type') == 'block' or param.get('name', '').startswith('&'):
                        metadata['has_block_param'] = True

        # Check for yield usage (recursively)
        metadata['has_yield'] = self._has_yield(method_node)

        # Fallback: reconstruye el texto completo del método y busca 'yield' y '&block'
        # Usa start_byte y end_byte si existen
        try:
            start_byte = getattr(method_node, 'start_byte', None)
            end_byte = getattr(method_node, 'end_byte', None)
            if (
                start_byte is not None
                and end_byte is not None
                and hasattr(self, '_current_file_text')
            ):
                # Reconstruye el texto del método
                file_text = self._current_file_text
                method_text = file_text[start_byte:end_byte]
                if not metadata['has_yield'] and 'yield' in method_text:
                    metadata['has_yield'] = True
                if not metadata['has_block_param'] and '&block' in method_text:
                    metadata['has_block_param'] = True
            else:
                # Fallback anterior
                method_text = (
                    getattr(method_node, 'text', b'').decode('utf8')
                    if hasattr(method_node, 'text')
                    else ''
                )
                if not metadata['has_yield'] and 'yield' in method_text:
                    metadata['has_yield'] = True
                if not metadata['has_block_param'] and '&block' in method_text:
                    metadata['has_block_param'] = True
        except Exception:
            pass

        # Check for visibility modifiers before this method
        metadata['visibility'] = self._get_method_visibility(method_node)

        # Extract TODOs and FIXMEs
        metadata['todos'] = self._extract_todos(method_node)

        # Check for common patterns
        patterns_result = self._detect_patterns(method_node)
        metadata['patterns'] = patterns_result

        # Detect security issues
        metadata['security'] = self._detect_security_issues(method_node)

        # Extract dependencies
        deps = self._analyze_dependencies(method_node)
        if deps['internal'] or deps['external']:
            metadata['dependencies'] = deps

        return metadata

    def _extract_parameters(self, params_node) -> List[Dict[str, Any]]:
        """Extract parameter details from method_parameters node."""
        params = []

        for child in params_node.children:
            param = {'name': '', 'type': 'positional', 'default': None}

            if child.type == 'identifier':
                param['name'] = child.text.decode('utf8')
            elif child.type == 'optional_parameter':
                # param = default_value
                for subchild in child.children:
                    if subchild.type == 'identifier':
                        param['name'] = subchild.text.decode('utf8')
                        param['type'] = 'optional'
                    # Could extract default value here
            elif child.type == 'keyword_parameter':
                # param: value
                param['type'] = 'keyword'
                for subchild in child.children:
                    if subchild.type == 'identifier':
                        param['name'] = subchild.text.decode('utf8')
            elif child.type == 'block_parameter':
                # &block
                param['type'] = 'block'
                param['name'] = child.text.decode('utf8')
            elif child.type == 'splat_parameter':
                # *args
                param['type'] = 'splat'
                param['name'] = child.text.decode('utf8')
            elif child.type == 'hash_splat_parameter':
                # **kwargs
                param['type'] = 'hash_splat'
                param['name'] = child.text.decode('utf8')

            if param['name']:
                params.append(param)

        return params

    def _has_yield(self, node) -> bool:
        """Check recursively if any descendant node is a 'yield'."""
        if getattr(node, 'type', None) == 'yield':
            return True
        for child in getattr(node, 'children', []):
            if self._has_yield(child):
                return True
        return False

    def _get_method_visibility(self, method_node) -> str:
        """Determine method visibility based on Ruby visibility modifiers."""
        name = self._extract_node_name(method_node)
        parent = getattr(method_node, 'parent', None)

        # 1. Search in subsequent siblings and their descendants (in the same body_statement)
        if parent and hasattr(parent, 'children'):
            idx = parent.children.index(method_node)

            def search_in_subtree(node):
                if getattr(node, 'type', None) == 'call':
                    call_method = self._get_call_method(node)
                    if call_method in ['private', 'protected', 'public']:
                        for child in node.children:
                            if child.type == 'argument_list':
                                for arg in child.children:
                                    if arg.type in ['symbol', 'simple_symbol']:
                                        arg_text = arg.text.decode('utf8').lstrip(':')
                                        if arg_text == name:
                                            return call_method
                for child in getattr(node, 'children', []):
                    result = search_in_subtree(child)
                    if result:
                        return result
                return None

            for node in parent.children[idx + 1 :]:
                result = search_in_subtree(node)
                if result:
                    return result

            # 2. Search for modifiers before the method (only the nearest one)
            for prev in reversed(parent.children[:idx]):
                if getattr(prev, 'type', None) == 'identifier':
                    val = prev.text.decode('utf8')
                    if val in ['private', 'protected', 'public']:
                        logger.info("[UNTESTED PATH] ruby visibility modifier found")
                        return val
                if getattr(prev, 'type', None) == 'call':
                    call_method = self._get_call_method(prev)
                    if call_method in ['private', 'protected', 'public']:
                        for child in prev.children:
                            if child.type == 'argument_list':
                                for arg in child.children:
                                    if arg.type in ['symbol', 'simple_symbol']:
                                        arg_text = arg.text.decode('utf8').lstrip(':')
                                        if arg_text == name:
                                            return call_method

        # 3. If not found, return 'public' (Ruby default)
        return 'public'

    def _extract_class_metadata(self, class_node) -> Dict[str, Any]:
        """Extract Ruby-specific class metadata."""
        metadata: Dict[str, Any] = {
            'superclass': None,
            'included_modules': [],
            'extended_modules': [],
            'prepended_modules': [],
            'methods': [],
            'class_methods': [],
            'attr_accessors': [],
            'attr_readers': [],
            'attr_writers': [],
            'constants': [],
            'has_initialize': False,
            'complexity': self._calculate_complexity(class_node),
            'dependencies': {'internal': [], 'external': []},
        }

        # Extract superclass
        for child in class_node.children:
            if child.type == 'superclass':
                metadata['superclass'] = child.text.decode('utf8').strip('< ')

        # Walk the class body
        for child in class_node.children:
            if child.type == 'body_statement' or child.type == 'class_body':
                self._analyze_class_body(child, metadata)

        # Extract TODOs
        metadata['todos'] = self._extract_todos(class_node)

        # Detect patterns
        patterns_result = self._detect_patterns(class_node)
        metadata['patterns'] = patterns_result

        # Detect security issues
        metadata['security'] = self._detect_security_issues(class_node)

        # Extract dependencies
        deps = self._analyze_dependencies(class_node)
        metadata['dependencies'] = deps

        return metadata

    def _extract_module_metadata(self, module_node) -> Dict[str, Any]:
        """Extract Ruby-specific module metadata."""
        metadata: Dict[str, Any] = {
            'methods': [],
            'module_methods': [],
            'included_modules': [],
            'extended_modules': [],
            'prepended_modules': [],
            'attr_accessors': [],
            'attr_readers': [],
            'attr_writers': [],
            'constants': [],
            'complexity': self._calculate_complexity(module_node),
            'dependencies': {'internal': [], 'external': []},
        }

        # Walk the module body
        for child in module_node.children:
            if child.type == 'body_statement' or child.type == 'module_body':
                self._analyze_module_body(child, metadata)

        # Extract TODOs
        metadata['todos'] = self._extract_todos(module_node)

        # Detect patterns
        patterns_result = self._detect_patterns(module_node)
        metadata['patterns'] = patterns_result

        # Detect security issues
        metadata['security'] = self._detect_security_issues(module_node)

        # Extract dependencies
        deps = self._analyze_dependencies(module_node)
        if deps['internal'] or deps['external']:
            metadata['dependencies'] = deps

        return metadata

    def _analyze_class_body(self, body_node, metadata: Dict[str, Any]) -> None:
        """Analyze class body for methods, constants, and module inclusions."""
        for child in body_node.children:
            if child.type == 'method':
                name = self._extract_node_name(child)
                if name == 'initialize':
                    metadata['has_initialize'] = True
                if name and 'methods' in metadata and isinstance(metadata['methods'], list):
                    metadata['methods'].append(name)
            elif child.type == 'singleton_method':
                name = self._extract_node_name(child)
                if (
                    name
                    and 'class_methods' in metadata
                    and isinstance(metadata['class_methods'], list)
                ):
                    metadata['class_methods'].append(name)
            elif child.type == 'call':
                self._analyze_class_call(child, metadata)
            elif child.type == 'assignment' and self._is_constant_assignment(child):
                for subchild in child.children:
                    if (
                        subchild.type == 'constant'
                        and 'constants' in metadata
                        and isinstance(metadata['constants'], list)
                    ):
                        metadata['constants'].append(subchild.text.decode('utf8'))

    def _analyze_module_body(self, body_node, metadata: Dict[str, Any]) -> None:
        """Analyze module body for methods, constants, includes, extends, prepends, autoload, attr_accessor, attr_reader, attr_writer, etc."""
        for child in body_node.children:
            if child.type == 'method':
                name = self._extract_node_name(child)
                if name and 'methods' in metadata and isinstance(metadata['methods'], list):
                    metadata['methods'].append(name)
            elif child.type == 'singleton_method':
                name = self._extract_node_name(child)
                if (
                    name
                    and 'module_methods' in metadata
                    and isinstance(metadata['module_methods'], list)
                ):
                    metadata['module_methods'].append(name)
            elif child.type == 'assignment' and self._is_constant_assignment(child):
                for subchild in child.children:
                    if (
                        subchild.type == 'constant'
                        and 'constants' in metadata
                        and isinstance(metadata['constants'], list)
                    ):
                        metadata['constants'].append(subchild.text.decode('utf8'))
            elif child.type == 'call':
                method = self._get_call_method(child)
                if method == 'include':
                    for arg in child.children:
                        if arg.type == 'argument_list':
                            for module_arg in arg.children:
                                if (
                                    module_arg.type == 'constant'
                                    and 'included_modules' in metadata
                                    and isinstance(metadata['included_modules'], list)
                                ):
                                    metadata['included_modules'].append(
                                        module_arg.text.decode('utf8')
                                    )
                elif method == 'extend':
                    for arg in child.children:
                        if arg.type == 'argument_list':
                            for module_arg in arg.children:
                                if (
                                    module_arg.type == 'constant'
                                    and 'extended_modules' in metadata
                                    and isinstance(metadata['extended_modules'], list)
                                ):
                                    metadata['extended_modules'].append(
                                        module_arg.text.decode('utf8')
                                    )
                elif method == 'prepend':
                    for arg in child.children:
                        if arg.type == 'argument_list':
                            for module_arg in arg.children:
                                if (
                                    module_arg.type == 'constant'
                                    and 'prepended_modules' in metadata
                                    and isinstance(metadata['prepended_modules'], list)
                                ):
                                    metadata['prepended_modules'].append(
                                        module_arg.text.decode('utf8')
                                    )
                elif method == 'autoload':
                    for arg in child.children:
                        if arg.type == 'argument_list':
                            for a in arg.children:
                                if a.type == 'string':
                                    dep = a.text.decode('utf8').strip("'\"")
                                    if 'dependencies' in metadata:
                                        if isinstance(metadata['dependencies'], list):
                                            metadata['dependencies'].append(dep)
                                        elif isinstance(metadata['dependencies'], dict):
                                            metadata['dependencies'].setdefault(
                                                'external', []
                                            ).append(dep)
                elif method == 'attr_accessor':
                    for arg in child.children:
                        if arg.type == 'argument_list':
                            for symbol in arg.children:
                                if (
                                    (symbol.type == 'symbol' or symbol.type == 'simple_symbol')
                                    and 'attr_accessors' in metadata
                                    and isinstance(metadata['attr_accessors'], list)
                                ):
                                    attr_text = symbol.text.decode('utf8')
                                    attr_name = attr_text.lstrip(':').strip("'\"")
                                    metadata['attr_accessors'].append(attr_name)
                elif method == 'attr_reader':
                    for arg in child.children:
                        if arg.type == 'argument_list':
                            for symbol in arg.children:
                                if (
                                    (symbol.type == 'symbol' or symbol.type == 'simple_symbol')
                                    and 'attr_readers' in metadata
                                    and isinstance(metadata['attr_readers'], list)
                                ):
                                    attr_text = symbol.text.decode('utf8')
                                    attr_name = attr_text.lstrip(':').strip("'\"")
                                    metadata['attr_readers'].append(attr_name)
                elif method == 'attr_writer':
                    for arg in child.children:
                        if arg.type == 'argument_list':
                            for symbol in arg.children:
                                if (
                                    (symbol.type == 'symbol' or symbol.type == 'simple_symbol')
                                    and 'attr_writers' in metadata
                                    and isinstance(metadata['attr_writers'], list)
                                ):
                                    attr_text = symbol.text.decode('utf8')
                                    attr_name = attr_text.lstrip(':').strip("'\"")
                                    metadata['attr_writers'].append(attr_name)

    def _analyze_class_call(self, call_node, metadata: Dict[str, Any]) -> None:
        """Analyze call nodes within class body (attr_*, include, extend, etc.)."""
        method = self._get_call_method(call_node)

        if method == 'attr_accessor':
            for arg in call_node.children:
                if arg.type == 'argument_list':
                    for symbol in arg.children:
                        if (
                            (symbol.type == 'symbol' or symbol.type == 'simple_symbol')
                            and 'attr_accessors' in metadata
                            and isinstance(metadata['attr_accessors'], list)
                        ):
                            attr_text = symbol.text.decode('utf8')
                            attr_name = attr_text.lstrip(':').strip("'\"")
                            metadata['attr_accessors'].append(attr_name)
        elif method == 'attr_reader':
            for arg in call_node.children:
                if arg.type == 'argument_list':
                    for symbol in arg.children:
                        if (
                            (symbol.type == 'symbol' or symbol.type == 'simple_symbol')
                            and 'attr_readers' in metadata
                            and isinstance(metadata['attr_readers'], list)
                        ):
                            attr_text = symbol.text.decode('utf8')
                            attr_name = attr_text.lstrip(':').strip("'\"")
                            metadata['attr_readers'].append(attr_name)
        elif method == 'attr_writer':
            for arg in call_node.children:
                if arg.type == 'argument_list':
                    for symbol in arg.children:
                        if (
                            (symbol.type == 'symbol' or symbol.type == 'simple_symbol')
                            and 'attr_writers' in metadata
                            and isinstance(metadata['attr_writers'], list)
                        ):
                            attr_text = symbol.text.decode('utf8')
                            attr_name = attr_text.lstrip(':').strip("'\"")
                            metadata['attr_writers'].append(attr_name)
        elif method == 'include':
            for arg in call_node.children:
                if arg.type == 'argument_list':
                    for module_arg in arg.children:
                        if (
                            module_arg.type == 'constant'
                            and 'included_modules' in metadata
                            and isinstance(metadata['included_modules'], list)
                        ):
                            metadata['included_modules'].append(module_arg.text.decode('utf8'))
        elif method == 'extend':
            for arg in call_node.children:
                if arg.type == 'argument_list':
                    for module_arg in arg.children:
                        if (
                            module_arg.type == 'constant'
                            and 'extended_modules' in metadata
                            and isinstance(metadata['extended_modules'], list)
                        ):
                            metadata['extended_modules'].append(module_arg.text.decode('utf8'))
        elif method == 'prepend':
            for arg in call_node.children:
                if arg.type == 'argument_list':
                    for module_arg in arg.children:
                        if (
                            module_arg.type == 'constant'
                            and 'prepended_modules' in metadata
                            and isinstance(metadata['prepended_modules'], list)
                        ):
                            metadata['prepended_modules'].append(module_arg.text.decode('utf8'))
        elif method == 'autoload':
            # autoload :X, 'path' => extract dependency
            for arg in call_node.children:
                if arg.type == 'argument_list':
                    for a in arg.children:
                        if a.type == 'string':
                            dep = a.text.decode('utf8').strip("'\"")
                            if 'dependencies' in metadata:
                                if isinstance(metadata['dependencies'], list):
                                    metadata['dependencies'].append(dep)
                                elif isinstance(metadata['dependencies'], dict):
                                    metadata['dependencies'].setdefault('external', []).append(dep)

    def _get_decision_node_types(self) -> Set[str]:
        """Ruby-specific decision nodes."""
        return {
            'if',
            'unless',
            'while',
            'until',
            'for',
            'case',
            'when',
            'rescue',
            'elsif',
            'conditional',
        }

    def _detect_language_patterns(
        self, node, patterns: Dict[str, List[str]], metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[str]]:
        """Detect Ruby-specific patterns like Rails, RSpec, etc."""
        text = ''
        if hasattr(node, 'text') and node.text is not None:
            try:
                text = node.text.decode('utf8')
            except Exception:
                text = ''

        # Rails patterns
        if 'ActiveRecord::Base' in text:
            patterns['framework'].append('rails_model')
        if 'ApplicationController' in text:
            patterns['framework'].append('rails_controller')

        # RSpec patterns
        def find_rspec_and_describe(n):
            types = []
            texts = []
            stack = [n]
            while stack:
                current = stack.pop()
                t = getattr(current, 'type', None)
                tx = (
                    getattr(current, 'text', b'').decode('utf8').strip()
                    if hasattr(current, 'text')
                    else ''
                )
                types.append(t)
                texts.append(tx)
                stack.extend(getattr(current, 'children', []))
            has_rspec = any(t == 'constant' and tx == 'RSpec' for t, tx in zip(types, texts))
            has_describe = any(
                t == 'identifier' and tx == 'describe' for t, tx in zip(types, texts)
            )
            return has_rspec and has_describe

        if getattr(node, 'type', None) == 'call':
            if find_rspec_and_describe(node):
                patterns['framework'].append('rspec_test')

        if 'RSpec.describe' in text:
            patterns['framework'].append('rspec_test')

        # Minitest patterns
        if 'test_' in text and 'def test_' in text:
            patterns['framework'].append('minitest')

        return patterns

    def _get_import_node_types(self) -> List[str]:
        """Get node types that represent imports for Ruby."""
        # Ruby uses method calls for imports, not specific nodes
        return []  # Ruby doesn't have import nodes, uses require calls

    def _is_comment_node(self, node) -> bool:
        """Check if node is a Ruby comment."""
        return node.type == 'comment'

    def _check_hardcoded_credentials(self, node, text: str) -> Optional[Dict[str, Any]]:
        """Override to check for Ruby-specific hardcoded credentials."""
        # First use the base implementation
        base_result = super()._check_hardcoded_credentials(node, text)
        if base_result:
            return base_result

        # Ruby-specific checks
        credential_patterns = [
            'password',
            'passwd',
            'pwd',
            'secret',
            'api_key',
            'apikey',
            'access_token',
            'auth_token',
            'private_key',
            'salt',
        ]

        text_upper = text.upper()

        # Check for Ruby constant assignment patterns
        for pattern in credential_patterns:
            if pattern.upper() in text_upper:
                # Check for string assignment
                if '=' in text and any(quote in text for quote in ['"', "'", '%q', '%Q']):
                    # Check if it's a hardcoded value (not reading from ENV)
                    if 'ENV[' not in text and 'ENV.fetch' not in text:
                        # Look for patterns like SECRET = "hardcoded_value"
                        const_pattern = (
                            rf'\b[A-Z_]*{re.escape(pattern.upper())}[A-Z_]*\s*=\s*["\'][^"\']+'
                        )
                        if re.search(const_pattern, text_upper):
                            return {
                                'type': 'hardcoded_credential',
                                'severity': 'critical',
                                'description': f'Possible hardcoded {pattern} in constant',
                            }

        return None

    def _extract_dependencies_from_imports(self, import_nodes) -> List[str]:
        """Extract Ruby-specific import dependencies."""
        deps = set()

        for node in import_nodes:
            # In Ruby, imports are call nodes
            if node.type == 'call':
                method = self._get_call_method(node)
                if method in ['require', 'require_relative', 'load', 'autoload']:
                    # Extract the required file/gem
                    for child in node.children:
                        if child.type == 'argument_list':
                            for arg in child.children:
                                if arg.type == 'string':
                                    dep_text = arg.text.decode('utf8')
                                    # Remove quotes
                                    dep = dep_text.strip("'\"")
                                    # Clean up path
                                    if dep.startswith('./') or dep.startswith('../'):
                                        deps.add(dep)  # Keep relative paths
                                    else:
                                        # Extract gem name from require 'gem/module'
                                        deps.add(dep.split('/')[0])
                                elif arg.type == 'simple_symbol' and method == 'autoload':
                                    # autoload :Symbol, 'path' - skip the symbol, get the path from next arg
                                    pass

        return sorted(list(deps))

    def _extract_dependencies_from_import(self, import_node) -> List[str]:
        """Extract dependencies from a single import node (for DependencyAnalysisMixin)."""
        deps = []

        if import_node.type == 'call':
            method = self._get_call_method(import_node)
            if method in ['require', 'require_relative', 'load']:
                # Extract the required file/gem
                for child in import_node.children:
                    if child.type == 'argument_list':
                        for arg in child.children:
                            if arg.type == 'string':
                                dep = arg.text.decode('utf8').strip("'\"")
                                # Clean up path
                                if dep.startswith('./') or dep.startswith('../'):
                                    deps.append(dep)  # Keep relative paths
                                else:
                                    # Extract gem name
                                    deps.append(dep.split('/')[0])

        return deps

    def _extract_dependencies_from_node(self, node) -> List[str]:
        """Extract dependencies from a single node."""
        deps = []
        if node.type == 'call':
            method = self._get_call_method(node)
            if method in ['require', 'require_relative', 'load']:
                for child in node.children:
                    if child.type == 'argument_list':
                        for arg in child.children:
                            if arg.type == 'string':
                                dep = arg.text.decode('utf8').strip("'\"")
                                deps.append(dep)
        return deps

    def _extract_test_metadata(self, node) -> Dict[str, Any]:
        """Extract metadata for test blocks (it, describe, context)."""
        metadata = {
            'test_type': self._get_call_method(node),  # 'it', 'describe', 'context'
            'complexity': self._calculate_complexity(node),
            'todos': self._extract_todos(node),
        }

        # Extract test description from the first string argument
        for child in node.children:
            if child.type == 'argument_list':
                for arg in child.children:
                    if arg.type == 'string':
                        description = arg.text.decode('utf8').strip("'\"")
                        metadata['description'] = description
                        break

        # Detect patterns
        patterns_result = self._detect_patterns(node)
        metadata['patterns'] = patterns_result

        # Detect security issues in tests
        metadata['security'] = self._detect_security_issues(node)

        return metadata

    def _extract_node_name(self, node) -> Optional[str]:
        """Extract name from Ruby-specific nodes."""
        # For methods (regular or singleton)
        if node.type in ['method', 'singleton_method']:
            for child in node.children:
                if child.type == 'identifier':
                    return child.text.decode('utf8')

        # For classes
        elif node.type == 'class':
            for child in node.children:
                if child.type == 'constant':
                    return child.text.decode('utf8')

        # For modules
        elif node.type == 'module':
            for child in node.children:
                if child.type == 'constant':
                    return child.text.decode('utf8')

        # For calls (test blocks, imports)
        elif node.type == 'call':
            method_name = self._get_call_method(node)
            if method_name in ['it', 'describe', 'context', 'test']:
                # Extract test description from first string argument
                for child in node.children:
                    if child.type == 'argument_list':
                        for arg in child.children:
                            if arg.type == 'string':
                                desc = arg.text.decode('utf8').strip("'\"")
                                # Truncate long descriptions
                                return desc[:50] + '...' if len(desc) > 50 else desc
            return method_name

        # Fall back to base implementation
        return super()._extract_node_name(node)

    async def chunk(self, content: str, file_path: str):
        """Chunk Ruby code using tree-sitter AST parsing.

        Stores file lines and text for internal use by metadata extraction methods.

        Args:
            content: Ruby source code as string
            file_path: Path to the Ruby file

        Returns:
            List of Chunk objects with Ruby-specific metadata
        """
        # Store file content for metadata extraction methods that need access to full text
        self._current_file_lines = content.split('\n')
        self._current_file_text = content

        return await super().chunk(content, file_path)
