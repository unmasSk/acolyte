"""
Python chunker using tree-sitter-languages.
Extracts pragmatic metadata for superior search and context quality.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    pass

from typing import Dict, List, Any, Set
from tree_sitter_languages import get_language  # type: ignore

from acolyte.models.chunk import ChunkType
from acolyte.core.logging import logger
from acolyte.rag.chunking.base import LanguageChunker
from acolyte.rag.chunking.mixins import (
    SecurityAnalysisMixin,
    PatternDetectionMixin,
    DependencyAnalysisMixin,
)


class PythonChunker(
    LanguageChunker, SecurityAnalysisMixin, PatternDetectionMixin, DependencyAnalysisMixin
):
    """
    Python-specific chunker using tree-sitter.

    Extracts useful metadata without academic metrics that nobody uses.
    Focus on what developers actually search for.
    """

    def __init__(self):
        """Initialize with tree-sitter parser and project configuration."""
        super().__init__()

        # Get project name from config for dependency analysis
        self.project_name = self.config.get('project.name', 'acolyte')

    def _get_language_name(self) -> str:
        """Return language identifier."""
        return 'python'

    def _get_tree_sitter_language(self) -> Any:
        """Get Python language for tree-sitter."""
        logger.info("[UNTESTED PATH] python._get_tree_sitter_language called")
        return get_language('python')

    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """
        Python-specific node types to chunk.

        Comprehensive mapping for all relevant Python constructs.
        """
        return {
            # Functions
            'function_definition': ChunkType.FUNCTION,
            'async_function_definition': ChunkType.FUNCTION,
            # Classes
            'class_definition': ChunkType.CLASS,
            # Imports (BaseChunker groups these)
            'import_statement': ChunkType.IMPORTS,
            'import_from_statement': ChunkType.IMPORTS,
            'future_import_statement': ChunkType.IMPORTS,
            # Special handling
            'decorated_definition': ChunkType.UNKNOWN,  # Refined in _create_chunk_from_node
        }

    def _get_import_node_types(self) -> List[str]:
        """Python-specific import node types."""
        return [
            'import_statement',
            'import_from_statement',
            'future_import_statement',
        ]

    def _get_decision_node_types(self) -> Set[str]:
        """Python-specific decision nodes for complexity calculation."""
        # Get base types from mixin
        base_types: Set[str] = set()
        if hasattr(super(), '_get_decision_node_types'):
            base_types = super()._get_decision_node_types()
        # Add Python-specific decision nodes
        base_types.update({'elif_clause', 'except_clause', 'with_statement', 'boolean_operator'})
        return base_types

    def _get_security_patterns(self) -> List[Callable[[Any, str], dict]]:
        """Override to add Python-specific security checks."""
        base_patterns = super()._get_security_patterns()
        base_patterns.extend(
            [self._check_python_exec_eval, self._check_pickle_usage, self._check_ssl_verification]
        )
        return base_patterns

    def _check_python_exec_eval(self, node, text: str) -> Dict[str, Any]:
        """Check for exec/eval usage."""
        if 'exec(' in text or 'eval(' in text:
            return {
                'type': 'code_execution',
                'severity': 'critical',
                'description': 'Dynamic code execution with exec/eval',
            }
        return {}

    def _check_pickle_usage(self, node, text: str) -> Dict[str, Any]:
        """Check for unsafe pickle usage."""
        if 'pickle.load' in text or 'pickle.loads' in text:
            return {
                'type': 'unsafe_deserialization',
                'severity': 'high',
                'description': 'Pickle can execute arbitrary code',
            }
        return {}

    def _check_ssl_verification(self, node, text: str) -> Dict[str, Any]:
        """Check for disabled SSL verification."""
        if 'verify=False' in text or 'verify = False' in text:
            return {
                'type': 'ssl_verification_disabled',
                'severity': 'medium',
                'description': 'SSL certificate verification disabled',
            }
        return {}

    def _create_chunk_from_node(
        self, node, lines: List[str], file_path: str, chunk_type: ChunkType, processed_ranges
    ):
        """Override to handle Python-specific cases and extract rich metadata."""
        # Handle decorated definitions
        if node.type == 'decorated_definition':
            for child in node.children:
                if child.type in ['function_definition', 'class_definition']:
                    if child.type == 'function_definition':
                        chunk_type = self._determine_function_type(child, node)
                    else:
                        chunk_type = ChunkType.CLASS

                    return self._create_decorated_chunk(
                        node, child, lines, file_path, chunk_type, processed_ranges
                    )

        # Check if it's a method
        if node.type == 'function_definition':
            chunk_type = self._determine_function_type(node)

        # Standard processing
        chunk = super()._create_chunk_from_node(
            node, lines, file_path, chunk_type, processed_ranges
        )

        if not chunk:
            return None

        # Extract and populate language-specific metadata
        self._populate_language_metadata(chunk, node)

        return chunk

    def _determine_function_type(self, func_node, decorated_node=None) -> ChunkType:
        """Determine if a function is a method, constructor, property, etc."""
        # Get function name
        name = None
        for child in func_node.children:
            if child.type == 'identifier':
                name = child.text.decode('utf8')
                break

        # Check if it's a magic method
        if name and name.startswith('__') and name.endswith('__'):
            if name == '__init__':
                logger.info("[UNTESTED PATH] python constructor detected")
                return ChunkType.CONSTRUCTOR
            # Other magic methods are still methods
            logger.info("[UNTESTED PATH] python magic method detected")
            return ChunkType.METHOD

        # Check if it's inside a class
        parent = func_node.parent
        while parent:
            if parent.type == 'class_definition':
                return ChunkType.METHOD
            parent = parent.parent

        # Check for test functions
        if name and (name.startswith('test_') or name.startswith('Test')):
            logger.info("[UNTESTED PATH] python test function detected")
            return ChunkType.TESTS

        # Check decorators
        if decorated_node:
            decorators = self._get_decorators(decorated_node)
            if any('@property' in d for d in decorators):
                logger.info("[UNTESTED PATH] python property decorator detected")
                return ChunkType.PROPERTY
            if any('@staticmethod' in d or '@classmethod' in d for d in decorators):
                logger.info("[UNTESTED PATH] python static/classmethod detected")
                return ChunkType.METHOD
            if any('test' in d.lower() or 'pytest' in d for d in decorators):
                return ChunkType.TESTS

        return ChunkType.FUNCTION

    def _detect_language_patterns(
        self, node, patterns: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """Override to detect Python-specific patterns."""
        content = node.text.decode('utf8')

        # Check for mutable default arguments
        if re.search(r'def\s+\w+\s*\([^)]*=\s*(\[\]|\{\})', content):
            logger.info("[UNTESTED PATH] python mutable default argument")
            patterns['anti'].append('mutable_default_argument')

        # Framework patterns
        if any(d in content for d in ['@app.route', '@router.get', '@router.post']):
            patterns['framework'].append('web_endpoint')

        if '@pytest.' in content or 'pytest.fixture' in content:
            patterns['framework'].append('pytest')

        if 'django' in content.lower() and any(
            k in content for k in ['Model', 'View', 'Serializer']
        ):
            patterns['framework'].append('django')

        return patterns

    def _calculate_quality_metrics(self, node) -> Dict[str, Any]:
        """Calculate simple quality indicators."""
        content = node.text.decode('utf8')
        test_hints = []

        if any(imp in content for imp in ['unittest', 'pytest', 'mock', 'Mock']):
            test_hints.append('has_test_imports')
        if 'assert' in content:
            logger.info("[UNTESTED PATH] python assertions detected")
            test_hints.append('uses_assertions')
        if any(m in content for m in ['mock', 'Mock', 'patch']):
            test_hints.append('uses_mocks')

        return {
            'has_docstring': self._has_docstring(node),
            'test_coverage_hints': test_hints,
        }

    def _has_docstring(self, node) -> bool:
        """Check if the node has a docstring."""
        for child in node.children:
            if child.type == 'block':
                # First statement in block
                for stmt in child.children:
                    if stmt.type == 'expression_statement':
                        for expr in stmt.children:
                            if expr.type == 'string':
                                return True
                    break
        return False

    def _create_decorated_chunk(
        self,
        decorated_node,
        definition_node,
        lines: List[str],
        file_path: str,
        chunk_type: ChunkType,
        processed_ranges,
    ):
        """Create chunk for decorated definition."""
        start_line = decorated_node.start_point[0]
        end_line = decorated_node.end_point[0]

        # Check if already processed for this node type
        node_type_ranges = processed_ranges.get(decorated_node.type, set())
        if any(start_line <= s <= end_line and end_line >= e for (s, e) in node_type_ranges):
            return None

        content = '\n'.join(lines[start_line : end_line + 1])
        name = self._extract_node_name(definition_node)

        # Mark as processed
        if decorated_node.type not in processed_ranges:
            processed_ranges[decorated_node.type] = set()
        processed_ranges[decorated_node.type].add((start_line, end_line))

        chunk = self._create_chunk(
            content=content,
            chunk_type=chunk_type,
            file_path=file_path,
            start_line=start_line + 1,
            end_line=end_line + 1,
            name=name,
        )

        if chunk:
            decorators = self._get_decorators(decorated_node)

            # Extract and populate language-specific metadata
            self._populate_language_metadata(chunk, definition_node, decorators)

        return chunk

    def _populate_language_metadata(self, chunk, node, decorators=None):
        """Helper to populate language-specific metadata for a chunk.

        Args:
            chunk: The chunk to populate metadata for
            node: The AST node to extract metadata from
            decorators: Optional list of decorators (for decorated nodes)
        """
        # Extract base metadata based on node type
        metadata = {}

        if decorators:
            metadata['decorators'] = decorators

        if node.type in ['function_definition', 'async_function_definition']:
            func_metadata = self._extract_function_metadata(node)
            metadata.update(func_metadata)
            # If decorators were provided, override those from function metadata
            if decorators:
                metadata['decorators'] = decorators
        elif node.type == 'class_definition':
            class_metadata = self._extract_class_metadata(node)
            metadata.update(class_metadata)
            # If decorators were provided, override those from class metadata
            if decorators:
                metadata['decorators'] = decorators

        # Add common metadata
        if hasattr(self, '_calculate_complexity') and node.type in [
            'function_definition',
            'async_function_definition',
        ]:
            metadata['complexity'] = self._calculate_complexity(node)

        # Extract patterns
        patterns = self._detect_language_patterns(node, {'anti': [], 'framework': []})
        if patterns['anti'] or patterns['framework']:
            metadata['patterns'] = patterns

        # Extract quality metrics
        quality = self._calculate_quality_metrics(node)
        if quality:
            metadata['quality'] = quality

        # Extract TODOs
        todos = self._extract_todos(node)
        if todos:
            metadata['todos'] = todos

        # Security patterns
        security_issues = []
        for check in self._get_security_patterns():
            issue = check(node, node.text.decode('utf8'))
            if issue:
                security_issues.append(issue)
        if security_issues:
            metadata['security'] = security_issues

        # Assign the metadata
        if metadata:
            chunk.metadata.language_specific = metadata

    def _get_decorators(self, decorated_node) -> List[str]:
        """Extract decorator names from decorated definition."""
        decorators = []

        for child in decorated_node.children:
            if child.type == 'decorator':
                dec_text = child.text.decode('utf8').strip()
                decorators.append(dec_text)

        return decorators

    def _extract_function_metadata(self, func_node) -> Dict[str, Any]:
        """Extract Python-specific function metadata."""
        # Check if function is async by looking for 'async' keyword at the start
        is_async = False
        if func_node.children:
            first_child = func_node.children[0]
            if first_child.type == 'async' or (
                hasattr(first_child, 'text') and first_child.text.decode('utf8') == 'async'
            ):
                is_async = True

        metadata: Dict[str, Any] = {
            'is_async': is_async,
            'is_generator': False,
            'is_abstract': False,
            'parameters': [],
            'return_type': None,
            'throws': [],
            'decorators': [],
            'visibility': 'public',  # Python default
            'modifiers': [],
        }

        # Get function name for special checks
        func_name = None
        for child in func_node.children:
            if child.type == 'identifier':
                func_name = child.text.decode('utf8')
                break

        # Visibility based on naming convention
        if func_name:
            if func_name.startswith('__') and not func_name.endswith('__'):
                logger.info("[UNTESTED PATH] python private method")
                metadata['visibility'] = 'private'
                metadata['modifiers'].append('private')
            elif func_name.startswith('_'):
                metadata['visibility'] = 'protected'
                metadata['modifiers'].append('protected')

            # Check if it's a magic method
            if func_name.startswith('__') and func_name.endswith('__'):
                metadata['modifiers'].append('magic')

        # Extract parameters with types
        for child in func_node.children:
            if child.type == 'parameters':
                metadata['parameters'] = self._extract_parameters_with_types(child)

        # Extract return type from annotation
        for child in func_node.children:
            if child.type == 'type':
                logger.info("[UNTESTED PATH] python function return type")
                metadata['return_type'] = child.text.decode('utf8')

        # Check if generator
        def has_yield(n: Any) -> bool:
            if n.type in ['yield_expression', 'yield', 'yield_statement']:
                return True
            # Also check text content for yield keyword
            if hasattr(n, 'text') and b'yield' in n.text:
                # Verify it's actually a yield, not just in a string or comment
                text = n.text.decode('utf8')
                if 'yield ' in text and not any(q in text for q in ['"yield', "'yield", '#', '//']):
                    return True
            return any(has_yield(c) for c in n.children)

        metadata['is_generator'] = has_yield(func_node)
        if metadata['is_generator']:
            metadata['modifiers'].append('generator')

        # Get decorators from parent
        if func_node.parent and func_node.parent.type == 'decorated_definition':
            decorators = self._get_decorators(func_node.parent)
            metadata['decorators'] = decorators

            # Check for abstract
            if any('@abstractmethod' in d for d in decorators):
                logger.info("[UNTESTED PATH] python abstract method")
                metadata['is_abstract'] = True
                metadata['modifiers'].append('abstract')

            # Check for static/classmethod
            if any('@staticmethod' in d for d in decorators):
                metadata['modifiers'].append('static')
            if any('@classmethod' in d for d in decorators):
                metadata['modifiers'].append('classmethod')

        # Extract potential exceptions from raises
        metadata['throws'] = self._extract_exceptions(func_node)

        return metadata

    def _extract_parameters_with_types(self, params_node) -> List[Dict[str, Any]]:
        """Extract parameters with type annotations and defaults."""
        parameters = []

        for param in params_node.children:
            if param.type in [
                'identifier',
                'typed_parameter',
                'default_parameter',
                'typed_default_parameter',
            ]:
                param_info = {
                    'name': None,
                    'type': None,
                    'optional': False,
                    'default': None,
                }

                if param.type == 'identifier':
                    param_info['name'] = param.text.decode('utf8')
                else:
                    # Complex parameter node
                    for child in param.children:
                        if child.type == 'identifier':
                            param_info['name'] = child.text.decode('utf8')
                        elif child.type == 'type':
                            logger.info("[UNTESTED PATH] python parameter type annotation")
                            param_info['type'] = child.text.decode('utf8')
                        elif child.type in [
                            'integer',
                            'string',
                            'true',
                            'false',
                            'none',
                            'list',
                            'dictionary',
                        ]:
                            param_info['optional'] = True
                            param_info['default'] = child.text.decode('utf8')

                if param_info['name'] and param_info['name'] not in ['self', 'cls']:
                    parameters.append(param_info)

        return parameters

    def _extract_exceptions(self, func_node) -> List[str]:
        """Extract exceptions that might be raised."""
        exceptions: List[str] = []

        # Look for raise statements
        def find_raises(n: Any) -> None:
            if n.type == 'raise_statement':
                for child in n.children:
                    if child.type == 'call':
                        # Get the exception class name
                        for c in child.children:
                            if c.type == 'identifier':
                                exceptions.append(c.text.decode('utf8'))
                                logger.info("[UNTESTED PATH] python exception found in raise")
                                break
            for child in n.children:
                find_raises(child)

        find_raises(func_node)

        return list(set(exceptions))

    def _extract_class_metadata(self, class_node) -> Dict[str, Any]:
        """Extract Python-specific class metadata."""
        metadata: Dict[str, Any] = {
            'base_classes': [],
            'methods': [],
            'properties': [],
            'class_variables': [],
            'instance_variables': [],
            'decorators': [],
            'is_abstract': False,
            'is_dataclass': False,
            'has_slots': False,
            'metaclass': None,
            'visibility': 'public',
            'modifiers': [],
        }

        # Get class name for visibility check
        class_name = None
        for child in class_node.children:
            if child.type == 'identifier':
                class_name = child.text.decode('utf8')
                break

        if class_name and class_name.startswith('_'):
            metadata['visibility'] = 'internal'
            metadata['modifiers'].append('internal')

        # Extract base classes and metaclass
        for child in class_node.children:
            if child.type == 'argument_list':
                for arg in child.children:
                    if arg.type == 'identifier':
                        base = arg.text.decode('utf8')
                        metadata['base_classes'].append(base)
                        # Check for ABC
                        if base in ['ABC', 'ABCMeta']:
                            metadata['is_abstract'] = True
                            metadata['modifiers'].append('abstract')
                    elif arg.type == 'keyword_argument':
                        # Check for metaclass
                        if 'metaclass' in arg.text.decode('utf8'):
                            metadata['metaclass'] = arg.text.decode('utf8').split('=')[1].strip()

        # Get methods and properties
        for child in class_node.children:
            if child.type == 'block':
                for stmt in child.children:
                    if stmt.type in ['function_definition', 'async_function_definition']:
                        method_name = self._extract_node_name(stmt)
                        if method_name:
                            metadata['methods'].append(method_name)

                            # Check if it's a property
                            if stmt.parent and stmt.parent.type == 'decorated_definition':
                                decorators = self._get_decorators(stmt.parent)
                                if any('@property' in d for d in decorators):
                                    metadata['properties'].append(method_name)

                    # Class variables
                    elif stmt.type == 'expression_statement':
                        for expr in stmt.children:
                            if expr.type == 'assignment':
                                for target in expr.children:
                                    if target.type == 'identifier':
                                        var_name = target.text.decode('utf8')
                                        if not var_name.startswith('_'):
                                            metadata['class_variables'].append(var_name)
                                        break

                    # Check for __slots__
                    if '__slots__' in stmt.text.decode('utf8'):
                        logger.info("[UNTESTED PATH] python __slots__ detected")
                        metadata['has_slots'] = True
                        metadata['modifiers'].append('slots')

        # Get decorators
        if class_node.parent and class_node.parent.type == 'decorated_definition':
            metadata['decorators'] = self._get_decorators(class_node.parent)
            metadata['is_dataclass'] = any('@dataclass' in d for d in metadata['decorators'])
            if metadata['is_dataclass']:
                metadata['modifiers'].append('dataclass')

        return metadata

    def _extract_dependencies_from_imports(self, import_nodes) -> List[str]:
        """Extract Python-specific import dependencies with better parsing."""
        deps = set()

        for node in import_nodes:
            if node.type == 'import_statement':
                # import x, y, z
                for child in node.children:
                    if child.type in ['dotted_name', 'aliased_import']:
                        text = child.text.decode('utf8')
                        if ' as ' in text:
                            text = text.split(' as ')[0]
                        # Only keep the root module
                        deps.add(text.split('.')[0])

            elif node.type == 'import_from_statement':
                # from x import y
                module_name = None
                for child in node.children:
                    if child.type == 'dotted_name':
                        module_name = child.text.decode('utf8')
                    elif child.type == 'relative_import':
                        # Skip relative imports for external deps
                        continue

                if module_name:
                    deps.add(module_name.split('.')[0])

        return sorted(list(deps))

    def _is_internal_dependency(self, dep: str) -> bool:
        """Override to check if dependency is internal to Python project."""
        return dep.startswith('.') or dep.startswith(self.project_name)

    def _is_comment_node(self, node) -> bool:
        """Check if node is a Python comment."""
        logger.info("[UNTESTED PATH] python._is_comment_node called")
        return node.type in ['comment', 'line_comment', 'block_comment']
