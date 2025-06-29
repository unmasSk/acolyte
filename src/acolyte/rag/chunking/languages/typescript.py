"""
TypeScript/JavaScript chunker using tree-sitter-languages.
Handles JS, TS, JSX, and TSX files with rich metadata extraction.
"""

from typing import Dict, List, Any, Callable, Optional
import re
from tree_sitter_languages import get_language  # type: ignore

from acolyte.models.chunk import Chunk, ChunkType
from acolyte.core.logging import logger
from acolyte.rag.chunking.base import BaseChunker
from acolyte.rag.chunking.mixins import SecurityAnalysisMixin, PatternDetectionMixin


class TypeScriptChunker(BaseChunker, SecurityAnalysisMixin, PatternDetectionMixin):
    """
    TypeScript/JavaScript chunker using tree-sitter.

    Handles:
    - JavaScript (.js, .mjs, .cjs)
    - TypeScript (.ts, .mts, .cts)
    - JSX (.jsx)
    - TSX (.tsx)

    Extracts rich metadata for superior search and context.
    """

    def __init__(self):
        """Initialize with appropriate parser based on file extension."""
        # Will be set based on file type
        self._is_typescript = False
        self._is_jsx = False
        super().__init__()

    def _get_language_name(self) -> str:
        """Return language identifier."""
        # Always return 'javascript' for config lookup
        return 'javascript'

    def _get_tree_sitter_language(self) -> Any:
        """Get appropriate language for tree-sitter."""
        # Default to JavaScript, will be overridden in chunk()
        return get_language('javascript')

    async def chunk(self, content: str, file_path: str) -> List[Chunk]:
        """Override to set correct parser based on file extension."""
        # Determine file type safely
        ext = ''
        if file_path and '.' in file_path:
            ext = file_path.lower().split('.')[-1]

        # Check file extension
        if ext == 'tsx':
            # TSX is both TypeScript and JSX
            self._is_typescript = True
            self._is_jsx = True
            self.language = get_language('tsx')
            self.parser.set_language(self.language)
        elif ext in ['ts', 'mts', 'cts']:
            self._is_typescript = True
            self.language = get_language('typescript')
            self.parser.set_language(self.language)
        elif ext == 'jsx':
            self._is_jsx = True
            self.language = get_language('javascript')  # JSX uses JS parser
            self.parser.set_language(self.language)
        else:
            # Regular JavaScript (.js, .mjs, .cjs, or no extension)
            self.language = get_language('javascript')
            self.parser.set_language(self.language)

        return await super().chunk(content, file_path)

    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """
        JavaScript/TypeScript node types to chunk.

        Comprehensive mapping for all variants (JS/TS/JSX/TSX).
        """
        return {
            # Functions
            'function_declaration': ChunkType.FUNCTION,
            'function_expression': ChunkType.FUNCTION,
            'arrow_function': ChunkType.FUNCTION,
            'generator_function_declaration': ChunkType.FUNCTION,
            'method_definition': ChunkType.METHOD,
            # Classes
            'class_declaration': ChunkType.CLASS,
            'class_expression': ChunkType.CLASS,
            # Interfaces and Types (TypeScript)
            'interface_declaration': ChunkType.INTERFACE,
            'type_alias_declaration': ChunkType.TYPES,
            'enum_declaration': ChunkType.TYPES,
            # Modules
            'module': ChunkType.MODULE,
            'namespace_declaration': ChunkType.NAMESPACE,
            # Imports/Exports
            'import_statement': ChunkType.IMPORTS,
            'import_declaration': ChunkType.IMPORTS,
            'export_statement': ChunkType.IMPORTS,
            # Constants
            'variable_declaration': ChunkType.UNKNOWN,  # Will refine based on content
            'lexical_declaration': ChunkType.UNKNOWN,  # const/let - will refine
        }

    def _create_chunk_from_node(
        self, node, lines: List[str], file_path: str, chunk_type: ChunkType, processed_ranges
    ):
        """Override to handle JavaScript/TypeScript specific cases."""
        # Refine chunk type for variable declarations
        if node.type in ['variable_declaration', 'lexical_declaration']:
            chunk_type = self._determine_variable_type(node)
            if chunk_type == ChunkType.UNKNOWN:
                # Skip non-constant variables
                return None

        # Standard processing
        chunk = super()._create_chunk_from_node(
            node, lines, file_path, chunk_type, processed_ranges
        )

        if not chunk:
            return None

        # Extract comprehensive metadata
        chunk.metadata.language_specific = self._extract_comprehensive_metadata(node, lines)

        return chunk

    def _determine_variable_type(self, node) -> ChunkType:
        """Determine if a variable declaration should be chunked."""
        # Look for const declarations with UPPER_CASE names
        text = node.text.decode('utf8')

        # Check if it's a const
        if not text.startswith('const'):
            return ChunkType.UNKNOWN

        # Extract variable name
        for child in node.children:
            if child.type == 'variable_declarator':
                for subchild in child.children:
                    if subchild.type == 'identifier':
                        name = subchild.text.decode('utf8')
                        # Check if it's UPPER_CASE (constant)
                        if name.isupper() or '_' in name and name.upper() == name:
                            return ChunkType.CONSTANTS

        return ChunkType.UNKNOWN

    def _extract_comprehensive_metadata(self, node, lines: List[str]) -> Dict[str, Any]:
        """Extract rich metadata following README specifications."""
        metadata = {
            # Basic info
            'node_type': node.type,
            'modifiers': [],
            'visibility': 'public',  # Default in JS/TS
            'is_abstract': False,
            'is_async': False,
            'is_generator': False,
            'is_static': False,
            'is_exported': False,
            'parameters': [],
            'return_type': None,
            'throws': [],
            # TypeScript specific
            'generics': [],
            'decorators': [],
            'type_guards': [],
            # Quality indicators
            'quality': {'has_docstring': False, 'has_jsdoc': False, 'test_coverage_hints': []},
            # Dependencies
            'dependencies': {'internal': [], 'external': []},
        }

        # Extract modifiers and basic info
        self._extract_modifiers(node, metadata)

        # Extract parameters and return type
        if node.type in [
            'function_declaration',
            'function_expression',
            'arrow_function',
            'method_definition',
        ]:
            self._extract_function_info(node, metadata)

        # Calculate complexity using mixin
        metadata['complexity'] = self._calculate_complexity(node)

        # Detect patterns using mixin
        metadata['patterns'] = self._detect_patterns(node, metadata)

        # Extract TODOs using mixin
        metadata['todos'] = self._extract_todos(node)

        # Check quality indicators
        self._check_quality(node, lines, metadata)

        # Detect security issues using mixin
        metadata['security'] = self._detect_security_issues(node)

        # TypeScript specific extractions
        if self._is_typescript:
            self._extract_typescript_features(node, metadata)

        return metadata

    def _extract_modifiers(self, node, metadata: Dict[str, Any]):
        """Extract modifiers like async, static, export, etc."""
        text = node.text.decode('utf8')

        # Check common modifiers - look at first few lines, not just 50 chars
        first_lines = text.split('\n')[:3]
        text_start = ' '.join(first_lines).lower()

        if 'async' in text_start:
            metadata['is_async'] = True
            metadata['modifiers'].append('async')

        if (
            'function*' in text
            or 'generator' in node.type
            or ('async' in text_start and '*' in text)
        ):
            metadata['is_generator'] = True
            metadata['modifiers'].append('generator')

        if 'static' in text_start:
            metadata['is_static'] = True
            metadata['modifiers'].append('static')

        if 'export' in text_start:
            metadata['is_exported'] = True
            metadata['modifiers'].append('export')

        if 'abstract' in text_start:
            metadata['is_abstract'] = True
            metadata['modifiers'].append('abstract')

        # Check visibility (TypeScript)
        if 'private' in text_start:
            metadata['visibility'] = 'private'
            metadata['modifiers'].append('private')
        elif 'protected' in text_start:
            metadata['visibility'] = 'protected'
            metadata['modifiers'].append('protected')

    def _extract_function_info(self, node, metadata: Dict[str, Any]):
        """Extract function parameters and return type."""
        # Extract parameters
        for child in node.children:
            if child.type in ['formal_parameters', 'parameters']:
                params = []
                for param in child.children:
                    if param.type in ['identifier', 'required_parameter', 'optional_parameter']:
                        param_info = {
                            'name': None,
                            'type': None,
                            'optional': param.type == 'optional_parameter',
                            'default': None,
                        }

                        # Extract parameter details
                        param_text = param.text.decode('utf8')
                        if ':' in param_text:
                            # TypeScript typed parameter
                            parts = param_text.split(':')
                            param_info['name'] = parts[0].strip().rstrip('?')
                            param_info['type'] = parts[1].strip()
                        else:
                            param_info['name'] = param_text.strip()

                        if param_info['name']:
                            params.append(param_info)

                metadata['parameters'] = params

            # Extract return type (TypeScript)
            elif child.type == 'type_annotation':
                metadata['return_type'] = child.text.decode('utf8').lstrip(':').strip()

    def _check_quality(self, node, lines: List[str], metadata: Dict[str, Any]):
        """Check quality indicators."""
        text = node.text.decode('utf8')

        # Check for JSDoc
        if node.start_point[0] > 0:
            prev_line_idx = node.start_point[0] - 1
            if prev_line_idx < len(lines):
                prev_lines = '\n'.join(lines[max(0, prev_line_idx - 5) : node.start_point[0]])
                if '/**' in prev_lines and '*/' in prev_lines:
                    metadata['quality']['has_jsdoc'] = True
                    metadata['quality']['has_docstring'] = True

        # Test coverage hints
        if any(imp in text for imp in ['jest', 'mocha', 'chai', 'expect', 'test', 'describe']):
            metadata['quality']['test_coverage_hints'].append('has_test_imports')

        if 'mock' in text.lower() or 'stub' in text.lower():
            metadata['quality']['test_coverage_hints'].append('uses_mocks')

        if node.type in ['function_declaration', 'method_definition']:
            name = self._extract_node_name(node)
            if name and ('test' in name.lower() or 'spec' in name.lower()):
                metadata['quality']['test_coverage_hints'].append('is_test_function')

    def _extract_typescript_features(self, node, metadata: Dict[str, Any]):
        """Extract TypeScript-specific features."""
        text = node.text.decode('utf8')

        # Extract generics
        if '<' in text and '>' in text:
            # Simple generic extraction
            start = text.find('<')
            end = text.find('>')
            if start < end:
                generics_text = text[start + 1 : end]
                if generics_text and not any(op in generics_text for op in ['=', '(', ')']):
                    logger.info("[UNTESTED PATH] typescript generics extraction")
                    metadata['generics'] = [g.strip() for g in generics_text.split(',')]

        # Extract decorators
        if '@' in text:
            decorators = re.findall(r'@\w+(?:\([^)]*\))?', text)
            metadata['decorators'] = decorators

        # Type guards
        if 'is ' in text and node.type in ['function_declaration', 'method_definition']:
            metadata['type_guards'].append('has_type_predicate')

    # Override security patterns for JS/TS specific issues
    def _get_security_patterns(self) -> List[Callable[[Any, str], dict]]:
        """Override to add JS/TS specific security checks."""
        patterns = super()._get_security_patterns()
        patterns.extend([self._check_eval_usage, self._check_unsafe_regex, self._check_innerHTML])
        return patterns

    def _check_eval_usage(self, node, text: str) -> Dict[str, Any]:
        """Check for eval usage."""
        if 'eval(' in text:
            return {
                'type': 'eval_usage',
                'severity': 'high',
                'description': 'eval() can execute arbitrary code',
            }
        return {}

    def _check_unsafe_regex(self, node, text: str) -> Dict[str, Any]:
        """Check for unsafe regex patterns."""
        if 'new RegExp' in text and 'escape' not in text:
            logger.info("[UNTESTED PATH] typescript unsafe regex detected")
            return {
                'type': 'unsafe_regex',
                'severity': 'medium',
                'description': 'RegExp without proper escaping',
            }
        return {}

    def _check_innerHTML(self, node, text: str) -> Dict[str, Any]:
        """Check for innerHTML usage."""
        if 'innerHTML' in text:
            return {
                'type': 'unsafe_html_injection',
                'severity': 'high',
                'description': 'innerHTML can lead to XSS attacks',
            }
        return {}

    # Override pattern detection for JS/TS specific patterns
    def _detect_language_patterns(
        self, node, patterns: Dict[str, List[str]], metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[str]]:
        """Detect JS/TS specific patterns."""
        text = node.text.decode('utf8')

        # Anti-patterns
        # Check for callback hell - nested callbacks or promises
        if (text.count('function') > 3 or text.count('=>') > 3) and (
            'callback' in text.lower() or text.count('(') > 10
        ):
            patterns['anti'].append('callback_hell')

        # Framework patterns
        if 'useState' in text or 'useEffect' in text:
            patterns['framework'].append('react_hook')

        if any(
            pattern in text
            for pattern in [
                'app.get',
                'app.post',
                'router.',
                'Router()',
                '.use(',
                '.get(',
                '.post(',
            ]
        ):
            patterns['framework'].append('express_route')

        if '@Component' in text or '@Injectable' in text:
            patterns['framework'].append('angular_decorator')

        if 'Vue.component' in text or 'export default {' in text and 'data()' in text:
            patterns['framework'].append('vue_component')

        return patterns

    def _get_import_node_types(self) -> List[str]:
        """Get node types that represent imports for JavaScript/TypeScript."""
        return [
            'import_statement',
            'import_declaration',
            'export_statement',
            'require_call',  # CommonJS
        ]

    def _is_comment_node(self, node) -> bool:
        """Check if node is a comment."""
        return node.type in ['comment', 'line_comment', 'block_comment']

    def _extract_dependencies_from_imports(self, import_nodes) -> List[str]:
        """Extract JavaScript/TypeScript import dependencies."""
        deps = set()

        for node in import_nodes:
            text = node.text.decode('utf8')

            # ES6 imports: import ... from 'module'
            if 'from' in text:
                # Extract module name between quotes
                matches = re.findall(r"from\s+['\"]([^'\"]+)['\"]", text)
                for module in matches:
                    if not module.startswith('.'):
                        # External module - take base package name
                        deps.add(module.split('/')[0])

            # CommonJS: require('module')
            if 'require(' in text:
                requires = re.findall(r"require\(['\"]([^'\"]+)['\"]\)", text)
                for req in requires:
                    if not req.startswith('.'):
                        logger.info("[UNTESTED PATH] typescript CommonJS require dependency")
                        deps.add(req.split('/')[0])

        return sorted(list(deps))
