"""
Swift chunker with enhanced pattern-based parsing.
Provides rich metadata extraction despite lack of tree-sitter support.

Features:
- Modern Swift 5+ syntax support
- Classes, structs, enums, protocols, actors
- Property wrappers and result builders
- Async/await and structured concurrency
- SwiftUI view detection
- Extension and protocol conformance tracking
- Generics and associated types
- Comprehensive metadata
"""

from typing import Dict, List, Optional, Any
import re

from acolyte.models.chunk import Chunk, ChunkType
from acolyte.core.logging import logger
from acolyte.rag.chunking.base import BaseChunker
from acolyte.rag.chunking.mixins import (
    ComplexityMixin,
    TodoExtractionMixin,
    PatternDetectionMixin,
    SecurityAnalysisMixin,
)


class SwiftChunker(
    BaseChunker, ComplexityMixin, TodoExtractionMixin, PatternDetectionMixin, SecurityAnalysisMixin
):
    """
    Swift-specific chunker using advanced pattern matching.

    Since tree-sitter-languages doesn't support Swift, we use sophisticated
    regex patterns to identify code structures and extract rich metadata.

    Handles:
    - Modern Swift features (async/await, actors, property wrappers)
    - SwiftUI declarative syntax
    - Protocol-oriented programming patterns
    - Complex type system with generics
    - Access control and modifiers
    - Objective-C interoperability
    """

    def __init__(self):
        """Initialize with Swift language configuration."""
        super().__init__()
        self._init_patterns()
        logger.info("SwiftChunker: Using enhanced pattern-based chunking")

    def _get_language_name(self) -> str:
        """Return language identifier."""
        return 'swift'

    def _get_tree_sitter_language(self) -> Any:
        """Swift is not supported by tree-sitter-languages."""
        logger.info("[UNTESTED PATH] swift._get_tree_sitter_language called")
        return None

    def _get_import_node_types(self) -> List[str]:
        """Not used for pattern-based chunking."""
        logger.info("[UNTESTED PATH] swift._get_import_node_types called")
        return []

    def _is_comment_node(self, node) -> bool:
        """Not used for pattern-based chunking."""
        logger.info("[UNTESTED PATH] swift._is_comment_node called")
        return False

    def _init_patterns(self):
        """Initialize regex patterns for Swift constructs."""
        # Access modifiers
        self.access_modifiers = r'(?:open|public|internal|fileprivate|private)'
        self.declaration_modifiers = r'(?:static|class|final|lazy|weak|unowned|mutating|nonmutating|override|convenience|required|indirect)'

        # Import statements
        self.import_pattern = re.compile(
            r'^\s*import\s+(?:(class|struct|enum|protocol|typealias|func|let|var)\s+)?(\w+(?:\.\w+)*)',
            re.MULTILINE,
        )

        # Type declarations - ALL patterns now have consistent groups: (attributes)(access)(modifiers)name(generics)(inheritance)(where)
        self.class_pattern = re.compile(
            r'^\s*(?:@\w+(?:\([^)]*\))?\s+)*'  # Optional attributes
            rf'({self.access_modifiers}\s+)?((?:{self.declaration_modifiers}\s+)*)'
            r'class\s+(\w+)'
            r'(?:<([^>]+)>)?'  # Generic parameters
            r'(?:\s*:\s*([^{]+))?'  # Inheritance
            r'\s*(?:where\s+([^{]+))?\s*\{',
            re.MULTILINE,
        )

        self.struct_pattern = re.compile(
            r'^\s*(?:@\w+(?:\([^)]*\))?\s+)*'  # Optional attributes
            rf'({self.access_modifiers}\s+)?((?:{self.declaration_modifiers}\s+)*)'
            r'struct\s+(\w+)'
            r'(?:<([^>]+)>)?'
            r'(?:\s*:\s*([^{]+))?'
            r'\s*(?:where\s+([^{]+))?\s*\{',
            re.MULTILINE,
        )

        self.enum_pattern = re.compile(
            r'^\s*(?:@\w+(?:\([^)]*\))?\s+)*'  # Optional attributes
            rf'({self.access_modifiers}\s+)?((?:{self.declaration_modifiers}\s+)*)'
            r'enum\s+(\w+)'
            r'(?:<([^>]+)>)?'
            r'(?:\s*:\s*([^{]+))?'
            r'\s*(?:where\s+([^{]+))?\s*\{',
            re.MULTILINE,
        )

        self.protocol_pattern = re.compile(
            rf'^\s*({self.access_modifiers}\s+)?((?:{self.declaration_modifiers}\s+)*)'
            r'protocol\s+(\w+)'
            r'(?:<([^>]+)>)?'
            r'(?:\s*:\s*([^{]+))?'
            r'\s*(?:where\s+([^{]+))?\s*\{',
            re.MULTILINE,
        )

        self.actor_pattern = re.compile(
            rf'^\s*({self.access_modifiers}\s+)?((?:{self.declaration_modifiers}\s+)*)'
            r'actor\s+(\w+)'
            r'(?:<([^>]+)>)?'
            r'(?:\s*:\s*([^{]+))?'
            r'\s*(?:where\s+([^{]+))?\s*\{',
            re.MULTILINE,
        )

        # Extension pattern
        self.extension_pattern = re.compile(
            rf'^\s*({self.access_modifiers}\s+)?'
            r'extension\s+(\w+(?:\.\w+)*)'
            r'(?:<([^>]+)>)?'
            r'(?:\s*:\s*([^{]+))?'
            r'\s*(?:where\s+([^{]+))?\s*\{',
            re.MULTILINE,
        )

        # Function/method pattern - Fixed to properly capture ** operator
        self.function_pattern = re.compile(
            rf'^\s*(?:@\w+(?:\([^)]*\))?\s+)*'  # Attributes
            rf'({self.access_modifiers}\s+)?'
            rf'((?:{self.declaration_modifiers}\s+)*)'
            r'func\s+([\w\+\-\/=<>!&|~?%^*]+)'  # Name or operator (including **)
            r'(?:<([^>]+)>)?'  # Generic parameters
            r'\s*\('  # Opening paren
            r'([\s\S]*?)'  # Parameters (match everything including newlines)
            r'\)'  # Parameters
            r'(?:\s*async)?'
            r'(?:\s*throws)?'
            r'(?:\s*rethrows)?'
            r'(?:\s*->\s*([^{]+))?'  # Return type
            r'\s*(?:where\s+([^{]+))?'
            r'\s*(?:\{|$)',  # Opening brace or end of line (protocol methods)
            re.MULTILINE | re.DOTALL,
        )

        # Initializer pattern - Fixed to better capture throws
        self.init_pattern = re.compile(
            rf'^\s*({self.access_modifiers}\s+)?'
            rf'((?:{self.declaration_modifiers}\s+)*)'
            r'init(?:\?|!)?'  # init, init?, init!
            r'(?:<([^>]+)>)?'
            r'\s*\(([^)]*)\)'
            r'(?:\s*async)?'
            r'(?:\s*throws)?'
            r'(?:\s*rethrows)?'
            r'\s*(?:where\s+([^{]+))?'
            r'[\s\S]*?'  # Match any whitespace/keywords between ) and {
            r'\{',  # Opening brace
            re.MULTILINE | re.DOTALL,
        )

        # Property patterns
        self.property_pattern = re.compile(
            rf'^\s*(?:@\w+(?:\([^)]*\))?\s+)*'  # Property wrappers
            rf'({self.access_modifiers}\s+)?'
            rf'((?:{self.declaration_modifiers}\s+)*)'
            r'(?:var|let)\s+(\w+)(?:\s*:\s*([^=\n{]+))?'  # Type is optional for properties with initializers
            r'(?:\s*=\s*([^{\n]+))?'  # Default value
            r'(?:\s*\{[^}]+\})?',  # Computed property
            re.MULTILINE,
        )

        # SwiftUI view pattern
        self.swiftui_view_pattern = re.compile(
            r'struct\s+(\w+)\s*:\s*(?:[^{]*\b)?View\b', re.MULTILINE
        )

        # Property wrapper pattern
        self.property_wrapper_pattern = re.compile(
            r'@propertyWrapper\s+(?:public\s+)?(?:struct|class|enum)\s+(\w+)', re.MULTILINE
        )

        # Result builder pattern
        self.result_builder_pattern = re.compile(
            r'@resultBuilder\s+(?:public\s+)?(?:struct|class|enum)\s+(\w+)', re.MULTILINE
        )

        # Async/await patterns
        self.async_patterns = {
            'async_func': re.compile(r'\basync\s+(?:throws\s+)?(?:->|{)'),
            'await_call': re.compile(r'\bawait\s+\w+'),
            'task_group': re.compile(r'\bwithTaskGroup\b|\bwithThrowingTaskGroup\b'),
            'async_let': re.compile(r'\basync\s+let\s+\w+'),
        }

        # Type alias and associated type
        self.typealias_pattern = re.compile(
            rf'^\s*({self.access_modifiers}\s+)?typealias\s+(\w+)\s*=\s*(.+)$', re.MULTILINE
        )

        self.associatedtype_pattern = re.compile(
            r'^\s*associatedtype\s+(\w+)(?:\s*:\s*([^=\n]+))?(?:\s*=\s*(.+))?$', re.MULTILINE
        )

    async def chunk(self, content: str, file_path: str) -> List[Chunk]:
        """
        Chunk Swift content using pattern matching with rich metadata.

        Strategy:
        1. Extract imports
        2. Extract type declarations (classes, structs, etc.)
        3. Extract extensions
        4. Extract global functions
        5. Enrich with metadata
        6. Detect property wrappers at top-level
        """
        chunks = []
        lines = content.split('\n')
        processed_lines = set()

        # Extract imports
        import_chunk = self._extract_imports_swift(content, lines, file_path)
        if import_chunk:
            chunks.append(import_chunk)
            for i in range(import_chunk.metadata.start_line - 1, import_chunk.metadata.end_line):
                processed_lines.add(i)

        # Extract type declarations
        type_chunks = self._extract_type_declarations(content, lines, file_path, processed_lines)
        chunks.extend(type_chunks)

        # Extract extensions
        extension_chunks = self._extract_extensions(content, lines, file_path, processed_lines)
        chunks.extend(extension_chunks)

        # Extract global functions
        global_func_chunks = self._extract_global_functions(
            content, lines, file_path, processed_lines
        )
        chunks.extend(global_func_chunks)

        # Property wrapper definitions are already handled by type declarations
        # No need for special handling as they'll be detected as struct/class/enum

        # Sort by start line
        chunks.sort(key=lambda c: c.metadata.start_line)

        # Validate and add smart overlap
        chunks = self._validate_chunks(chunks)
        chunks = self._add_smart_overlap(chunks, preserve_imports=True)

        return chunks

    def _extract_imports_swift(
        self, content: str, lines: List[str], file_path: str
    ) -> Optional[Chunk]:
        """Extract and group all import statements."""
        imports: List[Dict[str, Any]] = []
        import_lines = []

        for match in self.import_pattern.finditer(content):
            import_kind = match.group(1)  # class, struct, etc.
            module_name = match.group(2)
            line_num = content[: match.start()].count('\n')

            # Use cast to inform mypy about the types
            from typing import cast

            import_kind_str = cast(str, import_kind) if import_kind else ""
            module_name_str = cast(str, module_name)

            imports.append(
                {'kind': import_kind_str, 'module': module_name_str, 'line': line_num + 1}
            )
            import_lines.append(line_num)

        if not imports:
            return None

        # Find the range
        start_line = min(import_lines)
        end_line = max(import_lines)

        # Include leading comments
        while start_line > 0 and (
            lines[start_line - 1].strip().startswith('//')
            or lines[start_line - 1].strip().startswith('/*')
            or not lines[start_line - 1].strip()
        ):
            start_line -= 1

        chunk_content = '\n'.join(lines[start_line : end_line + 1])

        # Don't create empty chunks
        if not chunk_content.strip():
            return None

        chunk = self._create_chunk(
            content=chunk_content,
            chunk_type=ChunkType.IMPORTS,
            file_path=file_path,
            start_line=start_line + 1,
            end_line=end_line + 1,
            name='imports',
        )

        # Extract dependencies
        dependencies = list(set(imp['module'].split('.')[0] for imp in imports))

        chunk.metadata.language_specific = {
            'imports': imports,
            'dependencies': dependencies,
            'import_count': len(imports),
            'has_uikit': any('UIKit' in imp['module'] for imp in imports),
            'has_swiftui': any('SwiftUI' in imp['module'] for imp in imports),
            'has_combine': any('Combine' in imp['module'] for imp in imports),
            'frameworks': self._detect_frameworks(imports),
        }

        return chunk

    def _extract_type_declarations(
        self, content: str, lines: List[str], file_path: str, processed_lines: set
    ) -> List[Chunk]:
        """Extract all type declarations (class, struct, enum, protocol, actor)."""
        chunks = []

        type_patterns = [
            (self.class_pattern, 'class', ChunkType.CLASS),
            (self.struct_pattern, 'struct', ChunkType.CLASS),
            (self.enum_pattern, 'enum', ChunkType.CLASS),
            (self.protocol_pattern, 'protocol', ChunkType.INTERFACE),
            (self.actor_pattern, 'actor', ChunkType.CLASS),
        ]

        for pattern, type_kind, chunk_type in type_patterns:
            for match in pattern.finditer(content):
                # Calculate initial start line from match position
                match_start_line = content[: match.start()].count('\n')

                if match_start_line in processed_lines:
                    continue

                # Extract match groups - now all patterns have same structure
                access = match.group(1)
                modifiers = match.group(2) or ''
                name = match.group(3)
                generics = match.group(4)
                inheritance = match.group(5)
                where_clause = match.group(6)

                # Find the actual start line including attributes
                match_start = match.start()

                # Look for attributes before the type declaration
                # Split content into lines and find the line where the match starts
                all_lines = content.split('\n')
                match_line_num = content[:match_start].count('\n')

                # Search backwards from match line to find attributes
                actual_start_line = match_line_num
                for i in range(match_line_num - 1, -1, -1):
                    line = all_lines[i].strip()
                    # Continue including lines that are attributes or empty
                    if line.startswith('@') or line == '':
                        actual_start_line = i
                    else:
                        # Stop when we hit a non-attribute, non-empty line
                        break

                # Adjust start_line if attributes were found
                start_line = actual_start_line
                if start_line < match_line_num:
                    # Include doc comments before attributes if present
                    if start_line > 0 and all_lines[start_line - 1].strip().startswith('///'):
                        while start_line > 0 and all_lines[start_line - 1].strip().startswith(
                            '///'
                        ):
                            start_line -= 1

                # Find the end of the type
                end_line = self._find_block_end(lines, start_line)

                # Mark lines as processed
                for i in range(start_line, end_line + 1):
                    processed_lines.add(i)

                # Create chunk
                chunk_content = '\n'.join(lines[start_line : end_line + 1])

                if not chunk_content.strip():
                    continue

                chunk = self._create_chunk(
                    content=chunk_content,
                    chunk_type=chunk_type,
                    file_path=file_path,
                    start_line=start_line + 1,
                    end_line=end_line + 1,
                    name=name,
                )

                # Extract metadata
                chunk.metadata.language_specific = self._extract_type_metadata(
                    chunk_content,
                    type_kind,
                    name,
                    access,
                    modifiers,
                    generics,
                    inheritance,
                    where_clause,
                )

                chunks.append(chunk)

        return chunks

    def _extract_type_metadata(
        self,
        content: str,
        type_kind: str,
        name: str,
        access: str,
        modifiers: str,
        generics: str,
        inheritance: str,
        where_clause: str,
    ) -> Dict[str, Any]:
        """Extract comprehensive metadata for a type declaration."""
        metadata = {
            'type_kind': type_kind,
            'visibility': (access or 'internal').strip(),
            'modifiers': modifiers.split() if modifiers else [],
            'is_final': 'final' in modifiers,
            'is_public': ((access or '').strip() == 'public'),
            'is_open': (access or '').strip() == 'open',
            'generic_parameters': self._parse_generics(generics),
            'inheritance': self._parse_inheritance(inheritance),
            'where_clause': where_clause,
            'members': {
                'properties': [],
                'methods': [],
                'initializers': [],
                'nested_types': [],
                'type_aliases': [],
            },
            'attributes': self._extract_attributes(content),
            'complexity': self._calculate_complexity_from_content(content),
            'patterns': {
                'is_swiftui_view': False,
                'has_property_wrappers': False,
                'uses_async': False,
                'uses_combine': False,
                'is_codable': False,
                'design_patterns': [],
                'anti_patterns': [],
            },
            'todos': self._extract_todos_from_content(content),
            'security': self._detect_security_issues_in_content(content),
            'quality': {
                'has_documentation': False,
                'line_count': content.count('\n') + 1,
                'member_count': 0,
            },
        }

        # Check for generics
        if metadata['generic_parameters']:
            patterns_dict = metadata.get('patterns', {})
            if isinstance(patterns_dict, dict):
                patterns_dict['uses_generics'] = True

        # Check if it's a SwiftUI View
        inheritance_list = metadata.get('inheritance', [])
        if isinstance(inheritance_list, list) and 'View' in inheritance_list:
            patterns_dict = metadata.get('patterns', {})
            if isinstance(patterns_dict, dict):
                patterns_dict['is_swiftui_view'] = True

        # Check for Codable
        if isinstance(inheritance_list, list):
            if 'Codable' in inheritance_list or (
                'Encodable' in inheritance_list and 'Decodable' in inheritance_list
            ):
                patterns_dict = metadata.get('patterns', {})
                if isinstance(patterns_dict, dict):
                    patterns_dict['is_codable'] = True

        # Extract members
        self._extract_members(content, metadata)

        # Detect patterns
        self._detect_swift_patterns(content, metadata)

        # Check documentation
        if '///' in content or '/**' in content:
            quality_dict = metadata.get('quality', {})
            if isinstance(quality_dict, dict):
                quality_dict['has_documentation'] = True

        return metadata

    def _extract_members(self, content: str, metadata: Dict[str, Any]) -> None:
        """Extract type members (properties, methods, etc.)."""
        # Get members dict with explicit type
        members_dict = metadata.get('members', {})
        if not isinstance(members_dict, dict):
            return

        # Get properties list
        properties_list = members_dict.get('properties', [])
        if not isinstance(properties_list, list):
            return

        # Extract properties
        for match in self.property_pattern.finditer(content):
            prop_access = match.group(1)
            prop_modifiers = match.group(2) or ''
            prop_name = match.group(3)
            prop_type = (match.group(4) or '').strip()
            prop_default = match.group(5)

            prop_info = {
                'name': prop_name,
                'type': prop_type,
                'visibility': prop_access or 'internal',
                'modifiers': prop_modifiers.split() if prop_modifiers else [],
                'has_default': prop_default is not None,
                'is_computed': '{' in match.group(0) and '}' in match.group(0),
                'is_optional': '?' in prop_type,
                'property_wrappers': self._extract_property_wrappers(match.group(0)),
            }

            properties_list.append(prop_info)

            if prop_info['property_wrappers']:
                patterns_dict = metadata.get('patterns', {})
                if isinstance(patterns_dict, dict):
                    patterns_dict['has_property_wrappers'] = True

        # Get methods list
        methods_list = members_dict.get('methods', [])
        if isinstance(methods_list, list):
            # Extract methods
            for match in self.function_pattern.finditer(content):
                # For structs/classes, accept all functions found within the content
                method_access = match.group(1)
                method_modifiers = match.group(2) or ''
                method_name = match.group(3)
                method_generics = match.group(4)
                method_params = match.group(5)
                method_return = match.group(6)
                method_where = match.group(7)

                method_info = {
                    'name': method_name,
                    'visibility': method_access or 'internal',
                    'modifiers': method_modifiers.split() if method_modifiers else [],
                    'is_async': 'async' in match.group(0),
                    'is_throwing': 'throws' in match.group(0),
                    'is_mutating': 'mutating' in method_modifiers,
                    'is_static': 'static' in method_modifiers,
                    'is_class': 'class' in method_modifiers,
                    'is_override': 'override' in method_modifiers,
                    'parameters': self._parse_parameters(method_params),
                    'return_type': method_return.strip() if method_return else 'Void',
                    'generic_parameters': self._parse_generics(method_generics),
                    'where_clause': method_where,
                }

                methods_list.append(method_info)

                if method_info['is_async']:
                    patterns_dict = metadata.get('patterns', {})
                    if isinstance(patterns_dict, dict):
                        patterns_dict['uses_async'] = True

        # Get initializers list
        initializers_list = members_dict.get('initializers', [])
        if isinstance(initializers_list, list):
            # Extract initializers
            for match in self.init_pattern.finditer(content):
                init_access = match.group(1)
                init_modifiers = match.group(2) or ''
                init_generics = match.group(3)
                init_params = match.group(4)
                init_where = match.group(5)

                init_info = {
                    'visibility': init_access or 'internal',
                    'modifiers': init_modifiers.split() if init_modifiers else [],
                    'is_failable': '?' in match.group(0),
                    'is_throwing': 'throws' in match.group(0),
                    'is_async': 'async' in match.group(0),
                    'is_required': 'required' in init_modifiers,
                    'is_convenience': 'convenience' in init_modifiers,
                    'parameters': self._parse_parameters(init_params),
                    'generic_parameters': self._parse_generics(init_generics),
                    'where_clause': init_where,
                }

                initializers_list.append(init_info)

        # Count total members
        quality_dict = metadata.get('quality', {})
        if isinstance(quality_dict, dict):
            quality_dict['member_count'] = (
                len(properties_list) + len(methods_list) + len(initializers_list)
            )

    def _extract_extensions(
        self, content: str, lines: List[str], file_path: str, processed_lines: set
    ) -> List[Chunk]:
        """Extract extension declarations."""
        chunks = []

        for match in self.extension_pattern.finditer(content):
            start_line = content[: match.start()].count('\n')

            if start_line in processed_lines:
                continue

            access = match.group(1)
            extended_type = match.group(2)
            generics = match.group(3)
            conformances = match.group(4)
            where_clause = match.group(5)

            # Find the end
            end_line = self._find_block_end(lines, start_line)

            # Mark as processed
            for i in range(start_line, end_line + 1):
                processed_lines.add(i)

            chunk_content = '\n'.join(lines[start_line : end_line + 1])

            if not chunk_content.strip():
                continue

            chunk = self._create_chunk(
                content=chunk_content,
                chunk_type=ChunkType.CLASS,  # Extensions modify types
                file_path=file_path,
                start_line=start_line + 1,
                end_line=end_line + 1,
                name=f'extension_{extended_type}',
            )

            # Extract metadata
            chunk.metadata.language_specific = self._extract_extension_metadata(
                chunk_content, extended_type, access, generics, conformances, where_clause
            )
            # Extract metadata for extension

            chunks.append(chunk)

        return chunks

    def _extract_extension_metadata(
        self,
        content: str,
        extended_type: str,
        access: str,
        generics: str,
        conformances: str,
        where_clause: str,
    ) -> Dict[str, Any]:
        """Extract metadata for an extension."""
        metadata = {
            'type_kind': 'extension',
            'extended_type': extended_type,
            'visibility': access or 'internal',
            'generic_constraints': self._parse_generics(generics),
            'conformances': self._parse_conformances(conformances),
            'where_clause': where_clause,
            'members': {'properties': [], 'methods': [], 'initializers': []},
            'purpose': self._determine_extension_purpose(conformances, content),
            'todos': self._extract_todos_from_content(content),
        }

        # Extract members (reuse same logic)
        self._extract_members(content, metadata)

        return metadata

    def _extract_global_functions(
        self, content: str, lines: List[str], file_path: str, processed_lines: set
    ) -> List[Chunk]:
        """Extract global functions (not inside types)."""
        chunks = []

        for match in self.function_pattern.finditer(content):
            start_line = content[: match.start()].count('\n')

            if start_line in processed_lines:
                continue

            # Check if it's truly global (at root level)
            line_start = content.rfind('\n', 0, match.start()) + 1
            indent = len(content[line_start : match.start()])
            if indent > 0:  # Skip if indented
                continue

            # Extract function details
            access = match.group(1)
            modifiers = match.group(2) or ''
            name = match.group(3)
            generics = match.group(4)
            params = match.group(5)
            return_type = match.group(6)
            where_clause = match.group(7)

            # Find end
            end_line = self._find_function_end(lines, start_line)

            # Mark as processed
            for i in range(start_line, end_line + 1):
                processed_lines.add(i)

            chunk_content = '\n'.join(lines[start_line : end_line + 1])

            # Determine if it's a test
            chunk_type = ChunkType.TESTS if name.startswith('test') else ChunkType.FUNCTION

            if not chunk_content.strip():
                continue

            chunk = self._create_chunk(
                content=chunk_content,
                chunk_type=chunk_type,
                file_path=file_path,
                start_line=start_line + 1,
                end_line=end_line + 1,
                name=name,
            )

            # Extract metadata
            chunk.metadata.language_specific = self._extract_function_metadata(
                chunk_content, name, access, modifiers, generics, params, return_type, where_clause
            )

            chunks.append(chunk)

        return chunks

    def _extract_function_metadata(
        self,
        content: str,
        name: str,
        access: str,
        modifiers: str,
        generics: str,
        params: str,
        return_type: str,
        where_clause: str,
    ) -> Dict[str, Any]:
        """Extract metadata for a function."""
        metadata = {
            'visibility': (access or 'internal').strip(),
            'modifiers': modifiers.split() if modifiers else [],
            'is_async': 'async' in content.split('\n')[0],
            'is_throwing': 'throws' in content.split('\n')[0],
            'parameters': self._parse_parameters(params),
            'return_type': return_type.strip() if return_type else 'Void',
            'generic_parameters': self._parse_generics(generics),
            'where_clause': where_clause,
            'complexity': self._calculate_complexity_from_content(content),
            'calls': self._extract_function_calls(content),
            'patterns': {
                'uses_force_unwrap': len(re.findall(r'(?<=[\w\]\)\?])!(?![!=])', content)) > 2,
                'uses_guard': 'guard ' in content,
                'uses_defer': 'defer ' in content,
                'has_early_return': content.count('return') > 1,
                'async_patterns': self._detect_async_patterns(content),
            },
            'todos': self._extract_todos_from_content(content),
            'security': self._detect_security_issues_in_content(content),
        }

        return metadata

    def _find_block_end(self, lines: List[str], start_line: int) -> int:
        """Find the end of a code block by counting braces."""
        brace_count = 0
        found_opening = False
        in_string = False
        in_comment = False

        for i in range(start_line, len(lines)):
            line = lines[i]

            # Process character by character to handle strings and comments properly
            j = 0
            while j < len(line):
                # Handle multi-line comments
                if not in_string and j < len(line) - 1 and line[j : j + 2] == '/*':
                    in_comment = True
                    j += 2
                    continue
                elif in_comment and j < len(line) - 1 and line[j : j + 2] == '*/':
                    in_comment = False
                    j += 2
                    continue
                elif in_comment:
                    j += 1
                    continue

                # Handle single-line comments
                if not in_string and j < len(line) - 1 and line[j : j + 2] == '//':
                    break  # Rest of line is comment

                # Handle strings
                if line[j] in ['"', "'"] and (j == 0 or line[j - 1] != '\\'):
                    in_string = not in_string

                # Count braces only if not in string or comment
                if not in_string:
                    if line[j] == '{':
                        brace_count += 1
                        found_opening = True
                    elif line[j] == '}':
                        brace_count -= 1

                j += 1

            if found_opening and brace_count == 0:
                return i

        return len(lines) - 1

    def _find_function_end(self, lines: List[str], start_line: int) -> int:
        """Find the end of a function (similar to block but handles one-liners)."""
        first_line = lines[start_line]

        # Check for one-liner (expression body)
        if '{' not in first_line:
            # Look for the arrow and find the end of expression
            for i in range(start_line, min(start_line + 10, len(lines))):
                if not lines[i].rstrip().endswith('\\'):
                    return i

        # Otherwise use block end logic
        return self._find_block_end(lines, start_line)

    def _remove_strings_and_comments(self, line: str) -> str:
        """Remove string literals and comments from a line."""
        # Remove single-line comments
        if '//' in line:
            line = line[: line.index('//')]

        # Remove string literals (simplified)
        line = re.sub(r'"[^"]*"', '""', line)
        line = re.sub(r"'[^']*'", "''", line)

        return line

    def _parse_generics(self, generics_str: str) -> List[str]:
        """Parse generic type parameters."""
        if not generics_str:
            return []

        # Remove angle brackets
        generics_str = generics_str.strip('<>')

        # Split by comma (simplified - doesn't handle nested generics perfectly)
        return [param.strip() for param in generics_str.split(',')]

    def _parse_inheritance(self, inheritance_str: str) -> List[str]:
        """Parse inherited types and protocol conformances."""
        if not inheritance_str:
            return []

        # Split by comma
        return [item.strip() for item in inheritance_str.split(',')]

    def _parse_conformances(self, conformances_str: str) -> List[str]:
        """Parse protocol conformances."""
        return self._parse_inheritance(conformances_str)

    def _parse_parameters(self, params_str: str) -> List[Dict[str, Any]]:
        """Parse function parameters."""
        if not params_str.strip():
            return []

        params = []

        # Split by comma (simplified)
        param_parts = params_str.split(',')

        for param in param_parts:
            param = param.strip()
            if not param:
                continue

            # Parse parameter pattern
            # external_name internal_name: Type = default
            param_match = re.match(r'^(?:(\w+)\s+)?(\w+)\s*:\s*([^=]+)(?:\s*=\s*(.+))?$', param)

            if param_match:
                external_name = param_match.group(1)
                internal_name = param_match.group(2)
                param_type = param_match.group(3).strip()
                default_value = param_match.group(4)

                param_info = {
                    'external_name': external_name or internal_name,
                    'internal_name': internal_name,
                    'type': param_type,
                    'is_optional': '?' in param_type,
                    'is_inout': 'inout' in param_type,
                    'is_variadic': '...' in param_type,
                    'has_default': default_value is not None,
                }

                params.append(param_info)

        return params

    def _extract_attributes(self, content: str) -> List[str]:
        """Extract attributes (@propertyWrapper, @State, etc.)."""
        attributes = []

        # Pattern for attributes - handles camelCase and PascalCase
        # This will match @propertyWrapper, @State, @MainActor, etc.
        attr_pattern = re.compile(r'@([a-zA-Z_][a-zA-Z0-9_]*)', re.MULTILINE)

        for match in attr_pattern.finditer(content):
            attr_name = match.group(1)
            if attr_name not in attributes:
                attributes.append(attr_name)

        return attributes

    def _extract_property_wrappers(self, property_def: str) -> List[str]:
        """Extract property wrappers from a property definition."""
        wrappers = []

        # Look for @Something at the start
        wrapper_pattern = re.compile(r'@(\w+)(?:\([^)]*\))?')

        for match in wrapper_pattern.finditer(property_def):
            wrappers.append(match.group(1))

        return wrappers

    def _detect_frameworks(self, imports: List[Dict[str, str]]) -> List[str]:
        """Detect which frameworks are being used."""
        frameworks = set()

        for imp in imports:
            module = imp['module']

            # Apple frameworks
            if module in [
                'UIKit',
                'AppKit',
                'SwiftUI',
                'Combine',
                'CoreData',
                'CloudKit',
                'WidgetKit',
                'WatchKit',
            ]:
                frameworks.add(module)

            # Third-party framework patterns
            elif module.startswith('Alamofire'):
                frameworks.add('Alamofire')
            elif module.startswith('RxSwift'):
                frameworks.add('RxSwift')
            elif module.startswith('SnapKit'):
                frameworks.add('SnapKit')

        return list(frameworks)

    def _determine_extension_purpose(self, conformances: str, content: str) -> str:
        """Determine the purpose of an extension."""
        if not conformances:
            return 'additional_functionality'

        conformances_lower = conformances.lower()

        if 'codable' in conformances_lower or 'decodable' in conformances_lower:
            return 'serialization'
        elif 'equatable' in conformances_lower or 'hashable' in conformances_lower:
            return 'equality_and_hashing'
        elif 'comparable' in conformances_lower:
            return 'comparison'
        elif 'collection' in conformances_lower:
            return 'collection_conformance'
        elif 'view' in conformances:
            return 'swiftui_view_conformance'
        else:
            return 'protocol_conformance'

    def _extract_function_calls(self, content: str) -> List[str]:
        """Extract function calls from content."""
        calls = []

        # Pattern for function calls
        call_pattern = re.compile(r'\b(\w+)\s*\(')

        for match in call_pattern.finditer(content):
            func_name = match.group(1)
            # Skip keywords
            if func_name not in ['if', 'for', 'while', 'switch', 'guard', 'func', 'init']:
                calls.append(func_name)

        return list(set(calls))

    def _detect_async_patterns(self, content: str) -> List[str]:
        """Detect async/await patterns in content."""
        patterns = []

        for pattern_name, pattern in self.async_patterns.items():
            if pattern.search(content):
                patterns.append(pattern_name)

        # Additional patterns
        if 'Task {' in content or 'Task.detached' in content:
            patterns.append('task_creation')

        if 'MainActor' in content:
            patterns.append('main_actor')

        return patterns

    def _detect_swift_patterns(self, content: str, metadata: Dict[str, Any]) -> None:
        """Detect Swift-specific patterns and anti-patterns."""
        # Get the patterns dict with proper type
        patterns: Dict[str, Any] = metadata.get('patterns', {})

        # Ensure lists exist
        if 'design_patterns' not in patterns:
            patterns['design_patterns'] = []
        if 'anti_patterns' not in patterns:
            patterns['anti_patterns'] = []

        design_patterns: List[str] = patterns['design_patterns']
        anti_patterns: List[str] = patterns['anti_patterns']

        # Design patterns
        if '@Published' in content:
            design_patterns.append('observable_object')
            patterns['uses_combine'] = True

        if metadata['type_kind'] == 'protocol' and len(metadata['members']['methods']) > 0:
            if all(not m.get('has_implementation', True) for m in metadata['members']['methods']):
                design_patterns.append('protocol_oriented')

        if '@Singleton' in content or (
            'static let shared' in content and 'private init' in content
        ):
            design_patterns.append('singleton')

        # Anti-patterns
        force_unwrap_count = content.count('!')
        if force_unwrap_count > 5:
            anti_patterns.append('excessive_force_unwrapping')

        if metadata['complexity']['cyclomatic'] > 10:
            anti_patterns.append('high_complexity')

        if metadata['quality']['member_count'] > 20:
            anti_patterns.append('god_object')

        # Nested types check
        if content.count('class ') + content.count('struct ') + content.count('enum ') > 3:
            anti_patterns.append('excessive_nesting')

        # Ensure patterns dict is properly assigned back
        metadata['patterns'] = patterns

    def _calculate_complexity_from_content(self, content: str) -> Dict[str, int]:
        """Calculate complexity metrics from content."""
        lines = content.split('\n')

        complexity = {
            'cyclomatic': 1,  # Base complexity
            'lines_of_code': len(
                [line for line in lines if line.strip() and not line.strip().startswith('//')]
            ),
            'max_nesting': 0,
            'decision_points': 0,
        }

        # Count decision points
        decision_keywords = ['if', 'else', 'switch', 'for', 'while', 'guard', 'catch']
        for keyword in decision_keywords:
            complexity['decision_points'] += len(re.findall(r'\b' + keyword + r'\b', content))

        # Count ternary operators separately (? is a regex special character)
        complexity['decision_points'] += len(re.findall(r'\?', content))

        complexity['cyclomatic'] += complexity['decision_points']

        # Estimate nesting
        current_nesting = 0
        for line in lines:
            current_nesting += line.count('{')
            current_nesting -= line.count('}')
            complexity['max_nesting'] = max(complexity['max_nesting'], current_nesting)

        return complexity

    def _extract_todos_from_content(self, content: str) -> List[str]:
        """Extract TODO/FIXME comments."""
        todos = []

        # Single-line comment pattern
        todo_pattern = re.compile(
            r'//\s*(TODO|FIXME|HACK|NOTE|XXX|BUG)[:\s](.+)$', re.MULTILINE | re.IGNORECASE
        )

        for match in todo_pattern.finditer(content):
            todos.append(f"{match.group(1)}: {match.group(2).strip()}")

        # Multi-line comment pattern
        multiline_pattern = re.compile(
            r'/\*.*?(TODO|FIXME|HACK|NOTE|XXX|BUG)[:\s](.+?)\*/', re.DOTALL | re.IGNORECASE
        )

        for match in multiline_pattern.finditer(content):
            todos.append(f"{match.group(1)}: {match.group(2).strip()}")

        return todos

    def _detect_security_issues_in_content(self, content: str) -> List[Dict[str, Any]]:
        """Detect potential security issues."""
        issues = []

        # Force unwrapping of optionals (can cause crashes)
        force_unwrap_pattern = r'(?<=[\w\]\)\?])!(?![!=])'
        force_unwraps = re.findall(force_unwrap_pattern, content)
        if len(force_unwraps) > 3:
            issues.append(
                {
                    'type': 'excessive_force_unwrapping',
                    'severity': 'medium',
                    'count': len(force_unwraps),
                    'description': 'Excessive force unwrapping can cause runtime crashes',
                }
            )

        # Unsafe string interpolation
        if 'String(format:' in content:
            issues.append(
                {
                    'type': 'format_string_vulnerability',
                    'severity': 'medium',
                    'description': 'String format can be vulnerable to format string attacks',
                }
            )

        # Fixed: Hardcoded credentials detection
        # Pattern 1: let/var SOMETHING_KEY = "value" (case insensitive)
        if re.search(
            r'(let|var)\s+\w*(KEY|TOKEN|SECRET|PASSWORD|API_KEY)\w*\s*=\s*"[^"]+"',
            content,
            re.IGNORECASE,
        ):
            issues.append(
                {
                    'type': 'hardcoded_credentials',
                    'severity': 'critical',
                    'description': 'Hardcoded credentials detected',
                }
            )

        # SQL Injection: detect dynamic SQL queries using user input
        # Look for SELECT statements that are concatenated or interpolated with variables (potentially user input)
        sql_select_pattern = re.compile(r'SELECT\s+.+FROM\s+\w+.*([+]|\\\(|\{).*', re.IGNORECASE)
        if sql_select_pattern.search(content):
            # Heuristic: also check for common user input variable names
            if any(
                var in content
                for var in ['userInput', 'input', 'query', 'search', 'request', 'param', 'args']
            ):
                issues.append(
                    {
                        'type': 'sql_injection',
                        'severity': 'critical',
                        'description': 'Possible SQL injection vulnerability: dynamic SQL query constructed with user input',
                    }
                )

        # Unsafe URL construction
        if 'URL(string:' in content and '+' in content:
            issues.append(
                {
                    'type': 'unsafe_url_construction',
                    'severity': 'low',
                    'description': 'URL construction with string concatenation can be unsafe',
                }
            )

        return issues
