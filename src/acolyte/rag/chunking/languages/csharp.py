"""
C# chunker with enhanced pattern-based parsing.
Provides rich metadata extraction despite lack of tree-sitter support.

Features:
- Classes, interfaces, structs, records extraction
- Method and property detection with modifiers
- Async/await pattern detection
- LINQ usage detection
- Attribute extraction
- Generic type handling
- Comprehensive metadata
"""

from typing import Dict, List, Optional, Any, Set, Tuple
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


class CSharpChunker(
    BaseChunker, ComplexityMixin, TodoExtractionMixin, PatternDetectionMixin, SecurityAnalysisMixin
):
    """
    C#-specific chunker using advanced pattern matching.

    Since tree-sitter-languages doesn't support C#, we use sophisticated
    regex patterns to identify code structures and extract rich metadata.

    Handles:
    - Modern C# features (records, pattern matching, nullable reference types)
    - .NET attributes and metadata
    - Async/await patterns
    - LINQ expressions
    - Properties with getters/setters
    - Events and delegates
    - Extension methods
    - Partial classes
    """

    def __init__(self) -> None:
        """Initialize with C# language configuration."""
        super().__init__()
        self._init_patterns()
        logger.info("CSharpChunker: Using enhanced pattern-based chunking")

    def _get_language_name(self) -> str:
        """Return language identifier."""
        return 'csharp'

    def _get_tree_sitter_language(self) -> Any:
        """C# is not supported by tree-sitter-languages."""
        logger.info("[UNTESTED PATH] csharp._get_tree_sitter_language called")
        return None

    def _get_import_node_types(self) -> List[str]:
        """Not used for pattern-based chunking."""
        logger.info("[UNTESTED PATH] csharp._get_import_node_types called")
        return []

    def _is_comment_node(self, node: Any) -> bool:
        """Not used for pattern-based chunking."""
        logger.info("[UNTESTED PATH] csharp._is_comment_node called")
        return False

    def _init_patterns(self) -> None:
        """Initialize regex patterns for C# constructs."""
        # Using directives
        self.using_pattern = re.compile(
            r'^\s*(global\s+)?using\s+(static\s+)?([^;]+);', re.MULTILINE
        )

        # Namespace
        self.namespace_pattern = re.compile(r'^\s*namespace\s+([\w.]+)\s*(\{|;)?', re.MULTILINE)

        # Class/Interface/Struct/Record
        self.type_pattern = re.compile(
            r'^\s*(?P<mods>(?:public|private|protected|internal|abstract|sealed|static|partial)\s+)*'
            r'(class|interface|struct|record|enum)\s+(\w+)'
            r'(<[^>]+>)?'  # Generic parameters
            r'(\s*:\s*[^{]+)?'  # Inheritance
            r'\s*(\{|where)',
            re.MULTILINE,
        )

        # Methods (including constructors)
        self.method_pattern = re.compile(
            r'^\s*(?P<mods>(?:public|private|protected|internal|static|virtual|override|abstract|async|partial)\s+)*'
            r'([\w<>\[\]?]+\s+)?'  # Return type (optional for constructors)
            r'(\w+)\s*'  # Method name
            r'(<[^>]+>)?\s*'  # Generic parameters
            r'\([^)]*\)\s*'  # Parameters
            r'(where[^{]+)?'  # Generic constraints
            r'(\{|=>|;)',  # Method body or expression body or abstract
            re.MULTILINE,
        )

        # Properties
        self.property_pattern = re.compile(
            r'^\s*(?P<mods>(?:public|private|protected|internal|static|virtual|override|abstract)\s+)*'
            r'([\w<>\[\]?]+)\s+'  # Type
            r'(\w+)\s*'  # Property name
            r'(\{[^}]*\})',  # Property accessors
            re.MULTILINE | re.DOTALL,
        )

        # Attributes
        self.attribute_pattern = re.compile(r'\[([^\]]+)\]', re.MULTILINE)

        # Async patterns
        self.async_pattern = re.compile(r'\b(async|await)\b')

        # LINQ patterns
        self.linq_pattern = re.compile(
            r'\b(from|where|select|orderby|group|join|let|Where|Select|OrderBy|GroupBy|Join)\b'
        )

        # Event patterns
        self.event_pattern = re.compile(
            r'^\s*(?P<mods>(?:public|private|protected|internal|static|virtual)\s+)*event\s+([\w<>?]+)\s+(\w+)',
            re.MULTILINE,
        )

    async def chunk(self, content: str, file_path: str) -> List[Chunk]:
        """
        Chunk C# content using pattern matching with rich metadata.

        Strategy:
        1. Extract using directives
        2. Extract namespace and top-level types
        3. Extract members (methods, properties, etc.)
        4. Enrich with metadata
        """
        chunks = []
        lines = content.split('\n')

        # Track what's been processed
        processed_lines: Set[int] = set()

        # Extract using directives
        using_chunk = self._extract_using_directives(content, lines, file_path)
        if using_chunk:
            chunks.append(using_chunk)
            for i in range(using_chunk.metadata.start_line - 1, using_chunk.metadata.end_line):
                processed_lines.add(i)

        # Extract types (classes, interfaces, etc.)
        type_chunks = self._extract_types(content, lines, file_path, processed_lines)
        chunks.extend(type_chunks)

        # Extract any remaining significant code
        # Create a dummy processed_ranges for compatibility with parent class
        processed_ranges: Dict[str, Set[Tuple[int, int]]] = {}
        for line in processed_lines:
            if 'processed' not in processed_ranges:
                processed_ranges['processed'] = set()
            processed_ranges['processed'].add((line, line))

        remaining_chunks = self._extract_remaining_code(None, lines, file_path, processed_ranges)
        chunks.extend(remaining_chunks)

        # Sort by start line
        chunks.sort(key=lambda c: c.metadata.start_line)

        # Validate and add smart overlap
        chunks = self._validate_chunks(chunks)
        chunks = self._add_smart_overlap(chunks, preserve_imports=True)

        return chunks

    def _extract_using_directives(
        self, content: str, lines: List[str], file_path: str
    ) -> Optional[Chunk]:
        """Extract and group all using directives."""
        using_matches = list(self.using_pattern.finditer(content))
        if not using_matches:
            return None

        # Find the range of using directives
        first_match = using_matches[0]
        last_match = using_matches[-1]

        start_line = content[: first_match.start()].count('\n')
        end_line = content[: last_match.end()].count('\n')

        # Include any leading comments
        while start_line > 0 and (
            lines[start_line - 1].strip().startswith('//')
            or lines[start_line - 1].strip().startswith('/*')
            or not lines[start_line - 1].strip()
        ):
            start_line -= 1

        chunk_content = '\n'.join(lines[start_line : end_line + 1])

        # Extract dependencies
        dependencies = []
        for match in using_matches:
            is_static = bool(match.group(2))
            namespace = match.group(3).strip()

            if not is_static and '=' not in namespace:  # Skip aliases
                # Add all namespace parts, not just the first
                parts = namespace.split('.')
                dependencies.append(parts[0])
                # Also add second-level namespaces for common patterns
                if len(parts) > 1:
                    dependencies.append(f"{parts[0]}.{parts[1]}")

        chunk = self._create_chunk(
            content=chunk_content,
            chunk_type=ChunkType.IMPORTS,
            file_path=file_path,
            start_line=start_line + 1,
            end_line=end_line + 1,
            name='imports',
        )

        # Add metadata
        chunk.metadata.language_specific = {
            'dependencies': list(set(dependencies)),
            'global_usings': sum(1 for m in using_matches if m.group(1)),
            'static_usings': sum(1 for m in using_matches if m.group(2)),
            'total_usings': len(using_matches),
        }

        return chunk

    def _extract_types(
        self, content: str, lines: List[str], file_path: str, processed_lines: Set[int]
    ) -> List[Chunk]:
        """Extract classes, interfaces, structs, records, and enums."""
        chunks = []

        for match in self.type_pattern.finditer(content):
            start_pos = match.start()
            start_line = content[:start_pos].count('\n')

            # Skip if already processed
            if start_line in processed_lines:
                continue

            # Extract type information
            modifiers = (match.group('mods') or '').strip().split()
            type_kind = match.group(2)
            type_name = match.group(3)
            generics = match.group(4) or ''
            inheritance = match.group(5) or ''

            # Find the end of the type definition
            end_line = self._find_block_end(lines, start_line)

            # Mark lines as processed
            for i in range(start_line, end_line + 1):
                processed_lines.add(i)

            # Create chunk - include all lines of the type
            chunk_content = '\n'.join(lines[start_line : end_line + 1])

            # Determine chunk type
            chunk_type_map = {
                'class': ChunkType.CLASS,
                'interface': ChunkType.INTERFACE,
                'struct': ChunkType.CLASS,
                'record': ChunkType.CLASS,
                'enum': ChunkType.TYPES,
            }
            chunk_type = chunk_type_map.get(type_kind, ChunkType.UNKNOWN)

            chunk = self._create_chunk(
                content=chunk_content,
                chunk_type=chunk_type,
                file_path=file_path,
                start_line=start_line + 1,
                end_line=end_line + 1,
                name=type_name,
            )

            # Extract rich metadata
            chunk.metadata.language_specific = self._extract_type_metadata(
                chunk_content, type_kind, type_name, modifiers, generics, inheritance
            )

            chunks.append(chunk)

        return chunks

    def _extract_type_metadata(
        self,
        content: str,
        type_kind: str,
        type_name: str,
        modifiers: List[str],
        generics: str,
        inheritance: str,
    ) -> Dict[str, Any]:
        """Extract comprehensive metadata for a type."""
        metadata: Dict[str, Any] = {
            'type_kind': type_kind,
            'type_name': type_name,
            'modifiers': modifiers,
            'is_public': 'public' in modifiers,
            'is_abstract': 'abstract' in modifiers,
            'is_sealed': 'sealed' in modifiers,
            'is_static': 'static' in modifiers,
            'is_partial': 'partial' in modifiers,
            'generics': self._parse_generics(generics),
            'base_types': self._parse_inheritance(inheritance),
            'attributes': self._extract_attributes(content),
            'methods': [],
            'properties': [],
            'events': [],
            'complexity': self._calculate_complexity_from_content(content),
            'patterns': {
                'uses_async': bool(self.async_pattern.search(content)),
                'uses_linq': bool(self.linq_pattern.search(content)),
                'uses_generics': bool(generics),
                'uses_attributes': bool(self.attribute_pattern.search(content)),
            },
            'todos': self._extract_todos_from_content(content),
            'security': [],
        }

        # Extract methods
        for method_match in self.method_pattern.finditer(content):
            method_mods = (method_match.group('mods') or '').strip()
            method_info = {
                'name': method_match.group(3),
                'modifiers': method_mods.split(),
                'return_type': method_match.group(2) or 'void',
                'is_async': 'async' in method_mods,
                'is_abstract': ';' in method_match.group(6),
                'is_expression_body': '=>' in method_match.group(6),
            }
            methods_list = metadata.get('methods', [])
            if isinstance(methods_list, list):
                methods_list.append(method_info)

        # Extract properties
        for prop_match in self.property_pattern.finditer(content):
            prop_info = {
                'name': prop_match.group(3),
                'type': prop_match.group(2),
                'modifiers': (prop_match.group('mods') or '').strip().split(),
                'accessors': self._parse_property_accessors(prop_match.group(4)),
            }
            properties_list = metadata.get('properties', [])
            if isinstance(properties_list, list):
                properties_list.append(prop_info)

        # Extract events - simplified pattern
        event_lines = [line for line in content.split('\n') if 'event ' in line]
        for line in event_lines:
            event_match = re.search(r'event\s+([^\s]+)\s+(\w+)', line)
            if event_match:
                event_info: Dict[str, Any] = {
                    'name': event_match.group(2),
                    'type': event_match.group(1),
                    'modifiers': [],
                }
                # Extract modifiers
                mod_match = re.search(
                    r'^\s*((?:public|private|protected|internal|static|virtual)(?:\s+(?:public|private|protected|internal|static|virtual))*)',
                    line,
                )
                if mod_match:
                    event_info['modifiers'] = mod_match.group(1).split()
                events_list = metadata.get('events', [])
                if isinstance(events_list, list):
                    events_list.append(event_info)

        # Analyze patterns
        patterns_dict = metadata.get('patterns', {})
        if isinstance(patterns_dict, dict):
            patterns_dict.update(self._analyze_csharp_patterns(content, metadata))

        # Security analysis
        metadata['security'] = self._analyze_security_issues(content)

        return metadata

    def _parse_generics(self, generics_str: str) -> List[str]:
        """Parse generic type parameters."""
        if not generics_str:
            return []

        # Remove < and >
        generics_str = generics_str.strip('<>')

        # Simple split (doesn't handle nested generics perfectly)
        return [param.strip() for param in generics_str.split(',')]

    def _parse_inheritance(self, inheritance_str: str) -> List[str]:
        """Parse base types and interfaces."""
        if not inheritance_str:
            return []

        # Remove leading colon
        inheritance_str = inheritance_str.strip().lstrip(':').strip()

        # Split by comma
        return [base.strip() for base in inheritance_str.split(',')]

    def _extract_attributes(self, content: str) -> List[Dict[str, str]]:
        """Extract C# attributes."""
        attributes = []

        for match in self.attribute_pattern.finditer(content):
            attr_content = match.group(1)
            # Simple parsing - doesn't handle all cases
            parts = attr_content.split('(', 1)
            attr_name = parts[0].strip()
            attr_params = parts[1].rstrip(')') if len(parts) > 1 else ''

            attributes.append({'name': attr_name, 'parameters': attr_params})

        return attributes

    def _parse_property_accessors(self, accessors_str: str) -> Dict[str, Any]:
        """Parse property getter/setter."""
        accessors = {
            'has_getter': 'get' in accessors_str,
            'has_setter': 'set' in accessors_str,
            'has_init': 'init' in accessors_str,  # C# 9.0+
            'getter_visibility': 'public',
            'setter_visibility': 'public',
        }

        # Check for private/protected accessors
        if 'private get' in accessors_str:
            accessors['getter_visibility'] = 'private'
        elif 'protected get' in accessors_str:
            accessors['getter_visibility'] = 'protected'

        if 'private set' in accessors_str:
            accessors['setter_visibility'] = 'private'
        elif 'protected set' in accessors_str:
            accessors['setter_visibility'] = 'protected'

        return accessors

    def _analyze_csharp_patterns(
        self, content: str, metadata: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Analyze C#-specific patterns and practices."""
        patterns: Dict[str, List[str]] = {'good': [], 'anti': [], 'modern': []}

        # Good patterns
        if metadata['is_sealed'] and not metadata['is_abstract']:
            patterns['good'].append('sealed_class')

        if any(attr['name'] == 'Obsolete' for attr in metadata['attributes']):
            patterns['good'].append('obsolete_marked')

        if metadata['patterns']['uses_async']:
            patterns['good'].append('async_await')

        # Check for dependency injection pattern
        type_name = metadata.get('type_name', '')
        base_types = metadata.get('base_types', [])
        if type_name and len(type_name) > 1 and isinstance(base_types, list):
            interface_name = 'I' + type_name[1:]
            if interface_name in base_types:
                logger.info("[UNTESTED PATH] csharp interface implementation pattern")
                patterns['good'].append('interface_implementation')

        # Anti-patterns
        if len(metadata['methods']) > 20:
            patterns['anti'].append('god_class')

        if metadata['is_static'] and len(metadata['methods']) > 10:
            logger.info("[UNTESTED PATH] csharp static god class anti-pattern")
            patterns['anti'].append('static_god_class')

        # Modern C# features
        if re.search(r'\b(record)\s+\w+', content):
            logger.info("[UNTESTED PATH] csharp records pattern")
            patterns['modern'].append('records')

        if 'init' in content:
            patterns['modern'].append('init_only_setters')

        if '??' in content or '?.' in content:
            logger.info("[UNTESTED PATH] csharp null conditional pattern")
            patterns['modern'].append('null_conditional')

        if 'switch' in content and '=>' in content:
            logger.info("[UNTESTED PATH] csharp switch expressions pattern")
            patterns['modern'].append('switch_expressions')

        if 'is not' in content or 'and' in content or 'or' in content:
            logger.info("[UNTESTED PATH] csharp pattern matching")
            patterns['modern'].append('pattern_matching')

        return patterns

    def _analyze_security_issues(self, content: str) -> List[Dict[str, Any]]:
        """Analyze potential security issues in C# code."""
        issues = []

        # SQL injection risk - check for string interpolation or concatenation with SQL
        sql_patterns = [
            r'\$".*(?:SELECT|INSERT|UPDATE|DELETE|DROP).*\{',  # String interpolation
            r'".*(?:SELECT|INSERT|UPDATE|DELETE|DROP).*"\s*\+',  # String concatenation
            r'string\.Format\(.*(?:SELECT|INSERT|UPDATE|DELETE)',  # String.Format
        ]

        for pattern in sql_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append(
                    {
                        'type': 'sql_injection_risk',
                        'severity': 'high',
                        'description': 'Use parameterized queries instead of string concatenation',
                    }
                )
                break

        # Hardcoded credentials
        if re.search(r'(password|pwd|secret|key)\s*=\s*"[^"]+"', content, re.IGNORECASE):
            issues.append(
                {
                    'type': 'hardcoded_credentials',
                    'severity': 'critical',
                    'description': 'Store credentials in secure configuration',
                }
            )

        # Unsafe deserialization
        if 'BinaryFormatter' in content or 'NetDataContractSerializer' in content:
            logger.info("[UNTESTED PATH] csharp unsafe deserialization")
            issues.append(
                {
                    'type': 'unsafe_deserialization',
                    'severity': 'high',
                    'description': 'These serializers are vulnerable to deserialization attacks',
                }
            )

        return issues

    def _find_block_end(self, lines: List[str], start_line: int) -> int:
        """Find the end of a code block by counting braces."""
        brace_count = 0
        found_opening = False

        for i in range(start_line, len(lines)):
            line = lines[i]

            # Count braces
            open_braces = line.count('{')
            close_braces = line.count('}')

            if open_braces > 0:
                found_opening = True

            brace_count += open_braces - close_braces

            # End of block
            if found_opening and brace_count == 0:
                return i

        # If no matching brace found, return end of file
        return len(lines) - 1

    def _extract_remaining_code(
        self,
        root_node: Any,
        lines: List[str],
        file_path: str,
        processed_ranges: Dict[str, Set[Tuple[int, int]]],
    ) -> List[Chunk]:
        """Extract any remaining significant code blocks."""
        # Convert processed_ranges to processed_lines for backward compatibility
        processed_lines: Set[int] = set()
        for ranges in processed_ranges.values():
            for start, end in ranges:
                for line in range(start, end + 1):
                    processed_lines.add(line)

        chunks = []
        current_chunk_lines: List[str] = []
        current_start: Optional[int] = None

        for i, line_content in enumerate(lines):
            if i not in processed_lines and line_content.strip():
                if current_start is None:
                    current_start = i
                current_chunk_lines.append(line_content)
            else:
                # End of unprocessed block
                if current_start is not None and len(current_chunk_lines) >= 1:  # Minimum size
                    chunk = self._create_chunk(
                        content='\n'.join(current_chunk_lines),
                        chunk_type=ChunkType.MODULE,
                        file_path=file_path,
                        start_line=current_start + 1,
                        end_line=i,
                        name='module_code',
                    )
                    chunks.append(chunk)

                current_chunk_lines = []
                current_start = None

        # Handle final chunk
        if current_start is not None and len(current_chunk_lines) >= 1:
            chunk = self._create_chunk(
                content='\n'.join(current_chunk_lines),
                chunk_type=ChunkType.MODULE,
                file_path=file_path,
                start_line=current_start + 1,
                end_line=len(lines),
                name='module_code',
            )
            chunks.append(chunk)

        return chunks

    def _calculate_complexity_from_content(self, content: str) -> Dict[str, int]:
        """Calculate complexity metrics from content."""
        lines = content.split('\n')

        complexity = {
            'cyclomatic': 1,  # Base complexity
            'lines_of_code': len(
                [line for line in lines if line.strip() and not line.strip().startswith('//')]
            ),
            'max_nesting': 0,
            'parameters': 0,
        }

        # Count decision points for cyclomatic complexity
        decision_keywords = [
            'if',
            'else',
            'case',
            'for',
            'foreach',
            'while',
            'catch',
            'switch',
            'when',
        ]
        for keyword in decision_keywords:
            complexity['cyclomatic'] += len(re.findall(r'\b' + keyword + r'\b', content))

        # Count logical operators
        complexity['cyclomatic'] += content.count('||')
        complexity['cyclomatic'] += content.count('&&')

        # Count LINQ operators as they add complexity
        linq_operators = [
            '.Where(',
            '.Select(',
            '.OrderBy(',
            '.GroupBy(',
            '.Join(',
            '.Any(',
            '.All(',
        ]
        for op in linq_operators:
            complexity['cyclomatic'] += content.count(op)

        # Count ternary and null-coalescing operators
        # Don't count ? in nullable types or generics
        ternary_count = len(re.findall(r'[^<>?]\?[^<>?]', content))
        null_coalescing_count = content.count('??')
        complexity['cyclomatic'] += ternary_count + null_coalescing_count

        # Count lambda expressions
        complexity['cyclomatic'] += content.count('=>')

        # Estimate nesting
        current_nesting = 0
        for line in lines:
            current_nesting += line.count('{')
            current_nesting -= line.count('}')
            complexity['max_nesting'] = max(complexity['max_nesting'], current_nesting)

        if complexity['max_nesting'] > 10:
            logger.info("[UNTESTED PATH] csharp high nesting complexity")

        return complexity

    def _extract_todos_from_content(self, content: str) -> List[str]:
        """Extract TODO/FIXME comments."""
        todos = []
        todo_pattern = re.compile(
            r'//\s*(TODO|FIXME|HACK|NOTE|XXX|BUG)[:\s](.+)$', re.MULTILINE | re.IGNORECASE
        )

        for match in todo_pattern.finditer(content):
            todos.append(f"{match.group(1)}: {match.group(2).strip()}")

        return todos
