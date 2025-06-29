"""
Perl chunker using tree-sitter-languages.
Handles Perl's unique syntax and constructs.
"""

import re
from typing import Dict, List, Any, Optional
from tree_sitter_languages import get_language  # type: ignore

from acolyte.models.chunk import ChunkType, Chunk, ChunkMetadata
from ..base import BaseChunker


class PerlChunker(BaseChunker):
    """
    Perl-specific chunker using tree-sitter.

    Handles Perl's rich syntax including:
    - Subroutines (named and anonymous)
    - Packages and modules
    - POD documentation
    - Special variables and filehandles
    """

    def _get_language_name(self) -> str:
        """Return language identifier."""
        return 'perl'

    def _get_import_node_types(self) -> List[str]:
        """Get node types that represent imports for perl."""
        return ['use_statement', 'require_statement', 'use_no_statement']

    def _is_comment_node(self, node) -> bool:
        """Check if node is a comment."""
        return node.type in ['comment', 'line_comment', 'block_comment']

    def _get_tree_sitter_language(self) -> Any:
        """Get Perl language for tree-sitter."""
        return get_language('perl')

    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """
        Perl-specific node types to chunk.

        Tree-sitter Perl node types mapped to ChunkTypes.
        """
        return {
            # Subroutines
            'subroutine_declaration_statement': ChunkType.FUNCTION,
            'anonymous_subroutine_expression': ChunkType.FUNCTION,
            'method_declaration': ChunkType.METHOD,
            # Packages/Modules
            'package_statement': ChunkType.MODULE,
            # Imports
            'use_statement': ChunkType.IMPORTS,
            'require_statement': ChunkType.IMPORTS,
            'use_no_statement': ChunkType.IMPORTS,
            # Constants
            'use_constant_declaration': ChunkType.CONSTANTS,
            # POD documentation
            'pod': ChunkType.DOCSTRING,
            # Special blocks
            'phaser_statement': ChunkType.MODULE,  # BEGIN, END, etc.
        }

    async def chunk(self, content: str, file_path: str) -> List[Chunk]:
        """Override chunk method to use pattern-based extraction for Perl."""
        lines = content.splitlines(keepends=True)
        chunks = []

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Look for POD documentation blocks
            if line.startswith('=') and not line == '=cut':
                pod_start = i + 1
                j = i + 1
                # Find the end of POD (=cut or end of file)
                while j < len(lines):
                    if lines[j].strip() == '=cut':
                        j += 1  # Include =cut
                        break
                    j += 1

                # Only create chunk if we have content
                if j > i:
                    content_chunk = ''.join(lines[i:j])
                    chunk = Chunk(
                        content=content_chunk,
                        metadata=ChunkMetadata(
                            chunk_type=ChunkType.DOCSTRING,
                            start_line=pod_start,
                            end_line=j,
                            language="perl",
                            file_path=file_path,
                            name="pod_documentation",
                            language_specific={},
                        ),
                    )
                    chunks.append(chunk)
                    i = j - 1

            # Look for use constant declarations (before other use statements)
            elif line.startswith('use constant'):
                start_line = i + 1
                # Check if it's multi-line with hash
                if '{' in line:
                    # Multi-line constant declaration
                    j = i
                    brace_count = 0
                    while j < len(lines):
                        brace_count += lines[j].count('{') - lines[j].count('}')
                        j += 1
                        if brace_count == 0:
                            break
                else:
                    # Single line constant
                    j = i + 1

                content_chunk = ''.join(lines[i:j])
                chunk = Chunk(
                    content=content_chunk,
                    metadata=ChunkMetadata(
                        chunk_type=ChunkType.CONSTANTS,
                        start_line=start_line,
                        end_line=j,
                        language="perl",
                        file_path=file_path,
                        name="constants",
                        language_specific={},
                    ),
                )
                chunks.append(chunk)
                i = j - 1

            # Look for anonymous subroutines (my $var = sub { ... })
            elif 'sub {' in line or '= sub' in line:
                if self._is_anonymous_sub_assignment(line, lines[i + 1 : i + 3]):
                    start_line = i + 1
                    # Find the end of the anonymous sub
                    brace_count = 0
                    j = i
                    found_opening = False

                    while j < len(lines):
                        for char in lines[j]:
                            if char == '{':
                                brace_count += 1
                                found_opening = True
                            elif char == '}':
                                brace_count -= 1

                        if found_opening and brace_count == 0:
                            # Include the line with closing brace and semicolon
                            j += 1
                            # Check if semicolon is on next line
                            if j < len(lines) and lines[j].strip().startswith(';'):
                                j += 1

                            content_chunk = ''.join(lines[i:j])
                            chunk = Chunk(
                                content=content_chunk,
                                metadata=ChunkMetadata(
                                    chunk_type=ChunkType.FUNCTION,
                                    start_line=start_line,
                                    end_line=j,
                                    language="perl",
                                    file_path=file_path,
                                    name="anonymous_sub",
                                    language_specific=self._extract_subroutine_metadata_from_text(
                                        content_chunk
                                    ),
                                ),
                            )
                            chunks.append(chunk)
                            i = j - 1
                            break
                        j += 1

            # Look for subroutine declarations
            elif re.match(r'^sub\s+\w+', line):
                sub_match = re.match(r'^sub\s+(\w+)', line)
                if sub_match:
                    sub_name = sub_match.group(1)
                    start_line = i + 1

                    # Find the end of the subroutine
                    brace_count = 0
                    j = i
                    found_opening = False

                    while j < len(lines):
                        for char in lines[j]:
                            if char == '{':
                                brace_count += 1
                                found_opening = True
                            elif char == '}':
                                brace_count -= 1

                        if found_opening and brace_count == 0:
                            # Found the end
                            end_line = j + 1
                            content_chunk = ''.join(lines[i : j + 1])

                            # Determine chunk type
                            chunk_type = ChunkType.FUNCTION
                            if sub_name == 'new':
                                chunk_type = ChunkType.CONSTRUCTOR
                            elif sub_name.startswith('_'):
                                chunk_type = ChunkType.METHOD

                            # Create chunk
                            chunk = Chunk(
                                content=content_chunk,
                                metadata=ChunkMetadata(
                                    chunk_type=chunk_type,
                                    start_line=start_line,
                                    end_line=end_line,
                                    language="perl",
                                    file_path=file_path,
                                    name=sub_name,
                                    language_specific=self._extract_subroutine_metadata_from_text(
                                        content_chunk
                                    ),
                                ),
                            )
                            chunks.append(chunk)
                            i = j
                            break
                        j += 1
                    else:
                        # Couldn't find proper end, take next 10 lines
                        end_line = min(i + 10, len(lines))
                        content_chunk = ''.join(lines[i:end_line])

                        chunk_type = ChunkType.FUNCTION
                        if sub_name == 'new':
                            chunk_type = ChunkType.CONSTRUCTOR
                        elif sub_name.startswith('_'):
                            chunk_type = ChunkType.METHOD

                        chunk = Chunk(
                            content=content_chunk,
                            metadata=ChunkMetadata(
                                chunk_type=chunk_type,
                                start_line=start_line,
                                end_line=end_line,
                                language="perl",
                                file_path=file_path,
                                name=sub_name,
                                language_specific=self._extract_subroutine_metadata_from_text(
                                    content_chunk
                                ),
                            ),
                        )
                        chunks.append(chunk)
                        i = end_line - 1

            # Look for package declarations
            elif line.startswith('package '):
                pkg_match = re.match(r'^package\s+([\w:]+)', line)
                if pkg_match:
                    pkg_name = pkg_match.group(1)
                    start_line = i + 1

                    # Find next package or significant boundary
                    j = i + 1
                    # Opcional: límite configurable (None = sin límite)
                    PACKAGE_LINE_LIMIT = None  # Cambia a un entero si quieres un límite
                    while j < len(lines):
                        next_line = lines[j].strip()
                        # Stop at next package, sub, or POD
                        if (
                            next_line.startswith('package ')
                            or re.match(r'^sub\s+\w+', next_line)
                            or next_line.startswith('=')
                        ):
                            break
                        if PACKAGE_LINE_LIMIT is not None and (j - i > PACKAGE_LINE_LIMIT):
                            break
                        j += 1

                    end_line = j
                    content_chunk = ''.join(lines[i:j])

                    chunk = Chunk(
                        content=content_chunk,
                        metadata=ChunkMetadata(
                            chunk_type=ChunkType.MODULE,
                            start_line=start_line,
                            end_line=end_line,
                            language="perl",
                            file_path=file_path,
                            name=pkg_name,
                            language_specific=self._extract_package_metadata_from_text(
                                content_chunk, lines, i
                            ),
                        ),
                    )
                    chunks.append(chunk)
                    i = j - 1

            # Look for use/require statements (but not use constant)
            elif (line.startswith('use ') or line.startswith('require ')) and not line.startswith(
                'use constant'
            ):
                # Group imports together
                import_start = i + 1
                j = i
                import_lines = []

                while j < len(lines):
                    curr_line = lines[j].strip()
                    if (
                        curr_line.startswith('use ') or curr_line.startswith('require ')
                    ) and not curr_line.startswith('use constant'):
                        import_lines.append(lines[j])
                        j += 1
                    elif curr_line == '' or curr_line.startswith('#'):
                        # Include empty lines and comments between imports
                        import_lines.append(lines[j])
                        j += 1
                    else:
                        break

                if import_lines:
                    content_chunk = ''.join(import_lines)
                    chunk = Chunk(
                        content=content_chunk,
                        metadata=ChunkMetadata(
                            chunk_type=ChunkType.IMPORTS,
                            start_line=import_start,
                            end_line=j,
                            language="perl",
                            file_path=file_path,
                            name="imports",
                            language_specific={},
                        ),
                    )
                    chunks.append(chunk)
                    i = j - 1

            i += 1

        # If no chunks found, create a single module chunk
        if not chunks:
            chunk = Chunk(
                content=content,
                metadata=ChunkMetadata(
                    chunk_type=ChunkType.MODULE,
                    start_line=1,
                    end_line=len(lines),
                    language="perl",
                    file_path=file_path,
                    name=file_path.split('/')[-1].replace('.pl', ''),
                    language_specific={},
                ),
            )
            chunks.append(chunk)

        return chunks

    def _extract_subroutine_metadata_from_text(self, sub_text: str) -> Dict[str, Any]:
        """Extract Perl-specific subroutine metadata from text."""
        metadata = {
            'parameters': [],
            'has_prototype': False,
            'prototype': None,
            'is_anonymous': ('= sub' in sub_text or '=sub' in sub_text)
            and not re.match(r'^\s*sub\s+\w+', sub_text.strip()),
            'attributes': [],
            'uses_shift': False,
            'uses_at_underscore': False,
        }

        # Check for prototype - look for pattern like sub name (prototype)
        proto_match = re.search(r'sub\s+\w+\s*\(([^)]*)\)', sub_text)
        if proto_match:
            metadata['has_prototype'] = True
            metadata['prototype'] = f"({proto_match.group(1)})"

        # Check for attributes like :lvalue, :method
        attr_match = re.findall(r':(\w+)', sub_text)
        if attr_match:
            metadata['attributes'] = [f":{attr}" for attr in attr_match]

        # Check for parameter extraction patterns
        if 'shift' in sub_text:
            metadata['uses_shift'] = True
        if '@_' in sub_text:
            metadata['uses_at_underscore'] = True

        # Try to extract parameter names
        # Look for my ($param1, $param2) = @_; pattern
        param_match = re.search(r'my\s*\(([^)]+)\)\s*=\s*@_', sub_text)
        if param_match:
            params = param_match.group(1)
            # Extract variable names
            var_names = re.findall(r'\$(\w+)', params)
            metadata['parameters'] = var_names
        # Also check for shift patterns
        elif 'shift' in sub_text:
            shift_vars = re.findall(r'my\s+\$(\w+)\s*=\s*shift', sub_text)
            metadata['parameters'] = shift_vars

        return metadata

    def _extract_package_metadata_from_text(
        self, pkg_text: str, lines: List[str], pkg_line_idx: int
    ) -> Dict[str, Any]:
        """Extract Perl-specific package metadata from text."""
        metadata: Dict[str, Any] = {
            'version': None,
            'parent_classes': [],
            'exports': [],
            'imports': [],
        }

        # Check for $VERSION assignment
        version_match = re.search(r'\$VERSION\s*=\s*[\'"]?([\d\.]+)[\'"]?', pkg_text)
        if version_match:
            metadata['version'] = version_match.group(1)

        # Check for @ISA
        isa_matches = re.findall(
            r'@ISA\s*=\s*\([\'"]?(\w+(?:::\w+)*)[\'"]?(?:,\s*[\'"]?(\w+(?:::\w+)*)[\'"]?)*\)',
            pkg_text,
        )
        for match in isa_matches:
            for cls in match:
                if cls:
                    metadata['parent_classes'].append(cls)

        # Also check simple @ISA patterns
        simple_isa = re.findall(r'[\'"](\w+(?:::\w+)*)[\'"]', pkg_text)
        if '@ISA' in pkg_text and simple_isa:
            for cls in simple_isa:
                if cls not in metadata['parent_classes']:
                    metadata['parent_classes'].append(cls)

        # Check for use base/parent
        base_match = re.findall(r'use\s+(?:base|parent)\s+[\'"]?(\w+(?:::\w+)*)[\'"]?', pkg_text)
        metadata['parent_classes'].extend(base_match)

        # Remove duplicates
        metadata['parent_classes'] = list(set(metadata['parent_classes']))

        return metadata

    def _extract_subroutine_metadata(self, sub_node) -> Dict[str, Any]:
        """Extract Perl-specific subroutine metadata."""
        return self._extract_subroutine_metadata_from_text(sub_node.text.decode('utf8'))

    def _extract_package_metadata(self, package_node, lines: List[str]) -> Dict[str, Any]:
        """Extract Perl-specific package metadata."""
        # Get the line number of the package
        pkg_line = package_node.start_point[0]
        return self._extract_package_metadata_from_text(
            package_node.text.decode('utf8'), lines, pkg_line
        )

    def _extract_version_from_line(self, line: str) -> Optional[str]:
        """Extract version string from a line."""
        # Look for version patterns
        match = re.search(r'\$VERSION\s*=\s*[\'"]?([\d\.]+)[\'"]?', line)
        return match.group(1) if match else None

    def _extract_isa_from_line(self, line: str) -> List[str]:
        """Extract parent classes from @ISA assignment line."""
        # Extract quoted strings
        classes = re.findall(r'[\'"](\w+(?:::\w+)*)[\'"]', line)
        return classes

    def _extract_version(self, node) -> Optional[str]:
        """Extract version string from assignment."""
        text = node.text.decode('utf8')
        # Look for version patterns
        match = re.search(r'[\'"]?([\d\.]+)[\'"]?', text)
        return match.group(1) if match else None

    def _extract_isa(self, node) -> List[str]:
        """Extract parent classes from @ISA assignment."""
        text = node.text.decode('utf8')
        # Extract quoted strings
        classes = re.findall(r'[\'"](\w+(?:::\w+)*)[\'"]', text)
        return classes

    def _extract_base_classes(self, text: str) -> List[str]:
        """Extract base classes from use base/parent statement."""
        # Match quoted module names
        classes = re.findall(r'[\'"](\w+(?:::\w+)*)[\'"]', text)
        if not classes:
            # Try qw() syntax
            qw_match = re.search(r'qw[/\(\[](.*?)[/\)\]]', text)
            if qw_match:
                classes = qw_match.group(1).split()
        return classes

    def _extract_dependencies_from_imports(self, import_nodes) -> List[str]:
        """Extract Perl-specific import dependencies."""
        deps = set()

        for node in import_nodes:
            text = node.text.decode('utf8')

            # Skip pragmas
            if any(
                pragma in text
                for pragma in ['use strict', 'use warnings', 'use utf8', 'use feature']
            ):
                continue

            # Extract module name
            if text.startswith('use '):
                # use Module::Name
                match = re.search(r'use\s+(\w+(?:::\w+)*)', text)
                if match:
                    module = match.group(1)
                    # Skip version declarations and special modules
                    if module not in ['constant', 'base', 'parent', 'lib']:
                        # Convert :: to root module
                        deps.add(module.split('::')[0])

            elif text.startswith('require '):
                # require Module::Name or require "file.pl"
                match = re.search(r'require\s+[\'"]?(\w+(?:::\w+)*)', text)
                if match:
                    module = match.group(1)
                    deps.add(module.split('::')[0])

        return sorted(list(deps))

    def _extract_node_name(self, node) -> Optional[str]:
        """Extract name from Perl nodes."""
        node_text = node.text.decode('utf8')

        # For subroutines
        if 'sub' in node_text:
            # Get the subroutine text and extract name
            match = re.search(r'sub\s+(\w+)', node_text)
            if match:
                return match.group(1)

        # For packages
        elif node.type == 'package_statement' or 'package' in node_text:
            # Extract package name from text
            match = re.search(r'package\s+([\w:]+)', node_text)
            if match:
                return match.group(1)

        return super()._extract_node_name(node)

    @staticmethod
    def _is_anonymous_sub_assignment(line: str, next_lines: list) -> bool:
        """Detects if the current line (and possibly next lines) is an anonymous sub assignment."""
        import re

        # Compiled regex: matches inline or newline-separated anonymous sub assignment
        pattern = re.compile(r"^(?:my\s+)?\$\w+\s*=\s*sub\s*(?:\{|$)")
        if pattern.search(line):
            if line.rstrip().endswith('sub'):
                # Check if next non-empty line starts with '{'
                for next_line in next_lines:
                    if next_line.strip() == '':
                        continue
                    return next_line.strip().startswith('{')
                return False
            return True
        return False
