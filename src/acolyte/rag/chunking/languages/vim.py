"""
VimScript chunker with enhanced pattern-based parsing.
Provides rich metadata extraction despite lack of tree-sitter support.

Features:
- Function definitions with script-local and autoload detection
- Command and mapping extraction
- Autocommand groups and events
- Plugin declarations (Plug, Vundle, etc.)
- Configuration sections
- Folding markers
- Comprehensive metadata
"""

from typing import Dict, List, Optional, Any
import re

from acolyte.models.chunk import Chunk, ChunkType
from acolyte.core.logging import logger
from acolyte.rag.chunking.base import BaseChunker
from acolyte.rag.chunking.mixins import ComplexityMixin, TodoExtractionMixin, PatternDetectionMixin


class VimChunker(BaseChunker, ComplexityMixin, TodoExtractionMixin, PatternDetectionMixin):
    """
    VimScript-specific chunker using advanced pattern matching.

    Since tree-sitter-languages doesn't support VimScript, we use sophisticated
    regex patterns to identify code structures and extract rich metadata.

    Handles:
    - Function definitions (function!, autoload functions)
    - Commands and mappings
    - Autocommand groups
    - Plugin configurations (Plug, Vundle, NeoBundle)
    - Settings and options
    - Conditionals and script structure
    """

    def __init__(self):
        """Initialize with VimScript language configuration."""
        super().__init__()
        self._init_patterns()
        logger.info("VimChunker: Using enhanced pattern-based chunking")

    def _get_language_name(self) -> str:
        """Return language identifier."""
        return 'vimscript'

    def _get_tree_sitter_language(self) -> Any:
        """VimScript is not supported by tree-sitter-languages."""
        return None

    def _get_import_node_types(self) -> List[str]:
        """Not used for pattern-based chunking."""
        return []

    def _is_comment_node(self, node) -> bool:
        """Not used for pattern-based chunking."""
        logger.info("[UNTESTED PATH] vim._is_comment_node called")
        return False

    def _init_patterns(self):
        """Initialize regex patterns for VimScript constructs."""
        # Function definitions
        self.function_pattern = re.compile(
            r'^\s*function!?\s+([sg]:|<SID>)?([a-zA-Z_#][a-zA-Z0-9_#]*)\s*\((.*?)\)(\s+.*)?$',
            re.MULTILINE,
        )

        # Command definitions
        self.command_pattern = re.compile(
            r'^\s*command!?\s+(-\w+\s+)*([A-Z]\w*)\s+(.+)$', re.MULTILINE
        )

        # Autocommand groups
        self.augroup_pattern = re.compile(r'^\s*augroup\s+(\w+)\s*$', re.MULTILINE)

        # Autocommands
        self.autocmd_pattern = re.compile(
            r'^\s*(au|autocmd)!?\s+(\w+)?\s*([A-Za-z,*]+)\s+([^\s]+)\s+(.+)$', re.MULTILINE
        )

        # Mappings
        self.mapping_pattern = re.compile(
            r'^\s*([nvxsoilct]?)n?o?re?map!?\s+(<\w+>)*\s*(\S+)\s+(.+)$', re.MULTILINE
        )

        # Plugin declarations
        self.plugin_patterns = {
            'plug': re.compile(r"^\s*Plug\s+'([^']+)'", re.MULTILINE),
            'vundle': re.compile(r'^\s*Plugin\s+["\']([^"\']+)["\']', re.MULTILINE),
            'neobundle': re.compile(r'^\s*NeoBundle\s+["\']([^"\']+)["\']', re.MULTILINE),
            'packadd': re.compile(r'^\s*packadd!?\s+(\S+)', re.MULTILINE),
        }

        # Source/runtime commands
        self.source_pattern = re.compile(r'^\s*(so|source|ru|runtime)!?\s+(.+)$', re.MULTILINE)

        # Set options
        self.set_pattern = re.compile(r'^\s*set?\s+(no)?(\w+)([=:]\S*)?', re.MULTILINE)

        # Let assignments
        self.let_pattern = re.compile(
            r'^\s*let\s+([gblstwvaf]:)?(\w+)\s*([.+\-*/]?=)\s*(.+)$', re.MULTILINE
        )

        # If/endif blocks
        self.if_pattern = re.compile(r'^\s*if\s+(.+)$', re.MULTILINE)

        # Fold markers
        self.fold_pattern = re.compile(r'\{\{\{(\d*)|["}]\s*\}\}\}(\d*)', re.MULTILINE)

    async def chunk(self, content: str, file_path: str) -> List[Chunk]:
        """
        Chunk VimScript content using pattern matching with rich metadata.

        Strategy:
        1. Extract plugin declarations as imports
        2. Extract functions with full metadata
        3. Extract commands and mappings
        4. Extract autocommand groups
        5. Group configuration sections
        """
        chunks = []
        lines = content.split('\n')
        processed_lines = set()

        # Extract plugin declarations first
        plugin_chunk = self._extract_plugins(content, lines, file_path)
        if plugin_chunk:
            chunks.append(plugin_chunk)
            for i in range(plugin_chunk.metadata.start_line - 1, plugin_chunk.metadata.end_line):
                processed_lines.add(i)

        # Extract functions
        function_chunks = self._extract_functions(content, lines, file_path, processed_lines)
        chunks.extend(function_chunks)

        # Extract autocommand groups
        augroup_chunks = self._extract_augroups(content, lines, file_path, processed_lines)
        chunks.extend(augroup_chunks)

        # Extract commands
        command_chunks = self._extract_commands(content, lines, file_path, processed_lines)
        chunks.extend(command_chunks)

        # Extract configuration sections
        config_chunks = self._extract_config_sections(lines, file_path, processed_lines)
        chunks.extend(config_chunks)

        # Sort by start line
        chunks.sort(key=lambda c: c.metadata.start_line)

        # Validate and add smart overlap
        chunks = self._validate_chunks(chunks)
        chunks = self._add_smart_overlap(chunks, preserve_imports=True)

        return chunks

    def _extract_plugins(self, content: str, lines: List[str], file_path: str) -> Optional[Chunk]:
        """Extract and group all plugin declarations."""
        all_plugins = []
        plugin_lines = []

        # Check each pattern type
        for plugin_type, pattern in self.plugin_patterns.items():
            for match in pattern.finditer(content):
                plugin_name = match.group(1)
                all_plugins.append({'type': plugin_type, 'name': plugin_name})
                line_num = content[: match.start()].count('\n')
                plugin_lines.append(line_num)

        # Also check source commands for plugin loading
        for match in self.source_pattern.finditer(content):
            path = match.group(2)
            if 'plugin' in path.lower() or 'bundle' in path.lower():
                all_plugins.append({'type': 'source', 'name': path})
                line_num = content[: match.start()].count('\n')
                plugin_lines.append(line_num)

        if not plugin_lines:
            return None

        # Find the range
        start_line = min(plugin_lines)
        end_line = max(plugin_lines)

        # Expand to include nearby related lines
        while start_line > 0 and (
            lines[start_line - 1].strip().startswith('"')  # Comments
            or not lines[start_line - 1].strip()  # Empty lines
        ):
            start_line -= 1

        while end_line < len(lines) - 1 and (
            'Plug' in lines[end_line + 1]
            or 'Plugin' in lines[end_line + 1]
            or 'NeoBundle' in lines[end_line + 1]
            or lines[end_line + 1].strip().startswith('"')
        ):
            end_line += 1

        chunk_content = '\n'.join(lines[start_line : end_line + 1])

        chunk = self._create_chunk(
            content=chunk_content,
            chunk_type=ChunkType.IMPORTS,
            file_path=file_path,
            start_line=start_line + 1,
            end_line=end_line + 1,
            name='plugins',
        )

        # Extract unique plugin names for dependencies
        dependencies = list(set(p['name'].split('/')[-1].replace('.vim', '') for p in all_plugins))

        chunk.metadata.language_specific = {
            'plugins': all_plugins,
            'dependencies': dependencies,
            'plugin_managers': list(set(p['type'] for p in all_plugins)),
            'total_plugins': len(all_plugins),
        }

        return chunk

    def _extract_functions(
        self, content: str, lines: List[str], file_path: str, processed_lines: set
    ) -> List[Chunk]:
        """Extract function definitions with metadata."""
        chunks = []

        for match in self.function_pattern.finditer(content):
            start_pos = match.start()
            start_line = content[:start_pos].count('\n')

            if start_line in processed_lines:
                continue

            # Extract function info
            scope = match.group(1) or ''
            name = match.group(2)
            params = match.group(3) or ''
            flags = match.group(4) or ''

            # Find the end of the function
            end_line = self._find_function_end(lines, start_line)

            # Mark lines as processed
            for i in range(start_line, end_line + 1):
                processed_lines.add(i)

            # Create chunk
            chunk_content = '\n'.join(lines[start_line : end_line + 1])

            chunk = self._create_chunk(
                content=chunk_content,
                chunk_type=ChunkType.FUNCTION,
                file_path=file_path,
                start_line=start_line + 1,
                end_line=end_line + 1,
                name=name,
            )

            # Extract metadata
            chunk.metadata.language_specific = self._extract_function_metadata(
                chunk_content, scope, name, params, flags
            )

            chunks.append(chunk)

        return chunks

    def _extract_function_metadata(
        self, content: str, scope: str, name: str, params: str, flags: str
    ) -> Dict[str, Any]:
        """Extract comprehensive function metadata."""
        metadata = {
            'scope': scope,
            'is_script_local': scope in ['s:', '<SID>'],
            'is_global': scope == 'g:' or not scope,
            'is_autoload': '#' in name,
            'has_bang': 'function!' in content,
            'parameters': self._parse_parameters(params),
            'flags': {
                'abort': 'abort' in flags,
                'range': 'range' in flags,
                'dict': 'dict' in flags,
                'closure': 'closure' in flags,
            },
            'complexity': self._calculate_complexity_from_content(content),
            'patterns': {
                'uses_execute': 'execute' in content or 'exe' in content,
                'uses_eval': 'eval(' in content,
                'uses_normal': 'normal' in content,
                'has_try_catch': 'try' in content and 'catch' in content,
            },
            'todos': self._extract_todos_from_content(content),
            'calls': self._extract_function_calls(content),
            'variables': self._extract_variables(content),
        }

        # Detect autoload function details
        if metadata['is_autoload']:
            parts = name.split('#')
            metadata['autoload_plugin'] = parts[0] if parts else ''
            metadata['autoload_function'] = parts[-1] if parts else ''

        return metadata

    def _parse_parameters(self, params_str: str) -> List[Dict[str, Any]]:
        """Parse function parameters."""
        if not params_str.strip():
            return []

        params = []

        # Handle variadic functions
        if '...' in params_str:
            logger.info("[UNTESTED PATH] vim variadic function detected")
            params.append({'name': '...', 'type': 'variadic', 'optional': False})
        else:
            # Parse regular parameters
            for param in params_str.split(','):
                param = param.strip()
                if param:
                    param_info = {
                        'name': param.lstrip('a:'),  # Remove a: prefix if present
                        'type': 'regular',
                        'optional': '=' in param,  # Has default value
                    }
                    params.append(param_info)

        return params

    def _extract_function_calls(self, content: str) -> List[str]:
        """Extract function calls from content."""
        calls = []

        # Pattern for function calls
        call_pattern = re.compile(r'\b(call\s+)?([sg]:|<SID>)?([a-zA-Z_#][a-zA-Z0-9_#]*)\s*\(')

        for match in call_pattern.finditer(content):
            func_name = match.group(3)
            if func_name not in ['if', 'for', 'while', 'function']:  # Skip keywords
                logger.info("[UNTESTED PATH] vim function call detected")
                calls.append(func_name)

        return list(set(calls))

    def _extract_variables(self, content: str) -> Dict[str, List[str]]:
        """Extract variable definitions by scope."""
        variables: Dict[str, List[str]] = {
            'global': [],
            'script': [],
            'local': [],
            'window': [],
            'buffer': [],
            'tabpage': [],
        }

        for match in self.let_pattern.finditer(content):
            scope = match.group(1) or 'l:'  # Default to local
            var_name = match.group(2)

            scope_map = {
                'g:': 'global',
                's:': 'script',
                'l:': 'local',
                'w:': 'window',
                'b:': 'buffer',
                't:': 'tabpage',
            }

            scope_key = scope_map.get(scope, 'local')
            if var_name not in variables[scope_key]:
                variables[scope_key].append(var_name)

        # Clean up empty lists
        return {k: v for k, v in variables.items() if v}

    def _find_function_end(self, lines: List[str], start_line: int) -> int:
        """Find the endfunction line."""
        indent_level = len(lines[start_line]) - len(lines[start_line].lstrip())

        for i in range(start_line + 1, len(lines)):
            line = lines[i]

            # Check for endfunction at same or less indentation
            if re.match(r'^\s*endf(u|un|unc|unct|uncti|unctio|unction)?!?\s*$', line):
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= indent_level:
                    return i

        # If no endfunction found, assume it goes to end of file
        return len(lines) - 1

    def _extract_augroups(
        self, content: str, lines: List[str], file_path: str, processed_lines: set
    ) -> List[Chunk]:
        """Extract autocommand groups."""
        chunks = []

        for match in self.augroup_pattern.finditer(content):
            start_line = content[: match.start()].count('\n')

            if start_line in processed_lines:
                continue

            group_name = match.group(1)

            # Skip END marker
            if group_name.upper() == 'END':
                continue

            # Find the end of the augroup
            end_line = self._find_augroup_end(lines, start_line)
            logger.info("[UNTESTED PATH] vim augroup processing")

            # Mark lines as processed
            for i in range(start_line, end_line + 1):
                processed_lines.add(i)

            chunk_content = '\n'.join(lines[start_line : end_line + 1])

            chunk = self._create_chunk(
                content=chunk_content,
                chunk_type=ChunkType.MODULE,  # Augroups are like modules
                file_path=file_path,
                start_line=start_line + 1,
                end_line=end_line + 1,
                name=f'augroup_{group_name}',
            )

            # Extract metadata
            chunk.metadata.language_specific = self._extract_augroup_metadata(
                chunk_content, group_name
            )

            chunks.append(chunk)

        return chunks

    def _extract_augroup_metadata(self, content: str, group_name: str) -> Dict[str, Any]:
        """Extract autocommand group metadata."""
        metadata: Dict[str, Any] = {
            'group_name': group_name,
            'autocmds': [],
            'events': set(),
            'patterns': set(),
            'has_clear': 'au!' in content or 'autocmd!' in content,
        }

        # Extract individual autocmds
        for match in self.autocmd_pattern.finditer(content):
            events = match.group(3).split(',')
            pattern = match.group(4)
            command = match.group(5)

            autocmd_info = {
                'events': events,
                'pattern': pattern,
                'command': command.strip(),
            }

            metadata['autocmds'].append(autocmd_info)
            metadata['events'].update(events)
            metadata['patterns'].add(pattern)

        metadata['events'] = list(metadata['events'])
        metadata['patterns'] = list(metadata['patterns'])
        metadata['total_autocmds'] = len(metadata['autocmds'])

        return metadata

    def _find_augroup_end(self, lines: List[str], start_line: int) -> int:
        """Find the augroup END line."""
        for i in range(start_line + 1, len(lines)):
            if re.match(r'^\s*augroup\s+END\s*$', lines[i], re.IGNORECASE):
                return i

        # If no END found, look for next augroup
        for i in range(start_line + 1, len(lines)):
            if re.match(r'^\s*augroup\s+\w+\s*$', lines[i]):
                return i - 1

        return len(lines) - 1

    def _extract_commands(
        self, content: str, lines: List[str], file_path: str, processed_lines: set
    ) -> List[Chunk]:
        """Extract command definitions."""
        chunks = []

        for match in self.command_pattern.finditer(content):
            start_line = content[: match.start()].count('\n')

            if start_line in processed_lines:
                continue

            cmd_name = match.group(2)
            cmd_args = match.group(1) or ''
            cmd_definition = match.group(3)

            # Commands are usually single line, but check for line continuations
            end_line = start_line
            while end_line < len(lines) - 1 and lines[end_line].rstrip().endswith('\\'):
                logger.info("[UNTESTED PATH] vim command line continuation")
                end_line += 1

            # Mark as processed
            for i in range(start_line, end_line + 1):
                processed_lines.add(i)

            chunk_content = '\n'.join(lines[start_line : end_line + 1])

            chunk = self._create_chunk(
                content=chunk_content,
                chunk_type=ChunkType.FUNCTION,  # Commands are like functions
                file_path=file_path,
                start_line=start_line + 1,
                end_line=end_line + 1,
                name=f'command_{cmd_name}',
            )

            # Extract metadata
            chunk.metadata.language_specific = {
                'command_name': cmd_name,
                'arguments': self._parse_command_args(cmd_args),
                'definition': cmd_definition,
                'has_bang': 'command!' in chunk_content,
            }

            chunks.append(chunk)

        return chunks

    def _parse_command_args(self, args_str: str) -> Dict[str, Any]:
        """Parse command arguments like -nargs, -complete, etc."""
        args_info = {
            'nargs': '0',  # default
            'complete': None,
            'range': False,
            'count': False,
            'bang': False,
            'bar': False,
            'register': False,
            'buffer': False,
        }

        if not args_str:
            return args_info

        # Parse each -arg
        args = args_str.split()
        for arg in args:
            if arg.startswith('-nargs='):
                args_info['nargs'] = arg.split('=')[1]
            elif arg == '-nargs':
                args_info['nargs'] = '*'
            elif arg.startswith('-complete='):
                args_info['complete'] = arg.split('=')[1]
            elif arg == '-range':
                logger.info("[UNTESTED PATH] vim command -range argument")
                args_info['range'] = True
            elif arg == '-count':
                args_info['count'] = True
            elif arg == '-bang':
                args_info['bang'] = True
            elif arg == '-bar':
                args_info['bar'] = True
            elif arg == '-register':
                args_info['register'] = True
            elif arg == '-buffer':
                args_info['buffer'] = True

        return args_info

    def _extract_config_sections(
        self, lines: List[str], file_path: str, processed_lines: set
    ) -> List[Chunk]:
        """Extract configuration sections based on fold markers or comments."""
        chunks = []
        current_section = None
        section_start = None
        section_lines = []

        for i, line in enumerate(lines):
            if i in processed_lines:
                if current_section:
                    # End current section
                    if section_start is not None and len(section_lines) >= 5:
                        chunk = self._create_config_chunk(
                            section_lines, file_path, section_start, i - 1, current_section
                        )
                        chunks.append(chunk)
                    current_section = None
                    section_lines = []
                continue

            # Check for section markers
            section_match = self._is_section_marker(line)
            if section_match:
                # Save previous section if exists
                if current_section and section_start is not None and len(section_lines) >= 5:
                    chunk = self._create_config_chunk(
                        section_lines, file_path, section_start, i - 1, current_section
                    )
                    chunks.append(chunk)

                # Start new section
                current_section = section_match
                section_start = i
                section_lines = [line]
            elif current_section:
                section_lines.append(line)

        # Handle final section
        if current_section and section_start is not None and len(section_lines) >= 5:
            chunk = self._create_config_chunk(
                section_lines, file_path, section_start, len(lines) - 1, current_section
            )
            chunks.append(chunk)

        return chunks

    def _is_section_marker(self, line: str) -> Optional[str]:
        """Check if line marks a section start."""
        # Fold markers
        if '{{{' in line:
            # Extract section name from comment
            comment_match = re.search(r'"\s*(.+?)\s*{{{', line)
            if comment_match:
                return comment_match.group(1)

        # Section comments
        section_patterns = [
            r'^\s*"\s*=+\s*(.+?)\s*=+',  # "==== Section ===="
            r'^\s*"\s*-+\s*(.+?)\s*-+',  # "---- Section ----"
            r'^\s*"\s*\*+\s*(.+?)\s*\*+',  # "**** Section ****"
            r'^\s*"\s*#+\s*(.+)',  # "### Section"
        ]

        for pattern in section_patterns:
            match = re.match(pattern, line)
            if match:
                return match.group(1).strip()

        return None

    def _create_config_chunk(
        self, lines: List[str], file_path: str, start: int, end: int, section_name: str
    ) -> Chunk:
        """Create a configuration section chunk."""
        content = '\n'.join(lines)

        chunk = self._create_chunk(
            content=content,
            chunk_type=ChunkType.CONSTANTS,  # Config sections are like constants
            file_path=file_path,
            start_line=start + 1,
            end_line=end + 1,
            name=f'config_{section_name.lower().replace(" ", "_")}',
        )

        # Extract configuration metadata
        chunk.metadata.language_specific = self._extract_config_metadata(content, section_name)

        return chunk

    def _extract_config_metadata(self, content: str, section_name: str) -> Dict[str, Any]:
        """Extract metadata from configuration section."""
        metadata: Dict[str, Any] = {
            'section_name': section_name,
            'settings': [],
            'mappings': [],
            'variables': {},
            'options': [],
            'has_conditionals': False,
            'patterns': {
                'uses_has': 'has(' in content,
                'uses_exists': 'exists(' in content,
                'has_os_check': any(x in content for x in ['has("win', 'has("mac', 'has("unix']),
            },
        }

        # Extract set options
        for match in self.set_pattern.finditer(content):
            option = {
                'name': match.group(2),
                'negated': bool(match.group(1)),
                'value': match.group(3).lstrip('=:') if match.group(3) else None,
            }
            metadata['options'].append(option)

        # Extract mappings
        for match in self.mapping_pattern.finditer(content):
            mapping = {
                'mode': match.group(1) or 'n',  # Default to normal mode
                'lhs': match.group(3),
                'rhs': match.group(4),
            }
            metadata['mappings'].append(mapping)

        # Extract let assignments
        for match in self.let_pattern.finditer(content):
            scope = match.group(1) or 'l:'
            var_name = match.group(2)
            value = match.group(4)

            if scope not in metadata['variables']:
                metadata['variables'][scope] = []

            # Help mypy understand this is a list
            vars_list = metadata['variables'][scope]
            assert isinstance(vars_list, list)
            vars_list.append({'name': var_name, 'value': value.strip()})

        # Check for conditionals
        metadata['has_conditionals'] = bool(self.if_pattern.search(content))

        metadata['settings'] = len(metadata['options']) + len(metadata['mappings'])

        return metadata

    def _calculate_complexity_from_content(self, content: str) -> Dict[str, int]:
        """Calculate complexity metrics from content."""
        lines = content.split('\n')

        complexity = {
            'cyclomatic': 1,  # Base complexity
            'lines_of_code': len(
                [line for line in lines if line.strip() and not line.strip().startswith('"')]
            ),
            'max_nesting': 0,
            'conditionals': 0,
        }

        # Count decision points
        decision_patterns = [
            r'\bif\b',
            r'\belseif\b',
            r'\bwhile\b',
            r'\bfor\b',
            r'\btry\b',
            r'\bcatch\b',
            r'\?',
            r'\|\|',
            r'&&',
        ]

        for pattern in decision_patterns:
            complexity['cyclomatic'] += len(re.findall(pattern, content))

        # Count conditionals
        complexity['conditionals'] = len(self.if_pattern.findall(content))

        # Estimate nesting (simple approach)
        current_nesting = 0
        for line in lines:
            if re.match(r'^\s*(if|while|for|try|function)', line):
                current_nesting += 1
                complexity['max_nesting'] = max(complexity['max_nesting'], current_nesting)
            elif re.match(r'^\s*(endif|endwhile|endfor|endtry|endfunction)', line):
                logger.info("[UNTESTED PATH] vim complexity nesting decrease")
                current_nesting = max(0, current_nesting - 1)

        return complexity

    def _extract_todos_from_content(self, content: str) -> List[str]:
        """Extract TODO/FIXME comments."""
        todos = []
        todo_pattern = re.compile(
            r'"\s*(TODO|FIXME|HACK|NOTE|XXX|BUG)[:\s](.+)$', re.MULTILINE | re.IGNORECASE
        )

        for match in todo_pattern.finditer(content):
            todos.append(f"{match.group(1)}: {match.group(2).strip()}")

        return todos
