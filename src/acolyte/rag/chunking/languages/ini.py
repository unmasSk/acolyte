"""
INI chunker with enhanced pattern-based parsing.
Provides rich metadata extraction despite lack of tree-sitter support.

Features:
- Section-based chunking with hierarchy detection
- Security issue detection (passwords, tokens, keys)
- Environment variable and path reference tracking
- Duplicate key detection
- Quality metrics (comments, organization)
- Configuration pattern recognition
- Comprehensive metadata
"""

from typing import Dict, List, Any
import re
import os

from acolyte.models.chunk import Chunk, ChunkType
from acolyte.core.logging import logger
from acolyte.rag.chunking.languages.config_base import ConfigChunkerBase
from acolyte.rag.chunking.mixins import (
    TodoExtractionMixin,
    SecurityAnalysisMixin,
    PatternDetectionMixin,
)


class IniChunker(
    ConfigChunkerBase, TodoExtractionMixin, SecurityAnalysisMixin, PatternDetectionMixin
):
    """
    INI-specific chunker using advanced pattern matching.

    Since tree-sitter-languages doesn't support INI, we use sophisticated
    regex patterns to parse configuration files and extract rich metadata.

    Handles:
    - Standard INI format with [sections] and key=value pairs
    - Comments with # or ; prefixes
    - Multi-line values with indentation
    - Environment variable references
    - File path detection
    - Security sensitive data detection
    - Quality and organization analysis
    """

    def __init__(self):
        """Initialize with INI language configuration."""
        super().__init__()
        self._init_patterns()
        logger.info("IniChunker: Using enhanced pattern-based chunking")

    def _get_language_name(self) -> str:
        """Return language identifier."""
        return 'ini'

    def _get_tree_sitter_language(self) -> Any:
        """INI is not supported by tree-sitter-languages."""
        return None

    def _get_import_node_types(self) -> List[str]:
        """Not used for pattern-based chunking."""
        return []

    def _is_comment_node(self, node) -> bool:
        """Not used for pattern-based chunking."""
        return False

    def _init_patterns(self) -> None:
        """Initialize regex patterns for INI constructs."""
        # Section headers
        self.section_pattern = re.compile(r'^\s*\[([^\]]+)\]\s*$', re.MULTILINE)

        # Key-value pairs (various formats)
        self.setting_patterns = [
            # Standard: key = value
            re.compile(r'^\s*([^=;#\s][^=]*?)\s*=\s*(.*)$', re.MULTILINE),
            # Colon separator: key: value
            re.compile(r'^\s*([^:;#\s][^:]*?)\s*:\s*(.*)$', re.MULTILINE),
            # Space separator: key value (menos común)
            re.compile(r'^\s*([^;#\s]\S+)\s+([^;#].*)$', re.MULTILINE),
        ]

        # Comments
        self.comment_pattern = re.compile(r'^\s*[;#]\s*(.*)$', re.MULTILINE)

        # Multi-line value continuation
        self.continuation_pattern = re.compile(r'^\s+(.+)$', re.MULTILINE)

        # Environment variables
        self.env_var_patterns = [
            re.compile(r'\$\{?([A-Z_][A-Z0-9_]*)\}?'),  # Unix style
            re.compile(r'%([A-Z_][A-Z0-9_]*)%'),  # Windows style
        ]

        # File path patterns
        self.path_patterns = [
            re.compile(r'[A-Za-z]:\\[^;#\s]+'),  # Windows absolute
            re.compile(r'/[A-Za-z0-9_\-./]+'),  # Unix absolute
            re.compile(r'\./[A-Za-z0-9_\-./]+'),  # Relative with ./
            re.compile(r'[A-Za-z0-9_\-]+\.[A-Za-z]{2,4}$'),  # File with extension
        ]

        # Security sensitive patterns
        self.security_patterns = {
            'password': re.compile(
                r'(password|passwd|pwd|pass)\s*[=:]\s*([^;#\s]+)', re.IGNORECASE
            ),
            'token': re.compile(
                r'(token|api_key|apikey|auth_key|access_key|secret_key)\s*[=:]\s*([^;#\s]+)',
                re.IGNORECASE,
            ),
            'url_creds': re.compile(r'(https?|ftp|jdbc|mongodb)://[^:]+:([^@]+)@', re.IGNORECASE),
            'private_key': re.compile(r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----'),
            'connection_string': re.compile(
                r'(Server|Data Source|User ID|Password)=([^;]+)', re.IGNORECASE
            ),
        }

    async def chunk(self, content: str, file_path: str) -> List[Chunk]:
        """
        Chunk INI content using pattern matching with rich metadata.

        Strategy:
        1. For small files (<50 lines), keep as single chunk
        2. For larger files, chunk by section
        3. Extract comprehensive metadata at both file and section level
        """
        lines = content.split('\n')

        # For small INI files, analyze as single chunk
        # Reduced threshold to 50 lines for better chunking granularity
        if len(lines) < 50:
            return await self._chunk_as_single_file(content, lines, file_path)

        # For larger files, chunk by section
        return await self._chunk_by_sections(content, lines, file_path)

    async def _chunk_as_single_file(
        self, content: str, lines: List[str], file_path: str
    ) -> List[Chunk]:
        """Create a single chunk for the entire INI file."""
        # Handle empty files by providing minimal content
        if not content.strip():
            content = "# Empty configuration file"
            lines = [content]

        chunk = self._create_chunk(
            content=content,
            chunk_type=ChunkType.MODULE,
            file_path=file_path,
            start_line=1,
            end_line=len(lines),
            name=os.path.basename(file_path)
            .replace('.ini', '')
            .replace('.cfg', '')
            .replace('.conf', ''),
        )

        # Extract comprehensive file-level metadata
        chunk.metadata.language_specific = self._extract_file_metadata(content, lines)

        return [chunk]

    async def _chunk_by_sections(
        self, content: str, lines: List[str], file_path: str
    ) -> List[Chunk]:
        """Chunk INI file by sections."""
        chunks = []

        # Find all sections
        sections: List[Dict[str, Any]] = []
        for match in self.section_pattern.finditer(content):
            section_name = match.group(1)
            line_num = content[: match.start()].count('\n')
            sections.append({'name': section_name, 'start_line': line_num, 'end_line': None})

        # Determine section boundaries
        for i, section in enumerate(sections):
            if i < len(sections) - 1:
                next_start = sections[i + 1]['start_line']
                if isinstance(next_start, int):
                    section['end_line'] = next_start - 1
            else:
                section['end_line'] = len(lines) - 1

        # Handle content before first section (global settings)
        if sections and sections[0]['start_line'] > 0:
            start_line = sections[0]['start_line']
            if isinstance(start_line, int):
                global_content = '\n'.join(lines[:start_line])
                if self._has_significant_content(global_content):
                    global_chunk = self._create_chunk(
                        content=global_content,
                        chunk_type=ChunkType.CONSTANTS,
                        file_path=file_path,
                        start_line=1,
                        end_line=start_line,
                        name='global_settings',
                    )
                    global_chunk.metadata.language_specific = self._extract_section_metadata(
                        global_content, 'global', lines[:start_line]
                    )
                    chunks.append(global_chunk)

        # Create chunks for each section
        for section in sections:
            start_line = section.get('start_line')
            end_line = section.get('end_line')
            section_name = section.get('name')

            if not all(isinstance(x, int) for x in [start_line, end_line]) or not isinstance(
                section_name, str
            ):
                continue

            # Type assertions for mypy
            assert isinstance(start_line, int)
            assert isinstance(end_line, int)
            assert isinstance(section_name, str)

            section_lines = lines[start_line : end_line + 1]
            section_content = '\n'.join(section_lines)

            # Skip trivial sections
            if not self._has_significant_content(section_content):
                logger.info("[UNTESTED PATH] ini skipping trivial section")
                continue

            chunk = self._create_chunk(
                content=section_content,
                chunk_type=self._determine_section_type(section_name),
                file_path=file_path,
                start_line=start_line + 1,
                end_line=end_line + 1,
                name=section_name,
            )

            # Extract section-specific metadata
            chunk.metadata.language_specific = self._extract_section_metadata(
                section_content, section_name, section_lines
            )

            chunks.append(chunk)

        return chunks

    def _has_significant_content(self, content: str) -> bool:
        """Check if content has enough substance to warrant a chunk."""
        # Remove comments and empty lines
        significant_lines = []
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith(('#', ';')):
                significant_lines.append(line)

        # Need at least 3 non-comment lines
        return len(significant_lines) >= 3

    def _determine_section_type(self, section_name: str) -> ChunkType:
        """Determine appropriate ChunkType based on section name."""
        section_lower = section_name.lower()

        # Database/connection sections
        if any(
            db in section_lower for db in ['database', 'db', 'mysql', 'postgres', 'mongo', 'redis']
        ):
            return ChunkType.CONSTANTS

        # Server/network sections
        if any(srv in section_lower for srv in ['server', 'http', 'api', 'network', 'connection']):
            return ChunkType.CONSTANTS

        # Security sections
        if any(sec in section_lower for sec in ['auth', 'security', 'credentials', 'keys']):
            return ChunkType.CONSTANTS

        # Logging sections
        if any(log in section_lower for log in ['log', 'logger', 'logging']):
            return ChunkType.CONSTANTS

        # Path/directory sections
        if any(path in section_lower for path in ['path', 'directory', 'folder', 'file']):
            return ChunkType.CONSTANTS

        # Test sections
        if 'test' in section_lower:
            return ChunkType.TESTS

        # Default to namespace for general sections
        return ChunkType.NAMESPACE

    def _extract_file_metadata(self, content: str, lines: List[str]) -> Dict[str, Any]:
        """Extract comprehensive file-level metadata."""
        metadata: Dict[str, Any] = {
            'sections': [],
            'global_settings': {},
            'total_settings': 0,
            'security_summary': {
                'issues': [],
                'has_passwords': False,
                'has_tokens': False,
                'has_secrets': False,
                'exposed_ports': [],
                'exposed_hosts': [],
                'risk_level': 'low',
            },
            'quality': {
                'has_comments': False,
                'comment_ratio': 0.0,
                'is_organized': True,
                'has_descriptions': False,
                'duplicate_keys': [],
                'empty_values': [],
            },
            'dependencies': {
                'files': [],
                'env_vars': [],
                'external_services': [],
            },
            'patterns': {
                'config_type': [],
                'frameworks': [],
                'anti_patterns': [],
            },
            'todos': [],
        }

        # Extract sections
        sections_list: List[str] = metadata['sections']
        for match in self.section_pattern.finditer(content):
            sections_list.append(match.group(1))

        # Analyze settings and security
        all_keys = []
        comment_lines = 0
        total_lines = len([line for line in lines if line.strip()])

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            # Count comments
            if line_stripped.startswith(('#', ';')):
                comment_lines += 1
                metadata['quality']['has_comments'] = True

                # Check for TODOs
                todo_match = re.search(r'(TODO|FIXME|HACK|XXX|BUG)[:\s](.+)', line, re.IGNORECASE)
                if todo_match:
                    todos_list: List[str] = metadata['todos']
                    todos_list.append(f"{todo_match.group(1)}: {todo_match.group(2).strip()}")

                # Check if comment describes next line
                if i + 1 < len(lines) and not lines[i + 1].strip().startswith(('#', ';')):
                    metadata['quality']['has_descriptions'] = True
                continue

            # Extract settings
            for pattern in self.setting_patterns:
                setting_match = pattern.match(line)
                if setting_match:
                    key = setting_match.group(1).strip()
                    value = setting_match.group(2).strip()

                    # Track all keys for duplicate detection
                    all_keys.append(key.lower())

                    # Check for empty values
                    if not value or value in ['""', "''", '[]', '{}']:
                        empty_values_list: List[str] = metadata['quality']['empty_values']
                        logger.info("[UNTESTED PATH] ini empty value in file metadata")
                        empty_values_list.append(key)

                    # Analyze setting
                    self._analyze_setting(key, value, metadata)
                    metadata['total_settings'] += 1
                    break

        # Also check content directly for private keys
        if '-----BEGIN' in content and 'PRIVATE' in content and 'KEY-----' in content:
            # Found private key in content
            issue = {
                'type': 'private_key',
                'key': 'embedded_private_key',
                'severity': 'critical',
                'description': 'Private key found in configuration',
            }
            issues_list: List[Dict[str, Any]] = metadata['security_summary']['issues']
            issues_list.append(issue)
            metadata['security_summary']['has_secrets'] = True

        # Calculate quality metrics
        metadata['quality']['comment_ratio'] = comment_lines / max(total_lines, 1)

        # Find duplicate keys
        seen = set()
        for key in all_keys:
            if key in seen:
                duplicate_keys_list: List[str] = metadata['quality']['duplicate_keys']
                if key not in duplicate_keys_list:
                    duplicate_keys_list.append(key)
            seen.add(key)

        # Determine patterns
        self._detect_ini_patterns(metadata)

        # Assess security risk level
        security_issues = metadata['security_summary']['issues']
        if any(issue['severity'] == 'critical' for issue in security_issues):
            metadata['security_summary']['risk_level'] = 'critical'
        elif any(issue['severity'] == 'high' for issue in security_issues):
            metadata['security_summary']['risk_level'] = 'high'
        elif security_issues:
            metadata['security_summary']['risk_level'] = 'medium'

        return metadata

    def _extract_section_metadata(
        self, content: str, section_name: str, lines: List[str]
    ) -> Dict[str, Any]:
        """Extract section-specific metadata."""
        metadata: Dict[str, Any] = {
            'section_name': section_name,
            'settings': {},
            'security_issues': [],
            'file_references': [],
            'env_vars': [],
            'duplicate_keys': [],
            'empty_values': [],
            'todos': [],
            'patterns': {
                'type': self._categorize_section(section_name),
                'anti_patterns': [],
                'good_practices': [],
            },
            'complexity': {
                'settings_count': 0,
                'max_value_length': 0,
                'has_multiline_values': False,
                'nesting_level': 0,
            },
        }

        # Check for private key in section content
        if '-----BEGIN' in content and 'PRIVATE' in content and 'KEY-----' in content:
            metadata['security_issues'].append(
                {
                    'type': 'private_key',
                    'severity': 'critical',
                    'key': 'embedded_private_key',
                    'recommendation': 'Store private keys in secure vault, not in config files',
                }
            )

        # Parse settings in section
        current_key = None
        current_value: List[str] = []
        all_keys = []

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            # Skip section header
            if line_stripped.startswith('[') and line_stripped.endswith(']'):
                continue

            # Handle comments
            if line_stripped.startswith(('#', ';')):
                # Extract TODOs
                todo_match = re.search(r'(TODO|FIXME|HACK|XXX|BUG)[:\s](.+)', line, re.IGNORECASE)
                if todo_match:
                    todos_list: List[str] = metadata['todos']
                    todos_list.append(f"{todo_match.group(1)}: {todo_match.group(2).strip()}")
                continue

            # 1. Si la línea está indentada y hay clave activa, es continuación
            if line.startswith((' ', '\t')) and current_key:
                line_content = line.strip()
                current_value.append(line_content)
                metadata['complexity']['has_multiline_values'] = True
                continue

            # 2. Si la línea NO está indentada, intentamos hacer match con setting_patterns
            setting_found = False
            for pattern in self.setting_patterns:
                setting_match = pattern.match(line)
                if setting_match:
                    # Guardar el valor anterior si existe
                    if current_key:
                        self._save_setting(current_key, '\n'.join(current_value), metadata)
                    current_key = setting_match.group(1).strip()
                    value_part = setting_match.group(2).strip()
                    all_keys.append(current_key.lower())
                    setting_found = True
                    # Multilínea por backslash
                    if value_part.rstrip().endswith('\\\\'):
                        logger.info("[UNTESTED PATH] ini multiline value with double backslash")
                        current_value = [value_part.rstrip()[:-2].rstrip()]
                        metadata['complexity']['has_multiline_values'] = True
                    elif value_part.rstrip().endswith('\\'):
                        current_value = [value_part.rstrip()[:-1].rstrip()]
                        metadata['complexity']['has_multiline_values'] = True
                    else:
                        current_value = [value_part]
                    break

            # 3. Si no hay match y hay clave activa, cerramos el valor anterior
            if not setting_found and current_key:
                logger.info("[UNTESTED PATH] ini closing previous value")
                self._save_setting(current_key, '\n'.join(current_value), metadata)
                current_key = None
                current_value = []

        # Save final setting
        if current_key:
            self._save_setting(current_key, '\n'.join(current_value), metadata)

        # Find duplicate keys
        seen = set()
        for key in all_keys:
            if key in seen:
                duplicate_keys_list: List[str] = metadata['duplicate_keys']
                if key not in duplicate_keys_list:
                    duplicate_keys_list.append(key)
            seen.add(key)

        # Detect section-specific patterns
        self._detect_section_patterns(metadata, section_name)

        metadata['complexity']['settings_count'] = len(metadata['settings'])

        return metadata

    def _save_setting(self, key: str, value: str, metadata: Dict[str, Any]) -> None:
        """Save a setting and analyze it."""
        # Store setting
        settings_dict: Dict[str, str] = metadata['settings']
        settings_dict[key] = value

        # Track empty values
        if not value or value in ['""', "''", '[]', '{}']:
            empty_values_list: List[str] = metadata['empty_values']
            empty_values_list.append(key)

        # Update max value length
        complexity_dict: Dict[str, Any] = metadata['complexity']
        complexity_dict['max_value_length'] = max(complexity_dict['max_value_length'], len(value))

        # Security analysis
        self._check_setting_security(key, value, metadata)

        # File reference detection
        if self._is_file_reference(key, value):
            logger.info("[UNTESTED PATH] ini file reference detected")
            file_refs_list: List[Dict[str, str]] = metadata['file_references']
            file_refs_list.append({'key': key, 'path': value})

        # Environment variable detection
        env_vars = []
        for pattern in self.env_var_patterns:
            env_vars.extend(pattern.findall(value))
        if env_vars:
            env_vars_list = metadata['env_vars']  # type: List[str]
            env_vars_list.extend(env_vars)

    def _analyze_setting(self, key: str, value: str, file_metadata: Dict[str, Any]) -> None:
        """Analyze a setting for file-level metadata."""
        # Security checks
        for check_type, pattern in self.security_patterns.items():
            sec_match = pattern.search(f"{key}={value}")
            if sec_match:
                issue = {
                    'type': check_type,
                    'key': key,
                    'severity': 'critical' if check_type in ['password', 'private_key'] else 'high',
                    'description': f"Potential {check_type} exposure in {key}",
                }
                issues_list: List[Dict[str, Any]] = file_metadata['security_summary']['issues']
                issues_list.append(issue)

                if check_type == 'password':
                    file_metadata['security_summary']['has_passwords'] = True
                elif check_type == 'token':
                    file_metadata['security_summary']['has_tokens'] = True
                elif check_type in ['private_key', 'connection_string']:
                    file_metadata['security_summary']['has_secrets'] = True

        # Port/host detection
        if 'port' in key.lower():
            try:
                port = int(value)
                logger.info("[UNTESTED PATH] ini port detected")
                ports_list: List[int] = file_metadata['security_summary']['exposed_ports']
                ports_list.append(port)
            except ValueError:
                pass

        if any(h in key.lower() for h in ['host', 'server', 'endpoint', 'url']):
            if value and not value.startswith(('$', '%', '{')):
                logger.info("[UNTESTED PATH] ini host/server detected")
                hosts_list = file_metadata['security_summary']['exposed_hosts']
                hosts_list.append(value)

        # Dependency detection - improved file reference detection
        if self._is_file_reference(key, value):
            files_list: List[str] = file_metadata['dependencies']['files']
            if value not in files_list:  # Avoid duplicates
                files_list.append(value)

        # Environment variable detection
        for pattern in self.env_var_patterns:
            env_vars = pattern.findall(value)
            env_vars_list = file_metadata['dependencies']['env_vars']
            env_vars_list.extend(env_vars)

    def _check_setting_security(self, key: str, value: str, metadata: Dict[str, Any]) -> None:
        """Check individual setting for security issues."""
        key_lower = key.lower()

        # Password detection
        if any(pwd in key_lower for pwd in ['password', 'passwd', 'pwd', 'pass']):
            if value and not value.startswith(('$', '%', '${')) and value != '********':
                sec_issues_pwd: List[Dict[str, str]] = metadata['security_issues']
                sec_issues_pwd.append(
                    {
                        'type': 'hardcoded_password',
                        'severity': 'critical',
                        'key': key,
                        'recommendation': 'Use environment variable or secure vault',
                    }
                )

        # API key/token detection
        if any(token in key_lower for token in ['api_key', 'apikey', 'token', 'secret', 'auth']):
            if value and len(value) > 10 and not value.startswith(('$', '%', '${')):
                sec_issues_token: List[Dict[str, str]] = metadata['security_issues']
                sec_issues_token.append(
                    {
                        'type': 'exposed_credential',
                        'severity': 'high',
                        'key': key,
                        'recommendation': 'Move to environment variable',
                    }
                )

        # Connection string with credentials
        if (
            'connection' in key_lower
            or 'dsn' in key_lower
            or 'sql_server' in key_lower
            or 'database' in key_lower
            or
            # Check if we're in a connection-related section
            (
                hasattr(metadata, 'get')
                and isinstance(metadata.get('section_name', ''), str)
                and 'connection' in metadata.get('section_name', '').lower()
            )
        ):
            logger.info("[UNTESTED PATH] ini checking connection string")
            # Check for connection string patterns
            if (
                ('@' in value and ':' in value and '//' in value)
                or ('Server=' in value and 'Password=' in value)
                or ('User ID=' in value and 'Password=' in value)
            ):
                logger.info("[UNTESTED PATH] ini found credentials in connection string")
                sec_issues_conn: List[Dict[str, str]] = metadata['security_issues']
                sec_issues_conn.append(
                    {
                        'type': 'credentials_in_connection_string',
                        'severity': 'high',
                        'key': key,
                        'recommendation': 'Use separate username/password fields with env vars',
                    }
                )

    def _is_file_reference(self, key: str, value: str) -> bool:
        """Check if a setting value is a file path."""
        key_lower = key.lower()

        # Key hints - expanded list
        path_keywords = [
            'path',
            'file',
            'dir',
            'directory',
            'folder',
            'location',
            'log',
            'cache',
            'temp',
            'config',
            'backup',
            'data',
            'script',
            'style',
        ]
        if any(kw in key_lower for kw in path_keywords):
            return True

        # Value patterns
        for pattern in self.path_patterns:
            if pattern.search(value):
                return True

        # Additional simple checks for common file patterns
        # Check for common file extensions
        common_extensions = [
            '.log',
            '.db',
            '.json',
            '.xml',
            '.ini',
            '.config',
            '.txt',
            '.css',
            '.js',
            '.py',
            '.java',
            '.cpp',
            '.tar.gz',
        ]
        if any(value.endswith(ext) for ext in common_extensions):
            logger.info("[UNTESTED PATH] ini file extension match")
            return True

        # Check for path separators
        if '/' in value or '\\' in value:
            return True

        return False

    def _categorize_section(self, section_name: str) -> str:
        """Categorize section type based on name."""
        section_lower = section_name.lower()

        if any(db in section_lower for db in ['database', 'db', 'mysql', 'postgres', 'mongo']):
            return 'database'
        elif any(srv in section_lower for srv in ['server', 'http', 'api', 'web']):
            return 'server'
        elif any(auth in section_lower for auth in ['auth', 'security', 'credentials']):
            return 'authentication'
        elif any(log in section_lower for log in ['log', 'logger', 'logging']):
            return 'logging'
        elif any(path in section_lower for path in ['path', 'directory', 'file']):
            return 'paths'
        elif 'test' in section_lower:
            logger.info("[UNTESTED PATH] ini test section detected")
            return 'testing'
        elif 'dev' in section_lower:
            logger.info("[UNTESTED PATH] ini dev section detected")
            return 'development'
        elif 'prod' in section_lower:
            logger.info("[UNTESTED PATH] ini prod section detected")
            return 'production'
        else:
            return 'general'

    def _detect_ini_patterns(self, metadata: Dict[str, Any]) -> None:
        """Detect configuration patterns and anti-patterns."""
        # Determine config type
        sections = [s.lower() for s in metadata['sections']]

        config_type_list: List[str] = metadata['patterns']['config_type']
        if any('database' in s or 'db' in s for s in sections):
            config_type_list.append('database_config')
        if any('server' in s or 'http' in s or 'api' in s for s in sections):
            logger.info("[UNTESTED PATH] ini web server config detected")
            config_type_list.append('web_server_config')
        if any('log' in s for s in sections):
            logger.info("[UNTESTED PATH] ini logging config detected")
            config_type_list.append('logging_config')
        if any('test' in s for s in sections):
            config_type_list.append('test_config')

        # Detect frameworks
        settings_text = str(metadata).lower()
        frameworks_list: List[str] = metadata['patterns']['frameworks']
        if 'django' in settings_text:
            frameworks_list.append('django')
        if 'flask' in settings_text:
            frameworks_list.append('flask')
        if 'pytest' in settings_text:
            frameworks_list.append('pytest')

        # Anti-patterns
        anti_patterns_list: List[str] = metadata['patterns']['anti_patterns']
        if metadata['quality']['duplicate_keys']:
            anti_patterns_list.append('duplicate_keys')
        if len(metadata['quality']['empty_values']) > 5:
            anti_patterns_list.append('many_empty_values')
        if metadata['security_summary']['has_passwords']:
            anti_patterns_list.append('hardcoded_passwords')
        if metadata['total_settings'] > 100:
            anti_patterns_list.append('oversized_config')
        if not metadata['quality']['has_comments']:
            anti_patterns_list.append('no_documentation')

    def _detect_section_patterns(self, metadata: Dict[str, Any], section_name: str) -> None:
        """Detect section-specific patterns."""
        settings_count = len(metadata['settings'])

        # Anti-patterns
        anti_patterns: List[str] = metadata['patterns']['anti_patterns']
        if settings_count > 30:
            anti_patterns.append('large_section')
        if metadata['duplicate_keys']:
            anti_patterns.append('duplicate_keys')
        if len(metadata['empty_values']) > settings_count * 0.3:
            anti_patterns.append('many_empty_values')

        # Good practices
        good_practices: List[str] = metadata['patterns']['good_practices']
        if metadata['env_vars'] and not metadata['security_issues']:
            logger.info("[UNTESTED PATH] ini good practice: uses env vars")
            good_practices.append('uses_env_vars')
        if metadata['todos']:
            logger.info("[UNTESTED PATH] ini good practice: has todos")
            good_practices.append('has_todos')
        if metadata['complexity']['has_multiline_values']:
            good_practices.append('structured_values')
