"""
Base chunker for configuration files.
Provides common functionality for JSON, YAML, TOML, INI, etc.
"""

import re
from typing import Dict, List, Any
from abc import abstractmethod
from tree_sitter import Node
from acolyte.rag.chunking.base import BaseChunker
from acolyte.core.logging import logger


class ConfigChunkerBase(BaseChunker):
    """
    Base class for configuration file chunkers.

    Configuration files have different needs than code:
    - Focus on semantic sections rather than functions/classes
    - Extract security-sensitive information
    - Detect configuration patterns
    - Handle environment variables and paths
    """

    def _get_chunk_size(self) -> int:
        """Config files should have smaller chunks than code."""
        language = self._get_language_name()

        # Try language-specific size first
        size = self.config.get(f'indexing.chunk_sizes.{language}', None)

        # Try generic config size
        if size is None:
            size = self.config.get('indexing.chunk_sizes.config', 50)

        # Fallback to smaller default for config files
        if size is None:
            logger.info("[UNTESTED PATH] config_base using default size")
            size = 50

        return int(size)

    def _extract_env_vars(self, content: str) -> List[Dict[str, Any]]:
        """Extract environment variable references."""
        env_vars = []

        # Common patterns for env vars
        patterns = [
            r'\$\{([A-Z_][A-Z0-9_]*)\}',  # ${VAR}
            r'\$([A-Z_][A-Z0-9_]*)',  # $VAR
            r'%([A-Z_][A-Z0-9_]*)%',  # %VAR%
            r'\{\{([A-Z_][A-Z0-9_]*)\}\}',  # {{VAR}}
            r'env\.([A-Z_][A-Z0-9_]*)',  # env.VAR
            r'ENV\[[\'"](.*?)[\'"]\]',  # ENV['VAR']
            r'process\.env\.([A-Z_][A-Z0-9_]*)',  # process.env.VAR
        ]

        seen = set()
        for pattern in patterns:
            for match in re.finditer(pattern, content):
                var_name = match.group(1)
                if var_name not in seen:
                    env_vars.append(
                        {
                            'name': var_name,
                            'pattern': match.group(0),
                            'line': content[: match.start()].count('\n') + 1,
                        }
                    )
                    seen.add(var_name)

        return env_vars

    def _detect_secrets(self, content: str) -> List[Dict[str, Any]]:
        """Detect potential secrets in configuration."""
        secrets = []
        lines = content.split('\n')

        # Patterns that indicate secrets
        secret_patterns = [
            'password',
            'passwd',
            'pwd',
            'secret',
            'api_key',
            'apikey',
            'access_key',
            'private_key',
            'auth_token',
            'access_token',
            'client_secret',
            'database_password',
            'db_password',
        ]

        for i, line in enumerate(lines):
            line_lower = line.lower()

            # Skip if it's using env vars
            if any(pattern in line for pattern in ['${', '{{', 'env.', 'ENV[', '%']):
                continue

            for pattern in secret_patterns:
                if pattern in line_lower:
                    # Check if there's a value that looks like a secret
                    if ':' in line or '=' in line:
                        # Extract the value part
                        value_part = (
                            line.split(':', 1)[-1] if ':' in line else line.split('=', 1)[-1]
                        )
                        value_part = value_part.strip().strip('"\'')

                        # Check if it looks like an actual secret (not placeholder)
                        if len(value_part) > 5 and value_part not in [
                            'null',
                            'undefined',
                            'none',
                            'changeme',
                            'your-password-here',
                            '********',
                            'xxx',
                        ]:
                            secrets.append(
                                {
                                    'type': pattern,
                                    'line': i + 1,
                                    'severity': 'critical' if 'key' in pattern else 'high',
                                    'pattern': pattern,
                                }
                            )
                            break

        return secrets

    def _extract_urls(self, content: str) -> List[Dict[str, Any]]:
        """Extract URLs from configuration."""
        urls = []

        # URL pattern
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]\'()]+'

        seen = set()
        for match in re.finditer(url_pattern, content):
            url = match.group(0).rstrip(',.;:')
            if url not in seen:
                urls.append({'url': url, 'line': content[: match.start()].count('\n') + 1})
                seen.add(url)

        return urls

    def _extract_paths(self, content: str) -> List[Dict[str, Any]]:
        """Extract file paths from configuration."""
        paths = []

        # Path patterns
        path_patterns = [
            r'[\'"]?(/[a-zA-Z0-9_\-./]+)',  # Unix absolute
            r'[\'"]?([a-zA-Z]:\\[^\'"\n]+)',  # Windows absolute
            r'[\'"]?(\.{1,2}/[a-zA-Z0-9_\-./]+)',  # Relative paths
            r'[\'"]?([a-zA-Z0-9_\-]+\.[a-zA-Z]{2,4})',  # Simple filenames
        ]

        seen = set()
        for pattern in path_patterns:
            for match in re.finditer(pattern, content):
                path = match.group(1).strip('"\'')

                # Filter out common false positives
                if (
                    path not in seen
                    and len(path) > 3
                    and not path.startswith('http')
                    and '.' in path
                    or '/' in path
                    or '\\' in path
                ):

                    paths.append(
                        {
                            'path': path,
                            'type': 'absolute' if path.startswith(('/', 'C:\\')) else 'relative',
                            'line': content[: match.start()].count('\n') + 1,
                        }
                    )
                    seen.add(path)

        return paths

    def _extract_todos(self, content: str) -> List[Dict[str, Any]]:
        """Extract TODO/FIXME comments from config files."""
        todos = []
        lines = content.split('\n')

        # TO-DO patterns for different comment styles
        patterns = [
            (r'#\s*(TODO|FIXME|HACK|NOTE|WARNING|OPTIMIZE):\s*(.+)', '#'),
            (r'//\s*(TODO|FIXME|HACK|NOTE|WARNING|OPTIMIZE):\s*(.+)', '//'),
            (r'/\*\s*(TODO|FIXME|HACK|NOTE|WARNING|OPTIMIZE):\s*(.+)\*/', '/*'),
            (r'<!--\s*(TODO|FIXME|HACK|NOTE|WARNING|OPTIMIZE):\s*(.+)-->', '<!--'),
        ]

        for i, line in enumerate(lines):
            for pattern, comment_type in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    todos.append(
                        {
                            'type': match.group(1).upper(),
                            'text': match.group(2).strip(),
                            'line': i + 1,
                            'comment_style': comment_type,
                        }
                    )
                    break

        return todos

    def _detect_config_patterns(self, content: str) -> Dict[str, List[str]]:
        """Detect common configuration patterns."""
        patterns: Dict[str, List[str]] = {
            'database': [],
            'api': [],
            'auth': [],
            'cache': [],
            'logging': [],
            'deployment': [],
            'testing': [],
        }

        content_lower = content.lower()

        # Database patterns
        db_keywords = ['database', 'db_', 'mysql', 'postgres', 'mongodb', 'redis', 'elasticsearch']
        if any(kw in content_lower for kw in db_keywords):
            if 'host' in content_lower or 'port' in content_lower:
                patterns['database'].append('connection_config')
            if 'pool' in content_lower:
                patterns['database'].append('connection_pooling')
            if 'replica' in content_lower or 'slave' in content_lower:
                patterns['database'].append('replication')

        # API patterns
        api_keywords = ['endpoint', 'api_', 'base_url', 'service_url']
        if any(kw in content_lower for kw in api_keywords):
            patterns['api'].append('external_service')
            if 'timeout' in content_lower:
                patterns['api'].append('timeout_config')
            if 'retry' in content_lower:
                patterns['api'].append('retry_config')

        # Auth patterns
        auth_keywords = ['auth', 'oauth', 'jwt', 'token', 'session']
        if any(kw in content_lower for kw in auth_keywords):
            patterns['auth'].append('authentication')
            if 'oauth' in content_lower:
                patterns['auth'].append('oauth_config')
            if 'jwt' in content_lower:
                patterns['auth'].append('jwt_config')

        # Cache patterns
        if 'cache' in content_lower:
            patterns['cache'].append('cache_config')
            if 'ttl' in content_lower or 'expire' in content_lower:
                patterns['cache'].append('ttl_config')
            if 'redis' in content_lower:
                patterns['cache'].append('redis_cache')

        # Logging patterns
        log_keywords = ['log', 'logger', 'logging']
        if any(kw in content_lower for kw in log_keywords):
            patterns['logging'].append('logging_config')
            if 'level' in content_lower:
                patterns['logging'].append('log_levels')
            if 'rotate' in content_lower:
                patterns['logging'].append('log_rotation')

        # Deployment patterns
        deploy_keywords = ['prod', 'staging', 'development', 'environment']
        if any(kw in content_lower for kw in deploy_keywords):
            patterns['deployment'].append('multi_environment')

        if 'docker' in content_lower or 'container' in content_lower:
            patterns['deployment'].append('containerized')

        if 'kubernetes' in content_lower or 'k8s' in content_lower:
            patterns['deployment'].append('kubernetes')

        # Testing patterns
        if 'test' in content_lower and ('mock' in content_lower or 'fixture' in content_lower):
            patterns['testing'].append('test_config')

        # Remove empty pattern categories
        return {k: v for k, v in patterns.items() if v}

    def _analyze_structure_complexity(self, content: str) -> Dict[str, Any]:
        """Analyze the structural complexity of config file."""
        lines = content.split('\n')

        complexity: Dict[str, Any] = {
            'lines': len(lines),
            'non_empty_lines': len([line for line in lines if line.strip()]),
            'max_line_length': max(len(line) for line in lines) if lines else 0,
            'nesting_indicators': 0,
            'has_arrays': False,
            'has_objects': False,
            'estimated_depth': 0,
        }

        # Count nesting indicators
        for line in lines:
            complexity['nesting_indicators'] += line.count('{') + line.count('[')
            complexity['nesting_indicators'] -= line.count('}') + line.count(']')

            if '[' in line:
                complexity['has_arrays'] = True
            if '{' in line:
                complexity['has_objects'] = True

        # Estimate depth by indentation
        indentations = []
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                indentations.append(indent)

        if indentations:
            # Assume 2 or 4 space indentation
            min_indent = (
                min(i for i in indentations if i > 0) if any(i > 0 for i in indentations) else 2
            )
            complexity['estimated_depth'] = (
                max(i // min_indent for i in indentations) if min_indent > 0 else 0
            )

        return complexity

    def _validate_config_syntax(self, content: str, file_type: str) -> Dict[str, Any]:
        """Basic syntax validation for config files."""
        validation: Dict[str, Any] = {'valid': True, 'errors': [], 'warnings': []}

        # Basic bracket/brace matching
        stack: List[tuple[str, int]] = []
        lines = content.split('\n')

        brackets = {'[': ']', '{': '}', '(': ')'}
        closing = {']': '[', '}': '{', ')': '('}

        for i, line in enumerate(lines):
            # Skip comments
            if line.strip().startswith(('#', '//', '/*', '<!--')):
                continue

            for char in line:
                if char in brackets:
                    stack.append((char, i + 1))
                elif char in closing:
                    if not stack:
                        validation['valid'] = False
                        validation['errors'].append(
                            {'line': i + 1, 'message': f'Unexpected closing {char}'}
                        )
                    elif stack[-1][0] != closing[char]:
                        validation['valid'] = False
                        validation['errors'].append(
                            {
                                'line': i + 1,
                                'message': f'Mismatched bracket: expected {brackets[stack[-1][0]]}, got {char}',
                            }
                        )
                    else:
                        stack.pop()

        # Check unclosed brackets
        if stack:
            validation['valid'] = False
            logger.info("[UNTESTED PATH] config_base unclosed brackets")
            for bracket, line_num in stack:
                validation['errors'].append({'line': line_num, 'message': f'Unclosed {bracket}'})

        # File-type specific checks
        if file_type == 'json' and not content.strip().startswith(('[', '{')):
            validation['warnings'].append({'line': 1, 'message': 'JSON should start with { or ['})

        return validation

    def _get_import_node_types(self) -> List[str]:
        """Config files generally don't have imports."""
        return []  # Override in specific implementations if needed

    def _is_comment_node(self, node: Node) -> bool:
        """Check if node is a comment - override in specific implementations."""
        logger.info("[UNTESTED PATH] config_base._is_comment_node called")
        return node.type == 'comment'  # Generic fallback

    def _extract_dependencies_from_imports(self, import_nodes: List[Node]) -> List[str]:
        """Extract dependencies - config files usually don't have imports."""
        return []  # Override if specific config type has dependencies

    # Keep these abstract to force implementation
    @abstractmethod
    def _get_language_name(self) -> str:
        """Get the language/config type name."""
        pass

    @abstractmethod
    def _get_tree_sitter_language(self) -> Any:
        """Get tree-sitter language for this config type."""
        pass
