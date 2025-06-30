"""
Mixins for chunking functionality.
Provides reusable components for language-specific chunkers.
"""

from typing import Dict, List, Any, Optional, Callable
import re
from acolyte.core.logging import logger


class ComplexityMixin:
    """
    Mixin for calculating code complexity metrics.

    Provides generic complexity calculation that works for most languages.
    Override specific methods if language needs custom behavior.
    """

    def _calculate_complexity(self, node) -> Dict[str, int]:
        """
        Calculate cyclomatic complexity and other metrics using tree-sitter AST.

        Returns:
            Dictionary with complexity metrics:
            - cyclomatic: McCabe cyclomatic complexity
            - nesting_depth: Maximum nesting level
            - lines_of_code: Actual lines (excluding blanks/comments)
            - branches: Number of decision points
        """
        complexity = {
            'cyclomatic': 1,  # Start at 1
            'nesting_depth': 0,
            'lines_of_code': node.end_point[0] - node.start_point[0] + 1,
            'branches': 0,
        }

        self._analyze_complexity_recursive(node, complexity, 0)

        return complexity

    def _analyze_complexity_recursive(
        self, node, complexity: Dict[str, int], depth: int, max_depth: int = 100
    ):
        """Recursively analyze node for complexity metrics with depth limit."""
        # Prevent stack overflow
        if depth > max_depth:
            logger.info("[UNTESTED PATH] Max recursion depth reached in complexity analysis")
            return

        # Update max nesting
        complexity['nesting_depth'] = max(complexity['nesting_depth'], depth)

        # Get decision point node types for this language
        decision_types = self._get_decision_node_types()
        nesting_types = self._get_nesting_node_types()

        # Count decision points
        if node.type in decision_types:
            complexity['cyclomatic'] += 1
            complexity['branches'] += 1

        # Recurse with proper depth tracking
        for child in node.children:
            next_depth = depth + 1 if node.type in nesting_types else depth
            self._analyze_complexity_recursive(child, complexity, next_depth, max_depth)

    def _get_decision_node_types(self):
        """
        Get node types that represent decision points.
        Override in language-specific chunkers for customization.
        """
        # Common decision nodes across languages
        return {
            'if_statement',
            'if_expression',
            'elif_clause',
            'else_clause',
            'while_statement',
            'while_expression',
            'for_statement',
            'for_expression',
            'switch_statement',
            'switch_expression',
            'case_statement',
            'when_entry',
            'try_statement',
            'except_clause',
            'catch_clause',
            'finally_clause',
            'conditional_expression',
            'ternary_expression',
            'match_expression',
            'match_arm',
        }

    def _get_nesting_node_types(self):
        """
        Get node types that increase nesting depth.
        Override in language-specific chunkers for customization.
        """
        return {
            'block',
            'compound_statement',
            'if_statement',
            'for_statement',
            'while_statement',
            'with_statement',
            'try_statement',
            'function_definition',
            'class_definition',
            'method_definition',
        }


class TodoExtractionMixin:
    """
    Mixin for extracting TODO/FIXME/HACK comments.

    Works with tree-sitter comment nodes.
    """

    # Default patterns - can be overridden
    TODO_PATTERNS = [
        'TODO',
        'FIXME',
        'HACK',
        'BUG',
        'OPTIMIZE',
        'REFACTOR',
        'NOTE',
        'WARNING',
        'XXX',
    ]

    def _extract_todos(self, node) -> List[Dict[str, Any]]:
        """
        Extract TODO-style comments from AST.

        Args:
            node: Tree-sitter node to analyse

        Returns:
            List of TODO items with type, text, and line number (1-based)
        """
        todos = []
        self._find_todos_recursive(node, todos)
        return todos

    def _find_todos_recursive(self, node, todos: List[Dict[str, Any]]):
        """Recursively find TODO comments in AST."""
        # Check if this is a comment node
        # The language chunker must implement _is_comment_node()
        if hasattr(self, '_is_comment_node') and self._is_comment_node(node):  # type: ignore
            text = node.text.decode('utf8', errors='ignore')
            # Convert 0-based to 1-based line number for external use
            line = node.start_point[0] + 1

            # Cache compiled regexes for performance
            _todo_regexps = getattr(self, "_todo_regexps", None)
            if _todo_regexps is None:  # cache on the fly
                self._todo_regexps = {
                    p: re.compile(rf"{re.escape(p)}[:\s]*(.*)", re.IGNORECASE)
                    for p in self.TODO_PATTERNS
                }
                _todo_regexps = self._todo_regexps

            # Check for TODO patterns
            for pattern, rx in _todo_regexps.items():
                if pattern in text.upper():
                    # Extract the actual comment text after the pattern
                    match = rx.search(text)
                    if match:
                        todo_text = match.group(1).strip()
                        # Clean up comment markers
                        todo_text = self._clean_comment_text(todo_text)

                        todos.append({'type': pattern, 'text': todo_text, 'line': line})
                        break  # Only match first pattern

        # Recurse into children
        for child in node.children:
            self._find_todos_recursive(child, todos)

    def _clean_comment_text(self, text: str) -> str:
        """Clean up comment markers from text."""
        # Remove common comment markers
        text = text.strip()
        text = re.sub(r'^[/*#\s-]+', '', text)  # Leading markers
        text = re.sub(r'[*/\s-]+$', '', text)  # Trailing markers
        return text.strip()


class SecurityAnalysisMixin:
    """
    Mixin for detecting common security issues.

    Provides base implementation for security pattern detection.
    Languages should override SECURITY_PATTERNS for specific checks.
    """

    def _detect_security_issues(self, node) -> List[Dict[str, Any]]:
        """
        Detect potential security vulnerabilities in code.

        Returns:
            List of security issues with type, severity, line (1-based), and description
        """
        issues = []
        self._analyze_security_recursive(node, issues)
        return issues

    def _analyze_security_recursive(self, node, issues: List[Dict[str, Any]]):
        """Recursively analyze nodes for security issues."""
        node_text = node.text.decode('utf8', errors='ignore')
        # Convert 0-based to 1-based line number for external use
        line = node.start_point[0] + 1

        # Check each security pattern
        for pattern_check in self._get_security_patterns():
            if pattern_check(node, node_text):
                issue = pattern_check(node, node_text)
                if issue:
                    issue['line'] = line
                    issues.append(issue)

        # Recurse
        for child in node.children:
            self._analyze_security_recursive(child, issues)

    def _get_security_patterns(self) -> List[Callable[[Any, str], Optional[Dict[str, Any]]]]:
        """
        Get list of security check functions.
        Override in language-specific classes.
        """
        return [
            self._check_sql_injection,
            self._check_hardcoded_credentials,
            self._check_weak_crypto,
            self._check_unsafe_deserialization,
        ]

    def _check_sql_injection(self, node, text: str) -> Optional[Dict[str, Any]]:
        """Check for potential SQL injection."""
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE']

        if any(keyword in text.upper() for keyword in sql_keywords):
            # Check for string concatenation or interpolation
            if any(op in text for op in ['+', 'concat', 'format', '%', 'f"', "f'", '${']):
                return {
                    'type': 'sql_injection_risk',
                    'severity': 'high',
                    'description': 'Potential SQL injection via string manipulation',
                }
        return None

    def _check_hardcoded_credentials(self, node, text: str) -> Optional[Dict[str, Any]]:
        """Check for hardcoded passwords or API keys."""
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
        ]

        text_lower = text.lower()
        for pattern in credential_patterns:
            if pattern in text_lower and '=' in text:
                # Check if it looks like a hardcoded value
                if any(quote in text for quote in ['"', "'"]) and len(text) < 200:
                    # Exclude if reading from config/env
                    if not any(
                        safe in text_lower for safe in ['env', 'config', 'getenv', 'environ']
                    ):
                        return {
                            'type': 'hardcoded_credential',
                            'severity': 'critical',
                            'description': f'Possible hardcoded {pattern}',
                        }
        return None

    def _check_weak_crypto(self, node, text: str) -> Optional[Dict[str, Any]]:
        """Check for weak cryptographic algorithms."""
        weak_algos = ['MD5', 'SHA1', 'DES', 'RC4', 'RC2']

        for algo in weak_algos:
            if algo in text.upper():
                return {
                    'type': 'weak_cryptography',
                    'severity': 'medium',
                    'description': f'Use of weak algorithm: {algo}',
                }
        return None

    def _check_unsafe_deserialization(self, node, text: str) -> Optional[Dict[str, Any]]:
        """Check for unsafe deserialization patterns."""
        unsafe_patterns = [
            'pickle.loads',
            'eval(',
            'exec(',
            'marshal.loads',
            'yaml.load(',
            'deserialize(',
            'unserialize(',
        ]

        for pattern in unsafe_patterns:
            if pattern in text:
                return {
                    'type': 'unsafe_deserialization',
                    'severity': 'high',
                    'description': f'Potential unsafe deserialization with {pattern}',
                }
        return None


class PatternDetectionMixin:
    """
    Mixin for detecting code patterns and anti-patterns.

    Provides framework for pattern analysis.
    """

    def _detect_patterns(
        self, node, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[str]]:
        """
        Detect patterns and anti-patterns in code.

        Args:
            node: AST node to analyze
            metadata: Optional metadata with complexity metrics

        Returns:
            Dictionary with 'anti' and 'good' pattern lists
        """
        patterns = {'anti': [], 'good': [], 'framework': []}

        # Use metadata if provided, otherwise calculate
        if not metadata:
            if hasattr(self, '_calculate_complexity'):
                complexity = self._calculate_complexity(node)  # type: ignore
            else:
                logger.info("[UNTESTED PATH] No _calculate_complexity method, using defaults")
                complexity = {'cyclomatic': 0, 'nesting_depth': 0, 'lines_of_code': 0}
        else:
            complexity = metadata.get('complexity', {})

        # Anti-patterns based on complexity
        if complexity.get('cyclomatic', 0) > 10:
            patterns['anti'].append('high_complexity')

        if complexity.get('nesting_depth', 0) > 4:
            patterns['anti'].append('deep_nesting')

        if complexity.get('lines_of_code', 0) > 100:
            patterns['anti'].append('long_function')

        # Language-specific patterns
        patterns = self._detect_language_patterns(node, patterns, metadata)

        return patterns

    def _detect_language_patterns(
        self, node, patterns: Dict[str, List[str]], metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[str]]:
        """Override in language-specific chunkers for custom patterns."""
        return patterns


class DependencyAnalysisMixin:
    """
    Mixin for analyzing code dependencies.

    Provides base implementation for dependency extraction.
    """

    def _analyze_dependencies(self, node) -> Dict[str, List[str]]:
        """
        Analyze internal and external dependencies.

        Returns:
            Dictionary with 'internal' and 'external' dependency lists
        """
        dependencies = {'internal': [], 'external': []}

        # Extract imports
        import_nodes = self._find_import_nodes(node)
        for import_node in import_nodes:
            deps = self._extract_dependencies_from_import(import_node)
            for dep in deps:
                if self._is_internal_dependency(dep):
                    dependencies['internal'].append(dep)
                else:
                    dependencies['external'].append(dep)

        # Remove duplicates and sort
        dependencies['internal'] = sorted(list(set(dependencies['internal'])))
        dependencies['external'] = sorted(list(set(dependencies['external'])))

        return dependencies

    def _find_import_nodes(self, node) -> List:
        """Find all import nodes in AST. Override for language-specific imports."""
        imports = []

        # Check if method exists before calling
        if hasattr(self, '_get_import_node_types'):
            import_types = self._get_import_node_types()  # type: ignore
            if node.type in import_types:
                imports.append(node)

        for child in node.children:
            imports.extend(self._find_import_nodes(child))

        return imports

    def _extract_dependencies_from_import(self, import_node) -> List[str]:
        """Extract dependency names from import node. Override in language chunkers."""
        # Base implementation - languages should override
        return []

    def _is_internal_dependency(self, dep: str) -> bool:
        """Determine if dependency is internal to project. Override as needed."""
        # Get project name from config if available
        if hasattr(self, 'config'):
            project_name = self.config.get('project.name', 'acolyte')  # type: ignore
        else:
            logger.info("[UNTESTED PATH] No config available, using default project name")
            project_name = 'acolyte'

        # Simple heuristic - relative imports or project name
        return dep.startswith('.') or dep.startswith(project_name)
