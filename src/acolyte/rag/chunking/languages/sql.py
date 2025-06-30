"""
SQL chunker using tree-sitter-languages.
Handles DDL, DML, stored procedures, and complex queries.
"""

from typing import Dict, List, Any, Optional, Callable
from tree_sitter_languages import get_language  # type: ignore

from acolyte.models.chunk import ChunkType
from acolyte.core.logging import logger
from acolyte.rag.chunking.base import LanguageChunker
from acolyte.rag.chunking.mixins import PatternDetectionMixin, SecurityAnalysisMixin


class SQLChunker(LanguageChunker, PatternDetectionMixin, SecurityAnalysisMixin):
    """
    SQL-specific chunker using tree-sitter.

    Handles:
    - DDL (CREATE, ALTER, DROP)
    - DML (SELECT, INSERT, UPDATE, DELETE)
    - Stored procedures and functions
    - Views and triggers
    - CTEs and complex queries
    """

    def _get_language_name(self) -> str:
        """Return language identifier."""
        return 'sql'

    def _get_import_node_types(self) -> List[str]:
        """Get node types that represent imports for sql."""
        return []

    def _is_comment_node(self, node) -> bool:
        """Check if node is a comment."""
        return node.type in ['comment', 'line_comment', 'block_comment']

    def _get_tree_sitter_language(self) -> Any:
        """Get SQL language for tree-sitter."""
        return get_language('sql')

    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """
        SQL-specific node types to chunk.

        Maps SQL constructs to appropriate ChunkTypes.
        """
        return {
            # DDL - Tables and schema objects
            'create_table_statement': ChunkType.CLASS,  # Tables as "classes"
            'create_view_statement': ChunkType.INTERFACE,  # Views as "interfaces"
            'create_index_statement': ChunkType.PROPERTY,  # Indexes as "properties"
            # Stored procedures and functions
            'create_procedure_statement': ChunkType.FUNCTION,
            'create_function_statement': ChunkType.FUNCTION,
            'create_trigger_statement': ChunkType.METHOD,  # Triggers as "methods"
            # Complex queries as functions
            'with_clause': ChunkType.FUNCTION,  # CTEs
            'select_statement': ChunkType.FUNCTION,  # Complex SELECTs
            'insert_statement': ChunkType.FUNCTION,
            'update_statement': ChunkType.FUNCTION,
            'delete_statement': ChunkType.FUNCTION,
            # Type definitions
            'create_type_statement': ChunkType.TYPES,
            # Schema/database level
            'create_schema_statement': ChunkType.NAMESPACE,
            'create_database_statement': ChunkType.NAMESPACE,
        }

    def _create_chunk_from_node(
        self, node, lines: List[str], file_path: str, chunk_type: ChunkType, processed_ranges
    ):
        """Override to handle SQL-specific cases."""
        # For small DML statements, check if they should be grouped
        if node.type in [
            'select_statement',
            'insert_statement',
            'update_statement',
            'delete_statement',
        ]:
            # Calculate statement size
            start_line = node.start_point[0]
            end_line = node.end_point[0]
            statement_lines = end_line - start_line + 1

            # Small statements (< 5 lines) might not be worth individual chunks
            if statement_lines < 5:
                # Check if it's a simple statement without CTEs or subqueries
                if not self._has_complex_structure(node):
                    return None  # Let it be grouped with module code

        # Standard processing
        chunk = super()._create_chunk_from_node(
            node, lines, file_path, chunk_type, processed_ranges
        )

        # Add SQL-specific metadata
        if chunk:
            metadata = self._extract_sql_metadata(node)

            # Add enrichments
            metadata['complexity'] = self._calculate_complexity(node)
            metadata['todos'] = self._extract_todos(node)
            metadata['patterns'] = self._detect_patterns(node)

            # Security analysis
            security_issues = []
            for check in self._get_security_patterns():
                issue = check(node, node.text.decode('utf8'))
                if issue:
                    security_issues.append(issue)
            if security_issues:
                metadata['security'] = security_issues

            # Quality metrics
            metadata['quality'] = self._analyze_sql_quality(node)

            chunk.metadata.language_specific = metadata

        return chunk

    def _has_complex_structure(self, node) -> bool:
        """Check if a SQL statement has complex structures worth chunking."""
        complex_indicators = [
            'with_clause',  # CTEs
            'subquery',
            'join_clause',
            'union_statement',
            'case_expression',
            'window_function',
        ]

        def check_node(n):
            if n.type in complex_indicators:
                return True
            for child in n.children:
                if check_node(child):
                    return True
            return False

        return check_node(node)

    def _extract_sql_metadata(self, node) -> Dict[str, Any]:
        """Extract SQL-specific metadata."""
        metadata = {
            'statement_type': node.type,
            'object_name': None,
            'dependencies': [],
            'is_temporary': False,
            'has_conditions': False,
        }

        # Extract object name based on statement type
        if 'create' in node.type:
            metadata['object_name'] = self._extract_object_name(node)
            metadata['is_temporary'] = self._is_temporary_object(node)

        # Extract table dependencies
        metadata['dependencies'] = self._extract_table_references(node)

        # Check for WHERE/HAVING clauses
        metadata['has_conditions'] = self._has_conditions(node)

        return metadata

    def _extract_object_name(self, node) -> Optional[str]:
        """Extract the name of the object being created/modified."""
        # Buscar todos los qualified_name en el árbol y devolver el más largo
        qualified_names = []

        def find_qualified_names(n):
            for child in getattr(n, 'children', []):
                if child.type == 'qualified_name':
                    parts = [
                        subchild.text.decode('utf8')
                        for subchild in child.children
                        if subchild.type == 'identifier'
                    ]
                    if parts:
                        qualified_names.append('.'.join(parts))
                find_qualified_names(child)

        find_qualified_names(node)
        if qualified_names:
            # Devuelve el nombre calificado más largo
            return max(qualified_names, key=lambda s: s.count('.'))

        # Si no hay qualified_name, busca secuencia identifier-dot-identifier
        def find_identifiers_sequence(n):
            identifiers = []
            for child in getattr(n, 'children', []):
                if child.type == 'identifier':
                    identifiers.append(child.text.decode('utf8'))
                elif child.type == 'dot':
                    continue
                else:
                    # Busca recursivamente
                    result = find_identifiers_sequence(child)
                    if result:
                        identifiers.extend(result)
            return identifiers

        ids = find_identifiers_sequence(node)
        if len(ids) >= 2:
            return '.'.join(ids)

        # Fallback: busca el primer identifier
        for child in getattr(node, 'children', []):
            if child.type == 'identifier':
                logger.info("[UNTESTED PATH] sql fallback identifier found")
                return child.text.decode('utf8')
        return None

    def _is_temporary_object(self, node) -> bool:
        """Check if object is temporary (TEMP/TEMPORARY keyword)."""
        text = node.text.decode('utf8').upper()
        return 'TEMPORARY' in text or ' TEMP ' in text

    def _extract_table_references(self, node) -> List[str]:
        """Extract all table references from a SQL statement."""
        tables = set()

        def find_tables(n):
            if n.type == 'from_clause' or n.type == 'join_clause':
                # Look for table references in FROM/JOIN
                for child in n.children:
                    if child.type == 'identifier' or child.type == 'qualified_name':
                        table_name = child.text.decode('utf8')
                        # Skip aliases and keywords
                        if table_name.upper() not in [
                            'FROM',
                            'JOIN',
                            'ON',
                            'AS',
                            'LEFT',
                            'RIGHT',
                            'INNER',
                            'OUTER',
                        ]:
                            logger.info("[UNTESTED PATH] sql table reference found")
                            tables.add(table_name)

            # Recurse
            for child in n.children:
                find_tables(child)

        find_tables(node)
        return sorted(list(tables))

    def _has_conditions(self, node) -> bool:
        """Check if statement has WHERE, HAVING, or ON conditions."""
        condition_types = ['where_clause', 'having_clause', 'on_clause']

        def check_conditions(n):
            if n.type in condition_types:
                return True
            for child in n.children:
                if check_conditions(child):
                    return True
            return False

        return check_conditions(node)

    def _extract_dependencies_from_imports(self, import_nodes) -> List[str]:
        """
        SQL doesn't have imports like other languages.

        We track schema/database references instead.
        """
        schemas = set()

        for node in import_nodes:
            # Look for USE statements or schema references
            text = node.text.decode('utf8')
            if text.upper().startswith('USE '):
                schema = text[4:].strip().rstrip(';')
                schemas.add(schema)

        return sorted(list(schemas))

    def _get_decision_node_types(self):
        """SQL-specific decision nodes."""
        return {
            'case_expression',
            'case_when',
            'if_expression',
            'where_clause',
            'having_clause',
            'when_clause',
        }

    def _get_security_patterns(self) -> List[Callable[[Any, str], dict]]:
        """SQL-specific security patterns."""
        base = super()._get_security_patterns()
        sql_patterns = [
            self._check_sql_injection_risk,
            self._check_dynamic_sql,
            self._check_grant_statements,
        ]
        return base + sql_patterns

    def _check_sql_injection_risk(self, node, text: str) -> Dict[str, Any]:
        """Check for SQL injection vulnerabilities."""
        if 'EXEC(' in text.upper() or 'EXECUTE IMMEDIATE' in text.upper():
            return {
                'type': 'dynamic_sql_execution',
                'severity': 'high',
                'description': 'Dynamic SQL execution can lead to injection',
            }
        return {}

    def _check_dynamic_sql(self, node, text: str) -> Dict[str, Any]:
        """Check for string concatenation in SQL."""
        if "'||'" in text or "' + '" in text or 'CONCAT(' in text.upper():
            return {
                'type': 'string_concatenation',
                'severity': 'medium',
                'description': 'String concatenation in SQL queries',
            }
        return {}

    def _check_grant_statements(self, node, text: str) -> Dict[str, Any]:
        """Check for overly permissive grants."""
        if 'GRANT ALL' in text.upper() or 'TO PUBLIC' in text.upper():
            return {
                'type': 'permissive_grant',
                'severity': 'high',
                'description': 'Overly permissive database permissions',
            }
        return {}

    def _analyze_sql_quality(self, node) -> Dict[str, Any]:
        """Analyze SQL quality indicators."""
        text = node.text.decode('utf8').upper()
        return {
            'has_comments': '--' in text or '/*' in text,
            'uses_transactions': any(
                kw in text
                for kw in ['BEGIN', 'COMMIT', 'ROLLBACK', 'START TRANSACTION', 'SAVEPOINT']
            ),
            'has_error_handling': 'EXCEPTION' in text or 'TRY' in text,
            'uses_cte': 'WITH' in text and node.type != 'with_clause',
            'has_indexes': 'INDEX' in text,
        }

    def _detect_language_patterns(
        self, node, patterns: Dict[str, List[str]], metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[str]]:
        """Detect SQL-specific patterns."""
        text = node.text.decode('utf8').upper()

        # Initialize optimization list if not present
        if 'optimization' not in patterns:
            patterns['optimization'] = []

        # Anti-patterns
        if 'SELECT *' in text:
            patterns['anti'].append('select_star')
        if text.count('JOIN') > 5:
            patterns['anti'].append('excessive_joins')
        if 'CURSOR' in text:
            logger.info("[UNTESTED PATH] sql cursor usage detected")
            patterns['anti'].append('cursor_usage')

        # Good patterns
        if 'EXPLAIN' in text:
            patterns['optimization'].append('uses_explain')
        if 'PARTITION' in text:
            patterns['optimization'].append('partitioned_tables')

        return patterns
