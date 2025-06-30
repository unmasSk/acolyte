"""
R chunker using tree-sitter-languages.
Handles R's unique syntax including <- assignment and S4 classes.
"""

from typing import Dict, List, Any, Optional, Callable
from tree_sitter_languages import get_language  # type: ignore

from acolyte.models.chunk import Chunk, ChunkType
from acolyte.core.logging import logger
from acolyte.rag.chunking.base import LanguageChunker
from acolyte.rag.chunking.mixins import PatternDetectionMixin, SecurityAnalysisMixin


class RChunker(LanguageChunker, PatternDetectionMixin, SecurityAnalysisMixin):
    """
    R-specific chunker using tree-sitter.

    Handles R's unique features:
    - Assignment with <- and =
    - S4 classes (setClass, setMethod)
    - library/require imports
    - Vectorized operations
    """

    async def chunk(self, content: str, file_path: str) -> List[Chunk]:
        """Override to ensure module chunks capture top-level code."""
        logger.info("[UNTESTED PATH] r.chunk override called")
        # Get chunks from base implementation
        chunks = await super().chunk(content, file_path)

        # If no chunks were created (e.g., only variable assignments), create a module chunk
        if not chunks and content.strip():
            logger.info("[UNTESTED PATH] r creating module chunk for unchunked content")
            lines = content.split('\n')
            chunk = self._create_chunk(
                content=content,
                chunk_type=ChunkType.MODULE,
                file_path=file_path,
                start_line=1,
                end_line=len(lines),
                name='module',
            )
            # Add metadata including security checks
            metadata = {
                'security': self._detect_security_issues_in_text(content),
                'todos': [],
                'patterns': {},
                'complexity': {'cyclomatic': 1, 'nesting_depth': 0},
            }
            chunk.metadata.language_specific = metadata
            chunks = [chunk]

        return chunks

    def _detect_security_issues_in_text(self, text: str) -> List[Dict[str, Any]]:
        """Detect security issues in plain text (not node)."""
        logger.info("[UNTESTED PATH] r._detect_security_issues_in_text called")
        issues = []

        # Split into lines for line-by-line analysis
        lines = text.split('\n')
        for i, line in enumerate(lines):
            # Check each security pattern
            for pattern_check in self._get_security_patterns():
                result = pattern_check(None, line)
                if result:
                    result['line'] = i + 1
                    issues.append(result)

        return issues

    def _get_language_name(self) -> str:
        """Return language identifier."""
        return 'r'

    def _get_import_node_types(self) -> List[str]:
        """Get node types that represent imports for r."""
        return ['library_call', 'source_call']

    def _is_comment_node(self, node) -> bool:
        """Check if node is a comment."""
        return node.type in ['comment', 'line_comment', 'block_comment']

    def _get_tree_sitter_language(self) -> Any:
        """Get R language for tree-sitter."""
        return get_language('r')

    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """
        R-specific node types to chunk.

        Tree-sitter R node types for functions, classes, and imports.
        """
        return {
            # Functions (including assigned functions)
            'function_definition': ChunkType.FUNCTION,
            'assignment': ChunkType.UNKNOWN,  # Will be refined in _create_chunk_from_node
            # S4 class system
            'call': ChunkType.UNKNOWN,  # Will check for setClass, setMethod
            # Imports - R uses function calls
            # library(), require(), source() handled in _create_chunk_from_node
        }

    def _create_chunk_from_node(
        self, node, lines: List[str], file_path: str, chunk_type: ChunkType, processed_ranges
    ):
        """Override to handle R-specific patterns and include roxygen docs."""
        # For functions, we need to ensure roxygen comments are included
        if (
            node.type == 'assignment'
            and self._check_function_assignment(node) == ChunkType.FUNCTION
        ):
            # Get the base chunk first
            chunk = super()._create_chunk_from_node(
                node, lines, file_path, ChunkType.FUNCTION, processed_ranges
            )

            if chunk and node.start_point[0] > 0:
                # Check for roxygen comments above the function
                start_line = node.start_point[0]
                roxygen_lines: list[str] = []

                # Look backwards for roxygen comments
                check_line = start_line - 1
                while check_line >= 0:
                    line = lines[check_line].strip()
                    if line.startswith("#'"):
                        roxygen_lines.insert(0, lines[check_line])
                        check_line -= 1
                        logger.info("[UNTESTED PATH] r roxygen check backwards")
                    elif (
                        line == ''
                        and check_line > 0
                        and lines[check_line - 1].strip().startswith("#'")
                    ):
                        # Empty line between roxygen blocks
                        roxygen_lines.insert(0, lines[check_line])
                        check_line -= 1
                        logger.info("[UNTESTED PATH] r roxygen check backwards")
                    else:
                        break

                # If we found roxygen comments, prepend them to the chunk
                if roxygen_lines:
                    logger.info("[UNTESTED PATH] r roxygen lines found")
                    roxygen_content = '\n'.join(roxygen_lines)
                    chunk.content = roxygen_content + '\n' + chunk.content
                    # Update start line
                    chunk.metadata.start_line = check_line + 2  # +1 for 0-based, +1 for next line

            return chunk

        # Handle assignment of functions (name <- function())
        if node.type == 'assignment':
            chunk_type = self._check_function_assignment(node)
            if chunk_type == ChunkType.UNKNOWN:
                return None  # Skip non-function assignments

        # Handle S4 class definitions and imports via calls
        elif node.type == 'call':
            function_name = self._get_call_name(node)
            logger.info("[UNTESTED PATH] r call node type check")

            # S4 class system
            if function_name == 'setClass':
                chunk_type = ChunkType.CLASS
            elif function_name in ['setMethod', 'setGeneric']:
                chunk_type = ChunkType.METHOD
            # Imports
            elif function_name in ['library', 'require', 'source']:
                # Group imports in base._extract_imports instead
                return None
            else:
                return None  # Skip other function calls

        # Standard processing
        chunk = super()._create_chunk_from_node(
            node, lines, file_path, chunk_type, processed_ranges
        )

        # Add R-specific metadata
        if chunk:
            metadata = {}

            # Base metadata by type
            if chunk.metadata.chunk_type == ChunkType.FUNCTION:
                metadata.update(self._extract_function_metadata(node))
            elif chunk.metadata.chunk_type == ChunkType.CLASS:
                metadata.update(self._extract_s4_class_metadata(node))

            # Add common enrichments
            metadata['complexity'] = self._calculate_complexity(node)
            metadata['todos'] = self._extract_todos(node)
            metadata['patterns'] = self._detect_patterns(node)
            metadata['security'] = self._detect_security_issues(node)

        # Quality metrics - use chunk content to check for roxygen
        if chunk:
            logger.info("[UNTESTED PATH] r adding quality metrics")
            quality = {
                'has_docstring': self._has_roxygen_docs_in_chunk(chunk.content),
                'uses_tidyverse': self._uses_tidyverse(node),
                'has_tests': 'test_' in file_path or 'test(' in chunk.content,
            }
            metadata['quality'] = quality

            chunk.metadata.language_specific = metadata

        return chunk

    def _has_roxygen_docs_in_chunk(self, content: str) -> bool:
        """Check for Roxygen documentation in chunk content."""
        # Roxygen uses #' at the beginning of lines
        lines = content.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#'"):
                return True
        return False

    def _check_function_assignment(self, assignment_node) -> ChunkType:
        """Check if assignment is a function definition."""
        logger.info("[UNTESTED PATH] r._check_function_assignment called")
        # R function assignment: name <- function(args) { body }
        for child in assignment_node.children:
            if child.type == 'function_definition':
                return ChunkType.FUNCTION
        return ChunkType.UNKNOWN

    def _get_call_name(self, call_node) -> str:
        """Extract function name from call node."""
        logger.info("[UNTESTED PATH] r._get_call_name called")
        for child in call_node.children:
            if child.type == 'identifier':
                return child.text.decode('utf8')
        return ""

    def _extract_node_name(self, node) -> Optional[str]:
        """Override to handle R's assignment syntax."""
        # For assignments, get the left side (first identifier before <- or =)
        if node.type == 'assignment':
            # R tree-sitter assignment structure:
            # assignment has children: [identifier, operator, expression]
            # We want the first child which should be the identifier
            if node.children and len(node.children) > 0:
                first_child = node.children[0]
                # The first child should be the identifier being assigned to
                if first_child.type == 'identifier':
                    return first_child.text.decode('utf8')
                # Sometimes the identifier might be nested
                elif first_child.children:
                    logger.info("[UNTESTED PATH] r nested identifier search")
                    for subchild in first_child.children:
                        if subchild.type == 'identifier':
                            return subchild.text.decode('utf8')

        # For setClass calls, extract class name from arguments
        elif node.type == 'call':
            function_name = self._get_call_name(node)
            if function_name == 'setClass':
                # First argument is usually the class name
                for child in node.children:
                    if child.type == 'argument_list':
                        for arg in child.children:
                            if arg.type == 'string':
                                return arg.text.decode('utf8').strip('"\'')

        return super()._extract_node_name(node)

    def _extract_function_metadata(self, node) -> Dict[str, Any]:
        """Extract R-specific function metadata."""
        metadata = {
            'parameters': [],
            'has_return': False,
            'uses_vectorization': False,
            'assignment_type': '<-',  # or '='
        }

        # Find the function definition
        func_def = None
        if node.type == 'assignment':
            for child in node.children:
                if child.type == 'function_definition':
                    func_def = child
                    break
                # Check for assignment operator in children
                for sub_child in child.children:
                    if sub_child.type in ['<-', '=', 'left_assignment', 'equals_assignment']:
                        metadata['assignment_type'] = sub_child.text.decode('utf8').strip()
                        break
                    elif sub_child.text.decode('utf8').strip() in ['<-', '=']:
                        metadata['assignment_type'] = sub_child.text.decode('utf8').strip()
                        break
        else:
            func_def = node

        if func_def:
            # Extract parameters
            for child in func_def.children:
                if child.type == 'parameters':
                    params = []
                    for param in child.children:
                        if param.type == 'identifier':
                            params.append(param.text.decode('utf8'))
                        elif param.type == 'default_parameter':
                            # parameter = default_value
                            logger.info("[UNTESTED PATH] r default parameter extraction")
                            for subchild in param.children:
                                if subchild.type == 'identifier':
                                    params.append(subchild.text.decode('utf8'))
                                    break
                    metadata['parameters'] = params

        # Check for return() calls
        def has_return(node):
            if node.type == 'call':
                name = self._get_call_name(node)
                if name == 'return':
                    return True
            for child in node.children:
                if has_return(child):
                    return True
            return False

        metadata['has_return'] = has_return(func_def or node)

        # Check for vectorization patterns (sapply, lapply, etc.)
        def uses_vectorization(node):
            if node.type == 'call':
                name = self._get_call_name(node)
                if name in ['sapply', 'lapply', 'vapply', 'mapply', 'apply']:
                    return True
            for child in node.children:
                if uses_vectorization(child):
                    return True
            return False

        metadata['uses_vectorization'] = uses_vectorization(func_def or node)

        return metadata

    def _extract_s4_class_metadata(self, class_node) -> Dict[str, Any]:
        """Extract S4 class metadata from setClass call."""
        metadata: Dict[str, Any] = {
            'slots': [],
            'contains': [],  # inheritance
            'methods': [],
        }

        # Parse setClass arguments
        # setClass("ClassName", slots = c(...), contains = "ParentClass")
        # This would require deeper AST analysis of the arguments
        # For now, keep it simple

        return metadata

    def _extract_dependencies_from_imports(self, import_nodes) -> List[str]:
        """Extract R-specific import dependencies."""
        deps = set()

        for node in import_nodes:
            if node.type == 'call':
                func_name = self._get_call_name(node)

                if func_name in ['library', 'require']:
                    # Extract package name from arguments
                    for child in node.children:
                        if child.type == 'argument_list':
                            for arg in child.children:
                                if arg.type == 'identifier':
                                    deps.add(arg.text.decode('utf8'))
                                elif arg.type == 'string':
                                    deps.add(arg.text.decode('utf8').strip('"\''))

                elif func_name == 'source':
                    # source() loads R files, track as dependency
                    for child in node.children:
                        if child.type == 'argument_list':
                            for arg in child.children:
                                if arg.type == 'string':
                                    file_path = arg.text.decode('utf8').strip('"\'')
                                    logger.info("[UNTESTED PATH] r source dependency found")
                                    deps.add(f"source:{file_path}")

        return sorted(list(deps))

    def _get_decision_node_types(self):
        """R-specific decision nodes."""
        logger.info("[UNTESTED PATH] r._get_decision_node_types called")
        # R tree-sitter uses different node types than other languages
        return {
            'if_statement',
            'if',  # R might use 'if' instead of 'if_statement'
            'else_clause',
            'else',  # R might use 'else' instead of 'else_clause'
            'while_statement',
            'while',  # R might use 'while' instead of 'while_statement'
            'for_statement',
            'for',  # R might use 'for' instead of 'for_statement'
            'repeat_statement',
            'repeat',  # R might use 'repeat' instead of 'repeat_statement'
            'switch_statement',
            'switch',  # R might use 'switch'
            'binary_operator',  # for conditions like x > 0
            'call',  # for function calls that might be decision points
        }

    def _has_roxygen_docs(self, node) -> bool:
        """Check for Roxygen documentation."""
        # Look for #' comments in the node content
        content = node.text.decode('utf8')
        return "#'" in content

    def _uses_tidyverse(self, node) -> bool:
        """Check if code uses tidyverse patterns."""
        content = node.text.decode('utf8')
        tidyverse_indicators = ['%>%', '|>', 'mutate(', 'filter(', 'select(', 'ggplot(']
        return any(indicator in content for indicator in tidyverse_indicators)

    def _detect_language_patterns(
        self, node, patterns: Dict[str, List[str]], metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[str]]:
        """Detect R-specific patterns."""
        content = node.text.decode('utf8')

        # Anti-patterns
        # Check for inefficient for loops over vectors (only in long functions)
        if 'for' in content and 'in' in content and len(content) > 200:
            patterns['anti'].append('for_loop_over_vector')
        if '<<-' in content:
            patterns['anti'].append('global_assignment')

        # Framework patterns
        if 'shiny' in content.lower():
            patterns['framework'].append('shiny')
        if 'plumber' in content.lower():
            patterns['framework'].append('plumber_api')

        return patterns

    def _get_security_patterns(self) -> List[Callable[[Any, str], Dict[str, Any]]]:
        """Override to add R-specific security patterns."""
        # Get base patterns but override hardcoded_credentials
        base_patterns = super()._get_security_patterns()
        # Remove the base hardcoded_credentials check
        filtered_patterns = [
            p for p in base_patterns if p.__name__ != '_check_hardcoded_credentials'
        ]
        # Add our custom patterns (cast para tipado)
        r_patterns: List[Callable[[Any, str], Dict[str, Any]]] = [
            self._check_hardcoded_credentials,  # type: ignore
            self._check_r_eval,  # type: ignore
            self._check_r_sql_injection,  # type: ignore
        ]
        filtered_patterns.extend(r_patterns)
        return filtered_patterns

    def _check_r_eval(self, node, text: str) -> Optional[Dict[str, Any]]:
        """Check for R eval() usage."""
        logger.info("[UNTESTED PATH] r._check_r_eval called")
        if 'eval(' in text and 'parse(' in text:
            return {
                'type': 'code_execution',
                'severity': 'high',
                'description': 'Potential code execution with eval(parse())',
            }
        return None

    def _check_r_sql_injection(self, node, text: str) -> Optional[Dict[str, Any]]:
        """Check for R-specific SQL injection patterns."""
        if 'paste0(' in text and any(
            sql in text.upper() for sql in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']
        ):
            return {
                'type': 'sql_injection_risk',
                'severity': 'high',
                'description': 'Potential SQL injection via paste0() concatenation',
            }
        return None

    def _check_hardcoded_credentials(self, node, text: str) -> Optional[Dict[str, Any]]:
        """Check for hardcoded passwords or API keys in R code."""
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
            'token',
        ]
        for pattern in credential_patterns:
            if pattern in text.lower() or pattern.upper() in text:
                if ('<-' in text or '=' in text) and any(quote in text for quote in ['"', "'"]):
                    if not any(
                        safe in text.lower() for safe in ['env', 'config', 'getenv', 'environ']
                    ):
                        return {
                            'type': 'hardcoded_credential',
                            'severity': 'critical',
                            'description': f'Possible hardcoded {pattern}',
                        }
        # Also use base implementation
        return super()._check_hardcoded_credentials(node, text) or None

    def _check_unsafe_deserialization(self, node, text: str) -> Optional[Dict[str, Any]]:
        """Override to exclude R's eval() which is handled separately."""
        if 'eval(' in text and 'parse(' in text:
            return None
        return super()._check_unsafe_deserialization(node, text) or None

    def _extract_imports(
        self, root_node, lines: List[str], file_path: str, processed_ranges
    ) -> List[Chunk]:
        """Override to handle R's import style."""
        import_chunks = []
        import_nodes = []

        # Find all library/require/source calls
        def find_imports(node):
            if node.type == 'call':
                func_name = self._get_call_name(node)
                if func_name in ['library', 'require', 'source']:
                    import_nodes.append(node)
            for child in node.children:
                find_imports(child)

        find_imports(root_node)

        if not import_nodes:
            return []

        # Group consecutive imports (R typically has them at the top)
        if import_nodes:
            first_import = import_nodes[0]
            last_import = import_nodes[-1]

            start_line = first_import.start_point[0]
            end_line = last_import.end_point[0]

            # Include any leading comments
            while start_line > 0 and (
                lines[start_line - 1].strip().startswith('#') or not lines[start_line - 1].strip()
            ):
                logger.info("[UNTESTED PATH] r including leading comments in imports")
                start_line -= 1

            content = '\n'.join(lines[start_line : end_line + 1])

            chunk = self._create_chunk(
                content=content,
                chunk_type=ChunkType.IMPORTS,
                file_path=file_path,
                start_line=start_line + 1,
                end_line=end_line + 1,
                name='imports',
            )

            # Mark as processed for imports
            if 'imports' not in processed_ranges:
                processed_ranges['imports'] = set()
            for line in range(start_line, end_line + 1):
                processed_ranges['imports'].add((line, line))

            import_chunks.append(chunk)

        return import_chunks
