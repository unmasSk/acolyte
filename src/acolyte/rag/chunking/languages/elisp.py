"""
Emacs Lisp chunker using tree-sitter-languages.
Handles elisp-specific constructs like defun, defvar, defmacro, etc.
"""

from typing import Dict, List, Any, Optional
from tree_sitter_languages import get_language

from acolyte.models.chunk import Chunk, ChunkType
from acolyte.core.logging import logger
from acolyte.rag.chunking.base import BaseChunker


class ElispChunker(BaseChunker):
    """
    Emacs Lisp specific chunker using tree-sitter.

    Handles:
    - Functions (defun, defmacro, defsubst)
    - Variables (defvar, defcustom, defconst)
    - Classes (defclass for EIEIO)
    - Requires and loads
    - Section comments (;;;)
    """

    def _get_language_name(self) -> str:
        """Return language identifier."""
        return 'elisp'

    def _get_import_node_types(self) -> List[str]:
        """Get node types that represent imports for elisp."""
        return ['require', 'load']

    def _is_comment_node(self, node: Any) -> bool:
        """Check if node is a comment."""
        return node.type in ['comment', 'line_comment', 'block_comment']

    def _get_tree_sitter_language(self) -> Any:
        """Get Emacs Lisp language for tree-sitter."""
        logger.info("[UNTESTED PATH] elisp._get_tree_sitter_language called")
        return get_language('elisp')

    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """
        Elisp-specific node types to chunk.

        Tree-sitter elisp uses generic 'list' nodes,
        so we need to check the first element.
        """
        # Elisp is different - most things are 'list' nodes
        # We'll override _create_chunk_from_node to handle this
        return {
            'list': ChunkType.UNKNOWN,  # Will be refined
        }

    def _create_chunk_from_node(
        self,
        node: Any,
        lines: List[str],
        file_path: str,
        chunk_type: ChunkType,
        processed_ranges: Dict[str, Any],
    ) -> Optional[Chunk]:
        """Override to handle elisp's list-based structure."""
        if node.type == 'list':
            # Check first element to determine actual type
            chunk_type = self._determine_list_type(node)
            if chunk_type == ChunkType.UNKNOWN:
                # Not a definition we care about
                return None

        # Standard processing
        chunk = super()._create_chunk_from_node(
            node, lines, file_path, chunk_type, processed_ranges
        )

        # Add elisp-specific metadata
        if chunk:
            if chunk.metadata.chunk_type in [ChunkType.FUNCTION, ChunkType.METHOD]:
                chunk.metadata.language_specific = self._extract_function_metadata(node)
            elif chunk.metadata.chunk_type == ChunkType.CLASS:
                chunk.metadata.language_specific = self._extract_class_metadata(node)
            elif chunk.metadata.chunk_type == ChunkType.CONSTANTS:
                chunk.metadata.language_specific = self._extract_variable_metadata(node)

        return chunk

    def _determine_list_type(self, list_node: Any) -> ChunkType:
        """Determine the type of elisp definition from list node."""
        if not list_node.children:
            return ChunkType.UNKNOWN

        # Get the first symbol (function name)
        # Skip opening paren and whitespace
        first_child = None
        for child in list_node.children:
            if child.type == 'symbol':
                first_child = child.text.decode('utf8')
                break
            elif child.type not in ['(', ')', ' ', '\n', '\t']:
                # Non-symbol, non-whitespace found first
                break

        if not first_child:
            return ChunkType.UNKNOWN

        # Map elisp forms to chunk types
        type_map = {
            # Functions
            'defun': ChunkType.FUNCTION,
            'defmacro': ChunkType.FUNCTION,
            'defsubst': ChunkType.FUNCTION,
            'cl-defun': ChunkType.FUNCTION,
            'cl-defmacro': ChunkType.FUNCTION,
            'cl-defmethod': ChunkType.METHOD,
            'defgeneric': ChunkType.FUNCTION,
            # Variables
            'defvar': ChunkType.CONSTANTS,
            'defcustom': ChunkType.CONSTANTS,
            'defconst': ChunkType.CONSTANTS,
            'defparameter': ChunkType.CONSTANTS,
            # Classes (EIEIO)
            'defclass': ChunkType.CLASS,
            'cl-defstruct': ChunkType.CLASS,
            # Imports
            'require': ChunkType.IMPORTS,
            'load': ChunkType.IMPORTS,
            'load-file': ChunkType.IMPORTS,
            'autoload': ChunkType.IMPORTS,
            # Tests
            'ert-deftest': ChunkType.TESTS,
        }

        return type_map.get(first_child, ChunkType.UNKNOWN)

    def _extract_node_name(self, node: Any) -> Optional[str]:
        """Extract name from elisp definition."""
        if node.type != 'list':
            return super()._extract_node_name(node)

        # For lists, the name is usually the second symbol
        symbols = []
        for child in node.children:
            if child.type == 'symbol':
                symbols.append(child.text.decode('utf8'))

        # Return the second symbol (first is defun/defvar/etc)
        if len(symbols) >= 2:
            return symbols[1]

        logger.info("[UNTESTED PATH] elisp not enough symbols for name")
        return None

    def _extract_function_metadata(self, func_node: Any) -> Dict[str, Any]:
        """Extract elisp-specific function metadata."""
        metadata: Dict[str, Any] = {
            'type': 'unknown',
            'parameters': [],
            'has_docstring': False,
            'is_interactive': False,
            'is_macro': False,
        }

        # Get function type
        for child in func_node.children:
            if child.type == 'symbol':
                func_type = child.text.decode('utf8')
                metadata['type'] = func_type
                metadata['is_macro'] = 'macro' in func_type
                break

        # Extract parameters from parameter list
        param_list_found = False
        for i, child in enumerate(func_node.children):
            if param_list_found and child.type == 'list':
                # This is the parameter list
                params = []
                for param_child in child.children:
                    if param_child.type == 'symbol':
                        param_name = param_child.text.decode('utf8')
                        if not param_name.startswith('&'):  # Skip &optional, &rest
                            params.append(param_name)
                metadata['parameters'] = params
                break
            elif child.type == 'symbol' and i == 1:
                # Found function name, next list is parameters
                param_list_found = True

        # Check for docstring and interactive
        strings_found = 0
        for child in func_node.children:
            if child.type == 'string':
                strings_found += 1
                if strings_found == 1:
                    metadata['has_docstring'] = True
            elif child.type == 'list':
                # Check if it's (interactive)
                for subchild in child.children:
                    if subchild.type == 'symbol' and subchild.text.decode('utf8') == 'interactive':
                        logger.info("[UNTESTED PATH] elisp interactive function")
                        metadata['is_interactive'] = True
                        break

        return metadata

    def _extract_class_metadata(self, class_node: Any) -> Dict[str, Any]:
        """Extract elisp class metadata (EIEIO)."""
        metadata: Dict[str, Any] = {
            'type': 'defclass',
            'superclasses': [],
            'slots': [],
            'has_docstring': False,
        }

        # EIEIO classes have format:
        # (defclass name (superclasses) ((slot1) (slot2)) "docstring")

        found_name = False
        found_supers = False

        for child in class_node.children:
            if child.type == 'symbol':
                if not found_name and child.text.decode('utf8') != 'defclass':
                    found_name = True
            elif child.type == 'list' and not found_supers:
                # Superclasses list
                found_supers = True
                for super_child in child.children:
                    if super_child.type == 'symbol':
                        metadata['superclasses'].append(super_child.text.decode('utf8'))
            elif child.type == 'list' and found_supers:
                # Slots list
                for slot_child in child.children:
                    if slot_child.type == 'list':
                        # Each slot is a list, get first symbol
                        for slot_elem in slot_child.children:
                            if slot_elem.type == 'symbol':
                                metadata['slots'].append(slot_elem.text.decode('utf8'))
                                break
            elif child.type == 'string':
                metadata['has_docstring'] = True

        return metadata

    def _extract_variable_metadata(self, var_node: Any) -> Dict[str, Any]:
        """Extract elisp variable metadata."""
        metadata: Dict[str, Any] = {
            'type': 'unknown',
            'has_docstring': False,
            'is_customizable': False,
            'is_constant': False,
        }

        # Get variable type
        for child in var_node.children:
            if child.type == 'symbol':
                var_type = child.text.decode('utf8')
                metadata['type'] = var_type
                metadata['is_customizable'] = var_type == 'defcustom'
                metadata['is_constant'] = var_type in ['defconst', 'defparameter']
                break

        # Check for docstring
        for child in var_node.children:
            if child.type == 'string':
                logger.info("[UNTESTED PATH] elisp variable has docstring")
                metadata['has_docstring'] = True
                break

        return metadata

    def _extract_imports(
        self, root_node: Any, lines: List[str], file_path: str, processed_ranges: Dict[str, Any]
    ) -> List[Chunk]:
        """Override to handle elisp require/load forms."""
        import_chunks: List[Chunk] = []
        import_nodes: List[Any] = []

        # Find all import-like nodes
        def find_imports(node: Any, depth: int = 0) -> None:
            logger.info("[UNTESTED PATH] elisp find_imports function")
            if node.type == 'list' and depth < 2:  # Only check top-level and immediate children
                chunk_type = self._determine_list_type(node)
                if chunk_type == ChunkType.IMPORTS:
                    import_nodes.append(node)
            for child in node.children:
                find_imports(child, depth + 1)

        find_imports(root_node)

        if not import_nodes:
            logger.info("[UNTESTED PATH] elisp no import nodes found")
            return []

        # Group imports that are close together
        # (Different from BaseChunker because elisp imports can be scattered)
        for node in import_nodes:
            start_line = node.start_point[0]
            end_line = node.end_point[0]

            # Check if already processed
            already_done = any(
                s <= start_line <= e for (s, e) in processed_ranges.get('imports', set())
            )
            if already_done:
                logger.info("[UNTESTED PATH] elisp import already processed")
                continue

            content = '\n'.join(lines[start_line : end_line + 1])

            chunk = self._create_chunk(
                content=content,
                chunk_type=ChunkType.IMPORTS,
                file_path=file_path,
                start_line=start_line + 1,
                end_line=end_line + 1,
                name='require',
            )

            # Mark as processed using the base class method
            if 'imports' not in processed_ranges:
                processed_ranges['imports'] = set()
            processed_ranges['imports'].add((start_line, end_line))
            import_chunks.append(chunk)

        return import_chunks

    def _extract_dependencies_from_imports(self, import_nodes: List[Any]) -> List[str]:
        """Extract elisp dependencies from require/load nodes."""
        deps = set()

        for node in import_nodes:
            # Look for (require 'package-name)
            found_require = False
            for child in node.children:
                if child.type == 'symbol' and child.text.decode('utf8') in [
                    'require',
                    'load',
                    'load-file',
                ]:
                    found_require = True
                elif found_require and child.type == 'quote':
                    # Next child should be the package symbol
                    for quote_child in child.children:
                        if quote_child.type == 'symbol':
                            deps.add(quote_child.text.decode('utf8'))
                            break
                    found_require = False

        return sorted(list(deps))
