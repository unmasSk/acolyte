"""
Lua chunker using tree-sitter-languages.
Handles Lua-specific constructs and patterns.
"""

from typing import Dict, List, Any, Optional
from tree_sitter_languages import get_language  # type: ignore

from acolyte.models.chunk import ChunkType
from acolyte.core.logging import logger
from acolyte.rag.chunking.base import BaseChunker


class LuaChunker(BaseChunker):
    """
    Lua-specific chunker using tree-sitter.

    Handles:
    - Functions (global and local)
    - Tables (as classes/modules)
    - Metatables and metamethods
    - Requires/modules
    """

    def _get_language_name(self) -> str:
        """Return language identifier."""
        return 'lua'

    def _get_import_node_types(self) -> List[str]:
        """Get node types that represent imports for lua."""
        return ['function_call']

    def _is_comment_node(self, node) -> bool:
        """Check if node is a comment."""
        return node.type in ['comment', 'line_comment', 'block_comment']

    def _get_tree_sitter_language(self) -> Any:
        """Get Lua language for tree-sitter."""
        logger.info("[UNTESTED PATH] lua._get_tree_sitter_language")
        return get_language('lua')

    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """
        Lua-specific node types to chunk.

        Tree-sitter Lua node types mapped to ChunkTypes.
        """
        return {
            # Functions
            'function_declaration': ChunkType.FUNCTION,
            'function_definition': ChunkType.FUNCTION,
            'function': ChunkType.FUNCTION,
            'local_function': ChunkType.FUNCTION,
            'method_definition': ChunkType.METHOD,
            # Tables (used as classes/modules in Lua)
            'table_constructor': ChunkType.CLASS,
            # Assignments that create module-like structures
            'assignment': ChunkType.MODULE,
            'local_declaration': ChunkType.MODULE,
            # Requires/imports
            'function_call': ChunkType.IMPORTS,  # Will filter for require()
        }

    def _create_chunk_from_node(
        self, node, lines: List[str], file_path: str, chunk_type: ChunkType, processed_ranges
    ):
        """Override to handle Lua-specific cases."""
        # Filter function calls - only keep require() as imports
        if node.type == 'function_call':
            call_text = node.text.decode('utf8')
            if 'require' not in call_text:
                logger.info("[UNTESTED PATH] lua non-require function call")
                return None

        # Handle table constructors that define modules/classes
        if node.type == 'table_constructor':
            # Check if this is a module-like table
            parent = node.parent
            if parent and parent.type == 'assignment':
                logger.info("[UNTESTED PATH] lua table constructor in assignment")
                # This is likely a module/class definition
                chunk_type = self._determine_table_type(parent)
                # Process the entire assignment, not just the table
                node = parent

        # Handle assignments that might be module definitions
        if node.type == 'assignment':
            if not self._is_module_assignment(node):
                logger.info("[UNTESTED PATH] lua non-module assignment")
                return None

        # Handle local declarations similarly
        if node.type == 'local_declaration':
            if not self._is_module_declaration(node):
                logger.info("[UNTESTED PATH] lua non-module local declaration")
                return None

        # Standard processing
        chunk = super()._create_chunk_from_node(
            node, lines, file_path, chunk_type, processed_ranges
        )

        # Add Lua-specific metadata
        if chunk:
            if chunk.metadata.chunk_type in [ChunkType.FUNCTION, ChunkType.METHOD]:
                logger.info("[UNTESTED PATH] lua function/method metadata")
                chunk.metadata.language_specific = self._extract_function_metadata(node)
            elif chunk.metadata.chunk_type == ChunkType.CLASS:
                logger.info("[UNTESTED PATH] lua class metadata")
                chunk.metadata.language_specific = self._extract_table_metadata(node)

        return chunk

    def _determine_table_type(self, assignment_node) -> ChunkType:
        """Determine if table is used as class, module, or namespace."""
        # Get the variable name being assigned
        var_name = self._get_assignment_target(assignment_node)

        if not var_name:
            logger.info("[UNTESTED PATH] lua no var name for table type")
            return ChunkType.UNKNOWN

        # Common patterns in Lua
        if var_name and (var_name[0].isupper() or var_name.endswith('Class')):
            return ChunkType.CLASS
        elif var_name and var_name.startswith('M') and len(var_name) > 1 and var_name[1].isupper():
            return ChunkType.MODULE  # Common pattern: M, Module, etc.
        else:
            return ChunkType.NAMESPACE

    def _is_module_assignment(self, node) -> bool:
        """Check if assignment creates a module-like structure."""
        # Look for patterns like: MyModule = {} or MyModule = { ... }
        for child in node.children:
            if child.type == 'table_constructor':
                return True
            # Also check for function assignments that create modules
            if child.type == 'function_definition':
                var_name = self._get_assignment_target(node)
                # Module functions often have uppercase names
                if var_name and (var_name[0].isupper() or '.' in var_name):
                    return True
        return False

    def _is_module_declaration(self, node) -> bool:
        """Check if local declaration creates a module."""
        # Look for: local M = {} or local MyModule = { ... }
        for child in node.children:
            if child.type == 'assignment_list':
                for assignment in child.children:
                    if assignment.type == 'table_constructor':
                        return True
        return False

    def _get_assignment_target(self, node) -> Optional[str]:
        """Extract the target variable name from an assignment."""
        for child in node.children:
            if child.type == 'variable_list':
                for var in child.children:
                    if var.type == 'identifier':
                        return var.text.decode('utf8')
                    elif var.type == 'field_expression':
                        # Handle Module.function assignments
                        return var.text.decode('utf8')
        return None

    def _extract_function_metadata(self, func_node) -> Dict[str, Any]:
        """Extract Lua-specific function metadata."""
        metadata = {
            'is_local': func_node.type == 'local_function',
            'parameters': [],
            'is_method': False,
            'is_metamethod': False,
        }

        # Get function name to check if it's a metamethod
        name = self._extract_node_name(func_node)
        if name and name.startswith('__'):
            metadata['is_metamethod'] = True

        # Check if it's a method (has self/this parameter or uses :)
        for child in func_node.children:
            if child.type == 'parameters':
                params = []
                has_self = False
                for param in child.children:
                    if param.type == 'identifier':
                        param_name = param.text.decode('utf8')
                        params.append(param_name)
                        if param_name in ['self', 'this']:
                            has_self = True
                metadata['parameters'] = params
                metadata['is_method'] = has_self

        # Also check for colon syntax (Class:method)
        parent = func_node.parent
        if parent and parent.type == 'field_expression':
            if ':' in parent.text.decode('utf8'):
                metadata['is_method'] = True

        return metadata

    def _extract_table_metadata(self, table_node) -> Dict[str, Any]:
        """Extract metadata from table definitions."""
        metadata: Dict[str, Any] = {
            'fields': [],
            'methods': [],
            'has_metatable': False,
            'is_class': False,
        }

        # Analyze table contents
        table_content = table_node
        if table_node.type == 'assignment':
            # Find the actual table constructor
            for child in table_node.children:
                if child.type == 'expression_list':
                    for expr in child.children:
                        if expr.type == 'table_constructor':
                            table_content = expr
                            break

        if table_content.type == 'table_constructor':
            for child in table_content.children:
                if child.type == 'field':
                    field_name = self._get_field_name(child)
                    if field_name:
                        # Check if it's a function
                        for field_child in child.children:
                            if field_child.type in ['function', 'function_definition']:
                                methods_list = metadata.get('methods', [])
                                if isinstance(methods_list, list):
                                    methods_list.append(field_name)
                                break
                        else:
                            fields_list = metadata.get('fields', [])
                            if isinstance(fields_list, list):
                                fields_list.append(field_name)

                        # Check for metatable
                        if field_name == '__index' or field_name.startswith('__'):
                            metadata['has_metatable'] = True

        # Heuristic: if has metatable or many methods, it's likely a class
        methods = metadata.get('methods', [])
        if metadata['has_metatable'] or (isinstance(methods, list) and len(methods) > 2):
            metadata['is_class'] = True

        return metadata

    def _get_field_name(self, field_node) -> Optional[str]:
        """Extract field name from a table field."""
        for child in field_node.children:
            if child.type == 'identifier':
                return child.text.decode('utf8')
            elif child.type == 'string':
                # Handle ["field"] syntax
                return child.text.decode('utf8').strip('"\'')
        return None

    def _extract_dependencies_from_imports(self, import_nodes) -> List[str]:
        """Extract Lua module dependencies from require() calls."""
        deps = set()

        for node in import_nodes:
            if node.type == 'function_call':
                call_text = node.text.decode('utf8')
                # Match require("module") or require('module')
                if 'require' in call_text:
                    # Simple extraction - could be improved with proper AST walking
                    import re

                    matches = re.findall(r'require\s*\(\s*["\']([^"\']+)["\']\s*\)', call_text)
                    for match in matches:
                        # Convert Lua module paths (dots) to root module
                        root_module = match.split('.')[0]
                        deps.add(root_module)

        return sorted(list(deps))
