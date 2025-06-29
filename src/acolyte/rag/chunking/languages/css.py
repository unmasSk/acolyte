"""
CSS/SCSS chunker using tree-sitter-languages.
Handles both CSS and SCSS (Sass) files with rich metadata extraction.
"""

import re
from typing import Dict, List, Any, Set, Optional, Tuple
from tree_sitter_languages import get_language

from acolyte.models.chunk import ChunkType, Chunk
from acolyte.core.logging import logger
from acolyte.rag.chunking.base import BaseChunker


class CssChunker(BaseChunker):
    """
    CSS/SCSS chunker using tree-sitter.

    Handles both CSS and SCSS files since SCSS is a superset of CSS.
    Extracts rich metadata for style analysis and optimization.
    """

    def _get_language_name(self) -> str:
        """Return language identifier."""
        return 'css'  # Both CSS and SCSS use 'css' in tree-sitter-languages

    def _get_import_node_types(self) -> List[str]:
        """Get node types that represent imports for css."""
        return ['import_statement', 'use_statement', 'forward_statement']

    def _is_comment_node(self, node: Any) -> bool:
        """Check if node is a comment."""
        return node.type in ['comment', 'line_comment', 'block_comment']

    def _get_tree_sitter_language(self) -> Any:
        """Get CSS language for tree-sitter."""
        # tree-sitter-languages uses 'css' for both CSS and SCSS
        logger.info("[UNTESTED PATH] css._get_tree_sitter_language called")
        return get_language('css')

    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """
        CSS/SCSS node types to chunk.

        Note: CSS is mostly flat, so we chunk by rule sets and at-rules.
        """
        return {
            # Main structures
            'rule_set': ChunkType.UNKNOWN,  # Will refine based on content
            'at_rule': ChunkType.UNKNOWN,  # @media, @import, @keyframes, etc.
            # SCSS specific
            'mixin_statement': ChunkType.FUNCTION,  # @mixin in SCSS
            'include_statement': ChunkType.UNKNOWN,  # @include in SCSS
            'function_statement': ChunkType.FUNCTION,  # @function in SCSS
            # Imports
            'import_statement': ChunkType.IMPORTS,
            'use_statement': ChunkType.IMPORTS,  # SCSS @use
            'forward_statement': ChunkType.IMPORTS,  # SCSS @forward
        }

    def _create_chunk_from_node(
        self,
        node: Any,
        lines: List[str],
        file_path: str,
        chunk_type: ChunkType,
        processed_ranges: Dict[str, Set[Tuple[int, int]]],
    ) -> Optional[Chunk]:
        """Override to handle CSS-specific cases and extract metadata."""
        # Refine chunk type based on content
        if node.type == 'rule_set':
            chunk_type = self._determine_rule_type(node)
        elif node.type == 'at_rule':
            chunk_type = self._determine_at_rule_type(node)
            if chunk_type == ChunkType.UNKNOWN:
                logger.info("[UNTESTED PATH] css unknown at_rule type")

        # Standard processing
        chunk = super()._create_chunk_from_node(
            node, lines, file_path, chunk_type, processed_ranges
        )

        if not chunk:
            return None

        # Extract CSS-specific metadata
        metadata = self._extract_css_metadata(node, lines)
        chunk.metadata.language_specific = metadata

        # Set name based on what we found
        if metadata.get('selectors'):
            chunk.metadata.name = ', '.join(metadata['selectors'][:3])
            if len(metadata['selectors']) > 3:
                chunk.metadata.name += f' (+{len(metadata["selectors"]) - 3} more)'
        elif metadata.get('at_rule'):
            chunk.metadata.name = metadata['at_rule']
        elif metadata.get('mixin_name'):
            chunk.metadata.name = f"@mixin {metadata['mixin_name']}"
        elif metadata.get('function_name'):
            chunk.metadata.name = f"@function {metadata['function_name']}"

        return chunk

    def _determine_rule_type(self, node: Any) -> ChunkType:
        """Determine the type of CSS rule."""
        # For now, treat all rule sets as UNKNOWN
        # Could be enhanced to detect component styles, utility classes, etc.
        return ChunkType.UNKNOWN

    def _determine_at_rule_type(self, node: Any) -> ChunkType:
        """Determine the type of @rule."""
        # Get the @ keyword
        for child in node.children:
            if child.type == 'at_keyword':
                keyword = child.text.decode('utf8').lower()
                if keyword in ['@import', '@use', '@forward']:
                    return ChunkType.IMPORTS
                elif keyword in ['@mixin', '@function']:
                    return ChunkType.FUNCTION
                # @media, @keyframes, @font-face, etc.
                break

        return ChunkType.UNKNOWN

    def _extract_css_metadata(self, node: Any, lines: List[str]) -> Dict[str, Any]:
        """Extract CSS-specific metadata."""
        metadata: Dict[str, Any] = {}
        selectors: List[str] = []  # Initialize selectors

        if node.type == 'rule_set':
            # Extract selectors
            selectors = self._extract_selectors(node)
            if selectors:
                metadata['selectors'] = selectors
                metadata['selector_types'] = self._classify_selectors(selectors)
                metadata['specificity'] = self._calculate_max_specificity(selectors)

            # Extract properties
            properties = self._extract_properties(node)
            if properties:
                metadata['properties'] = list(properties.keys())
                metadata['property_count'] = len(properties)

                # Detect patterns
                patterns = self._detect_css_patterns(properties, selectors)
                if patterns:
                    metadata['patterns'] = patterns

                # Extract variables used
                variables = self._extract_variables_used(properties)
                if variables:
                    metadata['variables_used'] = variables
            else:
                logger.info("[UNTESTED PATH] css no patterns detected")

        elif node.type == 'at_rule':
            at_rule = self._extract_at_rule_info(node)
            metadata.update(at_rule)

        elif node.type == 'mixin_statement':
            mixin_info = self._extract_mixin_info(node)
            metadata.update(mixin_info)

        elif node.type == 'function_statement':
            func_info = self._extract_function_info(node)
            metadata.update(func_info)

        # Calculate complexity
        complexity = self._calculate_css_complexity(node)
        if complexity:
            metadata['complexity'] = complexity

        # Extract TODOs using mixin
        todos = self._extract_todos(node)
        if todos:
            metadata['todos'] = todos

        return metadata

    def _extract_selectors(self, node: Any) -> List[str]:
        """Extract all selectors from a rule set."""
        selectors = []

        for child in node.children:
            if child.type == 'selectors':
                # Each selector in the comma-separated list
                selector_text = child.text.decode('utf8').strip()
                # Split by comma but preserve complex selectors
                parts = [s.strip() for s in selector_text.split(',')]
                selectors.extend(parts)
                break

        return selectors

    def _classify_selectors(self, selectors: List[str]) -> Dict[str, int]:
        """Classify selectors by type."""
        types = {
            'id': 0,  # #id
            'class': 0,  # .class
            'element': 0,  # div, p, etc.
            'attribute': 0,  # [attr]
            'pseudo': 0,  # :hover, ::before
            'complex': 0,  # combinators
        }

        for selector in selectors:
            if '#' in selector:
                types['id'] += 1
            if '.' in selector:
                types['class'] += 1
            if '[' in selector and ']' in selector:
                types['attribute'] += 1
            if ':' in selector:
                types['pseudo'] += 1
            if any(c in selector for c in ['>', '+', '~', ' ']):
                types['complex'] += 1
            # Simple element selector
            if not any(c in selector for c in ['#', '.', '[', ':', '>', '+', '~']):
                types['element'] += 1

        return {k: v for k, v in types.items() if v > 0}

    def _calculate_max_specificity(self, selectors: List[str]) -> List[int]:
        """
        Calculate CSS specificity for the most specific selector.
        Returns [id_count, class_count, element_count].
        """
        max_spec = [0, 0, 0]

        for selector in selectors:
            spec = [0, 0, 0]

            # Count IDs
            spec[0] = selector.count('#')

            # Count classes, attributes, and pseudo-classes (but not pseudo-elements)
            spec[1] = (
                selector.count('.')
                + selector.count('[')
                + len([1 for _ in re.finditer(r':(?!:)', selector)])  # Single : not followed by :
            )

            # Count elements and pseudo-elements
            # Count actual element names
            # Element selectors are identifiers **not** preceded by ':' / '::'
            element_tokens = re.findall(r'(?<![:.#\[\]>+~\s])\b([a-zA-Z][-a-zA-Z0-9]*)\b', selector)
            element_count = len(element_tokens)
            pseudo_element_count = len(re.findall(r'::[a-zA-Z-]+', selector))
            spec[2] = element_count + pseudo_element_count

            # Update max if this is more specific
            if spec > max_spec:
                max_spec = spec

        return max_spec

    def _extract_properties(self, node: Any) -> Dict[str, str]:
        """Extract CSS properties and their values."""
        properties = {}

        # Find declaration block
        for child in node.children:
            if child.type == 'block':
                # Each declaration in the block
                for declaration in child.children:
                    if declaration.type == 'declaration':
                        prop_name = None
                        prop_value = None

                        for part in declaration.children:
                            if part.type == 'property_name':
                                prop_name = part.text.decode('utf8').strip()
                            elif part.type == ':':
                                # Find the next sibling which should be the value
                                idx = declaration.children.index(part)
                                if idx + 1 < len(declaration.children):
                                    value_node = declaration.children[idx + 1]
                                    prop_value = value_node.text.decode('utf8').strip()

                        if prop_name and prop_value:
                            properties[prop_name] = prop_value

        return properties

    def _detect_css_patterns(
        self, properties: Dict[str, str], selectors: List[str]
    ) -> Dict[str, List[str]]:
        """Detect CSS patterns and potential issues."""
        patterns: Dict[str, List[str]] = {
            'anti': [],
            'performance': [],
            'maintenance': [],
        }

        # Check for !important abuse
        important_count = sum(1 for v in properties.values() if '!important' in v)
        if important_count > 2:
            patterns['anti'].append('important_overuse')

        # Check for overly specific selectors
        for selector in selectors:
            depth = selector.count(' ') + selector.count('>') + selector.count('+')
            if depth > 3:
                patterns['anti'].append('deep_nesting')
                break

        # Check for hardcoded colors (not using variables)
        color_pattern = re.compile(r'#[0-9a-fA-F]{3,6}|rgb|hsl')
        hardcoded_colors = sum(1 for v in properties.values() if color_pattern.search(v))
        if hardcoded_colors > 3:
            patterns['maintenance'].append('hardcoded_colors')

        # Check for vendor prefixes
        if any(p.startswith('-webkit-') or p.startswith('-moz-') for p in properties):
            patterns['maintenance'].append('vendor_prefixes')

        # Performance patterns
        if any(p in properties for p in ['box-shadow', 'filter', 'transform']):
            patterns['performance'].append('expensive_properties')

        return {k: v for k, v in patterns.items() if v}

    def _extract_variables_used(self, properties: Dict[str, str]) -> List[str]:
        """Extract CSS/SCSS variables used in properties."""
        variables: Set[str] = set()

        for value in properties.values():
            # CSS custom properties
            css_vars = re.findall(r'var\(--([^)]+)\)', value)
            variables.update(f'--{var}' for var in css_vars)

            # SCSS variables
            scss_vars = re.findall(r'\$[\w-]+', value)
            variables.update(scss_vars)

        return sorted(list(variables))

    def _extract_at_rule_info(self, node: Any) -> Dict[str, Any]:
        """Extract information about @rules."""
        info = {}

        # In CSS tree-sitter, at-rules start with the @ symbol
        # The first token is usually the rule name
        if node.text:
            text = node.text.decode('utf8').strip()
            # Extract @rule name
            match = re.match(r'(@\w+)', text)
            if match:
                info['at_rule'] = match.group(1)

        # Extract specific info based on type
        if info.get('at_rule') == '@media':
            # Extract media query
            for child in node.children:
                if child.type == 'query_list':
                    info['media_query'] = child.text.decode('utf8').strip()
                    break

        elif info.get('at_rule') == '@keyframes':
            # Extract animation name
            for child in node.children:
                if child.type == 'keyframes_name':
                    info['animation_name'] = child.text.decode('utf8').strip()
                    break

        return info

    def _extract_mixin_info(self, node: Any) -> Dict[str, Any]:
        """Extract SCSS mixin information."""
        info: Dict[str, Any] = {'is_mixin': True}
        # Extract mixin name and parameters
        for child in node.children:
            if child.type == 'name':
                info['mixin_name'] = child.text.decode('utf8').strip()
            elif child.type == 'parameters':
                params = []
                param_text = child.text.decode('utf8').strip('()')
                if param_text:
                    params = [p.strip() for p in param_text.split(',')]
                info['parameters'] = params
        return info

    def _extract_function_info(self, node: Any) -> Dict[str, Any]:
        """Extract SCSS function information."""
        info: Dict[str, Any] = {'is_function': True}
        # Extract function name and parameters
        for child in node.children:
            if child.type == 'name':
                info['function_name'] = child.text.decode('utf8').strip()
            elif child.type == 'parameters':
                params = []
                param_text = child.text.decode('utf8').strip('()')
                if param_text:
                    params = [p.strip() for p in param_text.split(',')]
                info['parameters'] = params
        return info

    def _calculate_css_complexity(self, node: Any) -> Dict[str, Any]:
        """Calculate CSS-specific complexity metrics."""
        complexity = {}

        # Count nested rules (SCSS)
        nesting_depth = self._calculate_nesting_depth(node)
        if nesting_depth > 0:
            complexity['nesting_depth'] = nesting_depth

        # Count declarations
        declaration_count = self._count_declarations(node)
        if declaration_count > 0:
            complexity['declarations'] = declaration_count

        # Lines of code
        if hasattr(node, 'end_point') and hasattr(node, 'start_point'):
            loc = node.end_point[0] - node.start_point[0] + 1
            complexity['lines_of_code'] = loc

        return complexity

    def _calculate_nesting_depth(self, node: Any, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth in SCSS."""
        max_depth = current_depth

        # Look for nested rule sets
        for child in node.children:
            if child.type == 'block':
                for block_child in child.children:
                    if block_child.type == 'rule_set':
                        child_depth = self._calculate_nesting_depth(block_child, current_depth + 1)
                        max_depth = max(max_depth, child_depth)

        return max_depth

    def _count_declarations(self, node: Any) -> int:
        """Count CSS declarations in a node."""
        count = 0

        def count_in_node(n: Any) -> None:
            nonlocal count
            if n.type == 'declaration':
                count += 1
            for child in n.children:
                count_in_node(child)

        count_in_node(node)
        return count

    def _extract_dependencies_from_imports(self, import_nodes: List[Any]) -> List[str]:
        """Extract dependency names from CSS import nodes."""
        deps = set()

        for node in import_nodes:
            text = node.text.decode('utf8')

            # CSS @import patterns
            if '@import' in text:
                # Extract URL or file path
                # @import url("style.css");
                # @import "style.css";
                # @import 'style.css';
                import_match = re.search(r'@import\s+(?:url\()?["\']([^"\')]+)["\']', text)
                if import_match:
                    deps.add(import_match.group(1))
                else:
                    logger.info("[UNTESTED PATH] css @import pattern not matched")

            # SCSS @use and @forward
            elif '@use' in text or '@forward' in text:
                # @use 'sass:math';
                # @forward 'src/list';
                use_match = re.search(r'@(?:use|forward)\s+["\']([^"\')]+)["\']', text)
                if use_match:
                    deps.add(use_match.group(1))

        return list(deps)
