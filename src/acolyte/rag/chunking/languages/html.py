"""
HTML chunker using tree-sitter-languages.
Specialized for markup structure and semantic HTML understanding.
"""

from typing import Dict, List, Any
from tree_sitter_languages import get_language

from acolyte.models.chunk import ChunkType
from acolyte.core.logging import logger
from acolyte.rag.chunking.base import BaseChunker


class HtmlChunker(BaseChunker):
    """
    HTML-specific chunker using tree-sitter.

    Focuses on semantic structure, components, and metadata extraction
    rather than traditional code constructs.
    """

    def _get_language_name(self) -> str:
        """Return language identifier."""
        return 'html'

    def _get_import_node_types(self) -> List[str]:
        """Get node types that represent imports for html."""
        return ['link_element', 'script_element']

    def _is_comment_node(self, node) -> bool:
        """Check if a node is a comment."""
        logger.info("[UNTESTED PATH] html._is_comment_node")
        return node.type == 'comment'

    def _get_tree_sitter_language(self) -> Any:
        """Get HTML language for tree-sitter."""
        logger.info("[UNTESTED PATH] html._get_tree_sitter_language")
        return get_language('html')

    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """
        HTML-specific node types to chunk.

        Tree-sitter HTML node types focus on document structure.
        """
        return {
            # Document structure
            'element': ChunkType.UNKNOWN,  # Will be refined based on tag
            'script_element': ChunkType.FUNCTION,  # Inline scripts
            'style_element': ChunkType.TYPES,  # Inline styles
            # Special handling for these will be in _create_chunk_from_node
            'doctype': ChunkType.TYPES,
            'comment': ChunkType.COMMENT,
        }

    def _create_chunk_from_node(
        self, node, lines: List[str], file_path: str, chunk_type: ChunkType, processed_ranges
    ):
        """Override to handle HTML-specific cases."""
        # For regular elements, determine chunk type by tag name
        if node.type == 'element':
            tag_name = self._get_tag_name(node)
            chunk_type = self._determine_element_type(tag_name)

            # Skip if not a significant element
            if chunk_type == ChunkType.UNKNOWN and not self._is_significant_element(node, tag_name):
                return None

        # Standard processing
        chunk = super()._create_chunk_from_node(
            node, lines, file_path, chunk_type, processed_ranges
        )

        # Add HTML-specific metadata
        if chunk:
            chunk.metadata.language_specific = self._extract_element_metadata(node)

            # Extract TODOs from comments
            if node.type == 'comment':
                chunk.metadata.language_specific['todos'] = self._extract_html_todos(node, lines)

            # Override name with something more meaningful for HTML
            if node.type == 'element':
                tag_name = self._get_tag_name(node)
                element_id = self._get_attribute_value(node, 'id')
                element_class = self._get_attribute_value(node, 'class')

                # Create meaningful name
                name_parts = [tag_name]
                if element_id:
                    name_parts.append(f"#{element_id}")
                elif element_class:
                    # Use first class as identifier
                    first_class = element_class.split()[0]
                    name_parts.append(f".{first_class}")

                chunk.metadata.name = ' '.join(name_parts)

        return chunk

    def _get_tag_name(self, element_node) -> str:
        """Extract tag name from element node."""
        for child in element_node.children:
            if child.type in ['start_tag', 'self_closing_tag']:
                for tag_child in child.children:
                    if tag_child.type == 'tag_name':
                        logger.info("[UNTESTED PATH] html tag_name extraction")
                        return tag_child.text.decode('utf8').lower()
        return 'unknown'

    def _determine_element_type(self, tag_name: str) -> ChunkType:
        """Determine ChunkType based on HTML tag name."""
        # Semantic sections
        if tag_name in ['header', 'nav', 'main', 'footer', 'article', 'section', 'aside']:
            return ChunkType.MODULE

        # Forms and interactive
        elif tag_name in ['form', 'dialog']:
            return ChunkType.INTERFACE

        # Templates and components
        elif tag_name in ['template', 'slot'] or tag_name.startswith('x-') or '-' in tag_name:
            logger.info("[UNTESTED PATH] html template/component detected")
            return ChunkType.CLASS  # Web components

        # Scripts and styles (handled separately)
        elif tag_name == 'script':
            logger.info("[UNTESTED PATH] html script tag")
            return ChunkType.FUNCTION
        elif tag_name == 'style':
            return ChunkType.TYPES

        # Documentation
        elif tag_name in ['head', 'meta', 'title']:
            logger.info("[UNTESTED PATH] html documentation tag")
            return ChunkType.DOCSTRING

        return ChunkType.UNKNOWN

    def _is_significant_element(self, node, tag_name: str) -> bool:
        """Determine if an element is significant enough to chunk."""
        # Always chunk semantic elements
        if tag_name in ['header', 'nav', 'main', 'footer', 'article', 'section', 'aside', 'form']:
            return True

        # Check for significant attributes
        if self._get_attribute_value(node, 'id') or self._get_attribute_value(
            node, 'data-component'
        ):
            return True

        # Check for significant classes that might indicate components
        classes = self._get_attribute_value(node, 'class') or ''
        component_indicators = ['component', 'widget', 'module', 'container', 'wrapper']
        if any(indicator in classes.lower() for indicator in component_indicators):
            return True

        # Check if it has substantial content (more than 10 lines)
        start_line = node.start_point[0]
        end_line = node.end_point[0]
        if end_line - start_line > 10:
            return True

        return False

    def _get_attribute_value(self, element_node, attr_name: str) -> str:
        """Extract attribute value from element."""
        for child in element_node.children:
            if child.type in ['start_tag', 'self_closing_tag']:
                for tag_child in child.children:
                    if tag_child.type == 'attribute':
                        attr_name_node = None
                        attr_value_node = None

                        for attr_child in tag_child.children:
                            if attr_child.type == 'attribute_name':
                                attr_name_node = attr_child
                            elif (
                                attr_child.type == 'attribute_value'
                                or attr_child.type == 'quoted_attribute_value'
                            ):
                                attr_value_node = attr_child

                        if attr_name_node and attr_name_node.text.decode('utf8') == attr_name:
                            if attr_value_node:
                                value = attr_value_node.text.decode('utf8')
                                # Remove quotes if present
                                if value.startswith('"') and value.endswith('"'):
                                    logger.info("[UNTESTED PATH] html double-quoted attribute")
                                    value = value[1:-1]
                                elif value.startswith("'") and value.endswith("'"):
                                    logger.info("[UNTESTED PATH] html single-quoted attribute")
                                    value = value[1:-1]
                                return value
        return ''

    def _extract_html_todos(self, comment_node, lines: List[str]) -> List[Dict[str, Any]]:
        """Extract TODOs from HTML comments."""
        todos = []
        comment_text = comment_node.text.decode('utf8')
        # Remove comment markers
        import re

        # Remove HTML comment delimiters exactly once
        comment_text = re.sub(r'^<!--\s*|\s*-->', '', comment_text).strip()

        # Check for TODO patterns
        for pattern in ['TODO', 'FIXME', 'HACK', 'BUG', 'OPTIMIZE', 'NOTE']:
            if pattern in comment_text.upper():
                import re

                match = re.search(f'{pattern}[:\\s]*(.*)', comment_text, re.IGNORECASE)
                if match:
                    todos.append(
                        {
                            'type': pattern,
                            'text': match.group(1).strip(),
                            'line': comment_node.start_point[0] + 1,  # Convert to 1-based
                        }
                    )
                    break

        return todos

    def _extract_element_metadata(self, node) -> Dict[str, Any]:
        """Extract HTML-specific metadata."""
        metadata: Dict[str, Any] = {
            'tag_name': self._get_tag_name(node) if node.type == 'element' else node.type,
            'attributes': {},
            'semantic_role': None,
            'accessibility': {},
            'seo': {},
            'resources': [],
            'scripts': [],
            'forms': [],
            'todos': [],
            'security': [],
            'quality': {
                'has_semantic_structure': False,
                'has_accessibility': False,
                'has_meta_description': False,
            },
        }

        if node.type != 'element':
            return metadata

        # Extract all attributes
        attributes = self._extract_all_attributes(node)
        metadata['attributes'] = attributes

        # Determine semantic role
        tag_name = metadata['tag_name']
        if tag_name in ['header', 'nav', 'main', 'footer', 'article', 'section', 'aside']:
            metadata['semantic_role'] = tag_name
            metadata['quality']['has_semantic_structure'] = True

        # Extract accessibility info
        for attr, value in attributes.items():
            if attr.startswith('aria-'):
                metadata['accessibility'][attr] = value
                metadata['quality']['has_accessibility'] = True
            elif attr == 'role':
                metadata['semantic_role'] = value
            elif attr == 'alt' and tag_name == 'img':
                metadata['accessibility']['alt_text'] = value
                metadata['quality']['has_accessibility'] = True

        # Extract SEO-relevant info
        if tag_name == 'meta':
            logger.info("[UNTESTED PATH] html meta tag analysis")
            meta_name = attributes.get('name', '')
            meta_property = attributes.get('property', '')
            content = attributes.get('content', '')

            if meta_name == 'description' or meta_property == 'og:description':
                metadata['seo']['description'] = content
                metadata['quality']['has_meta_description'] = True
            elif meta_name == 'keywords':
                metadata['seo']['keywords'] = content
            elif meta_property.startswith('og:'):
                logger.info("[UNTESTED PATH] html og: property")
                metadata['seo'][meta_property] = content

        elif tag_name == 'title':
            # Extract text content
            text_content = self._extract_text_content(node)
            metadata['seo']['title'] = text_content

        # Extract resources
        if tag_name == 'link' and 'href' in attributes:
            metadata['resources'].append(
                {'type': attributes.get('rel', 'unknown'), 'href': attributes['href']}
            )
        elif tag_name == 'script' and 'src' in attributes:
            metadata['scripts'].append(
                {
                    'src': attributes['src'],
                    'type': attributes.get('type', 'text/javascript'),
                    'async': 'async' in attributes,
                    'defer': 'defer' in attributes,
                }
            )
        elif tag_name == 'img' and 'src' in attributes:
            logger.info("[UNTESTED PATH] html img resource")
            metadata['resources'].append(
                {'type': 'image', 'src': attributes['src'], 'alt': attributes.get('alt', '')}
            )

        # Form analysis
        if tag_name == 'form':
            metadata['forms'].append(
                {
                    'action': attributes.get('action', ''),
                    'method': attributes.get('method', 'get').upper(),
                    'id': attributes.get('id', ''),
                    'name': attributes.get('name', ''),
                }
            )

        # Security checks
        if tag_name == 'script':
            # Check for inline scripts without CSP
            if 'src' not in attributes:
                metadata['security'].append(
                    {
                        'type': 'inline_script',
                        'severity': 'medium',
                        'message': 'Inline script without Content Security Policy',
                    }
                )

        elif tag_name == 'form':
            # Check for forms without CSRF protection indication
            if not any(attr.startswith('data-csrf') for attr in attributes):
                logger.info("[UNTESTED PATH] html form without CSRF")
                metadata['security'].append(
                    {
                        'type': 'csrf_protection_missing',
                        'severity': 'medium',
                        'message': 'Form without visible CSRF protection',
                    }
                )

        # Check for common security attributes
        if (
            'onclick' in attributes
            or 'onload' in attributes
            or 'onmouseover' in attributes
            or 'onclick' in attributes
            or 'onmouseover' in attributes
        ):
            logger.info("[UNTESTED PATH] html inline event handlers")
            metadata['security'].append(
                {
                    'type': 'inline_event_handler',
                    'severity': 'low',
                    'message': 'Inline event handlers should be avoided',
                }
            )

        return metadata

    def _extract_all_attributes(self, element_node) -> Dict[str, str]:
        """Extract all attributes from an element."""
        attributes = {}

        for child in element_node.children:
            if child.type in ['start_tag', 'self_closing_tag']:
                for tag_child in child.children:
                    if tag_child.type == 'attribute':
                        attr_name = None
                        attr_value = ''

                        for attr_child in tag_child.children:
                            if attr_child.type == 'attribute_name':
                                attr_name = attr_child.text.decode('utf8')
                            elif (
                                attr_child.type == 'attribute_value'
                                or attr_child.type == 'quoted_attribute_value'
                            ):
                                attr_value = attr_child.text.decode('utf8')
                                # Remove quotes
                                if attr_value.startswith('"') and attr_value.endswith('"'):
                                    logger.info("[UNTESTED PATH] html attr double quotes")
                                    attr_value = attr_value[1:-1]
                                elif attr_value.startswith("'") and attr_value.endswith("'"):
                                    logger.info("[UNTESTED PATH] html attr single quotes")
                                    attr_value = attr_value[1:-1]

                        if attr_name:
                            attributes[attr_name] = attr_value

        return attributes

    def _extract_text_content(self, node) -> str:
        """Extract text content from a node."""
        text_parts = []

        def extract_text(n):
            if n.type == 'text':
                text_parts.append(n.text.decode('utf8'))
            for child in n.children:
                extract_text(child)

        extract_text(node)
        return ' '.join(text_parts).strip()
