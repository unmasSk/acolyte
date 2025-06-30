"""
Markdown chunker using tree-sitter-languages.
Extracts rich metadata from documentation and README files.
"""

from typing import Dict, List, Any, cast
from tree_sitter_languages import get_language  # type: ignore

from acolyte.models.chunk import Chunk, ChunkType
from acolyte.core.logging import logger
from acolyte.rag.chunking.base import BaseChunker


class MarkdownChunker(BaseChunker):
    """
    Markdown-specific chunker using tree-sitter.

    Extracts structure, code blocks, TODOs, and links for better searchability.
    """

    def _get_language_name(self) -> str:
        """Return language identifier."""
        return 'markdown'

    def _get_tree_sitter_language(self) -> Any:
        """Get Markdown language for tree-sitter."""
        return get_language('markdown')

    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """
        Markdown-specific node types to chunk.

        Tree-sitter Markdown node types:
        - atx_heading: Headers with # syntax
        - setext_heading: Headers with === or --- underline
        - fenced_code_block: Code blocks with ``` or ~~~
        - list: Ordered and unordered lists
        - section: Logical sections (heading + content)
        """
        return {
            # Headers become README chunks
            'atx_heading': ChunkType.README,
            'setext_heading': ChunkType.README,
            # Code blocks are special
            'fenced_code_block': ChunkType.UNKNOWN,  # Will be refined based on language
            # Lists and sections
            'list': ChunkType.README,
            'section': ChunkType.README,
            # Block quotes and paragraphs for documentation
            'block_quote': ChunkType.DOCSTRING,
            'paragraph': ChunkType.README,
        }

    def _get_chunk_size(self) -> int:
        """Markdown uses smaller chunks (50 lines default)."""
        return self.config.get('indexing.chunk_sizes.markdown', 50)

    def _create_chunk_from_node(
        self, node, lines: List[str], file_path: str, chunk_type: ChunkType, processed_ranges
    ):
        """Override to handle Markdown-specific cases and extract rich metadata."""
        # Handle fenced code blocks specially
        if node.type == 'fenced_code_block':
            logger.info("[UNTESTED PATH] markdown fenced_code_block")
            chunk_type = self._determine_code_block_type(node)

        # For sections, include the heading and all content until next heading
        if node.type == 'section':
            return self._create_section_chunk(node, lines, file_path, processed_ranges)

        # Standard processing
        chunk = super()._create_chunk_from_node(
            node, lines, file_path, chunk_type, processed_ranges
        )

        if not chunk:
            logger.info("[UNTESTED PATH] markdown no chunk created")
            return None

        # Extract Markdown-specific metadata
        metadata = self._extract_markdown_metadata(node, lines, chunk)
        if metadata:
            chunk.metadata.language_specific = metadata

        return chunk

    def _determine_code_block_type(self, code_node) -> ChunkType:
        """Determine chunk type based on code block language."""
        # Look for info_string child that contains the language
        for child in code_node.children:
            if child.type == 'info_string':
                lang = child.text.decode('utf8').strip().lower()
                # Common test patterns
                if any(test in lang for test in ['test', 'spec', 'jest', 'pytest']):
                    return ChunkType.TESTS
                # Common config formats
                elif lang in ['json', 'yaml', 'yml', 'toml', 'ini', 'env']:
                    return ChunkType.CONSTANTS  # Using for config
                # Code examples
                elif lang in ['python', 'javascript', 'typescript', 'java', 'go', 'rust']:
                    return ChunkType.UNKNOWN  # Generic code example
                break

        return ChunkType.UNKNOWN

    def _create_section_chunk(self, section_node, lines, file_path, processed_ranges):
        """Create chunk for a section (heading + its content)."""
        # Find the heading within the section
        heading_node = None
        heading_text = ""
        heading_level = 0

        for child in section_node.children:
            if child.type in ['atx_heading', 'setext_heading']:
                heading_node = child
                heading_text, heading_level = self._extract_heading_info(child)
                break

        if not heading_node:
            return None

        # Get section boundaries
        start_line = section_node.start_point[0]
        end_line = section_node.end_point[0]

        # Check if already processed
        if any(start_line <= s <= end_line for (s, e) in processed_ranges):
            return None

        # Extract content
        content = '\n'.join(lines[start_line : end_line + 1])

        # Mark as processed
        processed_ranges.add((start_line, end_line))

        # Create chunk with section name
        chunk = self._create_chunk(
            content=content,
            chunk_type=ChunkType.README,
            file_path=file_path,
            start_line=start_line + 1,
            end_line=end_line + 1,
            name=heading_text,
        )

        return chunk

    def _extract_heading_info(self, heading_node) -> tuple[str, int]:
        """Extract heading text and level."""
        text = ""
        level = 0

        if heading_node.type == 'atx_heading':
            # Count # symbols for level
            for child in heading_node.children:
                if child.type == 'atx_h1_marker':
                    level = 1
                elif child.type == 'atx_h2_marker':
                    level = 2
                elif child.type == 'atx_h3_marker':
                    level = 3
                elif child.type == 'atx_h4_marker':
                    level = 4
                elif child.type == 'atx_h5_marker':
                    level = 5
                elif child.type == 'atx_h6_marker':
                    level = 6
                elif child.type in ['heading_content', 'inline']:
                    logger.info("[UNTESTED PATH] markdown heading content extraction")
                    text = child.text.decode('utf8').strip()

        elif heading_node.type == 'setext_heading':
            # Setext only supports h1 (===) and h2 (---)
            text = heading_node.children[0].text.decode('utf8').strip()
            # Check the underline
            for child in heading_node.children:
                if child.type == 'setext_h1_underline':
                    level = 1
                elif child.type == 'setext_h2_underline':
                    logger.info("[UNTESTED PATH] markdown setext h2 underline")
                    level = 2

        return text, level

    def _extract_markdown_metadata(self, node, lines, chunk) -> Dict[str, Any]:
        """Extract rich metadata from Markdown nodes."""
        metadata = {}

        # Extract heading info
        if node.type in ['atx_heading', 'setext_heading']:
            text, level = self._extract_heading_info(node)
            metadata.update(
                {
                    'heading_level': level,
                    'heading_text': text,
                    'is_title': level == 1,
                }
            )

        # Extract code block info
        elif node.type == 'fenced_code_block':
            metadata.update(self._extract_code_block_metadata(node))

        # Extract list metadata
        elif node.type == 'list':
            metadata.update(self._extract_list_metadata(node))

        # Extract TODOs and special markers from content
        todos = self._extract_todos_from_content(chunk.content)
        if todos:
            metadata['todos'] = todos

        # Extract links and references
        links = self._extract_links(node)
        if links:
            metadata['links'] = links

        # Analyze structure
        metadata['structure'] = self._analyze_structure(chunk.content)

        # Quality indicators
        metadata['quality'] = self._analyze_markdown_quality(chunk.content, metadata)

        return metadata

    def _extract_code_block_metadata(self, code_node) -> Dict[str, Any]:
        """Extract metadata from code blocks."""
        metadata = {
            'code_language': None,
            'is_executable': False,
            'has_output': False,
        }

        # Get language from info string
        for child in code_node.children:
            if child.type == 'info_string':
                info = child.text.decode('utf8').strip()
                if info:
                    # First word is usually the language
                    parts = info.split()
                    metadata['code_language'] = parts[0].lower()

                    # Check for special markers
                    if 'executable' in info or 'runnable' in info:
                        logger.info("[UNTESTED PATH] markdown executable code block")
                        metadata['is_executable'] = True
                    if 'output' in info or 'result' in info:
                        logger.info("[UNTESTED PATH] markdown output code block")
                        metadata['has_output'] = True
                break

            # Extract code content for analysis
            elif child.type == 'code_fence_content':
                code_text = child.text.decode('utf8')
                # Detect shell commands
                if any(code_text.strip().startswith(p) for p in ['$', '>', '#']):
                    metadata['is_shell_command'] = True
                # Detect test code
                if any(t in code_text for t in ['assert', 'expect', 'test.', 'describe(']):
                    metadata['is_test_code'] = True

        return metadata

    def _extract_list_metadata(self, list_node) -> Dict[str, Any]:
        """Extract metadata from lists."""
        metadata = {
            'list_type': 'unordered',  # default
            'item_count': 0,
            'has_nested_lists': False,
            'has_code_items': False,
            'has_checkboxes': False,
        }

        # Analyze list items
        def analyze_list(node, depth=0):
            for child in node.children:
                if child.type == 'list_item':
                    metadata['item_count'] = cast(int, metadata['item_count']) + 1

                    # Check for task list items
                    text = child.text.decode('utf8')
                    if '[ ]' in text or '[x]' in text or '[X]' in text:
                        metadata['has_checkboxes'] = True

                    # Check for code in items
                    if '`' in text:
                        metadata['has_code_items'] = True

                    # Check for ordered list markers
                    if any(c.type == 'list_marker_dot' for c in child.children):
                        metadata['list_type'] = 'ordered'

                elif child.type == 'list' and depth > 0:
                    metadata['has_nested_lists'] = True

                # Recurse
                if child.type in ['list', 'list_item']:
                    analyze_list(child, depth + 1)

        analyze_list(list_node)
        return metadata

    def _extract_todos_from_content(self, content: str) -> List[Dict[str, Any]]:
        """Extract TODO, FIXME, NOTE, etc. from markdown content."""
        import re

        todos = []
        patterns = {
            'TODO': r'(?:^|\s)(?:TODO|todo):?\s*(.+?)(?:\n|$)',
            'FIXME': r'(?:^|\s)(?:FIXME|fixme):?\s*(.+?)(?:\n|$)',
            'NOTE': r'(?:^|\s)(?:NOTE|note):?\s*(.+?)(?:\n|$)',
            'WARNING': r'(?:^|\s)(?:WARNING|⚠️):?\s*(.+?)(?:\n|$)',
            'IMPORTANT': r'(?:^|\s)(?:IMPORTANT|‼️|❗):?\s*(.+?)(?:\n|$)',
            'TIP': r'(?:^|\s)(?:TIP|💡):?\s*(.+?)(?:\n|$)',
        }

        lines = content.split('\n')
        for i, line in enumerate(lines):
            for todo_type, pattern in patterns.items():
                matches = re.finditer(pattern, line)
                for match in matches:
                    todos.append({'type': todo_type, 'text': match.group(1).strip(), 'line': i + 1})

        return todos

    def _extract_links(self, node) -> List[Dict[str, Any]]:
        """Extract all links from the node."""
        links = []

        def find_links(n):
            if n.type == 'link':
                link_text = ""
                link_url = ""

                for child in n.children:
                    if child.type == 'link_text':
                        link_text = child.text.decode('utf8').strip('[]')
                    elif child.type == 'link_destination':
                        link_url = child.text.decode('utf8').strip('()')

                if link_url:
                    links.append(
                        {
                            'text': link_text,
                            'url': link_url,
                            'is_external': link_url.startswith(('http://', 'https://')),
                            'is_anchor': link_url.startswith('#'),
                            'is_relative': link_url.startswith(('./', '../')),
                        }
                    )

            # Recurse
            for child in n.children:
                find_links(child)

        find_links(node)
        return links

    def _analyze_structure(self, content: str) -> Dict[str, Any]:
        """Analyze markdown structure metrics."""
        lines = content.split('\n')

        structure = {
            'total_lines': len(lines),
            'blank_lines': sum(1 for line in lines if not line.strip()),
            'has_front_matter': content.strip().startswith('---'),
            'has_table_of_contents': any('table of contents' in line.lower() for line in lines),
            'code_block_count': content.count('```'),
            'has_images': '![' in content,
            'has_tables': '|' in content and lines.count('|') > 2,
            'emphasis_count': content.count('**') + content.count('*') + content.count('_'),
        }

        return structure

    def _analyze_markdown_quality(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze quality indicators for documentation."""
        quality = {
            'has_title': metadata.get('heading_level') == 1,
            'has_description': len(content.split('\n')) > 5,
            'has_code_examples': metadata.get('structure', {}).get('code_block_count', 0) > 0,
            'has_links': len(metadata.get('links', [])) > 0,
            'completeness_hints': [],
        }

        # Check for common documentation sections
        content_lower = content.lower()
        common_sections = {
            'installation': ['install', 'setup', 'getting started'],
            'usage': ['usage', 'how to use', 'example', 'quick start'],
            'api': ['api', 'reference', 'methods', 'functions'],
            'configuration': ['config', 'settings', 'options'],
            'contributing': ['contribut', 'development', 'pull request'],
            'license': ['license', 'copyright'],
        }

        for section, keywords in common_sections.items():
            if any(kw in content_lower for kw in keywords):
                quality['completeness_hints'].append(f'has_{section}')

        # Documentation completeness score (0-1)
        quality['completeness_score'] = len(quality['completeness_hints']) / len(common_sections)

        return quality

    async def chunk(self, content: str, file_path: str) -> List[Chunk]:
        """
        Override to handle Markdown-specific preprocessing.

        Markdown benefits from keeping sections together, so we'll
        use a hybrid approach of AST + section awareness.
        """
        # Remove front matter if present (it's metadata, not content)
        if content.strip().startswith('---'):
            # Find the closing ---
            lines = content.split('\n')
            end_idx = -1
            for i in range(1, len(lines)):
                if lines[i].strip() == '---':
                    end_idx = i
                    break

            if end_idx > 0:
                # Extract front matter for potential metadata
                self._front_matter = '\n'.join(lines[1:end_idx])
                # Remove from content
                content = '\n'.join(lines[end_idx + 1 :])

        # Let base class handle the rest
        chunks = await super().chunk(content, file_path)

        # Post-process to ensure README type for all markdown
        for chunk in chunks:
            if chunk.metadata.chunk_type == ChunkType.UNKNOWN:
                chunk.metadata.chunk_type = ChunkType.README

        return chunks

    def _get_import_node_types(self) -> List[str]:
        """Get node types that represent imports for Markdown."""
        # Markdown doesn't have imports
        return []

    def _is_comment_node(self, node) -> bool:
        """Check if node is a Markdown comment."""
        # Markdown comments are HTML comments
        return node.type == 'html_comment'

    def _extract_dependencies_from_imports(self, import_nodes) -> List[str]:
        """Extract dependencies - Markdown has no imports."""
        return []
