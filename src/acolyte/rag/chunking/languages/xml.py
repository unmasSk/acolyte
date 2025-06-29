"""
XML chunker with enhanced pattern-based parsing.
Provides rich metadata extraction despite lack of tree-sitter support.

Features:
- Element-based chunking with namespace awareness
- Attribute extraction and analysis
- CDATA section handling
- DTD and schema reference detection
- Configuration file pattern recognition (web.xml, pom.xml, etc.)
- Depth and complexity analysis
- Security issue detection
- Comprehensive metadata
"""

from typing import Dict, List, Optional, Any, Set
import re
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError

from acolyte.models.chunk import Chunk, ChunkType
from acolyte.core.logging import logger
from acolyte.rag.chunking.base import BaseChunker
from acolyte.rag.chunking.mixins import (
    ComplexityMixin,
    TodoExtractionMixin,
    SecurityAnalysisMixin,
    PatternDetectionMixin,
)


class XmlChunker(
    BaseChunker, ComplexityMixin, TodoExtractionMixin, SecurityAnalysisMixin, PatternDetectionMixin
):
    """
    XML-specific chunker using advanced pattern matching and XML parsing.

    Since tree-sitter-languages doesn't support XML, we use a combination of
    ElementTree parsing (when possible) and regex patterns for robustness.

    Handles:
    - Standard XML documents with proper structure
    - HTML/XHTML documents
    - Configuration files (web.xml, pom.xml, build.xml, etc.)
    - SOAP/WSDL service definitions
    - RSS/Atom feeds
    - SVG graphics
    - Malformed XML (fallback to pattern matching)
    """

    def __init__(self):
        """Initialize with XML language configuration."""
        super().__init__()
        self._init_patterns()
        self._init_type_mappings()
        logger.info("XmlChunker: Using enhanced pattern-based chunking with XML parsing")

    def _get_language_name(self) -> str:
        """Return language identifier."""
        return 'xml'

    def _get_tree_sitter_language(self) -> Any:
        """XML is not supported by tree-sitter-languages."""
        return None

    def _get_import_node_types(self) -> List[str]:
        """Not used for pattern-based chunking."""
        return []

    def _is_comment_node(self, node) -> bool:
        """Not used for pattern-based chunking."""
        return False

    def _init_patterns(self):
        """Initialize regex patterns for XML constructs."""
        # XML declaration
        self.xml_decl_pattern = re.compile(
            r'<\?xml\s+version\s*=\s*["\']([^"\']+)["\']\s*(?:encoding\s*=\s*["\']([^"\']+)["\'])?\s*\?>',
            re.IGNORECASE,
        )

        # DOCTYPE declaration
        self.doctype_pattern = re.compile(
            r'<!DOCTYPE\s+(\w+)(?:\s+PUBLIC\s+"([^"]+)"\s+"([^"]+)"|\s+SYSTEM\s+"([^"]+)")?\s*(?:\[.*?\])?\s*>',
            re.DOTALL | re.IGNORECASE,
        )

        # Elements with attributes
        self.element_pattern = re.compile(
            r'<(\w+(?::\w+)?)((?:\s+[\w:.-]+\s*=\s*"[^"]*")*)\s*(?:/>|>)', re.MULTILINE
        )

        # CDATA sections
        self.cdata_pattern = re.compile(r'<!\[CDATA\[(.*?)\]\]>', re.DOTALL)

        # Comments
        self.comment_pattern = re.compile(r'<!--(.*?)-->', re.DOTALL)

        # Processing instructions
        self.pi_pattern = re.compile(r'<\?(\w+)(?:\s+([^?]+))?\?>', re.MULTILINE)

        # Namespace declarations
        self.namespace_pattern = re.compile(r'xmlns(?::(\w+))?\s*=\s*"([^"]+)"')

        # Schema references
        self.schema_patterns = {
            'xsi:schemaLocation': re.compile(r'xsi:schemaLocation\s*=\s*"([^"]+)"'),
            'xsi:noNamespaceSchemaLocation': re.compile(
                r'xsi:noNamespaceSchemaLocation\s*=\s*"([^"]+)"'
            ),
        }

        # Security sensitive patterns
        self.security_patterns = {
            'password': re.compile(
                r'password["\']?\s*(?:=|>)\s*["\']?([^"\'<>\s]+)', re.IGNORECASE
            ),
            'credential': re.compile(
                r'(api[_-]?key|token|secret|auth)["\']?\s*(?:=|>)\s*["\']?([^"\'<>\s]+)',
                re.IGNORECASE,
            ),
            'connection': re.compile(r'(jdbc:|mongodb:|mysql:|postgres:)//[^<>\s]+', re.IGNORECASE),
            'file_path': re.compile(
                r'(?:file|path)["\']?\s*(?:=|>)\s*["\']?([A-Za-z]:[\\/][^"\'<>]+|/[^"\'<>\s]+)',
                re.IGNORECASE,
            ),
        }

    def _init_type_mappings(self):
        """Initialize mappings for common XML file types and elements."""
        # File type to root element mappings
        self.file_type_mappings = {
            'pom.xml': 'maven_project',
            'build.xml': 'ant_build',
            'web.xml': 'web_app_config',
            'applicationContext.xml': 'spring_config',
            'persistence.xml': 'jpa_config',
            'faces-config.xml': 'jsf_config',
            'struts-config.xml': 'struts_config',
            'hibernate.cfg.xml': 'hibernate_config',
            '.project': 'eclipse_project',
            '.classpath': 'eclipse_classpath',
            'AndroidManifest.xml': 'android_manifest',
        }

        # Element to ChunkType mappings
        self.element_type_mappings = {
            # Configuration elements
            'configuration': ChunkType.CONSTANTS,
            'settings': ChunkType.CONSTANTS,
            'properties': ChunkType.CONSTANTS,
            'config': ChunkType.CONSTANTS,
            # Type definitions
            'complexType': ChunkType.TYPES,
            'simpleType': ChunkType.TYPES,
            'element': ChunkType.TYPES,
            'attribute': ChunkType.TYPES,
            # Service definitions
            'services': ChunkType.MODULE,  # Container for services
            'service': ChunkType.CLASS,
            'portType': ChunkType.INTERFACE,
            'binding': ChunkType.CLASS,
            'operation': ChunkType.METHOD,
            # Dependencies
            'dependency': ChunkType.IMPORTS,
            'dependencies': ChunkType.IMPORTS,
            'import': ChunkType.IMPORTS,
            'include': ChunkType.IMPORTS,
            # Documentation
            'documentation': ChunkType.DOCSTRING,
            'description': ChunkType.DOCSTRING,
            'comment': ChunkType.COMMENT,
            # Modules/packages
            'module': ChunkType.MODULE,
            'package': ChunkType.MODULE,
            'namespace': ChunkType.NAMESPACE,
            # Build/deployment
            'target': ChunkType.FUNCTION,
            'task': ChunkType.FUNCTION,
            'goal': ChunkType.FUNCTION,
            'plugin': ChunkType.CLASS,
            # Testing
            'test': ChunkType.TESTS,
            'testcase': ChunkType.TESTS,
            'testsuite': ChunkType.TESTS,
        }

    async def chunk(self, content: str, file_path: str) -> List[Chunk]:
        """
        Chunk XML content using XML parsing with pattern matching fallback.

        Strategy:
        1. Try to parse with ElementTree for proper structure
        2. Fall back to pattern matching if parsing fails
        3. Extract rich metadata for each significant element
        4. Group related elements intelligently
        """
        chunks = []
        lines = content.split('\n')

        # Determine file type
        file_type = self._determine_file_type(file_path)

        # Try XML parsing first
        try:
            chunks = await self._chunk_with_parser(content, lines, file_path, file_type)
        except (ParseError, ET.ParseError) as e:
            logger.warning(f"XML parsing failed for {file_path}: {e}. Using pattern matching.")
            chunks = await self._chunk_with_patterns(content, lines, file_path, file_type)

        # Sort by start line
        chunks.sort(key=lambda c: c.metadata.start_line)

        # Validate and add smart overlap
        chunks = self._validate_chunks(chunks)
        chunks = self._add_smart_overlap(chunks, preserve_imports=True)

        return chunks

    async def _chunk_with_parser(
        self, content: str, lines: List[str], file_path: str, file_type: str
    ) -> List[Chunk]:
        """Chunk using ElementTree parser."""
        chunks = []

        # Extract XML header first (declaration + DOCTYPE)
        header_chunk = self._extract_xml_header(content, lines, file_path)
        if header_chunk:
            chunks.append(header_chunk)

        # Parse XML
        root = ET.fromstring(content)

        # Build parent map for depth calculation
        parent_map = {child: parent for parent in root.iter() for child in parent}

        # Process root element
        root_chunk = self._process_element(
            root, content, lines, file_path, 0, file_type, parent_map, root
        )
        if root_chunk:
            # Special handling for root element namespaces
            # ElementTree strips xmlns attributes, so extract from original content
            root_pattern = r'<\w+[^>]*>'
            root_match = re.search(root_pattern, content)
            if root_match and root_chunk.metadata.language_specific:
                opening_tag = root_match.group(0)
                for ns_match in self.namespace_pattern.finditer(opening_tag):
                    logger.info("[UNTESTED PATH] xml namespace extraction from opening tag")
                    ns_prefix = ns_match.group(1) or 'default'
                    ns_uri = ns_match.group(2)
                    if (
                        isinstance(root_chunk.metadata.language_specific, dict)
                        and 'namespaces' in root_chunk.metadata.language_specific
                        and isinstance(root_chunk.metadata.language_specific['namespaces'], dict)
                    ):
                        logger.info("[UNTESTED PATH] xml namespace assignment to root chunk")
                        root_chunk.metadata.language_specific['namespaces'][ns_prefix] = ns_uri
            chunks.append(root_chunk)

        # Process significant child elements
        processed_elements = {root}
        chunks.extend(
            self._process_children(
                root, content, lines, file_path, processed_elements, file_type, parent_map, root
            )
        )

        return chunks

    async def _chunk_with_patterns(
        self, content: str, lines: List[str], file_path: str, file_type: str
    ) -> List[Chunk]:
        """Chunk using regex patterns when parsing fails."""
        chunks = []
        processed_lines = set()

        # Extract XML declaration and DOCTYPE
        header_chunk = self._extract_xml_header(content, lines, file_path)
        if header_chunk:
            chunks.append(header_chunk)
            for i in range(header_chunk.metadata.start_line - 1, header_chunk.metadata.end_line):
                processed_lines.add(i)

        # Extract major elements using patterns
        element_chunks = self._extract_elements_by_pattern(
            content, lines, file_path, processed_lines, file_type
        )
        chunks.extend(element_chunks)

        return chunks

    def _determine_file_type(self, file_path: str) -> str:
        """Determine XML file type from filename."""
        filename = file_path.split('/')[-1].split('\\')[-1].lower()

        for pattern, file_type in self.file_type_mappings.items():
            if pattern in filename:
                return file_type

        # Check for common patterns
        if filename.endswith('.xml'):
            if 'config' in filename:
                return 'config_file'
            elif 'build' in filename:
                return 'build_file'
            elif 'test' in filename:
                return 'test_file'

        return 'generic_xml'

    def _process_element(
        self,
        element: ET.Element,
        content: str,
        lines: List[str],
        file_path: str,
        depth: int,
        file_type: str,
        parent_map: Dict[ET.Element, ET.Element],
        root: ET.Element,
    ) -> Optional[Chunk]:
        """Process a single XML element into a chunk."""
        # Skip trivial elements
        if not self._is_significant_element(element, depth):
            return None

        # Get element position in content
        element_str = ET.tostring(element, encoding='unicode', method='xml')

        # Try to find the element in the original content
        tag_name = element.tag.split('}')[-1] if '}' in element.tag else element.tag

        # For better matching, let's use a more robust approach
        # First, check if we can find a unique identifier
        unique_attrs = []
        for attr in ['name', 'id', 'class', 'type']:
            if attr in element.attrib:
                unique_attrs.append(f'{attr}="{element.attrib[attr]}"')

        # Build a pattern that includes attributes
        if unique_attrs:
            pattern = f'<{tag_name}[^>]*{unique_attrs[0]}[^>]*>'
        else:
            pattern = f'<{tag_name}[^>]*>'

        # Find all matches and try to locate the right one
        matches = list(re.finditer(pattern, content, re.IGNORECASE))

        # If no matches with the simple tag name, try without namespace
        if not matches and ':' in tag_name:
            logger.info("[UNTESTED PATH] xml namespace tag fallback")
            simple_tag = tag_name.split(':')[-1]
            pattern = rf'<\w*:?{simple_tag}[^>]*>'
            matches = list(re.finditer(pattern, content, re.IGNORECASE))

        if not matches:
            # Fallback to ElementTree serialization
            return self._create_chunk(
                content=element_str,
                chunk_type=self._determine_element_chunk_type(element, file_type),
                file_path=file_path,
                start_line=1,
                end_line=element_str.count('\n') + 1,
                name=self._get_element_name(element),
            )

        # Use the first match
        match = matches[0]
        start_pos = match.start()

        # Extract content using a stack-based approach to handle nested tags
        original_element_content = self._extract_element_content(content, start_pos, tag_name)
        if not original_element_content:
            original_element_content = element_str

        start_line = content[:start_pos].count('\n')
        end_line = start_line + original_element_content.count('\n')

        # Determine chunk type
        chunk_type = self._determine_element_chunk_type(element, file_type)

        # Create chunk
        chunk = self._create_chunk(
            content=original_element_content,
            chunk_type=chunk_type,
            file_path=file_path,
            start_line=start_line + 1,
            end_line=end_line + 1,
            name=self._get_element_name(element),
        )

        # Extract metadata using the original content
        chunk.metadata.language_specific = self._extract_element_metadata(
            element, original_element_content, depth, file_type
        )

        return chunk

    def _process_children(
        self,
        parent: ET.Element,
        content: str,
        lines: List[str],
        file_path: str,
        processed: Set[ET.Element],
        file_type: str,
        parent_map: Dict[ET.Element, ET.Element],
        root: ET.Element,
    ) -> List[Chunk]:
        """Recursively process child elements."""
        chunks = []

        for child in parent:
            if child in processed:
                continue

            processed.add(child)

            # Calculate depth
            depth = self._calculate_element_depth(child, parent_map, root)

            # Process this element
            chunk = self._process_element(
                child, content, lines, file_path, depth, file_type, parent_map, root
            )
            if chunk:
                chunks.append(chunk)

                # Don't process children if this element is chunked
                # UNLESS it's a container element that should have its children processed
                tag_name = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                container_elements = {
                    'services',
                    'modules',
                    'dependencies',
                    'plugins',
                    'beans',
                    'targets',
                    'tasks',
                }
                if tag_name.lower() not in container_elements:
                    continue

            # Process children if this element wasn't chunked
            chunks.extend(
                self._process_children(
                    child, content, lines, file_path, processed, file_type, parent_map, root
                )
            )

        return chunks

    def _is_significant_element(self, element: ET.Element, depth: int) -> bool:
        """Determine if an element is significant enough to chunk."""
        tag_name = element.tag.split('}')[-1] if '}' in element.tag else element.tag
        tag_lower = tag_name.lower()

        # Root element at depth 0 is always significant
        if depth == 0:
            return True

        # Always chunk certain important elements regardless of size
        if tag_lower in self.element_type_mappings:
            # These elements are always significant, even if small
            always_significant = {
                'service',
                'bean',
                'dependency',
                'module',
                'plugin',
                'property',
                'configuration',
                'settings',
                'task',
                'target',
                'root',
                'project',
                'manifest',
                'application',
            }
            if tag_lower in always_significant:
                return True

            # For other mapped elements, check size
            element_str = ET.tostring(element, encoding='unicode', method='xml')
            if len(element_str) >= 100:
                return True

        # Skip if too deep (unless important)
        if depth > 3 and tag_lower not in ['dependency', 'plugin', 'test', 'service']:
            logger.info("[UNTESTED PATH] xml skipping deep element")
            return False

        # Check size for non-mapped elements
        element_str = ET.tostring(element, encoding='unicode', method='xml')
        if len(element_str) < 100:  # Too small
            return False

        # Check complexity
        child_count = len(list(element))
        if child_count < 2 and not element.text:  # Too simple
            return False

        # Check for important attributes
        important_attrs = {'id', 'name', 'class', 'type', 'ref'}
        if any(attr in element.attrib for attr in important_attrs):
            return True

        return child_count >= 3 or len(element_str) > 500

    def _determine_element_chunk_type(self, element: ET.Element, file_type: str) -> ChunkType:
        """Determine ChunkType for an element."""
        tag_name = element.tag.split('}')[-1] if '}' in element.tag else element.tag
        tag_lower = tag_name.lower()

        # Check mappings
        if tag_lower in self.element_type_mappings:
            return self.element_type_mappings[tag_lower]

        # File-type specific mappings
        if file_type == 'maven_project':
            if tag_lower in ['groupid', 'artifactid', 'version']:
                return ChunkType.CONSTANTS
            elif tag_lower in ['build', 'reporting']:
                return ChunkType.MODULE
        elif file_type == 'spring_config':
            if tag_lower == 'bean':
                return ChunkType.CLASS
            elif tag_lower in ['property', 'constructor-arg']:
                return ChunkType.PROPERTY

        # Default based on content and tag name
        if tag_lower == 'modules' or tag_lower == 'module':
            return ChunkType.MODULE
        elif len(list(element)) > 5:
            return ChunkType.MODULE
        else:
            return ChunkType.UNKNOWN

    def _get_element_name(self, element: ET.Element) -> str:
        """Get a meaningful name for an element."""
        tag_name = element.tag.split('}')[-1] if '}' in element.tag else element.tag

        # Try common naming attributes
        for attr in ['name', 'id', 'ref', 'class', 'type']:
            if attr in element.attrib:
                return f"{tag_name}_{element.attrib[attr]}"

        # For specific elements
        if tag_name.lower() == 'dependency':
            artifact = element.find('.//artifactId')
            if artifact is not None and artifact.text:
                logger.info("[UNTESTED PATH] xml dependency artifact name")
                return f"dependency_{artifact.text}"

        return tag_name

    def _extract_element_metadata(
        self, element: ET.Element, element_str: str, depth: int, file_type: str
    ) -> Dict[str, Any]:
        """Extract comprehensive metadata for an element."""
        tag_name = element.tag.split('}')[-1] if '}' in element.tag else element.tag

        metadata: Dict[str, Any] = {
            'tag_name': tag_name,
            'namespace': self._extract_namespace(element.tag),
            'attributes': dict(element.attrib),
            'depth': depth,
            'child_count': len(list(element)),
            'text_content': (element.text or '').strip()[:100],  # First 100 chars
            'has_cdata': '<![CDATA[' in element_str,
            'namespaces': {},
            'complexity': {
                'total_elements': 1,
                'max_depth': 0,
                'attribute_count': len(element.attrib),
                'has_mixed_content': False,
            },
            'patterns': {
                'type': file_type,
                'uses': [],
                'anti_patterns': [],
            },
            'security': [],
            'todos': [],
            'dependencies': [],
        }

        # Extract namespace declarations from the original content
        # ElementTree doesn't preserve xmlns attributes on elements
        # Look for the opening tag in the content
        tag_pattern = f'<{tag_name}[^>]*>'
        tag_match = re.search(tag_pattern, element_str)
        if tag_match:
            opening_tag = tag_match.group(0)
            for ns_match in self.namespace_pattern.finditer(opening_tag):
                ns_prefix = ns_match.group(1) or 'default'
                ns_uri = ns_match.group(2)
                metadata['namespaces'][ns_prefix] = ns_uri

        # Calculate complexity
        if isinstance(metadata['complexity'], dict):
            self._calculate_element_complexity(element, metadata['complexity'])

        # Check for mixed content
        if element.text and element.text.strip() and len(list(element)) > 0:
            if isinstance(metadata['complexity'], dict):
                metadata['complexity']['has_mixed_content'] = True

        # Extract patterns and uses
        self._detect_xml_patterns(element, metadata, file_type)

        # Security analysis
        metadata['security'] = self._analyze_xml_security(element, element_str)

        # Extract TODOs from comments
        for comment_match in self.comment_pattern.finditer(element_str):
            comment_text = comment_match.group(1)
            todo_match = re.search(
                r'(TODO|FIXME|HACK|XXX|BUG)[:\s](.+)', comment_text, re.IGNORECASE
            )
            if todo_match:
                metadata['todos'].append(f"{todo_match.group(1)}: {todo_match.group(2).strip()}")

        # Extract dependencies for specific file types
        if file_type == 'maven_project' or (
            tag_name.lower() in ['dependency', 'dependencies'] and 'groupid' in element_str.lower()
        ):
            if tag_name.lower() == 'dependency':
                dep_info = self._extract_maven_dependency(element)
                if dep_info:
                    if isinstance(metadata['dependencies'], list):
                        metadata['dependencies'].append(dep_info)
            elif tag_name.lower() == 'dependencies':
                # For dependencies container, extract all child dependencies
                for child in element:
                    child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                    if child_tag.lower() == 'dependency':
                        dep_info = self._extract_maven_dependency(child)
                        if dep_info and isinstance(metadata['dependencies'], list):
                            metadata['dependencies'].append(dep_info)

        return metadata

    def _extract_namespace(self, tag: str) -> Optional[str]:
        """Extract namespace from tag."""
        if '}' in tag:
            return tag.split('}')[0].lstrip('{')
        return None

    def _calculate_element_complexity(self, element: ET.Element, complexity: Dict[str, int]):
        """Calculate complexity metrics for an element tree."""

        def traverse(elem, depth=0):
            complexity['total_elements'] += 1
            complexity['max_depth'] = max(complexity['max_depth'], depth)
            complexity['attribute_count'] += len(elem.attrib)

            for child in elem:
                traverse(child, depth + 1)

        for child in element:
            traverse(child, 1)

    def _calculate_element_depth(
        self, element: ET.Element, parent_map: Dict[ET.Element, ET.Element], root: ET.Element
    ) -> int:
        """Calculate depth of element in tree using parent map."""
        depth = 0
        current = element

        while current != root and current in parent_map:
            current = parent_map[current]
            depth += 1

        return depth

    def _detect_xml_patterns(self, element: ET.Element, metadata: Dict[str, Any], file_type: str):
        """Detect patterns and uses in XML element."""
        # Framework detection
        if file_type == 'spring_config':
            if isinstance(metadata['patterns'], dict) and isinstance(
                metadata['patterns'].get('uses'), list
            ):
                metadata['patterns']['uses'].append('spring_framework')
            if 'class' in element.attrib:
                class_name = element.attrib['class']
                if 'Controller' in class_name:
                    if isinstance(metadata['patterns'], dict) and isinstance(
                        metadata['patterns'].get('uses'), list
                    ):
                        metadata['patterns']['uses'].append('spring_mvc')
                elif 'Repository' in class_name:
                    if isinstance(metadata['patterns'], dict) and isinstance(
                        metadata['patterns'].get('uses'), list
                    ):
                        metadata['patterns']['uses'].append('spring_data')

        elif file_type == 'maven_project':
            if isinstance(metadata['patterns'], dict) and isinstance(
                metadata['patterns'].get('uses'), list
            ):
                metadata['patterns']['uses'].append('maven_build')

        elif file_type == 'hibernate_config':
            if isinstance(metadata['patterns'], dict) and isinstance(
                metadata['patterns'].get('uses'), list
            ):
                metadata['patterns']['uses'].append('hibernate_orm')

        # Anti-patterns
        if isinstance(metadata['complexity'], dict) and metadata['complexity']['max_depth'] > 10:
            if isinstance(metadata['patterns'], dict) and isinstance(
                metadata['patterns'].get('anti_patterns'), list
            ):
                metadata['patterns']['anti_patterns'].append('excessive_nesting')

        if (
            isinstance(metadata['complexity'], dict)
            and metadata['complexity']['total_elements'] > 100
        ):
            if isinstance(metadata['patterns'], dict) and isinstance(
                metadata['patterns'].get('anti_patterns'), list
            ):
                metadata['patterns']['anti_patterns'].append('oversized_element')

        # Check for inline styles/scripts (bad practice in config)
        if 'style' in element.attrib or 'onclick' in element.attrib:
            if isinstance(metadata['patterns'], dict) and isinstance(
                metadata['patterns'].get('anti_patterns'), list
            ):
                metadata['patterns']['anti_patterns'].append('inline_code')

    def _analyze_xml_security(self, element: ET.Element, element_str: str) -> List[Dict[str, Any]]:
        """Analyze security issues in XML element."""
        issues = []

        # Check element text and attributes
        tag_name = element.tag.split('}')[-1] if '}' in element.tag else element.tag

        # Password detection in element text
        if 'password' in tag_name.lower() and element.text:
            pwd = element.text.strip()
            if pwd and not pwd.startswith('${'):
                issues.append(
                    {
                        'type': 'hardcoded_password',
                        'severity': 'critical',
                        'location': f'element:{tag_name}',
                        'description': 'Hardcoded password in element text',
                    }
                )

        # Check for api-key elements
        if 'key' in tag_name.lower() and element.text:
            key = element.text.strip()
            if key and len(key) > 10 and not key.startswith('${'):
                issues.append(
                    {
                        'type': 'exposed_credential',
                        'severity': 'high',
                        'location': f'element:{tag_name}',
                        'description': 'API key exposed in element text',
                    }
                )

        # Password/credential detection in attributes
        for attr, value in element.attrib.items():
            if 'password' in attr.lower() and value and not value.startswith('${'):
                issues.append(
                    {
                        'type': 'hardcoded_password',
                        'severity': 'critical',
                        'location': f'attribute:{attr}',
                        'description': 'Hardcoded password in attribute',
                    }
                )

        # Search for hardcoded passwords in content
        password_matches = re.findall(
            r'<password[^>]*>([^<]+)</password>', element_str, re.IGNORECASE
        )
        for pwd in password_matches:
            if pwd.strip() and not pwd.strip().startswith('${'):
                issues.append(
                    {
                        'type': 'hardcoded_password',
                        'severity': 'critical',
                        'location': 'element:password',
                        'description': 'Hardcoded password in content',
                    }
                )

        # API key/token detection
        key_patterns = [
            (r'<api[_-]?key[^>]*>([^<]+)</api[_-]?key>', 'api-key'),
            (r'<secret[_-]?key[^>]*>([^<]+)</secret[_-]?key>', 'secret-key'),
            (r'<token[^>]*>([^<]+)</token>', 'token'),
        ]

        for pattern, key_type in key_patterns:
            matches = re.findall(pattern, element_str, re.IGNORECASE)
            for match in matches:
                if match.strip() and len(match.strip()) > 10 and not match.strip().startswith('${'):
                    issues.append(
                        {
                            'type': 'exposed_credential',
                            'severity': 'high',
                            'location': f'element:{key_type}',
                            'description': f'Exposed {key_type} in content',
                        }
                    )

        # File path exposure
        if '/var/log' in element_str or '/home/' in element_str or 'C:\\' in element_str:
            issues.append(
                {
                    'type': 'absolute_path',
                    'severity': 'medium',
                    'location': f'element:{tag_name}',
                    'description': 'Absolute file path detected',
                }
            )

        return issues

    def _extract_maven_dependency(self, element: ET.Element) -> Optional[Dict[str, str]]:
        """Extract Maven dependency information."""
        dep_info = {}

        for child in element:
            tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if child.text:
                dep_info[tag.lower()] = child.text.strip()

        if 'groupid' in dep_info and 'artifactid' in dep_info:
            return dep_info

        return None

    def _extract_xml_header(
        self, content: str, lines: List[str], file_path: str
    ) -> Optional[Chunk]:
        """Extract XML declaration and DOCTYPE as a header chunk."""
        header_lines: List[int] = []
        header_end = 0

        # Check for XML declaration
        xml_match = self.xml_decl_pattern.search(content)
        if xml_match:
            line_num = content[: xml_match.end()].count('\n')
            header_lines.extend(range(0, line_num + 1))
            header_end = line_num

        # Check for DOCTYPE
        doctype_match = self.doctype_pattern.search(content)
        if doctype_match:
            start_line = content[: doctype_match.start()].count('\n')
            end_line = content[: doctype_match.end()].count('\n')
            header_lines.extend(range(start_line, end_line + 1))
            header_end = max(header_end, end_line)

        if not header_lines:
            return None

        # Create header chunk
        header_content = '\n'.join(lines[: header_end + 1])

        chunk = self._create_chunk(
            content=header_content,
            chunk_type=ChunkType.IMPORTS,  # Headers are like imports
            file_path=file_path,
            start_line=1,
            end_line=header_end + 1,
            name='xml_header',
        )

        # Extract header metadata
        metadata: Dict[str, Any] = {
            'has_xml_declaration': xml_match is not None,
            'xml_version': xml_match.group(1) if xml_match else None,
            'encoding': xml_match.group(2) if xml_match and xml_match.group(2) else 'UTF-8',
            'has_doctype': doctype_match is not None,
            'doctype_root': doctype_match.group(1) if doctype_match else None,
            'doctype_public': (
                doctype_match.group(2) if doctype_match and doctype_match.group(2) else None
            ),
            'schemas': {},
        }

        # Check for schema references
        for schema_type, pattern in self.schema_patterns.items():
            match = pattern.search(header_content)
            if match and isinstance(metadata['schemas'], dict):
                metadata['schemas'][schema_type] = match.group(1)

        chunk.metadata.language_specific = metadata

        return chunk

    def _extract_elements_by_pattern(
        self, content: str, lines: List[str], file_path: str, processed_lines: set, file_type: str
    ) -> List[Chunk]:
        """Extract elements using regex patterns (fallback method)."""
        chunks = []

        # Find major elements
        element_stack: List[Dict[str, Any]] = []  # Track nested elements
        current_element = None

        for i, line in enumerate(lines):
            if i in processed_lines:
                continue

            # Check for element start
            start_match = re.match(r'^\s*<(\w+(?::\w+)?)((?:\s+[\w:.-]+\s*=\s*"[^"]*")*)\s*>', line)
            if start_match:
                tag_name = start_match.group(1)
                attrs_str = start_match.group(2)

                # Start new element
                element_info = {
                    'tag': tag_name,
                    'start_line': i,
                    'end_line': None,
                    'attrs': attrs_str,
                    'content_lines': [line],
                }

                # Check if it's a significant element
                if tag_name.lower() in self.element_type_mappings or len(element_stack) == 0:
                    current_element = element_info

                element_stack.append(element_info)

            # Check for element end
            end_match = re.match(r'^\s*</(\w+(?::\w+)?)\s*>', line)
            if end_match and element_stack:
                tag_name = end_match.group(1)

                # Find matching start tag
                for j in range(len(element_stack) - 1, -1, -1):
                    if element_stack[j]['tag'] == tag_name:
                        element_stack[j]['end_line'] = i
                        if isinstance(element_stack[j]['content_lines'], list):
                            element_stack[j]['content_lines'].append(line)

                        # If this was our current element, create chunk
                        if current_element and current_element['tag'] == tag_name:
                            chunk = self._create_element_chunk_from_pattern(
                                current_element, lines, file_path, file_type
                            )
                            if chunk:
                                chunks.append(chunk)
                                logger.info("[UNTESTED PATH] xml element chunk from pattern")

                                # Mark lines as processed
                                start_line = current_element.get('start_line')
                                end_line = current_element.get('end_line')
                                if isinstance(start_line, int) and isinstance(end_line, int):
                                    for k in range(start_line, end_line + 1):
                                        processed_lines.add(k)

                            current_element = None

                        # Remove completed elements
                        element_stack = element_stack[:j]
                        break

            # Add line to current element
            elif current_element:
                start_line = current_element.get('start_line')
                if isinstance(start_line, int) and i > start_line:
                    if isinstance(current_element['content_lines'], list):
                        current_element['content_lines'].append(line)

        return chunks

    def _create_element_chunk_from_pattern(
        self, element_info: Dict[str, Any], lines: List[str], file_path: str, file_type: str
    ) -> Optional[Chunk]:
        """Create chunk from element found by pattern matching."""
        content = '\n'.join(element_info['content_lines'])

        # Skip if too small
        if len(content) < 100:
            return None

        # Validate start_line and end_line
        start_line = element_info.get('start_line')
        end_line = element_info.get('end_line')
        if not isinstance(start_line, int) or not isinstance(end_line, int):
            return None

        tag_name = element_info['tag']
        chunk_type = self.element_type_mappings.get(
            tag_name.lower(), ChunkType.MODULE if len(content) > 500 else ChunkType.UNKNOWN
        )

        chunk = self._create_chunk(
            content=content,
            chunk_type=chunk_type,
            file_path=file_path,
            start_line=start_line + 1,
            end_line=end_line + 1,
            name=tag_name,
        )

        # Extract comprehensive metadata using pattern matching
        metadata = {
            'tag_name': tag_name,
            'namespace': None,
            'attributes': self._parse_attributes_string(element_info['attrs']),
            'depth': 0,  # Unknown in pattern mode
            'child_count': 0,  # Not calculated in pattern mode
            'text_content': '',
            'has_cdata': '<![CDATA[' in content,
            'namespaces': {},
            'complexity': {
                'total_elements': 1,
                'max_depth': 0,
                'attribute_count': len(self._parse_attributes_string(element_info['attrs'])),
                'has_mixed_content': False,
            },
            'patterns': {
                'type': file_type,
                'uses': [],
                'anti_patterns': [],
            },
            'security': [],
            'todos': [],
            'dependencies': [],
            'parsing_method': 'pattern',
        }

        # Extract namespaces from attributes
        for attr_name, attr_value in metadata['attributes'].items():
            if attr_name.startswith('xmlns'):
                ns_prefix = attr_name.split(':')[1] if ':' in attr_name else 'default'
                metadata['namespaces'][ns_prefix] = attr_value

        # Security analysis on content
        all_text = content.lower()

        # Password detection
        if 'password' in all_text:
            password_matches = self.security_patterns['password'].findall(content)
            for match in password_matches:
                if match and match != '${' and not match.startswith('${'):
                    metadata['security'].append(
                        {
                            'type': 'hardcoded_password',
                            'severity': 'critical',
                            'location': f'element:{tag_name}',
                            'description': 'Hardcoded password detected',
                        }
                    )

        # Credential detection
        credential_matches = self.security_patterns['credential'].findall(content)
        for match in credential_matches:
            if isinstance(match, tuple) and len(match) > 1 and match[1] and len(match[1]) > 10:
                metadata['security'].append(
                    {
                        'type': 'exposed_credential',
                        'severity': 'high',
                        'location': f'element:{tag_name}',
                        'description': f'Potential {match[0]} exposure',
                    }
                )

        # File path detection
        if (
            'file://' in content
            or 'c:\\' in all_text
            or '/home/' in all_text
            or '/var/' in all_text
        ):
            metadata['security'].append(
                {
                    'type': 'absolute_path',
                    'severity': 'medium',
                    'location': f'element:{tag_name}',
                    'description': 'Absolute file path detected',
                }
            )

        # TO-DO extraction from comments
        for comment_match in self.comment_pattern.finditer(content):
            comment_text = comment_match.group(1)
            todo_match = re.search(
                r'(TODO|FIXME|HACK|XXX|BUG|OPTIMIZE|NOTE)[:\s](.+)', comment_text, re.IGNORECASE
            )
            if todo_match:
                metadata['todos'].append(f"{todo_match.group(1)}: {todo_match.group(2).strip()}")

        # Framework detection based on file type
        if file_type == 'spring_config' and 'bean' in tag_name.lower():
            if isinstance(metadata['patterns'], dict) and isinstance(
                metadata['patterns'].get('uses'), list
            ):
                metadata['patterns']['uses'].append('spring_framework')
        elif file_type == 'maven_project':
            if isinstance(metadata['patterns'], dict) and isinstance(
                metadata['patterns'].get('uses'), list
            ):
                metadata['patterns']['uses'].append('maven_build')
            # Extract Maven dependencies
            if tag_name.lower() == 'dependency':
                dep_info = {}
                # Simple extraction from content
                for dep_tag in ['groupId', 'artifactId', 'version']:
                    pattern = f'<{dep_tag}>([^<]+)</{dep_tag}>'
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        dep_info[dep_tag.lower()] = match.group(1).strip()
                if 'groupid' in dep_info and 'artifactid' in dep_info:
                    if isinstance(metadata['dependencies'], list):
                        metadata['dependencies'].append(dep_info)

        chunk.metadata.language_specific = metadata

        return chunk

    def _extract_element_content(self, content: str, start_pos: int, tag_name: str) -> str:
        """Extract element content using stack-based parsing to handle nested tags."""
        # Find the end of the opening tag
        tag_end = content.find('>', start_pos)
        if tag_end == -1:
            return ""

        # Check if it's a self-closing tag
        if content[tag_end - 1] == '/':
            return content[start_pos : tag_end + 1]

        # Stack-based parsing to find matching closing tag
        stack = 1
        pos = tag_end + 1

        while stack > 0 and pos < len(content):
            # Look for next tag
            next_tag = content.find('<', pos)
            if next_tag == -1:
                break

            # Check if it's a closing tag
            if content[next_tag + 1] == '/':
                # Extract tag name
                close_tag_end = content.find('>', next_tag)
                if close_tag_end != -1:
                    close_tag_name = content[next_tag + 2 : close_tag_end].strip().split()[0]
                    if close_tag_name == tag_name or close_tag_name.endswith(':' + tag_name):
                        stack -= 1
                        if stack == 0:
                            return content[start_pos : close_tag_end + 1]
                pos = close_tag_end + 1
            else:
                # Check if it's an opening tag of the same type
                tag_name_end = next_tag + 1
                while tag_name_end < len(content) and content[tag_name_end] not in ' >/':
                    tag_name_end += 1
                logger.info("[UNTESTED PATH] xml nested tag stack increment")

                found_tag_name = content[next_tag + 1 : tag_name_end]
                if found_tag_name == tag_name or found_tag_name.endswith(':' + tag_name):
                    stack += 1

                pos = content.find('>', next_tag) + 1

        return ""

    def _parse_attributes_string(self, attrs_str: str) -> Dict[str, str]:
        """Parse attributes from string."""
        attrs: Dict[str, str] = {}

        if not attrs_str:
            return attrs

        # Pattern for name="value" pairs (including hyphens)
        attr_pattern = re.compile(r'([\w:.-]+)\s*=\s*"([^"]*)"')

        for match in attr_pattern.finditer(attrs_str):
            attrs[match.group(1)] = match.group(2)

        return attrs
