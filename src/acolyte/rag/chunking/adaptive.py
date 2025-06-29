"""
Adaptive chunking for ACOLYTE.
Adjusts chunking parameters based on content analysis using tree-sitter AST.
"""

from typing import List
from dataclasses import dataclass
from enum import Enum

from acolyte.models.chunk import Chunk
from acolyte.core.logging import logger
from acolyte.core.secure_config import Settings
from acolyte.rag.chunking.factory import ChunkerFactory


class ChunkingStrategy(Enum):
    """Chunking strategies based on code characteristics."""

    DENSE = "dense"  # Dense code - smaller chunks
    SPARSE = "sparse"  # Sparse code - larger chunks
    BALANCED = "balanced"  # Normal code - default sizes
    DOCUMENTATION = "documentation"  # Docs/comments heavy


@dataclass
class ContentAnalysis:
    """AST-based content analysis for adaptive decisions."""

    total_nodes: int
    function_nodes: int
    class_nodes: int
    import_nodes: int
    comment_ratio: float
    avg_function_size: float
    max_function_size: int
    has_complex_nesting: bool
    is_test_file: bool
    recommended_strategy: ChunkingStrategy


class AdaptiveChunker:
    """
    Adaptive chunker that adjusts parameters based on AST analysis.

    Does NOT inherit from BaseChunker to avoid initialization issues.
    Instead, delegates to language-specific chunkers dynamically.
    """

    def __init__(self):
        """Initialize adaptive chunker."""
        self.config = Settings()
        self._current_strategy = ChunkingStrategy.BALANCED
        self._current_language = None
        self._current_chunker = None

    async def chunk(self, content: str, file_path: str) -> List[Chunk]:
        """
        Adaptively chunk content using tree-sitter AST analysis.

        Process:
        1. Detect language and set up appropriate tree-sitter parser
        2. Parse content to AST
        3. Analyze AST structure
        4. Adjust chunking parameters
        5. Use parent's chunk method with adjusted params
        """
        if not content.strip():
            logger.warning(f"Empty content for {file_path}")
            return []

        # Detect language and get appropriate chunker
        language = ChunkerFactory.detect_language(file_path, content)
        self._current_language = language

        # Get the appropriate language chunker
        self._current_chunker = ChunkerFactory._get_language_chunker(language)
        if not self._current_chunker:
            # Create a default chunker for fallback
            logger.warning(f"No tree-sitter support for {language}, using default chunker")
            from .languages.default import DefaultChunker

            self._current_chunker = DefaultChunker()
            return await self._current_chunker.chunk(content, file_path)

        # Store original parameters
        self._base_chunk_size = self._current_chunker.chunk_size
        self._base_overlap = self._current_chunker.overlap

        # Parse with tree-sitter using the language chunker's parser
        tree = self._current_chunker.parser.parse(bytes(content, 'utf8'))

        # Analyze AST structure
        analysis = self._analyze_ast(tree, file_path)

        logger.info(
            f"Adaptive chunking {file_path}: "
            f"language={language}, "
            f"strategy={analysis.recommended_strategy.value}, "
            f"functions={analysis.function_nodes}, "
            f"classes={analysis.class_nodes}"
        )

        # Adjust parameters based on analysis
        self._adjust_parameters(analysis)

        # Use the language chunker with adjusted params
        chunks = await self._current_chunker.chunk(content, file_path)

        # Restore original parameters
        self._current_chunker.chunk_size = self._base_chunk_size
        self._current_chunker.overlap = self._base_overlap

        logger.info(f"Created {len(chunks)} adaptive chunks for {file_path}")

        return chunks

    def _analyze_ast(self, tree, file_path: str) -> ContentAnalysis:
        """
        Analyze AST to determine code characteristics.

        Uses tree-sitter AST instead of regex/string matching.
        """
        # Counters
        total_nodes = 0
        function_nodes = 0
        class_nodes = 0
        import_nodes = 0
        comment_nodes = 0
        function_sizes = []
        max_depth = 0

        # Walk the AST
        def walk_tree(node, depth=0):
            nonlocal total_nodes, function_nodes, class_nodes, import_nodes
            nonlocal comment_nodes, max_depth

            total_nodes += 1
            max_depth = max(max_depth, depth)

            node_type = node.type

            # Count different node types
            if 'function' in node_type or node_type in ['method_definition', 'arrow_function']:
                function_nodes += 1
                # Calculate function size
                start_line = node.start_point[0]
                end_line = node.end_point[0]
                function_sizes.append(end_line - start_line + 1)

            elif 'class' in node_type:
                class_nodes += 1

            elif 'import' in node_type or 'require' in node_type:
                import_nodes += 1

            elif 'comment' in node_type:
                comment_nodes += 1

            # Recurse
            for child in node.children:
                walk_tree(child, depth + 1)

        walk_tree(tree.root_node)

        # Calculate metrics
        comment_ratio = comment_nodes / max(total_nodes, 1)
        avg_function_size = (
            sum(function_sizes) / max(len(function_sizes), 1) if function_sizes else 0
        )
        max_function_size = max(function_sizes) if function_sizes else 0
        has_complex_nesting = max_depth > 10

        # Detect if test file
        path_lower = file_path.lower()
        is_test_file = any(
            pattern in path_lower for pattern in ['test', 'spec', '_test.', '.test.']
        )

        # Determine strategy based on AST analysis
        if is_test_file:
            strategy = ChunkingStrategy.BALANCED
        elif comment_ratio > 0.3:
            strategy = ChunkingStrategy.DOCUMENTATION
        elif avg_function_size > 50 or has_complex_nesting:
            strategy = ChunkingStrategy.DENSE
        elif function_nodes < 5 and class_nodes < 2:
            strategy = ChunkingStrategy.SPARSE
        else:
            strategy = ChunkingStrategy.BALANCED

        return ContentAnalysis(
            total_nodes=total_nodes,
            function_nodes=function_nodes,
            class_nodes=class_nodes,
            import_nodes=import_nodes,
            comment_ratio=comment_ratio,
            avg_function_size=avg_function_size,
            max_function_size=max_function_size,
            has_complex_nesting=has_complex_nesting,
            is_test_file=is_test_file,
            recommended_strategy=strategy,
        )

    def _adjust_parameters(self, analysis: ContentAnalysis) -> None:
        """
        Adjust chunking parameters based on AST analysis.

        Modifies the current chunker's parameters.
        """
        self._current_strategy = analysis.recommended_strategy

        # Apply adjustments to the current chunker
        if self._current_chunker:
            if analysis.recommended_strategy == ChunkingStrategy.DENSE:
                # Smaller chunks for dense/complex code
                self._current_chunker.chunk_size = int(self._base_chunk_size * 0.6)
                self._current_chunker.overlap = min(0.3, self._base_overlap * 1.5)

            elif analysis.recommended_strategy == ChunkingStrategy.SPARSE:
                # Larger chunks for sparse code
                self._current_chunker.chunk_size = int(self._base_chunk_size * 1.5)
                self._current_chunker.overlap = max(0.1, self._base_overlap * 0.5)

            elif analysis.recommended_strategy == ChunkingStrategy.DOCUMENTATION:
                # Moderate chunks for doc-heavy files
                self._current_chunker.chunk_size = int(self._base_chunk_size * 0.8)
                self._current_chunker.overlap = self._base_overlap

            # Special adjustments
            if analysis.is_test_file:
                # Test files often have many small functions
                self._current_chunker.chunk_size = min(self._current_chunker.chunk_size, 100)

            if analysis.max_function_size > 100:
                # Ensure we can handle very long functions
                self._current_chunker.chunk_size = max(self._current_chunker.chunk_size, 150)

        overlap_str = f"{self._current_chunker.overlap:.1f}" if self._current_chunker else "N/A"
        logger.debug(
            f"Adjusted chunking parameters: "
            f"size {self._base_chunk_size} -> {self._current_chunker.chunk_size if self._current_chunker else 'N/A'}, "
            f"overlap {self._base_overlap:.1f} -> {overlap_str} "
            f"for {analysis.recommended_strategy.value} strategy"
        )
