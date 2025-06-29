"""
GraphBuilder - Updates neural graph from enriched chunks.

Extracts relationships (imports, calls, extends) from code
and updates the graph database.

CREATED: To automatically update the neural graph (/rag/graph/) during
the enrichment process. This ensures the graph stays synchronized with
code changes without manual intervention.

INTEGRATION: Called automatically by EnrichmentService.enrich_chunks()
after Git metadata extraction to build/update code relationships.
"""

from typing import List, Dict, Any, Set
import re
import ast

from acolyte.core.logging import logger
from acolyte.core.tracing import MetricsCollector
from acolyte.rag.graph import NeuralGraph
from acolyte.models.chunk import Chunk, ChunkType


class GraphBuilder:
    """
    Builds and updates the neural graph from code analysis.

    Called by EnrichmentService after chunks are enriched.
    """

    def __init__(self):
        self.graph = NeuralGraph()
        self._processed_files: Set[str] = set()
        self.metrics = MetricsCollector()

    async def update_from_chunks(self, chunks: List[Chunk], metadata: Dict[str, Any]) -> None:
        """
        Update graph from enriched chunks.

        Args:
            chunks: List of code chunks to analyze
            metadata: Enrichment metadata (includes Git info)
        """
        logger.info(f"Updating graph from {len(chunks)} chunks")
        self.metrics.increment("rag.graph_builder.update_calls")
        self.metrics.gauge("rag.graph_builder.chunks_per_update", len(chunks))

        # Track files for co-modification updates
        files_in_batch = {chunk.metadata.file_path for chunk in chunks}

        for chunk in chunks:
            try:
                await self._process_chunk(chunk, metadata)
            except Exception as e:
                logger.error(f"Error processing chunk for graph: {e}")
                continue

        # Update co-modification patterns if Git metadata available
        if metadata.get("git", {}).get("co_modified_with"):
            await self._update_co_modifications(files_in_batch, metadata["git"])

        logger.info("Graph update completed")

    async def _process_chunk(self, chunk: Chunk, metadata: Dict[str, Any]) -> None:
        """Process a single chunk to extract relationships."""
        file_path = chunk.metadata.file_path
        logger.info(
            f"Processing chunk: file_path={file_path}, chunk_type={chunk.metadata.chunk_type}"
        )

        chunk_type = chunk.metadata.chunk_type
        if isinstance(chunk_type, ChunkType):
            chunk_type = chunk_type.value

        # Add file node if not processed
        if file_path not in self._processed_files:
            await self.graph.add_node("FILE", file_path, file_path.split("/")[-1])
            self._processed_files.add(file_path)
            self.metrics.increment("rag.graph_builder.nodes_created", labels={"type": "FILE"})

        # Extract relationships based on chunk type
        logger.info(
            f"About to check chunk type: {chunk_type} == 'function'? {chunk_type == 'function'}"
        )

        if chunk_type == ChunkType.FUNCTION.value or chunk_type == ChunkType.METHOD.value:
            logger.info("Processing as function")
            await self._process_function(chunk, file_path)
        elif chunk_type == ChunkType.CLASS.value:
            await self._process_class(chunk, file_path)
        elif chunk_type == ChunkType.MODULE.value:
            await self._process_module(chunk, file_path)
        elif chunk_type == ChunkType.IMPORTS.value:
            await self._process_imports(chunk, file_path)
        else:
            logger.info(f"Chunk type {chunk_type} not handled")

    async def _process_function(self, chunk: Chunk, file_path: str) -> None:
        """Extract function relationships."""
        logger.info(f"_process_function called for {file_path}")
        func_name = chunk.metadata.name or "unknown_function"
        func_path = f"{file_path}::{func_name}"
        logger.info(f"About to add FUNCTION node: {func_path}")

        # Add function node
        await self.graph.add_node("FUNCTION", func_path, func_name)
        self.metrics.increment("rag.graph_builder.nodes_created", labels={"type": "FUNCTION"})

        # Add CONTAINS relationship
        await self.graph.add_edge(file_path, func_path, "CONTAINS")
        self.metrics.increment("rag.graph_builder.edges_created", labels={"type": "CONTAINS"})

        # Extract function calls using simple pattern matching
        # (More sophisticated AST parsing would be better but requires language detection)
        call_pattern = r'(\w+)\s*\('
        calls = re.findall(call_pattern, chunk.content)

        for called_func in set(calls):
            if called_func not in ['if', 'for', 'while', 'return', 'print']:  # Skip keywords
                called_path = f"unknown::{called_func}"
                # Create unknown node first to avoid NotFoundError
                await self.graph.add_node(
                    "FUNCTION",
                    called_path,
                    called_func,
                    metadata={"inferred": True, "incomplete": True, "source": "function_call"},
                )
                await self.graph.add_edge(func_path, called_path, "CALLS")

    async def _process_class(self, chunk: Chunk, file_path: str) -> None:
        """Extract class relationships."""
        class_name = chunk.metadata.name or "unknown_class"
        class_path = f"{file_path}::{class_name}"

        # Add class node
        await self.graph.add_node("CLASS", class_path, class_name)
        self.metrics.increment("rag.graph_builder.nodes_created", labels={"type": "CLASS"})

        # Add CONTAINS relationship
        await self.graph.add_edge(file_path, class_path, "CONTAINS")
        self.metrics.increment("rag.graph_builder.edges_created", labels={"type": "CONTAINS"})

        # Try to extract inheritance (simple pattern)
        inherit_pattern = r'class\s+\w+\s*\(([^)]+)\)'
        match = re.search(inherit_pattern, chunk.content)

        if match:
            parents = match.group(1).split(',')
            for parent in parents:
                parent_name = parent.strip()
                if parent_name and parent_name != 'object':
                    parent_path = f"unknown::{parent_name}"
                    # Create unknown parent class node first
                    await self.graph.add_node(
                        "CLASS",
                        parent_path,
                        parent_name,
                        metadata={"inferred": True, "incomplete": True, "source": "inheritance"},
                    )
                    await self.graph.add_edge(class_path, parent_path, "EXTENDS")

    async def _process_module(self, chunk: Chunk, file_path: str) -> None:
        """Process module-level code."""
        # For Python, extract imports
        await self._process_imports(chunk, file_path)

    async def _process_imports(self, chunk: Chunk, file_path: str) -> None:
        """Extract import relationships."""
        content = chunk.content

        # Python imports
        import_patterns = [r'from\s+([\w.]+)\s+import', r'import\s+([\w.]+)']

        for pattern in import_patterns:
            imports = re.findall(pattern, content)
            for module in imports:
                # Convert module to approximate file path
                module_path = module.replace('.', '/')
                if not module_path.endswith('.py'):
                    module_path += '.py'

                # Create module node if it doesn't exist
                await self.graph.add_node(
                    "MODULE", module_path, module, metadata={"inferred": True, "from_import": True}
                )
                await self.graph.add_edge(
                    file_path, module_path, "IMPORTS", metadata={"import_type": "python"}
                )

    async def _update_co_modifications(
        self, files_in_batch: Set[str], git_metadata: Dict[str, Any]
    ) -> None:
        """Update co-modification relationships from Git data."""
        co_modified = git_metadata.get("co_modified_with", [])

        for file_path in files_in_batch:
            for co_file in co_modified:
                if co_file != file_path:
                    # Add or strengthen co-modification edge
                    await self.graph.add_edge(
                        file_path, co_file, "MODIFIES_TOGETHER", discovered_by="GIT_ACTIVITY"
                    )

    async def extract_relationships_from_ast(
        self, chunk: Chunk, language: str = "python"
    ) -> Dict[str, List[str]]:
        """
        Extract relationships using AST parsing (more accurate).

        Currently only supports Python. Other languages would need
        their own parsers.

        WARNING: This method only extracts relationship names. If you use
        these to create edges, ensure target nodes exist first or create
        them as 'unknown' nodes to avoid NotFoundError.
        """
        if language != "python":
            return {}

        try:
            tree = ast.parse(chunk.content)
            relationships: Dict[str, List[str]] = {"imports": [], "calls": [], "extends": []}

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        relationships["imports"].append(alias.name)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        relationships["imports"].append(node.module)

                elif isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            relationships["extends"].append(base.id)

                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        relationships["calls"].append(node.func.id)

            return relationships

        except SyntaxError:
            # [UNTESTED PATH] SyntaxError when parsing AST - line 196, 222-224 not covered
            logger.debug("Could not parse chunk as Python AST")
            return {}
