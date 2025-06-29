"""
Hybrid search implementation combining semantic (70%) and lexical (30%) search.

This module implements the CRITICAL hybrid search that is the heart of the RAG system.
ConversationService and other modules DEPEND on this implementation.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from acolyte.core.logging import logger
from acolyte.core.secure_config import Settings
from acolyte.core.token_counter import SmartTokenCounter
from acolyte.models.chunk import Chunk, ChunkType, ChunkMetadata
from acolyte.rag.compression import ContextualCompressor
from acolyte.rag.retrieval.fuzzy_matcher import get_fuzzy_matcher
from acolyte.rag.retrieval.cache import SearchCache


@dataclass
class ScoredChunk:
    """Chunk with relevance score."""

    chunk: Chunk
    score: float
    source: str = ""  # 'semantic', 'lexical', or 'hybrid'


@dataclass
class SearchFilters:
    """Optional filters for search."""

    file_path: Optional[str] = None
    file_types: Optional[List[str]] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    chunk_types: Optional[List[str]] = None


class HybridSearch:
    """
    Implements hybrid search 70/30 for code retrieval.

    This is the ONLY search implementation in ACOLYTE.
    Other modules must use this class, not reimplement.
    """

    def __init__(
        self,
        weaviate_client,
        lexical_index=None,  # TBD: Actual lexical search implementation
        semantic_weight: float = 0.7,
        lexical_weight: float = 0.3,
        enable_compression: bool = True,
    ):
        """
        Initialize hybrid search.

        Args:
            weaviate_client: Weaviate client for semantic search
            lexical_index: Lexical search implementation (TBD)
            semantic_weight: Weight for semantic results (default 0.7)
            lexical_weight: Weight for lexical results (default 0.3)
            enable_compression: Enable contextual compression
        """
        self.weaviate_client = weaviate_client
        self.lexical_index = lexical_index
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight

        # Ensure weights sum to 1.0
        total_weight = semantic_weight + lexical_weight
        if abs(total_weight - 1.0) > 0.001:
            logger.warning("Weights don't sum to 1.0, normalizing", total_weight=total_weight)
            self.semantic_weight = semantic_weight / total_weight
            self.lexical_weight = lexical_weight / total_weight

        # Initialize compression if enabled
        self.enable_compression = enable_compression
        if enable_compression:
            self.token_counter = SmartTokenCounter()
            self.compressor = ContextualCompressor(token_counter=self.token_counter)
        else:
            self.compressor = None

        # Initialize proper LRU cache with TTL
        config = Settings()
        self.cache = SearchCache(
            max_size=config.get("cache.max_size", 1000), ttl=config.get("cache.ttl_seconds", 3600)
        )

        logger.info(
            "HybridSearch initialized",
            semantic_weight=self.semantic_weight,
            lexical_weight=self.lexical_weight,
            compression="enabled" if enable_compression else "disabled",
        )

    async def search(
        self, query: str, max_chunks: int = 10, filters: Optional[SearchFilters] = None
    ) -> List[ScoredChunk]:
        """
        Perform hybrid search without compression.

        Args:
            query: Search query text
            max_chunks: Maximum number of results
            filters: Optional search filters

        Returns:
            List of chunks sorted by hybrid score
        """
        # Convert filters to dict for cache key
        filters_dict = filters.__dict__ if filters else None

        # Try to get from cache first
        cached_chunks = self.cache.get(query, filters_dict)
        if cached_chunks is not None:
            # Convert cached Chunks back to ScoredChunks
            # Since we don't store scores, mark them as cached with score 1.0
            cached_results = [
                ScoredChunk(chunk=chunk, score=1.0, source="cached")
                for chunk in cached_chunks[:max_chunks]
            ]
            logger.debug("Cache hit for query", query=query[:50])
            return cached_results

        # Get more results than needed for better combination
        search_limit = max_chunks * 2

        # Perform both searches in parallel
        semantic_results = await self._semantic_search(query, search_limit, filters)
        lexical_results = await self._lexical_search(query, search_limit, filters)

        # Combine results with weights
        combined_results = self._combine_results(semantic_results, lexical_results)

        # Sort by final score and return top results
        combined_results.sort(key=lambda x: x.score, reverse=True)
        final_results = combined_results[:max_chunks]

        # Cache the chunks (not the scores)
        chunks_to_cache = [
            r.chunk for r in combined_results[:search_limit]
        ]  # Cache more for flexibility
        self.cache.set(query, chunks_to_cache, filters_dict)

        logger.debug(
            "Search completed",
            query=query[:50],
            semantic=len(semantic_results),
            lexical=len(lexical_results),
            combined=len(combined_results),
            returned=len(final_results),
        )

        return final_results

    async def search_with_compression(
        self,
        query: str,
        max_chunks: int = 10,
        token_budget: Optional[int] = None,
        compression_ratio: Optional[float] = None,
        filters: Optional[SearchFilters] = None,
    ) -> List[Chunk]:
        """
        Hybrid search with contextual compression.

        Args:
            query: User query
            max_chunks: Maximum chunks to return
            token_budget: Token budget (if not provided, uses ratio)
            compression_ratio: Target compression ratio (default from config)
            filters: Optional search filters

        Returns:
            List of chunks, possibly compressed
        """
        if not self.enable_compression or not self.compressor:
            # If compression disabled, use normal search
            scored_results = await self.search(query, max_chunks, filters)
            return [r.chunk for r in scored_results]

        # Create cache key for compressed results
        # Note: We cache the compressed results separately with budget info
        cache_key = f"compressed:{query}:{max_chunks}:{token_budget}"
        filters_dict = filters.__dict__ if filters else None

        # Try cache first
        cached_compressed = self.cache.get(cache_key, filters_dict)
        if cached_compressed is not None:
            logger.debug("Cache hit for compressed query", query=query[:50])
            return cached_compressed[:max_chunks]

        # Get configuration
        config = Settings()
        compression_config = config.get("rag.compression", {})

        if compression_ratio is None:
            compression_ratio = compression_config.get("ratio", 0.7)

        # Search for more chunks than needed (to have margin)
        search_multiplier = compression_config.get("search_multiplier", 1.5)
        scored_results = await self.search(
            query=query, max_chunks=int(max_chunks * search_multiplier), filters=filters
        )

        # Extract chunks from scored results
        raw_chunks = [r.chunk for r in scored_results]

        # If no token_budget, calculate based on ratio
        if token_budget is None:
            # Estimate average tokens per chunk
            avg_chunk_size = compression_config.get("avg_chunk_tokens", 200)
            token_budget = int(max_chunks * avg_chunk_size * compression_ratio)

        # Decide if compression is needed
        if not self.compressor.should_compress(query, raw_chunks, token_budget):
            # Not worth compressing, return first max_chunks
            uncompressed_results = raw_chunks[:max_chunks]
            # Cache even uncompressed results
            self.cache.set(cache_key, uncompressed_results, filters_dict)
            return uncompressed_results

        # Compress intelligently
        compressed_chunks, compression_result = self.compressor.compress_chunks(
            chunks=raw_chunks, query=query, token_budget=token_budget
        )

        # Cache compressed results
        self.cache.set(cache_key, compressed_chunks, filters_dict)

        # Log compression metrics
        logger.info(
            "Compression applied",
            query_type=compression_result.query_type,
            tokens_saved=compression_result.tokens_saved,
            compression_ratio=compression_result.compression_ratio,
        )

        return compressed_chunks

    async def _semantic_search(
        self, query: str, limit: int, filters: Optional[SearchFilters] = None
    ) -> List[ScoredChunk]:
        """
        Perform semantic search using embeddings.

        Uses query embedding to find similar chunks in vector space.
        """
        try:
            # Get query embedding from embeddings service
            from acolyte.embeddings import get_embeddings

            embedder = get_embeddings()

            # encode() is NOT async, it's a regular method
            query_embedding = embedder.encode(query)

            # Convert to Weaviate format (float64)
            query_vector = query_embedding.to_weaviate()

            # Build Weaviate query
            query_builder = (
                self.weaviate_client.query.get(
                    "CodeChunk",
                    [
                        "content",
                        "file_path",
                        "chunk_type",
                        "start_line",
                        "end_line",
                        "language",
                        "git_last_author",
                        "git_last_modified",
                    ],
                )
                .with_near_vector(
                    {"vector": query_vector, "certainty": 0.7}  # Minimum similarity threshold
                )
                .with_limit(limit)
                .with_additional(["certainty"])  # To get the score
            )

            # Apply filters if provided
            if filters:
                where_conditions = []

                if filters.file_path:
                    where_conditions.append(
                        {
                            "path": ["file_path"],
                            "operator": "Equal",
                            "valueString": filters.file_path,
                        }
                    )

                if filters.chunk_types:
                    where_conditions.append(
                        {
                            "path": ["chunk_type"],
                            "operator": "In",
                            "valueStringArray": [ct.upper() for ct in filters.chunk_types],
                        }
                    )

                if filters.file_types:
                    # File types filter based on file extension
                    where_conditions.append(
                        {
                            "path": ["language"],
                            "operator": "In",
                            "valueStringArray": filters.file_types,
                        }
                    )

                if where_conditions:
                    if len(where_conditions) > 1:
                        logger.info("[UNTESTED PATH] Multiple where conditions in semantic search")
                        where_clause = {"operator": "And", "operands": where_conditions}
                    else:
                        where_clause = where_conditions[0]
                    query_builder = query_builder.with_where(where_clause)

            # Execute search
            results = query_builder.do()

            # Convert to ScoredChunks
            scored_chunks = []

            # Verify response structure
            code_chunks = results.get("data", {}).get("Get", {}).get("CodeChunk", [])

            for item in code_chunks:
                # Create Chunk object from result
                metadata = ChunkMetadata(
                    file_path=item.get("file_path", ""),
                    language=item.get("language", "unknown"),
                    start_line=item.get("start_line", 1),
                    end_line=item.get("end_line", 1),
                    chunk_type=ChunkType(item.get("chunk_type", "unknown")),
                    name=item.get("name"),
                    last_modified=item.get("git_last_modified"),
                )

                chunk = Chunk(content=item.get("content", ""), metadata=metadata)

                # Extract similarity score
                score = item.get("_additional", {}).get("certainty", 0.0)

                scored_chunks.append(ScoredChunk(chunk=chunk, score=score, source="semantic"))

            logger.debug(
                "Semantic search completed",
                query=query[:50],
                limit=limit,
                results=len(scored_chunks),
            )

            return scored_chunks

        except ImportError as e:
            logger.info("[UNTESTED PATH] Failed to import embeddings service")
            logger.error("Failed to import embeddings service", error=str(e))
            return []
        except Exception as e:
            logger.info("[UNTESTED PATH] Semantic search failed with exception")
            logger.error("Semantic search failed", error=str(e))
            return []

    async def _lexical_search(
        self, query: str, limit: int, filters: Optional[SearchFilters] = None
    ) -> List[ScoredChunk]:
        """
        Perform lexical search for exact term matches.

        Uses Weaviate BM25 search with fuzzy query expansion to find chunks
        containing query terms regardless of naming convention.
        """
        try:
            # Expand query with fuzzy variations
            fuzzy_matcher = get_fuzzy_matcher()
            query_variations = fuzzy_matcher.expand_query(query)

            # Collect all results from variations
            all_scored_chunks = []

            for i, variation in enumerate(query_variations):
                # Reduce weight for variations (original gets full weight)
                variation_weight = 1.0 if i == 0 else 0.8

                # Use Weaviate BM25 search
                query_builder = (
                    self.weaviate_client.query.get(
                        "CodeChunk",
                        [
                            "content",
                            "file_path",
                            "chunk_type",
                            "start_line",
                            "end_line",
                            "language",
                            "git_last_author",
                            "git_last_modified",
                        ],
                    )
                    .with_bm25(
                        query=variation,
                        properties=["content", "file_path"],  # Search in these fields
                    )
                    .with_limit(limit // len(query_variations) + 1)  # Distribute limit
                    .with_additional(["score"])  # Get BM25 score
                )

                # Apply filters if provided
                if filters:
                    where_conditions = []

                    if filters.file_path:
                        where_conditions.append(
                            {
                                "path": ["file_path"],
                                "operator": "Equal",
                                "valueString": filters.file_path,
                            }
                        )

                    if filters.chunk_types:
                        where_conditions.append(
                            {
                                "path": ["chunk_type"],
                                "operator": "In",
                                "valueStringArray": [ct.upper() for ct in filters.chunk_types],
                            }
                        )

                    if filters.file_types:
                        where_conditions.append(
                            {
                                "path": ["language"],
                                "operator": "In",
                                "valueStringArray": filters.file_types,
                            }
                        )

                    if where_conditions:
                        if len(where_conditions) > 1:
                            logger.info(
                                "[UNTESTED PATH] Multiple where conditions in lexical search"
                            )
                            where_clause = {"operator": "And", "operands": where_conditions}
                        else:
                            where_clause = where_conditions[0]
                        query_builder = query_builder.with_where(where_clause)

                # Execute search
                results = query_builder.do()

                # Process results for this variation
                code_chunks = results.get("data", {}).get("Get", {}).get("CodeChunk", [])

                for item in code_chunks:
                    # Create Chunk object from result
                    metadata = ChunkMetadata(
                        file_path=item.get("file_path", ""),
                        language=item.get("language", "unknown"),
                        start_line=item.get("start_line", 1),
                        end_line=item.get("end_line", 1),
                        chunk_type=ChunkType(item.get("chunk_type", "unknown")),
                        name=item.get("name"),
                        last_modified=item.get("git_last_modified"),
                    )

                    chunk = Chunk(content=item.get("content", ""), metadata=metadata)

                    # Extract BM25 score and apply variation weight
                    base_score = item.get("_additional", {}).get("score", 0.0)
                    weighted_score = base_score * variation_weight

                    all_scored_chunks.append(
                        ScoredChunk(chunk=chunk, score=weighted_score, source="lexical")
                    )

            # Deduplicate results based on chunk ID
            seen_chunks = {}
            for scored_chunk in all_scored_chunks:
                chunk_id = scored_chunk.chunk.id
                if chunk_id not in seen_chunks or scored_chunk.score > seen_chunks[chunk_id].score:
                    seen_chunks[chunk_id] = scored_chunk

            # Get unique results and sort by score
            unique_chunks = list(seen_chunks.values())
            unique_chunks.sort(key=lambda x: x.score, reverse=True)

            # Return top results up to limit
            final_results = unique_chunks[:limit]

            logger.debug(
                "Lexical search completed",
                query=query[:50],
                variations=len(query_variations),
                limit=limit,
                results=len(final_results),
            )

            return final_results

        except Exception as e:
            logger.info("[UNTESTED PATH] Lexical search failed with exception")
            logger.error("Lexical search failed", error=str(e))
            # Fallback to empty results on error
            return []

    def _combine_results(
        self, semantic_results: List[ScoredChunk], lexical_results: List[ScoredChunk]
    ) -> List[ScoredChunk]:
        """
        Combine semantic and lexical results with 70/30 weights.

        Handles deduplication and re-scoring when chunks appear in both.
        """
        # Normalize scores to [0, 1] range
        semantic_normalized = self._normalize_scores(semantic_results)
        lexical_normalized = self._normalize_scores(lexical_results)

        # Create dictionaries for efficient lookup
        semantic_dict = {r.chunk.id: r for r in semantic_normalized}
        lexical_dict = {r.chunk.id: r for r in lexical_normalized}

        # Combine results
        combined_dict: Dict[str, ScoredChunk] = {}

        # Process semantic results
        for chunk_id, result in semantic_dict.items():
            if chunk_id in lexical_dict:
                # Chunk appears in both - combine scores
                semantic_score = result.score * self.semantic_weight
                lexical_score = lexical_dict[chunk_id].score * self.lexical_weight
                combined_score = semantic_score + lexical_score

                combined_dict[chunk_id] = ScoredChunk(
                    chunk=result.chunk, score=combined_score, source="hybrid"
                )
            else:
                # Only in semantic
                combined_dict[chunk_id] = ScoredChunk(
                    chunk=result.chunk, score=result.score * self.semantic_weight, source="semantic"
                )

        # Process lexical-only results
        for chunk_id, result in lexical_dict.items():
            if chunk_id not in semantic_dict:
                combined_dict[chunk_id] = ScoredChunk(
                    chunk=result.chunk, score=result.score * self.lexical_weight, source="lexical"
                )

        return list(combined_dict.values())

    def _normalize_scores(self, results: List[ScoredChunk]) -> List[ScoredChunk]:
        """Normalize scores to [0, 1] range."""
        if not results:
            return results

        scores = [r.score for r in results]
        max_score = max(scores)
        min_score = min(scores)

        # If all scores are the same
        if max_score == min_score:
            return [ScoredChunk(chunk=r.chunk, score=1.0, source=r.source) for r in results]

        # Normalize to [0, 1]
        normalized = []
        for result in results:
            normalized_score = (result.score - min_score) / (max_score - min_score)
            normalized.append(
                ScoredChunk(chunk=result.chunk, score=normalized_score, source=result.source)
            )

        return normalized

    def invalidate_cache(self, pattern: Optional[str] = None):
        """
        Invalidate cache entries.

        Args:
            pattern: Optional pattern to match. If None or "*", invalidates all.
                    Otherwise, invalidates entries matching the pattern.
        """
        if not pattern or pattern == "*":
            # Clear entire cache
            stats = self.cache.get_stats()
            self.cache.clear()
            logger.info("Invalidated entire search cache", entries=stats['size'])
        else:
            # Delegate pattern-based invalidation to SearchCache
            self.cache.invalidate_by_pattern(pattern)
            logger.info("Invalidated cache entries matching pattern", pattern=pattern)

    def invalidate_cache_for_file(self, file_path: str):
        """
        Invalidate cache entries for a specific file.

        This is more precise than pattern-based invalidation.

        Args:
            file_path: Path of the modified file
        """
        self.cache.invalidate_by_file(file_path)
        logger.info("Invalidated cache entries for file", file_path=file_path)

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for monitoring.

        Returns:
            Dict with cache statistics including size, hit rate, etc.
        """
        return self.cache.get_stats()

    async def search_with_graph_expansion(
        self,
        query: str,
        max_results: int = 10,
        expansion_depth: int = 2,
        filters: Optional[SearchFilters] = None,
    ) -> List[Chunk]:
        """
        Hybrid search with neural graph expansion.

        THIS REPLACES search_with_clustering().

        Args:
            query: Search query
            max_results: Maximum results to return
            expansion_depth: Expansion depth in graph (default: 2 hops)
            filters: Optional filters

        Returns:
            Expanded and re-ranked chunks
        """
        # 1. Initial search to get "seeds"
        initial_results = await self.search(query, max_results // 3, filters)

        if not initial_results:
            return []

        # 2. Expand via neural graph
        from acolyte.rag.graph.neural_graph import NeuralGraph

        graph = NeuralGraph()

        expanded_chunks = []
        seen_paths = set()

        for scored_chunk in initial_results[:5]:  # Top 5 as seeds
            chunk = scored_chunk.chunk
            file_path = chunk.metadata.file_path

            # Find related files
            related_nodes = await graph.find_related(
                node=file_path, max_distance=expansion_depth, min_strength=0.3
            )

            # Load chunks from related files
            for node in related_nodes:
                if node['path'] not in seen_paths:
                    seen_paths.add(node['path'])
                    file_chunks = await self._load_chunks_from_file(node['path'])
                    expanded_chunks.extend(file_chunks)

        # 3. Combine original results + expanded
        all_chunks = [r.chunk for r in initial_results] + expanded_chunks

        # 4. Re-rank all by relevance to query
        reranked = await self._rerank_by_relevance(query, all_chunks, max_results)

        logger.info(
            "Graph expansion results",
            seeds=len(initial_results),
            total_chunks=len(all_chunks),
            final_results=len(reranked),
        )

        return reranked

    async def _load_chunks_from_file(self, file_path: str) -> List[Chunk]:
        """
        Load all chunks from a specific file from Weaviate.

        Args:
            file_path: Path of the file to load chunks from

        Returns:
            List of chunks from the file
        """
        logger.info("[UNTESTED PATH] _load_chunks_from_file method called")
        try:
            # Query Weaviate for all chunks from this file
            query_builder = (
                self.weaviate_client.query.get(
                    "CodeChunk",
                    [
                        "content",
                        "file_path",
                        "chunk_type",
                        "start_line",
                        "end_line",
                        "language",
                        "name",
                        "git_last_author",
                        "git_last_modified",
                    ],
                )
                .with_where({"path": ["file_path"], "operator": "Equal", "valueString": file_path})
                .with_limit(100)  # Reasonable limit per file
            )

            # Execute query
            results = query_builder.do()

            # Convert results to chunks
            chunks = []
            code_chunks = results.get("data", {}).get("Get", {}).get("CodeChunk", [])

            for item in code_chunks:
                # Create ChunkMetadata
                metadata = ChunkMetadata(
                    file_path=item.get("file_path", ""),
                    language=item.get("language", "unknown"),
                    start_line=item.get("start_line", 1),
                    end_line=item.get("end_line", 1),
                    chunk_type=ChunkType(item.get("chunk_type", "unknown")),
                    name=item.get("name"),
                    last_modified=item.get("git_last_modified"),
                )

                # Create Chunk
                chunk = Chunk(content=item.get("content", ""), metadata=metadata)

                chunks.append(chunk)

            logger.debug("Loaded chunks from file", file_path=file_path, chunks=len(chunks))
            return chunks

        except Exception as e:
            logger.error("Failed to load chunks from file", file_path=file_path, error=str(e))
            return []

    async def _rerank_by_relevance(
        self, query: str, chunks: List[Chunk], max_results: int
    ) -> List[Chunk]:
        """
        Re-rank chunks by relevance to query using embeddings.

        Args:
            query: User query
            chunks: Chunks to re-rank
            max_results: Maximum number of results to return

        Returns:
            Top max_results chunks ordered by relevance
        """
        try:
            # Get embeddings service
            from acolyte.embeddings import get_embeddings

            embedder = get_embeddings()

            # Get query embedding
            query_embedding = embedder.encode(query)

            # Score each chunk
            scored_chunks = []
            for chunk in chunks:
                # Get chunk embedding
                chunk_text = chunk.to_search_text()
                chunk_embedding = embedder.encode(chunk_text)

                # Calculate similarity
                similarity = query_embedding.cosine_similarity(chunk_embedding)

                scored_chunks.append(ScoredChunk(chunk=chunk, score=similarity, source="reranked"))

            # Sort by score and return top results
            scored_chunks.sort(key=lambda x: x.score, reverse=True)

            # Return just the chunks (not ScoredChunk)
            return [sc.chunk for sc in scored_chunks[:max_results]]

        except Exception as e:
            logger.info("[UNTESTED PATH] Failed to re-rank chunks")
            logger.error("Failed to re-rank chunks", error=str(e))
            # Fallback: return original chunks truncated
            return chunks[:max_results]
