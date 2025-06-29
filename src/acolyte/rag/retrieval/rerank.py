"""
Simple re-ranking strategies for search results.

Re-orders search results based on additional signals like recency,
file modifications, and chunk types. No ML, just heuristics.
"""

from datetime import timezone
from typing import List, Optional
from collections import defaultdict

from acolyte.core.logging import logger
from acolyte.core.utils.datetime_utils import utc_now, parse_iso_datetime
from acolyte.models.chunk import ChunkType
from acolyte.rag.retrieval.hybrid_search import ScoredChunk


class SimpleReranker:
    """
    Basic re-ranking for search results.

    Applies simple heuristics to improve result ordering.
    Designed for mono-user local usage - no personalization needed.
    """

    # Priority order for chunk types (higher index = higher priority)
    CHUNK_TYPE_PRIORITY = {
        ChunkType.UNKNOWN: 0,
        ChunkType.COMMENT: 1,
        ChunkType.DOCSTRING: 2,
        ChunkType.CONSTANTS: 3,
        ChunkType.IMPORTS: 4,
        ChunkType.TYPES: 5,
        ChunkType.PROPERTY: 6,
        ChunkType.TESTS: 7,
        ChunkType.README: 8,
        ChunkType.MODULE: 9,
        ChunkType.NAMESPACE: 10,
        ChunkType.INTERFACE: 11,
        ChunkType.CONSTRUCTOR: 12,
        ChunkType.METHOD: 13,
        ChunkType.FUNCTION: 14,
        ChunkType.CLASS: 15,
        ChunkType.SUMMARY: 16,
        ChunkType.SUPER_SUMMARY: 17,
    }

    def rerank_by_recency(
        self, results: List[ScoredChunk], decay_factor: float = 0.95
    ) -> List[ScoredChunk]:
        """
        Boost recently modified files.

        Args:
            results: Scored chunks to rerank
            decay_factor: How much to decay score per day old

        Returns:
            Re-ranked results
        """
        reranked = []
        now = utc_now()

        for result in results:
            chunk = result.chunk
            score = result.score

            # Apply recency boost if we have modification date
            if chunk.metadata.last_modified:
                # Parse ISO string to datetime if it's a string
                if isinstance(chunk.metadata.last_modified, str):
                    modified_dt = parse_iso_datetime(chunk.metadata.last_modified)
                else:
                    modified_dt = chunk.metadata.last_modified
                    # Ensure datetime is timezone-aware
                    if modified_dt.tzinfo is None:
                        logger.info("[UNTESTED PATH] Datetime without timezone in rerank")
                        modified_dt = modified_dt.replace(tzinfo=timezone.utc)

                days_old = (now - modified_dt).days
                # Apply exponential decay
                recency_multiplier = decay_factor**days_old
                new_score = score * recency_multiplier

                reranked.append(ScoredChunk(chunk=chunk, score=new_score, source=result.source))
            else:
                # No date info, keep original score
                reranked.append(result)

        # Sort by new scores
        reranked.sort(key=lambda x: x.score, reverse=True)

        logger.debug("Reranked by recency", count=len(results), decay_factor=decay_factor)

        return reranked

    def boost_modified_files(
        self,
        results: List[ScoredChunk],
        boost_factor: float = 1.2,
        modified_files: Optional[List[str]] = None,
    ) -> List[ScoredChunk]:
        """
        Boost chunks from recently modified files.

        Args:
            results: Scored chunks
            boost_factor: Multiplier for modified files
            modified_files: List of recently modified file paths

        Returns:
            Re-ranked results
        """
        if not modified_files:
            return results

        modified_set = set(modified_files)
        reranked = []

        for result in results:
            chunk = result.chunk
            score = result.score

            # Boost if from modified file
            if chunk.metadata.file_path in modified_set:
                new_score = score * boost_factor
                reranked.append(ScoredChunk(chunk=chunk, score=new_score, source=result.source))
            else:
                reranked.append(result)

        # Sort by new scores
        reranked.sort(key=lambda x: x.score, reverse=True)

        logger.debug("Boosted modified files", count=len(modified_files), boost_factor=boost_factor)

        return reranked

    def prioritize_by_chunk_type(
        self, results: List[ScoredChunk], priority_types: Optional[List[ChunkType]] = None
    ) -> List[ScoredChunk]:
        """
        Re-order based on chunk type priority.

        Functions/methods usually more relevant than comments.

        Args:
            results: Scored chunks
            priority_types: Custom priority list (optional)

        Returns:
            Re-ranked results
        """
        # Group by score bands to maintain relevance
        score_bands = defaultdict(list)
        band_size = 0.1  # Group scores in 0.1 increments

        for result in results:
            band = int(result.score / band_size)
            score_bands[band].append(result)

        # Within each band, sort by chunk type priority
        reranked = []
        for band in sorted(score_bands.keys(), reverse=True):
            band_results = score_bands[band]

            # Sort by chunk type priority within band
            band_results.sort(
                key=lambda r: self.CHUNK_TYPE_PRIORITY.get(r.chunk.metadata.chunk_type, 0),
                reverse=True,
            )

            reranked.extend(band_results)

        logger.debug("Prioritized by chunk type", bands=len(score_bands), results=len(results))

        return reranked

    def rerank(
        self, results: List[ScoredChunk], strategy: str = "mixed", **kwargs
    ) -> List[ScoredChunk]:
        """
        Apply re-ranking strategy.

        Args:
            results: Initial search results
            strategy: One of 'recency', 'chunk_type', 'mixed'
            **kwargs: Additional arguments for strategies

        Returns:
            Re-ranked results
        """
        if not results:
            return results

        if strategy == "recency":
            return self.rerank_by_recency(results, decay_factor=kwargs.get("decay_factor", 0.95))

        elif strategy == "chunk_type":
            return self.prioritize_by_chunk_type(
                results, priority_types=kwargs.get("priority_types")
            )

        elif strategy == "mixed":
            # Apply multiple strategies in sequence
            # First by recency, then by chunk type
            results = self.rerank_by_recency(results, decay_factor=kwargs.get("decay_factor", 0.95))
            results = self.prioritize_by_chunk_type(
                results, priority_types=kwargs.get("priority_types")
            )

            # Apply file boost if provided
            modified_files = kwargs.get("modified_files")
            if modified_files:
                results = self.boost_modified_files(
                    results,
                    boost_factor=kwargs.get("boost_factor", 1.2),
                    modified_files=modified_files,
                )

            return results

        else:
            logger.warning("Unknown rerank strategy", strategy=strategy)
            return results

    def diversity_rerank(
        self, results: List[ScoredChunk], soft_max_per_file: int = 3
    ) -> List[ScoredChunk]:
        """
        Ensure diversity by limiting chunks per file.

        Prevents one file from dominating results. After enforcing the per-file
        limit, any remaining slots are filled with the best-scoring overflow
        chunks regardless of file, allowing files to exceed the soft limit.

        Args:
            results: Scored chunks
            soft_max_per_file: Soft limit on chunks from same file. Can be
                             exceeded if there are remaining slots after
                             initial distribution.

        Returns:
            Diversified results with same total count as input
        """
        file_counts = defaultdict(int)
        diversified = []
        overflow = []  # Chunks that exceed limit

        for result in results:
            file_path = result.chunk.metadata.file_path

            if file_counts[file_path] < soft_max_per_file:
                diversified.append(result)
                file_counts[file_path] += 1
            else:
                overflow.append(result)

        # Add overflow at end if space available
        remaining_slots = len(results) - len(diversified)
        if remaining_slots > 0:
            diversified.extend(overflow[:remaining_slots])

        logger.debug(
            "Diversity rerank", files=len(file_counts), soft_max_per_file=soft_max_per_file
        )

        return diversified
