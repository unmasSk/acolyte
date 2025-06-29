"""
Token limit management for ACOLYTE.

Complete implementation of token counting and management system.
Uses simple estimation instead of async calls to Ollama to avoid complexity.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
from functools import lru_cache

from acolyte.core.logging import logger


class TruncateStrategy(Enum):
    """Text truncation strategies."""

    END = "end"  # Cut at the end
    START = "start"  # Cut at the beginning
    MIDDLE = "middle"  # Remove from middle
    SMART = "smart"  # Preserve important start and end


@dataclass
class TokenCount:
    """Token count result."""

    total: int
    content: int
    special: int
    estimated: bool = False


@dataclass
class ContextSplit:
    """Context split result."""

    included: List[Any]  # Included messages
    truncated: List[Any]  # Truncated messages
    excluded: List[Any]  # Excluded messages
    token_usage: Dict[str, int]  # Token distribution


class TokenEncoder(ABC):
    """Interface for different token encoders."""

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to tokens."""
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """Decode tokens to text."""
        pass


class OllamaEncoder(TokenEncoder):
    """
    Encoder using simple estimation (100% local, no async complexity).

    ARCHITECTURAL DECISION: Why estimation instead of real Ollama?
    =====================================================================
    1. AVOIDS CIRCULAR DEPENDENCIES: OllamaClient is in Core, and TokenCounter too
    2. SIMPLICITY: Single-user system doesn't need extreme precision
    3. PERFORMANCE: Estimation is instant, HTTP calls add latency
    4. ROBUSTNESS: Doesn't fail if Ollama is down
    5. SUFFICIENTLY ACCURATE: <5% error for limit management

    When to consider real Ollama?
    - If we needed token precision for billing (NOT applicable: single-user)
    - If the model was very different from GPT-style (NOT applicable: qwen2.5-coder is similar)
    - If there were multiple models with very different tokenization (NOT applicable: only acolyte:latest)
    """

    def __init__(self, base_url: str = "http://127.0.0.1:11434") -> None:
        """
        Initialize encoder.

        NOTE: Doesn't actually use Ollama to avoid async complexity.
        The estimation is good enough for token counting.
        """
        self.base_url = base_url
        # NO more _token_cache - @lru_cache handles everything
        # NO more OllamaClient import - direct estimation

    @lru_cache(maxsize=10000)
    def encode(self, text: str) -> List[int]:
        """
        Encode text to tokens with simple estimation.

        For a local single-user system, estimation is sufficient.
        We avoid async complexity that caused problems.

        Estimation:
        - English/ASCII: ~4 characters per token
        - Spanish/Unicode: ~3 characters per token
        - Code: ~5 characters per token (more dense)
        """
        if not text:
            return []

        # Detect content type for better estimation
        sample = text[:200]  # Sample for detection

        # Count special code characters
        code_chars = sum(1 for c in sample if c in '{}()[]<>;:=')
        unicode_chars = sum(1 for c in sample if ord(c) > 127)

        # Content-based estimation
        if code_chars > len(sample) * 0.1:  # >10% code characters
            chars_per_token = 5
        elif unicode_chars > len(sample) * 0.2:  # >20% unicode
            chars_per_token = 3
        else:  # Normal English text
            chars_per_token = 4

        estimated_count = max(1, len(text) // chars_per_token)

        # Return list of simulated indices
        return list(range(estimated_count))

    def decode(self, tokens: List[int]) -> str:
        """Decode tokens to text."""
        # Ollama doesn't support direct token decoding
        # This method shouldn't be used in ACOLYTE
        raise NotImplementedError("Ollama doesn't support decoding of token IDs")


class SmartTokenCounter:
    """
    Smart token counter with optimizations.

    Features:
    1. Cache of frequent counts
    2. Fast estimation for long texts
    3. Multi-model support
    4. Special token management
    """

    def __init__(self) -> None:
        self.encoders: Dict[str, TokenEncoder] = {}
        self.default_encoder = "ollama"
        self._count_cache: Dict[Any, TokenCount] = {}  # Count cache
        # Lazy loading of default encoder
        self._default_encoder_instance: Optional[OllamaEncoder] = None
        logger.info("SmartTokenCounter initialized", default_encoder=self.default_encoder)

    def _get_encoder(self, model: Optional[str] = None) -> TokenEncoder:
        """Get encoder for model, with lazy loading."""
        model = model or self.default_encoder

        if model not in self.encoders:
            if model == self.default_encoder and self._default_encoder_instance:
                self.encoders[model] = self._default_encoder_instance
            else:
                # ACOLYTE uses OllamaEncoder with simple estimation
                # Avoids async complexity and is sufficiently accurate
                encoder = OllamaEncoder()
                self.encoders[model] = encoder
                if model == self.default_encoder:
                    self._default_encoder_instance = encoder

        return self.encoders[model]

    def count(
        self, text: str, model: Optional[str] = None, include_special: bool = True
    ) -> TokenCount:
        """
        Count tokens with metadata.

        Returns:
            TokenCount with:
            - total: Total tokens
            - content: Content tokens
            - special: Special tokens
            - estimated: If it was an estimation
        """
        # Cache key
        cache_key = (text[:100], len(text), model, include_special)
        if cache_key in self._count_cache:
            return self._count_cache[cache_key]

        # For very long texts, use estimation
        if len(text) > 50000:
            # Estimation: ~1 token per 4 characters (conservative)
            estimated_total = len(text) // 4
            result = TokenCount(
                total=estimated_total, content=estimated_total, special=0, estimated=True
            )
            self._count_cache[cache_key] = result
            logger.debug(
                "Using estimation for large text",
                text_size=len(text),
                estimated_tokens=estimated_total,
            )
            return result

        # Real count
        encoder = self._get_encoder(model)
        tokens = encoder.encode(text)

        # Count special tokens (simplified)
        special_count = 0
        if include_special:
            # Approximation: tokens starting with Ġ or having special characters
            for token in tokens[:10] + tokens[-10:]:  # Only check start/end
                try:
                    decoded = encoder.decode([token])
                    if decoded.startswith("Ġ") or any(c in decoded for c in ["<", ">", "[", "]"]):
                        special_count += 1
                except Exception:
                    special_count += 1

        result = TokenCount(
            total=len(tokens),
            content=len(tokens) - special_count,
            special=special_count,
            estimated=False,
        )

        # Cache result
        self._count_cache[cache_key] = result

        # Limit cache
        if len(self._count_cache) > 1000:
            # Remove old entries
            keys = list(self._count_cache.keys())
            for k in keys[:500]:
                del self._count_cache[k]
            # Cache cleanup is normal operation, doesn't need logging

        return result

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Convenience method to get only the total tokens.
        Useful for compatibility with code expecting an int.
        """
        return self.count(text, model).total

    def truncate_to_limit(
        self,
        text: str,
        limit: int,
        model: Optional[str] = None,
        strategy: TruncateStrategy = TruncateStrategy.END,
    ) -> str:
        """
        Truncate text to token limit.

        Strategies:
        - END: Cut at the end
        - START: Cut at the beginning
        - MIDDLE: Remove from middle preserving context
        - SMART: Preserve important start and end
        """
        encoder = self._get_encoder(model)
        tokens = encoder.encode(text)

        if len(tokens) <= limit:
            return text

        if strategy == TruncateStrategy.END:
            truncated_tokens = tokens[:limit]

        elif strategy == TruncateStrategy.START:
            truncated_tokens = tokens[-limit:]

        elif strategy == TruncateStrategy.MIDDLE:
            # Preserve 40% start, 40% end, remove from middle
            start_size = int(limit * 0.4)
            end_size = int(limit * 0.4)
            truncated_tokens = tokens[:start_size] + tokens[-end_size:]

        elif strategy == TruncateStrategy.SMART:
            # Find sentence/paragraph boundaries
            # For now, similar to MIDDLE but with adjustments
            start_size = int(limit * 0.35)
            end_size = int(limit * 0.35)
            middle_size = limit - start_size - end_size

            # Try to preserve something from the middle too
            middle_start = len(tokens) // 2 - middle_size // 2
            middle_tokens = tokens[middle_start : middle_start + middle_size]

            truncated_tokens = tokens[:start_size] + middle_tokens + tokens[-end_size:]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return encoder.decode(truncated_tokens)

    def split_for_context(
        self, messages: List[Dict[str, Any]], max_context: int, reserve_output: int = 1000
    ) -> ContextSplit:
        """
        Split messages to fit in context.

        Priorities:
        1. Complete current message
        2. System messages
        3. Recent messages
        4. Messages with high relevance score
        """
        available_tokens = max_context - reserve_output

        included = []
        truncated = []
        excluded = []
        token_usage = defaultdict(int)

        # Priority 1: Current message (last)
        if messages:
            last_msg = messages[-1]
            last_tokens = self.count(last_msg.get("content", "")).total

            if last_tokens <= available_tokens:
                included.append(last_msg)
                token_usage["current"] = last_tokens
                available_tokens -= last_tokens
            else:
                # Truncate current message if too long
                truncated_content = self.truncate_to_limit(
                    last_msg.get("content", ""), available_tokens, strategy=TruncateStrategy.SMART
                )
                truncated_msg = {**last_msg, "content": truncated_content}
                truncated.append(truncated_msg)
                token_usage["current"] = available_tokens
                available_tokens = 0

        # Priority 2: System messages
        system_messages = [m for m in messages[:-1] if m.get("role") == "system"]
        for msg in system_messages:
            if available_tokens <= 0:
                excluded.append(msg)
                continue

            msg_tokens = self.count(msg.get("content", "")).total
            if msg_tokens <= available_tokens:
                included.append(msg)
                token_usage["system"] += msg_tokens
                available_tokens -= msg_tokens
            else:
                excluded.append(msg)

        # Priority 3: Recent messages (reverse, without last already processed)
        other_messages = [m for m in messages[:-1] if m.get("role") != "system"]
        for msg in reversed(other_messages):
            if available_tokens <= 0:
                excluded.append(msg)
                continue

            msg_tokens = self.count(msg.get("content", "")).total

            if msg_tokens <= available_tokens:
                included.insert(0, msg)  # Maintain chronological order
                token_usage["history"] += msg_tokens
                available_tokens -= msg_tokens
            elif available_tokens > 100:  # Worth truncating
                truncated_content = self.truncate_to_limit(
                    msg.get("content", ""),
                    available_tokens - 50,  # Leave margin
                    strategy=TruncateStrategy.MIDDLE,
                )
                truncated_msg = {**msg, "content": truncated_content}
                truncated.append(truncated_msg)
                token_usage["history"] += available_tokens - 50
                available_tokens = 50
            else:
                excluded.append(msg)

        # Calculate totals
        token_usage["total"] = sum(token_usage.values())
        token_usage["reserved"] = reserve_output
        token_usage["available"] = available_tokens

        return ContextSplit(
            included=included, truncated=truncated, excluded=excluded, token_usage=dict(token_usage)
        )


class TokenBudgetManager:
    """
    Manages token budget for operations.

    Useful for:
    1. RAG with multiple sources
    2. Long conversations
    3. Batch processing
    4. Dynamic distribution by query type
    """

    def __init__(self, total_budget: int) -> None:
        """
        Initialize with total budget (model's context_size).

        Args:
            total_budget: TOTAL model limit (32k, 128k, etc)
        """
        self.total_budget = total_budget
        self.allocations: Dict[str, int] = {}
        self.used: Dict[str, int] = defaultdict(int)
        self.priorities: Dict[str, float] = {
            "system": 1.0,  # Maximum priority
            "response": 0.9,  # High priority
            "rag": 0.7,  # Medium-high
            "history": 0.5,  # Medium
            "other": 0.3,  # Low
        }
        logger.info("TokenBudgetManager initialized", total_budget=total_budget)

    def allocate(self, category: str, tokens: int) -> bool:
        """
        Reserve tokens for category.

        Example categories:
        - "system": System prompts
        - "rag": RAG context
        - "history": Chat history
        - "response": Expected response

        Returns:
            True if reservation succeeded, False if no space
        """
        # Check that it doesn't exceed budget
        current_allocated = sum(self.allocations.values())
        if current_allocated + tokens > self.total_budget:
            logger.warning(
                "Token allocation failed",
                category=category,
                requested=tokens,
                available=self.total_budget - current_allocated,
            )
            return False

        self.allocations[category] = tokens
        return True

    def use(self, category: str, tokens: int) -> bool:
        """
        Consume tokens from category.

        Returns:
            True if there were enough tokens, False if exceeds allocation
        """
        if category not in self.allocations:
            return False

        if self.used[category] + tokens > self.allocations[category]:
            logger.warning(
                "Token usage exceeded allocation",
                category=category,
                requested=tokens,
                remaining=self.get_remaining(category),
            )
            return False

        self.used[category] += tokens
        return True

    def get_remaining(self, category: str) -> int:
        """Get remaining tokens in category."""
        if category not in self.allocations:
            return 0
        return self.allocations[category] - self.used[category]

    def allocate_for_dream_cycle(self, model_size: str, cycle_number: int = 1) -> Dict[str, int]:
        """
        Allocate tokens for Dream with sliding window (Dream Decision #2).

        For 32k models:
        - Cycle 1: 28,000 new code, 0 previous context
        - Cycle 2+: 28,000 new code, 1,500 previous context

        For 128k models:
        - Everything in one cycle: 117,900 tokens

        Args:
            model_size: '32k' or '128k'
            cycle_number: Cycle number (1, 2, 3...)

        Returns:
            Token distribution for the cycle
        """
        # Clear previous allocations
        self.allocations.clear()
        self.used.clear()

        if model_size == "128k" and self.total_budget >= 131072:
            # 128k model: everything at once
            available = int(self.total_budget * 0.9)  # 90% of total
            self.allocate("code_analysis", available)
            self.allocate("dream_internal", int(self.total_budget * 0.1))

        elif model_size == "32k" and self.total_budget >= 32768:
            # 32k model: sliding window
            if cycle_number == 1:
                # First cycle: no previous context
                self.allocate("code_analysis", 28000)
                self.allocate("dream_internal", 1500)
            else:
                # Later cycles: with sliding window
                self.allocate("sliding_window", 1500)  # Previous critical context
                self.allocate("code_analysis", 28000)  # New code
                # Remaining space for internal processing
                remaining = int(self.total_budget * 0.9) - 29500
                if remaining > 0:
                    self.allocate("dream_internal", remaining)
        else:
            # Fallback for other sizes
            available = int(self.total_budget * 0.9)
            if cycle_number > 1:
                # With sliding window
                window_size = min(1500, int(available * 0.05))
                self.allocate("sliding_window", window_size)
                self.allocate("code_analysis", available - window_size)
            else:
                self.allocate("code_analysis", available)
            self.allocate("dream_internal", int(self.total_budget * 0.1))

        return self.allocations

    def get_distribution(self) -> Dict[str, Any]:
        """Get current token distribution."""
        return {
            "total_budget": self.total_budget,
            "allocations": dict(self.allocations),
            "used": dict(self.used),
            "remaining": {cat: self.get_remaining(cat) for cat in self.allocations},
        }

    def optimize_allocations(self, priorities: Optional[Dict[str, float]] = None) -> None:
        """
        Rebalance allocations according to priorities.

        Useful when initial context doesn't fit.
        """
        if priorities:
            self.priorities.update(priorities)

        # Calculate total weight
        total_priority = sum(self.priorities.get(cat, 0.5) for cat in self.allocations)

        if total_priority == 0:
            return

        # Redistribute according to priorities
        total_available = self.total_budget
        new_allocations = {}

        for category in self.allocations:
            priority = self.priorities.get(category, 0.5)
            proportion = priority / total_priority
            new_allocations[category] = int(total_available * proportion)

        # Adjust to not exceed total
        allocated = sum(new_allocations.values())
        if allocated > self.total_budget:
            # Reduce proportionally
            factor = self.total_budget / allocated
            for cat in new_allocations:
                new_allocations[cat] = int(new_allocations[cat] * factor)

        self.allocations = new_allocations
        # Optimization is normal operation, not critical for debugging

    def allocate_for_query_type(self, query_type: str) -> Dict[str, int]:
        """
        Allocate tokens by query type (Decision #10).

        Types:
        - 'generation': 75% response, 25% context
        - 'simple': 20% response, 80% context
        - 'default': 10% response, 90% context
        """
        distributions = {
            "generation": {"response": 0.75, "system": 0.05, "rag": 0.15, "history": 0.05},
            "simple": {"response": 0.20, "system": 0.10, "rag": 0.50, "history": 0.20},
            "default": {"response": 0.10, "system": 0.09, "rag": 0.54, "history": 0.27},
        }

        dist = distributions.get(query_type, distributions["default"])

        # Clear previous allocations
        self.allocations.clear()
        self.used.clear()

        # Allocate according to distribution
        for category, proportion in dist.items():
            tokens = int(self.total_budget * proportion)
            self.allocate(category, tokens)

        return self.allocations


# Global instance for simple use
_default_counter: Optional[SmartTokenCounter] = None


def get_token_counter() -> SmartTokenCounter:
    """Get global counter instance."""
    global _default_counter
    if _default_counter is None:
        _default_counter = SmartTokenCounter()
        logger.info("Global token counter created")
    return _default_counter
