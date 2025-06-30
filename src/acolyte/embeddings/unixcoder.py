"""
Embeddings implementation using UniXcoder for multi-language code.

UniXcoder is a pre-trained model specifically designed to understand code
in multiple programming languages.
"""

from typing import List, Optional, Tuple, Union
import time
import threading

import torch
from transformers import AutoModel, AutoTokenizer

from acolyte.core.logging import logger
from acolyte.core.secure_config import Settings
from acolyte.core.exceptions import ConfigurationError, AcolyteError, ExternalServiceError
from acolyte.core.database import DatabaseManager
from acolyte.embeddings.types import EmbeddingVector, EmbeddingsMetricsSummaryDict, MetricsProvider
from acolyte.embeddings.persistent_cache import SmartPersistentCache
from acolyte.embeddings.context import RichCodeContext
from acolyte.models.chunk import Chunk


class UniXcoderEmbeddings:
    """Embeddings generator using microsoft/unixcoder-base.

    UniXcoder is a model specifically trained to understand
    code in multiple programming languages.
    """

    def __init__(self, metrics: Optional[MetricsProvider] = None):
        """Initializes the model and auxiliary components.

        Args:
            metrics: Optional metrics provider. If not provided,
                    uses the singleton of the module if enabled.
        """
        self.config = Settings()

        # Validar configuración antes de continuar
        self._validate_config()

        # Métricas: usar parámetro o singleton del módulo
        if metrics is not None:
            self.metrics = metrics
        else:
            from acolyte.embeddings import get_embeddings_metrics

            self.metrics = get_embeddings_metrics()
        # Leer TTL de la configuración, default 3600 segundos (1 hora)
        cache_ttl = self.config.get("cache.ttl_seconds", 3600)
        self.cache = SmartPersistentCache(
            max_size=self.config.get("embeddings.cache_size", 10000),
            ttl_seconds=cache_ttl,
            save_interval=self.config.get("cache.save_interval", 300),  # 5 minutes by default
        )

        # Lazy loading del modelo
        self._model = None
        self._tokenizer = None
        self._device = None
        self._device_lock = threading.Lock()

        # Configuración
        self.model_name = "microsoft/unixcoder-base"
        self.max_length = 512
        self.batch_size = self.config.get("embeddings.batch_size", 20)

        logger.info("UniXcoderEmbeddings initialized (lazy loading enabled)")

        # Reranker instance (lazy loading)
        self._reranker = None

        # Database manager for runtime state
        self._db = None

        # Cargar device state persistido si existe
        self._load_device_state()

    def _validate_config(self):
        """Validates the embeddings module configuration.

        Following the ConfigValidator pattern in secure_config.py, validates all
        embeddings-specific parameters to catch errors early.

        Raises:
            ConfigurationError: If any parameter is invalid
        """
        # Validate device
        device = self.config.get("embeddings.device", "auto")
        if device not in ["auto", "cuda", "cpu"]:
            logger.error("Invalid embeddings device", device=device)
            raise ConfigurationError(
                f"Invalid embeddings.device: {device}. " f"Must be one of: auto, cuda, cpu"
            )

        # Validate cache_size
        cache_size = self.config.get("embeddings.cache_size", 10000)
        if not isinstance(cache_size, int) or cache_size < 100:
            logger.error("Invalid cache size", cache_size=cache_size)
            raise ConfigurationError(
                f"Invalid embeddings.cache_size: {cache_size}. " f"Must be an integer >= 100"
            )

        # Validate batch_size
        batch_size = self.config.get("embeddings.batch_size", 20)
        if not isinstance(batch_size, int) or batch_size < 1 or batch_size > 100:
            logger.error("Invalid batch size", batch_size=batch_size)
            raise ConfigurationError(
                f"Invalid embeddings.batch_size: {batch_size}. "
                f"Must be an integer between 1 and 100"
            )

        # NOTE: Clustering was removed in favor of the neural graph in /rag/graph/
        # We do not validate clustering parameters as they are not used

        # Validate cache.ttl_seconds (used by ContextAwareCache)
        cache_ttl = self.config.get("cache.ttl_seconds", 3600)
        if not isinstance(cache_ttl, int) or cache_ttl < 60:
            logger.error("Invalid cache TTL", ttl=cache_ttl)
            raise ConfigurationError(
                f"Invalid cache.ttl_seconds: {cache_ttl}. " f"Must be an integer >= 60 seconds"
            )

        # Validate enable_metrics
        enable_metrics = self.config.get("embeddings.enable_metrics", True)
        if not isinstance(enable_metrics, bool):
            logger.error("Invalid enable_metrics", enable_metrics=enable_metrics)
            raise ConfigurationError(
                f"Invalid embeddings.enable_metrics: {enable_metrics}. "
                f"Must be a boolean (true/false)"
            )

        # Validate reranker_batch_size
        reranker_batch_size = self.config.get("embeddings.reranker_batch_size", 32)
        if (
            not isinstance(reranker_batch_size, int)
            or reranker_batch_size < 1
            or reranker_batch_size > 100
        ):
            logger.error("Invalid reranker batch size", batch_size=reranker_batch_size)
            raise ConfigurationError(
                f"Invalid embeddings.reranker_batch_size: {reranker_batch_size}. "
                f"Must be an integer between 1 and 100"
            )

        # Validate max_tokens_per_batch
        max_tokens_per_batch = self.config.get("embeddings.max_tokens_per_batch", 10000)
        if not isinstance(max_tokens_per_batch, int) or max_tokens_per_batch < 512:
            logger.error("Invalid max_tokens_per_batch", max_tokens=max_tokens_per_batch)
            raise ConfigurationError(
                f"Invalid embeddings.max_tokens_per_batch: {max_tokens_per_batch}. "
                f"Must be an integer >= 512"
            )

        logger.debug(
            "Embeddings configuration validated",
            device=device,
            cache_size=cache_size,
            batch_size=batch_size,
            max_tokens_per_batch=max_tokens_per_batch,
        )

    async def _get_db(self) -> DatabaseManager:
        """Obtiene instancia de base de datos con lazy loading."""
        if self._db is None:
            self._db = DatabaseManager()  # type: ignore[attr-defined]
            await self._db.initialize()  # type: ignore[attr-defined]
        return self._db

    def _load_device_state(self):
        """Loads the persisted device state from SQLite.

        Runs synchronously at startup for simplicity.
        Only one read at service startup.
        """
        import asyncio

        async def load():
            try:
                db = await self._get_db()
                result = await db.fetch_one(  # type: ignore[attr-defined]
                    "SELECT value FROM runtime_state WHERE key = ?", ("embeddings.device",)
                )

                if result and result["value"]:
                    device_str = result["value"]
                    # Validate that it is a valid device
                    if device_str in ["cuda", "cpu"]:
                        self._device = torch.device(device_str)
                        logger.info("Device state loaded from runtime_state", device=device_str)
                    else:
                        logger.warning("Invalid device state in runtime_state", device=device_str)

            except Exception as e:
                # Not critical if it fails
                logger.debug("Could not load device state", error=str(e))

        # Run synchronously
        try:
            asyncio.run(load())
        except RuntimeError:
            # If no event loop, ignore
            pass

    async def _save_device_state(self, device_str: str):
        """Saves the device state in SQLite.

        Only called when the device changes (fallback).
        """
        try:
            db = await self._get_db()
            await db.execute(  # type: ignore[attr-defined]
                "INSERT OR REPLACE INTO runtime_state (key, value) VALUES (?, ?)",
                ("embeddings.device", device_str),
            )
            logger.info("Device state saved in runtime_state", device=device_str)
        except Exception as e:
            # Not critical if it fails
            logger.error("Error saving device state", error=str(e))

    @property
    def device(self) -> torch.device:
        """Detects and returns the device (CUDA/CPU) to use with thread safety."""
        if self._device is None:
            with self._device_lock:
                # Double-check locking pattern
                if self._device is None:
                    device_config = self.config.get("embeddings.device", "auto")

                    if device_config == "auto":
                        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    else:
                        self._device = torch.device(device_config)

                    logger.info("Using device", device=self._device)

        return self._device

    def _load_model(self):
        """Loads the model and tokenizer when needed.

        Handles specific errors:
        - Checks CUDA availability
        - Automatic fallback to CPU if CUDA fails
        - Model download/load errors
        - Insufficient memory errors
        """
        if self._model is None:
            logger.info("Loading model", model_name=self.model_name)
            start_time = time.time()

            # Check CUDA availability before attempting to load
            if self.device.type == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, changing to CPU")
                self._device = torch.device("cpu")
                # Persist the change for future restarts
                import asyncio

                try:
                    asyncio.create_task(self._save_device_state("cpu"))
                except RuntimeError:
                    # If no active event loop, try to create one
                    try:
                        loop = asyncio.new_event_loop()
                        loop.run_until_complete(self._save_device_state("cpu"))
                        loop.close()
                    except Exception:
                        logger.debug("Could not persist device state")

            try:
                # Try loading tokenizer first (lighter)
                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                except Exception as e:
                    error = ExternalServiceError(
                        f"Could not download tokenizer {self.model_name}",
                        context={
                            "model": self.model_name,
                            "error_type": type(e).__name__,
                            "error_details": str(e),
                        },
                    )
                    error.add_suggestion("Check internet connection")
                    error.add_suggestion("Check that the model name is correct")
                    error.add_suggestion(
                        f"Try manually: transformers-cli download {self.model_name}"
                    )
                    logger.error("Error downloading tokenizer", error=error.to_dict())
                    raise error

                # Try loading model
                try:
                    self._model = AutoModel.from_pretrained(self.model_name)
                except AcolyteError:
                    raise
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                        # Memory error
                        error = ExternalServiceError(
                            "Insufficient memory to load model",
                            context={
                                "model": self.model_name,
                                "device": str(self.device),
                                "error": str(e),
                            },
                        )
                        error.add_suggestion("Reduce batch_size in configuration")
                        error.add_suggestion("Use a smaller model")
                        if self.device.type == "cuda":
                            error.add_suggestion(
                                "Change to CPU in configuration: embeddings.device = 'cpu'"
                            )
                        logger.error("Memory error", error=error.to_dict())
                        raise error
                    else:
                        raise  # Re-throw other RuntimeErrors
                except Exception as e:
                    # Other download/load errors
                    error = ExternalServiceError(
                        f"Could not load model {self.model_name}",
                        context={
                            "model": self.model_name,
                            "error_type": type(e).__name__,
                            "error_details": str(e),
                        },
                    )
                    error.add_suggestion("Check available disk space")
                    error.add_suggestion("Clean transformers cache: ~/.cache/huggingface/")
                    logger.error("Error loading model", error=error.to_dict())
                    raise error

                # Try moving to device
                try:
                    self._model.to(self.device)
                except RuntimeError as e:
                    if "cuda" in str(e).lower() and self.device.type == "cuda":
                        # CUDA error, try fallback to CPU
                        logger.warning("Error moving model to CUDA, trying CPU", error=str(e))
                        self._device = torch.device("cpu")
                        try:
                            self._model.to(self.device)
                            logger.info("Model loaded successfully on CPU as fallback")
                            # Persist the fallback for future restarts
                            import asyncio

                            try:
                                asyncio.create_task(self._save_device_state("cpu"))
                            except RuntimeError:
                                # If no active event loop, try to create one
                                try:
                                    loop = asyncio.new_event_loop()
                                    loop.run_until_complete(self._save_device_state("cpu"))
                                    loop.close()
                                except Exception:
                                    logger.debug("Could not persist device state")
                        except Exception as cpu_error:
                            # Even CPU doesn't work
                            error = ExternalServiceError(
                                "Could not load model on any device",
                                context={
                                    "cuda_error": str(e),
                                    "cpu_error": str(cpu_error),
                                    "model": self.model_name,
                                },
                            )
                            error.add_suggestion("Check PyTorch installation")
                            error.add_suggestion("Reinstall with: pip install torch torchvision")
                            logger.error("Critical device error", error=error.to_dict())
                            raise error
                    else:
                        # Other type of error moving to device
                        error = ExternalServiceError(
                            f"Error moving model to {self.device}",
                            context={"device": str(self.device), "error": str(e)},
                        )
                        logger.error("Device error", error=error.to_dict())
                        raise error

                # Configure evaluation mode
                self._model.eval()

                # Register success
                load_time = (time.time() - start_time) * 1000
                logger.info("Model loaded successfully", device=self.device, load_time_ms=load_time)
                if self.metrics:
                    self.metrics.record_operation("model_load", load_time)

                # Log available memory if is CUDA
                if self.device.type == "cuda":
                    memory_info = torch.cuda.get_device_properties(0)
                    memory_used = torch.cuda.memory_allocated(0) / 1024**3  # GB
                    memory_total = memory_info.total_memory / 1024**3  # GB
                    logger.info(
                        f"GPU {memory_info.name}: {memory_used:.1f}GB / {memory_total:.1f}GB used"
                    )

            except AcolyteError:
                raise
            except Exception as e:
                error = ExternalServiceError(
                    "Unexpected error loading model",
                    context={
                        "model": self.model_name,
                        "device": str(self.device),
                        "error_type": type(e).__name__,
                        "error_details": str(e),
                    },
                )
                error.add_suggestion("Check logs for more details")
                error.add_suggestion("Report this error if persists")
                logger.error("Unanticipated error", error=error.to_dict())
                raise error

    def _prepare_input(self, text: Union[str, Chunk], context: Optional[RichCodeContext]) -> str:
        """Prepares the input using the standard to_search_text() interface.

        UNIFICATION: Delegates to chunk.to_search_text() to avoid duplication.
        Adds the chunk type as a prefix for better relevance.
        """
        # If it's a Chunk, use its standard interface with type as prefix
        if isinstance(text, Chunk):
            base_text = text.to_search_text(context)
            # Add type of chunk as prefix for better embeddings
            if hasattr(text, "chunk_type") and getattr(text, "chunk_type", None):
                type_prefix = f"<{getattr(text.chunk_type, 'value', str(text.chunk_type))}>"  # type: ignore[attr-defined]
                return f"{type_prefix} {base_text}"
            return base_text
        # If the object has to_search_text but is not a Chunk
        elif hasattr(text, "to_search_text") and callable(getattr(text, "to_search_text")):
            return text.to_search_text(context)  # type: ignore[attr-defined]
        # Fallback for plain text (when not a Chunk)
        return self._prepare_plain_text(text, context)  # type: ignore[arg-type]

    def _prepare_plain_text(self, text: str, context: Optional[RichCodeContext]) -> str:
        """Prepares plain text (no-Chunk) with rich context."""
        if context is None:
            return text
        # Build enriched input for plain text
        parts = [f"<{context.language}>"]
        # Add relevant metadata
        if context.imports:
            parts.append(f"imports: {', '.join(context.imports[:5])}")
        if context.semantic_tags:
            parts.append(f"tags: {', '.join(context.semantic_tags[:3])}")
        if context.dependencies:
            parts.append(f"uses: {', '.join(context.dependencies[:3])}")
        # The code itself
        parts.append(text)
        return " ".join(parts)

    def encode(
        self, text: Union[str, Chunk], context: Optional[RichCodeContext] = None
    ) -> EmbeddingVector:
        """Generates standard embedding for a text or Chunk with optional context.

        Args:
            text: Code, text or Chunk to vectorize
            context: Additional context for better relevance

        Returns:
            Standard 768-dimensional EmbeddingVector
        """
        start_time = time.time()
        # Check cache
        cached = self.cache.get(text, context)
        if cached is not None:
            if self.metrics:
                self.metrics.record_cache_hit()
            return cached
        if self.metrics:
            self.metrics.record_cache_miss()
        # Load model if needed
        self._load_model()
        # Explicit check to avoid None errors
        if self._tokenizer is None or self._model is None:
            raise RuntimeError("The model or the tokenizer are not initialized")
        # Prepare input with context
        prepared_text = self._prepare_input(text, context)
        # Tokenize
        inputs = self._tokenizer(
            prepared_text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        # Generate embedding
        with torch.no_grad():
            outputs = self._model(**inputs)
            # UniXcoder uses CLS token pooling
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        # Create Standard EmbeddingVector
        result = EmbeddingVector(embedding)
        # Save to cache
        self.cache.set(text, context, result)
        # Metrics
        latency = (time.time() - start_time) * 1000
        if self.metrics:
            self.metrics.record_operation("encode", latency)
        return result

    def encode_batch(
        self,
        texts: List[Union[str, Chunk]],
        contexts: Optional[List[Optional[RichCodeContext]]] = None,
        batch_size: Optional[int] = None,
        max_tokens_per_batch: Optional[int] = None,
    ) -> List[EmbeddingVector]:
        """Generates embeddings for multiple texts efficiently.

        Args:
            texts: List of texts or Chunks to vectorize
            contexts: Optional list of contexts (same order as texts)
            batch_size: Batch size (uses config if not specified)
            max_tokens_per_batch: Total token limit per batch (new)

        Returns:
            List of EmbeddingVector in the same order as the inputs
        """
        if batch_size is None:
            batch_size = self.batch_size
        if max_tokens_per_batch is None:
            max_tokens_per_batch = self.config.get("embeddings.max_tokens_per_batch", 10000)
        if contexts is None:
            contexts = [None for _ in range(len(texts))]
        if len(texts) != len(contexts):
            raise ValueError("texts and contexts must have the same size")
        # Guarantee that batch_size and max_tokens_per_batch are not None
        assert batch_size is not None, "batch_size cannot be None"
        assert max_tokens_per_batch is not None, "max_tokens_per_batch cannot be None"
        # If only batch_size is specified and max_tokens is the default, use previous behavior
        if batch_size != self.batch_size and max_tokens_per_batch == 10000:
            return self._encode_batch_legacy(texts, contexts, batch_size)
        # New behavior with token limit
        return self._encode_batch_token_limited(texts, contexts, max_tokens_per_batch)

    def _encode_batch_legacy(
        self,
        texts: List[Union[str, Chunk]],
        contexts: List[Optional[RichCodeContext]],
        batch_size: int,
    ) -> List[EmbeddingVector]:
        """Previous behavior with fixed batch size for compatibility."""
        # Initialize result array with placeholders
        results: List[Optional[EmbeddingVector]] = [None] * len(texts)
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_contexts = contexts[i : i + batch_size]
            # Check cache first
            batch_to_encode = []
            batch_indices = []
            for j, (text, ctx) in enumerate(zip(batch_texts, batch_contexts)):
                global_idx = i + j
                cached = self.cache.get(text, ctx)
                if cached is not None:
                    results[global_idx] = cached
                    if self.metrics:
                        self.metrics.record_cache_hit()
                else:
                    batch_to_encode.append(self._prepare_input(text, ctx))
                    batch_indices.append(global_idx)
                    if self.metrics:
                        self.metrics.record_cache_miss()
            # Encode the ones not in cache
            if batch_to_encode:
                self._load_model()
                if self._tokenizer is None or self._model is None:
                    raise RuntimeError("The model or the tokenizer are not initialized")
                start_time = time.time()
                # Tokenize batch
                inputs = self._tokenizer(
                    batch_to_encode,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                # Generate embeddings
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                # Assign results directly to correct positions O(1)
                for emb, idx in zip(embeddings, batch_indices):
                    vec = EmbeddingVector(emb)
                    results[idx] = vec
                    # Update cache with original text/chunk
                    self.cache.set(texts[idx], contexts[idx], vec)
                # Metrics
                latency = (time.time() - start_time) * 1000
                if self.metrics:
                    self.metrics.record_operation("encode_batch", latency / len(batch_to_encode))
        # Check that no None before returning
        return [r for r in results if r is not None]

    def _estimate_tokens(self, text: Union[str, Chunk]) -> int:
        """Estimates the number of tokens without tokenizing completely.

        Uses a quick approximation to avoid tokenizing the entire text.
        """
        # If it's a Chunk, get the content
        if isinstance(text, Chunk):
            text_content = text.content
        else:
            text_content = text  # type: ignore[assignment]
        # Quick estimation: ~1.3 tokens per word in code
        # We use only the first 1000 characters to estimate
        sample = text_content[:1000]  # type: ignore[index]
        word_count = len(sample.split())
        # Scale if the text is longer than the sample
        if isinstance(text_content, str) and len(text_content) > 1000:
            word_count = word_count * (len(text_content) / 1000)
        return int(word_count * 1.3)

    def _encode_batch_token_limited(
        self,
        texts: List[Union[str, Chunk]],
        contexts: List[Optional[RichCodeContext]],
        max_tokens_per_batch: int,
    ) -> List[EmbeddingVector]:
        """Processes batches limiting total tokens instead of item count.

        This prevents OOM errors with very long texts.
        """
        # Initialize result
        results: List[Optional[EmbeddingVector]] = [None] * len(texts)
        # Process with token limit
        current_batch_texts: List[Union[str, Chunk]] = []
        current_batch_contexts: List[Optional[RichCodeContext]] = []
        current_batch_indices: List[int] = []
        current_tokens = 0
        for i, (text, ctx) in enumerate(zip(texts, contexts)):
            # Check cache first
            cached = self.cache.get(text, ctx)
            if cached is not None:
                results[i] = cached
                if self.metrics:
                    self.metrics.record_cache_hit()
                continue
            # Estimate tokens for this text
            estimated_tokens = self._estimate_tokens(text)
            # If this text alone exceeds the limit, process it only
            if estimated_tokens > max_tokens_per_batch:
                logger.warning(
                    f"Individual text exceeds token limit "
                    f"({estimated_tokens} > {max_tokens_per_batch}), "
                    f"processing it only"
                )
                # Process current batch if it has elements
                if current_batch_texts:
                    self._process_batch_tokens(
                        current_batch_texts,
                        current_batch_contexts,
                        current_batch_indices,
                        texts,
                        contexts,
                        results,
                    )
                    current_batch_texts = []
                    current_batch_contexts = []
                    current_batch_indices = []
                    current_tokens = 0
                # Process this text only
                self._process_batch_tokens([text], [ctx], [i], texts, contexts, results)
                continue
            # If adding this text exceeds the limit, process current batch
            if current_tokens + estimated_tokens > max_tokens_per_batch and current_batch_texts:
                self._process_batch_tokens(
                    current_batch_texts,
                    current_batch_contexts,
                    current_batch_indices,
                    texts,
                    contexts,
                    results,
                )
                # Restart for next batch
                current_batch_texts = [text]
                current_batch_contexts = [ctx]
                current_batch_indices = [i]
                current_tokens = estimated_tokens
            else:
                # Add to current batch
                current_batch_texts.append(text)
                current_batch_contexts.append(ctx)
                current_batch_indices.append(i)
                current_tokens += estimated_tokens
        # Process last batch if something remains
        if current_batch_texts:
            self._process_batch_tokens(
                current_batch_texts,
                current_batch_contexts,
                current_batch_indices,
                texts,
                contexts,
                results,
            )
        # Check that no None before returning
        return [r for r in results if r is not None]

    def _process_batch_tokens(
        self,
        batch_texts: List[Union[str, Chunk]],
        batch_contexts: List[Optional[RichCodeContext]],
        batch_indices: List[int],
        all_texts: List[Union[str, Chunk]],
        all_contexts: List[Optional[RichCodeContext]],
        results: List[Optional[EmbeddingVector]],
    ):
        """Processes a batch of texts and updates results."""
        if not batch_texts:
            return
        self._load_model()
        if self._tokenizer is None or self._model is None:
            raise RuntimeError("The model or the tokenizer are not initialized")
        start_time = time.time()
        # Prepare texts
        prepared_texts = [
            self._prepare_input(text, ctx) for text, ctx in zip(batch_texts, batch_contexts)
        ]
        # Tokenize
        inputs = self._tokenizer(
            prepared_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        # Generate embeddings
        with torch.no_grad():
            outputs = self._model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        # Assign results and update cache
        for i, (emb, idx) in enumerate(zip(embeddings, batch_indices)):
            vec = EmbeddingVector(emb)
            results[idx] = vec
            self.cache.set(all_texts[idx], all_contexts[idx], vec)
        # Metrics
        latency = (time.time() - start_time) * 1000
        if self.metrics:
            self.metrics.record_operation("encode_batch", latency / len(batch_texts))
            self.metrics.record_cache_miss()  # Miss already registered above

    def _get_reranker(self):
        """Gets instance of CrossEncoderReranker with lazy loading.

        Returns:
            CrossEncoderReranker: Instance of the re-ranker
        """
        if self._reranker is None:
            from acolyte.embeddings.reranker import CrossEncoderReranker

            logger.debug("Initializing CrossEncoderReranker...")
            self._reranker = CrossEncoderReranker()
        return self._reranker

    def encode_with_rerank(
        self, query: str, candidates: List[str], top_k: int = 10, initial_retrieval_factor: int = 3
    ) -> List[Tuple[str, float]]:
        """Generates embeddings and re-ranks results for maximum relevance.

        Implements the two-stage flow mentioned in the README:
        1. Wide initial search with embeddings (fast)
        2. Precise re-ranking with CrossEncoder (slow but precise)

        Args:
            query: Search query
            candidates: List of candidates to rank
            top_k: Number of best results to return
            initial_retrieval_factor: Multiplier for initial retrieval
                                    (e.g., 3 = retrieve 3*top_k to re-rank)

        Returns:
            List of tuples (candidate, score) ordered by relevance

        Example:
            >>> embeddings = UniXcoderEmbeddings()
            >>> candidates = [code1, code2, code3, ...]
            >>> results = embeddings.encode_with_rerank("find user auth", candidates, top_k=5)
            >>> # Returns the 5 best results after precise re-ranking
        """
        if not candidates:
            return []

        start_time = time.time()

        # Stage 1: Wide initial search with embeddings
        # Retrieve more candidates than requested to re-rank
        initial_top_k = min(len(candidates), top_k * initial_retrieval_factor)

        # Generate embedding of query
        query_embedding = self.encode(query)

        # Calculate similarities with all candidates
        similarities = []
        for i, candidate in enumerate(candidates):
            candidate_embedding = self.encode(candidate)
            similarity = query_embedding.cosine_similarity(candidate_embedding)
            similarities.append((candidate, similarity, i))  # Save original index

        # Sort by similarity and take the best initial_top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        initial_candidates = [(cand, score) for cand, score, _ in similarities[:initial_top_k]]

        # If we have few candidates, it's not worth re-ranking
        if len(initial_candidates) <= top_k:
            logger.debug(
                f"Few candidates ({len(initial_candidates)}), " f"returning without re-ranking"
            )
            return initial_candidates

        # Stage 2: Re-ranking with CrossEncoder
        reranker = self._get_reranker()

        # Extract only the candidates (without scores)
        candidates_to_rerank = [cand for cand, _ in initial_candidates]

        # Re-rank
        reranked_results = reranker.rerank(
            query=query, candidates=candidates_to_rerank, top_k=top_k
        )

        # Metrics
        total_time = (time.time() - start_time) * 1000
        if self.metrics:
            self.metrics.record_operation("encode_with_rerank", total_time)

        logger.debug(
            f"Complete re-ranking: {len(candidates)} → {initial_top_k} → {len(reranked_results)} "
            f"in {total_time:.1f}ms"
        )

        return reranked_results

    def get_metrics_summary(self) -> EmbeddingsMetricsSummaryDict:
        """Returns module metrics summary.

        Returns:
            EmbeddingsMetricsSummaryDict with module state information
        """
        base_info = {
            "cache_size": self.cache.size,
            "cache_ttl_seconds": self.cache.ttl_seconds,
            "model_loaded": self._model is not None,
            "device": str(self.device) if self._device else "not_set",
            "metrics_enabled": self.metrics is not None,
        }
        # Add persistence info if the cache supports it
        if hasattr(self.cache, 'get_persistent_stats'):
            base_info['cache_persistence'] = self.cache.get_persistent_stats()
        if self.metrics:
            base_info.update(
                {
                    "p95_latency_ms": self.metrics.get_p95_latency(),
                    "cache_hit_rate": self.metrics.get_cache_hit_rate(),
                }
            )
        # Explicit cast to comply with typing
        return base_info  # type: ignore[return-value]
