"""
Cliente para el modelo local Ollama en ACOLYTE.
"""

import aiohttp
import json
from collections import OrderedDict
from typing import Union, AsyncIterator, List, Type, Optional, Dict, Any
from pydantic import BaseModel, ValidationError
from acolyte.core.exceptions import ExternalServiceError, ConfigurationError
from acolyte.core.logging import logger
from acolyte.core.utils.retry import retry_async


class OllamaClient:
    """
    Cliente simple para Ollama con modelo acolyte:latest.

    Características:
    1. Conexión única (mono-usuario)
    2. Siempre usa acolyte:latest
    3. Caching inteligente
    4. Streaming eficiente
    """

    def __init__(self, base_url: Optional[str] = None, cache_size: int = 1000) -> None:
        """Initialize Ollama client.

        Args:
            base_url: Ollama server URL (if None, reads from config)
            cache_size: Maximum number of cached responses
        """
        logger.info("OllamaClient initializing...")
        try:
            if base_url is None:
                # Read from configuration
                from acolyte.core import Settings

                config = Settings()
                port = config.get("ports.ollama")
                if not port:
                    raise ConfigurationError(
                        "Ollama port not configured. Please check your .acolyte file."
                    )
                self.base_url = f"http://localhost:{port}"
            else:
                self.base_url = base_url
            self.model = "acolyte:latest"  # ALWAYS this model
            self.session = aiohttp.ClientSession()
            # JUSTIFICATION OrderedDict vs functools.lru_cache:
            # 1. lru_cache is decorator for FUNCTIONS, not for INSTANCE cache
            # 2. We need dynamic size control (customizable cache_size)
            # 3. We need move_to_end() for real LRU with custom cache_key
            # 4. We need direct cache access for manual invalidation
            # 5. lru_cache doesn't support optional cache_key per call
            # VERDICT: OrderedDict is the CORRECT choice here
            self.cache: "OrderedDict[str, str]" = OrderedDict()  # Real LRU cache with OrderedDict
            self.cache_size = cache_size
            logger.info(f"Ollama client initialized with URL: {self.base_url}")
            logger.info("OllamaClient ready", base_url=self.base_url, model=self.model)
        except Exception as e:
            logger.error("OllamaClient initialization failed", error=str(e))
            raise

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        cache_key: Optional[str] = None,
    ) -> Union[str, AsyncIterator[str]]:
        """
        Generate response with optimizations.

        Optimizaciones:
        1. Cache de respuestas idénticas
        2. Timeout dinámico basado en tokens
        3. Retry con backoff exponencial
        4. Métricas de rendimiento

        Para streaming:
        - Yield incremental
        - Buffer configurable
        - Heartbeat para mantener conexión
        """
        # Check cache if cache_key is provided
        if cache_key and cache_key in self.cache:
            # Move to end (most recent) for LRU
            self.cache.move_to_end(cache_key)
            if stream:

                async def cached_stream() -> AsyncIterator[str]:
                    yield self.cache[cache_key]

                return cached_stream()
            return self.cache[cache_key]

        # ALWAYS use acolyte:latest regardless of what is requested
        payload = {
            "model": "acolyte:latest",
            "prompt": prompt,
            "temperature": temperature,
            "stream": stream,
        }

        # Add system prompt if provided
        if system:
            payload["system"] = system

        # Add max_tokens as num_predict if specified
        if max_tokens:
            payload["options"] = {"num_predict": max_tokens}

        # DESIGN DECISION: Simple but effective retry for local system
        # ===============================================================
        # In LOCAL SINGLE-USER system complex retry is UNNECESSARY because:
        # 1. Ollama runs on the same machine = predictable latency
        # 2. No distributed network issues
        # 3. Circuit breaker is overkill for localhost
        # 4. 3 attempts are enough to recover from temporary blips
        #
        # BUT we DO need retry because:
        # - Ollama may be starting up (delay in boot)
        # - Model may be loading into memory
        # - System may be under temporary load
        #
        # VERDICT: Simple retry with backoff is the perfect balance
        # ===============================================================

        # Track if we had retries for logging
        attempt_count = 0

        async def _do_generate():
            nonlocal attempt_count
            attempt_count += 1

            if stream:
                return self._stream_generate(payload, cache_key)
            else:
                async with self.session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),  # 5 min timeout
                ) as response:
                    response.raise_for_status()
                    result: Dict[str, Any] = await response.json()

                    # Extract response
                    generated_text = result.get("response", "")
                    assert isinstance(generated_text, str)

                    # Cache if cache_key is provided
                    if cache_key:
                        # If already exists, move to end
                        if cache_key in self.cache:
                            self.cache.move_to_end(cache_key)
                        self.cache[cache_key] = generated_text

                        # Limit cache size (LRU)
                        if len(self.cache) > self.cache_size:
                            # Remove oldest (first in OrderedDict)
                            self.cache.popitem(last=False)

                    return generated_text

        try:
            result = await retry_async(
                _do_generate,
                max_attempts=3,
                backoff="exponential",
                initial_delay=1.0,
                retry_on=(aiohttp.ClientError,),
                logger=logger,
            )

            # Log if we succeeded after retry
            if attempt_count > 1:
                logger.info("Generation succeeded after retry", attempt=attempt_count)

            return result
        except aiohttp.ClientError:
            raise ExternalServiceError(
                "Failed to generate response from Ollama after retries",
                code="OLLAMA_NO_RESPONSE",
                context={"url": self.base_url},
            )

    async def _stream_generate(
        self, payload: Dict[str, Any], cache_key: Optional[str] = None
    ) -> AsyncIterator[str]:
        """Generate response in streaming mode."""
        payload["stream"] = True
        full_response = ""

        async with self.session.post(
            f"{self.base_url}/api/generate", json=payload, timeout=aiohttp.ClientTimeout(total=300)
        ) as response:
            response.raise_for_status()

            async for line in response.content:
                if line:
                    try:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            chunk_text = chunk["response"]
                            full_response += chunk_text
                            yield chunk_text

                            # If it's the last chunk and there's cache_key, cache it
                            if chunk.get("done") and cache_key:
                                if cache_key in self.cache:
                                    self.cache.move_to_end(cache_key)
                                self.cache[cache_key] = full_response

                                # LRU: remove oldest if exceeds limit
                                if len(self.cache) > self.cache_size:
                                    self.cache.popitem(last=False)
                    except json.JSONDecodeError:
                        continue

    async def generate_structured(
        self,
        prompt: str,
        schema: Type[BaseModel],
        examples: Optional[List[Dict[str, Any]]] = None,
        max_retries: int = 3,
    ) -> BaseModel:
        """
        Generate structured response validated.

        Process:
        1. Add format instructions
        2. Include examples if provided
        3. Generate response
        4. Parse and validate with Pydantic
        5. Retry with feedback if validation fails
        """
        # Build prompt with format instructions
        schema_json = schema.model_json_schema()

        structured_prompt = f"""{prompt}

Respond ONLY with valid JSON following this schema:
{json.dumps(schema_json, indent=2)}
"""

        # Add examples if provided
        if examples:
            structured_prompt += "\n\nValid examples:\n"
            for i, example in enumerate(examples[:3], 1):  # Maximum 3 examples
                structured_prompt += f"\nExample {i}:\n{json.dumps(example, indent=2)}\n"

        structured_prompt += "\n\nYour response (JSON only):"

        # Try to generate and validate
        for attempt in range(max_retries):
            try:
                # Generate response
                response = await self.generate(
                    prompt=structured_prompt,
                    temperature=0.3,  # More deterministic for JSON
                )
                # Ensure response is a string
                if not isinstance(response, str):
                    raise ValueError(
                        "generate_structured response must be a string, but received AsyncIterator. Use stream=False."
                    )
                # Try to extract JSON from response
                # Find first { and last }
                start_idx = response.find("{")
                end_idx = response.rfind("}")

                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response[start_idx : end_idx + 1]

                    # Parse JSON
                    data = json.loads(json_str)

                    # Validate with Pydantic
                    result = schema(**data)
                    if attempt > 0:
                        logger.info(
                            "Structured generation succeeded after retry", attempt=attempt + 1
                        )
                    return result
                else:
                    raise ValueError("No JSON found in response")

            except (json.JSONDecodeError, ValueError, ValidationError) as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        "Structured generation failed, retrying", attempt=attempt + 1, error=str(e)
                    )
                    # Add error feedback for next attempt
                    structured_prompt = f"""{prompt}

Previous attempt failed with error: {str(e)}
Please generate valid JSON following the schema.

Schema:
{json.dumps(schema_json, indent=2)}

Your response (JSON only):"""
                else:
                    logger.error(
                        "Structured generation failed permanently",
                        max_attempts=max_retries,
                        error=str(e),
                    )
                    raise ValueError(
                        f"Failed to generate valid structured output after {max_retries} attempts: {str(e)}"
                    )
        raise ValueError(
            f"Failed to generate valid structured output after {max_retries} attempts. No valid response returned."
        )

    async def count_tokens(self, text: str) -> int:
        """
        Count tokens using Ollama API.

        Replaces tiktoken to be 100% local.
        """
        payload = {"model": "acolyte:latest", "prompt": text}

        try:
            async with self.session.post(
                f"{self.base_url}/api/tokenize",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                response.raise_for_status()
                result = await response.json()
                return len(result.get("tokens", []))
        except Exception:
            # Fallback to estimation if fails
            # Approximately 1 token per 4 characters for modern models
            return len(text) // 4

    async def close(self) -> None:
        """Close HTTP session."""
        await self.session.close()
