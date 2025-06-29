from typing import Union, AsyncIterator, List, Type, Optional, Dict, Any, TypeVar
from collections import OrderedDict
from pydantic import BaseModel
import aiohttp

T = TypeVar('T', bound=BaseModel)

class OllamaClient:
    base_url: str
    model: str
    session: aiohttp.ClientSession
    cache: OrderedDict[str, str]
    cache_size: int

    def __init__(self, base_url: str = ..., cache_size: int = ...) -> None: ...
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = ...,
        temperature: float = ...,
        max_tokens: Optional[int] = ...,
        stream: bool = ...,
        cache_key: Optional[str] = ...,
    ) -> Union[str, AsyncIterator[str]]: ...
    async def _stream_generate(
        self, payload: Dict[str, Any], cache_key: Optional[str] = ...
    ) -> AsyncIterator[str]: ...
    async def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        examples: Optional[List[Dict[str, Any]]] = ...,
        max_retries: int = ...,
    ) -> T: ...
    async def count_tokens(self, text: str) -> int: ...
    async def close(self) -> None: ...
