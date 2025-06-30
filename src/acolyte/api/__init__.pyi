from fastapi import FastAPI, Request
from fastapi.responses import Response
from typing import Callable, Awaitable, AsyncIterator
from acolyte.core.exceptions import AcolyteError
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException

app: FastAPI

async def lifespan(app: FastAPI) -> AsyncIterator[None]: ...
async def add_request_metadata(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response: ...
async def acolyte_exception_handler(request: Request, exc: AcolyteError) -> Response: ...
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> Response: ...
async def http_exception_handler(request: Request, exc: HTTPException) -> Response: ...
async def general_exception_handler(request: Request, exc: Exception) -> Response: ...

__all__ = ["app"]
