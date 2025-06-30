from fastapi import APIRouter
from typing import Dict, Any

router: APIRouter

def get_active_tasks() -> list[str]: ...
def get_connection_stats() -> Dict[str, Dict[str, Any]]: ...

__all__: list[str]
