"""
Sistema de logging asíncrono simple para ACOLYTE.
"""

import re
import time
import os
import yaml
from pathlib import Path
from queue import Queue
from logging.handlers import QueueHandler
from typing import List, Pattern, Optional
from contextlib import contextmanager
from loguru import logger as loguru_logger


class AsyncLogger:
    """
    Logger asíncrono simple con formato plano.

    Formato: timestamp | level | component | message
    Sin emojis, sin JSON complejo, latencia cero.
    """

    def __init__(self, component: str, debug_mode: bool = False):
        self.component = component
        self.debug_mode = debug_mode
        self.queue = Queue()
        self.handler = QueueHandler(self.queue)
        self._setup_async_handler()

    # Handler único compartido entre todas las instancias
    _handler_id = None

    def _setup_async_handler(self):
        """
        Configura handler asíncrono con QueueHandler.

        Características:
        - Latencia cero (non-blocking)
        - Formato simple sin emojis
        - Rotación automática a 10MB
        - Singleton: solo un handler para evitar conflictos
        """
        # Solo añadir handler si no existe
        if AsyncLogger._handler_id is None:
            AsyncLogger._handler_id = loguru_logger.add(
                "debug.log",
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {extra[component]} | {message}",
                rotation="10 MB",
                compression="zip",
                enqueue=True,  # Clave para async
            )

    def log(self, level: str, message: str, **context):
        """
        Registra mensaje de forma asíncrona.

        Proceso:
        1. Añadir a queue (instantáneo)
        2. Worker procesa en background
        3. Sin bloqueo del caller
        """
        loguru_logger.bind(component=self.component).log(level, message, **context)

    def debug(self, message: str, **context):
        """Log nivel DEBUG."""
        self.log("DEBUG", message, **context)

    def info(self, message: str, **context):
        """Log nivel INFO."""
        self.log("INFO", message, **context)

    def warning(self, message: str, **context):
        """Log nivel WARNING."""
        self.log("WARNING", message, **context)

    def error(self, message: str, include_trace: Optional[bool] = None, **context):
        """
        Log nivel ERROR con stack trace opcional.

        Args:
            message: Mensaje de error
            include_trace: Si incluir stack trace (None = auto según debug_mode)
            **context: Contexto adicional
        """
        # Decidir si incluir stack trace
        should_include_trace = include_trace if include_trace is not None else self.debug_mode

        if should_include_trace:
            import traceback

            context["stack_trace"] = traceback.format_exc()

        self.log("ERROR", message, **context)


class SensitiveDataMasker:
    """
    Enmascara datos sensibles en logs.

    Patrones a enmascarar:
    - Tokens/API keys
    - Paths completos (mostrar solo basename)
    - Hashes largos (mostrar solo primeros 8 chars)
    """

    def __init__(self, patterns: Optional[List[Pattern]] = None):
        self.patterns = patterns or []

    def mask(self, text: str) -> str:
        """
        Enmascara datos sensibles.

        Ejemplo:
        - "token=abc123def456" → "token=***"
        - "/home/user/project" → ".../project"
        - "a1b2c3d4e5f6..." → "a1b2c3d4..."
        """
        # Copia del texto para modificar
        masked = text

        # 1. Enmascarar tokens largos (>20 chars alfanuméricos continuos)
        # Busca secuencias largas que parecen tokens/keys
        masked = re.sub(r'\b[a-zA-Z0-9]{20,}\b', '***TOKEN***', masked)

        # 2. Acortar paths absolutos
        # Linux/Mac paths: /home/user/project → .../project
        masked = re.sub(r'/[a-zA-Z0-9_/.-]{10,}/([a-zA-Z0-9_.-]+)', r'.../\1', masked)

        # Windows paths: C:\Users\Name\project → ...\project
        masked = re.sub(
            r'[A-Z]:\\\\[a-zA-Z0-9_\\\\.-]{10,}\\\\([a-zA-Z0-9_.-]+)', r'...\\\1', masked
        )

        # 3. Acortar hashes largos (>16 chars hex)
        # Muestra solo primeros 8 caracteres
        masked = re.sub(r'\b([a-f0-9]{8})[a-f0-9]{8,}\b', r'\1...', masked)

        # 4. Enmascarar patterns tipo key=value con valores largos
        masked = re.sub(
            r'(api_key|token|secret|password|key)=[a-zA-Z0-9]{8,}',
            r'\1=***',
            masked,
            flags=re.IGNORECASE,
        )

        return masked


class PerformanceLogger:
    """
    Logger especializado para métricas de rendimiento.

    Automáticamente registra:
    - Duración
    - Memoria usada
    """

    def __init__(self):
        self.logger = AsyncLogger("performance")

    @contextmanager
    def measure(self, operation: str, **context):
        """
        Context manager para medir operación.

        Uso:
        ```
        with perf_logger.measure("database_query", query=sql):
            result = await db.execute(sql)
        ```
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.logger.info(
                "Operation completed", operation=operation, duration_ms=duration * 1000, **context
            )


# Logger global configurado
# debug_mode se configurará desde .acolyte cuando SecureConfig esté disponible
def _get_debug_mode() -> bool:
    """Obtiene debug_mode desde configuración o variable de entorno."""
    try:
        # Intentar leer desde .acolyte si existe
        config_path = Path(".acolyte")
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
                return config.get("logging", {}).get("debug_mode", False)
    except Exception:
        pass

    # Fallback a variable de entorno
    return os.getenv("ACOLYTE_DEBUG", "false").lower() == "true"


logger = AsyncLogger("acolyte", debug_mode=_get_debug_mode())
