"""
Sistema de eventos para WebSockets en ACOLYTE.
"""

import json
import asyncio
from enum import Enum
from abc import ABC, abstractmethod
from typing import Dict, Set, List, Optional, Any
from datetime import datetime
from collections import defaultdict, deque

from fastapi import WebSocket
from acolyte.core.logging import logger
from acolyte.core.id_generator import generate_id
from acolyte.core.utils.datetime_utils import utc_now
from acolyte.core.utils.retry import retry_async


class EventType(Enum):
    """Tipos de eventos del sistema."""

    PROGRESS = "progress"
    LOG = "log"
    STATUS = "status"
    ERROR = "error"
    INSIGHT = "insight"
    OPTIMIZATION_NEEDED = "optimization_needed"
    CACHE_INVALIDATE = "cache_invalidate"  # Invalidación de cache coordinada


class Event(ABC):
    """
    Clase base para todos los eventos.

    Atributos requeridos:
    - id: Hex de 32 caracteres
    - type: Tipo de evento
    - timestamp: Momento del evento
    - source: Origen del evento
    """

    id: str
    timestamp: datetime
    type: EventType

    def __init__(self):
        self.id = generate_id()
        self.timestamp = utc_now()

    @abstractmethod
    def to_json(self) -> str:
        """Serializa a JSON para WebSocket."""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Valida integridad del evento."""
        pass


class CacheInvalidateEvent(Event):
    """
    Evento para invalidar cache entre servicios.

    Ejemplo:
        event = CacheInvalidateEvent(
            source="git_service",
            target_service="conversation",
            key_pattern="session_*",
            reason="Files updated after pull"
        )
    """

    def __init__(self, source: str, target_service: str, key_pattern: str = "*", reason: str = ""):
        super().__init__()
        self.type = EventType.CACHE_INVALIDATE
        self.source = source
        self.target_service = target_service
        self.key_pattern = key_pattern
        self.reason = reason

    def to_json(self) -> str:
        return json.dumps(
            {
                "id": self.id,
                "type": self.type.value,
                "timestamp": self.timestamp.isoformat(),
                "source": self.source,
                "target_service": self.target_service,
                "key_pattern": self.key_pattern,
                "reason": self.reason,
            }
        )

    def validate(self) -> bool:
        return all([self.source, self.target_service, self.key_pattern])


class ProgressEvent(Event):
    """
    Evento para notificar progreso de operaciones largas.

    Ejemplo:
        event = ProgressEvent(
            source="indexing_service",
            operation="indexing_files",
            current=50,
            total=100,
            message="Processing file: main.py",
            task_id="idx_123_abc",
            files_skipped=5,
            chunks_created=150,
            embeddings_generated=150,
            errors=2,
            current_file="src/main.py"
        )
    """

    def __init__(
        self,
        source: str,
        operation: str,
        current: int,
        total: int,
        message: str = "",
        task_id: Optional[str] = None,
        files_skipped: int = 0,
        chunks_created: int = 0,
        embeddings_generated: int = 0,
        errors: int = 0,
        current_file: Optional[str] = None,
    ):
        super().__init__()
        self.type = EventType.PROGRESS
        self.source = source
        self.operation = operation
        self.current = current
        self.total = total
        self.percentage = (current / total * 100) if total > 0 else 0
        self.message = message
        self.task_id = task_id
        # Nuevos campos para estadísticas completas
        self.files_skipped = files_skipped
        self.chunks_created = chunks_created
        self.embeddings_generated = embeddings_generated
        self.errors = errors
        self.current_file = current_file

    def to_json(self) -> str:
        return json.dumps(
            {
                "id": self.id,
                "type": self.type.value,
                "timestamp": self.timestamp.isoformat(),
                "source": self.source,
                "operation": self.operation,
                "current": self.current,
                "total": self.total,
                "percentage": round(self.percentage, 1),
                "message": self.message,
                "task_id": self.task_id,
                "files_skipped": self.files_skipped,
                "chunks_created": self.chunks_created,
                "embeddings_generated": self.embeddings_generated,
                "errors": self.errors,
                "current_file": self.current_file,
            }
        )

    def validate(self) -> bool:
        return all([self.source, self.operation, self.current >= 0, self.total >= 0])


class EventBus:
    """
    Bus de eventos centralizado para sistema mono-usuario.

    Características:
    1. Pub/Sub simple
    2. Filtrado por tipo/origen
    3. Persistencia opcional
    4. Replay de eventos
    """

    _subscribers: Dict[EventType, Set[Any]]
    _event_store: "deque[Event]"
    _filters: List[Any]

    def __init__(self):
        self._subscribers = defaultdict(set)
        self._event_store = deque(maxlen=10000)
        self._filters = []
        logger.info("EventBus initialized", max_events=10000)

    async def publish(self, event: Event):
        """
        Publica evento a suscriptores.

        Proceso:
        1. Validar evento
        2. Aplicar filtros globales
        3. Almacenar si está configurado
        4. Notificar suscriptores relevantes
        5. Manejar errores sin detener propagación
        """
        # Validar evento
        if not event.validate():
            logger.error("Invalid event validation failed", event_type=event.type.value)
            raise ValueError(f"Invalid event: {event}")

        # Almacenar en event store
        self._event_store.append(event)

        # Obtener suscriptores para este tipo de evento
        subscribers = self._subscribers.get(event.type, set())

        # Notificar a cada suscriptor
        for handler, filter_func in subscribers:
            try:
                # Aplicar filtro si existe
                if filter_func and not filter_func(event):
                    continue

                # Llamar al handler
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)

            except Exception as e:
                # Log error pero continuar con otros suscriptores
                logger.error(
                    "Event handler failed",
                    event_type=event.type.value,
                    handler=handler.__name__,
                    error=str(e),
                )

    def subscribe(self, event_type: EventType, handler, filter=None):
        """
        Suscribe handler a tipo de evento.

        Returns:
            Subscription que permite unsubscribe
        """
        # Añadir suscriptor
        self._subscribers[event_type].add((handler, filter))

        # Retornar función para desuscribirse
        def unsubscribe():
            self._subscribers[event_type].discard((handler, filter))

        return unsubscribe

    async def replay(
        self,
        from_timestamp: datetime,
        to_timestamp: Optional[datetime] = None,
        event_types: Optional[List[EventType]] = None,
    ) -> List[Event]:
        """
        Reproduce eventos históricos.

        Útil para:
        - Recuperación de estado
        - Debugging
        - Auditoría

        Args:
            from_timestamp: Timestamp inicial (inclusive)
            to_timestamp: Timestamp final (inclusive), None = hasta ahora
            event_types: Lista de tipos a filtrar, None = todos

        Returns:
            Lista de eventos que cumplen los criterios
        """
        filtered_events = []

        # Si no hay timestamp final, usar ahora
        if to_timestamp is None:
            to_timestamp = utc_now()

        # Filtrar eventos del store
        for event in self._event_store:
            # Verificar timestamp
            if event.timestamp < from_timestamp:
                continue
            if event.timestamp > to_timestamp:
                continue

            # Verificar tipo si se especificó filtro
            if event_types and event.type not in event_types:
                continue

            # Evento cumple todos los criterios
            filtered_events.append(event)

        return filtered_events


class WebSocketManager:
    """
    Gestor de conexión WebSocket única.

    Responsabilidades:
    1. Gestión de conexión única (mono-usuario)
    2. Heartbeat para detectar desconexiones
    3. Envío de eventos
    4. Reconexión automática
    """

    _websocket: Optional[WebSocket]
    _connection_id: Optional[str]
    _heartbeat_task: Optional["asyncio.Task[None]"]
    is_connected_flag: bool

    HEARTBEAT_INTERVAL = 30  # segundos
    MAX_SEND_ATTEMPTS = 2

    def __init__(self):
        """Inicializa el WebSocketManager."""
        self._websocket = None
        self._connection_id = None
        self._heartbeat_task = None
        self.is_connected_flag = False
        logger.info("WebSocketManager initialized")

    async def connect(self, websocket: WebSocket) -> str:
        """
        Conecta WebSocket único.

        Proceso:
        1. Cerrar conexión existente si hay
        2. Generar connection_id con generate_id()
        3. Iniciar heartbeat
        4. Enviar estado inicial
        """
        # Limpiar conexión anterior si existe
        if self._websocket:
            logger.warning(
                "New WebSocket connection requested while old one exists. Closing old one.",
                old_connection_id=self._connection_id,
            )
            await self.disconnect()

        # Aceptar nueva conexión
        await websocket.accept()
        self._websocket = websocket
        self._connection_id = generate_id()
        self.is_connected_flag = True
        logger.info("WebSocket connected", connection_id=self._connection_id)

        # Enviar mensaje de conexión exitosa
        await self._send_connection_message(self._connection_id)

        # Iniciar heartbeat
        self._start_heartbeat()

        return self._connection_id

    async def _send_connection_message(self, connection_id: str) -> None:
        """Envía el mensaje inicial de conexión."""
        if self._websocket:
            await self._websocket.send_json({"type": "connected", "connection_id": connection_id})

    async def send_event(self, event: Event):
        """
        Envía un evento a través del WebSocket.

        Implementa reintentos en caso de fallo.
        """
        if not self.is_connected() or not self._websocket:
            return

        try:
            payload = event.to_json()
        except Exception as e:
            logger.error(
                "Event serialization failed, cannot send via WebSocket",
                event_type=event.type.value,
                error=str(e),
            )
            return

        # Use centralized retry logic
        try:
            await retry_async(
                lambda: self._websocket.send_json(json.loads(payload)),
                max_attempts=self.MAX_SEND_ATTEMPTS,
                backoff="linear",
                initial_delay=0.1,
                retry_on=(Exception,),
                logger=logger,
            )
        except Exception as last_exception:
            logger.error(
                "WebSocket send failed permanently after multiple attempts, disconnecting.",
                connection_id=self._connection_id,
                max_attempts=self.MAX_SEND_ATTEMPTS,
                error=str(last_exception),
            )
            await self.disconnect()

    def _start_heartbeat(self):
        """Inicia el task de heartbeat en segundo plano. Usado en tests."""
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self):
        """Bucle de heartbeat que envía pings periódicos."""
        while self.is_connected():
            try:
                # Send ping immediately on first iteration, then wait
                if self.is_connected() and self._websocket:
                    await self._websocket.send_text("ping")
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(
                    "WebSocket heartbeat failed, disconnecting.",
                    connection_id=self._connection_id,
                    error=str(e),
                )
                await self.disconnect()
                break

    async def disconnect(self):
        """Cierra la conexión WebSocket y limpia el estado."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except (asyncio.CancelledError, TypeError):
                # TypeError puede ocurrir si la tarea es un mock no-awaitable en tests
                logger.warning("[UNTESTED BRANCH] TypeError handling for mock tasks in tests")
                pass

        ws = self._websocket
        cid = self._connection_id

        # Reset state immediately
        self._websocket = None
        self._connection_id = None
        self.is_connected_flag = False
        self._heartbeat_task = None

        if ws:
            try:
                await ws.close()
                logger.info("WebSocket disconnected", connection_id=cid)
            except Exception as e:
                logger.error("Error while closing WebSocket", connection_id=cid, error=str(e))

    def is_connected(self) -> bool:
        """Verifica si el WebSocket está conectado."""
        return self._websocket is not None and self.is_connected_flag


# Instancia global del event bus para todo el sistema
event_bus = EventBus()
