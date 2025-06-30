"""
Conversation Service - Conversation Management.

Implements persistence in SQLite with keyword search.
Includes retry logic for critical database operations.
Weaviate is only used for code (via IndexingService), not for conversations.
"""

from acolyte.core.logging import logger
from acolyte.core.tracing import MetricsCollector
from acolyte.core.database import get_db_manager, FetchType
from acolyte.core.exceptions import DatabaseError, NotFoundError
from acolyte.core.events import event_bus, EventType, CacheInvalidateEvent, Event
from acolyte.core.token_counter import SmartTokenCounter
from acolyte.core.secure_config import Settings
from acolyte.core.id_generator import generate_id
from acolyte.core.utils.retry import retry_async

from acolyte.models.conversation import ConversationSearchRequest, ConversationSearchResult
from typing import List, Optional, Dict, Any, cast
from datetime import datetime
from acolyte.core.utils.datetime_utils import utc_now_iso
import json
import time


class ConversationService:
    """
    Manages conversations with persistence in SQLite:
    - SQLite: Summaries (~90% reduction) + metadata + keyword search
    - Weaviate: Only for code indexing (not conversations)

    IMPORTANT:
    - Stores SUMMARIES, not full conversations (Decision #1)
    - session_id is generated automatically with generate_id()
    - related_sessions maintains temporal continuity (Decision #4)
    - total_tokens is a cumulative counter for statistics (Decision #12)
    - Uses retry_async() for critical DB operations with retries
    """

    def __init__(self) -> None:
        self.metrics = MetricsCollector()
        self.token_counter = SmartTokenCounter()  # Para cálculo preciso de tokens
        self.config = Settings()  # Para límites configurables
        self.db = get_db_manager()  # Obtener instancia singleton de DatabaseManager
        self._session_cache = None  # Cache opcional para linter

        # HybridSearch removed - conversations are in SQLite, not Weaviate
        # Weaviate is only for code chunks, not conversation data

        # Suscribirse a eventos de invalidación de cache
        self._cache_subscription = event_bus.subscribe(
            EventType.CACHE_INVALIDATE,
            self._handle_cache_invalidation,
            filter=lambda e: isinstance(e, CacheInvalidateEvent)
            and e.target_service in ["conversation", "all"],
        )

        logger.info("ConversationService initialized")

    async def _make_retryable_operation(
        self, operation_name: str, operation: Any, *args: Any, **kwargs: Any
    ) -> Any:
        """Wrapper to handle is_retryable() logic for retry_async."""
        try:
            return await operation(*args, **kwargs)
        except DatabaseError as e:
            if not e.is_retryable():
                raise  # Don't retry non-retryable errors
            raise  # Let retry_async handle retryable ones
        except Exception as e:
            # Convert to DatabaseError like original
            raise DatabaseError(f"{operation_name} error: {str(e)}", cause=e)

    async def create_session(self, initial_message: str) -> str:
        """
        Crea nueva sesión con ID único.

        IMPORTANTE:
        - Genera session_id with generate_id() - 32 chars hex
        - Busca automáticamente sesiones relacionadas
        - NO requiere task_checkpoint_id (puede ser None)

        Args:
            initial_message: Mensaje inicial para contexto de búsqueda

        Returns:
            ID de la sesión creada (32 caracteres hex)

        Raises:
            DatabaseError: Si falla la creación en BD
        """
        start_time = time.time()
        try:
            # Generar ID único
            session_id = generate_id()

            # Obtener última sesión para continuidad
            last_session = await self.get_last_session()
            related_sessions = []

            if last_session:
                # Mantener cadena de continuidad
                related_sessions = [last_session["id"]]
                # Si la última sesión tenía relacionadas, incluir algunas
                if last_session.get("related_sessions"):
                    prev_related = json.loads(last_session["related_sessions"])
                    # Mantener hasta 5 sesiones en la cadena (configurable)
                    max_chain = self.config.get("limits.related_sessions_chain", 5)
                    related_sessions.extend(prev_related[: max_chain - 1])

            # Buscar sesiones relacionadas por keywords en SQLite
            if initial_message:
                try:
                    keyword_related = await self.find_related_sessions(
                        query=initial_message, current_session_id=session_id, limit=3
                    )
                    # Añadir solo IDs únicos
                    for session in keyword_related:
                        if session["id"] not in related_sessions:
                            related_sessions.append(session["id"])
                except DatabaseError as e:
                    # No es crítico si no se encuentran sesiones relacionadas
                    logger.warning("Could not find related sessions", error=str(e))
                    # Continuar sin sesiones relacionadas por keywords

            # Crear entrada inicial de sesión en conversations
            # Usamos la tabla conversations adaptada para resúmenes de sesión
            operation_name = "create_session"
            attempt_count = 0

            async def db_operation_with_counter() -> Any:
                nonlocal attempt_count
                attempt_count += 1
                return await self.db.execute_async(
                    """
                    INSERT INTO conversations (
                        session_id, role, content, content_summary,
                        metadata, related_sessions, total_tokens
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,  # session_id único
                        "system",  # Role 'system' para resúmenes de sesión
                        "",  # Content vacío al inicio (se llenará con resúmenes)
                        "",  # Keywords vacíos al inicio
                        json.dumps(
                            {
                                "session_type": "conversation",
                                "created_at": utc_now_iso(),
                            }
                        ),
                        json.dumps(
                            related_sessions[: self.config.get("limits.max_related_sessions", 10)]
                        ),
                        0,  # total_tokens empieza en 0
                    ),
                    FetchType.NONE,
                )

            try:
                await retry_async(
                    lambda: self._make_retryable_operation(
                        operation_name, db_operation_with_counter
                    ),
                    max_attempts=3,
                    retry_on=(DatabaseError,),
                    initial_delay=0.5,
                    backoff="exponential",
                    logger=logger,
                )

                # Handle metrics for successful retries
                if attempt_count > 1:
                    self.metrics.increment("services.conversation_service.db_retries_successful")

            except DatabaseError:
                # All retries exhausted
                self.metrics.increment("services.conversation_service.db_retries_exhausted")
                raise

            self.metrics.increment("services.conversation_service.sessions_created")
            logger.info(
                "Created new session",
                session_id=session_id,
                related_count=len(related_sessions),
            )

            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record(
                "services.conversation_service.session_creation_time_ms", elapsed_ms
            )
            return str(session_id)

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record(
                "services.conversation_service.session_creation_time_ms", elapsed_ms
            )
            logger.error("Failed to create session", error=str(e))
            raise DatabaseError(f"Failed to create session: {str(e)}") from e

    async def save_conversation_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        summary: str,  # Viene de Semantic (~90% reducción)
        tokens_used: int,
        task_checkpoint_id: Optional[str] = None,
    ) -> None:
        """
        Guarda turno de conversación.

        CRÍTICO:
        - Guarda el RESUMEN, no los mensajes completos
        - Actualiza total_tokens acumulativo
        - Solo maneja resúmenes de conversación, no código

        Args:
            session_id: ID de la sesión
            user_message: Mensaje del usuario (para calcular tokens)
            assistant_response: Respuesta del asistente (para calcular tokens)
            summary: Resumen generado por Semantic
            tokens_used: Tokens consumidos en este turno
            task_checkpoint_id: ID de tarea asociada (opcional)

        Raises:
            NotFoundError: Si no existe la sesión
            DatabaseError: Si falla el guardado
        """
        start_time = time.time()
        try:
            # Obtener sesión actual
            operation_name = "get_session_for_update"
            attempt_count = 0

            async def db_operation_with_counter() -> Any:
                nonlocal attempt_count
                attempt_count += 1
                return await self.db.execute_async(
                    "SELECT total_tokens, content FROM conversations WHERE session_id = ? AND role = 'system'",
                    (session_id,),
                    FetchType.ONE,
                )

            try:
                result = await retry_async(
                    lambda: self._make_retryable_operation(
                        operation_name, db_operation_with_counter
                    ),
                    max_attempts=3,
                    retry_on=(DatabaseError,),
                    initial_delay=0.5,
                    backoff="exponential",
                    logger=logger,
                )

                # Handle metrics for successful retries
                if attempt_count > 1:
                    self.metrics.increment("services.conversation_service.db_retries_successful")

            except DatabaseError:
                # All retries exhausted
                self.metrics.increment("services.conversation_service.db_retries_exhausted")
                raise

            if not result.data:
                raise NotFoundError(f"Session {session_id} not found")

            row = cast(Dict[str, Any], result.data)  # FetchType.ONE returns Dict

            current_total = row["total_tokens"] or 0
            new_total = current_total + tokens_used

            # Concatenar resumen anterior con el nuevo (manteniendo límite)
            existing_content = row["content"] or ""
            if existing_content:
                # Mantener últimos N turnos resumidos (configurable)
                max_summary_turns = self.config.get("limits.max_summary_turns", 4)
                summary_parts = existing_content.split(" | ")[-max_summary_turns:]
                summary_parts.append(summary)
                combined_summary = " | ".join(summary_parts)
            else:
                combined_summary = summary

            # Extraer keywords para búsqueda rápida
            keywords = self._extract_keywords(combined_summary)

            # Actualizar sesión con resumen acumulado
            operation_name_update = "update_session_summary"
            attempt_count_update = 0

            async def db_operation_update_with_counter() -> Any:
                nonlocal attempt_count_update
                attempt_count_update += 1
                return await self.db.execute_async(
                    """
                    UPDATE conversations
                    SET content = ?,
                        content_summary = ?,
                        total_tokens = ?,
                        task_checkpoint_id = COALESCE(?, task_checkpoint_id),
                        metadata = json_set(metadata, '$.updated_at', ?)
                    WHERE session_id = ? AND role = 'system'
                    """,
                    (
                        combined_summary,
                        json.dumps(keywords),
                        new_total,
                        task_checkpoint_id,
                        utc_now_iso(),
                        session_id,
                    ),
                    FetchType.NONE,
                )

            try:
                await retry_async(
                    lambda: self._make_retryable_operation(
                        operation_name_update, db_operation_update_with_counter
                    ),
                    max_attempts=3,
                    retry_on=(DatabaseError,),
                    initial_delay=0.5,
                    backoff="exponential",
                    logger=logger,
                )

                # Handle metrics for successful retries
                if attempt_count_update > 1:
                    self.metrics.increment("services.conversation_service.db_retries_successful")

            except DatabaseError:
                # All retries exhausted
                self.metrics.increment("services.conversation_service.db_retries_exhausted")
                raise

            # NO guardamos mensajes completos - Decisión #1
            # Solo actualizamos el resumen acumulado en conversations

            # Calcular ratio de compresión basado en TOKENS, no caracteres
            original_tokens = self.token_counter.count_tokens(user_message + assistant_response)
            summary_tokens = self.token_counter.count_tokens(summary)
            compression_ratio = 1 - (summary_tokens / max(1, original_tokens))

            self.metrics.gauge(
                "services.conversation_service.summary_compression_ratio", compression_ratio
            )
            self.metrics.increment("services.conversation_service.conversation_turns_saved")
            self.metrics.increment(
                "services.conversation_service.total_tokens_processed", tokens_used
            )

            logger.info(
                "Saved conversation turn",
                session_id=session_id,
                compression_ratio=compression_ratio,
                tokens=tokens_used,
                original_tokens=original_tokens,
                summary_tokens=summary_tokens,
            )

            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record("services.conversation_service.save_turn_time_ms", elapsed_ms)

        except NotFoundError:
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record("services.conversation_service.save_turn_time_ms", elapsed_ms)
            raise
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record("services.conversation_service.save_turn_time_ms", elapsed_ms)
            logger.error("Failed to save turn", error=str(e))
            raise DatabaseError(f"Failed to save turn: {str(e)}") from e

    async def find_related_sessions(
        self, query: str, current_session_id: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Busca sesiones relacionadas en SQLite por keywords.

        Las conversaciones están almacenadas solo en SQLite con resúmenes.

        Args:
            query: Texto para buscar sesiones relacionadas
            current_session_id: ID de sesión actual para excluir
            limit: Máximo de resultados

        Returns:
            Lista de diccionarios con id, summary, score, created_at

        Raises:
            DatabaseError: Si falla la búsqueda en la base de datos
        """
        start_time = time.time()
        try:
            # Búsqueda directa en SQLite (las conversaciones NO están en Weaviate)
            result = await self._search_by_keywords(query, current_session_id, limit)
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record(
                "services.conversation_service.find_related_sessions_time_ms", elapsed_ms
            )
            return result

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record(
                "services.conversation_service.find_related_sessions_time_ms", elapsed_ms
            )
            logger.error("Failed to find related sessions", error=str(e))
            raise DatabaseError(f"Failed to find related sessions: {str(e)}") from e

    async def get_session_context(
        self, session_id: str, include_related: bool = True
    ) -> Dict[str, Any]:
        """
        Recupera contexto completo de sesión.

        Incluye:
        - Resúmenes de la sesión
        - Sesiones relacionadas si include_related=True
        - Task checkpoint si existe
        - Decisiones técnicas asociadas

        Args:
            session_id: ID de la sesión
            include_related: Si incluir sesiones relacionadas

        Returns:
            Dict con session, summary_turns, related_sessions, task, decisions

        Raises:
            NotFoundError: Si no existe la sesión
            DatabaseError: Si falla la recuperación
        """
        start_time = time.time()
        try:
            # Obtener sesión principal
            result = await self.db.execute_async(
                "SELECT * FROM conversations WHERE session_id = ? AND role = 'system'",
                (session_id,),
                FetchType.ONE,
            )

            if not result.data:
                raise NotFoundError(f"Session {session_id} not found")

            session = result.data
            session = cast(Dict[str, Any], session)

            context = {
                "session": session,
                "summary_turns": [],  # Resúmenes de turnos
                "related_sessions": [],
                "task": None,
                "decisions": [],
            }

            # Los resúmenes están en el campo content de conversations
            # Parsear resúmenes acumulados separados por " | "
            if session["content"]:
                summary_parts = session["content"].split(" | ")
                # Convertir a formato de turnos para compatibilidad
                for i, part in enumerate(summary_parts):
                    context["summary_turns"].append(
                        {
                            "id": f"{session_id}_turn_{i}",
                            "summary": part.strip(),
                            "tokens": len(part.split()) * 2,  # Estimación rápida
                        }
                    )

            # Obtener sesiones relacionadas si se solicita
            if include_related and session["related_sessions"]:
                related_ids: List[str] = json.loads(session["related_sessions"])
                if related_ids:
                    placeholders = ",".join("?" * len(related_ids))
                    result = await self.db.execute_async(
                        f"""
                        SELECT session_id as id, content as summary, timestamp as created_at, total_tokens
                        FROM conversations
                        WHERE session_id IN ({placeholders}) AND role = 'system'
                        """,
                        tuple(related_ids),
                        FetchType.ALL,
                    )

                    if result.data:
                        context["related_sessions"] = result.data

            # Obtener tarea si existe
            if session["task_checkpoint_id"]:
                result = await self.db.execute_async(
                    "SELECT * FROM tasks WHERE id = ?",
                    (session["task_checkpoint_id"],),
                    FetchType.ONE,
                )

                if result.data:
                    task_row = cast(Dict[str, Any], result.data)

                    # Crear objeto TaskCheckpoint para usar get_summary()
                    try:
                        from acolyte.models.task_checkpoint import (
                            TaskCheckpoint,
                            TaskType,
                            TaskStatus,
                        )

                        task_obj = TaskCheckpoint(
                            id=task_row["id"],
                            title=task_row["title"],
                            description=task_row["description"],
                            task_type=TaskType(task_row["task_type"]),
                            status=TaskStatus(task_row["status"]),
                            session_ids=[],  # Se llenará si es necesario
                            initial_context=task_row.get("description", ""),
                            key_decisions=[],  # Se llenará si es necesario
                            keywords=[],  # Se llenará si es necesario
                        )

                        # Añadir tanto datos raw como objeto para compatibilidad
                        context["task"] = task_row
                        context["task_object"] = task_obj
                        context["task_summary"] = task_obj.get_summary()  # USAR get_summary()

                    except Exception as e:
                        logger.warning("Could not create TaskCheckpoint object", error=str(e))
                        context["task"] = task_row
                        context["task_summary"] = f"{task_row['title']} ({task_row['task_type']})"

                    # Obtener decisiones de la tarea
                    result = await self.db.execute_async(
                        """
                        SELECT * FROM technical_decisions
                        WHERE task_id = ?
                        ORDER BY created_at DESC
                        LIMIT 5
                        """,
                        (session["task_checkpoint_id"],),
                        FetchType.ALL,
                    )

                    if result.data:
                        context["decisions"] = result.data

                        # Crear resumen de decisiones usando get_summary()
                        decision_summaries = []
                        for decision_row in result.data:
                            decision_row = cast(Dict[str, Any], decision_row)
                            try:
                                from acolyte.models.technical_decision import (
                                    TechnicalDecision,
                                    DecisionType,
                                )

                                # Parsear alternatives si existe
                                alternatives = []
                                if decision_row["alternatives_considered"]:
                                    alternatives = json.loads(
                                        decision_row["alternatives_considered"]
                                    )

                                decision_obj = TechnicalDecision(
                                    id=decision_row["id"],
                                    task_id=decision_row["task_id"],
                                    session_id=decision_row["session_id"],
                                    decision_type=DecisionType(decision_row["decision_type"]),
                                    title=decision_row["title"],
                                    description=decision_row["description"],
                                    rationale=decision_row["rationale"],
                                    alternatives_considered=alternatives,
                                    impact_level=decision_row["impact_level"],
                                )

                                # USAR get_summary()
                                decision_summaries.append(decision_obj.get_summary())

                            except Exception as e:
                                logger.warning(
                                    "Could not create TechnicalDecision object", error=str(e)
                                )
                                # Fallback a formato simple
                                decision_summaries.append(
                                    f"{decision_row['title']} - {decision_row['decision_type']} - Impacto: {decision_row['impact_level']}/5"
                                )

                        context["decision_summaries"] = decision_summaries

            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record(
                "services.conversation_service.get_session_context_time_ms", elapsed_ms
            )
            return context

        except NotFoundError:
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record(
                "services.conversation_service.get_session_context_time_ms", elapsed_ms
            )
            raise
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record(
                "services.conversation_service.get_session_context_time_ms", elapsed_ms
            )
            logger.error("Failed to get session context", error=str(e))
            raise DatabaseError(f"Failed to get session context: {str(e)}") from e

    async def search_conversations(
        self, request: ConversationSearchRequest
    ) -> List[ConversationSearchResult]:
        """
        Búsqueda en conversaciones usando modelos tipados.

        Ejemplo: "aquella vez que refactorizamos auth"
        Usa búsqueda SQL por keywords con validación Pydantic automática.

        Args:
            request: Búsqueda estructurada con validación Pydantic

        Returns:
            Lista de resultados tipados con validación

        Examples:
            >>> request = ConversationSearchRequest(
            ...     query="auth JWT",
            ...     limit=5,
            ...     include_completed=True
            ... )
            >>> results = await service.search_conversations(request)
            >>> results[0].relevance_score  # Type-safe access

        Raises:
            DatabaseError: Si falla la búsqueda en la base de datos
        """
        start_time = time.time()
        try:
            # Extraer parámetros del request tipado
            query = request.query

            # Preparar rango de tiempo para la búsqueda SQL
            time_range = None
            if request.date_from and request.date_to:
                time_range = (request.date_from, request.date_to)

            # Ejecutar búsqueda con todos los filtros del request
            result = await self._search_by_keywords_typed(
                query=query,
                exclude_session_id=None,
                limit=request.limit,
                time_range=time_range,
                include_completed=request.include_completed,
                task_id=request.task_id,
            )
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record(
                "services.conversation_service.search_conversations_time_ms", elapsed_ms
            )
            return result

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record(
                "services.conversation_service.search_conversations_time_ms", elapsed_ms
            )
            logger.error("Failed to search conversations", error=str(e))
            raise DatabaseError(f"Failed to search conversations: {str(e)}") from e

    async def get_last_session(self) -> Optional[Dict[str, Any]]:
        """
        Obtiene la última sesión del usuario.

        Usado para continuidad automática (Decision #7).

        Returns:
            Dict con datos de la sesión o None si no hay sesiones

        Note:
            No lanza excepciones, retorna None si falla
        """
        try:
            result = await self.db.execute_async(
                """
                SELECT * FROM conversations
                WHERE role = 'system'
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (),
                FetchType.ONE,
            )

            if not result.data:
                return None
            if isinstance(result.data, dict):
                return cast(Dict[str, Any], result.data)
            # Si por alguna razón es una lista, devolver el primer elemento casteado
            if isinstance(result.data, list) and result.data:
                logger.warning("[UNTESTED PATH] get_last_session returned list instead of dict")
                return cast(Dict[str, Any], result.data[0])
            return None

        except Exception as e:
            logger.error("Failed to get last session", error=str(e))
            return None

    async def complete_session(self, session_id: str) -> None:
        """
        Marca una sesión como completada.

        Args:
            session_id: ID de la sesión a completar

        Raises:
            DatabaseError: Si falla la actualización
        """
        try:
            # Marcar sesión como completada en metadata
            operation_name = "complete_session"
            attempt_count = 0

            async def db_operation_with_counter() -> Any:
                nonlocal attempt_count
                attempt_count += 1
                return await self.db.execute_async(
                    """
                    UPDATE conversations
                    SET metadata = json_set(
                        metadata,
                        '$.status', 'completed',
                        '$.completed_at', ?
                    )
                    WHERE session_id = ? AND role = 'system'
                    """,
                    (utc_now_iso(), session_id),
                    FetchType.NONE,
                )

            try:
                await retry_async(
                    lambda: self._make_retryable_operation(
                        operation_name, db_operation_with_counter
                    ),
                    max_attempts=3,
                    retry_on=(DatabaseError,),
                    initial_delay=0.5,
                    backoff="exponential",
                    logger=logger,
                )

                # Handle metrics for successful retries
                if attempt_count > 1:
                    self.metrics.increment("services.conversation_service.db_retries_successful")

            except DatabaseError:
                # All retries exhausted
                self.metrics.increment("services.conversation_service.db_retries_exhausted")
                raise

            logger.info("Session completed", session_id=session_id)

        except Exception as e:
            logger.warning("[UNTESTED PATH] complete_session error handling")
            logger.error("Failed to complete session", error=str(e))
            raise DatabaseError(f"Failed to complete session: {str(e)}") from e

    async def _search_by_keywords(
        self,
        query: str,
        exclude_session_id: Optional[str],
        limit: int,
        time_range: Optional[tuple[datetime, datetime]] = None,
        include_completed: bool = True,
        task_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Búsqueda en SQLite por keywords.
        """
        try:
            # Extraer palabras clave significativas
            keywords = [
                word.lower()
                for word in query.split()
                if len(word) > 3
                and word.lower()
                not in {
                    "como",
                    "cuando",
                    "donde",
                    "aquella",
                    "sobre",
                    "that",
                    "when",
                    "where",
                    "about",
                    "which",
                }
            ]

            if not keywords:
                logger.warning("[UNTESTED PATH] No keywords extracted from query")
                return []

            # Construir query SQL con LIKE
            conditions = []
            params: List[Any] = []

            for keyword in keywords[:5]:  # Máximo 5 keywords
                conditions.append("(content LIKE ? OR content_summary LIKE ?)")
                params.extend([f"%{keyword}%", f"%{keyword}%"])

            where_clause = " OR ".join(conditions)

            # Siempre filtrar por role = 'system'
            where_clause = f"role = 'system' AND ({where_clause})"

            # Añadir filtro de tiempo si existe
            if time_range:
                where_clause = f"{where_clause} AND timestamp BETWEEN ? AND ?"
                params.extend([time_range[0].isoformat(), time_range[1].isoformat()])

            # Excluir sesión actual si se especifica
            if exclude_session_id:
                where_clause = f"{where_clause} AND session_id != ?"
                params.append(exclude_session_id)

            # Filtro por estado completado
            if not include_completed:
                where_clause = (
                    f"{where_clause} AND json_extract(metadata, '$.status') != 'completed'"
                )

            # Filtro por tarea específica
            if task_id:
                where_clause = f"{where_clause} AND task_checkpoint_id = ?"
                params.append(task_id)

            query_sql = f"""
                SELECT session_id as id, content as summary, timestamp as created_at, task_checkpoint_id
                FROM conversations
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
            """
            params.append(limit)

            result = await self.db.execute_async(query_sql, tuple(params), FetchType.ALL)

            if not result.data:
                return []

            results = []
            for row in result.data:
                row = cast(Dict[str, Any], row)
                results.append(
                    {
                        "id": row["id"],
                        "summary": row["summary"] or "",
                        "score": 0.5,  # Score fijo para fallback
                        "created_at": row["created_at"],
                        "task_id": row["task_checkpoint_id"],
                    }
                )

            return results

        except Exception as e:
            logger.warning("[UNTESTED PATH] _search_by_keywords error handling")
            logger.error("Keyword search failed", error=str(e))
            raise  # Re-lanzar la excepción para que el llamador la maneje

    async def _search_by_keywords_typed(
        self,
        query: str,
        exclude_session_id: Optional[str],
        limit: int,
        time_range: Optional[tuple[datetime, datetime]],
        include_completed: bool = True,
        task_id: Optional[str] = None,
    ) -> List[ConversationSearchResult]:
        """
        Búsqueda en SQLite retornando modelos tipados.
        """
        try:
            # Usar método de búsqueda por keywords para obtener resultados raw
            raw_results = await self._search_by_keywords(
                query, exclude_session_id, limit, time_range, include_completed, task_id
            )

            # Convertir a modelos tipados
            typed_results = []
            for raw_result in raw_results:
                # Convertir cada dict a ConversationSearchResult
                typed_result = ConversationSearchResult(
                    conversation_id=raw_result["id"],
                    session_id=raw_result["id"],  # Mismo ID por compatibilidad
                    relevance_score=raw_result["score"],
                    summary=raw_result["summary"],
                    keywords=[],  # Extraer desde la BD si es necesario
                    message_count=0,  # Calcular si es necesario
                    created_at=raw_result["created_at"],
                    relevant_messages=[],  # Vacío por defecto
                )
                typed_results.append(typed_result)

            return typed_results

        except Exception as e:
            logger.warning("[UNTESTED PATH] _search_by_keywords_typed error handling")
            logger.error("Typed keyword search failed", error=str(e))
            raise  # Re-lanzar la excepción para que el llamador la maneje

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extrae keywords de un texto para búsqueda rápida.

        Simple extracción basada en palabras significativas.
        """
        if not text:
            return []

        # Palabras a ignorar (stopwords simplificadas)
        stopwords = {
            "el",
            "la",
            "de",
            "en",
            "y",
            "a",
            "que",
            "es",
            "por",
            "con",
            "para",
            "los",
            "las",
            "un",
            "una",
            "su",
            "al",
            "del",
            "se",
            "como",
            "más",
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "after",
        }

        # Extraer palabras significativas
        words = text.lower().split()
        keywords = []

        for word in words:
            # Limpiar puntuación
            word = word.strip(".,!?;:()[]{}\"'")

            # Filtrar por longitud y stopwords
            if len(word) > 2 and word not in stopwords:
                if word not in keywords:  # Evitar duplicados
                    keywords.append(word)

        # Limitar cantidad de keywords
        return keywords[:20]

    async def _handle_cache_invalidation(self, event: Event) -> None:
        """
        Maneja eventos de invalidación de cache.

        Args:
            event: Evento con información de qué invalidar
        """
        # Cast to CacheInvalidateEvent for type safety
        if not isinstance(event, CacheInvalidateEvent):
            logger.error("Invalid event type for cache invalidation", event_type=type(event))
            return

        cache_event = event  # Now we know it's CacheInvalidateEvent
        logger.info(
            "Handling cache invalidation",
            source=cache_event.source,
            pattern=cache_event.key_pattern,
            reason=cache_event.reason,
        )

        # No HybridSearch cache to invalidate - conversations are in SQLite
        # Just log the event for awareness
        self.metrics.increment("services.conversation_service.cache_invalidation_events")
        logger.info("Cache invalidation event received", pattern=cache_event.key_pattern)

        # Limpiar cache local si existe
        if hasattr(self, "_session_cache") and self._session_cache is not None:
            logger.warning("[UNTESTED PATH] _session_cache clearing")
            self._session_cache.clear()
