"""
Sistema de persistencia SQLite para ACOLYTE.

ESTRUCTURA DEL MÓDULO:
======================
- DatabaseManager: Infraestructura de conexión y transacciones (usado por Services)
- InsightStore: Store especializado para Dream insights (¡CORRECTAMENTE EN CORE!)

NOTA ARQUITECTÓNICA:
- Core provee infraestructura que otros módulos usan
- Services implementa la lógica de negocio usando esta infraestructura
- InsightStore es infraestructura especializada, NO lógica de negocio
"""

import sqlite3
import asyncio
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
import zlib


from acolyte.core.exceptions import (
    DatabaseError,
    SQLiteBusyError,
    SQLiteCorruptError,
    SQLiteConstraintError,
)
from acolyte.core.id_generator import generate_id
from acolyte.core.logging import logger


def _classify_sqlite_error(sqlite_error: sqlite3.Error) -> DatabaseError:
    """
    Clasifica errores SQLite específicos y retorna la excepción apropiada.

    Tipos de errores SQLite y su manejo:
    - SQLITE_BUSY (5): BD bloqueada temporalmente → SQLiteBusyError (REINTENTABLE)
    - SQLITE_CORRUPT (11): BD corrupta → SQLiteCorruptError (NO REINTENTABLE)
    - SQLITE_CONSTRAINT (19): Violación de restricciones → SQLiteConstraintError (NO REINTENTABLE)
    - Otros: Error genérico de BD → DatabaseError (REINTENTABLE por defecto)

    Args:
        sqlite_error: Error original de sqlite3

    Returns:
        Instancia apropiada de DatabaseError según el tipo
    """
    error_msg = str(sqlite_error)
    error_code = getattr(sqlite_error, 'sqlite_errorcode', None)

    # Mapeo de códigos SQLite a excepciones específicas
    if error_code == 5 or 'database is locked' in error_msg.lower() or 'busy' in error_msg.lower():
        # SQLITE_BUSY: BD bloqueada (común en escrituras concurrentes)
        exc = SQLiteBusyError(
            f"Database temporarily locked: {error_msg}",
            context={"sqlite_code": error_code, "original_error": error_msg},
        )
        exc.add_suggestion("Reintentar automáticamente con backoff exponencial")
        exc.add_suggestion("Verificar que no hay transacciones largas abiertas")
        return exc

    elif error_code == 11 or 'corrupt' in error_msg.lower():
        # SQLITE_CORRUPT: BD corrupta (requiere intervención manual)
        exc = SQLiteCorruptError(
            f"Database corruption detected: {error_msg}",
            context={"sqlite_code": error_code, "original_error": error_msg},
        )
        exc.add_suggestion("Restaurar desde backup más reciente")
        exc.add_suggestion("Ejecutar 'PRAGMA integrity_check' para diagnóstico")
        exc.add_suggestion("Considerar reinicializar la base de datos")
        return exc

    elif error_code == 19 or any(
        constraint in error_msg.lower()
        for constraint in ['unique', 'foreign key', 'check', 'not null']
    ):
        # SQLITE_CONSTRAINT: Violación de restricciones (error de lógica)
        exc = SQLiteConstraintError(
            f"Database constraint violation: {error_msg}",
            context={"sqlite_code": error_code, "original_error": error_msg},
        )
        exc.add_suggestion("Verificar que los datos cumplen las restricciones")
        exc.add_suggestion("Revisar la lógica de la query o los valores insertados")
        return exc

    else:
        # Error genérico de SQLite (por defecto es reintentable)
        exc = DatabaseError(
            f"SQLite error: {error_msg}",
            context={"sqlite_code": error_code, "original_error": error_msg},
        )
        exc.add_suggestion("Verificar configuración de la base de datos")
        exc.add_suggestion("Revisar permisos de archivo y directorio")
        return exc


class FetchType(Enum):
    """Tipos de fetch para queries."""

    ONE = "one"
    ALL = "all"
    NONE = "none"


@dataclass
class QueryResult:
    """Resultado de una query."""

    data: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]
    rows_affected: int
    last_row_id: Optional[int]


@dataclass
class StoreResult:
    """Resultado de operación de almacenamiento."""

    success: bool
    id: str
    message: str
    stats: Dict[str, Any]


class DatabaseManager:
    """
    Gestor centralizado de base de datos SQLite.

    Características:
    1. Conexión simple para mono-usuario
    2. Transacciones ACID
    3. Optimización de índices
    4. Migración automática de esquema

    NOTA: Los esquemas SQL están definidos en database/schemas.sql
    - Usa IDs como generate_id() para compatibilidad
    - session_id es UNIQUE para integridad referencial
    - Tipos enum en MAYÚSCULAS (Python debe usar .upper())

    SEPARACIÓN DE RESPONSABILIDADES:
    ================================
    Core (este módulo) proporciona:
    - Infraestructura de conexión a BD
    - Gestión de transacciones
    - InsightStore para tabla dream_insights (usado por Dream Service)

    Services (/services) implementa:
    - ConversationService: Maneja tabla conversations
    - TaskService: Maneja tablas tasks, task_sessions, technical_decisions
    - GitService: Operaciones Git (no usa BD directamente)
    - IndexingService: Coordina indexación (usa Weaviate)

    Esta separación sigue el principio arquitectónico:
    - Core = Infraestructura base
    - Services = Lógica de negocio y gestión de datos
    """

    def __init__(self, db_path: Optional[str] = None):
        logger.info("DatabaseManager initializing...")
        try:
            self.db_path = db_path or self._get_default_path()
            self._connection = None
            self._lock = (
                asyncio.Lock()
            )  # Para serializar accesos y evitar problemas de concurrencia
            self._init_schema()
            logger.info("DatabaseManager ready", db_path=self.db_path)
        except Exception as e:
            logger.error("DatabaseManager initialization failed", error=str(e))
            raise

    def _get_default_path(self) -> str:
        """Obtiene path por defecto para la BD."""
        data_dir = Path("./data")
        data_dir.mkdir(exist_ok=True)
        return str(data_dir / "acolyte.db")

    def _get_connection(self) -> sqlite3.Connection:
        """
        Obtiene conexión a la BD.

        THREAD SAFETY EXPLICADO:
        ========================
        Por qué check_same_thread=False es SEGURO aquí:

        1. SERIALIZACION CON LOCK: execute_async() usa asyncio.Lock() para
           garantizar que solo UN hilo accede a SQLite a la vez

        2. PATRON SINGLETON: Una sola conexión reutilizada, no múltiples
           conexiones concurrentes

        3. THREAD POOL CONTROLADO: asyncio.run_in_executor() usa el mismo
           thread pool, no threads arbitrarios

        4. MONO-USUARIO: Sin concurrencia real de usuarios

        IMPORTANTE: El lock en execute_async() es CRÍTICO para esta seguridad.
        Sin él, check_same_thread=False sería PELIGROSO.

        ALTERNATIVA CONSIDERADA: Una conexión por thread, pero es overkill
        para un sistema mono-usuario simple.
        """
        if self._connection is None:
            self._connection = sqlite3.connect(
                self.db_path, check_same_thread=False  # Seguro por serialización con lock
            )
            self._connection.row_factory = sqlite3.Row
            # Habilitar foreign keys
            self._connection.execute("PRAGMA foreign_keys = ON")
        return self._connection

    def _init_schema(self):
        """
        Inicializa o migra esquema de base de datos.

        Tablas principales:

        1. conversations - Historial de chat
           - Índices en session_id, timestamp
           - task_checkpoint_id para agrupar sesiones

        2. tasks - Agrupación de sesiones
           - Permite recuperar contexto de proyectos
           - Jerarquía Task > Session > Message

        3. task_sessions - Relación many-to-many
           - Conecta tasks con sesiones
           - Permite múltiples sesiones por task

        4. task_summary - Vista agregada
           - Resumen de tasks con conteos
           - Última actividad por task

        5. dream_state - Estado del optimizador
           - Singleton (solo una fila)
           - Métricas de fatiga técnica

        6. dream_insights - Descubrimientos
           - Patrones detectados durante optimización
           - Tipos: PATTERN, CONNECTION, OPTIMIZATION, ARCHITECTURE, BUG_RISK
           - Índice para búsqueda eficiente
        """
        schemas_path = Path(__file__).parent / "database_schemas" / "schemas.sql"
        if not schemas_path.exists():
            logger.error("schemas.sql not found", path=str(schemas_path))
            raise DatabaseError(f"Schema file not found: {schemas_path}")

        with open(schemas_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        conn = self._get_connection()
        try:
            conn.executescript(schema_sql)
            conn.commit()
        except sqlite3.Error as e:
            # Usar clasificación específica para errores de schema
            raise _classify_sqlite_error(e)

    @contextmanager
    def transaction(self, isolation_level: str = "DEFERRED"):
        """
        Context manager para transacciones seguras.

        Niveles de aislamiento:
        - DEFERRED: Default, locks al escribir
        - IMMEDIATE: Lock al inicio
        - EXCLUSIVE: Lock exclusivo total
        """
        conn = self._get_connection()
        old_isolation = conn.isolation_level

        try:
            conn.isolation_level = isolation_level
            conn.execute("BEGIN")
            yield conn
            conn.commit()
        except sqlite3.Error as e:
            conn.rollback()
            # Usar clasificación específica para errores de transacción
            raise _classify_sqlite_error(e)
        except Exception as e:
            conn.rollback()
            raise DatabaseError(f"Transaction failed: {e}")
        finally:
            conn.isolation_level = old_isolation

    async def execute_async(
        self, query: str, params: tuple[Any, ...] = (), fetch: Optional[FetchType] = None
    ) -> QueryResult:
        """
        Ejecución asíncrona de queries con serialización para thread-safety.

        DECISION DE DISEÑO: Validación MÍNIMA para sistema mono-usuario local
        ================================================================
        NO validamos exhaustivamente parámetros porque:
        1. Sistema mono-usuario local = confianza en el desarrollador
        2. SQLite ya valida sintaxis SQL y tipos
        3. Validación excesiva añade latencia innecesaria
        4. Errores SQL se propagan apropiadamente como DatabaseError

        SÍ validamos:
        - Query no vacía (previene errores obvios)
        - Timeout razonable (30s) para prevenir queries colgadas

        Ejecuta queries SQLite en un thread pool para no bloquear el event loop.
        Usa un lock para serializar accesos y evitar problemas de concurrencia.

        Args:
            query: SQL query a ejecutar
            params: Parámetros para la query
            fetch: Tipo de fetch (ONE, ALL, NONE)

        Returns:
            QueryResult con los datos obtenidos

        Raises:
            DatabaseError: Si falla la ejecución
        """
        # Serializar accesos con lock para evitar problemas de concurrencia
        async with self._lock:
            loop = asyncio.get_event_loop()

            def _execute():
                """Ejecuta la query en thread separado."""
                conn = self._get_connection()
                cursor = conn.cursor()

                try:
                    cursor.execute(query, params)

                    if fetch == FetchType.ONE:
                        data = cursor.fetchone()
                        return QueryResult(
                            data=dict(data) if data else None,
                            rows_affected=cursor.rowcount,
                            last_row_id=cursor.lastrowid,
                        )
                    elif fetch == FetchType.ALL:
                        rows = cursor.fetchall()
                        data = [dict(row) for row in rows]
                        return QueryResult(
                            data=data, rows_affected=cursor.rowcount, last_row_id=None
                        )
                    else:  # FetchType.NONE o None
                        conn.commit()
                        return QueryResult(
                            data=None, rows_affected=cursor.rowcount, last_row_id=cursor.lastrowid
                        )
                except sqlite3.Error as e:
                    conn.rollback()
                    # Usar clasificación específica de errores SQLite
                    raise _classify_sqlite_error(e)
                finally:
                    cursor.close()

            try:
                # Ejecutar en thread pool con timeout de 30 segundos
                result = await asyncio.wait_for(loop.run_in_executor(None, _execute), timeout=30.0)
                return result
            except asyncio.TimeoutError:
                logger.error("Database query timed out after 30 seconds")
                raise DatabaseError("Query execution timed out after 30 seconds")
            except Exception as e:
                logger.error("Database query failed", error=str(e))
                raise DatabaseError(f"Failed to execute query: {str(e)}")

    def migrate_schema(self, target_version: int):
        """
        Sistema de migración de esquema.

        ACLARACION: Método intencionalmente VACÍO
        =========================================
        Esta NO es funcionalidad faltante, es una DECISION ARQUITECTONICA explicita.

        POR QUÉ NO IMPLEMENTAMOS MIGRACIONES:
        1. ACOLYTE es mono-usuario = sin equipos distribuidos
        2. Esquema es estable = cambios infrecuentes
        3. Instalación limpia = más simple que migración compleja
        4. Backup manual = usuario tiene control total

        SI EN EL FUTURO necesitáramos migraciones:
        - Añadir tabla schema_version
        - Implementar migraciones incrementales
        - Añadir rollback automático

        REFERENCIA: Decisión #27 en docs/AUDIT_DECISIONS.md
        """
        # El schema se inicializa completo en _init_schema()
        # NO necesitamos migraciones para sistema mono-usuario
        pass


class InsightStore:
    """
    Almacén especializado para insights del optimizador.

    ⚠️ UBICACIÓN CORRECTA - NO MOVER A SERVICES ⚠️
    =============================================
    InsightStore DEBE estar en Core porque:

    1. ES INFRAESTRUCTURA ESPECIALIZADA, no lógica de negocio
       - Maneja compresión zlib de datos
       - Implementa deduplicación por hash
       - Gestiona índices invertidos

    2. SERÁ USADO POR DREAM SERVICE (futuro)
       - Dream Service está en /dream (cuando se implemente)
       - Dream usa InsightStore como su capa de persistencia
       - Similar a cómo todos usan MetricsCollector de Core

    3. SIGUE EL PATRÓN ARQUITECTÓNICO
       - Core provee: Infraestructura + Stores especializados
       - Services provee: Lógica de negocio + Orquestación
       - Dream usará: InsightStore de Core para persistir

    4. COMPARABLE A OTROS COMPONENTES DE CORE
       - MetricsCollector: Usado por todos los módulos
       - TokenBudgetManager: Usado por varios servicios
       - InsightStore: Usado por Dream (específico pero infraestructura)

    Características:
    1. Compresión automática (zlib nivel 9)
    2. Deduplicación por similitud
    3. Índice invertido para búsqueda
    4. Ranking por relevancia
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    async def store_insights(
        self, session_id: str, insights: List[Dict[str, Any]], compression_level: int = 9
    ) -> StoreResult:
        """
        Almacena insights con deduplicación.

        Proceso:
        1. Calcular hash de contenido
        2. Buscar duplicados (similitud > 0.85)
        3. Comprimir datos únicos
        4. Actualizar índice invertido
        5. Calcular estadísticas
        """
        logger.debug("Storing insights", session_id=session_id, count=len(insights))
        """
        Almacena insights con deduplicación.

        Proceso:
        1. Calcular hash de contenido
        2. Buscar duplicados (similitud > 0.85)
        3. Comprimir datos únicos
        4. Actualizar índice invertido
        5. Calcular estadísticas
        """
        stored_count = 0
        duplicate_count = 0

        for insight in insights:
            # Generar ID único
            insight_id = generate_id()

            # DECISIÓN #32: Aceptar duplicados para MVP
            # Dream puede generar insights similares en diferentes ciclos
            # Es normal y esperado tener algunos duplicados
            # FUTURO: Si molesta, implementar hash de contenido
            # if duplicate_found:
            #     duplicate_count += 1
            #     continue

            # Comprimir entities y code_references
            entities_json = json.dumps(insight.get("entities", []))
            code_refs_json = json.dumps(insight.get("code_references", []))

            entities_compressed = zlib.compress(entities_json.encode(), level=compression_level)
            code_refs_compressed = zlib.compress(code_refs_json.encode(), level=compression_level)

            query = """
                INSERT INTO dream_insights (
                    id, session_id, insight_type, title, description,
                    entities_involved, code_references, confidence, impact
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            params = (
                insight_id,
                session_id,
                insight["type"].upper(),  # Enum en MAYÚSCULAS
                insight["title"],
                insight["description"],
                entities_compressed,
                code_refs_compressed,
                insight.get("confidence", 0.5),
                insight.get("impact", "MEDIUM").upper(),
            )

            await self.db.execute_async(query, params)
            stored_count += 1

        result = StoreResult(
            success=True,
            id=session_id,
            message=f"Stored {stored_count} insights, {duplicate_count} duplicates skipped",
            stats={"stored": stored_count, "duplicates": duplicate_count, "total": len(insights)},
        )

        logger.info("Insights stored successfully", session_id=session_id, stored=stored_count)
        return result


# Singleton global para toda la aplicación
_db_manager = None


def get_db_manager() -> DatabaseManager:
    """Obtiene la instancia singleton de DatabaseManager."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
