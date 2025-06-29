"""
Git Service - Operaciones Git internas REACTIVAS.

IMPORTANTE:
- NO hace fetch automático (Decisión #11)
- Reacciona cuando usuario hace cambios
- Usa GitPython, NUNCA comandos shell
- Solo detecta y notifica
"""

from acolyte.core.logging import logger
from acolyte.core.tracing import MetricsCollector
from acolyte.core.exceptions import ExternalServiceError
from acolyte.core.events import EventBus, CacheInvalidateEvent, event_bus as global_event_bus
from git import Repo  # type: ignore
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import time
from configparser import NoOptionError, NoSectionError


class GitService:
    """
    Operaciones Git internas REACTIVAS.

    IMPORTANTE:
    - NO hace fetch automático (Decisión #11)
    - Reacciona cuando usuario hace cambios
    - Usa GitPython, NUNCA comandos shell
    - Solo detecta y notifica
    """

    def __init__(self, repo_path: str, event_bus: Optional[EventBus] = None):
        self.metrics = MetricsCollector()
        self.repo_path = Path(repo_path).resolve()

        # Cache con TTL
        self._repo_cache = None
        self._repo_cache_time = None
        self._repo_cache_ttl = timedelta(minutes=5)  # TTL de 5 minutos

        self.event_bus = event_bus or global_event_bus  # Usar global si no se pasa

        # Validar que es un repo Git
        if not (self.repo_path / ".git").exists():
            raise ExternalServiceError(f"No Git repository found at {repo_path}")

        logger.info("GitService initialized", repo_path=str(self.repo_path))

    @property
    def repo(self) -> Repo:
        """
        Lazy loading del repo con cache y TTL.

        El cache expira después de 5 minutos para detectar cambios externos.
        """
        now = datetime.now()

        # Verificar si el cache es válido
        if (
            self._repo_cache
            and self._repo_cache_time
            and now - self._repo_cache_time < self._repo_cache_ttl
        ):
            return self._repo_cache

        # Cache expirado o no existe, recargar
        try:
            self._repo_cache = Repo(self.repo_path)
            self._repo_cache_time = now
            return self._repo_cache
        except Exception as e:
            logger.error("Failed to load Git repository", error=str(e))
            raise ExternalServiceError(f"Failed to load Git repository: {str(e)}") from e

    def invalidate_repo_cache(self):
        """
        Invalida manualmente el cache del repo.

        Útil cuando se sabe que hubo cambios externos.
        """
        self._repo_cache = None
        self._repo_cache_time = None

    async def detect_changes_from_others(self) -> List[Dict[str, Any]]:
        """
        Detecta cambios de otros desarrolladores.

        SE LLAMA: Después que el usuario hace pull/fetch
        NO hace fetch automático

        Raises:
            ExternalServiceError: Si no se puede acceder al repositorio Git

        Returns:
            Lista de cambios detectados con commit info
        """
        start_time = time.time()
        try:
            changes = []

            # Buscar commits recientes (últimas 24 horas) no del usuario actual
            since = datetime.now() - timedelta(days=1)

            # Obtener identidad del usuario de forma más robusta
            current_identities = set()

            try:
                config = self.repo.config_reader()

                # Intentar obtener email
                try:
                    email = config.get_value("user", "email")
                    if email:
                        current_identities.add(str(email).lower())
                except (NoOptionError, NoSectionError):
                    logger.warning("[UNTESTED PATH] No git user email configured")
                    pass

                # Intentar obtener nombre
                try:
                    name = config.get_value("user", "name")
                    if name:
                        current_identities.add(name)
                except (NoOptionError, NoSectionError):
                    logger.warning("[UNTESTED PATH] No git user name configured")
                    pass

            except Exception as e:
                logger.warning("Could not read git config", error=str(e))

            # Si no hay identidades, advertir pero continuar
            if not current_identities:
                logger.warning("Git user not configured. Showing all recent commits.")

            for commit in self.repo.iter_commits(since=since):
                # Skip si es del usuario actual
                if current_identities:
                    commit_email = commit.author.email.lower() if commit.author.email else ""
                    commit_name = commit.author.name

                    if commit_email in current_identities or commit_name in current_identities:
                        continue

                # Archivos modificados en este commit
                modified_files = list(commit.stats.files.keys())

                changes.append(
                    {
                        "commit": commit.hexsha[:8],
                        "author": commit.author.name,
                        "email": commit.author.email,
                        "message": commit.message.strip(),
                        "files": modified_files,
                        "timestamp": datetime.fromtimestamp(commit.committed_date),
                        "is_merge": len(commit.parents) > 1,
                    }
                )

            self.metrics.gauge("services.git_service.changes_detected", len(changes))

            # Publicar evento de invalidación si hay cambios y event_bus disponible
            if changes and self.event_bus:
                await self._publish_cache_invalidation(
                    reason="Changes from other developers detected",
                    files=[f for change in changes for f in change["files"]],
                )

            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record(
                "services.git_service.detect_changes_from_others_time_ms", elapsed_ms
            )
            return changes

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record(
                "services.git_service.detect_changes_from_others_time_ms", elapsed_ms
            )
            logger.error("Error detecting changes", error=str(e))
            raise ExternalServiceError(f"Failed to detect changes: {str(e)}") from e

    async def analyze_potential_conflicts(self, files_to_modify: List[str]) -> Dict[str, Any]:
        """
        Analiza conflictos potenciales.

        Args:
            files_to_modify: Lista de archivos a verificar

        Returns:
            Dict con:
            - files_with_conflicts: Lista de archivos
            - severity: 0-10
            - suggestions: Qué hacer

        Raises:
            ExternalServiceError: Si no se puede analizar el repositorio
        """
        start_time = time.time()
        try:
            conflicts = []
            total_severity = 0

            # Verificar estado del repo
            if self.repo.is_dirty(untracked_files=False):
                logger.warning("Repository has uncommitted changes")

            for file_path in files_to_modify:
                path = Path(file_path)
                if not path.is_absolute():
                    path = self.repo_path / path

                if not path.exists():
                    logger.warning("[UNTESTED PATH] File does not exist during conflict analysis")
                    continue

                # Verificar si el archivo tiene cambios sin commit
                relative_path = str(path.relative_to(self.repo_path))

                # Analizar el historial reciente del archivo
                try:
                    # Obtener últimos 5 commits que tocaron este archivo
                    commits = list(self.repo.iter_commits(paths=relative_path, max_count=5))

                    if len(commits) > 1:
                        # Verificar si hay múltiples autores recientes
                        authors = {c.author.email for c in commits[:3]}
                        if len(authors) > 1:
                            # Scale severity based on number of authors: 5 + (2 * number_of_authors)
                            # So: 2 authors = 7, 3 authors = 9, 5 authors = 10 (capped)
                            severity = min(10, 5 + (2 * len(authors)))
                            conflicts.append(
                                {
                                    "file": relative_path,
                                    "reason": "Multiple recent authors",
                                    "authors": list(authors),
                                    "severity": severity,
                                }
                            )
                            total_severity += severity

                    # Verificar si está en staging
                    if relative_path in [item.a_path for item in self.repo.index.diff("HEAD")]:
                        conflicts.append(
                            {
                                "file": relative_path,
                                "reason": "File has staged changes",
                                "severity": 5,
                            }
                        )
                        total_severity += 5

                except Exception as e:
                    logger.warning(
                        "[UNTESTED PATH] Could not analyze history",
                        file=relative_path,
                        error=str(e),
                    )
                    logger.warning("Could not analyze history", file=relative_path, error=str(e))

            # Calcular severidad promedio (0-10)
            avg_severity = min(10, total_severity / max(1, len(files_to_modify)))

            # Generar sugerencias
            suggestions = []
            if avg_severity > 7:
                suggestions.append("Consider coordinating with team before modifying")
                suggestions.append("Review recent changes with 'git log -p <file>'")
            elif avg_severity > 4:
                suggestions.append("Check for recent changes before proceeding")
                suggestions.append("Consider creating a feature branch")
            else:
                suggestions.append("Low conflict risk, proceed normally")

            result = {
                "files_with_conflicts": [c["file"] for c in conflicts],
                "severity": round(avg_severity, 1),
                "suggestions": suggestions,
                "details": conflicts,
            }

            self.metrics.gauge("services.git_service.conflict_severity", avg_severity)
            self.metrics.increment("services.git_service.conflicts_analyzed", len(conflicts))

            # Si hay conflictos significativos, invalidar cache
            if conflicts and avg_severity > 4 and self.event_bus:
                await self._publish_cache_invalidation(
                    reason=f"Potential conflicts detected (severity: {avg_severity})",
                    files=[c["file"] for c in conflicts],
                )

            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record("services.git_service.analyze_conflicts_time_ms", elapsed_ms)
            return result

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record("services.git_service.analyze_conflicts_time_ms", elapsed_ms)
            logger.error("Error analyzing conflicts", error=str(e))
            raise ExternalServiceError(f"Failed to analyze conflicts: {str(e)}") from e

    async def get_co_modification_patterns(
        self, file_path: str, days_back: int = 30
    ) -> List[Tuple[str, float]]:
        """
        Archivos que cambian junto a este.

        Usado por: Grafo neuronal

        Args:
            file_path: Archivo a analizar
            days_back: Días hacia atrás para analizar

        Returns:
            Lista de tuplas (archivo, frecuencia) ordenadas por frecuencia

        Raises:
            ExternalServiceError: Si falla el análisis de co-modificaciones
        """
        start_time = time.time()
        try:
            # Normalizar path
            target_path = Path(file_path)
            if not target_path.is_absolute():
                target_path = self.repo_path / target_path
            relative_path = str(target_path.relative_to(self.repo_path))

            # Recopilar commits que modificaron el archivo objetivo
            since = datetime.now() - timedelta(days=days_back)
            target_commits = set()

            for commit in self.repo.iter_commits(paths=relative_path, since=since):
                target_commits.add(commit.hexsha)

            if not target_commits:
                return []

            # Contar co-modificaciones
            co_modifications = {}

            for commit_sha in target_commits:
                commit = self.repo.commit(commit_sha)
                # Obtener todos los archivos modificados en este commit
                for file in commit.stats.files.keys():
                    if file != relative_path:  # Excluir el archivo objetivo
                        co_modifications[file] = co_modifications.get(file, 0) + 1

            # Calcular frecuencias (0-1)
            total_commits = len(target_commits)
            patterns = [(file, count / total_commits) for file, count in co_modifications.items()]

            # Ordenar por frecuencia descendente
            patterns.sort(key=lambda x: x[1], reverse=True)

            # Tomar top 10
            patterns = patterns[:10]

            self.metrics.gauge("services.git_service.co_modification_patterns", len(patterns))

            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record("services.git_service.co_modification_analysis_time_ms", elapsed_ms)
            return patterns

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record("services.git_service.co_modification_analysis_time_ms", elapsed_ms)
            logger.error("Error analyzing co-modifications", error=str(e))
            raise ExternalServiceError(f"Failed to analyze co-modifications: {str(e)}") from e

    async def notify_in_chat(self, notification_type: str, data: Dict[str, Any]) -> str:
        """
        Genera notificación para mostrar en chat.

        IMPORTANTE: Este método NO lanza excepciones. En caso de error,
        retorna string vacío. Esto es por diseño ya que las notificaciones
        son opcionales y no deben interrumpir el flujo principal.

        Ejemplo: "Veo que actualizaste auth.py. ¿Quieres que revise los cambios?"

        Args:
            notification_type: Tipo de notificación (file_updated, conflicts_detected, etc.)
            data: Datos específicos para la notificación

        Returns:
            Mensaje de notificación o string vacío si hay error
        """
        start_time = time.time()
        try:
            notifications = {
                "file_updated": self._notify_file_updated,
                "conflicts_detected": self._notify_conflicts,
                "branch_changed": self._notify_branch_change,
                "changes_from_others": self._notify_others_changes,
            }

            handler = notifications.get(notification_type)
            if not handler:
                logger.warning("Unknown notification type", type=notification_type)
                return ""

            message = handler(data)
            self.metrics.increment(f"services.git_service.notifications.{notification_type}")

            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record("services.git_service.generate_notification_time_ms", elapsed_ms)
            return message

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record("services.git_service.generate_notification_time_ms", elapsed_ms)
            logger.warning("[UNTESTED PATH] notify_in_chat error handling")
            logger.error("Error generating notification", error=str(e))
            # Para notificaciones, retornar string vacío es aceptable
            # ya que no son críticas para el flujo principal
            return ""

    def _notify_file_updated(self, data: Dict[str, Any]) -> str:
        """Notificación de archivo actualizado."""
        files = data.get("files", [])
        if not files:
            return ""

        if len(files) == 1:
            return f"Veo que actualizaste {files[0]}. ¿Quieres que revise los cambios?"
        else:
            return (
                f"Veo que actualizaste {len(files)} archivos "
                f"({', '.join(files[:3])}{', ...' if len(files) > 3 else ''}). "
                "¿Quieres que revise los cambios?"
            )

    def _notify_conflicts(self, data: Dict[str, Any]) -> str:
        """Notificación de conflictos potenciales."""
        severity = data.get("severity", 0)
        files = data.get("files_with_conflicts", [])

        if severity > 7:
            return (
                f"⚠️ Detecté posibles conflictos en {', '.join(files[:2])}. "
                "Otros desarrolladores han estado trabajando en estos archivos. "
                "¿Revisamos los cambios recientes primero?"
            )
        elif severity > 4:
            return (
                f"Hay algunos cambios recientes en {files[0]}. "
                "Podría ser buena idea revisar antes de modificar."
            )
        else:
            return ""

    def _notify_branch_change(self, data: Dict[str, Any]) -> str:
        """Notificación de cambio de branch."""
        old_branch = data.get("old_branch", "unknown")
        new_branch = data.get("new_branch", "unknown")

        return (
            f"Cambiaste de la rama '{old_branch}' a '{new_branch}'. "
            "¿Necesitas contexto sobre lo que estábamos haciendo aquí?"
        )

    def _notify_others_changes(self, data: Dict[str, Any]) -> str:
        """Notificación de cambios de otros."""
        changes = data.get("changes", [])
        if not changes:
            return ""

        authors = {c.get("author") for c in changes[:3]}
        files_count = sum(len(c.get("files", [])) for c in changes)

        return (
            f"Hay {len(changes)} commits nuevos de {', '.join(authors)} "
            f"que modificaron {files_count} archivos. "
            "¿Quieres que revise qué cambió?"
        )

    async def _publish_cache_invalidation(
        self,
        reason: str,
        files: Optional[List[str]] = None,
        target_services: Optional[List[str]] = None,
    ):
        """
        Publica evento de invalidación de cache.

        Args:
            reason: Razón de la invalidación
            files: Archivos afectados (opcional)
            target_services: Servicios específicos a invalidar (por defecto todos)
        """
        if not self.event_bus:
            return

        # Si no se especifican servicios, invalidar los principales
        if target_services is None:
            target_services = ["conversation", "indexing", "enrichment"]

        try:
            for service in target_services:
                # Crear patrón de key basado en archivos si están disponibles
                if files:
                    # Para cada archivo, invalidar cache relacionado
                    for file in files[:10]:  # Limitar a 10 archivos
                        event = CacheInvalidateEvent(
                            source="git_service",
                            target_service=service,
                            key_pattern=f"*{file}*",
                            reason=f"{reason}: {file}",
                        )
                        await self.event_bus.publish(event)
                else:
                    # Invalidación general
                    event = CacheInvalidateEvent(
                        source="git_service", target_service=service, key_pattern="*", reason=reason
                    )
                    await self.event_bus.publish(event)

            logger.info(
                "Published cache invalidation events",
                reason=reason,
                services=target_services,
                files_count=len(files) if files else 0,
            )

        except Exception as e:
            # No fallar si la publicación de eventos falla
            logger.warning("[UNTESTED PATH] Failed to publish cache invalidation event")
            logger.warning("Failed to publish cache invalidation", error=str(e))

    def get_most_recent_files(self, max_files: int = 10, days_back: int = 7) -> List[str]:
        """
        Returns a list of files modified in the last `days_back` days, up to `max_files` unique files.

        Args:
            max_files: Maximum number of unique files to return.
            days_back: Number of days to look back for modified files.

        Returns:
            List of file paths (str) modified in the given period, up to max_files.
        """
        start_time = time.time()
        try:
            since = datetime.now() - timedelta(days=days_back)
            recent_files = []
            seen = set()
            for commit in self.repo.iter_commits(since=since):
                for file in commit.stats.files.keys():
                    if file not in seen:
                        recent_files.append(file)
                        seen.add(file)
                    if len(recent_files) >= max_files:
                        elapsed_ms = (time.time() - start_time) * 1000
                        self.metrics.record(
                            "services.git_service.get_most_recent_files_time_ms", elapsed_ms
                        )
                        self.metrics.gauge(
                            "services.git_service.most_recent_files_count", len(recent_files)
                        )
                        return recent_files
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record("services.git_service.get_most_recent_files_time_ms", elapsed_ms)
            self.metrics.gauge("services.git_service.most_recent_files_count", len(recent_files))
            return recent_files
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record("services.git_service.get_most_recent_files_time_ms", elapsed_ms)
            logger.error("Failed to get most recent files", error=str(e))
            return []
