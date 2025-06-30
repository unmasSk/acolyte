#!/usr/bin/env python3
"""
Script de inicialización de base de datos para ACOLYTE.

Este script:
1. Inicializa la base de datos SQLite con el esquema completo
2. Inicializa Weaviate con las collections necesarias
3. Verifica la integridad de ambas bases de datos

Uso:
    python -m acolyte.core.database.init_database

O directamente:
    python src/acolyte/core/database/init_database.py
"""

import sys
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional

# Añadir el directorio src al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from weaviate import Client
    from weaviate.exceptions import UnexpectedStatusCodeException
except ImportError:
    print("ERROR: Weaviate client no instalado. Ejecuta: pip install weaviate-client")
    sys.exit(1)

# Verificar que los archivos existen antes de importar
try:
    from acolyte.core.secure_config import Settings
    from acolyte.core.logging import AsyncLogger
except ImportError as e:
    print(f"ERROR: No se pueden importar módulos necesarios: {e}")
    print("Asegúrate de estar en el directorio correcto del proyecto.")
    sys.exit(1)

# Crear logger para este módulo
logger = AsyncLogger("database.init")


class DatabaseInitializer:
    """Inicializa tanto SQLite como Weaviate para ACOLYTE."""

    def __init__(self, config: Optional[Settings] = None):
        """
        Inicializa el instalador de base de datos.

        Args:
            config: Configuración opcional. Si no se proporciona, usa la default.
        """
        self.config = config or Settings()
        # DB_PATH debe estar en "./data/acolyte.db"
        self.project_root = Path(".")
        self.db_path = self.project_root / "data" / "acolyte.db"
        self.schemas_path = (
            Path(__file__).parent.parent / "src" / "acolyte" / "core" / "database" / "schemas.sql"
        )
        self.weaviate_schemas_path = (
            Path(__file__).parent.parent
            / "src"
            / "acolyte"
            / "rag"
            / "collections"
            / "schemas.json"
        )

        # Asegurar que el directorio data existe
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def init_sqlite(self) -> bool:
        """
        Inicializa la base de datos SQLite con el esquema completo.

        Returns:
            True si la inicialización fue exitosa
        """
        try:
            logger.info(f"Inicializando SQLite en: {self.db_path}")

            # Leer el esquema SQL
            if not self.schemas_path.exists():
                logger.error(f"No se encuentra el archivo de esquemas: {self.schemas_path}")
                return False

            with open(self.schemas_path, "r", encoding="utf-8") as f:
                schema_sql = f.read()

            # Conectar y ejecutar el esquema
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Habilitar foreign keys
            cursor.execute("PRAGMA foreign_keys = ON;")

            # Ejecutar el esquema completo
            cursor.executescript(schema_sql)

            # Insertar fila inicial en tablas singleton
            cursor.execute(
                """
                INSERT OR IGNORE INTO dream_state (id, fatigue_level, optimization_count)
                VALUES (1, 0.0, 0)
            """
            )

            cursor.execute(
                """
                INSERT OR IGNORE INTO code_graph_metrics (id, total_nodes, total_edges)
                VALUES (1, 0, 0)
            """
            )

            conn.commit()

            # Verificar que las tablas se crearon
            cursor.execute(
                """
                SELECT name FROM sqlite_master 
                WHERE type='table' 
                ORDER BY name;
            """
            )
            tables = [row[0] for row in cursor.fetchall()]

            expected_tables = [
                "code_graph_edges",
                "code_graph_metrics",
                "code_graph_nodes",
                "conversations",
                "dream_insights",
                "dream_state",
                "task_sessions",
                "tasks",
                "technical_decisions",
            ]

            missing_tables = set(expected_tables) - set(tables)
            if missing_tables:
                logger.error(f"Faltan tablas: {missing_tables}")
                return False

            logger.info(f"✅ SQLite inicializado correctamente con {len(tables)} tablas")
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Error inicializando SQLite: {e}")
            return False

    def init_weaviate(self) -> bool:
        """
        Inicializa Weaviate con las collections necesarias.

        Returns:
            True si la inicialización fue exitosa
        """
        try:
            logger.info("Inicializando Weaviate...")

            # Leer esquemas de Weaviate
            if not self.weaviate_schemas_path.exists():
                logger.error(
                    f"No se encuentra el archivo de schemas de Weaviate: {self.weaviate_schemas_path}"
                )
                return False

            with open(self.weaviate_schemas_path, "r", encoding="utf-8") as f:
                weaviate_config = json.load(f)

            # Conectar a Weaviate usando el puerto de la configuración
            ports = self.config.get("ports", {})
            weaviate_port = ports.get("weaviate", 8080)
            weaviate_url = f"http://localhost:{weaviate_port}"
            client = Client(url=weaviate_url)

            # Verificar conexión
            if not client.is_ready():
                logger.error("Weaviate no está disponible. ¿Está ejecutándose el contenedor?")
                return False

            # Obtener collections existentes
            try:
                existing_schema = client.schema.get()
                existing_classes = {cls["class"] for cls in existing_schema.get("classes", [])}
            except Exception:
                existing_classes = set()

            # Crear cada collection si no existe
            collections_created = 0
            collections_skipped = 0

            for collection_config in weaviate_config["collections"]:
                class_name = collection_config["class"]

                if class_name in existing_classes:
                    logger.info(f"Collection '{class_name}' ya existe, saltando...")
                    collections_skipped += 1
                    continue

                try:
                    # Añadir configuración de índice si está presente
                    if "indexing_config" in weaviate_config:
                        collection_config["vectorIndexType"] = weaviate_config[
                            "indexing_config"
                        ].get("vectorIndexType", "hnsw")
                        collection_config["vectorIndexConfig"] = weaviate_config[
                            "indexing_config"
                        ].get("vectorIndexConfig", {})

                    # Añadir configuración de replicación si está presente
                    if "replication_config" in weaviate_config:
                        collection_config["replicationConfig"] = weaviate_config[
                            "replication_config"
                        ]

                    # Añadir configuración de sharding si está presente
                    if "sharding_config" in weaviate_config:
                        collection_config["shardingConfig"] = weaviate_config["sharding_config"]

                    client.schema.create_class(collection_config)
                    logger.info(f"✅ Collection '{class_name}' creada exitosamente")
                    collections_created += 1

                except UnexpectedStatusCodeException as e:
                    logger.error(f"Error creando collection '{class_name}': {e}")
                    return False

            logger.info(
                f"✅ Weaviate inicializado: {collections_created} collections creadas, "
                f"{collections_skipped} ya existían"
            )
            return True

        except Exception as e:
            logger.error(f"Error inicializando Weaviate: {e}")
            return False

    def verify_installation(self) -> Dict[str, Any]:
        """
        Verifica que ambas bases de datos estén correctamente instaladas.

        Returns:
            Diccionario con el estado de la verificación
        """
        status: dict[str, dict[str, object]] = {
            "sqlite": {"ok": False, "tables": 0, "error": None},
            "weaviate": {"ok": False, "collections": 0, "error": None},
        }

        # Verificar SQLite
        try:
            if self.db_path.exists():
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                table_count = cursor.fetchone()[0]
                conn.close()

                status["sqlite"]["ok"] = table_count >= 9  # Esperamos al menos 9 tablas
                status["sqlite"]["tables"] = table_count
            else:
                status["sqlite"]["error"] = "Archivo de base de datos no existe"
        except Exception as e:
            status["sqlite"]["error"] = str(e)

        # Verificar Weaviate
        try:
            ports = self.config.get("ports", {})
            weaviate_port = ports.get("weaviate", 8080)
            weaviate_url = f"http://localhost:{weaviate_port}"
            client = Client(url=weaviate_url)

            if client.is_ready():
                schema = client.schema.get()
                collection_count = len(schema.get("classes", []))
                status["weaviate"]["ok"] = collection_count >= 5  # Esperamos al menos 5 collections
                status["weaviate"]["collections"] = collection_count
            else:
                status["weaviate"]["error"] = "Weaviate no está disponible"
        except Exception as e:
            status["weaviate"]["error"] = str(e)

        return status

    def run(self) -> bool:
        """
        Ejecuta la inicialización completa de las bases de datos.

        Returns:
            True si ambas inicializaciones fueron exitosas
        """
        logger.info("=== Iniciando instalación de bases de datos ACOLYTE ===")

        # Inicializar SQLite
        sqlite_ok = self.init_sqlite()
        if not sqlite_ok:
            logger.error("❌ Fallo en la inicialización de SQLite")
            return False

        # Inicializar Weaviate
        weaviate_ok = self.init_weaviate()
        if not weaviate_ok:
            logger.error("❌ Fallo en la inicialización de Weaviate")
            logger.warning(
                "Asegúrate de que Weaviate esté ejecutándose: docker-compose up -d weaviate"
            )
            return False

        # Verificar instalación
        logger.info("\n=== Verificando instalación ===")
        status = self.verify_installation()

        logger.info(
            f"SQLite: {'✅' if status['sqlite']['ok'] else '❌'} "
            f"({status['sqlite']['tables']} tablas)"
        )
        if status["sqlite"]["error"]:
            logger.error(f"  Error: {status['sqlite']['error']}")

        logger.info(
            f"Weaviate: {'✅' if status['weaviate']['ok'] else '❌'} "
            f"({status['weaviate']['collections']} collections)"
        )
        if status["weaviate"]["error"]:
            logger.error(f"  Error: {status['weaviate']['error']}")

        success = status["sqlite"]["ok"] and status["weaviate"]["ok"]

        if success:
            logger.info("\n✅ ¡Bases de datos inicializadas correctamente!")
            logger.info(f"SQLite: {self.db_path}")
            ports = self.config.get("ports", {})
            weaviate_port = ports.get("weaviate", 8080)
            logger.info(f"Weaviate: http://localhost:{weaviate_port}")
        else:
            logger.error("\n❌ La inicialización no se completó correctamente")

        return success


def main():
    """Función principal para ejecutar el script."""
    # Parsear argumentos si necesitamos en el futuro
    # Por ahora, simplemente ejecutar

    initializer = DatabaseInitializer()
    success = initializer.run()

    # Exit code basado en el éxito
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
