"""
ACOLYTE Database Initialization Module

Provides functions for initializing SQLite and Weaviate databases.
Converted from scripts/init_database.py to be importable.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional

from acolyte.core.logging import logger
from acolyte.install.common import print_error, print_info, print_success, show_spinner

# Try to import weaviate, but make it optional
try:
    from weaviate import Client
    from weaviate.exceptions import UnexpectedStatusCodeException

    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False
    Client = None
    UnexpectedStatusCodeException = None


class DatabaseInitializer:
    """Initialize both SQLite and Weaviate for ACOLYTE"""

    def __init__(self, project_path: Path, project_id: str, global_dir: Optional[Path] = None):
        """
        Initialize the database installer.

        Args:
            project_path: Path to the user's project
            project_id: Unique project ID
            global_dir: Global ACOLYTE directory (default: ~/.acolyte)
        """
        self.project_path = project_path.resolve()
        self.project_id = project_id
        self.global_dir = global_dir or Path.home() / ".acolyte"

        # Database paths
        self.project_data_dir = self.global_dir / "projects" / project_id / "data"
        self.db_path = self.project_data_dir / "acolyte.db"
        # Use importlib.resources for packaged files
        try:
            import importlib.resources as resources

            # For schemas.sql
            if hasattr(resources, 'files'):
                core_files = resources.files('acolyte.core.database_schemas')
                self.schemas_path = Path(str(core_files / 'schemas.sql'))
                rag_files = resources.files('acolyte.rag.collections')
                self.weaviate_schemas_path = Path(str(rag_files / 'schemas.json'))
            else:
                # Fallback for older Python
                with resources.path('acolyte.core.database_schemas', 'schemas.sql') as p:
                    self.schemas_path = Path(p)
                with resources.path('acolyte.rag.collections', 'schemas.json') as p:
                    self.weaviate_schemas_path = Path(p)
        except Exception:
            # Improved fallback for both development and pip installations
            import site

            schema_found = False
            weaviate_schema_found = False

            # Search in multiple possible locations
            search_paths = [
                Path(__file__).parent.parent,  # Development path
                *[Path(p) for p in site.getsitepackages()],  # System site-packages
                (
                    Path(site.getusersitepackages()) if site.getusersitepackages() else None
                ),  # User site-packages
            ]

            for base_path in filter(None, search_paths):
                if not schema_found:
                    potential_schema = (
                        base_path / "acolyte" / "core" / "database_schemas" / "schemas.sql"
                    )
                    if potential_schema.exists():
                        self.schemas_path = potential_schema
                        schema_found = True

                if not weaviate_schema_found:
                    potential_weaviate = (
                        base_path / "acolyte" / "rag" / "collections" / "schemas.json"
                    )
                    if potential_weaviate.exists():
                        self.weaviate_schemas_path = potential_weaviate
                        weaviate_schema_found = True

                if schema_found and weaviate_schema_found:
                    break

            # Final fallback if nothing found
            if not schema_found:
                self.schemas_path = (
                    Path(__file__).parent.parent / "core" / "database_schemas" / "schemas.sql"
                )
            if not weaviate_schema_found:
                self.weaviate_schemas_path = (
                    Path(__file__).parent.parent / "rag" / "collections" / "schemas.json"
                )

    def ensure_directories(self) -> bool:
        """Ensure all necessary directories exist"""
        try:
            self.project_data_dir.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error("Failed to create data directory", error=str(e))
            return False

    def init_sqlite(self) -> bool:
        """
        Initialize the SQLite database with the complete schema.

        Returns:
            True if initialization was successful
        """
        try:
            show_spinner("Initializing SQLite database...", 1.0)

            # Ensure directories exist
            if not self.ensure_directories():
                return False

            # Check if schemas file exists
            if not self.schemas_path.exists():
                logger.error(f"Schema file not found at: {self.schemas_path}")
                print_error("Schema file not found. Please ensure schemas.sql exists.")
                return False

            # Read schema SQL
            with open(self.schemas_path, "r", encoding="utf-8") as f:
                schema_sql = f.read()

            # Connect and execute schema
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Enable foreign keys
            cursor.execute("PRAGMA foreign_keys = ON;")

            # Execute the complete schema
            cursor.executescript(schema_sql)

            # Insert initial rows for singleton tables
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

            # Verify tables were created
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
                logger.error(f"Missing tables: {missing_tables}")
                print_error(f"Some tables were not created: {missing_tables}")
                return False

            logger.info(f"SQLite initialized with {len(tables)} tables")
            print_success(f"✅ SQLite database created with {len(tables)} tables")
            print_info(f"   Database location: {self.db_path}")

            conn.close()
            return True

        except Exception as e:
            logger.error("Error initializing SQLite", error=str(e))
            print_error(f"Failed to initialize SQLite: {e}")
            return False

    def load_weaviate_port(self) -> int:
        """Load Weaviate port from configuration"""
        try:
            config_file = self.global_dir / "projects" / self.project_id / ".acolyte"
            if config_file.exists():
                with open(config_file, "r") as f:
                    import yaml

                    config = yaml.safe_load(f)
                    return config.get("ports", {}).get("weaviate", 8080)
        except Exception:
            pass
        return 8080

    def init_weaviate(self) -> bool:
        """
        Initialize Weaviate with the necessary collections.

        Returns:
            True if initialization was successful
        """
        if not WEAVIATE_AVAILABLE:
            print_error("❌ Weaviate client not installed!")
            print_info("Install with: pip install weaviate-client")
            return False

        try:
            show_spinner("Initializing Weaviate collections...", 1.0)

            # Check if schemas file exists
            if not self.weaviate_schemas_path.exists():
                logger.error(f"Weaviate schema file not found at: {self.weaviate_schemas_path}")
                print_error("Weaviate schema file not found.")
                return False

            # Read Weaviate schemas
            with open(self.weaviate_schemas_path, "r", encoding="utf-8") as f:
                weaviate_config = json.load(f)

            # Connect to Weaviate
            weaviate_port = self.load_weaviate_port()
            weaviate_url = f"http://localhost:{weaviate_port}"
            client = Client(url=weaviate_url)  # type: ignore

            # Check connection
            if not client.is_ready():
                logger.error("Weaviate not available")
                print_error("❌ Weaviate is not running!")
                print_info("Start Weaviate with: docker-compose up -d weaviate")
                return False

            # Get existing collections
            try:
                existing_schema = client.schema.get()
                existing_classes = {cls["class"] for cls in existing_schema.get("classes", [])}
            except Exception:
                existing_classes = set()

            # Create each collection if it doesn't exist
            collections_created = 0
            collections_skipped = 0

            for collection_config in weaviate_config["collections"]:
                class_name = collection_config["class"]

                if class_name in existing_classes:
                    logger.info(f"Collection '{class_name}' already exists")
                    collections_skipped += 1
                    continue

                try:
                    # Add indexing config if present
                    if "indexing_config" in weaviate_config:
                        collection_config["vectorIndexType"] = weaviate_config[
                            "indexing_config"
                        ].get("vectorIndexType", "hnsw")
                        collection_config["vectorIndexConfig"] = weaviate_config[
                            "indexing_config"
                        ].get("vectorIndexConfig", {})

                    # Add replication config if present
                    if "replication_config" in weaviate_config:
                        collection_config["replicationConfig"] = weaviate_config[
                            "replication_config"
                        ]

                    # Add sharding config if present
                    if "sharding_config" in weaviate_config:
                        collection_config["shardingConfig"] = weaviate_config["sharding_config"]

                    client.schema.create_class(collection_config)
                    logger.info(f"Created collection '{class_name}'")
                    collections_created += 1

                except UnexpectedStatusCodeException as e:  # type: ignore
                    logger.error(f"Error creating collection '{class_name}'", error=str(e))
                    print_error(f"Failed to create collection '{class_name}': {e}")
                    return False

            print_success(
                f"✅ Weaviate initialized: {collections_created} new, {collections_skipped} existing"
            )
            print_info(f"   Weaviate URL: {weaviate_url}")

            return True

        except Exception as e:
            logger.error("Error initializing Weaviate", error=str(e))
            print_error(f"Failed to initialize Weaviate: {e}")
            return False

    def verify_installation(self) -> Dict[str, Any]:
        """
        Verify that both databases are correctly installed.

        Returns:
            Dictionary with verification status
        """
        status = {
            "sqlite": {"ok": False, "tables": 0, "error": None},
            "weaviate": {"ok": False, "collections": 0, "error": None},
        }

        # Verify SQLite
        try:
            if self.db_path.exists():
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                table_count = cursor.fetchone()[0]
                conn.close()

                status["sqlite"]["ok"] = table_count >= 9  # Expect at least 9 tables
                status["sqlite"]["tables"] = table_count
            else:
                status["sqlite"]["error"] = "Database file does not exist"
        except Exception as e:
            status["sqlite"]["error"] = str(e)

        # Verify Weaviate
        if WEAVIATE_AVAILABLE and Client is not None:
            try:
                weaviate_port = self.load_weaviate_port()
                weaviate_url = f"http://localhost:{weaviate_port}"
                client = Client(url=weaviate_url)  # type: ignore

                if client.is_ready():
                    schema = client.schema.get()
                    collection_count = len(schema.get("classes", []))
                    status["weaviate"]["ok"] = (
                        collection_count >= 5
                    )  # Expect at least 5 collections
                    status["weaviate"]["collections"] = collection_count
                else:
                    status["weaviate"]["error"] = "Weaviate is not available"
            except Exception as e:
                status["weaviate"]["error"] = str(e)
        else:
            status["weaviate"]["error"] = "Weaviate client not installed"

        return status

    def run(self, skip_weaviate: bool = False) -> bool:
        """
        Run the complete database initialization.

        Args:
            skip_weaviate: Skip Weaviate initialization (for testing)

        Returns:
            True if all initializations were successful
        """
        print_info("🗄️ Initializing ACOLYTE databases...")

        # Initialize SQLite
        sqlite_ok = self.init_sqlite()
        if not sqlite_ok:
            logger.error("SQLite initialization failed")
            return False

        # Initialize Weaviate (unless skipped)
        if not skip_weaviate:
            weaviate_ok = self.init_weaviate()
            if not weaviate_ok:
                logger.warning("Weaviate initialization failed - continuing anyway")
                print_info("⚠️ Weaviate initialization failed but SQLite is ready")
                print_info("You can initialize Weaviate later when Docker is running")
                return True  # Still return True as SQLite is the critical component
        else:
            print_info("Skipping Weaviate initialization")

        # Verify installation
        print_info("\n📊 Verifying installation...")
        status = self.verify_installation()

        if status["sqlite"]["ok"]:
            print_success(f"✅ SQLite: {status['sqlite']['tables']} tables")
        else:
            print_error(f"❌ SQLite: {status['sqlite']['error']}")

        if not skip_weaviate:
            if status["weaviate"]["ok"]:
                print_success(f"✅ Weaviate: {status['weaviate']['collections']} collections")
            else:
                print_error(f"❌ Weaviate: {status['weaviate']['error']}")

        success = status["sqlite"]["ok"] and (skip_weaviate or status["weaviate"]["ok"])

        if success:
            print_success("\n✅ Database initialization complete!")
        else:
            print_error("\n❌ Database initialization incomplete")

        return success


def initialize_databases(
    project_path: Path,
    project_id: str,
    global_dir: Optional[Path] = None,
    skip_weaviate: bool = False,
) -> bool:
    """
    Initialize databases for an ACOLYTE project.

    This is the main entry point for other modules to use.

    Args:
        project_path: Path to the project
        project_id: Unique project ID
        global_dir: Global ACOLYTE directory
        skip_weaviate: Skip Weaviate initialization

    Returns:
        True if successful
    """
    initializer = DatabaseInitializer(project_path, project_id, global_dir)
    return initializer.run(skip_weaviate)
