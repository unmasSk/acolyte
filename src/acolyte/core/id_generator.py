"""
Generador centralizado de IDs para ACOLYTE.
Resuelve inconsistencias entre formatos UUID4 y hex32.
"""

import secrets
import uuid
from typing import Literal, Optional


IDFormat = Literal["hex32", "uuid4"]


class IDGenerator:
    """
    Generador centralizado de IDs para todo ACOLYTE.

    Características:
    - Formato único consistente en todo el sistema
    - Conversión automática entre formatos
    - Compatible con SQLite (hex32) y estándares (UUID4)
    - Validación automática de formato
    """

    # Formato por defecto para ACOLYTE (compatible con SQLite)
    DEFAULT_FORMAT: IDFormat = "hex32"

    @staticmethod
    def generate(format: Optional[IDFormat] = None) -> str:
        """
        Genera ID en el formato especificado.

        Args:
            format: Formato deseado ("hex32" o "uuid4")

        Returns:
            ID generado en el formato solicitado

        Examples:
            >>> IDGenerator.generate("hex32")
            'a3f4b2c1d5e6f7a8b9c0d1e2f3a4b5c6'

            >>> IDGenerator.generate("uuid4")
            '550e8400-e29b-41d4-a716-446655440000'
        """
        if format is None:
            format = IDGenerator.DEFAULT_FORMAT

        if format == "hex32":
            return secrets.token_hex(16)
        elif format == "uuid4":
            return str(uuid.uuid4())
        else:
            raise ValueError(f"Formato no soportado: {format}")

    @staticmethod
    def to_db_format(id_str: str) -> str:
        """
        Convierte cualquier ID a formato compatible con SQLite.

        Args:
            id_str: ID en cualquier formato

        Returns:
            ID en formato hex32 (sin guiones)

        Examples:
            >>> IDGenerator.to_db_format("550e8400-e29b-41d4-a716-446655440000")
            '550e8400e29b41d4a716446655440000'

            >>> IDGenerator.to_db_format("a3f4b2c1d5e6f7a8b9c0d1e2f3a4b5c6")
            'a3f4b2c1d5e6f7a8b9c0d1e2f3a4b5c6'
        """
        if not id_str:
            raise ValueError("ID no puede estar vacío")

        # Si ya es hex32, devolverlo tal como está
        clean_id = id_str.replace("-", "").lower()

        # Validar que sea hex válido de 32 caracteres
        if len(clean_id) != 32:
            raise ValueError(f"ID debe tener 32 caracteres hex, recibido: {len(clean_id)}")

        try:
            int(clean_id, 16)  # Verificar que sea hex válido
        except ValueError:
            raise ValueError(f"ID contiene caracteres no-hex: {clean_id}")

        return clean_id

    @staticmethod
    def to_display_format(hex_str: str, target: IDFormat = "uuid4") -> str:
        """
        Convierte hex32 a formato legible para humanos.

        Args:
            hex_str: ID en formato hex32
            target: Formato objetivo ("uuid4")

        Returns:
            ID formateado para display

        Examples:
            >>> IDGenerator.to_display_format("550e8400e29b41d4a716446655440000")
            '550e8400-e29b-41d4-a716-446655440000'
        """
        if not hex_str:
            raise ValueError("ID hex no puede estar vacío")

        # Si ya tiene guiones, devolverlo tal como está
        if "-" in hex_str:
            return hex_str

        # Validar longitud
        if len(hex_str) != 32:
            raise ValueError(f"ID hex debe tener 32 caracteres, recibido: {len(hex_str)}")

        if target == "uuid4":
            # Formatear como UUID estándar
            return f"{hex_str[:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:]}"
        else:
            return hex_str

    @staticmethod
    def is_valid_id(id_str: str, format: Optional[IDFormat] = None) -> bool:
        """
        Valida si un string es un ID válido.

        Args:
            id_str: String a validar
            format: Formato esperado (None para auto-detectar)

        Returns:
            True si es válido, False en caso contrario
        """
        if not id_str:
            return False

        try:
            if format == "hex32":
                return len(id_str.replace("-", "")) == 32 and all(
                    c in "0123456789abcdef" for c in id_str.replace("-", "").lower()
                )
            elif format == "uuid4":
                uuid.UUID(id_str)  # Lanza excepción si no es válido
                return True
            else:
                # Auto-detectar formato
                if "-" in id_str:
                    uuid.UUID(id_str)
                    return True
                else:
                    return len(id_str) == 32 and all(
                        c in "0123456789abcdef" for c in id_str.lower()
                    )
        except (ValueError, TypeError):
            return False

    @staticmethod
    def detect_format(id_str: str) -> Optional[IDFormat]:
        """
        Detecta automáticamente el formato de un ID.

        Args:
            id_str: ID a analizar

        Returns:
            Formato detectado o None si no es válido
        """
        if not id_str:
            return None

        if "-" in id_str:
            return "uuid4" if IDGenerator.is_valid_id(id_str, "uuid4") else None
        else:
            return "hex32" if IDGenerator.is_valid_id(id_str, "hex32") else None


def generate_id(format: Optional[IDFormat] = None) -> str:
    """
    Función de conveniencia para generar IDs.
    Alias para IDGenerator.generate().
    """
    return IDGenerator.generate(format)


def is_valid_id(id_str: str) -> bool:
    """
    Función de conveniencia para validar IDs.
    Alias para IDGenerator.is_valid_id().
    """
    return IDGenerator.is_valid_id(id_str)
