#!/usr/bin/env python3
"""
Port management for multi-project support
Automatically finds available ports in ACOLYTE range
"""

import socket
from typing import Tuple, Optional


class PortManager:
    """Manages port allocation for ACOLYTE projects"""

    # ACOLYTE port ranges
    WEAVIATE_BASE = 42080
    OLLAMA_BASE = 42434
    BACKEND_BASE = 42000

    # Maximum offset to try
    MAX_OFFSET = 100

    @staticmethod
    def is_port_available(port: int) -> bool:
        """Check if a port is available"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return True
        except OSError:
            return False

    @classmethod
    def find_next_available(cls, base_port: int, max_attempts: int = 100) -> Optional[int]:
        """
        Find next available port starting from base_port

        Args:
            base_port: Starting port number
            max_attempts: Maximum ports to try

        Returns:
            Available port number or None if all are taken
        """
        for offset in range(max_attempts):
            port = base_port + offset
            if port > 65535:  # Max valid port
                return None

            if cls.is_port_available(port):
                return port

        return None

    @classmethod
    def find_available_ports(cls) -> Tuple[int, int, int]:
        """
        Find available ports for all ACOLYTE services

        Returns:
            Tuple of (weaviate_port, ollama_port, backend_port)

        Raises:
            RuntimeError: If cannot find available ports
        """
        # Find Weaviate port
        weaviate_port = cls.find_next_available(cls.WEAVIATE_BASE)
        if not weaviate_port:
            raise RuntimeError(f"Cannot find available port starting from {cls.WEAVIATE_BASE}")

        # Find Ollama port
        ollama_port = cls.find_next_available(cls.OLLAMA_BASE)
        if not ollama_port:
            raise RuntimeError(f"Cannot find available port starting from {cls.OLLAMA_BASE}")

        # Find Backend port
        backend_port = cls.find_next_available(cls.BACKEND_BASE)
        if not backend_port:
            raise RuntimeError(f"Cannot find available port starting from {cls.BACKEND_BASE}")

        return weaviate_port, ollama_port, backend_port

    @classmethod
    def suggest_ports(cls, preferred_ports: dict) -> dict:
        """
        Suggest available ports based on preferences

        Args:
            preferred_ports: Dict with preferred ports

        Returns:
            Dict with available ports (same or next available)
        """
        result = {}

        # Check Weaviate
        pref_weaviate = preferred_ports.get("weaviate", cls.WEAVIATE_BASE)
        if cls.is_port_available(pref_weaviate):
            result["weaviate"] = pref_weaviate
        else:
            result["weaviate"] = cls.find_next_available(pref_weaviate) or pref_weaviate

        # Check Ollama
        pref_ollama = preferred_ports.get("ollama", cls.OLLAMA_BASE)
        if cls.is_port_available(pref_ollama):
            result["ollama"] = pref_ollama
        else:
            result["ollama"] = cls.find_next_available(pref_ollama) or pref_ollama

        # Check Backend
        pref_backend = preferred_ports.get("backend", cls.BACKEND_BASE)
        if cls.is_port_available(pref_backend):
            result["backend"] = pref_backend
        else:
            result["backend"] = cls.find_next_available(pref_backend) or pref_backend

        return result
