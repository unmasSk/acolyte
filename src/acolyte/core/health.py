"""
Health checker for ACOLYTE services
"""

import time
import requests
from typing import Dict, Any
from acolyte.core.logging import logger


class ServiceHealthChecker:
    """Health checker for ACOLYTE services"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.timeout = 120  # 2 minutos máximo

    def _check_service_once(self, service_name: str, port: int, endpoint: str) -> bool:
        """Check if a service is ready (single attempt)"""
        url = f"http://localhost:{port}{endpoint}"
        try:
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _wait_for_service(self, service_name: str, port: int, endpoint: str) -> bool:
        """Generic method to wait for a service to be ready"""
        url = f"http://localhost:{port}{endpoint}"

        logger.info(f"Waiting for {service_name} to be ready...")
        for attempt in range(self.timeout):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"✓ {service_name} is ready!")
                    return True
            except requests.RequestException:
                pass

            if attempt % 10 == 0:  # Show progress every 10 seconds
                logger.info(f"  Attempt {attempt + 1}/{self.timeout}...")
            time.sleep(1)

        logger.error(f"✗ {service_name} failed to start within timeout")
        return False

    def wait_for_backend(self) -> bool:
        """Wait until backend is ready"""
        backend_port = self.config['ports']['backend']
        return self._wait_for_service("Backend", backend_port, "/api/health")

    def wait_for_weaviate(self) -> bool:
        """Wait until Weaviate is ready"""
        weaviate_port = self.config['ports']['weaviate']
        return self._wait_for_service("Weaviate", weaviate_port, "/v1/.well-known/ready")

    def wait_for_ollama(self) -> bool:
        """Wait until Ollama is ready"""
        ollama_port = self.config['ports']['ollama']
        url = f"http://localhost:{ollama_port}/api/tags"

        print("Waiting for Ollama to be ready...")
        for attempt in range(self.timeout):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print("✓ Ollama is ready!")
                    return True
            except requests.RequestException:
                pass

            if attempt % 10 == 0:  # Mostrar progreso cada 10 segundos
                print(f"  Attempt {attempt + 1}/{self.timeout}...")
            time.sleep(1)

        print("✗ Ollama failed to start within timeout")
        return False

    def check_all_services(self) -> Dict[str, bool]:
        """Verifica el estado de todos los servicios"""
        results = {}

        results['weaviate'] = self.wait_for_weaviate()
        results['ollama'] = self.wait_for_ollama()
        results['backend'] = self.wait_for_backend()

        return results
