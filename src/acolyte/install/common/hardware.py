#!/usr/bin/env python3
"""
Hardware detection utilities for ACOLYTE
"""

import platform
import subprocess
from typing import Dict, Optional, Tuple

import psutil
from acolyte.core.logging import logger


class SystemDetector:
    """Detects system characteristics with error handling"""

    @staticmethod
    def detect_os() -> Tuple[str, str]:
        """Detect operating system"""
        try:
            logger.info("Detecting operating system...")
            system = platform.system().lower()
            version = platform.version()

            if system == "linux":
                # Try to detect distribution
                try:
                    with open("/etc/os-release") as f:
                        lines = f.readlines()
                        for line in lines:
                            if line.startswith("PRETTY_NAME="):
                                version = line.split("=")[1].strip().strip('"')
                                break
                except Exception as e:
                    logger.warning(f"Could not read /etc/os-release: {e}")

            logger.info(f"Detected OS: {system} {version}")
            return system, version
        except Exception as e:
            logger.error(f"Error detecting OS: {e}")
            return "unknown", "unknown"

    @staticmethod
    def detect_cpu() -> Dict:
        """Detect CPU information"""
        try:
            logger.info("Detecting CPU information...")
            cpu_info = {
                "cores": psutil.cpu_count(logical=False) or 1,
                "threads": psutil.cpu_count(logical=True) or 1,
                "model": platform.processor() or "Unknown CPU",
            }
            logger.info(
                f"Detected CPU: {cpu_info['model']} with "
                f"{cpu_info['cores']} cores, {cpu_info['threads']} threads"
            )
            return cpu_info
        except Exception as e:
            logger.error(f"Error detecting CPU: {e}")
            return {"cores": 1, "threads": 1, "model": "Unknown"}

    @staticmethod
    def detect_memory() -> int:
        """Detect available RAM in GB"""
        try:
            logger.info("Detecting system memory...")
            mem = psutil.virtual_memory()
            ram_gb = round(mem.total / (1024**3))
            logger.info(f"Detected RAM: {ram_gb} GB")
            return ram_gb
        except Exception as e:
            logger.error(f"Error detecting memory: {e}")
            return 8  # Assume 8GB by default

    @staticmethod
    def detect_gpu() -> Optional[Dict]:
        """Detect available GPU"""
        gpu_info = None
        logger.info("Detecting GPU...")

        # Try NVIDIA
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                output = result.stdout.strip()
                if output:
                    # Validate output format before parsing
                    parts = output.split(", ")
                    if len(parts) != 2:
                        logger.debug(
                            f"Unexpected nvidia-smi output format: expected 2 parts, got {len(parts)}. Output: {output}"
                        )
                    else:
                        try:
                            # Parse GPU name (first part)
                            gpu_name = parts[0].strip()
                            if not gpu_name:
                                logger.debug("Empty GPU name in nvidia-smi output")
                                raise ValueError("Empty GPU name")

                            # Parse VRAM size (second part)
                            memory_str = parts[1].strip()
                            # Handle different memory formats: "8192 MiB", "8192 MB", etc.
                            memory_str = (
                                memory_str.replace(" MiB", "")
                                .replace(" MB", "")
                                .replace(" GiB", "")
                                .replace(" GB", "")
                            )

                            try:
                                vram_mb = int(memory_str)
                                # Convert GB to MB if the original string contained GB
                                if "GB" in parts[1] or "GiB" in parts[1]:
                                    vram_mb *= 1024
                            except ValueError as ve:
                                logger.debug(f"Failed to parse VRAM size '{parts[1]}': {ve}")
                                raise ValueError(f"Invalid VRAM format: {parts[1]}")

                            gpu_info = {
                                "type": "nvidia",
                                "name": gpu_name,
                                "vram_mb": vram_mb,
                            }
                            logger.info(
                                f"Detected NVIDIA GPU: {gpu_info['name']} with "
                                f"{gpu_info['vram_mb']} MB VRAM"
                            )
                        except ValueError as ve:
                            logger.debug(f"NVIDIA GPU parsing failed: {ve}")
                        except Exception as e:
                            logger.debug(f"Unexpected error parsing NVIDIA GPU info: {e}")
        except subprocess.TimeoutExpired:
            logger.debug("nvidia-smi command timed out")
        except FileNotFoundError:
            logger.debug("nvidia-smi command not found")
        except Exception as e:
            logger.debug(f"NVIDIA GPU detection failed: {e}")

        # If no NVIDIA, try AMD on Linux
        if not gpu_info and platform.system().lower() == "linux":
            try:
                result = subprocess.run(
                    ["rocm-smi", "--showmeminfo", "vram"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    # Parse VRAM from rocm-smi output
                    output = result.stdout.strip()
                    vram_mb = 8192  # Default fallback

                    # Try to parse VRAM size from output
                    try:
                        # Look for memory values in the output
                        import re

                        # Common patterns: "8192 MB", "8 GB", "8192MiB", etc.
                        memory_patterns = [
                            r'(\d+)\s*MB',  # 8192 MB
                            r'(\d+)\s*MiB',  # 8192 MiB
                            r'(\d+)\s*GB',  # 8 GB
                            r'(\d+)\s*GiB',  # 8 GiB
                        ]

                        for pattern in memory_patterns:
                            match = re.search(pattern, output, re.IGNORECASE)
                            if match:
                                value = int(match.group(1))
                                # Convert GB to MB if needed
                                if 'GB' in pattern.upper() or 'GIB' in pattern.upper():
                                    value *= 1024
                                vram_mb = value
                                break
                    except Exception:
                        pass  # Keep default if parsing fails

                    gpu_info = {
                        "type": "amd",
                        "name": "AMD GPU",
                        "vram_mb": vram_mb,
                    }
                    logger.info(f"Detected AMD GPU with {gpu_info['vram_mb']} MB VRAM")
            except Exception as e:
                logger.debug(f"AMD GPU not detected: {e}")

        # On macOS, check for Metal
        if not gpu_info and platform.system().lower() == "darwin":
            try:
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if "Apple" in result.stdout:
                    gpu_info = {
                        "type": "apple",
                        "name": "Apple Silicon",
                        "vram_mb": 8192,  # Shared with RAM
                    }
                    logger.info("Detected Apple Silicon GPU with shared memory")
            except Exception as e:
                logger.debug(f"Apple GPU not detected: {e}")

        if not gpu_info:
            logger.info("No GPU detected")

        return gpu_info

    @staticmethod
    def detect_disk_space() -> int:
        """Detect free disk space in GB"""
        import os

        try:
            logger.info("Detecting available disk space...")
            # Use the root of the current drive
            if platform.system().lower() == "windows":
                root_path = os.path.splitdrive(os.getcwd())[0] + os.sep
            else:
                root_path = "/"
            usage = psutil.disk_usage(root_path)
            disk_gb = round(usage.free / (1024**3))
            logger.info(f"Detected free disk space: {disk_gb} GB")
            return disk_gb
        except Exception as e:
            logger.error(f"Error detecting disk space: {e}")
            return 50  # Assume 50GB by default


class ModelRecommender:
    """Recommends model based on hardware"""

    MODELS = {
        "0.5b": {
            "ram_min": 2,
            "context": 32768,
            "size": "0.5B",
            "ollama_model": "qwen2.5-coder:0.5b",
        },
        "1.5b": {
            "ram_min": 4,
            "context": 32768,
            "size": "1.5B",
            "ollama_model": "qwen2.5-coder:1.5b",
        },
        "3b": {
            "ram_min": 8,
            "context": 32768,
            "size": "3B",
            "ollama_model": "qwen2.5-coder:3b",
        },
        "7b": {
            "ram_min": 16,
            "context": 32768,
            "size": "7B",
            "ollama_model": "qwen2.5-coder:7b",
        },
        "14b": {
            "ram_min": 32,
            "context": 32768,
            "size": "14B",
            "ollama_model": "qwen2.5-coder:14b",
        },
        "32b": {
            "ram_min": 64,
            "context": 32768,
            "size": "32B",
            "ollama_model": "qwen2.5-coder:32b",
        },
    }

    @classmethod
    def recommend(cls, ram_gb: int, gpu_info: Optional[Dict]) -> str:
        """Recommend best model based on hardware"""
        logger.info(f"Recommending model based on {ram_gb}GB RAM and GPU: {gpu_info}")

        # If GPU with enough VRAM
        vram_mb = gpu_info.get("vram_mb", 0) if gpu_info else 0

        # Validate VRAM value for reasonableness before any comparisons
        if gpu_info and isinstance(vram_mb, int) and vram_mb >= 8192:
            # Check if VRAM value is reasonable (between 1GB and 128GB)
            if vram_mb < 1024 or vram_mb > 131072:
                logger.warning(
                    f"Invalid VRAM value detected: {vram_mb} MB. Skipping GPU-based recommendation."
                )
                # Fall back to RAM-based recommendation
            else:
                if vram_mb >= 32768:
                    logger.info("Recommending 14B model based on GPU VRAM")
                    return "14b"
                elif vram_mb >= 16384:
                    logger.info("Recommending 7B model based on GPU VRAM")
                    return "7b"
                else:
                    logger.info("Recommending 3B model based on GPU VRAM")
                    return "3b"

        # Based on RAM
        if ram_gb >= 64:
            logger.info("Recommending 32B model based on RAM")
            return "32b"
        elif ram_gb >= 32:
            logger.info("Recommending 14B model based on RAM")
            return "14b"
        elif ram_gb >= 16:
            logger.info("Recommending 7B model based on RAM")
            return "7b"
        elif ram_gb >= 8:
            logger.info("Recommending 3B model based on RAM")
            return "3b"
        elif ram_gb >= 4:
            logger.info("Recommending 1.5B model based on RAM")
            return "1.5b"
        else:
            logger.info("Recommending 0.5B model based on RAM")
            return "0.5b"
