#!/usr/bin/env python3
"""
Post-installation script for ACOLYTE
Runs automatically after pip install
"""

import sys
import subprocess
import shutil
from pathlib import Path


def setup_acolyte_home():
    """Setup ACOLYTE home directory"""
    acolyte_home = Path.home() / ".acolyte"
    acolyte_home.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (acolyte_home / "projects").mkdir(exist_ok=True)
    (acolyte_home / "models").mkdir(exist_ok=True)
    (acolyte_home / "logs").mkdir(exist_ok=True)

    # Mark as pip installed
    (acolyte_home / ".pip_installed").touch()

    print(f"✓ ACOLYTE home directory created: {acolyte_home}")


def verify_path():
    """Verify that acolyte command is in PATH"""
    acolyte_path = shutil.which('acolyte')

    if acolyte_path is None:
        print("⚠️  Warning: 'acolyte' command not found in PATH")
        print("   This usually means the Scripts/ directory is not in your PATH")

        # Try to find where pip installed the script
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", "-f", "acolyte"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if line.startswith('acolyte'):
                        script_path = Path(line.split()[1])
                        scripts_dir = script_path.parent
                        print(f"   Script location: {scripts_dir}")
                        print("   Add this directory to your PATH environment variable")
                        break
        except Exception:
            pass

        print("   After adding to PATH, restart your terminal and run 'acolyte --version'")
    else:
        print(f"✓ ACOLYTE command found: {acolyte_path}")


def main():
    """Main post-installation setup"""
    print("🔧 ACOLYTE Post-Installation Setup")
    print("=" * 40)

    # Setup home directory
    setup_acolyte_home()

    # Verify PATH
    verify_path()

    print("\n✅ ACOLYTE installation completed!")
    print("   Run 'acolyte --help' to see available commands")


if __name__ == "__main__":
    main()
