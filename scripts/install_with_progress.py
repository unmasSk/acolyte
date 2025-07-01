#!/usr/bin/env python3
"""
Enhanced installation script with progress visualization
"""

import sys
import subprocess
import time
from pathlib import Path


class TqdmWrapper:
    """Wrapper to show progress during pip install"""

    def __init__(self, original_stdout):
        self.original_stdout = original_stdout
        self.buffer = ""
        self.last_update = time.time()

    def write(self, text):
        self.buffer += text
        current_time = time.time()

        # Update every 2 seconds to avoid spam
        if current_time - self.last_update > 2:
            if "Installing collected packages" in self.buffer:
                self.original_stdout.write("📦 Installing packages...\n")
            elif "Successfully installed" in self.buffer:
                self.original_stdout.write("✅ Packages installed successfully!\n")
            elif "Downloading" in self.buffer:
                self.original_stdout.write("⬇️  Downloading dependencies...\n")

            self.buffer = ""
            self.last_update = current_time

    def flush(self):
        self.original_stdout.flush()


def install_with_progress():
    """Install ACOLYTE with enhanced progress feedback"""

    print("\n" + "=" * 60)
    print("🚀 ACOLYTE INSTALLATION")
    print("=" * 60)
    print("\n📦 Installing dependencies (this includes PyTorch ~2GB)")
    print("☕ Grab a coffee, this will take 2-5 minutes...")
    print("💡 This is a one-time download - all projects share the same models")
    print("\n" + "=" * 60)

    # Get the current directory
    project_root = Path(__file__).parent.parent

    try:
        # Run pip install with real-time progress
        cmd = [sys.executable, "-m", "pip", "install", "-e", "."]

        print("\n🔄 Starting installation...")

        # Use Popen for real-time output
        process = subprocess.Popen(
            cmd,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Create progress wrapper
        progress_wrapper = TqdmWrapper(sys.stdout)

        # Read output in real-time
        if process.stdout:
            while True:
                output_line = process.stdout.readline()
                if output_line == '' and process.poll() is not None:
                    break
                if output_line:
                    progress_wrapper.write(output_line)

        # Wait for process to complete
        return_code = process.wait()

        if return_code == 0:
            print("\n" + "=" * 60)
            print("✅ Installation completed successfully!")
            print("📖 Next: Run 'acolyte init' in your project")
            print("🔍 Run 'acolyte doctor' to verify your setup")
            print("=" * 60 + "\n")
        else:
            print("\n❌ Installation failed!")
            print("Check the output above for error details")
            sys.exit(1)

    except subprocess.TimeoutExpired:
        print("\n⏰ Installation timed out after 10 minutes")
        print("Try again or check your internet connection")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Installation error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    install_with_progress()
