#!/usr/bin/env python3
"""
Test script to verify ACOLYTE pip installation
Run this after installing ACOLYTE to verify everything works
"""

import sys
import subprocess

def test_pip_installation():
    """Test ACOLYTE installation via pip"""
    
    print("🧪 Testing ACOLYTE pip installation...\n")
    
    # Test 1: Import the package
    print("1️⃣ Testing package import...")
    try:
        import acolyte
        print(f"✅ Package imported successfully")
        print(f"   Version: {acolyte.__version__}")
        print(f"   Author: {acolyte.__author__}")
        print(f"   License: {acolyte.__license__}")
    except ImportError as e:
        print(f"❌ Failed to import acolyte: {e}")
        return False
    
    # Test 2: Check CLI command
    print("\n2️⃣ Testing CLI command...")
    try:
        result = subprocess.run([sys.executable, "-m", "acolyte.cli", "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ CLI module works: {result.stdout.strip()}")
        else:
            print(f"❌ CLI module failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Failed to run CLI: {e}")
        return False
    
    # Test 3: Check if acolyte command is available
    print("\n3️⃣ Testing acolyte command...")
    try:
        result = subprocess.run(["acolyte", "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ acolyte command works: {result.stdout.strip()}")
        else:
            print(f"⚠️  acolyte command not in PATH")
            print("   You may need to add Python scripts to your PATH")
            print("   Or use: python -m acolyte.cli")
    except FileNotFoundError:
        print(f"⚠️  acolyte command not found in PATH")
        print("   Use: python -m acolyte.cli")
    except Exception as e:
        print(f"❌ Error testing acolyte command: {e}")
    
    # Test 4: Check core components
    print("\n4️⃣ Testing core components...")
    try:
        # Test configuration
        config = acolyte.get_config()
        print(f"✅ Configuration system works")
        
        # Test ID generation
        test_id = acolyte.generate_id()
        print(f"✅ ID generation works: {test_id[:8]}...")
        
        # Test logger
        acolyte.logger.info("Test log message")
        print(f"✅ Logging system works")
        
        # Test version check
        info = acolyte.check_version()
        print(f"✅ Version check works")
        print(f"   Python: {info['python'].split()[0]}")
        print(f"   Platform: {info['platform'].split('-')[0]}")
        
    except Exception as e:
        print(f"❌ Core component test failed: {e}")
        return False
    
    # Test 5: Check if ready
    print("\n5️⃣ Testing readiness...")
    try:
        if acolyte.is_ready():
            print(f"✅ ACOLYTE core is ready")
        else:
            print(f"⚠️  ACOLYTE core not fully initialized")
            print("   This is normal on first run")
    except Exception as e:
        print(f"❌ Readiness check failed: {e}")
    
    print("\n✨ All basic tests passed! ACOLYTE is installed correctly.")
    print("\nNext steps:")
    print("  1. Go to a project: cd /path/to/project")
    print("  2. Initialize: acolyte init")
    print("  3. Install services: acolyte install")
    print("  4. Start: acolyte start")
    
    return True

if __name__ == "__main__":
    success = test_pip_installation()
    sys.exit(0 if success else 1)
