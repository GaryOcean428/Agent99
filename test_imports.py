"""
Simple test for Agent99 basic functionality.
"""
import sys
import os

def test_basic_imports():
    """Test basic Python imports."""
    try:
        import json
        import logging
        import configparser
        print("✓ Basic imports successful")
        return True
    except ImportError as e:
        print(f"✗ Basic import failed: {e}")
        return False

def test_config_file():
    """Test config file exists and is readable."""
    try:
        if os.path.exists('config.toml'):
            print("✓ Config file exists")
            return True
        else:
            print("✗ Config file not found")
            return False
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def test_python_version():
    """Test Python version compatibility."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python version {version.major}.{version.minor} is compatible")
        return True
    else:
        print(f"✗ Python version {version.major}.{version.minor} is too old")
        return False

if __name__ == "__main__":
    print("Running Agent99 Simple Tests...")
    
    tests = [
        test_python_version,
        test_basic_imports,
        test_config_file
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! ✓")
        sys.exit(0)
    else:
        print("Some tests failed! ✗")
        sys.exit(1)
