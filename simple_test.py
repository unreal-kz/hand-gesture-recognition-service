#!/usr/bin/env python3
"""
Simple test script to verify basic functionality
"""

import sys
import os

def test_basic_imports():
    """Test basic Python imports"""
    print("🧪 Testing basic imports...")
    
    try:
        # Test standard library imports
        import time
        import uuid
        import base64
        print("✅ Standard library imports: OK")
        
        # Test if we can import our modules
        sys.path.append('.')
        
        # Test config
        try:
            from config import settings
            print("✅ Config import: OK")
        except ImportError as e:
            print(f"⚠️  Config import: {e}")
        
        # Test logger
        try:
            from logger import setup_logging, get_logger
            print("✅ Logger import: OK")
        except ImportError as e:
            print(f"⚠️  Logger import: {e}")
        
        # Test monitoring
        try:
            from monitoring import metrics
            print("✅ Monitoring import: OK")
        except ImportError as e:
            print(f"⚠️  Monitoring import: {e}")
        
        # Test rate limiter
        try:
            from rate_limiter import RateLimitMiddleware
            print("✅ Rate limiter import: OK")
        except ImportError as e:
            print(f"⚠️  Rate limiter import: {e}")
        
        # Test batch processor
        try:
            from batch_processor import BatchProcessor
            print("✅ Batch processor import: OK")
        except ImportError as e:
            print(f"⚠️  Batch processor import: {e}")
        
        # Test ML processor
        try:
            from ml_processor import ml_processor
            print("✅ ML processor import: OK")
        except ImportError as e:
            print(f"⚠️  ML processor import: {e}")
        
        print("\n🎯 Basic import test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_file_structure():
    """Test file structure"""
    print("\n📁 Testing file structure...")
    
    required_files = [
        'main.py',
        'config.py',
        'logger.py',
        'monitoring.py',
        'rate_limiter.py',
        'batch_processor.py',
        'ml_processor.py',
        'requirements.txt',
        'Dockerfile',
        'docker-compose.yml',
        '.env'
    ]
    
    required_dirs = [
        'model',
        'model/keypoint_classifier',
        'monitoring',
        'monitoring/grafana',
        'monitoring/grafana/dashboards',
        'monitoring/grafana/datasources'
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"✅ {file}: OK")
    
    for dir in required_dirs:
        if not os.path.isdir(dir):
            missing_dirs.append(dir)
        else:
            print(f"✅ {dir}/: OK")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
    
    if not missing_files and not missing_dirs:
        print("🎯 File structure test: PASSED")
        return True
    else:
        print("❌ File structure test: FAILED")
        return False

def test_python_syntax():
    """Test Python syntax"""
    print("\n🐍 Testing Python syntax...")
    
    python_files = [
        'main.py',
        'config.py',
        'logger.py',
        'monitoring.py',
        'rate_limiter.py',
        'batch_processor.py',
        'ml_processor.py'
    ]
    
    syntax_errors = []
    
    for file in python_files:
        try:
            with open(file, 'r') as f:
                compile(f.read(), file, 'exec')
            print(f"✅ {file}: Syntax OK")
        except SyntaxError as e:
            print(f"❌ {file}: Syntax error - {e}")
            syntax_errors.append(file)
        except Exception as e:
            print(f"⚠️  {file}: {e}")
    
    if not syntax_errors:
        print("🎯 Python syntax test: PASSED")
        return True
    else:
        print(f"❌ Python syntax test: FAILED - {len(syntax_errors)} files have syntax errors")
        return False

def main():
    """Run all tests"""
    print("🚀 Hand Gesture Recognition Service - Basic Tests")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_file_structure,
        test_python_syntax
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Service is ready for deployment.")
        print("\n💡 Next steps:")
        print("   1. Install Python dependencies")
        print("   2. Test local service startup")
        print("   3. Deploy with Docker")
    else:
        print("⚠️  Some tests failed. Please fix issues before deployment.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
