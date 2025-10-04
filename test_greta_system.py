#!/usr/bin/env python3
"""
GRETA PAI System Test Script
Comprehensive testing of the Personal AI Infrastructure
"""

import sys
import asyncio
import traceback
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

def test_import(module_name, import_path):
    """Test module import"""
    try:
        module = __import__(import_path, fromlist=[module_name])
        print(f"✅ {module_name} - Import successful")
        return True, module
    except Exception as e:
        print(f"❌ {module_name} - Import failed: {e}")
        return False, None

def test_function(function_name, func, *args):
    """Test function execution"""
    try:
        result = func(*args)
        print(f"✅ {function_name} - Execution successful")
        return True, result
    except Exception as e:
        print(f"❌ {function_name} - Execution failed: {e}")
        return False, None

async def test_async_function(function_name, func, *args):
    """Test async function execution"""
    try:
        result = await func(*args)
        print(f"✅ {function_name} - Async execution successful")
        return True, result
    except Exception as e:
        print(f"❌ {function_name} - Async execution failed: {e}")
        return False, None

def test_pai_directory_structure():
    """Test PAI directory structure"""
    print("\n🏗️ Testing PAI Directory Structure...")
    
    pai_dir = Path(__file__).parent / "PAI_DIRECTORY"
    required_dirs = ["context", "commands", "agents", "hooks", "voice-server"]
    
    all_good = True
    for dir_name in required_dirs:
        dir_path = pai_dir / dir_name
        if dir_path.exists():
            print(f"✅ PAI_DIRECTORY/{dir_name} - Exists")
        else:
            print(f"⚠️ PAI_DIRECTORY/{dir_name} - Not found (will be created)")
    
    # Check for CLAUDE.md
    claude_file = pai_dir / "context" / "CLAUDE.md"
    if claude_file.exists():
        print(f"✅ PAI_DIRECTORY/context/CLAUDE.md - PAI master context exists")
    else:
        print(f"❌ PAI_DIRECTORY/context/CLAUDE.md - Missing")
        all_good = False
    
    return all_good

def test_configuration():
    """Test configuration system"""
    print("\n⚙️ Testing Configuration System...")
    
    success, config = test_import("Settings", "backend.config")
    if not success:
        return False
    
    # Test settings instance
    try:
        settings = config.get_settings()
        print(f"✅ Settings instance - App: {settings.app_name}, Version: {settings.version}")
        print(f"✅ PAI Directory: {settings.pai_dir}")
        print(f"✅ Digital Assistant: {settings.digital_assistant_name}")
        return True
    except Exception as e:
        print(f"❌ Settings instance failed: {e}")
        return False

async def test_database():
    """Test database system"""
    print("\n💾 Testing Database System...")
    
    success, db = test_import("Database", "backend.database")
    if not success:
        return False
    
    # Test database instance
    try:
        database = await db.get_database()
        print(f"✅ Database instance - Type: {'MongoDB' if database.MOTOR_AVAILABLE else 'File-based'}")
        
        # Test health check
        health = await database.health_check()
        print(f"✅ Database health - Status: {health.get('status', 'unknown')}")
        print(f"✅ Storage type: {health.get('type', 'unknown')}")
        return True
    except Exception as e:
        print(f"❌ Database instance failed: {e}")
        return False

async def test_llm_service():
    """Test LLM service"""
    print("\n🤖 Testing LLM Service...")
    
    success, llm = test_import("LLMService", "backend.services.llm_service")
    if not success:
        return False
    
    # Test LLM service instance
    try:
        llm_service = await llm.get_llm_service()
        print(f"✅ LLM service instance - Available models: {len(llm_service.available_models)}")
        
        # Test health check
        health = await llm_service.health_check()
        print(f"✅ LLM service health - Status: {health.get('status', 'unknown')}")
        print(f"✅ PAI Intelligence: {health.get('pai_intelligence', 'unknown')}")
        return True
    except Exception as e:
        print(f"❌ LLM service instance failed: {e}")
        return False

def test_main_application():
    """Test main application"""
    print("\n🚀 Testing Main Application...")
    
    success, main = test_import("main", "backend.main")
    if not success:
        return False
    
    print("✅ Main application module imported successfully")
    return True

async def run_comprehensive_test():
    """Run comprehensive system test"""
    print("🎭 GRETA PAI System Test")
    print("=" * 50)
    print("Testing the World's Most Advanced Personal AI System")
    print("=" * 50)
    
    test_results = []
    
    # Test 1: PAI Directory Structure
    test_results.append(test_pai_directory_structure())
    
    # Test 2: Configuration System
    test_results.append(test_configuration())
    
    # Test 3: Database System
    test_results.append(await test_database())
    
    # Test 4: LLM Service
    test_results.append(await test_llm_service())
    
    # Test 5: Main Application
    test_results.append(test_main_application())
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"✅ Tests Passed: {passed}/{total}")
    print(f"❌ Tests Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("🚀 GRETA PAI System is ready to launch!")
        print("\n📋 Next Steps:")
        print("1. Run: ./Launch_Greta_PAI.command")
        print("2. Open: http://localhost:8000")
        print("3. Check health: http://localhost:8000/health")
        print("4. View API docs: http://localhost:8000/docs")
        return True
    else:
        print(f"\n⚠️ {total - passed} tests failed. Please fix issues before launching.")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(run_comprehensive_test())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Critical error during testing: {e}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        sys.exit(1)
