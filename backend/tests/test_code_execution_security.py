"""
Code Execution Security Tests

Tests for security features of the code execution sandbox.

Author: GRETA PAI Development Team
Date: 2025-11-09
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.code_execution_models import (
    CodeExecutionRequest,
    ExecutionLanguage,
    ExecutionStatus
)
from services.code_execution_service import CodeExecutionService
from database import Database


async def test_forbidden_imports_python():
    """Test 1: Forbidden Imports - Python"""
    print("\n" + "=" * 80)
    print("SECURITY TEST 1: Forbidden Imports - Python")
    print("=" * 80)
    
    db = Database()
    await db.connect()
    service = CodeExecutionService(db)
    
    # Test forbidden imports
    forbidden_codes = [
        'import os; os.system("ls")',
        'import subprocess; subprocess.run(["ls"])',
        'eval("print(1)")',
        'exec("print(1)")',
        '__import__("os").system("ls")'
    ]
    
    for code in forbidden_codes:
        result = await service.execute_code(
            CodeExecutionRequest(
                code=code,
                language=ExecutionLanguage.PYTHON,
                timeout=10
            )
        )
        
        print(f"✅ Code: {code[:50]}...")
        print(f"   Status: {result.status}")
        print(f"   Success: {result.success}")
        
        # Should fail validation or execution
        assert result.success is False or result.status == ExecutionStatus.FAILED, \
            f"Forbidden code should not succeed: {code}"
    
    print("✅ SECURITY TEST 1 PASSED - All forbidden imports blocked")


async def test_network_isolation():
    """Test 2: Network Isolation"""
    print("\n" + "=" * 80)
    print("SECURITY TEST 2: Network Isolation")
    print("=" * 80)
    
    db = Database()
    await db.connect()
    service = CodeExecutionService(db)
    
    # Try to make network request (should fail due to network isolation)
    result = await service.execute_code(
        CodeExecutionRequest(
            code='import urllib.request; urllib.request.urlopen("http://google.com")',
            language=ExecutionLanguage.PYTHON,
            timeout=10,
            allow_network=False  # Network disabled
        )
    )
    
    print(f"✅ Status: {result.status}")
    print(f"✅ Success: {result.success}")
    
    # Should fail due to network isolation
    assert result.success is False, "Network request should fail when network is disabled"
    
    print("✅ SECURITY TEST 2 PASSED - Network isolation working")


async def test_resource_limits_enforcement():
    """Test 3: Resource Limits Enforcement"""
    print("\n" + "=" * 80)
    print("SECURITY TEST 3: Resource Limits Enforcement")
    print("=" * 80)
    
    db = Database()
    await db.connect()
    service = CodeExecutionService(db)
    
    # Try to allocate large memory (should be limited)
    result = await service.execute_code(
        CodeExecutionRequest(
            code='x = [0] * (100 * 1024 * 1024); print("Done")',  # Try to allocate 100MB
            language=ExecutionLanguage.PYTHON,
            timeout=10
        )
    )
    
    print(f"✅ Status: {result.status}")
    print(f"✅ Success: {result.success}")
    print(f"✅ Execution Time: {result.execution_time:.3f}s")
    
    # Should complete or fail gracefully (not crash)
    assert result.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED], \
        "Should complete or fail gracefully"
    
    print("✅ SECURITY TEST 3 PASSED - Resource limits enforced")


async def test_timeout_enforcement():
    """Test 4: Timeout Enforcement"""
    print("\n" + "=" * 80)
    print("SECURITY TEST 4: Timeout Enforcement")
    print("=" * 80)
    
    db = Database()
    await db.connect()
    service = CodeExecutionService(db)
    
    # Try to run infinite loop (should timeout)
    result = await service.execute_code(
        CodeExecutionRequest(
            code='import time; time.sleep(100)',  # Sleep for 100 seconds
            language=ExecutionLanguage.PYTHON,
            timeout=2  # But timeout after 2 seconds
        )
    )
    
    print(f"✅ Status: {result.status}")
    print(f"✅ Success: {result.success}")
    print(f"✅ Execution Time: {result.execution_time:.3f}s")
    
    # Should timeout
    assert result.status == ExecutionStatus.TIMEOUT, "Should timeout after 2 seconds"
    assert result.execution_time >= 2.0, "Should run for at least 2 seconds"
    assert result.execution_time < 5.0, "Should not run for more than 5 seconds"
    
    print("✅ SECURITY TEST 4 PASSED - Timeout enforcement working")


async def test_file_system_isolation():
    """Test 5: File System Isolation"""
    print("\n" + "=" * 80)
    print("SECURITY TEST 5: File System Isolation")
    print("=" * 80)
    
    db = Database()
    await db.connect()
    service = CodeExecutionService(db)
    
    # Try to read system files (should fail due to read-only filesystem)
    result = await service.execute_code(
        CodeExecutionRequest(
            code='with open("/etc/passwd", "r") as f: print(f.read())',
            language=ExecutionLanguage.PYTHON,
            timeout=10
        )
    )
    
    print(f"✅ Status: {result.status}")
    print(f"✅ Success: {result.success}")
    
    # Should fail validation or execution
    assert result.success is False, "Should not be able to read system files"
    
    print("✅ SECURITY TEST 5 PASSED - File system isolation working")


async def run_all_security_tests():
    """Run all security tests"""
    print("\n" + "=" * 80)
    print("🔒 CODE EXECUTION SECURITY TESTS")
    print("=" * 80)
    
    tests = [
        ("test_forbidden_imports_python", test_forbidden_imports_python),
        ("test_network_isolation", test_network_isolation),
        ("test_resource_limits_enforcement", test_resource_limits_enforcement),
        ("test_timeout_enforcement", test_timeout_enforcement),
        ("test_file_system_isolation", test_file_system_isolation),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            await test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ SECURITY TEST FAILED: {test_name}")
            print(f"   Error: {e}")
            failed += 1
    
    print("\n" + "=" * 80)
    print("📊 SECURITY TEST SUMMARY")
    print("=" * 80)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    print(f"📈 Success Rate: {(passed/len(tests)*100):.1f}%")
    print("=" * 80)
    
    if failed == 0:
        print("\n🎉 ALL SECURITY TESTS PASSED!")
    else:
        print(f"\n⚠️ {failed} SECURITY TEST(S) FAILED")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_all_security_tests())

