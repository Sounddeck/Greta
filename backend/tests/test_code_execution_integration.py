"""
Code Execution Integration Tests

Simple integration tests for code execution service (no pytest required).

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
    ExecutionStatus,
    ResourceLimits
)
from services.code_execution_service import CodeExecutionService
from database import Database


async def test_python_hello_world():
    """Test 1: Python Hello World"""
    print("\n" + "=" * 80)
    print("TEST 1: Python Hello World")
    print("=" * 80)
    
    db = Database()
    await db.connect()
    service = CodeExecutionService(db)
    
    request = CodeExecutionRequest(
        code='print("Hello from Python!")',
        language=ExecutionLanguage.PYTHON,
        timeout=30
    )
    
    result = await service.execute_code(request)
    
    print(f"✅ Execution ID: {result.execution_id}")
    print(f"✅ Status: {result.status}")
    print(f"✅ Success: {result.success}")
    print(f"✅ Stdout: {result.stdout}")
    print(f"✅ Execution Time: {result.execution_time:.3f}s")
    
    assert result.success is True, "Python execution should succeed"
    assert result.status == ExecutionStatus.COMPLETED, "Status should be COMPLETED"
    assert "Hello from Python!" in result.stdout, "Output should contain message"

    print("✅ TEST 1 PASSED")


async def test_javascript_hello_world():
    """Test 2: JavaScript Hello World"""
    print("\n" + "=" * 80)
    print("TEST 2: JavaScript Hello World")
    print("=" * 80)
    
    db = Database()
    await db.connect()
    service = CodeExecutionService(db)
    
    request = CodeExecutionRequest(
        code='console.log("Hello from JavaScript!");',
        language=ExecutionLanguage.JAVASCRIPT,
        timeout=30
    )
    
    result = await service.execute_code(request)
    
    print(f"✅ Execution ID: {result.execution_id}")
    print(f"✅ Status: {result.status}")
    print(f"✅ Success: {result.success}")
    print(f"✅ Stdout: {result.stdout}")
    print(f"✅ Execution Time: {result.execution_time:.3f}s")
    
    assert result.success is True, "JavaScript execution should succeed"
    assert result.status == ExecutionStatus.COMPLETED, "Status should be COMPLETED"
    assert "Hello from JavaScript!" in result.stdout, "Output should contain message"

    print("✅ TEST 2 PASSED")


async def test_python_validation():
    """Test 3: Python Code Validation"""
    print("\n" + "=" * 80)
    print("TEST 3: Python Code Validation")
    print("=" * 80)
    
    db = Database()
    await db.connect()
    service = CodeExecutionService(db)
    
    # Valid code
    validation = await service.validate_code(
        'print("Valid code")',
        ExecutionLanguage.PYTHON
    )
    
    print(f"✅ Valid Code - Valid: {validation.valid}")
    print(f"✅ Errors: {validation.errors}")
    print(f"✅ Warnings: {validation.warnings}")
    
    assert validation.valid is True, "Valid code should pass validation"
    
    # Invalid code
    validation = await service.validate_code(
        'print("Invalid code"',  # Missing closing parenthesis
        ExecutionLanguage.PYTHON
    )
    
    print(f"✅ Invalid Code - Valid: {validation.valid}")
    print(f"✅ Errors: {validation.errors}")
    
    assert validation.valid is False, "Invalid code should fail validation"
    assert len(validation.errors) > 0, "Should have syntax errors"

    print("✅ TEST 3 PASSED")


async def test_error_handling():
    """Test 4: Error Handling"""
    print("\n" + "=" * 80)
    print("TEST 4: Error Handling")
    print("=" * 80)
    
    db = Database()
    await db.connect()
    service = CodeExecutionService(db)
    
    request = CodeExecutionRequest(
        code='raise ValueError("Test error")',
        language=ExecutionLanguage.PYTHON,
        timeout=30
    )
    
    result = await service.execute_code(request)
    
    print(f"✅ Execution ID: {result.execution_id}")
    print(f"✅ Status: {result.status}")
    print(f"✅ Success: {result.success}")
    print(f"✅ Error: {result.error}")
    print(f"✅ Stderr: {result.stderr[:200]}")
    
    assert result.success is False, "Error execution should fail"
    assert result.status == ExecutionStatus.FAILED, "Status should be FAILED"

    print("✅ TEST 4 PASSED")


async def test_resource_limits():
    """Test 5: Custom Resource Limits"""
    print("\n" + "=" * 80)
    print("TEST 5: Custom Resource Limits")
    print("=" * 80)
    
    db = Database()
    await db.connect()
    service = CodeExecutionService(db)
    
    limits = ResourceLimits(
        cpu_quota=25000,  # 25% CPU
        mem_limit="128m",  # 128 MB RAM
        pids_limit=25  # 25 processes max
    )
    
    request = CodeExecutionRequest(
        code='print("Hello with custom limits!")',
        language=ExecutionLanguage.PYTHON,
        timeout=30,
        resource_limits=limits
    )
    
    result = await service.execute_code(request)
    
    print(f"✅ Execution ID: {result.execution_id}")
    print(f"✅ Status: {result.status}")
    print(f"✅ Success: {result.success}")
    print(f"✅ CPU Quota: {limits.cpu_quota}")
    print(f"✅ Memory Limit: {limits.mem_limit}")
    
    assert result.success is True, "Execution with custom limits should succeed"

    print("✅ TEST 5 PASSED")


async def test_statistics():
    """Test 6: Statistics Tracking"""
    print("\n" + "=" * 80)
    print("TEST 6: Statistics Tracking")
    print("=" * 80)
    
    db = Database()
    await db.connect()
    service = CodeExecutionService(db)
    
    # Execute some code
    request = CodeExecutionRequest(
        code='print("Statistics test")',
        language=ExecutionLanguage.PYTHON,
        timeout=30
    )
    
    await service.execute_code(request)
    
    # Get statistics
    stats = await service.get_statistics()
    
    print(f"✅ Total Executions: {stats.total_executions}")
    print(f"✅ Successful: {stats.successful_executions}")
    print(f"✅ Failed: {stats.failed_executions}")
    print(f"✅ Timeouts: {stats.timeout_executions}")
    print(f"✅ Average Time: {stats.average_execution_time:.3f}s")
    print(f"✅ By Language: {stats.executions_by_language}")
    
    assert stats.total_executions > 0, "Should have executions"

    print("✅ TEST 6 PASSED")


async def test_health_check():
    """Test 7: Health Check"""
    print("\n" + "=" * 80)
    print("TEST 7: Health Check")
    print("=" * 80)
    
    db = Database()
    await db.connect()
    service = CodeExecutionService(db)
    
    health = await service.health_check()
    
    print(f"✅ Healthy: {health['healthy']}")
    print(f"✅ Docker Connected: {health['docker_connected']}")
    print(f"✅ Images Available: {health['images_available']}")
    print(f"✅ Total Executions: {health['total_executions']}")
    print(f"✅ Success Rate: {health['success_rate']:.1f}%")

    print("✅ TEST 7 PASSED")


async def run_all_tests():
    """Run all integration tests"""
    print("\n" + "=" * 80)
    print("🧪 CODE EXECUTION SERVICE INTEGRATION TESTS")
    print("=" * 80)
    
    tests = [
        test_python_hello_world,
        test_javascript_hello_world,
        test_python_validation,
        test_error_handling,
        test_resource_limits,
        test_statistics,
        test_health_check
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            print(f"\n❌ TEST FAILED: {test.__name__}")
            print(f"   Error: {e}")
            failed += 1
    
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    print(f"📈 Success Rate: {passed/len(tests)*100:.1f}%")
    print("=" * 80)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️ {failed} TEST(S) FAILED")
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)

