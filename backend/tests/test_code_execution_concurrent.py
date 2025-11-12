"""
Concurrent Execution Tests for Code Execution Service

Tests the ability to handle multiple simultaneous code executions.
"""

import asyncio
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import Database
from services.code_execution_service import CodeExecutionService
from models.code_execution_models import (
    CodeExecutionRequest,
    CodeExecutionResult,
    ExecutionLanguage,
    ExecutionStatus
)


async def test_concurrent_python_executions():
    """Test 1: 5 Concurrent Python Executions"""
    print("\n" + "=" * 80)
    print("CONCURRENT TEST 1: 5 Concurrent Python Executions")
    print("=" * 80)
    
    db = Database()
    await db.connect()
    service = CodeExecutionService(db)
    
    # Create 5 Python execution requests
    requests = [
        CodeExecutionRequest(
            code=f'import time; print("Execution {i}"); time.sleep(0.1); print("Done {i}")',
            language=ExecutionLanguage.PYTHON,
            timeout=10
        )
        for i in range(5)
    ]
    
    # Execute all concurrently
    start_time = time.time()
    results = await asyncio.gather(*[
        service.execute_code(req) for req in requests
    ])
    total_time = time.time() - start_time
    
    # Verify results
    success_count = sum(1 for r in results if r.success)
    failed_count = sum(1 for r in results if not r.success)
    avg_execution_time = sum(r.execution_time for r in results) / len(results)
    
    print(f"✅ Total Executions: {len(results)}")
    print(f"✅ Successful: {success_count}")
    print(f"✅ Failed: {failed_count}")
    print(f"✅ Total Time: {total_time:.2f}s")
    print(f"✅ Average Execution Time: {avg_execution_time:.2f}s")
    print(f"✅ Concurrency Benefit: {(sum(r.execution_time for r in results) / total_time):.1f}x")
    
    # All should succeed
    assert success_count == 5, f"Expected 5 successful executions, got {success_count}"
    assert failed_count == 0, f"Expected 0 failed executions, got {failed_count}"
    
    # Total time should be less than sequential execution
    sequential_time = sum(r.execution_time for r in results)
    assert total_time < sequential_time, f"Concurrent execution ({total_time:.2f}s) should be faster than sequential ({sequential_time:.2f}s)"
    
    print("✅ CONCURRENT TEST 1 PASSED")
    return results


async def test_concurrent_javascript_executions():
    """Test 2: 10 Concurrent JavaScript Executions"""
    print("\n" + "=" * 80)
    print("CONCURRENT TEST 2: 10 Concurrent JavaScript Executions")
    print("=" * 80)
    
    db = Database()
    await db.connect()
    service = CodeExecutionService(db)
    
    # Create 10 JavaScript execution requests
    requests = [
        CodeExecutionRequest(
            code=f'console.log("Execution {i}"); setTimeout(() => console.log("Done {i}"), 100);',
            language=ExecutionLanguage.JAVASCRIPT,
            timeout=10
        )
        for i in range(10)
    ]
    
    # Execute all concurrently
    start_time = time.time()
    results = await asyncio.gather(*[
        service.execute_code(req) for req in requests
    ])
    total_time = time.time() - start_time
    
    # Verify results
    success_count = sum(1 for r in results if r.success)
    failed_count = sum(1 for r in results if not r.success)
    avg_execution_time = sum(r.execution_time for r in results) / len(results)
    
    print(f"✅ Total Executions: {len(results)}")
    print(f"✅ Successful: {success_count}")
    print(f"✅ Failed: {failed_count}")
    print(f"✅ Total Time: {total_time:.2f}s")
    print(f"✅ Average Execution Time: {avg_execution_time:.2f}s")
    print(f"✅ Concurrency Benefit: {(sum(r.execution_time for r in results) / total_time):.1f}x")
    
    # All should succeed
    assert success_count == 10, f"Expected 10 successful executions, got {success_count}"
    assert failed_count == 0, f"Expected 0 failed executions, got {failed_count}"
    
    # Total time should be less than sequential execution
    sequential_time = sum(r.execution_time for r in results)
    assert total_time < sequential_time, f"Concurrent execution ({total_time:.2f}s) should be faster than sequential ({sequential_time:.2f}s)"
    
    print("✅ CONCURRENT TEST 2 PASSED")
    return results


async def test_concurrent_mixed_executions():
    """Test 3: 20 Concurrent Mixed (Python + JavaScript) Executions"""
    print("\n" + "=" * 80)
    print("CONCURRENT TEST 3: 20 Concurrent Mixed Executions")
    print("=" * 80)
    
    db = Database()
    await db.connect()
    service = CodeExecutionService(db)
    
    # Create 20 mixed execution requests (10 Python + 10 JavaScript)
    requests = []
    
    # 10 Python requests
    for i in range(10):
        requests.append(
            CodeExecutionRequest(
                code=f'print("Python {i}")',
                language=ExecutionLanguage.PYTHON,
                timeout=10
            )
        )
    
    # 10 JavaScript requests
    for i in range(10):
        requests.append(
            CodeExecutionRequest(
                code=f'console.log("JavaScript {i}");',
                language=ExecutionLanguage.JAVASCRIPT,
                timeout=10
            )
        )
    
    # Execute all concurrently
    start_time = time.time()
    results = await asyncio.gather(*[
        service.execute_code(req) for req in requests
    ])
    total_time = time.time() - start_time
    
    # Verify results
    success_count = sum(1 for r in results if r.success)
    failed_count = sum(1 for r in results if not r.success)
    python_count = sum(1 for r in results if r.language == ExecutionLanguage.PYTHON)
    js_count = sum(1 for r in results if r.language == ExecutionLanguage.JAVASCRIPT)
    avg_execution_time = sum(r.execution_time for r in results) / len(results)
    
    print(f"✅ Total Executions: {len(results)}")
    print(f"✅ Successful: {success_count}")
    print(f"✅ Failed: {failed_count}")
    print(f"✅ Python Executions: {python_count}")
    print(f"✅ JavaScript Executions: {js_count}")
    print(f"✅ Total Time: {total_time:.2f}s")
    print(f"✅ Average Execution Time: {avg_execution_time:.2f}s")
    print(f"✅ Concurrency Benefit: {(sum(r.execution_time for r in results) / total_time):.1f}x")
    
    # All should succeed
    assert success_count == 20, f"Expected 20 successful executions, got {success_count}"
    assert failed_count == 0, f"Expected 0 failed executions, got {failed_count}"
    assert python_count == 10, f"Expected 10 Python executions, got {python_count}"
    assert js_count == 10, f"Expected 10 JavaScript executions, got {js_count}"
    
    # Total time should be less than sequential execution
    sequential_time = sum(r.execution_time for r in results)
    assert total_time < sequential_time, f"Concurrent execution ({total_time:.2f}s) should be faster than sequential ({sequential_time:.2f}s)"
    
    print("✅ CONCURRENT TEST 3 PASSED")
    return results


async def test_concurrent_with_failures():
    """Test 4: Concurrent Executions with Some Failures"""
    print("\n" + "=" * 80)
    print("CONCURRENT TEST 4: Concurrent Executions with Failures")
    print("=" * 80)
    
    db = Database()
    await db.connect()
    service = CodeExecutionService(db)
    
    # Create mixed requests (some will succeed, some will fail)
    requests = [
        # Successful executions
        CodeExecutionRequest(code='print("Success 1")', language=ExecutionLanguage.PYTHON, timeout=10),
        CodeExecutionRequest(code='console.log("Success 2");', language=ExecutionLanguage.JAVASCRIPT, timeout=10),
        
        # Failed executions (syntax errors)
        CodeExecutionRequest(code='print("Missing paren"', language=ExecutionLanguage.PYTHON, timeout=10),
        CodeExecutionRequest(code='console.log("Missing paren"', language=ExecutionLanguage.JAVASCRIPT, timeout=10),
        
        # Runtime errors
        CodeExecutionRequest(code='raise ValueError("Test error")', language=ExecutionLanguage.PYTHON, timeout=10),
        CodeExecutionRequest(code='throw new Error("Test error");', language=ExecutionLanguage.JAVASCRIPT, timeout=10),
    ]
    
    # Execute all concurrently
    start_time = time.time()
    results = await asyncio.gather(*[
        service.execute_code(req) for req in requests
    ])
    total_time = time.time() - start_time
    
    # Verify results
    success_count = sum(1 for r in results if r.success)
    failed_count = sum(1 for r in results if not r.success)
    validation_errors = sum(1 for r in results if r.status == ExecutionStatus.FAILED and "validation" in str(r.error).lower())
    runtime_errors = sum(1 for r in results if r.status == ExecutionStatus.FAILED and "validation" not in str(r.error).lower())
    
    print(f"✅ Total Executions: {len(results)}")
    print(f"✅ Successful: {success_count}")
    print(f"✅ Failed: {failed_count}")
    print(f"✅ Validation Errors: {validation_errors}")
    print(f"✅ Runtime Errors: {runtime_errors}")
    print(f"✅ Total Time: {total_time:.2f}s")
    
    # Should have 2 successes and 4 failures
    assert success_count == 2, f"Expected 2 successful executions, got {success_count}"
    assert failed_count == 4, f"Expected 4 failed executions, got {failed_count}"
    
    print("✅ CONCURRENT TEST 4 PASSED")
    return results


async def test_resource_usage_under_load():
    """Test 5: Resource Usage Under Concurrent Load"""
    print("\n" + "=" * 80)
    print("CONCURRENT TEST 5: Resource Usage Under Load")
    print("=" * 80)
    
    db = Database()
    await db.connect()
    service = CodeExecutionService(db)
    
    # Create 15 CPU-intensive requests
    requests = [
        CodeExecutionRequest(
            code='sum(range(1000000))',
            language=ExecutionLanguage.PYTHON,
            timeout=10
        )
        for _ in range(15)
    ]
    
    # Execute all concurrently
    start_time = time.time()
    results = await asyncio.gather(*[
        service.execute_code(req) for req in requests
    ])
    total_time = time.time() - start_time
    
    # Verify results
    success_count = sum(1 for r in results if r.success)
    failed_count = sum(1 for r in results if not r.success)
    avg_execution_time = sum(r.execution_time for r in results) / len(results)
    max_execution_time = max(r.execution_time for r in results)
    min_execution_time = min(r.execution_time for r in results)
    
    print(f"✅ Total Executions: {len(results)}")
    print(f"✅ Successful: {success_count}")
    print(f"✅ Failed: {failed_count}")
    print(f"✅ Total Time: {total_time:.2f}s")
    print(f"✅ Average Execution Time: {avg_execution_time:.2f}s")
    print(f"✅ Min Execution Time: {min_execution_time:.2f}s")
    print(f"✅ Max Execution Time: {max_execution_time:.2f}s")
    print(f"✅ Concurrency Benefit: {(sum(r.execution_time for r in results) / total_time):.1f}x")
    
    # All should succeed
    assert success_count == 15, f"Expected 15 successful executions, got {success_count}"
    assert failed_count == 0, f"Expected 0 failed executions, got {failed_count}"
    
    print("✅ CONCURRENT TEST 5 PASSED")
    return results


async def main():
    """Run all concurrent execution tests"""
    print("\n" + "=" * 80)
    print("🧪 CODE EXECUTION CONCURRENT TESTS")
    print("=" * 80)
    
    tests = [
        ("Test 1: 5 Concurrent Python", test_concurrent_python_executions),
        ("Test 2: 10 Concurrent JavaScript", test_concurrent_javascript_executions),
        ("Test 3: 20 Concurrent Mixed", test_concurrent_mixed_executions),
        ("Test 4: Concurrent with Failures", test_concurrent_with_failures),
        ("Test 5: Resource Usage Under Load", test_resource_usage_under_load),
    ]
    
    passed = 0
    failed = 0
    failed_tests = []
    
    for test_name, test_func in tests:
        try:
            await test_func()
            passed += 1
        except AssertionError as e:
            failed += 1
            failed_tests.append((test_name, str(e)))
            print(f"\n❌ TEST FAILED: {test_name}")
            print(f"   Error: {e}")
        except Exception as e:
            failed += 1
            failed_tests.append((test_name, str(e)))
            print(f"\n❌ TEST ERROR: {test_name}")
            print(f"   Error: {e}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 CONCURRENT TEST SUMMARY")
    print("=" * 80)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    print(f"📈 Success Rate: {(passed / len(tests) * 100):.1f}%")
    print("=" * 80)
    
    if failed_tests:
        print("\n⚠️ FAILED TESTS:")
        for test_name, error in failed_tests:
            print(f"  - {test_name}: {error}")
        print("\n⚠️ {} TEST(S) FAILED".format(failed))
    else:
        print("\n🎉 ALL TESTS PASSED!")


if __name__ == "__main__":
    asyncio.run(main())

