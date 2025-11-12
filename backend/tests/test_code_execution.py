"""
Code Execution Service Tests

Comprehensive test suite for secure code execution sandbox.

Author: GRETA PAI Development Team
Date: 2025-11-09
"""

import pytest
import asyncio
import os
from datetime import datetime

from models.code_execution_models import (
    CodeExecutionRequest,
    CodeExecutionResult,
    ExecutionLanguage,
    ExecutionStatus,
    ResourceLimits
)
from services.code_execution_service import CodeExecutionService
from utils.database import Database


# Test fixtures
@pytest.fixture
async def db():
    """Database fixture"""
    database = Database()
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
async def service(db):
    """Code execution service fixture"""
    return CodeExecutionService(db)


# ============================================================================
# BASIC EXECUTION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_execute_python_hello_world(service):
    """Test basic Python execution"""
    request = CodeExecutionRequest(
        code='print("Hello, World!")',
        language=ExecutionLanguage.PYTHON,
        timeout=30
    )
    
    result = await service.execute_code(request)
    
    assert result.success is True
    assert result.status == ExecutionStatus.COMPLETED
    assert "Hello, World!" in result.stdout
    assert result.exit_code == 0
    assert result.execution_time > 0


@pytest.mark.asyncio
async def test_execute_javascript_hello_world(service):
    """Test basic JavaScript execution"""
    request = CodeExecutionRequest(
        code='console.log("Hello, World!");',
        language=ExecutionLanguage.JAVASCRIPT,
        timeout=30
    )
    
    result = await service.execute_code(request)
    
    assert result.success is True
    assert result.status == ExecutionStatus.COMPLETED
    assert "Hello, World!" in result.stdout
    assert result.exit_code == 0
    assert result.execution_time > 0


@pytest.mark.asyncio
async def test_execute_python_math(service):
    """Test Python mathematical operations"""
    request = CodeExecutionRequest(
        code='''
import math
result = math.sqrt(16) + math.pi
print(f"Result: {result:.2f}")
''',
        language=ExecutionLanguage.PYTHON,
        timeout=30
    )
    
    result = await service.execute_code(request)
    
    assert result.success is True
    assert "Result:" in result.stdout


@pytest.mark.asyncio
async def test_execute_javascript_math(service):
    """Test JavaScript mathematical operations"""
    request = CodeExecutionRequest(
        code='''
const result = Math.sqrt(16) + Math.PI;
console.log(`Result: ${result.toFixed(2)}`);
''',
        language=ExecutionLanguage.JAVASCRIPT,
        timeout=30
    )
    
    result = await service.execute_code(request)
    
    assert result.success is True
    assert "Result:" in result.stdout


# ============================================================================
# VALIDATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_validate_python_syntax_valid(service):
    """Test Python syntax validation - valid code"""
    validation = await service.validate_code(
        'print("Hello")',
        ExecutionLanguage.PYTHON
    )
    
    assert validation.valid is True
    assert len(validation.errors) == 0


@pytest.mark.asyncio
async def test_validate_python_syntax_invalid(service):
    """Test Python syntax validation - invalid code"""
    validation = await service.validate_code(
        'print("Hello"',  # Missing closing parenthesis
        ExecutionLanguage.PYTHON
    )
    
    assert validation.valid is False
    assert len(validation.errors) > 0


@pytest.mark.asyncio
async def test_validate_code_size_limit(service):
    """Test code size limit validation"""
    large_code = "x = 1\n" * 100000  # > 100 KB
    
    validation = await service.validate_code(
        large_code,
        ExecutionLanguage.PYTHON
    )
    
    assert validation.valid is False
    assert any("size" in error.lower() for error in validation.errors)


@pytest.mark.asyncio
async def test_validate_forbidden_patterns_python(service):
    """Test forbidden pattern detection - Python"""
    validation = await service.validate_code(
        'import os\nos.system("rm -rf /")',
        ExecutionLanguage.PYTHON
    )
    
    # Should have warnings about dangerous patterns
    assert len(validation.warnings) > 0 or len(validation.forbidden_patterns) > 0


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_execute_python_runtime_error(service):
    """Test Python runtime error handling"""
    request = CodeExecutionRequest(
        code='raise ValueError("Test error")',
        language=ExecutionLanguage.PYTHON,
        timeout=30
    )
    
    result = await service.execute_code(request)
    
    assert result.success is False
    assert result.status == ExecutionStatus.FAILED
    assert result.exit_code != 0


@pytest.mark.asyncio
async def test_execute_javascript_runtime_error(service):
    """Test JavaScript runtime error handling"""
    request = CodeExecutionRequest(
        code='throw new Error("Test error");',
        language=ExecutionLanguage.JAVASCRIPT,
        timeout=30
    )
    
    result = await service.execute_code(request)
    
    assert result.success is False
    assert result.status == ExecutionStatus.FAILED


@pytest.mark.asyncio
async def test_execute_python_syntax_error(service):
    """Test Python syntax error handling"""
    request = CodeExecutionRequest(
        code='print("Hello"',  # Missing closing parenthesis
        language=ExecutionLanguage.PYTHON,
        timeout=30
    )
    
    result = await service.execute_code(request)
    
    assert result.success is False
    assert "syntax" in result.stderr.lower() or "validation" in result.stderr.lower()


# ============================================================================
# RESOURCE LIMIT TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_execute_with_custom_resource_limits(service):
    """Test execution with custom resource limits"""
    limits = ResourceLimits(
        cpu_quota=25000,  # 25% CPU
        mem_limit="128m",  # 128 MB RAM
        pids_limit=25  # 25 processes max
    )
    
    request = CodeExecutionRequest(
        code='print("Hello with limits")',
        language=ExecutionLanguage.PYTHON,
        timeout=30,
        resource_limits=limits
    )
    
    result = await service.execute_code(request)
    
    assert result.success is True


# ============================================================================
# STATISTICS TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_get_statistics(service):
    """Test statistics retrieval"""
    # Execute some code first
    request = CodeExecutionRequest(
        code='print("Test")',
        language=ExecutionLanguage.PYTHON,
        timeout=30
    )
    
    await service.execute_code(request)
    
    # Get statistics
    stats = await service.get_statistics()
    
    assert stats.total_executions > 0
    assert stats.successful_executions >= 0
    assert stats.failed_executions >= 0


# ============================================================================
# STORAGE AND RETRIEVAL TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_get_execution_by_id(service):
    """Test execution retrieval by ID"""
    # Execute code
    request = CodeExecutionRequest(
        code='print("Test retrieval")',
        language=ExecutionLanguage.PYTHON,
        timeout=30
    )
    
    result = await service.execute_code(request)
    execution_id = result.execution_id
    
    # Retrieve execution
    retrieved = await service.get_execution(execution_id)
    
    assert retrieved is not None
    assert retrieved.execution_id == execution_id
    assert retrieved.success == result.success


@pytest.mark.asyncio
async def test_list_executions(service):
    """Test listing executions"""
    # Execute some code
    request = CodeExecutionRequest(
        code='print("Test listing")',
        language=ExecutionLanguage.PYTHON,
        timeout=30
    )
    
    await service.execute_code(request)
    
    # List executions
    executions = await service.list_executions(limit=10)
    
    assert isinstance(executions, list)
    assert len(executions) > 0


# ============================================================================
# HEALTH CHECK TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_health_check(service):
    """Test service health check"""
    health = await service.health_check()
    
    assert "healthy" in health
    assert "docker_connected" in health
    assert "images_available" in health


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    print("🧪 Running Code Execution Service Tests...")
    print("=" * 80)
    
    # Run tests
    pytest.main([__file__, "-v", "-s"])

