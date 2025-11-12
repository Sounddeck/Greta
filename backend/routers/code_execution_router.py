"""
Code Execution Router

REST API endpoints for secure code execution sandbox.

Author: GRETA PAI Development Team
Date: 2025-11-09
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional, List
from loguru import logger

from models.code_execution_models import (
    CodeExecutionRequest,
    CodeExecutionResult,
    ValidationResult,
    ExecutionLanguage,
    ExecutionStatus,
    ExecutionStatistics
)
from services.code_execution_service import CodeExecutionService
from database import Database
from middleware.auth import get_current_user


# Initialize router
router = APIRouter(
    prefix="/api/v1/code-execution",
    tags=["Code Execution"],
    responses={404: {"description": "Not found"}}
)


# Dependency to get code execution service
async def get_code_execution_service() -> CodeExecutionService:
    """Get code execution service instance"""
    db = Database()
    await db.connect()
    return CodeExecutionService(db)


@router.post("/execute", response_model=CodeExecutionResult)
async def execute_code(
    request: CodeExecutionRequest,
    service: CodeExecutionService = Depends(get_code_execution_service),
    current_user: dict = Depends(get_current_user)
):
    """
    Execute code in secure sandbox
    
    **Security Features**:
    - Isolated Docker containers
    - Resource limits (CPU, memory, disk)
    - Execution timeouts
    - Network isolation (default)
    - Code validation
    
    **Supported Languages**:
    - Python 3.11
    - JavaScript (Node.js 20)
    
    **Example Request**:
    ```json
    {
        "code": "print('Hello, World!')",
        "language": "python",
        "timeout": 30,
        "allow_network": false
    }
    ```
    
    **Example Response**:
    ```json
    {
        "execution_id": "exec_abc123",
        "status": "completed",
        "success": true,
        "stdout": "Hello, World!\\n",
        "stderr": "",
        "exit_code": 0,
        "execution_time": 0.15,
        "language": "python"
    }
    ```
    """
    try:
        logger.info(f"🚀 Code execution request from user {current_user.get('user_id')}")
        logger.info(f"   Language: {request.language}")
        logger.info(f"   Code size: {len(request.code)} bytes")
        
        # Execute code
        result = await service.execute_code(request)
        
        logger.info(f"✅ Execution {result.execution_id} completed")
        logger.info(f"   Status: {result.status}")
        logger.info(f"   Success: {result.success}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Code execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate", response_model=ValidationResult)
async def validate_code(
    code: str,
    language: ExecutionLanguage,
    service: CodeExecutionService = Depends(get_code_execution_service),
    current_user: dict = Depends(get_current_user)
):
    """
    Validate code without executing
    
    Checks:
    - Syntax errors
    - Code size limits
    - Forbidden patterns (optional)
    
    **Example Request**:
    ```json
    {
        "code": "print('Hello')",
        "language": "python"
    }
    ```
    
    **Example Response**:
    ```json
    {
        "valid": true,
        "language": "python",
        "errors": [],
        "warnings": [],
        "forbidden_patterns": [],
        "code_size": 15
    }
    ```
    """
    try:
        logger.info(f"🔍 Code validation request from user {current_user.get('user_id')}")
        
        result = await service.validate_code(code, language)
        
        logger.info(f"✅ Validation completed: valid={result.valid}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Code validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{execution_id}", response_model=CodeExecutionResult)
async def get_execution(
    execution_id: str,
    service: CodeExecutionService = Depends(get_code_execution_service),
    current_user: dict = Depends(get_current_user)
):
    """
    Retrieve execution result by ID
    
    **Example Response**:
    ```json
    {
        "execution_id": "exec_abc123",
        "status": "completed",
        "success": true,
        "stdout": "Hello, World!\\n",
        "stderr": "",
        "exit_code": 0,
        "execution_time": 0.15,
        "language": "python",
        "created_at": "2025-11-09T12:00:00Z"
    }
    ```
    """
    try:
        result = await service.get_execution(execution_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[CodeExecutionResult])
async def list_executions(
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    language: Optional[ExecutionLanguage] = Query(None, description="Filter by language"),
    status: Optional[ExecutionStatus] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Results offset"),
    service: CodeExecutionService = Depends(get_code_execution_service),
    current_user: dict = Depends(get_current_user)
):
    """
    List recent code executions
    
    **Query Parameters**:
    - `agent_id`: Filter by agent ID
    - `language`: Filter by language (python, javascript)
    - `status`: Filter by status (pending, running, completed, failed, timeout)
    - `limit`: Maximum results (1-1000, default: 100)
    - `offset`: Results offset (default: 0)
    
    **Example Response**:
    ```json
    [
        {
            "execution_id": "exec_abc123",
            "status": "completed",
            "success": true,
            "language": "python",
            "execution_time": 0.15,
            "created_at": "2025-11-09T12:00:00Z"
        },
        ...
    ]
    ```
    """
    try:
        results = await service.list_executions(
            agent_id=agent_id,
            language=language,
            status=status,
            limit=limit,
            offset=offset
        )
        
        logger.info(f"📋 Listed {len(results)} executions")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Failed to list executions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/global", response_model=ExecutionStatistics)
async def get_statistics(
    service: CodeExecutionService = Depends(get_code_execution_service),
    current_user: dict = Depends(get_current_user)
):
    """
    Get global execution statistics
    
    **Example Response**:
    ```json
    {
        "total_executions": 1000,
        "successful_executions": 950,
        "failed_executions": 40,
        "timeout_executions": 10,
        "average_execution_time": 0.25,
        "average_memory_usage": 45.5,
        "executions_by_language": {
            "python": 600,
            "javascript": 400
        },
        "executions_by_agent": {
            "agent_123": 500,
            "agent_456": 500
        },
        "last_updated": "2025-11-09T12:00:00Z"
    }
    ```
    """
    try:
        stats = await service.get_statistics()
        
        logger.info(f"📊 Retrieved execution statistics")
        
        return stats
        
    except Exception as e:
        logger.error(f"❌ Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cleanup")
async def cleanup_containers(
    service: CodeExecutionService = Depends(get_code_execution_service),
    current_user: dict = Depends(get_current_user)
):
    """
    Cleanup stale containers (admin only)
    
    **Note**: This endpoint should be restricted to admin users only.
    
    **Example Response**:
    ```json
    {
        "containers_removed": 5,
        "message": "Cleaned up 5 stale containers"
    }
    ```
    """
    try:
        # TODO: Add admin check
        # if not current_user.get('is_admin'):
        #     raise HTTPException(status_code=403, detail="Admin access required")
        
        removed = await service.cleanup_containers()
        
        logger.info(f"🗑️ Cleaned up {removed} containers")
        
        return {
            "containers_removed": removed,
            "message": f"Cleaned up {removed} stale containers"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to cleanup containers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/check")
async def health_check(
    service: CodeExecutionService = Depends(get_code_execution_service)
):
    """
    Health check for code execution service
    
    **Example Response**:
    ```json
    {
        "healthy": true,
        "docker_connected": true,
        "images_available": {
            "python": true,
            "javascript": true
        },
        "total_executions": 1000,
        "success_rate": 95.0
    }
    ```
    """
    try:
        health = await service.health_check()
        
        if not health["healthy"]:
            raise HTTPException(status_code=503, detail="Service unhealthy")
        
        return health
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

