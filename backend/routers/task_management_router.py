"""
Task Management Router for CPAS4
RESTful API for task persistence and lifecycle management
"""

import logging
from fastapi import APIRouter, HTTPException, Depends
from typing import List
from database import get_database, Database
from services.task_persistence_service import TaskPersistenceService
from models.task_models import (
    Task, TaskCreate, TaskUpdate, TaskQuery, TaskStats,
    TaskStatus, TaskResult
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/task-management",
    tags=["Task Management"]
)

# Global service instance
task_service: TaskPersistenceService = None


async def get_task_service() -> TaskPersistenceService:
    """Get task persistence service"""
    global task_service
    if task_service is None:
        db = await get_database()
        task_service = TaskPersistenceService(db)
    return task_service


@router.post("/tasks", response_model=Task)
async def create_task(
    task_create: TaskCreate,
    service: TaskPersistenceService = Depends(get_task_service)
):
    """
    Create a new task
    
    **Parameters:**
    - **title**: Task title
    - **description**: Task description
    - **priority**: Task priority (low, medium, high, urgent)
    - **task_type**: Task type (agent_execution, workflow, scheduled, manual, automated)
    - **input_data**: Task input data
    - **metadata**: Task metadata (agent_type, user_id, tags, etc.)
    
    **Returns:**
    - Created task object with task_id
    """
    try:
        task = await service.create_task(task_create)
        return task
    except Exception as e:
        logger.error(f"Error creating task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}", response_model=Task)
async def get_task(
    task_id: str,
    service: TaskPersistenceService = Depends(get_task_service)
):
    """
    Get task by ID
    
    **Parameters:**
    - **task_id**: Task ID
    
    **Returns:**
    - Task object
    """
    task = await service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return task


@router.patch("/tasks/{task_id}", response_model=Task)
async def update_task(
    task_id: str,
    task_update: TaskUpdate,
    service: TaskPersistenceService = Depends(get_task_service)
):
    """
    Update task
    
    **Parameters:**
    - **task_id**: Task ID
    - **task_update**: Fields to update
    
    **Returns:**
    - Updated task object
    """
    task = await service.update_task(task_id, task_update)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return task


@router.delete("/tasks/{task_id}")
async def delete_task(
    task_id: str,
    service: TaskPersistenceService = Depends(get_task_service)
):
    """
    Delete task
    
    **Parameters:**
    - **task_id**: Task ID
    
    **Returns:**
    - Success message
    """
    success = await service.delete_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return {"message": f"Task {task_id} deleted successfully"}


@router.post("/tasks/query", response_model=List[Task])
async def query_tasks(
    query: TaskQuery,
    service: TaskPersistenceService = Depends(get_task_service)
):
    """
    Query tasks with filters
    
    **Parameters:**
    - **status**: Filter by status
    - **priority**: Filter by priority
    - **task_type**: Filter by task type
    - **agent_type**: Filter by agent type
    - **user_id**: Filter by user ID
    - **tags**: Filter by tags
    - **limit**: Maximum number of results
    - **skip**: Number of results to skip
    
    **Returns:**
    - List of matching tasks
    """
    try:
        tasks = await service.query_tasks(query)
        return tasks
    except Exception as e:
        logger.error(f"Error querying tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks", response_model=List[Task])
async def list_tasks(
    status: str = None,
    priority: str = None,
    task_type: str = None,
    agent_type: str = None,
    user_id: str = None,
    limit: int = 100,
    service: TaskPersistenceService = Depends(get_task_service)
):
    """
    List tasks with optional filters
    
    **Query Parameters:**
    - **status**: Filter by status
    - **priority**: Filter by priority
    - **task_type**: Filter by task type
    - **agent_type**: Filter by agent type
    - **user_id**: Filter by user ID
    - **limit**: Maximum number of results (default: 100)
    
    **Returns:**
    - List of tasks
    """
    try:
        query = TaskQuery(
            status=status,
            priority=priority,
            task_type=task_type,
            agent_type=agent_type,
            user_id=user_id,
            limit=limit
        )
        tasks = await service.query_tasks(query)
        return tasks
    except Exception as e:
        logger.error(f"Error listing tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=TaskStats)
async def get_task_stats(
    service: TaskPersistenceService = Depends(get_task_service)
):
    """
    Get task statistics
    
    **Returns:**
    - Task statistics including counts by status, priority, type, agent
    - Average execution time
    - Success rate
    """
    try:
        stats = await service.get_task_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting task stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/{task_id}/start")
async def start_task(
    task_id: str,
    service: TaskPersistenceService = Depends(get_task_service)
):
    """
    Start task execution
    
    **Parameters:**
    - **task_id**: Task ID
    
    **Returns:**
    - Updated task with status IN_PROGRESS
    """
    task_update = TaskUpdate(status=TaskStatus.IN_PROGRESS)
    task = await service.update_task(task_id, task_update)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return task


@router.post("/tasks/{task_id}/complete")
async def complete_task(
    task_id: str,
    result: TaskResult,
    service: TaskPersistenceService = Depends(get_task_service)
):
    """
    Mark task as completed
    
    **Parameters:**
    - **task_id**: Task ID
    - **result**: Task execution result
    
    **Returns:**
    - Updated task with status COMPLETED
    """
    task_update = TaskUpdate(
        status=TaskStatus.COMPLETED,
        result=result,
        progress_percentage=100.0
    )
    task = await service.update_task(task_id, task_update)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return task


@router.post("/tasks/{task_id}/fail")
async def fail_task(
    task_id: str,
    error: str,
    service: TaskPersistenceService = Depends(get_task_service)
):
    """
    Mark task as failed
    
    **Parameters:**
    - **task_id**: Task ID
    - **error**: Error message
    
    **Returns:**
    - Updated task with status FAILED
    """
    result = TaskResult(success=False, error=error)
    task_update = TaskUpdate(status=TaskStatus.FAILED, result=result)
    task = await service.update_task(task_id, task_update)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return task


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(
    task_id: str,
    service: TaskPersistenceService = Depends(get_task_service)
):
    """
    Cancel task
    
    **Parameters:**
    - **task_id**: Task ID
    
    **Returns:**
    - Updated task with status CANCELLED
    """
    task_update = TaskUpdate(status=TaskStatus.CANCELLED)
    task = await service.update_task(task_id, task_update)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return task


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Task Management",
        "features": [
            "Task CRUD operations",
            "Task lifecycle management",
            "Task querying and filtering",
            "Task statistics",
            "Status transitions",
            "Progress tracking"
        ]
    }

