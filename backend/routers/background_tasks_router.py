"""
Background Tasks Router for GRETA PAI

This router provides API endpoints for managing background tasks including
task creation, status monitoring, queue statistics, and worker management.
It integrates with the task queue service to provide comprehensive task management.

The router provides endpoints for:
- Task creation and enqueueing
- Task status monitoring and tracking
- Queue statistics and analytics
- Worker management and monitoring
- Email sending via background tasks
- Task retry and cancellation
- Dead letter queue management

Example:
    Creating a background task:
        POST /background-tasks/email
        {
            "to": "user@example.com",
            "subject": "Welcome",
            "template": "welcome_email",
            "template_data": {"name": "John"}
        }

    Monitoring task status:
        GET /background-tasks/{task_id}/status

    Getting queue statistics:
        GET /background-tasks/stats

Attributes:
    router (APIRouter): FastAPI router instance
    task_queue_service (TaskQueueService): Background task queue service
    email_service (EmailService): Email service for email tasks
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from models.background_task_models import (
    BackgroundTask, BackgroundTaskType, TaskStatus, TaskPriority,
    EmailTask, TaskQueueStats
)
from models.api_models import StandardResponse
from services.task_queue_service import TaskQueueService
from services.email_service import EmailService
from utils.custom_exceptions import TaskQueueError, TaskNotFoundError
from middleware.auth import get_current_user
from models.user_models import UserInDB

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/background-tasks", tags=["Background Tasks"])

# Global service instances
task_queue_service: Optional[TaskQueueService] = None
email_service: Optional[EmailService] = None


async def get_task_queue_service() -> TaskQueueService:
    """
    Get task queue service dependency.
    
    Returns:
        TaskQueueService: Initialized task queue service
        
    Raises:
        HTTPException: If service initialization fails
    """
    global task_queue_service
    
    if task_queue_service is None:
        try:
            from database import database
            task_queue_service = TaskQueueService(database)
            await task_queue_service.initialize()
        except Exception as e:
            logger.error(f"❌ Failed to initialize task queue service: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Task queue service unavailable"
            )
    
    return task_queue_service


async def get_email_service() -> EmailService:
    """
    Get email service dependency.
    
    Returns:
        EmailService: Initialized email service
        
    Raises:
        HTTPException: If service initialization fails
    """
    global email_service
    
    if email_service is None:
        try:
            email_service = EmailService()
            await email_service.initialize()
        except Exception as e:
            logger.error(f"❌ Failed to initialize email service: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Email service unavailable"
            )
    
    return email_service


@router.post("/email", 
             response_model=StandardResponse,
             summary="Send Email via Background Task",
             description="""
**Send email asynchronously via background task queue**

This endpoint creates a background task for sending emails, allowing for:
- Asynchronous email delivery
- Retry logic for failed sends
- Template-based email rendering
- Priority-based processing
- Delivery tracking and monitoring

**Use Cases:**
- Welcome emails for new users
- Password reset notifications
- System alerts and notifications
- Marketing campaigns
- Transactional emails

**Features:**
- Template rendering with Jinja2
- HTML and plain text support
- Retry logic with exponential backoff
- Delivery status tracking
- Priority queue processing
             """,
             responses={
                 200: {
                     "description": "Email task created successfully",
                     "content": {
                         "application/json": {
                             "example": {
                                 "success": True,
                                 "message": "Email task created successfully",
                                 "data": {
                                     "task_id": "123e4567-e89b-12d3-a456-426614174000",
                                     "status": "pending",
                                     "priority": "high"
                                 }
                             }
                         }
                     }
                 },
                 400: {"description": "Invalid email parameters"},
                 503: {"description": "Email service unavailable"}
             })
async def send_email_task(
    email_data: Dict[str, Any],
    queue_service: TaskQueueService = Depends(get_task_queue_service),
    email_svc: EmailService = Depends(get_email_service),
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Create background task for sending email.
    
    Args:
        email_data: Email parameters (to, subject, template, etc.)
        queue_service: Task queue service dependency
        email_svc: Email service dependency
        current_user: Current authenticated user
        
    Returns:
        StandardResponse: Task creation result with task ID
    """
    try:
        # Validate required fields
        required_fields = ["to", "subject"]
        for field in required_fields:
            if field not in email_data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Missing required field: {field}"
                )
        
        # Create email task
        priority = TaskPriority(email_data.get("priority", "medium"))
        
        email_task = EmailTask.create_email_task(
            to=email_data["to"],
            subject=email_data["subject"],
            template=email_data.get("template"),
            body=email_data.get("body"),
            template_data=email_data.get("template_data", {}),
            priority=priority
        )
        
        # Add user context to metadata
        email_task.metadata.user_id = current_user.user_id
        
        # Enqueue task
        task_id = await queue_service.enqueue_task(email_task)
        
        logger.info(f"📧 Email task {task_id} created for {email_data['to']}")
        
        return StandardResponse(
            success=True,
            message="Email task created successfully",
            data={
                "task_id": task_id,
                "status": email_task.status.value,
                "priority": email_task.priority.value,
                "recipient": email_data["to"],
                "subject": email_data["subject"]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to create email task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create email task: {str(e)}"
        )


@router.get("/{task_id}/status",
            response_model=StandardResponse,
            summary="Get Task Status",
            description="""
**Get detailed status information for a background task**

Returns comprehensive information about a task including:
- Current status (pending, in_progress, completed, failed)
- Progress information with percentage completion
- Error details if task failed
- Retry information and next attempt timing
- Worker assignment and processing details
- Task metadata and timing information

**Task Statuses:**
- `pending`: Task is queued and waiting for processing
- `in_progress`: Task is currently being processed by a worker
- `completed`: Task completed successfully
- `failed`: Task failed (may be retried)
- `cancelled`: Task was cancelled by user or system
            """,
            responses={
                200: {
                    "description": "Task status retrieved successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "success": True,
                                "message": "Task status retrieved",
                                "data": {
                                    "task_id": "123e4567-e89b-12d3-a456-426614174000",
                                    "status": "completed",
                                    "progress": {"percentage": 100, "message": "Email sent successfully"},
                                    "created_at": "2024-01-01T12:00:00Z",
                                    "completed_at": "2024-01-01T12:00:30Z"
                                }
                            }
                        }
                    }
                },
                404: {"description": "Task not found"}
            })
async def get_task_status(
    task_id: str,
    queue_service: TaskQueueService = Depends(get_task_queue_service),
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Get status of a background task.
    
    Args:
        task_id: Task identifier
        queue_service: Task queue service dependency
        current_user: Current authenticated user
        
    Returns:
        StandardResponse: Task status information
    """
    try:
        task = await queue_service.get_task(task_id)
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        
        # Check if user has access to this task
        if task.metadata.user_id != current_user.user_id and not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this task"
            )
        
        return StandardResponse(
            success=True,
            message="Task status retrieved successfully",
            data={
                "task_id": task.task_id,
                "task_type": task.task_type.value,
                "status": task.status.value,
                "priority": task.priority.value,
                "progress": {
                    "percentage": task.progress.percentage,
                    "message": task.progress.message,
                    "current_step": task.progress.current_step,
                    "total_steps": task.progress.total_steps
                },
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "worker_id": task.worker_id,
                "current_attempt": task.current_attempt,
                "max_attempts": task.retry_policy.max_attempts,
                "last_error": task.last_error,
                "result": task.result.dict() if task.result else None
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get task status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task status: {str(e)}"
        )


@router.get("/stats",
            response_model=StandardResponse,
            summary="Get Queue Statistics",
            description="""
**Get comprehensive task queue statistics and metrics**

Returns detailed analytics about the task queue performance including:
- Task counts by status (pending, completed, failed, etc.)
- Queue depth and processing metrics
- Worker information and capacity
- Success and retry rates
- Performance metrics and trends

**Metrics Included:**
- Total tasks processed
- Current queue depth
- Active worker count
- Average processing time
- Success rate percentage
- Retry rate percentage
- Tasks per minute throughput

This endpoint is useful for monitoring system performance and identifying bottlenecks.
            """,
            responses={
                200: {
                    "description": "Queue statistics retrieved successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "success": True,
                                "message": "Queue statistics retrieved",
                                "data": {
                                    "total_tasks": 1000,
                                    "pending_tasks": 50,
                                    "completed_tasks": 900,
                                    "failed_tasks": 50,
                                    "success_rate": 90.0,
                                    "worker_count": 3
                                }
                            }
                        }
                    }
                }
            })
async def get_queue_stats(
    queue_service: TaskQueueService = Depends(get_task_queue_service),
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Get task queue statistics.
    
    Args:
        queue_service: Task queue service dependency
        current_user: Current authenticated user
        
    Returns:
        StandardResponse: Queue statistics and metrics
    """
    try:
        stats = await queue_service.get_queue_stats()
        
        return StandardResponse(
            success=True,
            message="Queue statistics retrieved successfully",
            data=stats.dict()
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to get queue stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get queue statistics: {str(e)}"
        )


@router.post("/workers/start",
             response_model=StandardResponse,
             summary="Start Background Worker",
             description="""
**Start a new background task worker**

Creates and starts a new worker process for handling background tasks.
Workers continuously poll the task queues and process tasks based on priority.

**Worker Features:**
- Priority-based task processing
- Automatic retry logic for failed tasks
- Graceful error handling and recovery
- Task progress tracking and monitoring
- Dead letter queue for permanently failed tasks

**Note:** This endpoint requires admin privileges.
             """)
async def start_worker(
    worker_config: Dict[str, Any],
    queue_service: TaskQueueService = Depends(get_task_queue_service),
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Start a background task worker.
    
    Args:
        worker_config: Worker configuration (worker_id, queue_names)
        queue_service: Task queue service dependency
        current_user: Current authenticated user
        
    Returns:
        StandardResponse: Worker start result
    """
    # Check admin privileges
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    
    try:
        worker_id = worker_config.get("worker_id", f"worker_{datetime.utcnow().timestamp()}")
        queue_names = worker_config.get("queue_names")
        
        worker = await queue_service.start_worker(worker_id, queue_names)
        
        return StandardResponse(
            success=True,
            message=f"Worker {worker_id} started successfully",
            data={
                "worker_id": worker_id,
                "queue_names": queue_names or list(queue_service.priority_queues.values()),
                "status": "running"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start worker: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start worker: {str(e)}"
        )
