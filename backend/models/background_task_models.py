"""
Background Task Models for GRETA PAI

This module provides comprehensive models for background task processing,
including task queuing, status tracking, retry logic, and monitoring.
It extends the basic task models with background processing capabilities.

The module supports:
- Async task queuing with Redis
- Task status tracking and monitoring
- Retry logic with exponential backoff
- Email sending and notification tasks
- Task result persistence
- Performance metrics and analytics

Example:
    Creating a background task:
        >>> from models.background_task_models import BackgroundTask, TaskType
        >>> task = BackgroundTask(
        ...     task_type=TaskType.EMAIL_SENDING,
        ...     payload={"to": "user@example.com", "subject": "Welcome"},
        ...     priority=TaskPriority.HIGH
        ... )
        >>> await task_queue.enqueue(task)

    Monitoring task status:
        >>> status = await task_queue.get_task_status(task.task_id)
        >>> print(f"Task {task.task_id}: {status.status}")

Attributes:
    TaskType: Extended enumeration for background task types
    BackgroundTask: Main task model with background processing features
    TaskQueue: Task queue management model
    TaskWorker: Worker configuration model
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum

from .task_models import TaskStatus, TaskPriority, TaskMetadata, TaskResult


class BackgroundTaskType(str, Enum):
    """
    Background task type enumeration.
    
    Defines the types of tasks that can be processed in the background,
    each with specific handling requirements and retry policies.
    """
    EMAIL_SENDING = "email_sending"
    FILE_PROCESSING = "file_processing"
    DATA_EXPORT = "data_export"
    REPORT_GENERATION = "report_generation"
    NOTIFICATION_DELIVERY = "notification_delivery"
    CACHE_WARMING = "cache_warming"
    DATABASE_CLEANUP = "database_cleanup"
    ANALYTICS_PROCESSING = "analytics_processing"
    AI_MODEL_TRAINING = "ai_model_training"
    WEBHOOK_DELIVERY = "webhook_delivery"


class RetryPolicy(BaseModel):
    """
    Task retry policy configuration.
    
    Defines how failed tasks should be retried, including maximum attempts,
    backoff strategy, and conditions for retry.
    
    Attributes:
        max_attempts: Maximum number of retry attempts
        backoff_factor: Exponential backoff multiplier
        initial_delay: Initial delay before first retry (seconds)
        max_delay: Maximum delay between retries (seconds)
        retry_on_exceptions: List of exception types to retry on
        
    Example:
        >>> policy = RetryPolicy(
        ...     max_attempts=3,
        ...     backoff_factor=2.0,
        ...     initial_delay=60,
        ...     max_delay=3600
        ... )
    """
    max_attempts: int = Field(default=3, ge=1, le=10)
    backoff_factor: float = Field(default=2.0, ge=1.0, le=10.0)
    initial_delay: int = Field(default=60, ge=1)  # seconds
    max_delay: int = Field(default=3600, ge=60)  # seconds
    retry_on_exceptions: List[str] = Field(default_factory=lambda: [
        "ConnectionError", "TimeoutError", "TemporaryFailure"
    ])
    
    def calculate_delay(self, attempt: int) -> int:
        """
        Calculate delay for given attempt number.
        
        Args:
            attempt: Current attempt number (1-based)
            
        Returns:
            int: Delay in seconds before next retry
        """
        delay = self.initial_delay * (self.backoff_factor ** (attempt - 1))
        return min(int(delay), self.max_delay)


class TaskProgress(BaseModel):
    """
    Task progress tracking model.
    
    Tracks the progress of long-running background tasks with
    percentage completion and status messages.
    
    Attributes:
        percentage: Completion percentage (0-100)
        current_step: Current processing step
        total_steps: Total number of steps
        message: Current status message
        details: Additional progress details
        
    Example:
        >>> progress = TaskProgress(
        ...     percentage=75,
        ...     current_step=3,
        ...     total_steps=4,
        ...     message="Processing data export..."
        ... )
    """
    percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    current_step: int = Field(default=0, ge=0)
    total_steps: int = Field(default=1, ge=1)
    message: str = Field(default="")
    details: Dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('percentage', pre=True)
    def calculate_percentage(cls, v, values):
        """Auto-calculate percentage from steps if not provided"""
        if v == 0.0 and 'current_step' in values and 'total_steps' in values:
            current = values.get('current_step', 0)
            total = values.get('total_steps', 1)
            return (current / total) * 100 if total > 0 else 0.0
        return v


class BackgroundTask(BaseModel):
    """
    Background task model with comprehensive tracking and retry capabilities.
    
    This model represents a task that can be processed asynchronously in the
    background with full status tracking, retry logic, and monitoring.
    
    Attributes:
        task_id: Unique task identifier
        task_type: Type of background task
        payload: Task-specific data and parameters
        status: Current task status
        priority: Task priority level
        retry_policy: Retry configuration
        progress: Task progress tracking
        metadata: Additional task metadata
        
    Example:
        >>> task = BackgroundTask(
        ...     task_type=BackgroundTaskType.EMAIL_SENDING,
        ...     payload={
        ...         "to": "user@example.com",
        ...         "subject": "Welcome to GRETA PAI",
        ...         "template": "welcome_email"
        ...     },
        ...     priority=TaskPriority.HIGH
        ... )
    """
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type: BackgroundTaskType
    payload: Dict[str, Any] = Field(default_factory=dict)
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM)
    
    # Retry and error handling
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    current_attempt: int = Field(default=0)
    last_error: Optional[str] = None
    error_details: Dict[str, Any] = Field(default_factory=dict)
    
    # Progress tracking
    progress: TaskProgress = Field(default_factory=TaskProgress)
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    scheduled_for: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # Results and metadata
    result: Optional[TaskResult] = None
    metadata: TaskMetadata = Field(default_factory=TaskMetadata)
    
    # Worker assignment
    worker_id: Optional[str] = None
    queue_name: str = Field(default="default")
    
    class Config:
        """Pydantic configuration"""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def should_retry(self) -> bool:
        """
        Check if task should be retried based on current state.
        
        Returns:
            bool: True if task should be retried, False otherwise
        """
        if self.status != TaskStatus.FAILED:
            return False
        
        if self.current_attempt >= self.retry_policy.max_attempts:
            return False
        
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        
        return True
    
    def get_next_retry_delay(self) -> int:
        """
        Get delay before next retry attempt.
        
        Returns:
            int: Delay in seconds
        """
        return self.retry_policy.calculate_delay(self.current_attempt + 1)
    
    def mark_started(self, worker_id: str) -> None:
        """
        Mark task as started by a worker.
        
        Args:
            worker_id: ID of the worker processing the task
        """
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.utcnow()
        self.worker_id = worker_id
    
    def mark_completed(self, result: TaskResult) -> None:
        """
        Mark task as completed with result.
        
        Args:
            result: Task execution result
        """
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.result = result
        self.progress.percentage = 100.0
        self.progress.message = "Task completed successfully"
    
    def mark_failed(self, error: str, error_details: Dict[str, Any] = None) -> None:
        """
        Mark task as failed with error information.
        
        Args:
            error: Error message
            error_details: Additional error details
        """
        self.status = TaskStatus.FAILED
        self.last_error = error
        self.error_details = error_details or {}
        self.current_attempt += 1
    
    def update_progress(self, percentage: float = None, message: str = None, 
                       current_step: int = None, details: Dict[str, Any] = None) -> None:
        """
        Update task progress information.
        
        Args:
            percentage: Completion percentage
            message: Progress message
            current_step: Current step number
            details: Additional progress details
        """
        if percentage is not None:
            self.progress.percentage = percentage
        if message is not None:
            self.progress.message = message
        if current_step is not None:
            self.progress.current_step = current_step
        if details is not None:
            self.progress.details.update(details)
        
        self.progress.updated_at = datetime.utcnow()


class EmailTask(BackgroundTask):
    """
    Specialized background task for email sending.
    
    Extends BackgroundTask with email-specific validation and helpers.
    
    Example:
        >>> email_task = EmailTask.create_email_task(
        ...     to="user@example.com",
        ...     subject="Welcome",
        ...     template="welcome_email",
        ...     template_data={"name": "John"}
        ... )
    """
    
    @classmethod
    def create_email_task(cls, to: str, subject: str, template: str = None, 
                         body: str = None, template_data: Dict[str, Any] = None,
                         priority: TaskPriority = TaskPriority.MEDIUM) -> 'EmailTask':
        """
        Create an email sending task.
        
        Args:
            to: Recipient email address
            subject: Email subject
            template: Email template name
            body: Email body (if not using template)
            template_data: Data for template rendering
            priority: Task priority
            
        Returns:
            EmailTask: Configured email task
            
        Raises:
            ValueError: If neither template nor body is provided
        """
        if not template and not body:
            raise ValueError("Either template or body must be provided")
        
        payload = {
            "to": to,
            "subject": subject,
            "template": template,
            "body": body,
            "template_data": template_data or {}
        }
        
        return cls(
            task_type=BackgroundTaskType.EMAIL_SENDING,
            payload=payload,
            priority=priority,
            retry_policy=RetryPolicy(max_attempts=3, initial_delay=30)
        )


class TaskQueueStats(BaseModel):
    """
    Task queue statistics model.
    
    Provides metrics and analytics for task queue performance monitoring.
    
    Attributes:
        total_tasks: Total number of tasks
        pending_tasks: Number of pending tasks
        in_progress_tasks: Number of tasks currently being processed
        completed_tasks: Number of completed tasks
        failed_tasks: Number of failed tasks
        average_processing_time: Average task processing time in seconds
        queue_depth: Current queue depth
        worker_count: Number of active workers
        
    Example:
        >>> stats = TaskQueueStats(
        ...     total_tasks=1000,
        ...     pending_tasks=50,
        ...     completed_tasks=900,
        ...     failed_tasks=50
        ... )
    """
    total_tasks: int = 0
    pending_tasks: int = 0
    in_progress_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0
    
    average_processing_time: float = 0.0  # seconds
    queue_depth: int = 0
    worker_count: int = 0
    
    # Performance metrics
    tasks_per_minute: float = 0.0
    success_rate: float = 0.0
    retry_rate: float = 0.0
    
    # Timing
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    def calculate_success_rate(self) -> float:
        """Calculate task success rate percentage"""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100
    
    def calculate_retry_rate(self) -> float:
        """Calculate task retry rate percentage"""
        if self.total_tasks == 0:
            return 0.0
        return (self.failed_tasks / self.total_tasks) * 100
