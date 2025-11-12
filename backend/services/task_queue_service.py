"""
Background Task Queue Service for GRETA PAI

This service provides comprehensive background task processing capabilities
using Redis as the message broker and MongoDB for task persistence.
It supports async task queuing, worker management, retry logic, and monitoring.

The service provides:
- Async task queuing with priority support
- Distributed worker management
- Task status tracking and monitoring
- Retry logic with exponential backoff
- Task result persistence
- Performance metrics and analytics
- Dead letter queue for failed tasks

Example:
    Basic task queue usage:
        >>> from services.task_queue_service import TaskQueueService
        >>> from models.background_task_models import EmailTask
        >>> 
        >>> queue_service = TaskQueueService()
        >>> await queue_service.initialize()
        >>> 
        >>> # Enqueue email task
        >>> email_task = EmailTask.create_email_task(
        ...     to="user@example.com",
        ...     subject="Welcome",
        ...     template="welcome_email"
        ... )
        >>> await queue_service.enqueue_task(email_task)
        >>> 
        >>> # Process tasks
        >>> await queue_service.start_worker("email_worker")

    Task monitoring:
        >>> stats = await queue_service.get_queue_stats()
        >>> print(f"Pending tasks: {stats.pending_tasks}")
        >>> 
        >>> task_status = await queue_service.get_task_status(task_id)
        >>> print(f"Task status: {task_status.status}")

Attributes:
    redis_client (Redis): Redis client for task queuing
    db (AsyncIOMotorDatabase): MongoDB database for task persistence
    workers (Dict): Active worker instances
    logger (logging.Logger): Service logger
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from motor.motor_asyncio import AsyncIOMotorDatabase

from models.background_task_models import (
    BackgroundTask, BackgroundTaskType, TaskStatus, TaskPriority,
    TaskQueueStats, EmailTask, RetryPolicy
)
from utils.cache import get_redis_client
from utils.custom_exceptions import (
    TaskQueueError, TaskNotFoundError, WorkerError
)
from config import get_settings

logger = logging.getLogger(__name__)


class TaskQueueService:
    """
    Background task queue service with Redis and MongoDB integration.
    
    This service manages the complete lifecycle of background tasks including
    queuing, processing, monitoring, and persistence.
    
    Attributes:
        redis_client: Redis client for task queuing
        db: MongoDB database for task persistence
        workers: Dictionary of active workers
        task_handlers: Registry of task type handlers
        
    Example:
        >>> queue_service = TaskQueueService()
        >>> await queue_service.initialize()
        >>> await queue_service.enqueue_task(task)
    """
    
    def __init__(self, database: AsyncIOMotorDatabase = None):
        """
        Initialize task queue service.
        
        Args:
            database: MongoDB database instance
        """
        self.redis_client = None
        self.db = database
        self.workers: Dict[str, 'TaskWorker'] = {}
        self.task_handlers: Dict[BackgroundTaskType, Callable] = {}
        self.settings = get_settings()
        self._running = False
        
        # Queue configuration
        self.default_queue = "greta_tasks"
        self.priority_queues = {
            TaskPriority.URGENT: "greta_tasks_urgent",
            TaskPriority.HIGH: "greta_tasks_high", 
            TaskPriority.MEDIUM: "greta_tasks_medium",
            TaskPriority.LOW: "greta_tasks_low"
        }
        self.dead_letter_queue = "greta_tasks_dlq"
        
        # Register default task handlers
        self._register_default_handlers()
    
    async def initialize(self) -> None:
        """
        Initialize the task queue service.
        
        Sets up Redis connection, MongoDB collections, and task handlers.
        
        Raises:
            TaskQueueError: If initialization fails
        """
        try:
            # Initialize Redis connection
            self.redis_client = await get_redis_client()
            logger.info("✅ Task queue Redis connection established")
            
            # Initialize MongoDB collections
            if self.db:
                await self._initialize_collections()
                logger.info("✅ Task queue MongoDB collections initialized")
            
            # Test connections
            await self._test_connections()
            
            self._running = True
            logger.info("✅ Task queue service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Task queue initialization failed: {e}")
            raise TaskQueueError(f"Failed to initialize task queue: {e}")
    
    async def _initialize_collections(self) -> None:
        """Initialize MongoDB collections for task persistence"""
        # Create tasks collection with indexes
        tasks_collection = self.db.background_tasks
        
        # Create indexes for efficient querying
        await tasks_collection.create_index("task_id", unique=True)
        await tasks_collection.create_index("status")
        await tasks_collection.create_index("task_type")
        await tasks_collection.create_index("priority")
        await tasks_collection.create_index("created_at")
        await tasks_collection.create_index("scheduled_for")
        await tasks_collection.create_index([("status", 1), ("priority", -1)])
        
        logger.info("✅ Task queue MongoDB indexes created")
    
    async def _test_connections(self) -> None:
        """Test Redis and MongoDB connections"""
        # Test Redis
        await self.redis_client.ping()
        
        # Test MongoDB
        if self.db:
            await self.db.command("ping")
    
    def _register_default_handlers(self) -> None:
        """Register default task handlers"""
        self.task_handlers[BackgroundTaskType.EMAIL_SENDING] = self._handle_email_task
        self.task_handlers[BackgroundTaskType.FILE_PROCESSING] = self._handle_file_processing
        self.task_handlers[BackgroundTaskType.CACHE_WARMING] = self._handle_cache_warming
        self.task_handlers[BackgroundTaskType.DATABASE_CLEANUP] = self._handle_database_cleanup
    
    async def enqueue_task(self, task: BackgroundTask, delay: int = 0) -> str:
        """
        Enqueue a background task for processing.
        
        Args:
            task: Background task to enqueue
            delay: Delay in seconds before task becomes available
            
        Returns:
            str: Task ID
            
        Raises:
            TaskQueueError: If task enqueueing fails
        """
        try:
            # Set scheduled time if delay is specified
            if delay > 0:
                task.scheduled_for = datetime.utcnow() + timedelta(seconds=delay)
            
            # Persist task to MongoDB
            if self.db:
                await self._persist_task(task)
            
            # Add to appropriate Redis queue
            queue_name = self.priority_queues.get(task.priority, self.default_queue)
            
            task_data = {
                "task_id": task.task_id,
                "task_type": task.task_type.value,
                "payload": task.payload,
                "priority": task.priority.value,
                "created_at": task.created_at.isoformat(),
                "scheduled_for": task.scheduled_for.isoformat() if task.scheduled_for else None
            }
            
            if delay > 0:
                # Schedule for later processing
                score = (datetime.utcnow() + timedelta(seconds=delay)).timestamp()
                await self.redis_client.zadd("greta_scheduled_tasks", {json.dumps(task_data): score})
            else:
                # Add to immediate processing queue
                await self.redis_client.lpush(queue_name, json.dumps(task_data))
            
            logger.info(f"✅ Task {task.task_id} enqueued to {queue_name}")
            return task.task_id
            
        except Exception as e:
            logger.error(f"❌ Failed to enqueue task {task.task_id}: {e}")
            raise TaskQueueError(f"Failed to enqueue task: {e}")
    
    async def _persist_task(self, task: BackgroundTask) -> None:
        """Persist task to MongoDB"""
        task_doc = task.dict()
        task_doc["_id"] = task.task_id
        
        await self.db.background_tasks.insert_one(task_doc)
    
    async def get_task(self, task_id: str) -> Optional[BackgroundTask]:
        """
        Retrieve a task by ID.
        
        Args:
            task_id: Task identifier
            
        Returns:
            BackgroundTask: Task instance or None if not found
        """
        if not self.db:
            return None
        
        task_doc = await self.db.background_tasks.find_one({"task_id": task_id})
        if task_doc:
            task_doc.pop("_id", None)  # Remove MongoDB _id
            return BackgroundTask(**task_doc)
        
        return None
    
    async def update_task_status(self, task_id: str, status: TaskStatus, 
                               error: str = None, result: Any = None) -> None:
        """
        Update task status in database.
        
        Args:
            task_id: Task identifier
            status: New task status
            error: Error message if failed
            result: Task result if completed
        """
        if not self.db:
            return
        
        update_data = {
            "status": status.value,
            "updated_at": datetime.utcnow()
        }
        
        if status == TaskStatus.IN_PROGRESS:
            update_data["started_at"] = datetime.utcnow()
        elif status == TaskStatus.COMPLETED:
            update_data["completed_at"] = datetime.utcnow()
            if result:
                update_data["result"] = result
        elif status == TaskStatus.FAILED:
            if error:
                update_data["last_error"] = error
        
        await self.db.background_tasks.update_one(
            {"task_id": task_id},
            {"$set": update_data}
        )
    
    async def get_queue_stats(self) -> TaskQueueStats:
        """
        Get task queue statistics.
        
        Returns:
            TaskQueueStats: Current queue statistics
        """
        stats = TaskQueueStats()
        
        if self.db:
            # Get task counts by status
            pipeline = [
                {"$group": {
                    "_id": "$status",
                    "count": {"$sum": 1}
                }}
            ]
            
            async for doc in self.db.background_tasks.aggregate(pipeline):
                status = doc["_id"]
                count = doc["count"]
                
                if status == TaskStatus.PENDING.value:
                    stats.pending_tasks = count
                elif status == TaskStatus.IN_PROGRESS.value:
                    stats.in_progress_tasks = count
                elif status == TaskStatus.COMPLETED.value:
                    stats.completed_tasks = count
                elif status == TaskStatus.FAILED.value:
                    stats.failed_tasks = count
                elif status == TaskStatus.CANCELLED.value:
                    stats.cancelled_tasks = count
            
            stats.total_tasks = (stats.pending_tasks + stats.in_progress_tasks + 
                               stats.completed_tasks + stats.failed_tasks + stats.cancelled_tasks)
        
        # Get Redis queue depths
        if self.redis_client:
            for queue_name in self.priority_queues.values():
                depth = await self.redis_client.llen(queue_name)
                stats.queue_depth += depth
        
        # Calculate metrics
        stats.success_rate = stats.calculate_success_rate()
        stats.retry_rate = stats.calculate_retry_rate()
        stats.worker_count = len(self.workers)
        
        return stats
    
    # Task handlers
    async def _handle_email_task(self, task: BackgroundTask) -> Any:
        """Handle email sending task"""
        payload = task.payload
        
        # Simulate email sending (replace with actual email service)
        await asyncio.sleep(1)  # Simulate processing time
        
        logger.info(f"📧 Email sent to {payload.get('to')} with subject: {payload.get('subject')}")
        
        return {
            "email_sent": True,
            "recipient": payload.get("to"),
            "subject": payload.get("subject"),
            "sent_at": datetime.utcnow().isoformat()
        }
    
    async def _handle_file_processing(self, task: BackgroundTask) -> Any:
        """Handle file processing task"""
        # Implement file processing logic
        await asyncio.sleep(2)  # Simulate processing
        return {"processed": True, "file_size": 1024}
    
    async def _handle_cache_warming(self, task: BackgroundTask) -> Any:
        """Handle cache warming task"""
        # Implement cache warming logic
        await asyncio.sleep(0.5)  # Simulate processing
        return {"cache_warmed": True, "entries": 100}
    
    async def _handle_database_cleanup(self, task: BackgroundTask) -> Any:
        """Handle database cleanup task"""
        # Implement database cleanup logic
        await asyncio.sleep(3)  # Simulate processing
        return {"cleanup_completed": True, "records_cleaned": 50}
    
    async def start_worker(self, worker_id: str, queue_names: List[str] = None) -> 'TaskWorker':
        """
        Start a background task worker.

        Args:
            worker_id: Unique worker identifier
            queue_names: List of queue names to process (default: all priority queues)

        Returns:
            TaskWorker: Started worker instance
        """
        if queue_names is None:
            queue_names = list(self.priority_queues.values())

        worker = TaskWorker(
            worker_id=worker_id,
            queue_service=self,
            queue_names=queue_names
        )

        await worker.start()
        self.workers[worker_id] = worker

        logger.info(f"✅ Worker {worker_id} started for queues: {queue_names}")
        return worker

    async def stop_worker(self, worker_id: str) -> None:
        """
        Stop a background task worker.

        Args:
            worker_id: Worker identifier to stop
        """
        if worker_id in self.workers:
            await self.workers[worker_id].stop()
            del self.workers[worker_id]
            logger.info(f"✅ Worker {worker_id} stopped")

    async def shutdown(self) -> None:
        """Shutdown the task queue service"""
        self._running = False

        # Stop all workers
        for worker in self.workers.values():
            await worker.stop()

        logger.info("✅ Task queue service shutdown complete")


class TaskWorker:
    """
    Background task worker for processing queued tasks.

    This worker continuously polls Redis queues for tasks and processes them
    using registered task handlers with retry logic and error handling.

    Attributes:
        worker_id: Unique worker identifier
        queue_service: Reference to task queue service
        queue_names: List of queue names to process
        running: Worker running state

    Example:
        >>> worker = TaskWorker("worker_1", queue_service, ["high_priority"])
        >>> await worker.start()
    """

    def __init__(self, worker_id: str, queue_service: TaskQueueService,
                 queue_names: List[str]):
        """
        Initialize task worker.

        Args:
            worker_id: Unique worker identifier
            queue_service: Task queue service instance
            queue_names: Queue names to process
        """
        self.worker_id = worker_id
        self.queue_service = queue_service
        self.queue_names = queue_names
        self.running = False
        self._task = None

    async def start(self) -> None:
        """Start the worker task processing loop"""
        if self.running:
            return

        self.running = True
        self._task = asyncio.create_task(self._process_loop())
        logger.info(f"🔄 Worker {self.worker_id} started processing")

    async def stop(self) -> None:
        """Stop the worker task processing loop"""
        self.running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info(f"⏹️ Worker {self.worker_id} stopped")

    async def _process_loop(self) -> None:
        """Main worker processing loop"""
        while self.running:
            try:
                # Process scheduled tasks first
                await self._process_scheduled_tasks()

                # Process immediate tasks
                task_data = await self._get_next_task()

                if task_data:
                    await self._process_task(task_data)
                else:
                    # No tasks available, wait before polling again
                    await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Worker {self.worker_id} error: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    async def _process_scheduled_tasks(self) -> None:
        """Process tasks scheduled for future execution"""
        redis_client = self.queue_service.redis_client
        current_time = datetime.utcnow().timestamp()

        # Get tasks scheduled for now or earlier
        scheduled_tasks = await redis_client.zrangebyscore(
            "greta_scheduled_tasks", 0, current_time, withscores=True
        )

        for task_json, score in scheduled_tasks:
            try:
                # Remove from scheduled set
                await redis_client.zrem("greta_scheduled_tasks", task_json)

                # Add to appropriate queue
                task_data = json.loads(task_json)
                priority = TaskPriority(task_data.get("priority", "medium"))
                queue_name = self.queue_service.priority_queues.get(priority,
                                                                   self.queue_service.default_queue)

                await redis_client.lpush(queue_name, task_json)

            except Exception as e:
                logger.error(f"❌ Failed to process scheduled task: {e}")

    async def _get_next_task(self) -> Optional[Dict[str, Any]]:
        """Get next task from queues (priority order)"""
        redis_client = self.queue_service.redis_client

        # Process queues in priority order
        for queue_name in self.queue_names:
            task_json = await redis_client.brpop(queue_name, timeout=1)
            if task_json:
                return json.loads(task_json[1])  # brpop returns (queue_name, value)

        return None

    async def _process_task(self, task_data: Dict[str, Any]) -> None:
        """Process a single task"""
        task_id = task_data.get("task_id")

        try:
            # Get full task from database
            task = await self.queue_service.get_task(task_id)
            if not task:
                logger.error(f"❌ Task {task_id} not found in database")
                return

            # Mark task as in progress
            task.mark_started(self.worker_id)
            await self.queue_service.update_task_status(task_id, TaskStatus.IN_PROGRESS)

            # Get task handler
            handler = self.queue_service.task_handlers.get(task.task_type)
            if not handler:
                raise ValueError(f"No handler registered for task type: {task.task_type}")

            # Execute task
            logger.info(f"🔄 Processing task {task_id} ({task.task_type})")
            result = await handler(task)

            # Mark task as completed
            from models.task_models import TaskResult
            task_result = TaskResult(success=True, data=result)
            task.mark_completed(task_result)

            await self.queue_service.update_task_status(
                task_id, TaskStatus.COMPLETED, result=result
            )

            logger.info(f"✅ Task {task_id} completed successfully")

        except Exception as e:
            logger.error(f"❌ Task {task_id} failed: {e}")

            # Handle task failure
            await self._handle_task_failure(task_id, str(e))

    async def _handle_task_failure(self, task_id: str, error: str) -> None:
        """Handle task failure with retry logic"""
        task = await self.queue_service.get_task(task_id)
        if not task:
            return

        task.mark_failed(error)

        if task.should_retry():
            # Schedule retry
            retry_delay = task.get_next_retry_delay()
            logger.info(f"🔄 Retrying task {task_id} in {retry_delay} seconds (attempt {task.current_attempt + 1})")

            # Re-enqueue with delay
            await self.queue_service.enqueue_task(task, delay=retry_delay)
        else:
            # Move to dead letter queue
            logger.error(f"💀 Task {task_id} moved to dead letter queue after {task.current_attempt} attempts")

            await self.queue_service.update_task_status(task_id, TaskStatus.FAILED, error=error)

            # Add to dead letter queue for manual inspection
            redis_client = self.queue_service.redis_client
            dlq_data = {
                "task_id": task_id,
                "error": error,
                "attempts": task.current_attempt,
                "failed_at": datetime.utcnow().isoformat()
            }
            await redis_client.lpush(self.queue_service.dead_letter_queue, json.dumps(dlq_data))
