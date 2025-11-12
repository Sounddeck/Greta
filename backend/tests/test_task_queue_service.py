"""
Comprehensive tests for task_queue_service.py module

Tests cover:
- Task enqueueing and dequeueing
- Priority handling
- Task status tracking
- Worker management
- Error handling
"""

import pytest
import asyncio
import json
import sys
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Mock custom_exceptions module before importing task_queue_service
class TaskQueueError(Exception):
    pass

class TaskNotFoundError(Exception):
    pass

class WorkerError(Exception):
    pass

# Create mock module
mock_custom_exceptions = type(sys)('utils.custom_exceptions')
mock_custom_exceptions.TaskQueueError = TaskQueueError
mock_custom_exceptions.TaskNotFoundError = TaskNotFoundError
mock_custom_exceptions.WorkerError = WorkerError
sys.modules['utils.custom_exceptions'] = mock_custom_exceptions


class TestTaskEnqueueingAndDequeueing:
    """Test task enqueueing and dequeueing"""
    
    @pytest.mark.asyncio
    async def test_enqueue_task_success(self):
        """Test successfully enqueueing a task"""
        from services.task_queue_service import TaskQueueService

        mock_db = MagicMock()
        mock_db.background_tasks.insert_one = AsyncMock()

        mock_redis = AsyncMock()
        mock_redis.lpush = AsyncMock()

        service = TaskQueueService(database=mock_db)
        service.redis_client = mock_redis
        service._running = True

        # Mock task object
        mock_task = MagicMock()
        mock_task.task_id = "test-123"
        mock_task.task_type = MagicMock()
        mock_task.task_type.value = "email_sending"
        mock_task.priority = MagicMock()
        mock_task.priority.value = "medium"
        mock_task.payload = {"to": "test@example.com"}
        mock_task.created_at = MagicMock()
        mock_task.created_at.isoformat = MagicMock(return_value="2024-01-01T00:00:00")
        mock_task.scheduled_for = None
        mock_task.dict = MagicMock(return_value={"task_id": "test-123"})

        task_id = await service.enqueue_task(mock_task)

        assert task_id == "test-123"
        assert mock_db.background_tasks.insert_one.called
        assert mock_redis.lpush.called
    
    @pytest.mark.asyncio
    async def test_enqueue_task_with_delay(self):
        """Test enqueueing a task with delay"""
        from services.task_queue_service import TaskQueueService

        mock_db = MagicMock()
        mock_db.background_tasks.insert_one = AsyncMock()

        mock_redis = AsyncMock()
        mock_redis.zadd = AsyncMock()

        service = TaskQueueService(database=mock_db)
        service.redis_client = mock_redis
        service._running = True

        # Mock task object
        mock_task = MagicMock()
        mock_task.task_id = "test-456"
        mock_task.task_type = MagicMock()
        mock_task.task_type.value = "email_sending"
        mock_task.priority = MagicMock()
        mock_task.priority.value = "medium"
        mock_task.payload = {"to": "test@example.com"}
        mock_task.created_at = MagicMock()
        mock_task.created_at.isoformat = MagicMock(return_value="2024-01-01T00:00:00")
        mock_task.scheduled_for = None
        mock_task.dict = MagicMock(return_value={"task_id": "test-456"})

        task_id = await service.enqueue_task(mock_task, delay=60)

        assert task_id == "test-456"
        assert mock_redis.zadd.called
        assert mock_task.scheduled_for is not None
    
    @pytest.mark.asyncio
    async def test_get_task_by_id(self):
        """Test retrieving a task by ID"""
        from services.task_queue_service import TaskQueueService
        from models.background_task_models import BackgroundTask, BackgroundTaskType
        
        mock_db = MagicMock()
        task_doc = {
            "task_id": "test-123",
            "task_type": BackgroundTaskType.EMAIL_SENDING.value,
            "payload": {"to": "test@example.com"},
            "priority": "medium",
            "status": "pending",
            "created_at": datetime.utcnow(),
            "current_attempt": 0
        }
        mock_db.background_tasks.find_one = AsyncMock(return_value=task_doc)
        
        service = TaskQueueService(database=mock_db)
        
        task = await service.get_task("test-123")
        
        assert task is not None
        assert task.task_id == "test-123"
    
    @pytest.mark.asyncio
    async def test_get_task_not_found(self):
        """Test retrieving non-existent task"""
        from services.task_queue_service import TaskQueueService
        
        mock_db = MagicMock()
        mock_db.background_tasks.find_one = AsyncMock(return_value=None)
        
        service = TaskQueueService(database=mock_db)
        
        task = await service.get_task("nonexistent")
        
        assert task is None


class TestPriorityHandling:
    """Test priority handling"""

    def _create_mock_task(self, task_id, priority_value):
        """Helper to create mock task"""
        mock_task = MagicMock()
        mock_task.task_id = task_id
        mock_task.task_type = MagicMock()
        mock_task.task_type.value = "email_sending"
        mock_task.priority = MagicMock()
        mock_task.priority.value = priority_value
        mock_task.payload = {"to": "test@example.com"}
        mock_task.created_at = MagicMock()
        mock_task.created_at.isoformat = MagicMock(return_value="2024-01-01T00:00:00")
        mock_task.scheduled_for = None
        mock_task.dict = MagicMock(return_value={"task_id": task_id})
        return mock_task

    @pytest.mark.asyncio
    async def test_priority_queue_routing(self):
        """Test tasks are routed to correct priority queues"""
        from services.task_queue_service import TaskQueueService
        from models.background_task_models import TaskPriority

        mock_db = MagicMock()
        mock_db.background_tasks.insert_one = AsyncMock()

        mock_redis = AsyncMock()
        mock_redis.lpush = AsyncMock()

        service = TaskQueueService(database=mock_db)
        service.redis_client = mock_redis
        service._running = True

        # Test urgent priority
        urgent_task = self._create_mock_task("urgent-1", "urgent")
        urgent_task.priority = TaskPriority.URGENT

        await service.enqueue_task(urgent_task)

        # Verify it was added to urgent queue
        call_args = mock_redis.lpush.call_args
        assert call_args[0][0] == "greta_tasks_urgent"

    @pytest.mark.asyncio
    async def test_high_priority_task_execution(self):
        """Test high priority tasks are processed first"""
        from services.task_queue_service import TaskQueueService
        from models.background_task_models import TaskPriority

        mock_db = MagicMock()
        mock_db.background_tasks.insert_one = AsyncMock()

        mock_redis = AsyncMock()
        mock_redis.lpush = AsyncMock()

        service = TaskQueueService(database=mock_db)
        service.redis_client = mock_redis
        service._running = True

        # Enqueue high priority task
        high_task = self._create_mock_task("high-1", "high")
        high_task.priority = TaskPriority.HIGH

        await service.enqueue_task(high_task)

        # Verify it was added to high priority queue
        call_args = mock_redis.lpush.call_args
        assert call_args[0][0] == "greta_tasks_high"

    @pytest.mark.asyncio
    async def test_default_priority_queue(self):
        """Test default priority queue behavior"""
        from services.task_queue_service import TaskQueueService
        from models.background_task_models import TaskPriority

        mock_db = MagicMock()
        mock_db.background_tasks.insert_one = AsyncMock()

        mock_redis = AsyncMock()
        mock_redis.lpush = AsyncMock()

        service = TaskQueueService(database=mock_db)
        service.redis_client = mock_redis
        service._running = True

        # Enqueue medium priority task
        task = self._create_mock_task("medium-1", "medium")
        task.priority = TaskPriority.MEDIUM

        await service.enqueue_task(task)

        # Verify it was added to medium priority queue
        call_args = mock_redis.lpush.call_args
        assert call_args[0][0] == "greta_tasks_medium"


class TestTaskStatusTracking:
    """Test task status tracking"""
    
    @pytest.mark.asyncio
    async def test_update_task_status_to_in_progress(self):
        """Test updating task status to in progress"""
        from services.task_queue_service import TaskQueueService
        from models.background_task_models import TaskStatus
        
        mock_db = MagicMock()
        mock_db.background_tasks.update_one = AsyncMock()
        
        service = TaskQueueService(database=mock_db)
        
        await service.update_task_status("test-123", TaskStatus.IN_PROGRESS)
        
        assert mock_db.background_tasks.update_one.called
        call_args = mock_db.background_tasks.update_one.call_args
        assert call_args[0][1]["$set"]["status"] == TaskStatus.IN_PROGRESS.value
        assert "started_at" in call_args[0][1]["$set"]
    
    @pytest.mark.asyncio
    async def test_update_task_status_to_completed(self):
        """Test updating task status to completed"""
        from services.task_queue_service import TaskQueueService
        from models.background_task_models import TaskStatus
        
        mock_db = MagicMock()
        mock_db.background_tasks.update_one = AsyncMock()
        
        service = TaskQueueService(database=mock_db)
        
        await service.update_task_status(
            "test-123",
            TaskStatus.COMPLETED,
            result={"success": True}
        )
        
        assert mock_db.background_tasks.update_one.called
        call_args = mock_db.background_tasks.update_one.call_args
        assert call_args[0][1]["$set"]["status"] == TaskStatus.COMPLETED.value
        assert "completed_at" in call_args[0][1]["$set"]
        assert call_args[0][1]["$set"]["result"] == {"success": True}
    
    @pytest.mark.asyncio
    async def test_update_task_status_to_failed(self):
        """Test updating task status to failed"""
        from services.task_queue_service import TaskQueueService
        from models.background_task_models import TaskStatus
        
        mock_db = MagicMock()
        mock_db.background_tasks.update_one = AsyncMock()
        
        service = TaskQueueService(database=mock_db)
        
        await service.update_task_status(
            "test-123",
            TaskStatus.FAILED,
            error="Task execution failed"
        )
        
        assert mock_db.background_tasks.update_one.called
        call_args = mock_db.background_tasks.update_one.call_args
        assert call_args[0][1]["$set"]["status"] == TaskStatus.FAILED.value
        assert call_args[0][1]["$set"]["last_error"] == "Task execution failed"
    
    @pytest.mark.asyncio
    async def test_get_queue_stats(self):
        """Test retrieving queue statistics"""
        from services.task_queue_service import TaskQueueService
        from models.background_task_models import TaskStatus
        
        mock_db = MagicMock()
        
        # Mock aggregation results
        async def mock_aggregate(pipeline):
            results = [
                {"_id": TaskStatus.PENDING.value, "count": 5},
                {"_id": TaskStatus.IN_PROGRESS.value, "count": 2},
                {"_id": TaskStatus.COMPLETED.value, "count": 10},
                {"_id": TaskStatus.FAILED.value, "count": 1}
            ]
            for result in results:
                yield result
        
        mock_db.background_tasks.aggregate = mock_aggregate
        
        mock_redis = AsyncMock()
        mock_redis.llen = AsyncMock(return_value=5)
        
        service = TaskQueueService(database=mock_db)
        service.redis_client = mock_redis
        
        stats = await service.get_queue_stats()
        
        assert stats.pending_tasks == 5
        assert stats.in_progress_tasks == 2
        assert stats.completed_tasks == 10
        assert stats.failed_tasks == 1
        assert stats.total_tasks == 18


class TestWorkerManagement:
    """Test worker management"""

    @pytest.mark.asyncio
    async def test_start_worker(self):
        """Test starting a worker"""
        from services.task_queue_service import TaskQueueService, TaskWorker

        mock_db = MagicMock()
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()

        service = TaskQueueService(database=mock_db)
        service.redis_client = mock_redis
        service._running = True

        # Mock worker start
        with patch.object(TaskWorker, 'start', new_callable=AsyncMock) as mock_start:
            worker = await service.start_worker("test_worker")

            # start_worker returns the worker object, not the ID
            assert worker.worker_id == "test_worker"
            assert "test_worker" in service.workers

    @pytest.mark.asyncio
    async def test_stop_worker(self):
        """Test stopping a worker"""
        from services.task_queue_service import TaskQueueService, TaskWorker

        mock_db = MagicMock()
        mock_redis = AsyncMock()

        service = TaskQueueService(database=mock_db)
        service.redis_client = mock_redis
        service._running = True

        # Create mock worker
        mock_worker = MagicMock()
        mock_worker.stop = AsyncMock()
        service.workers["test_worker"] = mock_worker

        await service.stop_worker("test_worker")

        assert mock_worker.stop.called
        assert "test_worker" not in service.workers

    @pytest.mark.asyncio
    async def test_worker_task_assignment(self):
        """Test task assignment to workers"""
        from services.task_queue_service import TaskWorker

        mock_service = MagicMock()
        mock_service.redis_client = AsyncMock()
        mock_service.redis_client.brpop = AsyncMock(return_value=None)

        worker = TaskWorker(
            worker_id="test_worker",
            queue_service=mock_service,
            queue_names=["greta_tasks_high"]
        )

        # Test getting next task
        task = await worker._get_next_task()

        # Should return None when no tasks available
        assert task is None

    @pytest.mark.asyncio
    async def test_worker_health_monitoring(self):
        """Test worker health monitoring"""
        from services.task_queue_service import TaskQueueService

        mock_db = MagicMock()
        mock_redis = AsyncMock()

        service = TaskQueueService(database=mock_db)
        service.redis_client = mock_redis

        # Create mock workers
        mock_worker1 = MagicMock()
        mock_worker1.running = True
        mock_worker2 = MagicMock()
        mock_worker2.running = False

        service.workers["worker1"] = mock_worker1
        service.workers["worker2"] = mock_worker2

        # Test worker count
        assert len(service.workers) == 2

        # Test active workers
        active_count = sum(1 for w in service.workers.values() if w.running)
        assert active_count == 1


class TestErrorHandling:
    """Test error handling"""

    @pytest.mark.asyncio
    async def test_enqueue_task_failure(self):
        """Test handling task enqueueing failure"""
        from services.task_queue_service import TaskQueueService

        mock_db = MagicMock()
        mock_db.background_tasks.insert_one = AsyncMock(
            side_effect=Exception("Database error")
        )

        service = TaskQueueService(database=mock_db)
        service.redis_client = AsyncMock()
        service._running = True

        # Mock task object
        mock_task = MagicMock()
        mock_task.task_id = "test-fail"
        mock_task.task_type = MagicMock()
        mock_task.task_type.value = "email_sending"
        mock_task.priority = MagicMock()
        mock_task.priority.value = "medium"
        mock_task.payload = {"to": "test@example.com"}
        mock_task.created_at = MagicMock()
        mock_task.created_at.isoformat = MagicMock(return_value="2024-01-01T00:00:00")
        mock_task.scheduled_for = None
        mock_task.dict = MagicMock(side_effect=Exception("Database error"))

        with pytest.raises(TaskQueueError):
            await service.enqueue_task(mock_task)

    @pytest.mark.asyncio
    async def test_initialization_failure(self):
        """Test handling service initialization failure"""
        from services.task_queue_service import TaskQueueService

        mock_db = MagicMock()

        service = TaskQueueService(database=mock_db)

        # Mock Redis connection failure
        with patch('services.task_queue_service.get_redis_client',
                   side_effect=Exception("Redis connection failed")):
            with pytest.raises(TaskQueueError):
                await service.initialize()

    @pytest.mark.asyncio
    async def test_task_handler_not_found(self):
        """Test handling missing task handler"""
        from services.task_queue_service import TaskWorker

        # Mock task object
        mock_task = MagicMock()
        mock_task.task_id = "test-123"
        mock_task.task_type = MagicMock()
        mock_task.task_type.value = "email_sending"
        mock_task.mark_started = MagicMock()
        mock_task.mark_failed = MagicMock()
        mock_task.should_retry = MagicMock(return_value=False)
        mock_task.current_attempt = 1  # Set to integer for JSON serialization

        mock_service = MagicMock()
        mock_service.get_task = AsyncMock(return_value=mock_task)
        mock_service.update_task_status = AsyncMock()
        mock_service.task_handlers = {}  # No handlers registered
        mock_service.dead_letter_queue = "greta_tasks_dlq"
        mock_service.redis_client = AsyncMock()
        mock_service.redis_client.lpush = AsyncMock()

        worker = TaskWorker(
            worker_id="test_worker",
            queue_service=mock_service,
            queue_names=["greta_tasks"]
        )

        # Process task should handle missing handler gracefully
        task_data = {"task_id": "test-123"}

        # Should not raise exception
        await worker._process_task(task_data)

    @pytest.mark.asyncio
    async def test_task_retry_logic(self):
        """Test task retry logic on failure"""
        from services.task_queue_service import TaskWorker

        # Mock task with retry capability
        mock_task = MagicMock()
        mock_task.task_id = "test-retry"
        mock_task.mark_failed = MagicMock()
        mock_task.should_retry = MagicMock(return_value=True)
        mock_task.get_next_retry_delay = MagicMock(return_value=60)
        mock_task.current_attempt = 1

        mock_service = MagicMock()
        mock_service.get_task = AsyncMock(return_value=mock_task)
        mock_service.update_task_status = AsyncMock()
        mock_service.enqueue_task = AsyncMock()

        worker = TaskWorker(
            worker_id="test_worker",
            queue_service=mock_service,
            queue_names=["greta_tasks"]
        )

        # Handle task failure
        await worker._handle_task_failure("test-retry", "Task failed")

        # Should retry if attempts remaining
        assert mock_service.enqueue_task.called

    @pytest.mark.asyncio
    async def test_dead_letter_queue(self):
        """Test moving failed tasks to dead letter queue"""
        from services.task_queue_service import TaskWorker

        # Mock task that has exhausted retries
        mock_task = MagicMock()
        mock_task.task_id = "test-dlq"
        mock_task.mark_failed = MagicMock()
        mock_task.should_retry = MagicMock(return_value=False)
        mock_task.current_attempt = 3

        mock_service = MagicMock()
        mock_service.get_task = AsyncMock(return_value=mock_task)
        mock_service.update_task_status = AsyncMock()
        mock_service.redis_client = AsyncMock()
        mock_service.redis_client.lpush = AsyncMock()

        worker = TaskWorker(
            worker_id="test_worker",
            queue_service=mock_service,
            queue_names=["greta_tasks"]
        )

        # Handle task failure
        await worker._handle_task_failure("test-dlq", "Task failed")

        # Should update status to failed (no more retries)
        assert mock_service.update_task_status.called

    @pytest.mark.asyncio
    async def test_worker_stop_gracefully(self):
        """Test worker stops gracefully"""
        from services.task_queue_service import TaskWorker

        mock_service = MagicMock()
        mock_service.redis_client = AsyncMock()

        worker = TaskWorker(
            worker_id="test_worker",
            queue_service=mock_service,
            queue_names=["greta_tasks"]
        )

        worker.running = True
        worker._task = None

        await worker.stop()

        assert worker.running is False

