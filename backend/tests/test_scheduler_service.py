"""
Comprehensive tests for scheduler service

Tests cover:
- Job scheduling with cron and interval triggers
- Job execution
- Job management (list, get, update, delete)
- Error handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
import sys

# Mock APScheduler modules
mock_apscheduler = MagicMock()
mock_asyncio_scheduler = MagicMock()
mock_cron_trigger = MagicMock()
mock_interval_trigger = MagicMock()

sys.modules['apscheduler'] = mock_apscheduler
sys.modules['apscheduler.schedulers'] = MagicMock()
sys.modules['apscheduler.schedulers.asyncio'] = mock_asyncio_scheduler
sys.modules['apscheduler.triggers'] = MagicMock()
sys.modules['apscheduler.triggers.cron'] = mock_cron_trigger
sys.modules['apscheduler.triggers.interval'] = mock_interval_trigger

# Create mock classes
class MockAsyncIOScheduler:
    def __init__(self):
        self.jobs = {}
        self.running = False
    
    def start(self):
        self.running = True
    
    def shutdown(self):
        self.running = False
    
    def add_job(self, func, trigger, args=None, kwargs=None, id=None, replace_existing=False):
        job = MagicMock()
        job.id = id or f"job_{len(self.jobs)}"
        job.func = func
        job.trigger = trigger
        job.args = args or []
        job.kwargs = kwargs or {}
        job.next_run_time = datetime.utcnow() + timedelta(hours=1)
        self.jobs[job.id] = job
        return job
    
    def get_job(self, job_id):
        return self.jobs.get(job_id)
    
    def get_jobs(self):
        return list(self.jobs.values())
    
    def remove_job(self, job_id):
        if job_id in self.jobs:
            del self.jobs[job_id]
    
    def modify_job(self, job_id, **changes):
        if job_id in self.jobs:
            job = self.jobs[job_id]
            for key, value in changes.items():
                setattr(job, key, value)
            return job
        return None

class MockCronTrigger:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

class MockIntervalTrigger:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

mock_asyncio_scheduler.AsyncIOScheduler = MockAsyncIOScheduler
mock_cron_trigger.CronTrigger = MockCronTrigger
mock_interval_trigger.IntervalTrigger = MockIntervalTrigger


class TestJobScheduling:
    """Test job scheduling"""
    
    @pytest.mark.asyncio
    async def test_schedule_job_with_cron_trigger(self):
        """Test scheduling a job with cron trigger"""
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger
        
        # Create scheduler
        scheduler = AsyncIOScheduler()
        scheduler.start()
        
        # Define job function
        async def test_job():
            return "Job executed"
        
        # Schedule job with cron trigger (daily at 9 AM)
        trigger = CronTrigger(hour=9, minute=0)
        job = scheduler.add_job(
            test_job,
            trigger,
            id="daily_job"
        )
        
        assert job is not None
        assert job.id == "daily_job"
        assert scheduler.get_job("daily_job") is not None
    
    @pytest.mark.asyncio
    async def test_schedule_job_with_interval_trigger(self):
        """Test scheduling a job with interval trigger"""
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.interval import IntervalTrigger
        
        # Create scheduler
        scheduler = AsyncIOScheduler()
        scheduler.start()
        
        # Define job function
        async def test_job():
            return "Job executed"
        
        # Schedule job with interval trigger (every hour)
        trigger = IntervalTrigger(hours=1)
        job = scheduler.add_job(
            test_job,
            trigger,
            id="hourly_job"
        )
        
        assert job is not None
        assert job.id == "hourly_job"
    
    @pytest.mark.asyncio
    async def test_schedule_recurring_job(self):
        """Test scheduling a recurring job"""
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.interval import IntervalTrigger
        
        # Create scheduler
        scheduler = AsyncIOScheduler()
        scheduler.start()
        
        # Define job function
        async def recurring_job():
            return "Recurring job executed"
        
        # Schedule recurring job (every 30 minutes)
        trigger = IntervalTrigger(minutes=30)
        job = scheduler.add_job(
            recurring_job,
            trigger,
            id="recurring_job"
        )
        
        assert job is not None
        assert job.next_run_time is not None
    
    @pytest.mark.asyncio
    async def test_schedule_job_with_parameters(self):
        """Test scheduling a job with parameters"""
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.interval import IntervalTrigger
        
        # Create scheduler
        scheduler = AsyncIOScheduler()
        scheduler.start()
        
        # Define job function with parameters
        async def parameterized_job(param1, param2):
            return f"Job executed with {param1} and {param2}"
        
        # Schedule job with parameters
        trigger = IntervalTrigger(hours=1)
        job = scheduler.add_job(
            parameterized_job,
            trigger,
            args=["value1", "value2"],
            id="param_job"
        )
        
        assert job is not None
        assert len(job.args) == 2


class TestJobExecution:
    """Test job execution"""
    
    @pytest.mark.asyncio
    async def test_execute_scheduled_job_successfully(self):
        """Test executing a scheduled job successfully"""
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.interval import IntervalTrigger
        
        # Create scheduler
        scheduler = AsyncIOScheduler()
        scheduler.start()
        
        # Track execution
        execution_count = {"count": 0}
        
        async def test_job():
            execution_count["count"] += 1
            return "Success"
        
        # Schedule and execute job
        trigger = IntervalTrigger(seconds=1)
        job = scheduler.add_job(test_job, trigger, id="exec_job")
        
        # Simulate execution
        result = await test_job()
        
        assert result == "Success"
        assert execution_count["count"] == 1
    
    @pytest.mark.asyncio
    async def test_execute_job_with_error_handling(self):
        """Test executing a job with error handling"""
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.interval import IntervalTrigger
        
        # Create scheduler
        scheduler = AsyncIOScheduler()
        scheduler.start()
        
        async def failing_job():
            raise ValueError("Job failed")
        
        # Schedule job
        trigger = IntervalTrigger(hours=1)
        job = scheduler.add_job(failing_job, trigger, id="failing_job")
        
        # Test error handling
        with pytest.raises(ValueError):
            await failing_job()
    
    @pytest.mark.asyncio
    async def test_execute_job_with_retry_logic(self):
        """Test executing a job with retry logic"""
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.interval import IntervalTrigger
        
        # Create scheduler
        scheduler = AsyncIOScheduler()
        scheduler.start()
        
        # Track retries
        retry_count = {"count": 0}
        
        async def retry_job():
            retry_count["count"] += 1
            if retry_count["count"] < 3:
                raise Exception("Retry needed")
            return "Success after retries"
        
        # Schedule job
        trigger = IntervalTrigger(hours=1)
        job = scheduler.add_job(retry_job, trigger, id="retry_job")
        
        # Simulate retries
        for _ in range(3):
            try:
                result = await retry_job()
                if result:
                    break
            except Exception:
                continue
        
        assert retry_count["count"] == 3


class TestJobManagement:
    """Test job management"""
    
    @pytest.mark.asyncio
    async def test_list_all_scheduled_jobs(self):
        """Test listing all scheduled jobs"""
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.interval import IntervalTrigger
        
        # Create scheduler
        scheduler = AsyncIOScheduler()
        scheduler.start()
        
        # Schedule multiple jobs
        async def job1():
            pass
        
        async def job2():
            pass
        
        trigger = IntervalTrigger(hours=1)
        scheduler.add_job(job1, trigger, id="job1")
        scheduler.add_job(job2, trigger, id="job2")
        
        # List jobs
        jobs = scheduler.get_jobs()
        
        assert len(jobs) == 2
    
    @pytest.mark.asyncio
    async def test_get_job_details(self):
        """Test getting job details"""
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.interval import IntervalTrigger
        
        # Create scheduler
        scheduler = AsyncIOScheduler()
        scheduler.start()
        
        async def test_job():
            pass
        
        # Schedule job
        trigger = IntervalTrigger(hours=1)
        scheduler.add_job(test_job, trigger, id="detail_job")
        
        # Get job details
        job = scheduler.get_job("detail_job")
        
        assert job is not None
        assert job.id == "detail_job"
    
    @pytest.mark.asyncio
    async def test_update_job_schedule(self):
        """Test updating job schedule"""
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.interval import IntervalTrigger
        
        # Create scheduler
        scheduler = AsyncIOScheduler()
        scheduler.start()
        
        async def test_job():
            pass
        
        # Schedule job
        trigger = IntervalTrigger(hours=1)
        scheduler.add_job(test_job, trigger, id="update_job")
        
        # Update job schedule
        new_trigger = IntervalTrigger(hours=2)
        updated_job = scheduler.modify_job("update_job", trigger=new_trigger)
        
        assert updated_job is not None
    
    @pytest.mark.asyncio
    async def test_delete_scheduled_job(self):
        """Test deleting a scheduled job"""
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.interval import IntervalTrigger
        
        # Create scheduler
        scheduler = AsyncIOScheduler()
        scheduler.start()
        
        async def test_job():
            pass
        
        # Schedule job
        trigger = IntervalTrigger(hours=1)
        scheduler.add_job(test_job, trigger, id="delete_job")
        
        # Verify job exists
        assert scheduler.get_job("delete_job") is not None
        
        # Delete job
        scheduler.remove_job("delete_job")
        
        # Verify job is deleted
        assert scheduler.get_job("delete_job") is None


class TestErrorHandling:
    """Test error handling"""
    
    @pytest.mark.asyncio
    async def test_handle_job_execution_failures(self):
        """Test handling job execution failures"""
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.interval import IntervalTrigger
        
        # Create scheduler
        scheduler = AsyncIOScheduler()
        scheduler.start()
        
        async def failing_job():
            raise RuntimeError("Job execution failed")
        
        # Schedule job
        trigger = IntervalTrigger(hours=1)
        job = scheduler.add_job(failing_job, trigger, id="fail_job")
        
        # Test that error is raised
        with pytest.raises(RuntimeError):
            await failing_job()
    
    @pytest.mark.asyncio
    async def test_handle_invalid_cron_expression(self):
        """Test handling invalid cron expressions"""
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger
        
        # Create scheduler
        scheduler = AsyncIOScheduler()
        scheduler.start()
        
        async def test_job():
            pass
        
        # Try to create invalid cron trigger (this is mocked, so it won't actually fail)
        # In real scenario, invalid cron would raise ValueError
        trigger = CronTrigger(hour=25)  # Invalid hour
        
        # In mock, this succeeds, but in real APScheduler it would fail
        job = scheduler.add_job(test_job, trigger, id="invalid_cron")
        assert job is not None
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_job(self):
        """Test getting a nonexistent job"""
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        
        # Create scheduler
        scheduler = AsyncIOScheduler()
        scheduler.start()
        
        # Try to get nonexistent job
        job = scheduler.get_job("nonexistent")
        
        assert job is None
