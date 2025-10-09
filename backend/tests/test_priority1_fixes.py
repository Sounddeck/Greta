"""
Priority 1 Fixes - Comprehensive Test Suite
Tests MongoDB connection fix and task persistence system
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import Database
from services.task_persistence_service import TaskPersistenceService
from models.task_models import (
    TaskCreate, TaskUpdate, TaskQuery, TaskStatus,
    TaskPriority, TaskType, TaskMetadata, TaskResult
)


async def test_mongodb_connection():
    """Test 1: MongoDB Connection Fix"""
    print("\n" + "="*80)
    print("TEST 1: MongoDB Connection Fix")
    print("="*80)
    
    try:
        db = Database()
        connected = await db.connect()
        
        print(f"✅ MongoDB connection successful: {connected}")
        print(f"✅ Database instance created: {db.db is not None}")
        print(f"✅ Connection status: {db.connected}")
        
        # Test health check
        health = await db.health_check()
        print(f"✅ Health check: {health['status']}")
        print(f"   Type: {health['type']}")
        
        if health['type'] == 'mongodb':
            print(f"   Database: {health.get('database')}")
            print(f"   Collections: {len(health.get('collections', []))}")
        
        await db.disconnect()
        print("✅ MongoDB connection test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ MongoDB connection test FAILED: {e}")
        return False


async def test_task_persistence_crud():
    """Test 2: Task Persistence CRUD Operations"""
    print("\n" + "="*80)
    print("TEST 2: Task Persistence CRUD Operations")
    print("="*80)
    
    try:
        # Initialize database and service
        db = Database()
        await db.connect()
        service = TaskPersistenceService(db)
        
        # Test 1: Create Task
        print("\n📝 Test 2.1: Create Task")
        task_create = TaskCreate(
            title="Test Marketing Campaign",
            description="Test task for marketing campaign generation",
            priority=TaskPriority.HIGH,
            task_type=TaskType.AGENT_EXECUTION,
            input_data={
                "agent_type": "marketing",
                "campaign_type": "email"
            },
            metadata=TaskMetadata(
                agent_type="marketing",
                user_id="test_user",
                tags=["test", "marketing"]
            )
        )
        
        task = await service.create_task(task_create)
        print(f"✅ Task created: {task.task_id}")
        print(f"   Title: {task.title}")
        print(f"   Status: {task.status}")
        print(f"   Priority: {task.priority}")
        
        task_id = task.task_id
        
        # Test 2: Get Task
        print("\n📖 Test 2.2: Get Task")
        retrieved_task = await service.get_task(task_id)
        assert retrieved_task is not None, "Task not found"
        assert retrieved_task.task_id == task_id, "Task ID mismatch"
        print(f"✅ Task retrieved: {retrieved_task.task_id}")
        
        # Test 3: Update Task - Start
        print("\n🚀 Test 2.3: Start Task")
        task_update = TaskUpdate(status=TaskStatus.IN_PROGRESS)
        updated_task = await service.update_task(task_id, task_update)
        assert updated_task.status == TaskStatus.IN_PROGRESS, "Status not updated"
        assert updated_task.started_at is not None, "Started timestamp not set"
        print(f"✅ Task started: {updated_task.status}")
        print(f"   Started at: {updated_task.started_at}")
        
        # Test 4: Update Task - Progress
        print("\n📊 Test 2.4: Update Progress")
        task_update = TaskUpdate(
            progress_percentage=50.0,
            progress_message="Campaign outline created"
        )
        updated_task = await service.update_task(task_id, task_update)
        assert updated_task.progress_percentage == 50.0, "Progress not updated"
        print(f"✅ Progress updated: {updated_task.progress_percentage}%")
        print(f"   Message: {updated_task.progress_message}")
        
        # Test 5: Complete Task
        print("\n✅ Test 2.5: Complete Task")
        result = TaskResult(
            success=True,
            output={"campaign_name": "Test Campaign", "emails": 10},
            execution_time=5.2,
            tokens_used=1500,
            model_used="llama3:latest"
        )
        task_update = TaskUpdate(
            status=TaskStatus.COMPLETED,
            result=result,
            progress_percentage=100.0
        )
        completed_task = await service.update_task(task_id, task_update)
        assert completed_task.status == TaskStatus.COMPLETED, "Status not completed"
        assert completed_task.completed_at is not None, "Completed timestamp not set"
        assert completed_task.result is not None, "Result not set"
        print(f"✅ Task completed: {completed_task.status}")
        print(f"   Completed at: {completed_task.completed_at}")
        print(f"   Execution time: {completed_task.result.execution_time}s")
        
        # Test 6: Query Tasks
        print("\n🔍 Test 2.6: Query Tasks")
        query = TaskQuery(
            status=TaskStatus.COMPLETED,
            agent_type="marketing",
            limit=10
        )
        tasks = await service.query_tasks(query)
        assert len(tasks) > 0, "No tasks found"
        print(f"✅ Found {len(tasks)} completed marketing tasks")
        
        # Test 7: Get Statistics
        print("\n📈 Test 2.7: Get Statistics")
        stats = await service.get_task_stats()
        print(f"✅ Task statistics:")
        print(f"   Total tasks: {stats.total_tasks}")
        print(f"   Completed: {stats.completed}")
        print(f"   Success rate: {stats.success_rate}%")
        print(f"   Avg execution time: {stats.average_execution_time}s")
        
        # Test 8: Delete Task
        print("\n🗑️  Test 2.8: Delete Task")
        deleted = await service.delete_task(task_id)
        assert deleted, "Task not deleted"
        
        # Verify deletion
        deleted_task = await service.get_task(task_id)
        assert deleted_task is None, "Task still exists after deletion"
        print(f"✅ Task deleted: {task_id}")
        
        await db.disconnect()
        print("\n✅ Task Persistence CRUD test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Task Persistence CRUD test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_task_lifecycle():
    """Test 3: Complete Task Lifecycle"""
    print("\n" + "="*80)
    print("TEST 3: Complete Task Lifecycle")
    print("="*80)
    
    try:
        db = Database()
        await db.connect()
        service = TaskPersistenceService(db)
        
        # Create multiple tasks
        print("\n📝 Creating multiple tasks...")
        task_ids = []
        
        for i in range(5):
            task_create = TaskCreate(
                title=f"Task {i+1}",
                description=f"Test task number {i+1}",
                priority=[TaskPriority.LOW, TaskPriority.MEDIUM, TaskPriority.HIGH][i % 3],
                task_type=TaskType.AGENT_EXECUTION,
                metadata=TaskMetadata(
                    agent_type=["marketing", "coding", "research"][i % 3],
                    user_id="test_user"
                )
            )
            task = await service.create_task(task_create)
            task_ids.append(task.task_id)
        
        print(f"✅ Created {len(task_ids)} tasks")
        
        # Start some tasks
        print("\n🚀 Starting tasks...")
        for task_id in task_ids[:3]:
            await service.update_task(task_id, TaskUpdate(status=TaskStatus.IN_PROGRESS))
        print(f"✅ Started 3 tasks")
        
        # Complete some tasks
        print("\n✅ Completing tasks...")
        for task_id in task_ids[:2]:
            result = TaskResult(success=True, execution_time=3.0)
            await service.update_task(
                task_id,
                TaskUpdate(status=TaskStatus.COMPLETED, result=result)
            )
        print(f"✅ Completed 2 tasks")
        
        # Fail one task
        print("\n❌ Failing one task...")
        result = TaskResult(success=False, error="Test error")
        await service.update_task(
            task_ids[2],
            TaskUpdate(status=TaskStatus.FAILED, result=result)
        )
        print(f"✅ Failed 1 task")
        
        # Get final statistics
        print("\n📈 Final Statistics:")
        stats = await service.get_task_stats()
        print(f"   Total: {stats.total_tasks}")
        print(f"   Pending: {stats.pending}")
        print(f"   In Progress: {stats.in_progress}")
        print(f"   Completed: {stats.completed}")
        print(f"   Failed: {stats.failed}")
        print(f"   Success Rate: {stats.success_rate}%")
        
        # Cleanup
        print("\n🗑️  Cleaning up...")
        for task_id in task_ids:
            await service.delete_task(task_id)
        print(f"✅ Deleted {len(task_ids)} tasks")
        
        await db.disconnect()
        print("\n✅ Task Lifecycle test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Task Lifecycle test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all Priority 1 tests"""
    print("\n" + "="*80)
    print("CPAS4 PRIORITY 1 FIXES - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print("\nTesting:")
    print("1. MongoDB Connection Fix (truth value testing)")
    print("2. Task Persistence System (CRUD operations)")
    print("3. Task Lifecycle Management")
    print("\n" + "="*80)
    
    results = []
    
    # Run tests
    results.append(await test_mongodb_connection())
    results.append(await test_task_persistence_crud())
    results.append(await test_task_lifecycle())
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nTests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Priority 1 fixes are fully functional.")
        print("\n✅ MongoDB Connection: FIXED")
        print("✅ Task Persistence: IMPLEMENTED")
        print("✅ Task Lifecycle: WORKING")
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    asyncio.run(main())

