"""
Task Persistence Service for CPAS4
Comprehensive task storage and lifecycle management
"""

import logging
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
from database import Database
from models.task_models import (
    Task, TaskCreate, TaskUpdate, TaskQuery, TaskStatus,
    TaskStats, TaskResult, TaskPriority, TaskType, TaskMetadata
)

logger = logging.getLogger(__name__)


class TaskPersistenceService:
    """
    Task Persistence Service
    Manages task storage, retrieval, and lifecycle in database
    """
    
    def __init__(self, database: Database):
        self.db = database
        self.collection_name = "tasks"
        logger.info("✅ Task Persistence Service initialized")
    
    async def create_task(self, task_create: TaskCreate) -> Task:
        """Create a new task"""
        try:
            # Generate task ID
            task_id = f"task_{uuid.uuid4().hex[:12]}"
            
            # Create task object
            task = Task(
                task_id=task_id,
                title=task_create.title,
                description=task_create.description,
                priority=task_create.priority,
                task_type=task_create.task_type,
                input_data=task_create.input_data,
                metadata=task_create.metadata or TaskMetadata(),
                parent_task_id=task_create.parent_task_id,
                depends_on=task_create.depends_on,
                status=TaskStatus.PENDING,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Store in database
            task_dict = task.model_dump()
            task_dict["created_at"] = task.created_at.isoformat()
            task_dict["updated_at"] = task.updated_at.isoformat()
            
            await self.db.insert_one(self.collection_name, task_dict)
            
            logger.info(f"✅ Task created: {task_id}")
            return task
            
        except Exception as e:
            logger.error(f"❌ Error creating task: {e}")
            raise
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        try:
            task_dict = await self.db.find_one(
                self.collection_name,
                {"task_id": task_id}
            )
            
            if not task_dict:
                return None
            
            # Convert ISO strings back to datetime
            if "created_at" in task_dict and isinstance(task_dict["created_at"], str):
                task_dict["created_at"] = datetime.fromisoformat(task_dict["created_at"])
            if "updated_at" in task_dict and isinstance(task_dict["updated_at"], str):
                task_dict["updated_at"] = datetime.fromisoformat(task_dict["updated_at"])
            if "started_at" in task_dict and task_dict["started_at"]:
                task_dict["started_at"] = datetime.fromisoformat(task_dict["started_at"])
            if "completed_at" in task_dict and task_dict["completed_at"]:
                task_dict["completed_at"] = datetime.fromisoformat(task_dict["completed_at"])
            
            return Task(**task_dict)
            
        except Exception as e:
            logger.error(f"❌ Error getting task {task_id}: {e}")
            return None
    
    async def update_task(self, task_id: str, task_update: TaskUpdate) -> Optional[Task]:
        """Update task"""
        try:
            # Get current task
            current_task = await self.get_task(task_id)
            if not current_task:
                return None
            
            # Prepare update data
            update_data = task_update.model_dump(exclude_unset=True)
            update_data["updated_at"] = datetime.utcnow().isoformat()
            
            # Handle status transitions
            if "status" in update_data:
                new_status = update_data["status"]
                
                # Set started_at when moving to in_progress
                if new_status == TaskStatus.IN_PROGRESS and not current_task.started_at:
                    update_data["started_at"] = datetime.utcnow().isoformat()
                
                # Set completed_at when moving to completed/failed/cancelled
                if new_status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    update_data["completed_at"] = datetime.utcnow().isoformat()
            
            # Update in database
            success = await self.db.update_one(
                self.collection_name,
                {"task_id": task_id},
                {"$set": update_data}
            )
            
            if success:
                logger.info(f"✅ Task updated: {task_id}")
                return await self.get_task(task_id)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error updating task {task_id}: {e}")
            return None
    
    async def delete_task(self, task_id: str) -> bool:
        """Delete task"""
        try:
            success = await self.db.delete_one(
                self.collection_name,
                {"task_id": task_id}
            )
            
            if success:
                logger.info(f"✅ Task deleted: {task_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error deleting task {task_id}: {e}")
            return False
    
    async def query_tasks(self, query: TaskQuery) -> List[Task]:
        """Query tasks with filters"""
        try:
            # Build query filter
            filter_dict = {}
            
            if query.status:
                filter_dict["status"] = query.status
            if query.priority:
                filter_dict["priority"] = query.priority
            if query.task_type:
                filter_dict["task_type"] = query.task_type
            if query.agent_type:
                filter_dict["metadata.agent_type"] = query.agent_type
            if query.user_id:
                filter_dict["metadata.user_id"] = query.user_id
            
            # Query database
            task_dicts = await self.db.find_many(
                self.collection_name,
                filter_dict,
                limit=query.limit
            )
            
            # Convert to Task objects
            tasks = []
            for task_dict in task_dicts:
                # Convert ISO strings back to datetime
                if "created_at" in task_dict and isinstance(task_dict["created_at"], str):
                    task_dict["created_at"] = datetime.fromisoformat(task_dict["created_at"])
                if "updated_at" in task_dict and isinstance(task_dict["updated_at"], str):
                    task_dict["updated_at"] = datetime.fromisoformat(task_dict["updated_at"])
                if "started_at" in task_dict and task_dict["started_at"]:
                    task_dict["started_at"] = datetime.fromisoformat(task_dict["started_at"])
                if "completed_at" in task_dict and task_dict["completed_at"]:
                    task_dict["completed_at"] = datetime.fromisoformat(task_dict["completed_at"])
                
                tasks.append(Task(**task_dict))
            
            return tasks
            
        except Exception as e:
            logger.error(f"❌ Error querying tasks: {e}")
            return []
    
    async def get_task_stats(self) -> TaskStats:
        """Get task statistics"""
        try:
            # Get all tasks
            all_tasks = await self.db.find_many(self.collection_name, {}, limit=10000)
            
            # Calculate stats
            total = len(all_tasks)
            by_status = {status.value: 0 for status in TaskStatus}
            by_priority = {priority.value: 0 for priority in TaskPriority}
            by_type = {task_type.value: 0 for task_type in TaskType}
            by_agent = {}
            
            execution_times = []
            completed_count = 0
            failed_count = 0
            
            for task_dict in all_tasks:
                # Count by status
                status = task_dict.get("status", "pending")
                by_status[status] = by_status.get(status, 0) + 1
                
                # Count by priority
                priority = task_dict.get("priority", "medium")
                by_priority[priority] = by_priority.get(priority, 0) + 1
                
                # Count by type
                task_type = task_dict.get("task_type", "manual")
                by_type[task_type] = by_type.get(task_type, 0) + 1
                
                # Count by agent
                agent_type = task_dict.get("metadata", {}).get("agent_type")
                if agent_type:
                    by_agent[agent_type] = by_agent.get(agent_type, 0) + 1
                
                # Calculate execution time
                if task_dict.get("result") and task_dict["result"].get("execution_time"):
                    execution_times.append(task_dict["result"]["execution_time"])
                
                # Count completed/failed
                if status == "completed":
                    completed_count += 1
                elif status == "failed":
                    failed_count += 1
            
            # Calculate averages
            avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else None
            success_rate = (completed_count / (completed_count + failed_count) * 100) if (completed_count + failed_count) > 0 else None
            
            return TaskStats(
                total_tasks=total,
                pending=by_status.get("pending", 0),
                in_progress=by_status.get("in_progress", 0),
                completed=by_status.get("completed", 0),
                failed=by_status.get("failed", 0),
                cancelled=by_status.get("cancelled", 0),
                paused=by_status.get("paused", 0),
                by_priority=by_priority,
                by_type=by_type,
                by_agent=by_agent,
                average_execution_time=avg_execution_time,
                success_rate=success_rate
            )
            
        except Exception as e:
            logger.error(f"❌ Error getting task stats: {e}")
            return TaskStats(
                total_tasks=0,
                pending=0,
                in_progress=0,
                completed=0,
                failed=0,
                cancelled=0,
                paused=0,
                by_priority={},
                by_type={},
                by_agent={}
            )

