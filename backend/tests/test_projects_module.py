"""
Project Management Module Test Suite
Comprehensive tests for Project Management functionality
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime, timedelta
from database import Database
from services.projects_service import ProjectsService
from models.projects_models import *

class TestProjectsModule:
    def __init__(self):
        self.db = None
        self.service = None
        self.test_results = []
        
    async def setup(self):
        """Initialize database and service"""
        self.db = Database()
        await self.db.connect()
        self.service = ProjectsService(self.db)
        print("✅ Projects module test setup complete")
    
    async def cleanup(self):
        """Clean up test data"""
        if self.db is not None:
            for collection in ["projects_projects", "projects_tasks", "projects_milestones", "projects_time_entries", "projects_resources"]:
                try:
                    docs = await self.db.find_many(collection, {}, limit=10000)
                    for doc in docs:
                        if "_id" in doc:
                            await self.db.delete_one(collection, {"_id": doc["_id"]})
                except:
                    pass
        print("✅ Test cleanup complete")
    
    def record_result(self, test_name: str, passed: bool, message: str = ""):
        """Record test result"""
        self.test_results.append({"test": test_name, "passed": passed, "message": message})
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}" + (f" - {message}" if message else ""))
    
    async def test_project_crud(self):
        """Test project CRUD operations"""
        try:
            # Create project
            project_create = ProjectCreate(
                name="Test Project",
                description="A test project",
                client_name="Test Client",
                start_date=datetime.utcnow(),
                end_date=datetime.utcnow() + timedelta(days=30),
                budget=10000.00,
                team_members=["user1", "user2"]
            )
            project = await self.service.create_project(project_create)
            assert project.project_id.startswith("prj_"), "Project ID format incorrect"
            assert project.name == "Test Project", "Project name mismatch"
            
            # Get project
            retrieved = await self.service.get_project(project.project_id)
            assert retrieved is not None, "Project not found"
            assert retrieved.project_id == project.project_id, "Project ID mismatch"
            
            # Update project
            update = ProjectUpdate(name="Updated Project", status=ProjectStatus.ACTIVE)
            updated = await self.service.update_project(project.project_id, update)
            assert updated is not None, "Project update failed"
            assert updated.name == "Updated Project", "Project name not updated"
            assert updated.status == ProjectStatus.ACTIVE, "Project status not updated"
            
            # List projects
            projects = await self.service.list_projects()
            assert len(projects) >= 1, "No projects found"
            
            # Delete project
            deleted = await self.service.delete_project(project.project_id)
            assert deleted, "Project deletion failed"
            
            self.record_result("Project CRUD", True, "All operations successful")
        except Exception as e:
            self.record_result("Project CRUD", False, str(e))
    
    async def test_task_crud(self):
        """Test task CRUD operations"""
        try:
            # Create project first
            project_create = ProjectCreate(name="Task Test Project", client_name="Test Client")
            project = await self.service.create_project(project_create)
            
            # Create task
            task_create = TaskCreate(
                project_id=project.project_id,
                title="Test Task",
                description="A test task",
                priority=TaskPriority.HIGH,
                estimated_hours=8.0,
                due_date=datetime.utcnow() + timedelta(days=7)
            )
            task = await self.service.create_task(task_create)
            assert task.task_id.startswith("tsk_"), "Task ID format incorrect"
            assert task.title == "Test Task", "Task title mismatch"
            
            # Get task
            retrieved = await self.service.get_task(task.task_id)
            assert retrieved is not None, "Task not found"
            
            # Update task
            update = TaskUpdate(status=TaskStatus.IN_PROGRESS, actual_hours=2.0)
            updated = await self.service.update_task(task.task_id, update)
            assert updated is not None, "Task update failed"
            assert updated.status == TaskStatus.IN_PROGRESS, "Task status not updated"
            
            # List tasks
            tasks = await self.service.list_tasks(project_id=project.project_id)
            assert len(tasks) >= 1, "No tasks found"
            
            self.record_result("Task CRUD", True, "All operations successful")
        except Exception as e:
            self.record_result("Task CRUD", False, str(e))
    
    async def test_milestone_crud(self):
        """Test milestone CRUD operations"""
        try:
            # Create project first
            project_create = ProjectCreate(name="Milestone Test Project", client_name="Test Client")
            project = await self.service.create_project(project_create)
            
            # Create milestone
            milestone_create = MilestoneCreate(
                project_id=project.project_id,
                name="Test Milestone",
                description="A test milestone",
                due_date=datetime.utcnow() + timedelta(days=14),
                deliverables=["Deliverable 1", "Deliverable 2"]
            )
            milestone = await self.service.create_milestone(milestone_create)
            assert milestone.milestone_id.startswith("mls_"), "Milestone ID format incorrect"
            assert milestone.name == "Test Milestone", "Milestone name mismatch"
            
            # Get milestone
            retrieved = await self.service.get_milestone(milestone.milestone_id)
            assert retrieved is not None, "Milestone not found"
            
            # Update milestone
            update = MilestoneUpdate(status=MilestoneStatus.IN_PROGRESS, progress=50.0)
            updated = await self.service.update_milestone(milestone.milestone_id, update)
            assert updated is not None, "Milestone update failed"
            assert updated.status == MilestoneStatus.IN_PROGRESS, "Milestone status not updated"
            
            # List milestones
            milestones = await self.service.list_milestones(project_id=project.project_id)
            assert len(milestones) >= 1, "No milestones found"
            
            self.record_result("Milestone CRUD", True, "All operations successful")
        except Exception as e:
            self.record_result("Milestone CRUD", False, str(e))
    
    async def test_time_entry_crud(self):
        """Test time entry CRUD operations"""
        try:
            # Create project first
            project_create = ProjectCreate(name="Time Entry Test Project", client_name="Test Client")
            project = await self.service.create_project(project_create)
            
            # Create time entry
            start_time = datetime.utcnow()
            end_time = start_time + timedelta(hours=2)
            time_entry_create = TimeEntryCreate(
                project_id=project.project_id,
                user_id="user1",
                user_name="Test User",
                entry_type=TimeEntryType.WORK,
                start_time=start_time,
                end_time=end_time,
                description="Working on project",
                billable=True,
                hourly_rate=50.00
            )
            time_entry = await self.service.create_time_entry(time_entry_create)
            assert time_entry.entry_id.startswith("te_"), "Time entry ID format incorrect"
            assert time_entry.duration_hours == 2.0, "Duration not calculated correctly"
            assert time_entry.total_cost == 100.00, "Total cost not calculated correctly"
            
            # Get time entry
            retrieved = await self.service.get_time_entry(time_entry.entry_id)
            assert retrieved is not None, "Time entry not found"
            
            # List time entries
            entries = await self.service.list_time_entries(project_id=project.project_id)
            assert len(entries) >= 1, "No time entries found"
            
            self.record_result("Time Entry CRUD", True, "All operations successful")
        except Exception as e:
            self.record_result("Time Entry CRUD", False, str(e))
    
    async def test_project_stats_update(self):
        """Test automatic project statistics updates"""
        try:
            # Create project
            project_create = ProjectCreate(name="Stats Test Project", client_name="Test Client")
            project = await self.service.create_project(project_create)
            
            # Create tasks
            for i in range(3):
                task_create = TaskCreate(
                    project_id=project.project_id,
                    title=f"Task {i+1}",
                    priority=TaskPriority.MEDIUM
                )
                await self.service.create_task(task_create)
            
            # Check project stats updated
            updated_project = await self.service.get_project(project.project_id)
            assert updated_project.total_tasks == 3, "Total tasks not updated"
            
            self.record_result("Project Stats Update", True, "Stats updated correctly")
        except Exception as e:
            self.record_result("Project Stats Update", False, str(e))
    
    async def test_statistics(self):
        """Test project statistics"""
        try:
            stats = await self.service.get_stats()
            assert stats is not None, "Stats not returned"
            assert hasattr(stats, 'total_projects'), "Missing total_projects"
            assert hasattr(stats, 'total_tasks'), "Missing total_tasks"
            assert hasattr(stats, 'total_milestones'), "Missing total_milestones"
            assert hasattr(stats, 'total_hours_logged'), "Missing total_hours_logged"
            
            self.record_result("Statistics", True, f"Stats: {stats.total_projects} projects, {stats.total_tasks} tasks, {stats.total_milestones} milestones")
        except Exception as e:
            self.record_result("Statistics", False, str(e))
    
    async def test_filtering(self):
        """Test filtering capabilities"""
        try:
            # Create projects with different statuses
            for i, status in enumerate([ProjectStatus.PLANNING, ProjectStatus.ACTIVE, ProjectStatus.COMPLETED]):
                project_create = ProjectCreate(
                    name=f"Test {status.value} {i}",
                    client_name="Test Client"
                )
                project = await self.service.create_project(project_create)
                if status != ProjectStatus.PLANNING:
                    await self.service.update_project(project.project_id, ProjectUpdate(status=status))
            
            # Filter by status
            active_projects = await self.service.list_projects(status=ProjectStatus.ACTIVE)
            assert len(active_projects) >= 1, "No active projects found"
            
            self.record_result("Filtering", True, "Filtering works correctly")
        except Exception as e:
            self.record_result("Filtering", False, str(e))
    
    async def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*80)
        print("🧪 PROJECT MANAGEMENT MODULE TEST SUITE")
        print("="*80 + "\n")
        
        await self.setup()
        
        # Run tests
        await self.test_project_crud()
        await self.test_task_crud()
        await self.test_milestone_crud()
        await self.test_time_entry_crud()
        await self.test_project_stats_update()
        await self.test_statistics()
        await self.test_filtering()
        
        await self.cleanup()
        
        # Print summary
        print("\n" + "="*80)
        print("📊 TEST SUMMARY")
        print("="*80)
        passed = sum(1 for r in self.test_results if r["passed"])
        total = len(self.test_results)
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Pass Rate: {(passed/total*100):.1f}%")
        print("="*80 + "\n")
        
        return passed == total

async def main():
    tester = TestProjectsModule()
    success = await tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())

