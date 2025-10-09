"""
Project Management Service for CPAS4 - Real Database Integration
Complete project management with projects, tasks, milestones, and time tracking
"""
import logging
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
from database import Database
from models.projects_models import *

logger = logging.getLogger(__name__)

class ProjectsService:
    def __init__(self, database: Database):
        self.db = database
        self.projects_collection = "projects_projects"
        self.tasks_collection = "projects_tasks"
        self.milestones_collection = "projects_milestones"
        self.time_entries_collection = "projects_time_entries"
        self.resources_collection = "projects_resources"
        logger.info("✅ Projects Service initialized with real database")
    
    # Project CRUD
    async def create_project(self, project_create: ProjectCreate) -> Project:
        project_id = f"prj_{uuid.uuid4().hex[:12]}"
        project = Project(project_id=project_id, **project_create.model_dump(), created_at=datetime.utcnow(), updated_at=datetime.utcnow())
        pd = project.model_dump()
        for k in ["created_at", "updated_at", "start_date", "end_date"]:
            if pd.get(k): pd[k] = pd[k].isoformat()
        await self.db.insert_one(self.projects_collection, pd)
        logger.info(f"✅ Project created: {project_id}")
        return project
    
    async def get_project(self, project_id: str) -> Optional[Project]:
        pd = await self.db.find_one(self.projects_collection, {"project_id": project_id})
        if not pd: return None
        for k in ["created_at", "updated_at", "start_date", "end_date"]:
            if k in pd and pd[k] and isinstance(pd[k], str): pd[k] = datetime.fromisoformat(pd[k])
        return Project(**pd)
    
    async def update_project(self, project_id: str, project_update: ProjectUpdate) -> Optional[Project]:
        ud = project_update.model_dump(exclude_unset=True)
        ud["updated_at"] = datetime.utcnow().isoformat()
        if "end_date" in ud and ud["end_date"]: ud["end_date"] = ud["end_date"].isoformat()
        return await self.get_project(project_id) if await self.db.update_one(self.projects_collection, {"project_id": project_id}, {"$set": ud}) else None
    
    async def delete_project(self, project_id: str) -> bool:
        return await self.db.delete_one(self.projects_collection, {"project_id": project_id})
    
    async def list_projects(self, status: Optional[ProjectStatus] = None, limit: int = 100) -> List[Project]:
        f = {}
        if status: f["status"] = status
        pds = await self.db.find_many(self.projects_collection, f, limit=limit)
        projects = []
        for pd in pds:
            for k in ["created_at", "updated_at", "start_date", "end_date"]:
                if k in pd and pd[k] and isinstance(pd[k], str): pd[k] = datetime.fromisoformat(pd[k])
            projects.append(Project(**pd))
        return projects
    
    async def update_project_stats(self, project_id: str):
        tasks = await self.list_tasks(project_id=project_id, limit=10000)
        time_entries = await self.list_time_entries(project_id=project_id, limit=10000)
        total_tasks = len(tasks)
        completed_tasks = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
        progress = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        total_hours = sum(te.duration_hours or 0 for te in time_entries)
        await self.db.update_one(self.projects_collection, {"project_id": project_id}, {"$set": {"total_tasks": total_tasks, "completed_tasks": completed_tasks, "progress": round(progress, 2), "total_hours": round(total_hours, 2)}})
    
    # Task CRUD
    async def create_task(self, task_create: TaskCreate) -> Task:
        task_id = f"tsk_{uuid.uuid4().hex[:12]}"
        task = Task(task_id=task_id, **task_create.model_dump(), created_at=datetime.utcnow(), updated_at=datetime.utcnow())
        td = task.model_dump()
        for k in ["created_at", "updated_at", "due_date", "started_at", "completed_at"]:
            if td.get(k): td[k] = td[k].isoformat()
        await self.db.insert_one(self.tasks_collection, td)
        await self.update_project_stats(task.project_id)
        logger.info(f"✅ Task created: {task_id}")
        return task
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        td = await self.db.find_one(self.tasks_collection, {"task_id": task_id})
        if not td: return None
        for k in ["created_at", "updated_at", "due_date", "started_at", "completed_at"]:
            if k in td and td[k] and isinstance(td[k], str): td[k] = datetime.fromisoformat(td[k])
        return Task(**td)
    
    async def update_task(self, task_id: str, task_update: TaskUpdate) -> Optional[Task]:
        ud = task_update.model_dump(exclude_unset=True)
        ud["updated_at"] = datetime.utcnow().isoformat()
        if "status" in ud:
            if ud["status"] == TaskStatus.IN_PROGRESS:
                task = await self.get_task(task_id)
                if task and not task.started_at: ud["started_at"] = datetime.utcnow().isoformat()
            elif ud["status"] == TaskStatus.COMPLETED:
                task = await self.get_task(task_id)
                if task and not task.completed_at:
                    ud["completed_at"] = datetime.utcnow().isoformat()
                    ud["progress"] = 100.0
        if "due_date" in ud and ud["due_date"]: ud["due_date"] = ud["due_date"].isoformat()
        task = await self.get_task(task_id)
        if task and await self.db.update_one(self.tasks_collection, {"task_id": task_id}, {"$set": ud}):
            await self.update_project_stats(task.project_id)
            return await self.get_task(task_id)
        return None
    
    async def delete_task(self, task_id: str) -> bool:
        task = await self.get_task(task_id)
        if task and await self.db.delete_one(self.tasks_collection, {"task_id": task_id}):
            await self.update_project_stats(task.project_id)
            return True
        return False
    
    async def list_tasks(self, project_id: Optional[str] = None, status: Optional[TaskStatus] = None, assignee_id: Optional[str] = None, limit: int = 100) -> List[Task]:
        f = {}
        if project_id: f["project_id"] = project_id
        if status: f["status"] = status
        if assignee_id: f["assignee_id"] = assignee_id
        tds = await self.db.find_many(self.tasks_collection, f, limit=limit)
        tasks = []
        for td in tds:
            for k in ["created_at", "updated_at", "due_date", "started_at", "completed_at"]:
                if k in td and td[k] and isinstance(td[k], str): td[k] = datetime.fromisoformat(td[k])
            tasks.append(Task(**td))
        return tasks
    
    # Milestone CRUD
    async def create_milestone(self, milestone_create: MilestoneCreate) -> Milestone:
        milestone_id = f"mls_{uuid.uuid4().hex[:12]}"
        milestone = Milestone(milestone_id=milestone_id, **milestone_create.model_dump(), created_at=datetime.utcnow(), updated_at=datetime.utcnow())
        md = milestone.model_dump()
        for k in ["created_at", "updated_at", "due_date", "completed_at"]:
            if md.get(k): md[k] = md[k].isoformat()
        await self.db.insert_one(self.milestones_collection, md)
        logger.info(f"✅ Milestone created: {milestone_id}")
        return milestone
    
    async def get_milestone(self, milestone_id: str) -> Optional[Milestone]:
        md = await self.db.find_one(self.milestones_collection, {"milestone_id": milestone_id})
        if not md: return None
        for k in ["created_at", "updated_at", "due_date", "completed_at"]:
            if k in md and md[k] and isinstance(md[k], str): md[k] = datetime.fromisoformat(md[k])
        return Milestone(**md)
    
    async def update_milestone(self, milestone_id: str, milestone_update: MilestoneUpdate) -> Optional[Milestone]:
        ud = milestone_update.model_dump(exclude_unset=True)
        ud["updated_at"] = datetime.utcnow().isoformat()
        if "status" in ud and ud["status"] == MilestoneStatus.COMPLETED:
            milestone = await self.get_milestone(milestone_id)
            if milestone and not milestone.completed_at: ud["completed_at"] = datetime.utcnow().isoformat()
        if "due_date" in ud and ud["due_date"]: ud["due_date"] = ud["due_date"].isoformat()
        return await self.get_milestone(milestone_id) if await self.db.update_one(self.milestones_collection, {"milestone_id": milestone_id}, {"$set": ud}) else None
    
    async def delete_milestone(self, milestone_id: str) -> bool:
        return await self.db.delete_one(self.milestones_collection, {"milestone_id": milestone_id})
    
    async def list_milestones(self, project_id: Optional[str] = None, status: Optional[MilestoneStatus] = None, limit: int = 100) -> List[Milestone]:
        f = {}
        if project_id: f["project_id"] = project_id
        if status: f["status"] = status
        mds = await self.db.find_many(self.milestones_collection, f, limit=limit)
        milestones = []
        for md in mds:
            for k in ["created_at", "updated_at", "due_date", "completed_at"]:
                if k in md and md[k] and isinstance(md[k], str): md[k] = datetime.fromisoformat(md[k])
            milestones.append(Milestone(**md))
        return milestones
    
    # Time Entry CRUD
    async def create_time_entry(self, time_entry_create: TimeEntryCreate) -> TimeEntry:
        entry_id = f"te_{uuid.uuid4().hex[:12]}"
        if time_entry_create.end_time and time_entry_create.start_time:
            duration = (time_entry_create.end_time - time_entry_create.start_time).total_seconds() / 3600
            time_entry_create.duration_hours = round(duration, 2)
        total_cost = (time_entry_create.duration_hours or 0) * (time_entry_create.hourly_rate or 0)
        time_entry = TimeEntry(entry_id=entry_id, **time_entry_create.model_dump(), total_cost=round(total_cost, 2), created_at=datetime.utcnow(), updated_at=datetime.utcnow())
        ted = time_entry.model_dump()
        for k in ["created_at", "updated_at", "start_time", "end_time"]:
            if ted.get(k): ted[k] = ted[k].isoformat()
        await self.db.insert_one(self.time_entries_collection, ted)
        await self.update_project_stats(time_entry.project_id)
        logger.info(f"✅ Time entry created: {entry_id}")
        return time_entry
    
    async def get_time_entry(self, entry_id: str) -> Optional[TimeEntry]:
        ted = await self.db.find_one(self.time_entries_collection, {"entry_id": entry_id})
        if not ted: return None
        for k in ["created_at", "updated_at", "start_time", "end_time"]:
            if k in ted and ted[k] and isinstance(ted[k], str): ted[k] = datetime.fromisoformat(ted[k])
        return TimeEntry(**ted)
    
    async def list_time_entries(self, project_id: Optional[str] = None, task_id: Optional[str] = None, user_id: Optional[str] = None, limit: int = 100) -> List[TimeEntry]:
        f = {}
        if project_id: f["project_id"] = project_id
        if task_id: f["task_id"] = task_id
        if user_id: f["user_id"] = user_id
        teds = await self.db.find_many(self.time_entries_collection, f, limit=limit)
        entries = []
        for ted in teds:
            for k in ["created_at", "updated_at", "start_time", "end_time"]:
                if k in ted and ted[k] and isinstance(ted[k], str): ted[k] = datetime.fromisoformat(ted[k])
            entries.append(TimeEntry(**ted))
        return entries
    
    # Statistics
    async def get_stats(self) -> ProjectStats:
        all_projects = await self.list_projects(limit=10000)
        all_tasks = await self.list_tasks(limit=10000)
        all_milestones = await self.list_milestones(limit=10000)
        all_time_entries = await self.list_time_entries(limit=10000)
        
        projects_by_status = {s.value: sum(1 for p in all_projects if p.status == s) for s in ProjectStatus}
        tasks_by_status = {s.value: sum(1 for t in all_tasks if t.status == s) for s in TaskStatus}
        tasks_by_priority = {p.value: sum(1 for t in all_tasks if t.priority == p) for p in TaskPriority}
        milestones_by_status = {s.value: sum(1 for m in all_milestones if m.status == s) for s in MilestoneStatus}
        time_by_type = {t.value: sum(te.duration_hours or 0 for te in all_time_entries if te.entry_type == t) for t in TimeEntryType}
        
        total_hours = sum(te.duration_hours or 0 for te in all_time_entries)
        billable_hours = sum(te.duration_hours or 0 for te in all_time_entries if te.billable)
        total_revenue = sum(te.total_cost for te in all_time_entries if te.billable)
        
        avg_project_completion = sum(p.progress for p in all_projects) / len(all_projects) if all_projects else 0
        avg_task_completion = sum(t.progress for t in all_tasks) / len(all_tasks) if all_tasks else 0
        
        return ProjectStats(
            total_projects=len(all_projects), active_projects=sum(1 for p in all_projects if p.status == ProjectStatus.ACTIVE),
            completed_projects=sum(1 for p in all_projects if p.status == ProjectStatus.COMPLETED),
            total_tasks=len(all_tasks), completed_tasks=sum(1 for t in all_tasks if t.status == TaskStatus.COMPLETED),
            total_milestones=len(all_milestones), completed_milestones=sum(1 for m in all_milestones if m.status == MilestoneStatus.COMPLETED),
            total_time_entries=len(all_time_entries), total_hours_logged=round(total_hours, 2),
            total_billable_hours=round(billable_hours, 2), total_revenue=round(total_revenue, 2),
            projects_by_status=projects_by_status, tasks_by_status=tasks_by_status,
            tasks_by_priority=tasks_by_priority, milestones_by_status=milestones_by_status,
            time_by_type={k: round(v, 2) for k, v in time_by_type.items()},
            average_project_completion=round(avg_project_completion, 2),
            average_task_completion=round(avg_task_completion, 2)
        )
