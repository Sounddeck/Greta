"""
Project Management Router for CPAS4
REST API endpoints for Project Management
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
import logging
from services.projects_service import ProjectsService
from models.projects_models import *
from database import Database

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/projects", tags=["Projects"])

async def get_service() -> ProjectsService:
    db = Database()
    await db.connect()
    return ProjectsService(db)

# Project endpoints
@router.post("/", response_model=Project, status_code=201)
async def create_project(project: ProjectCreate, svc: ProjectsService = Depends(get_service)):
    return await svc.create_project(project)

@router.get("/{project_id}", response_model=Project)
async def get_project(project_id: str, svc: ProjectsService = Depends(get_service)):
    prj = await svc.get_project(project_id)
    if not prj: raise HTTPException(404, "Project not found")
    return prj

@router.patch("/{project_id}", response_model=Project)
async def update_project(project_id: str, project_update: ProjectUpdate, svc: ProjectsService = Depends(get_service)):
    prj = await svc.update_project(project_id, project_update)
    if not prj: raise HTTPException(404, "Project not found")
    return prj

@router.delete("/{project_id}", status_code=204)
async def delete_project(project_id: str, svc: ProjectsService = Depends(get_service)):
    if not await svc.delete_project(project_id): raise HTTPException(404, "Project not found")

@router.get("/", response_model=List[Project])
async def list_projects(status: Optional[ProjectStatus] = None, limit: int = Query(100, le=1000), svc: ProjectsService = Depends(get_service)):
    return await svc.list_projects(status, limit)

# Task endpoints
@router.post("/tasks", response_model=Task, status_code=201)
async def create_task(task: TaskCreate, svc: ProjectsService = Depends(get_service)):
    return await svc.create_task(task)

@router.get("/tasks/{task_id}", response_model=Task)
async def get_task(task_id: str, svc: ProjectsService = Depends(get_service)):
    tsk = await svc.get_task(task_id)
    if not tsk: raise HTTPException(404, "Task not found")
    return tsk

@router.patch("/tasks/{task_id}", response_model=Task)
async def update_task(task_id: str, task_update: TaskUpdate, svc: ProjectsService = Depends(get_service)):
    tsk = await svc.update_task(task_id, task_update)
    if not tsk: raise HTTPException(404, "Task not found")
    return tsk

@router.delete("/tasks/{task_id}", status_code=204)
async def delete_task(task_id: str, svc: ProjectsService = Depends(get_service)):
    if not await svc.delete_task(task_id): raise HTTPException(404, "Task not found")

@router.get("/tasks", response_model=List[Task])
async def list_tasks(project_id: Optional[str] = None, status: Optional[TaskStatus] = None, assignee_id: Optional[str] = None, limit: int = Query(100, le=1000), svc: ProjectsService = Depends(get_service)):
    return await svc.list_tasks(project_id, status, assignee_id, limit)

# Milestone endpoints
@router.post("/milestones", response_model=Milestone, status_code=201)
async def create_milestone(milestone: MilestoneCreate, svc: ProjectsService = Depends(get_service)):
    return await svc.create_milestone(milestone)

@router.get("/milestones/{milestone_id}", response_model=Milestone)
async def get_milestone(milestone_id: str, svc: ProjectsService = Depends(get_service)):
    mls = await svc.get_milestone(milestone_id)
    if not mls: raise HTTPException(404, "Milestone not found")
    return mls

@router.patch("/milestones/{milestone_id}", response_model=Milestone)
async def update_milestone(milestone_id: str, milestone_update: MilestoneUpdate, svc: ProjectsService = Depends(get_service)):
    mls = await svc.update_milestone(milestone_id, milestone_update)
    if not mls: raise HTTPException(404, "Milestone not found")
    return mls

@router.delete("/milestones/{milestone_id}", status_code=204)
async def delete_milestone(milestone_id: str, svc: ProjectsService = Depends(get_service)):
    if not await svc.delete_milestone(milestone_id): raise HTTPException(404, "Milestone not found")

@router.get("/milestones", response_model=List[Milestone])
async def list_milestones(project_id: Optional[str] = None, status: Optional[MilestoneStatus] = None, limit: int = Query(100, le=1000), svc: ProjectsService = Depends(get_service)):
    return await svc.list_milestones(project_id, status, limit)

# Time Entry endpoints
@router.post("/time-entries", response_model=TimeEntry, status_code=201)
async def create_time_entry(time_entry: TimeEntryCreate, svc: ProjectsService = Depends(get_service)):
    return await svc.create_time_entry(time_entry)

@router.get("/time-entries/{entry_id}", response_model=TimeEntry)
async def get_time_entry(entry_id: str, svc: ProjectsService = Depends(get_service)):
    te = await svc.get_time_entry(entry_id)
    if not te: raise HTTPException(404, "Time entry not found")
    return te

@router.get("/time-entries", response_model=List[TimeEntry])
async def list_time_entries(project_id: Optional[str] = None, task_id: Optional[str] = None, user_id: Optional[str] = None, limit: int = Query(100, le=1000), svc: ProjectsService = Depends(get_service)):
    return await svc.list_time_entries(project_id, task_id, user_id, limit)

# Statistics
@router.get("/stats", response_model=ProjectStats)
async def get_stats(svc: ProjectsService = Depends(get_service)):
    return await svc.get_stats()

# Health check
@router.get("/health")
async def health_check():
    return {"status": "healthy", "module": "Projects", "features": ["Project Management", "Task Tracking", "Milestone Management", "Time Tracking", "Resource Management", "Statistics"], "endpoints": 19, "database": "MongoDB/File Storage"}

