"""
GRETA Agent Builder Router
Complete API endpoints for agent development environment
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import sys
import os

# Add backend directory to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import database
from services.llm_service import get_llm_service
from services.learning_service import LearningService
from services.agent_builder_service import AgentBuilderService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/builder", tags=["Agent Builder"])

# Global instance
builder_instance: Optional[AgentBuilderService] = None

# Pydantic models
class ProjectCreateRequest(BaseModel):
    """Project creation request"""
    name: str = Field(..., description="Project name")
    description: Optional[str] = Field("", description="Project description")
    template: str = Field(..., description="Project template (basic, advanced, ml_agent, langchain)")
    created_by: Optional[str] = Field("system", description="Creator")

class FileUpdateRequest(BaseModel):
    """File update request"""
    filename: str = Field(..., description="Filename")
    content: str = Field(..., description="File content")

class FileAddRequest(BaseModel):
    """File add request"""
    filename: str = Field(..., description="Filename")
    content: Optional[str] = Field("", description="File content")

class BuildRequest(BaseModel):
    """Build request"""
    config: Optional[Dict[str, Any]] = Field(None, description="Build configuration")

class TestRequest(BaseModel):
    """Test request"""
    config: Optional[Dict[str, Any]] = Field(None, description="Test configuration")

class DeployRequest(BaseModel):
    """Deploy request"""
    environment: Optional[str] = Field("production", description="Deployment environment")

async def get_builder() -> AgentBuilderService:
    """Get or initialize Agent Builder instance"""
    global builder_instance
    
    if builder_instance is None:
        db = database.database
        llm_service = await get_llm_service()
        learning_service = LearningService(db.client if hasattr(db, 'client') else None)
        await learning_service.initialize()
        
        builder_instance = AgentBuilderService(db, llm_service, learning_service)
        await builder_instance.initialize()
    
    return builder_instance

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "agent_builder",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/metrics")
async def get_metrics(service: AgentBuilderService = Depends(get_builder)):
    """Get service metrics"""
    return await service.get_metrics()

# ==================== PROJECTS ====================

@router.post("/projects")
async def create_project(
    request: ProjectCreateRequest,
    service: AgentBuilderService = Depends(get_builder)
):
    """Create a new agent project"""
    try:
        result = await service.create_project(request.dict())

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating project: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/projects")
async def list_projects(
    status: Optional[str] = None,
    framework: Optional[str] = None,
    service: AgentBuilderService = Depends(get_builder)
):
    """List all projects"""
    try:
        filters = {}
        if status:
            filters["status"] = status
        if framework:
            filters["framework"] = framework

        result = await service.list_projects(filters)
        return result
    except Exception as e:
        logger.error(f"Error listing projects: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/projects/{project_id}")
async def get_project(
    project_id: str,
    service: AgentBuilderService = Depends(get_builder)
):
    """Get project details"""
    try:
        result = await service.get_project(project_id)

        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting project: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/projects/{project_id}")
async def delete_project(
    project_id: str,
    service: AgentBuilderService = Depends(get_builder)
):
    """Delete a project"""
    try:
        result = await service.delete_project(project_id)

        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting project: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== FILES ====================

@router.put("/projects/{project_id}/files")
async def update_file(
    project_id: str,
    request: FileUpdateRequest,
    service: AgentBuilderService = Depends(get_builder)
):
    """Update a file in the project"""
    try:
        result = await service.update_file(project_id, request.filename, request.content)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/projects/{project_id}/files")
async def add_file(
    project_id: str,
    request: FileAddRequest,
    service: AgentBuilderService = Depends(get_builder)
):
    """Add a new file to the project"""
    try:
        result = await service.add_file(project_id, request.filename, request.content)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/projects/{project_id}/files/{filename}")
async def delete_file(
    project_id: str,
    filename: str,
    service: AgentBuilderService = Depends(get_builder)
):
    """Delete a file from the project"""
    try:
        result = await service.delete_file(project_id, filename)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== BUILD ====================

@router.post("/projects/{project_id}/build")
async def build_project(
    project_id: str,
    request: BuildRequest,
    service: AgentBuilderService = Depends(get_builder)
):
    """Build an agent project"""
    try:
        result = await service.build_project(project_id, request.config)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error building project: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/builds/{build_id}")
async def get_build(
    build_id: str,
    service: AgentBuilderService = Depends(get_builder)
):
    """Get build details"""
    try:
        result = await service.get_build(build_id)

        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting build: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== TESTING ====================

@router.post("/projects/{project_id}/test")
async def run_tests(
    project_id: str,
    request: TestRequest,
    service: AgentBuilderService = Depends(get_builder)
):
    """Run tests for the project"""
    try:
        result = await service.run_tests(project_id, request.config)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tests/{test_id}")
async def get_test(
    test_id: str,
    service: AgentBuilderService = Depends(get_builder)
):
    """Get test details"""
    try:
        result = await service.get_test(test_id)

        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== DEPLOYMENT ====================

@router.post("/projects/{project_id}/deploy")
async def deploy_project(
    project_id: str,
    request: DeployRequest,
    service: AgentBuilderService = Depends(get_builder)
):
    """Deploy the project"""
    try:
        result = await service.deploy_project(project_id, request.dict())

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deploying project: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== TEMPLATES ====================

@router.get("/templates")
async def get_templates(service: AgentBuilderService = Depends(get_builder)):
    """Get available project templates"""
    try:
        result = await service.get_templates()
        return result
    except Exception as e:
        logger.error(f"Error getting templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ACTIVITY ====================

@router.get("/activity-log")
async def get_activity_log(
    limit: int = 100,
    service: AgentBuilderService = Depends(get_builder)
):
    """Get activity log"""
    try:
        result = await service.get_activity_log(limit)
        return result
    except Exception as e:
        logger.error(f"Error getting activity log: {e}")
        raise HTTPException(status_code=500, detail=str(e))

