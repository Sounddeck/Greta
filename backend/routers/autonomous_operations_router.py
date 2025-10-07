"""
GRETA Autonomous Operations Router
Complete API endpoints for workflow automation
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
from services.autonomous_operations_service import AutonomousOperationsService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/operations", tags=["Autonomous Operations"])

# Global instance
operations_instance: Optional[AutonomousOperationsService] = None

# Pydantic models
class WorkflowCreateRequest(BaseModel):
    """Workflow creation request"""
    name: str = Field(..., description="Workflow name")
    type: str = Field(..., description="Workflow type (sequential, parallel, conditional, loop, scheduled)")
    description: Optional[str] = Field(None, description="Workflow description")
    steps: List[Dict[str, Any]] = Field(..., description="Workflow steps")
    config: Optional[Dict[str, Any]] = Field(None, description="Additional configuration")

async def get_operations() -> AutonomousOperationsService:
    """Get or initialize Autonomous Operations instance"""
    global operations_instance

    if operations_instance is None:
        db = database.database
        llm_service = await get_llm_service()
        learning_service = LearningService(db.client if hasattr(db, 'client') else None)
        await learning_service.initialize()

        operations_instance = AutonomousOperationsService(db, llm_service, learning_service)
        await operations_instance.initialize()

    return operations_instance

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Autonomous Operations Center",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/metrics")
async def get_metrics(operations: AutonomousOperationsService = Depends(get_operations)):
    """Get operations metrics"""
    try:
        metrics = await operations.get_metrics()
        return metrics
    except Exception as e:
        logger.error(f"❌ Get metrics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== WORKFLOW MANAGEMENT ENDPOINTS ====================

@router.get("/templates")
async def get_templates(operations: AutonomousOperationsService = Depends(get_operations)):
    """Get all workflow templates"""
    try:
        result = await operations.get_templates()
        return result
    except Exception as e:
        logger.error(f"Error getting templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/workflows")
async def create_workflow(
    request: WorkflowCreateRequest,
    operations: AutonomousOperationsService = Depends(get_operations)
):
    """Create a new workflow"""
    try:
        workflow_config = {
            "name": request.name,
            "type": request.type,
            "description": request.description,
            "steps": request.steps,
            **(request.config or {})
        }

        result = await operations.create_workflow(workflow_config)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workflows")
async def get_all_workflows(operations: AutonomousOperationsService = Depends(get_operations)):
    """Get all workflows"""
    try:
        result = await operations.get_all_workflows()
        return result
    except Exception as e:
        logger.error(f"Error getting workflows: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workflows/{workflow_id}")
async def get_workflow(
    workflow_id: str,
    operations: AutonomousOperationsService = Depends(get_operations)
):
    """Get a specific workflow"""
    try:
        result = await operations.get_workflow(workflow_id)

        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/workflows/{workflow_id}")
async def delete_workflow(
    workflow_id: str,
    operations: AutonomousOperationsService = Depends(get_operations)
):
    """Delete a workflow"""
    try:
        result = await operations.delete_workflow(workflow_id)

        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== WORKFLOW EXECUTION ENDPOINTS ====================

@router.post("/workflows/{workflow_id}/execute")
async def execute_workflow(
    workflow_id: str,
    operations: AutonomousOperationsService = Depends(get_operations)
):
    """Execute a workflow"""
    try:
        result = await operations.execute_workflow(workflow_id)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/workflows/{workflow_id}/pause")
async def pause_workflow(
    workflow_id: str,
    operations: AutonomousOperationsService = Depends(get_operations)
):
    """Pause a running workflow"""
    try:
        result = await operations.pause_workflow(workflow_id)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error pausing workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/workflows/{workflow_id}/resume")
async def resume_workflow(
    workflow_id: str,
    operations: AutonomousOperationsService = Depends(get_operations)
):
    """Resume a paused workflow"""
    try:
        result = await operations.resume_workflow(workflow_id)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resuming workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/workflows/{workflow_id}/cancel")
async def cancel_workflow(
    workflow_id: str,
    operations: AutonomousOperationsService = Depends(get_operations)
):
    """Cancel a workflow"""
    try:
        result = await operations.cancel_workflow(workflow_id)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ACTIVITY & METRICS ENDPOINTS ====================

@router.get("/activity-log")
async def get_activity_log(
    limit: int = 100,
    operations: AutonomousOperationsService = Depends(get_operations)
):
    """Get activity log"""
    try:
        result = await operations.get_activity_log(limit)
        return result
    except Exception as e:
        logger.error(f"Error getting activity log: {e}")
        raise HTTPException(status_code=500, detail=str(e))

