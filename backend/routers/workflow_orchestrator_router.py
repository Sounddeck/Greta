"""
Workflow Orchestrator Router

REST API endpoints for intelligent workflow orchestration
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from services.workflow_orchestrator import WorkflowOrchestrator
from services.tool_registry import tool_registry
from services.reasoning_bank_service import ReasoningBankService
from services.vector_search_service import VectorSearchService
from services.skills_service import skills_service
from database import get_database

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/workflows", tags=["Workflow Orchestration"])


# ==================== REQUEST/RESPONSE MODELS ====================

class WorkflowExecuteRequest(BaseModel):
    """Request to execute a workflow"""
    task_description: str = Field(..., description="High-level task description")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    user_id: str = Field(default="system", description="User identifier")


class WorkflowExecuteResponse(BaseModel):
    """Response from workflow execution"""
    workflow_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    subtasks_completed: Optional[int] = None
    total_subtasks: Optional[int] = None
    tools_used: Optional[list] = None
    patterns_used: Optional[list] = None
    execution_time: float
    error: Optional[str] = None


class WorkflowStatusResponse(BaseModel):
    """Workflow status response"""
    workflow_id: str
    task_description: str
    status: str
    start_time: str
    subtasks: list
    tools_used: list
    patterns_used: list
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float


# ==================== DEPENDENCY INJECTION ====================

_orchestrator_instance = None


async def get_orchestrator() -> WorkflowOrchestrator:
    """Get or create Workflow Orchestrator instance"""
    global _orchestrator_instance

    if _orchestrator_instance is None:
        try:
            # Get dependencies
            db = await get_database()

            # Initialize Vector Search Service
            try:
                vector_search = VectorSearchService()
                await vector_search.initialize()
            except Exception as e:
                logger.warning(f"Vector Search not available: {e}")
                vector_search = None

            # Initialize ReasoningBank Service
            try:
                reasoning_bank = ReasoningBankService(db, vector_search)
            except Exception as e:
                logger.warning(f"ReasoningBank not available: {e}")
                reasoning_bank = None
            
            # Create orchestrator
            _orchestrator_instance = WorkflowOrchestrator(
                tool_registry=tool_registry,
                reasoning_bank=reasoning_bank,
                vector_search=vector_search,
                skills_service=skills_service,
                db=db
            )
            
            logger.info("✅ Workflow Orchestrator initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Workflow Orchestrator: {e}")
            raise HTTPException(status_code=500, detail=f"Orchestrator initialization failed: {e}")
    
    return _orchestrator_instance


# ==================== API ENDPOINTS ====================

@router.post("/execute", response_model=WorkflowExecuteResponse)
async def execute_workflow(
    request: WorkflowExecuteRequest,
    orchestrator: WorkflowOrchestrator = Depends(get_orchestrator)
):
    """
    Execute an intelligent workflow
    
    This endpoint:
    1. Retrieves RAG context from similar past executions
    2. Finds matching patterns from ReasoningBank
    3. Decomposes task into subtasks
    4. Selects appropriate tools for each subtask
    5. Executes workflow with monitoring
    6. Stores results for future learning
    """
    try:
        logger.info(f"🚀 Executing workflow: {request.task_description[:100]}")
        
        result = await orchestrator.execute_workflow(
            task_description=request.task_description,
            context=request.context,
            user_id=request.user_id
        )
        
        return WorkflowExecuteResponse(**result)
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{workflow_id}", response_model=WorkflowStatusResponse)
async def get_workflow_status(
    workflow_id: str,
    orchestrator: WorkflowOrchestrator = Depends(get_orchestrator)
):
    """Get status of a specific workflow"""
    try:
        workflow = orchestrator.get_workflow_status(workflow_id)
        
        if not workflow:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
        
        return WorkflowStatusResponse(
            workflow_id=workflow["workflow_id"],
            task_description=workflow["task_description"],
            status=workflow["status"],
            start_time=workflow["start_time"].isoformat(),
            subtasks=workflow["subtasks"],
            tools_used=workflow.get("tools_used", []),
            patterns_used=workflow.get("patterns_used", []),
            result=workflow.get("result"),
            error=workflow.get("error"),
            execution_time=workflow.get("execution_time", 0.0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/active")
async def get_active_workflows(
    orchestrator: WorkflowOrchestrator = Depends(get_orchestrator)
):
    """Get all currently active workflows"""
    try:
        workflows = orchestrator.get_active_workflows()
        
        return {
            "active_workflows": len(workflows),
            "workflows": [
                {
                    "workflow_id": w["workflow_id"],
                    "task_description": w["task_description"],
                    "status": w["status"],
                    "start_time": w["start_time"].isoformat(),
                    "subtasks_count": len(w["subtasks"])
                }
                for w in workflows
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to get active workflows: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_workflow_history(
    limit: int = 10,
    orchestrator: WorkflowOrchestrator = Depends(get_orchestrator)
):
    """Get recent workflow execution history"""
    try:
        history = orchestrator.get_workflow_history(limit=limit)
        
        return {
            "total": len(history),
            "workflows": [
                {
                    "workflow_id": w["workflow_id"],
                    "task_description": w["task_description"],
                    "status": w["status"],
                    "start_time": w["start_time"].isoformat(),
                    "execution_time": w.get("execution_time", 0.0),
                    "subtasks_completed": len([s for s in w["subtasks"] if s["status"] == "completed"]),
                    "total_subtasks": len(w["subtasks"])
                }
                for w in history
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to get workflow history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check(orchestrator: WorkflowOrchestrator = Depends(get_orchestrator)):
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "workflow_orchestrator",
        "active_workflows": len(orchestrator.get_active_workflows()),
        "components": {
            "tool_registry": orchestrator.tool_registry is not None,
            "reasoning_bank": orchestrator.reasoning_bank is not None,
            "vector_search": orchestrator.vector_search is not None,
            "skills_service": orchestrator.skills_service is not None
        }
    }

