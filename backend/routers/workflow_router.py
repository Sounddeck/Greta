"""
Workflow Router

API endpoints for workflow management and execution
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio

from backend.services.workflow_execution_service import WorkflowExecutionService
from backend.services.event_bus import EventBus

router = APIRouter(prefix="/api/v1/workflows", tags=["workflows"])

# Dependency injection
def get_workflow_service() -> WorkflowExecutionService:
    return WorkflowExecutionService()

def get_event_bus() -> EventBus:
    return EventBus()


# Request/Response Models
class WorkflowNode(BaseModel):
    id: str
    type: str
    position: Dict[str, float]
    data: Dict[str, Any]


class WorkflowEdge(BaseModel):
    id: str
    source: str
    target: str
    sourceHandle: Optional[str] = None
    targetHandle: Optional[str] = None
    type: Optional[str] = "default"


class WorkflowDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    nodes: List[WorkflowNode]
    edges: List[WorkflowEdge]


class WorkflowExecutionRequest(BaseModel):
    workflow_id: str
    input_data: Optional[Dict[str, Any]] = None


class WorkflowExecutionResponse(BaseModel):
    execution_id: str
    workflow_id: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# Endpoints
@router.post("/", response_model=Dict[str, Any])
async def create_workflow(
    workflow: WorkflowDefinition,
    service: WorkflowExecutionService = Depends(get_workflow_service)
):
    """Create a new workflow"""
    try:
        workflow_id = await service.create_workflow(
            name=workflow.name,
            description=workflow.description,
            nodes=workflow.nodes,
            edges=workflow.edges
        )
        
        return {
            "workflow_id": workflow_id,
            "name": workflow.name,
            "status": "created",
            "created_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}", response_model=Dict[str, Any])
async def get_workflow(
    workflow_id: str,
    service: WorkflowExecutionService = Depends(get_workflow_service)
):
    """Get workflow by ID"""
    try:
        workflow = await service.get_workflow(workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        return workflow
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{workflow_id}/execute", response_model=WorkflowExecutionResponse)
async def execute_workflow(
    workflow_id: str,
    request: WorkflowExecutionRequest,
    service: WorkflowExecutionService = Depends(get_workflow_service),
    event_bus: EventBus = Depends(get_event_bus)
):
    """Execute a workflow"""
    try:
        # Start workflow execution
        execution_id = await service.start_execution(
            workflow_id=workflow_id,
            input_data=request.input_data or {}
        )
        
        # Publish workflow started event
        await event_bus.publish(
            event_type="workflow.started",
            data={
                "execution_id": execution_id,
                "workflow_id": workflow_id,
                "started_at": datetime.utcnow().isoformat()
            }
        )
        
        return WorkflowExecutionResponse(
            execution_id=execution_id,
            workflow_id=workflow_id,
            status="running",
            started_at=datetime.utcnow()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}/executions/{execution_id}", response_model=WorkflowExecutionResponse)
async def get_execution_status(
    workflow_id: str,
    execution_id: str,
    service: WorkflowExecutionService = Depends(get_workflow_service)
):
    """Get execution status"""
    try:
        execution = await service.get_execution(execution_id)
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        return WorkflowExecutionResponse(
            execution_id=execution["execution_id"],
            workflow_id=execution["workflow_id"],
            status=execution["status"],
            started_at=execution["started_at"],
            completed_at=execution.get("completed_at"),
            result=execution.get("result"),
            error=execution.get("error")
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{workflow_id}/executions/{execution_id}/stop")
async def stop_execution(
    workflow_id: str,
    execution_id: str,
    service: WorkflowExecutionService = Depends(get_workflow_service),
    event_bus: EventBus = Depends(get_event_bus)
):
    """Stop a running execution"""
    try:
        await service.stop_execution(execution_id)
        
        # Publish workflow stopped event
        await event_bus.publish(
            event_type="workflow.stopped",
            data={
                "execution_id": execution_id,
                "workflow_id": workflow_id,
                "stopped_at": datetime.utcnow().isoformat()
            }
        )
        
        return {"status": "stopped", "execution_id": execution_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[Dict[str, Any]])
async def list_workflows(
    service: WorkflowExecutionService = Depends(get_workflow_service)
):
    """List all workflows"""
    try:
        workflows = await service.list_workflows()
        return workflows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{workflow_id}")
async def delete_workflow(
    workflow_id: str,
    service: WorkflowExecutionService = Depends(get_workflow_service)
):
    """Delete a workflow"""
    try:
        await service.delete_workflow(workflow_id)
        return {"status": "deleted", "workflow_id": workflow_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

