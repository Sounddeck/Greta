"""
Multi-Agent Collaboration Router

REST API endpoints for multi-agent collaboration.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

from database import get_database
from services.multi_agent_collaboration import MultiAgentCollaborationService

router = APIRouter(prefix="/api/v1/multi-agent", tags=["Multi-Agent Collaboration"])


# Request/Response Models
class TeamFormationRequest(BaseModel):
    """Request to form an agent team"""
    task_description: str
    required_capabilities: List[str]
    max_team_size: int = Field(default=5, ge=1, le=20)


class SendMessageRequest(BaseModel):
    """Request to send inter-agent message"""
    from_agent: str
    message_type: str
    content: Dict[str, Any]
    to_agent: Optional[str] = None
    requires_response: bool = False


class ParallelExecutionRequest(BaseModel):
    """Request to execute parallel tasks"""
    team_id: str
    tasks: List[Dict[str, Any]]


class ConsensusRequest(BaseModel):
    """Request to reach consensus"""
    task_id: str
    team_id: str
    decision_options: List[str]
    timeout_seconds: int = Field(default=30, ge=5, le=300)


# Dependency to get service
async def get_collaboration_service():
    """Get multi-agent collaboration service instance"""
    db = await get_database()
    return MultiAgentCollaborationService(db)


@router.post("/form-team")
async def form_team(
    request: TeamFormationRequest,
    service: MultiAgentCollaborationService = Depends(get_collaboration_service)
):
    """
    Form an agent team based on task requirements
    
    **Example:**
    ```json
    {
      "task_description": "Analyze market data and create report",
      "required_capabilities": ["data_analysis", "report_generation"],
      "max_team_size": 3
    }
    ```
    """
    result = await service.form_team(
        task_description=request.task_description,
        required_capabilities=request.required_capabilities,
        max_team_size=request.max_team_size
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Team formation failed"))
    
    return result


@router.post("/send-message")
async def send_message(
    request: SendMessageRequest,
    service: MultiAgentCollaborationService = Depends(get_collaboration_service)
):
    """
    Send message between agents
    
    **Example:**
    ```json
    {
      "from_agent": "agent_123",
      "message_type": "request",
      "content": {"action": "analyze_data", "data_id": "dataset_456"},
      "to_agent": "agent_789",
      "requires_response": true
    }
    ```
    """
    message = await service.send_message(
        from_agent=request.from_agent,
        message_type=request.message_type,
        content=request.content,
        to_agent=request.to_agent,
        requires_response=request.requires_response
    )
    
    return {
        "success": True,
        "message_id": message.message_id,
        "timestamp": message.timestamp.isoformat()
    }


@router.post("/execute-parallel")
async def execute_parallel_tasks(
    request: ParallelExecutionRequest,
    service: MultiAgentCollaborationService = Depends(get_collaboration_service)
):
    """
    Execute multiple tasks in parallel across team
    
    **Example:**
    ```json
    {
      "team_id": "team_123",
      "tasks": [
        {"id": "task_1", "action": "analyze", "data": "dataset_1"},
        {"id": "task_2", "action": "analyze", "data": "dataset_2"}
      ]
    }
    ```
    """
    result = await service.execute_parallel_tasks(
        team_id=request.team_id,
        tasks=request.tasks
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Parallel execution failed"))
    
    return result


@router.post("/reach-consensus")
async def reach_consensus(
    request: ConsensusRequest,
    service: MultiAgentCollaborationService = Depends(get_collaboration_service)
):
    """
    Reach consensus among team members
    
    **Example:**
    ```json
    {
      "task_id": "task_123",
      "team_id": "team_456",
      "decision_options": ["option_a", "option_b", "option_c"],
      "timeout_seconds": 30
    }
    ```
    """
    result = await service.reach_consensus(
        task_id=request.task_id,
        team_id=request.team_id,
        decision_options=request.decision_options,
        timeout_seconds=request.timeout_seconds
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Consensus failed"))
    
    return result


@router.get("/team/{team_id}/status")
async def get_team_status(
    team_id: str,
    service: MultiAgentCollaborationService = Depends(get_collaboration_service)
):
    """
    Get status of a team
    
    **Example:** GET /api/v1/multi-agent/team/team_123/status
    """
    result = await service.get_team_status(team_id)
    
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error", "Team not found"))
    
    return result


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "multi-agent-collaboration",
        "timestamp": datetime.utcnow().isoformat()
    }

