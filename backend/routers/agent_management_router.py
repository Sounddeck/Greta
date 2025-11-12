"""
GRETA Agent Management Router
Complete API endpoints for agent lifecycle management
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
from services.agent_management_service import AgentManagementService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/agent-management", tags=["Agent Management"])

# Global instance
agent_mgmt_instance: Optional[AgentManagementService] = None

# Pydantic models
class AgentCreateRequest(BaseModel):
    """Agent creation request"""
    name: str = Field(..., description="Agent name")
    type: str = Field(..., description="Agent type (autonomous, reactive, proactive, collaborative, specialized)")
    config: Dict[str, Any] = Field(default_factory=dict, description="Agent configuration")

class AgentUpdateRequest(BaseModel):
    """Agent update request"""
    updates: Dict[str, Any] = Field(..., description="Fields to update")

class TaskExecuteRequest(BaseModel):
    """Task execution request"""
    task: Dict[str, Any] = Field(..., description="Task to execute")

class VersionCreateRequest(BaseModel):
    """Version creation request"""
    config: Dict[str, Any] = Field(..., description="Version configuration")
    changes: str = Field(..., description="Description of changes")
    created_by: Optional[str] = Field("system", description="Creator")

class PerformanceRecordRequest(BaseModel):
    """Performance recording request"""
    response_time: float = Field(..., description="Response time in seconds")
    success: bool = Field(..., description="Whether the operation was successful")

class ABTestCreateRequest(BaseModel):
    """A/B test creation request"""
    name: str = Field(..., description="Test name")
    variant_a: str = Field(..., description="Variant A agent ID or config")
    variant_b: str = Field(..., description="Variant B agent ID or config")
    metric: Optional[str] = Field("success_rate", description="Metric to measure")
    traffic_split: Optional[float] = Field(0.5, description="Traffic split (0.0-1.0)")

class ABTestResultRequest(BaseModel):
    """A/B test result recording request"""
    variant: str = Field(..., description="Variant (variant_a or variant_b)")
    success: bool = Field(..., description="Whether the operation was successful")
    response_time: float = Field(..., description="Response time in seconds")

async def get_agent_mgmt() -> AgentManagementService:
    """Get or initialize Agent Management instance"""
    global agent_mgmt_instance

    if agent_mgmt_instance is None:
        db = database.database
        llm_service = await get_llm_service()
        learning_service = LearningService(db.client if hasattr(db, 'client') else None)
        await learning_service.initialize()

        agent_mgmt_instance = AgentManagementService(db, llm_service, learning_service)
        await agent_mgmt_instance.initialize()
    
    return agent_mgmt_instance

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Agent Management Framework",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/metrics")
async def get_metrics(agent_mgmt: AgentManagementService = Depends(get_agent_mgmt)):
    """Get agent management metrics"""
    try:
        metrics = await agent_mgmt.get_metrics()
        return metrics
    except Exception as e:
        logger.error(f"❌ Get metrics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agent")
async def create_agent(
    request: AgentCreateRequest,
    agent_mgmt: AgentManagementService = Depends(get_agent_mgmt)
):
    """Create a new agent"""
    try:
        agent_config = {
            "name": request.name,
            "type": request.type,
            **request.config
        }
        
        result = await agent_mgmt.create_agent(agent_config)
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Agent creation failed"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Agent creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/agent/{agent_id}")
async def update_agent(
    agent_id: str,
    request: AgentUpdateRequest,
    agent_mgmt: AgentManagementService = Depends(get_agent_mgmt)
):
    """Update an existing agent"""
    try:
        result = await agent_mgmt.update_agent(agent_id, request.updates)
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Agent update failed"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Agent update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/agent/{agent_id}")
async def delete_agent(
    agent_id: str,
    agent_mgmt: AgentManagementService = Depends(get_agent_mgmt)
):
    """Delete an agent"""
    try:
        result = await agent_mgmt.delete_agent(agent_id)
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Agent deletion failed"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Agent deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents")
async def list_agents(
    status: Optional[str] = None,
    type: Optional[str] = None,
    agent_mgmt: AgentManagementService = Depends(get_agent_mgmt)
):
    """List all agents"""
    try:
        filters = {}
        if status:
            filters["status"] = status
        if type:
            filters["type"] = type
        
        agents = await agent_mgmt.list_agents(filters)
        
        return {
            "success": True,
            "agents": agents,
            "count": len(agents)
        }
        
    except Exception as e:
        logger.error(f"❌ List agents failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agent/{agent_id}")
async def get_agent(
    agent_id: str,
    agent_mgmt: AgentManagementService = Depends(get_agent_mgmt)
):
    """Get a specific agent"""
    try:
        agent = await agent_mgmt.get_agent(agent_id)
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return {
            "success": True,
            "agent": agent
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Get agent failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agent/{agent_id}/execute")
async def execute_task(
    agent_id: str,
    request: TaskExecuteRequest,
    agent_mgmt: AgentManagementService = Depends(get_agent_mgmt)
):
    """Execute a task with an agent"""
    try:
        result = await agent_mgmt.execute_task(agent_id, request.task)
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Task execution failed"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Task execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agent-types")
async def get_agent_types():
    """Get available agent types"""
    return {
        "success": True,
        "agent_types": [
            {
                "type": "autonomous",
                "name": "Autonomous Agent",
                "description": "Self-directed agent with decision-making capabilities",
                "capabilities": ["decision_making", "task_execution", "learning"]
            },
            {
                "type": "reactive",
                "name": "Reactive Agent",
                "description": "Event-driven agent that responds to stimuli",
                "capabilities": ["event_response", "monitoring"]
            },
            {
                "type": "proactive",
                "name": "Proactive Agent",
                "description": "Forward-thinking agent with planning capabilities",
                "capabilities": ["prediction", "planning", "optimization"]
            },
            {
                "type": "collaborative",
                "name": "Collaborative Agent",
                "description": "Team-oriented agent for multi-agent systems",
                "capabilities": ["communication", "coordination", "teamwork"]
            },
            {
                "type": "specialized",
                "name": "Specialized Agent",
                "description": "Domain-specific agent with expert knowledge",
                "capabilities": ["domain_expertise", "precision_tasks"]
            }
        ]
    }

# ==================== VERSIONING ====================

@router.post("/agent/{agent_id}/version")
async def create_version(
    agent_id: str,
    request: VersionCreateRequest,
    agent_mgmt: AgentManagementService = Depends(get_agent_mgmt)
):
    """Create a new version of an agent"""
    try:
        version_config = {
            "config": request.config,
            "changes": request.changes,
            "created_by": request.created_by
        }

        result = await agent_mgmt.create_version(agent_id, version_config)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Version creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agent/{agent_id}/versions")
async def get_versions(
    agent_id: str,
    agent_mgmt: AgentManagementService = Depends(get_agent_mgmt)
):
    """Get all versions of an agent"""
    try:
        result = await agent_mgmt.get_versions(agent_id)
        return result

    except Exception as e:
        logger.error(f"Get versions failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agent/{agent_id}/rollback/{version_number}")
async def rollback_version(
    agent_id: str,
    version_number: int,
    agent_mgmt: AgentManagementService = Depends(get_agent_mgmt)
):
    """Rollback agent to a specific version"""
    try:
        result = await agent_mgmt.rollback_version(agent_id, version_number)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Version rollback failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== PERFORMANCE MONITORING ====================

@router.post("/agent/{agent_id}/performance")
async def record_performance(
    agent_id: str,
    request: PerformanceRecordRequest,
    agent_mgmt: AgentManagementService = Depends(get_agent_mgmt)
):
    """Record performance metrics for an agent"""
    try:
        result = await agent_mgmt.record_performance(
            agent_id,
            request.response_time,
            request.success
        )

        return result

    except Exception as e:
        logger.error(f"Performance recording failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agent/{agent_id}/performance")
async def get_performance(
    agent_id: str,
    agent_mgmt: AgentManagementService = Depends(get_agent_mgmt)
):
    """Get performance metrics for an agent"""
    try:
        result = await agent_mgmt.get_performance(agent_id)

        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get performance failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== A/B TESTING ====================

@router.post("/ab-test")
async def create_ab_test(
    request: ABTestCreateRequest,
    agent_mgmt: AgentManagementService = Depends(get_agent_mgmt)
):
    """Create an A/B test"""
    try:
        test_config = {
            "name": request.name,
            "variant_a": request.variant_a,
            "variant_b": request.variant_b,
            "metric": request.metric,
            "traffic_split": request.traffic_split
        }

        result = await agent_mgmt.create_ab_test(test_config)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"A/B test creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ab-test/{test_id}/result")
async def record_ab_result(
    test_id: str,
    request: ABTestResultRequest,
    agent_mgmt: AgentManagementService = Depends(get_agent_mgmt)
):
    """Record result for an A/B test"""
    try:
        result = await agent_mgmt.record_ab_result(
            test_id,
            request.variant,
            request.success,
            request.response_time
        )

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"A/B result recording failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ab-test/{test_id}/results")
async def get_ab_test_results(
    test_id: str,
    agent_mgmt: AgentManagementService = Depends(get_agent_mgmt)
):
    """Get results of an A/B test"""
    try:
        result = await agent_mgmt.get_ab_test_results(test_id)

        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get A/B test results failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ab-test/{test_id}/stop")
async def stop_ab_test(
    test_id: str,
    agent_mgmt: AgentManagementService = Depends(get_agent_mgmt)
):
    """Stop an A/B test"""
    try:
        result = await agent_mgmt.stop_ab_test(test_id)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stop A/B test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ab-tests")
async def get_all_ab_tests(agent_mgmt: AgentManagementService = Depends(get_agent_mgmt)):
    """Get all A/B tests"""
    try:
        result = await agent_mgmt.get_all_ab_tests()
        return result

    except Exception as e:
        logger.error(f"Get A/B tests failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ACTIVITY LOG ====================

@router.get("/activity-log")
async def get_activity_log(
    limit: int = 100,
    agent_mgmt: AgentManagementService = Depends(get_agent_mgmt)
):
    """Get activity log"""
    try:
        result = await agent_mgmt.get_activity_log(limit)
        return result

    except Exception as e:
        logger.error(f"Get activity log failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

