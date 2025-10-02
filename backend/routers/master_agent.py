"""
GRETA PAI MASTER AGENT API ENDPOINTS
Exposes the complete master agent system through REST API
Fulfills original CPAS vision: master agent controls all aspects of Greta
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from utils.greta_master_agent import (
    initialize_greta_master_agent,
    execute_master_task,
    create_greta_agent,
    manage_greta_agents,
    greta_master_agent
)
from utils.error_handling import handle_errors
from utils.hooks import execute_hooks
from utils.ufc_context import ufc_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/master", tags=["master-agent"])

# Request/Response Models

class MasterTaskRequest(BaseModel):
    """Request for master agent task execution"""
    description: str = Field(..., description="Complex task description")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Task context and parameters")
    priority: Optional[str] = Field(default="medium", description="Task priority: low|medium|high")
    timeout: Optional[int] = Field(default=300, description="Timeout in seconds")

class AgentSpecRequest(BaseModel):
    """Request for custom agent creation"""
    spec: Dict[str, Any] = Field(..., description="Agent specification")

class AgentManagementRequest(BaseModel):
    """Request for agent lifecycle management"""
    action: str = Field(..., description="Action: deploy|update|remove|monitor")
    agent_name: str = Field(..., description="Agent name")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Action parameters")

class SystemStatusResponse(BaseModel):
    """Master agent system status"""
    master_agent: Dict[str, Any]
    framework_status: Dict[str, bool]
    agents: Dict[str, int]
    workflows: Dict[str, int]
    performance: Dict[str, Any]
    system_health: str

class WorkflowResponse(BaseModel):
    """Workflow execution response"""
    workflow_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    agents_used: List[str] = []
    custom_agents_created: List[str] = []
    execution_time: float
    performance_metrics: Dict[str, Any]

class AgentResponse(BaseModel):
    """Agent management response"""
    agent_name: str
    status: str
    agent_type: Optional[str] = None
    capabilities: Optional[List[str]] = None
    details: Optional[Dict[str, Any]] = None

# API Endpoints

@router.on_event("startup")
async def startup_event():
    """Initialize master agent system on startup"""
    try:
        success = await initialize_greta_master_agent()
        if success:
            logger.info("🎯 GRETA MASTER AGENT initialized via API router")
            await execute_hooks("master_agent_api_initialized")
        else:
            logger.error("❌ Master agent initialization failed during API startup")
    except Exception as e:
        logger.error(f"Master agent startup failed: {e}")

@router.get("/health")
async def check_master_agent_health() -> Dict[str, Any]:
    """Check master agent system health"""
    try:
        status = await greta_master_agent.get_system_status()
        return {
            "status": "healthy" if status.get("system_health") == "excellent" else "degraded",
            "message": f"GRETA Master Agent {status.get('system_health', 'unknown')} health",
            "details": status
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Master agent health check failed: {e}")

@router.get("/status")
async def get_master_agent_status() -> SystemStatusResponse:
    """Get comprehensive master agent system status"""
    try:
        status = await greta_master_agent.get_system_status()
        return SystemStatusResponse(**status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {e}")

@router.post("/task")
@handle_errors
async def execute_master_agent_task(
    request: MasterTaskRequest,
    background_tasks: BackgroundTasks
) -> WorkflowResponse:
    """
    Execute a complex multi-agent task via the master agent controller
    This fulfills the original CPAS vision of master agent controlling all Greta aspects
    """
    try:
        logger.info(f"🎯 Executing master task: {request.description[:100]}...")

        # Add task context
        context = request.context.copy()
        context.update({
            "request_time": datetime.utcnow().isoformat(),
            "priority": request.priority,
            "api_source": True,
            "timeout": request.timeout
        })

        # Execute the master workflow
        result = await execute_master_task(request.description, context)

        # Log successful execution
        logger.info(f"✅ Master task completed: {result.get('workflow_id')}")

        # Trigger success hooks
        await execute_hooks("master_task_completed", workflow_id=result.get('workflow_id'))

        return WorkflowResponse(**result)

    except Exception as e:
        error_msg = f"Master task execution failed: {e}"
        logger.error(error_msg)

        await execute_hooks("master_task_failed", error=str(e), task=request.description)

        raise HTTPException(status_code=500, detail=error_msg)

@router.post("/task/async")
@handle_errors
async def execute_master_agent_task_async(
    request: MasterTaskRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, str]:
    """
    Execute a complex task asynchronously for long-running workflows
    Returns workflow ID for status checking
    """
    try:
        # Add to background task queue
        background_tasks.add_task(
            _execute_master_task_background,
            request.description,
            request.context,
            request.priority,
            request.timeout
        )

        workflow_id = f"master_workflow_async_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"🎯 Queued async master task: {workflow_id}")

        return {
            "message": "Master task queued for execution",
            "workflow_id": workflow_id,
            "status": "queued",
            "monitor_endpoint": f"/api/master/workflow/{workflow_id}/status"
        }

    except Exception as e:
        error_msg = f"Async task queuing failed: {e}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@router.post("/agent/create")
@handle_errors
async def create_custom_agent(request: AgentSpecRequest) -> AgentResponse:
    """
    Create a custom agent using SmolAgents integration
    Fulfills the "agent builder" aspect of CPAS vision
    """
    try:
        logger.info(f"🏗️ Creating custom agent: {request.spec.get('name', 'Unnamed')}")

        result = await create_greta_agent(request.spec)

        # Log successful creation
        if result.get("status") == "active":
            logger.info(f"✅ Custom agent created: {result.get('agent_name')}")
            await execute_hooks("custom_agent_created_via_api", agent_name=result.get('agent_name'))
        else:
            logger.warning(f"⚠️ Custom agent creation had issues: {result}")

        return AgentResponse(
            agent_name=result.get("agent_name", "unknown"),
            status=result.get("status", "unknown"),
            agent_type=result.get("agent_type"),
            capabilities=result.get("capabilities", []),
            details=result
        )

    except Exception as e:
        error_msg = f"Custom agent creation failed: {e}"
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

@router.post("/agent/manage")
@handle_errors
async def manage_agent_lifecycle(request: AgentManagementRequest) -> AgentResponse:
    """
    Manage agent lifecycle: deploy, update, remove, monitor
    Fulfills the "agent deployment and management" aspect of CPAS vision
    """
    try:
        valid_actions = ["deploy", "update", "remove", "monitor"]
        if request.action not in valid_actions:
            raise HTTPException(status_code=400, detail=f"Invalid action. Must be one of: {valid_actions}")

        logger.info(f"⚙️ {request.action.capitalize()}ing agent: {request.agent_name}")

        result = await manage_greta_agents(
            request.action,
            request.agent_name,
            request.parameters
        )

        action_past_tense = {
            "deploy": "deployed",
            "update": "updated",
            "remove": "removed",
            "monitor": "monitored"
        }[request.action]

        logger.info(f"✅ Agent {action_past_tense}: {request.agent_name}")

        # Trigger appropriate hooks
        await execute_hooks(f"agent_{request.action}d_via_api",
                          agent_name=request.agent_name,
                          parameters=request.parameters)

        return AgentResponse(
            agent_name=request.agent_name,
            status=result.get("status", "completed"),
            details=result
        )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Agent management failed: {e}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@router.get("/agents/list")
@handle_errors
async def list_agents() -> Dict[str, Any]:
    """List all available agents (specialized and custom)"""
    try:
        status = await greta_master_agent.get_system_status()

        return {
            "specialized_agents": list(status["agents"].get("specialized_agents", [])),
            "custom_agents": list(greta_master_agent.custom_agents.keys()),
            "total_agents": status["agents"].get("total", 0),
            "agent_details": {
                agent_name: {
                    "type": "custom" if agent_name in greta_master_agent.custom_agents else "specialized",
                    "status": "active",
                    "created_at": getattr(greta_master_agent, '_init_time', datetime.utcnow()).isoformat()
                }
                for agent_name in list(status["agents"].get("specialized_agents", [])) + list(greta_master_agent.custom_agents.keys())
            }
        }

    except Exception as e:
        logger.error(f"Agent listing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent listing failed: {e}")

@router.get("/workflows/active")
@handle_errors
async def get_active_workflows() -> Dict[str, List[Dict[str, Any]]]:
    """Get list of currently active workflows"""
    try:
        active_workflows = [
            {k: v for k, v in workflow.items() if k != 'context'}  # Exclude large context
            for workflow in greta_master_agent.active_workflows.values()
            if workflow.get('status') in ['analyzing', 'running', 'coordinating']
        ]

        return {
            "active_workflows": active_workflows,
            "count": len(active_workflows),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Active workflows retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Active workflows retrieval failed: {e}")

@router.get("/workflow/{workflow_id}/status")
@handle_errors
async def get_workflow_status(workflow_id: str) -> Dict[str, Any]:
    """Get status of a specific workflow"""
    try:
        if workflow_id not in greta_master_agent.active_workflows:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")

        workflow = greta_master_agent.active_workflows[workflow_id]

        return {
            "workflow_id": workflow_id,
            "status": workflow.get("status"),
            "description": workflow.get("description"),
            "start_time": workflow.get("start_time"),
            "end_time": workflow.get("end_time"),
            "execution_time": None,  # Calculate if completed
            "agent_assignments": workflow.get("agent_assignments", {}),
            "subtasks": workflow.get("subtasks", []),
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Workflow status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Workflow status retrieval failed: {e}")

@router.post("/system/reset")
@handle_errors
async def reset_master_agent_system() -> Dict[str, str]:
    """Reset and reinitialize the master agent system"""
    try:
        # Reset all active workflows
        greta_master_agent.active_workflows.clear()
        greta_master_agent.custom_agents.clear()
        greta_master_agent.performance_metrics = {
            'tasks_processed': 0,
            'agents_created': 0,
            'workflows_completed': 0,
            'average_response_time': 0.0
        }

        # Reinitialize
        success = await initialize_greta_master_agent()

        if success:
            logger.info("🔄 Master agent system reset and reinitialized")
            await execute_hooks("master_agent_system_reset")

            return {
                "message": "GRETA Master Agent system reset and reinitialized successfully",
                "status": "operational",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Master agent reinitialization failed")

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"System reset failed: {e}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@router.get("/demo/scenarios")
async def get_demo_scenarios() -> Dict[str, List[Dict[str, str]]]:
    """Get available demo scenarios for testing master agent capabilities"""
    return {
        "demo_scenarios": [
            {
                "name": "ecommerce_platform",
                "description": "Build complete e-commerce platform with payment integration",
                "complexity": "high",
                "estimated_time": "5-10 minutes"
            },
            {
                "name": "ai_blog_platform",
                "description": "Create AI-powered blog platform with content generation",
                "complexity": "medium",
                "estimated_time": "3-7 minutes"
            },
            {
                "name": "mobile_app_backend",
                "description": "Develop mobile app with complete backend API",
                "complexity": "high",
                "estimated_time": "7-12 minutes"
            },
            {
                "name": "data_analytics_suite",
                "description": "Build data analytics dashboard with ML insights",
                "complexity": "medium",
                "estimated_time": "4-8 minutes"
            },
            {
                "name": "custom_agent_creation",
                "description": "Create specialized agent for specific domain",
                "complexity": "low",
                "estimated_time": "1-2 minutes"
            }
        ],
        "usage_examples": [
            "Execute complex multi-agent task coordination",
            "Dynamic agent creation and specialization",
            "Workflow orchestration and optimization",
            "Agent team formation and management"
        ]
    }

# Background Task Helper

async def _execute_master_task_background(
    description: str,
    context: Dict[str, Any],
    priority: str,
    timeout: int
) -> None:
    """Background execution of master tasks"""
    try:
        # Add background execution context
        bg_context = context.copy()
        bg_context.update({
            "background_execution": True,
            "priority": priority,
            "timeout": timeout
        })

        # Execute the task
        result = await execute_master_task(description, bg_context)

        logger.info(f"🎯 Background master task completed: {result.get('workflow_id')}")

    except Exception as e:
        logger.error(f"Background master task failed: {e}")

# Export router for main app integration
master_agent_router = router
