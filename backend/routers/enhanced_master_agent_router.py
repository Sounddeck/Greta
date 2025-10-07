"""
GRETA Enhanced Master Agent Router
Complete API endpoints for Master Agent system control
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from database import get_database
from services.llm_service import get_llm_service
from services.learning_service import LearningService
from services.enhanced_master_agent import EnhancedMasterAgent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/master", tags=["Master Agent"])

# Global master agent instance
master_agent_instance: Optional[EnhancedMasterAgent] = None

# Pydantic models
class CommandRequest(BaseModel):
    """Master agent command request"""
    type: str = Field(..., description="Command type")
    data: Dict[str, Any] = Field(default_factory=dict, description="Command data")

class AgentBuildRequest(BaseModel):
    """Agent build request"""
    type: str = Field(..., description="Agent type (research, code, creative, data, osint)")
    name: Optional[str] = Field(None, description="Agent name")
    config: Dict[str, Any] = Field(default_factory=dict, description="Agent configuration")

class AgentDeployRequest(BaseModel):
    """Agent deployment request"""
    agent_id: str = Field(..., description="Agent ID to deploy")
    deployment_config: Dict[str, Any] = Field(default_factory=dict, description="Deployment configuration")

class AgentManageRequest(BaseModel):
    """Agent management request"""
    agent_id: str = Field(..., description="Agent ID")
    action: str = Field(..., description="Action (start, stop, restart, delete)")

class WorkflowRequest(BaseModel):
    """Workflow execution request"""
    name: str = Field(..., description="Workflow name")
    description: Optional[str] = Field(None, description="Workflow description")
    steps: List[Dict[str, Any]] = Field(..., description="Workflow steps")

class ApplicationBuildRequest(BaseModel):
    """Application build request"""
    name: str = Field(..., description="Application name")
    type: str = Field(..., description="Application type (web_application, api_application, cli_application)")
    spec: Dict[str, Any] = Field(default_factory=dict, description="Application specification")

class OptimizationRequest(BaseModel):
    """System optimization request"""
    type: str = Field(default="general", description="Optimization type (memory, performance, resources, general)")

class LLMQueryRequest(BaseModel):
    """LLM query request"""
    query: str = Field(..., description="Query for LLM")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

async def get_master_agent() -> EnhancedMasterAgent:
    """Get or initialize master agent instance"""
    global master_agent_instance
    
    if master_agent_instance is None:
        db = await get_database()
        llm_service = await get_llm_service()
        learning_service = LearningService(db.client)
        await learning_service.initialize()
        
        master_agent_instance = EnhancedMasterAgent(db, llm_service, learning_service)
        await master_agent_instance.initialize()
    
    return master_agent_instance

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Enhanced Master Agent",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/status")
async def get_status(master_agent: EnhancedMasterAgent = Depends(get_master_agent)):
    """Get comprehensive system status"""
    try:
        status = await master_agent.get_system_status()
        return status
    except Exception as e:
        logger.error(f"❌ Status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/command")
async def execute_command(
    request: CommandRequest,
    master_agent: EnhancedMasterAgent = Depends(get_master_agent)
):
    """Execute a master agent command"""
    try:
        command = {
            "type": request.type,
            **request.data
        }
        
        result = await master_agent.execute_command(command)
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Command execution failed"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Command execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agent/build")
async def build_agent(
    request: AgentBuildRequest,
    master_agent: EnhancedMasterAgent = Depends(get_master_agent)
):
    """Build a new agent"""
    try:
        command = {
            "type": "build_agent",
            "config": {
                "type": request.type,
                "name": request.name,
                **request.config
            }
        }
        
        result = await master_agent.execute_command(command)
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Agent build failed"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Agent build failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agent/deploy")
async def deploy_agent(
    request: AgentDeployRequest,
    master_agent: EnhancedMasterAgent = Depends(get_master_agent)
):
    """Deploy an agent"""
    try:
        command = {
            "type": "deploy_agent",
            "agent_id": request.agent_id,
            "deployment_config": request.deployment_config
        }
        
        result = await master_agent.execute_command(command)
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Agent deployment failed"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Agent deployment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agent/manage")
async def manage_agent(
    request: AgentManageRequest,
    master_agent: EnhancedMasterAgent = Depends(get_master_agent)
):
    """Manage agent lifecycle"""
    try:
        command = {
            "type": "manage_agent",
            "agent_id": request.agent_id,
            "action": request.action
        }
        
        result = await master_agent.execute_command(command)
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Agent management failed"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Agent management failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents")
async def list_agents(
    status: Optional[str] = None,
    type: Optional[str] = None,
    master_agent: EnhancedMasterAgent = Depends(get_master_agent)
):
    """List all agents with optional filters"""
    try:
        filters = {}
        if status:
            filters["status"] = status
        if type:
            filters["type"] = type
        
        agents = await master_agent.list_agents(filters)
        
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
    master_agent: EnhancedMasterAgent = Depends(get_master_agent)
):
    """Get a specific agent"""
    try:
        agent = await master_agent.get_agent(agent_id)
        
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

@router.post("/workflow/execute")
async def execute_workflow(
    request: WorkflowRequest,
    background_tasks: BackgroundTasks,
    master_agent: EnhancedMasterAgent = Depends(get_master_agent)
):
    """Execute a workflow"""
    try:
        command = {
            "type": "execute_workflow",
            "workflow": {
                "name": request.name,
                "description": request.description,
                "steps": request.steps
            }
        }
        
        # Execute in background for long-running workflows
        if len(request.steps) > 5:
            background_tasks.add_task(master_agent.execute_command, command)
            return {
                "success": True,
                "message": "Workflow execution started in background",
                "workflow_name": request.name
            }
        else:
            result = await master_agent.execute_command(command)
            
            if not result.get("success"):
                raise HTTPException(status_code=400, detail=result.get("error", "Workflow execution failed"))
            
            return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Workflow execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workflows")
async def list_workflows(
    active_only: bool = False,
    master_agent: EnhancedMasterAgent = Depends(get_master_agent)
):
    """List workflows"""
    try:
        workflows = await master_agent.list_workflows(active_only)
        
        return {
            "success": True,
            "workflows": workflows,
            "count": len(workflows)
        }
        
    except Exception as e:
        logger.error(f"❌ List workflows failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workflow/{workflow_id}")
async def get_workflow(
    workflow_id: str,
    master_agent: EnhancedMasterAgent = Depends(get_master_agent)
):
    """Get a specific workflow"""
    try:
        workflow = await master_agent.get_workflow(workflow_id)
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return {
            "success": True,
            "workflow": workflow
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Get workflow failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/application/build")
async def build_application(
    request: ApplicationBuildRequest,
    background_tasks: BackgroundTasks,
    master_agent: EnhancedMasterAgent = Depends(get_master_agent)
):
    """Build a complete application"""
    try:
        command = {
            "type": "build_application",
            "spec": {
                "name": request.name,
                "type": request.type,
                **request.spec
            }
        }

        # Execute in background for complex applications
        background_tasks.add_task(master_agent.execute_command, command)

        return {
            "success": True,
            "message": "Application build started in background",
            "app_name": request.name
        }

    except Exception as e:
        logger.error(f"❌ Application build failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/system/optimize")
async def optimize_system(
    request: OptimizationRequest,
    master_agent: EnhancedMasterAgent = Depends(get_master_agent)
):
    """Optimize system performance"""
    try:
        command = {
            "type": "optimize_system",
            "type": request.type
        }

        result = await master_agent.execute_command(command)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "System optimization failed"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ System optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/llm/query")
async def query_llm(
    request: LLMQueryRequest,
    master_agent: EnhancedMasterAgent = Depends(get_master_agent)
):
    """Query the local LLM"""
    try:
        command = {
            "type": "query_llm",
            "query": request.query,
            "context": request.context
        }

        result = await master_agent.execute_command(command)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "LLM query failed"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ LLM query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_metrics(master_agent: EnhancedMasterAgent = Depends(get_master_agent)):
    """Get system metrics"""
    try:
        return {
            "success": True,
            "metrics": master_agent.metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Get metrics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/modules")
async def get_modules(master_agent: EnhancedMasterAgent = Depends(get_master_agent)):
    """Get all modules status"""
    try:
        return {
            "success": True,
            "modules": master_agent.modules,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Get modules failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/shutdown")
async def shutdown_master_agent(master_agent: EnhancedMasterAgent = Depends(get_master_agent)):
    """Shutdown the Master Agent"""
    try:
        await master_agent.shutdown()
        return {
            "success": True,
            "message": "Master Agent shutdown complete"
        }
    except Exception as e:
        logger.error(f"❌ Shutdown failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== SYSTEM MONITORING ENDPOINTS ====================

@router.get("/dashboard/metrics")
async def get_system_metrics(master_agent: EnhancedMasterAgent = Depends(get_master_agent)):
    """Get real-time system metrics for dashboard"""
    try:
        metrics = await master_agent.get_system_metrics()
        return {
            "success": True,
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard/module-status")
async def get_module_status(master_agent: EnhancedMasterAgent = Depends(get_master_agent)):
    """Get status of all modules"""
    try:
        status = await master_agent.get_module_status()
        return {
            "success": True,
            "modules": status
        }
    except Exception as e:
        logger.error(f"Error getting module status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard/agent-inventory")
async def get_agent_inventory(master_agent: EnhancedMasterAgent = Depends(get_master_agent)):
    """Get inventory of all deployed agents"""
    try:
        inventory = await master_agent.get_agent_inventory()
        return {
            "success": True,
            **inventory
        }
    except Exception as e:
        logger.error(f"Error getting agent inventory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard/activity-log")
async def get_activity_log(
    limit: int = 100,
    master_agent: EnhancedMasterAgent = Depends(get_master_agent)
):
    """Get recent activity log"""
    try:
        activities = await master_agent.get_activity_log(limit)
        return {
            "success": True,
            "activities": activities,
            "count": len(activities)
        }
    except Exception as e:
        logger.error(f"Error getting activity log: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard/health")
async def get_system_health(master_agent: EnhancedMasterAgent = Depends(get_master_agent)):
    """Get overall system health score"""
    try:
        health = await master_agent.calculate_system_health()
        return {
            "success": True,
            **health
        }
    except Exception as e:
        logger.error(f"Error calculating system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== AGENT ORCHESTRATION ENDPOINTS ====================

class AgentDeployFromTemplateRequest(BaseModel):
    """Deploy agent from template request"""
    template_id: str = Field(..., description="Template ID (researcher, engineer, designer, etc.)")
    name: Optional[str] = Field(None, description="Custom agent name")
    config: Optional[Dict[str, Any]] = Field(None, description="Custom configuration")

class TaskRoutingRequest(BaseModel):
    """Task routing request"""
    type: str = Field(..., description="Task type")
    description: str = Field(..., description="Task description")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class AgentLifecycleRequest(BaseModel):
    """Agent lifecycle management request"""
    agent_id: str = Field(..., description="Agent ID")
    action: str = Field(..., description="Action: start, stop, pause, resume, delete")

@router.get("/templates")
async def get_agent_templates(master_agent: EnhancedMasterAgent = Depends(get_master_agent)):
    """Get all available agent templates"""
    try:
        from services.agent_templates import AgentTemplates
        templates = AgentTemplates.get_all_templates()
        return {
            "success": True,
            "templates": templates,
            "count": len(templates)
        }
    except Exception as e:
        logger.error(f"Error getting agent templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agents/deploy")
async def deploy_agent_from_template(
    request: AgentDeployFromTemplateRequest,
    master_agent: EnhancedMasterAgent = Depends(get_master_agent)
):
    """Deploy an agent from a template"""
    try:
        result = await master_agent.deploy_agent_from_template(
            request.template_id,
            {"name": request.name, **(request.config or {})}
        )

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deploying agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tasks/route")
async def route_task(
    request: TaskRoutingRequest,
    master_agent: EnhancedMasterAgent = Depends(get_master_agent)
):
    """Route a task to the most appropriate agent"""
    try:
        task = {
            "type": request.type,
            "description": request.description,
            "metadata": request.metadata or {}
        }

        result = await master_agent.route_task_to_agent(task)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error routing task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agents/lifecycle")
async def manage_agent_lifecycle(
    request: AgentLifecycleRequest,
    master_agent: EnhancedMasterAgent = Depends(get_master_agent)
):
    """Manage agent lifecycle (start, stop, pause, resume, delete)"""
    try:
        result = await master_agent.manage_agent_lifecycle(request.agent_id, request.action)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error managing agent lifecycle: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== LEARNING COORDINATION ENDPOINTS ====================

@router.get("/learning/status")
async def get_learning_status(master_agent: EnhancedMasterAgent = Depends(get_master_agent)):
    """Get learning system status"""
    try:
        status = await master_agent.get_learning_status()
        return status
    except Exception as e:
        logger.error(f"Error getting learning status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/learning/collect")
async def collect_training_data(master_agent: EnhancedMasterAgent = Depends(get_master_agent)):
    """Collect training data from all modules"""
    try:
        result = await master_agent.collect_training_data()

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error collecting training data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/learning/train")
async def trigger_llm_training(
    background_tasks: BackgroundTasks,
    master_agent: EnhancedMasterAgent = Depends(get_master_agent)
):
    """Trigger LLM training with collected data"""
    try:
        # Run training in background
        background_tasks.add_task(master_agent.trigger_llm_training)

        return {
            "success": True,
            "message": "LLM training started in background"
        }
    except Exception as e:
        logger.error(f"Error triggering LLM training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== EMERGENCY CONTROLS ENDPOINTS ====================

@router.post("/emergency/pause")
async def pause_system(master_agent: EnhancedMasterAgent = Depends(get_master_agent)):
    """Pause all system operations"""
    try:
        result = await master_agent.pause_system()

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error pausing system: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/emergency/resume")
async def resume_system(master_agent: EnhancedMasterAgent = Depends(get_master_agent)):
    """Resume all system operations"""
    try:
        result = await master_agent.resume_system()

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resuming system: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/emergency/shutdown")
async def emergency_shutdown(master_agent: EnhancedMasterAgent = Depends(get_master_agent)):
    """Emergency shutdown of all operations"""
    try:
        result = await master_agent.emergency_shutdown()

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during emergency shutdown: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/emergency/restart")
async def restart_system(master_agent: EnhancedMasterAgent = Depends(get_master_agent)):
    """Restart the system"""
    try:
        result = await master_agent.restart_system()

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error restarting system: {e}")
        raise HTTPException(status_code=500, detail=str(e))

