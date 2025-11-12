"""
AutoGen 0.4 Router for CPAS4
REST API endpoints for AutoGen agent management, workflows, and execution
"""
from fastapi import APIRouter, HTTPException, Depends, Query, Body
from typing import List, Optional, Dict, Any
import logging

from services.autogen_agent_service import AutoGenAgentService
from models.autogen_agent_models import (
    AutoGenAgent, AutoGenAgentConfig, AgentStatus, AgentType,
    WorkflowDefinition, DeploymentConfig, DeploymentResult,
    ConversationResult, AgentTemplate, AgentMetrics,
    CreateAgentRequest, UpdateAgentRequest,
    CreateWorkflowRequest, DeployWorkflowRequest, ExecuteWorkflowRequest
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/agent-builder/autogen", tags=["AutoGen 0.4"])


# Dependency injection
async def get_service() -> AutoGenAgentService:
    """Get AutoGen service instance"""
    return AutoGenAgentService()


# ==================== Agent Management Endpoints ====================

@router.post("/agents", response_model=AutoGenAgent, status_code=201)
async def create_agent(
    request: CreateAgentRequest,
    service: AutoGenAgentService = Depends(get_service)
):
    """Create a new AutoGen agent"""
    try:
        agent = await service.create_agent(request.config)
        return agent
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_id}", response_model=AutoGenAgent)
async def get_agent(
    agent_id: str,
    service: AutoGenAgentService = Depends(get_service)
):
    """Get agent by ID"""
    agent = await service.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
    return agent


@router.get("/agents", response_model=List[AutoGenAgent])
async def list_agents(
    agent_type: Optional[AgentType] = Query(None, description="Filter by agent type"),
    status: Optional[AgentStatus] = Query(None, description="Filter by status"),
    limit: int = Query(100, le=1000, description="Maximum number of results"),
    service: AutoGenAgentService = Depends(get_service)
):
    """List all agents with optional filtering"""
    agents = await service.list_agents(agent_type=agent_type, status=status, limit=limit)
    return agents


@router.put("/agents/{agent_id}", response_model=AutoGenAgent)
async def update_agent(
    agent_id: str,
    request: UpdateAgentRequest,
    service: AutoGenAgentService = Depends(get_service)
):
    """Update agent configuration or status"""
    agent = await service.update_agent(
        agent_id=agent_id,
        config=request.config,
        status=request.status,
        tags=request.tags
    )
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
    return agent


@router.delete("/agents/{agent_id}", status_code=204)
async def delete_agent(
    agent_id: str,
    service: AutoGenAgentService = Depends(get_service)
):
    """Delete an agent"""
    success = await service.delete_agent(agent_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")


# ==================== Workflow Management Endpoints ====================

@router.post("/workflows", response_model=WorkflowDefinition, status_code=201)
async def create_workflow(
    request: CreateWorkflowRequest,
    service: AutoGenAgentService = Depends(get_service)
):
    """Create a new workflow"""
    try:
        workflow = await service.create_workflow(request.workflow)
        return workflow
    except Exception as e:
        logger.error(f"Error creating workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflows/{workflow_id}", response_model=WorkflowDefinition)
async def get_workflow(
    workflow_id: str,
    service: AutoGenAgentService = Depends(get_service)
):
    """Get workflow by ID"""
    workflow = await service.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail=f"Workflow not found: {workflow_id}")
    return workflow


@router.get("/workflows", response_model=List[WorkflowDefinition])
async def list_workflows(
    limit: int = Query(100, le=1000, description="Maximum number of results"),
    service: AutoGenAgentService = Depends(get_service)
):
    """List all workflows"""
    workflows = await service.list_workflows(limit=limit)
    return workflows


@router.put("/workflows/{workflow_id}", response_model=WorkflowDefinition)
async def update_workflow(
    workflow_id: str,
    workflow: WorkflowDefinition = Body(...),
    service: AutoGenAgentService = Depends(get_service)
):
    """Update workflow definition"""
    updated = await service.update_workflow(workflow_id, workflow)
    if not updated:
        raise HTTPException(status_code=404, detail=f"Workflow not found: {workflow_id}")
    return updated


@router.delete("/workflows/{workflow_id}", status_code=204)
async def delete_workflow(
    workflow_id: str,
    service: AutoGenAgentService = Depends(get_service)
):
    """Delete a workflow"""
    success = await service.delete_workflow(workflow_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Workflow not found: {workflow_id}")


# ==================== Deployment Endpoints ====================

@router.post("/workflows/{workflow_id}/deploy", response_model=DeploymentResult)
async def deploy_workflow(
    workflow_id: str,
    request: DeployWorkflowRequest,
    service: AutoGenAgentService = Depends(get_service)
):
    """Deploy a workflow to runtime environment"""
    try:
        result = await service.deploy_workflow(workflow_id, request.deployment_config)
        return result
    except Exception as e:
        logger.error(f"Error deploying workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deployments/{deployment_id}", response_model=DeploymentResult)
async def get_deployment(
    deployment_id: str,
    service: AutoGenAgentService = Depends(get_service)
):
    """Get deployment by ID"""
    deployment = await service.get_deployment(deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail=f"Deployment not found: {deployment_id}")
    return deployment


# ==================== Execution Endpoints ====================

@router.post("/workflows/{workflow_id}/execute", response_model=ConversationResult)
async def execute_workflow(
    workflow_id: str,
    request: ExecuteWorkflowRequest,
    service: AutoGenAgentService = Depends(get_service)
):
    """Execute a workflow with given input"""
    try:
        result = await service.execute_workflow(
            workflow_id=workflow_id,
            input_message=request.input_message,
            context=request.context,
            max_rounds=request.max_rounds,
            timeout=request.timeout
        )
        return result
    except Exception as e:
        logger.error(f"Error executing workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Template Endpoints ====================

@router.get("/templates", response_model=List[AgentTemplate])
async def list_templates(
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(100, le=1000, description="Maximum number of results"),
    service: AutoGenAgentService = Depends(get_service)
):
    """List agent templates"""
    templates = await service.list_templates(category=category, limit=limit)
    return templates


@router.get("/templates/{template_id}", response_model=AgentTemplate)
async def get_template(
    template_id: str,
    service: AutoGenAgentService = Depends(get_service)
):
    """Get template by ID"""
    template = await service.get_template(template_id)
    if not template:
        raise HTTPException(status_code=404, detail=f"Template not found: {template_id}")
    return template


@router.post("/templates/{template_id}/instantiate", response_model=WorkflowDefinition)
async def instantiate_template(
    template_id: str,
    customizations: Dict[str, Any] = Body(default={}),
    service: AutoGenAgentService = Depends(get_service)
):
    """Create a workflow from a template"""
    try:
        workflow = await service.instantiate_template(template_id, customizations)
        return workflow
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error instantiating template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Monitoring Endpoints ====================

@router.get("/agents/{agent_id}/status")
async def get_agent_status(
    agent_id: str,
    service: AutoGenAgentService = Depends(get_service)
):
    """Get agent runtime status"""
    status = await service.get_agent_status(agent_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
    return status


@router.get("/agents/{agent_id}/metrics", response_model=AgentMetrics)
async def get_agent_metrics(
    agent_id: str,
    service: AutoGenAgentService = Depends(get_service)
):
    """Get agent performance metrics"""
    metrics = await service.get_agent_metrics(agent_id)
    if not metrics:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
    return metrics


@router.get("/agents/{agent_id}/logs")
async def get_agent_logs(
    agent_id: str,
    limit: int = Query(100, le=1000, description="Maximum number of log entries"),
    service: AutoGenAgentService = Depends(get_service)
):
    """Get agent execution logs"""
    # Placeholder for log retrieval
    return {
        "agent_id": agent_id,
        "logs": [
            {"timestamp": "2025-10-10T12:00:00Z", "level": "INFO", "message": "Agent created"},
            {"timestamp": "2025-10-10T12:01:00Z", "level": "INFO", "message": "Agent configured"}
        ]
    }
