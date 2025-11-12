"""
Agent Builder Router for CPAS4
REST API endpoints for Agent Builder
"""
from fastapi import APIRouter, HTTPException, Depends, Query, Body
from typing import List, Optional, Dict, Any
import logging
from services.agent_builder_service import AgentBuilderService
from services.mcp_service import MCPHubService
from services.agent_collaboration import get_collaboration_service, WorkflowType
from models.agent_builder_models import (
    AgentTemplate, AgentTemplateCreate, AgentTemplateUpdate,
    CustomAgent, CustomAgentCreate, CustomAgentUpdate,
    AgentVersion, AgentVersionCreate, VersionDiff,
    Deployment, DeploymentCreate, DeploymentUpdate,
    AgentExecution, AgentExecutionCreate,
    CodeGenerationRequest, CodeGenerationResult,
    ValidationResult, AgentBuilderStats,
    AgentDraft, AgentDraftCreate, AgentDraftUpdate,
    EnhancedValidationResult,
    AgentStatus, AgentType, DeploymentStatus, ExecutionStatus,
    # Phase 2, Task 2.4: New validation models
    DependencyValidationResult, CompatibilityValidationResult,
    PerformanceEstimate, ComprehensiveValidationResult
)
from models.mcp_models import AgentMCPIntegration, AgentMCPIntegrationCreate, AgentMCPIntegrationUpdate
from database import Database

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/agent-builder", tags=["Agent Builder"])

async def get_service() -> AgentBuilderService:
    db = Database()
    await db.connect()
    return AgentBuilderService(db)

async def get_mcp_service() -> MCPHubService:
    db = Database()
    await db.connect()
    return MCPHubService(db)

# Templates
@router.post("/templates", response_model=AgentTemplate, status_code=201)
async def create_template(t: AgentTemplateCreate, svc: AgentBuilderService = Depends(get_service)):
    return await svc.create_template(t)

@router.get("/templates/{tid}", response_model=AgentTemplate)
async def get_template(tid: str, svc: AgentBuilderService = Depends(get_service)):
    t = await svc.get_template(tid)
    if not t: raise HTTPException(404, "Template not found")
    return t

@router.patch("/templates/{tid}", response_model=AgentTemplate)
async def update_template(tid: str, tu: AgentTemplateUpdate, svc: AgentBuilderService = Depends(get_service)):
    t = await svc.update_template(tid, tu)
    if not t: raise HTTPException(404, "Template not found")
    return t

@router.delete("/templates/{tid}", status_code=204)
async def delete_template(tid: str, svc: AgentBuilderService = Depends(get_service)):
    if not await svc.delete_template(tid): raise HTTPException(404, "Template not found")

@router.get("/templates", response_model=List[AgentTemplate])
async def list_templates(agent_type: Optional[AgentType] = None, tags: Optional[List[str]] = Query(None), limit: int = Query(100, le=1000), svc: AgentBuilderService = Depends(get_service)):
    return await svc.list_templates(agent_type, tags, limit)

# Agents
@router.post("/agents", response_model=CustomAgent, status_code=201)
async def create_agent(ac: CustomAgentCreate, svc: AgentBuilderService = Depends(get_service)):
    return await svc.create_agent(ac)

@router.get("/agents/{aid}", response_model=CustomAgent)
async def get_agent(aid: str, svc: AgentBuilderService = Depends(get_service)):
    a = await svc.get_agent(aid)
    if not a: raise HTTPException(404, "Agent not found")
    return a

@router.patch("/agents/{aid}", response_model=CustomAgent)
async def update_agent(
    aid: str,
    au: CustomAgentUpdate,
    auto_version: bool = Query(True, description="Automatically create a version snapshot before update"),
    version_changes: str = Query("Agent updated", description="Description of changes for version history"),
    svc: AgentBuilderService = Depends(get_service)
):
    """Update an agent and optionally create a version snapshot"""
    a = await svc.update_agent(aid, au, auto_version=auto_version, version_changes=version_changes)
    if not a: raise HTTPException(404, "Agent not found")
    return a

@router.delete("/agents/{aid}", status_code=204)
async def delete_agent(aid: str, svc: AgentBuilderService = Depends(get_service)):
    if not await svc.delete_agent(aid): raise HTTPException(404, "Agent not found")

@router.get("/agents", response_model=List[CustomAgent])
async def list_agents(status: Optional[AgentStatus] = None, agent_type: Optional[AgentType] = None, tags: Optional[List[str]] = Query(None), limit: int = Query(100, le=1000), svc: AgentBuilderService = Depends(get_service)):
    return await svc.list_agents(status, agent_type, tags, limit)

@router.post("/agents/{aid}/clone", response_model=CustomAgent)
async def clone_agent(aid: str, new_name: str = Query(...), svc: AgentBuilderService = Depends(get_service)):
    a = await svc.clone_agent(aid, new_name)
    if not a: raise HTTPException(404, "Agent not found")
    return a

# Versions
@router.post("/versions", response_model=AgentVersion, status_code=201)
async def create_version(vc: AgentVersionCreate, svc: AgentBuilderService = Depends(get_service)):
    """Create a new version of an agent"""
    return await svc.create_version(vc)

@router.get("/agents/{aid}/versions", response_model=List[AgentVersion])
async def list_versions(aid: str, limit: int = Query(50, le=100), svc: AgentBuilderService = Depends(get_service)):
    """List all versions for an agent"""
    return await svc.list_versions(aid, limit)

@router.get("/versions/{vid}", response_model=AgentVersion)
async def get_version(vid: str, svc: AgentBuilderService = Depends(get_service)):
    """Get a specific version by ID"""
    v = await svc.get_version(vid)
    if not v:
        raise HTTPException(404, "Version not found")
    return v

@router.get("/versions/{vid_a}/compare/{vid_b}", response_model=VersionDiff)
async def compare_versions(vid_a: str, vid_b: str, svc: AgentBuilderService = Depends(get_service)):
    """Compare two versions and return differences"""
    diff = await svc.compare_versions(vid_a, vid_b)
    if not diff:
        raise HTTPException(404, "One or both versions not found, or versions from different agents")
    return diff

@router.post("/agents/{aid}/rollback", response_model=CustomAgent)
async def rollback_version(aid: str, version_id: str = Query(...), svc: AgentBuilderService = Depends(get_service)):
    """Rollback agent to a specific version"""
    a = await svc.rollback_version(aid, version_id)
    if not a:
        raise HTTPException(404, "Version not found")
    return a

# Deployment
@router.post("/deployments", response_model=Deployment, status_code=201)
async def deploy_agent(dc: DeploymentCreate, svc: AgentBuilderService = Depends(get_service)):
    return await svc.deploy_agent(dc)

@router.get("/deployments/{did}", response_model=Deployment)
async def get_deployment(did: str, svc: AgentBuilderService = Depends(get_service)):
    d = await svc.get_deployment(did)
    if not d: raise HTTPException(404, "Deployment not found")
    return d

@router.get("/deployments", response_model=List[Deployment])
async def list_deployments(agent_id: Optional[str] = None, status: Optional[DeploymentStatus] = None, limit: int = Query(100, le=1000), svc: AgentBuilderService = Depends(get_service)):
    return await svc.list_deployments(agent_id, status, limit)

# Execution
@router.post("/executions", response_model=AgentExecution, status_code=201)
async def execute_agent(ec: AgentExecutionCreate, svc: AgentBuilderService = Depends(get_service)):
    return await svc.execute_agent(ec)

@router.get("/executions/{eid}", response_model=AgentExecution)
async def get_execution(eid: str, svc: AgentBuilderService = Depends(get_service)):
    e = await svc.get_execution(eid)
    if not e: raise HTTPException(404, "Execution not found")
    return e

@router.get("/executions", response_model=List[AgentExecution])
async def list_executions(agent_id: Optional[str] = None, status: Optional[ExecutionStatus] = None, limit: int = Query(100, le=1000), svc: AgentBuilderService = Depends(get_service)):
    return await svc.list_executions(agent_id, status, limit)

# Code Generation
@router.post("/agents/{aid}/generate-code", response_model=CodeGenerationResult)
async def generate_code(aid: str, include_tools: bool = True, include_tests: bool = False, svc: AgentBuilderService = Depends(get_service)):
    req = CodeGenerationRequest(agent_id=aid, include_tools=include_tools, include_tests=include_tests)
    return await svc.generate_code(req)

# Validation
@router.post("/agents/{aid}/validate", response_model=ValidationResult)
async def validate_agent(aid: str, svc: AgentBuilderService = Depends(get_service)):
    return await svc.validate_agent(aid)

# Statistics
@router.get("/stats", response_model=AgentBuilderStats)
async def get_stats(svc: AgentBuilderService = Depends(get_service)):
    return await svc.get_stats()

# MCP Configuration for Agents
@router.get("/agents/{aid}/mcp-config", response_model=List[AgentMCPIntegration])
async def get_agent_mcp_config(aid: str, mcp_svc: MCPHubService = Depends(get_mcp_service)):
    """Get MCP server integrations for a specific agent"""
    return await mcp_svc.list_integrations(agent_id=aid, limit=1000)

@router.post("/agents/{aid}/mcp-config", response_model=AgentMCPIntegration, status_code=201)
async def add_agent_mcp_config(aid: str, integration: AgentMCPIntegrationCreate, agent_svc: AgentBuilderService = Depends(get_service), mcp_svc: MCPHubService = Depends(get_mcp_service)):
    """Add MCP server integration to an agent"""
    # Verify agent exists
    agent = await agent_svc.get_agent(aid)
    if not agent:
        raise HTTPException(404, "Agent not found")

    # Create integration
    return await mcp_svc.create_integration(integration)

@router.put("/agents/{aid}/mcp-config/{integration_id}", response_model=AgentMCPIntegration)
async def update_agent_mcp_config(aid: str, integration_id: str, integration_update: AgentMCPIntegrationUpdate, mcp_svc: MCPHubService = Depends(get_mcp_service)):
    """Update MCP server integration for an agent"""
    intg = await mcp_svc.update_integration(integration_id, integration_update)
    if not intg:
        raise HTTPException(404, "Integration not found")
    return intg

@router.delete("/agents/{aid}/mcp-config/{integration_id}", status_code=204)
async def delete_agent_mcp_config(aid: str, integration_id: str, mcp_svc: MCPHubService = Depends(get_mcp_service)):
    """Remove MCP server integration from an agent"""
    if not await mcp_svc.delete_integration(integration_id):
        raise HTTPException(404, "Integration not found")

# Workflows - Collaboration Patterns
@router.post("/workflows", status_code=201)
async def create_workflow(
    name: str = Body(...),
    description: str = Body(...),
    workflow_type: WorkflowType = Body(...),
    agents: List[str] = Body(...),
    configuration: Optional[Dict[str, Any]] = Body(None),
    svc: AgentBuilderService = Depends(get_service)
):
    """Create a new collaboration workflow"""
    try:
        collab_service = await get_collaboration_service(svc)
        workflow = await collab_service.create_workflow(
            name=name,
            description=description,
            workflow_type=workflow_type,
            agents=agents,
            configuration=configuration
        )
        return workflow.model_dump()
    except Exception as e:
        logger.error(f"Failed to create workflow: {e}")
        raise HTTPException(500, str(e))

@router.post("/workflows/{workflow_id}/execute")
async def execute_workflow(
    workflow_id: str,
    input_data: Dict[str, Any] = Body(...),
    svc: AgentBuilderService = Depends(get_service)
):
    """Execute a collaboration workflow"""
    try:
        collab_service = await get_collaboration_service(svc)
        execution = await collab_service.execute_workflow(workflow_id, input_data)
        return execution.model_dump()
    except Exception as e:
        logger.error(f"Failed to execute workflow: {e}")
        raise HTTPException(500, str(e))

@router.get("/workflows/{workflow_id}/executions/{execution_id}")
async def get_workflow_execution(
    workflow_id: str,
    execution_id: str,
    svc: AgentBuilderService = Depends(get_service)
):
    """Get workflow execution details"""
    try:
        collab_service = await get_collaboration_service(svc)
        execution = await collab_service.get_workflow_execution(execution_id)
        if not execution:
            raise HTTPException(404, "Workflow execution not found")
        return execution.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow execution: {e}")
        raise HTTPException(500, str(e))

@router.get("/workflows")
async def list_workflows(
    workflow_type: Optional[WorkflowType] = Query(None, description="Filter by workflow type"),
    limit: int = Query(100, le=1000, description="Maximum number of workflows to return"),
    svc: AgentBuilderService = Depends(get_service)
):
    """List all workflows with optional filtering"""
    try:
        collab_service = await get_collaboration_service(svc)
        workflows = await collab_service.list_workflows(workflow_type=workflow_type, limit=limit)
        return {"workflows": [w.model_dump() for w in workflows], "count": len(workflows)}
    except Exception as e:
        logger.error(f"Failed to list workflows: {e}")
        raise HTTPException(500, str(e))

@router.get("/workflows/{workflow_id}")
async def get_workflow(workflow_id: str, svc: AgentBuilderService = Depends(get_service)):
    """Get workflow definition by ID"""
    try:
        collab_service = await get_collaboration_service(svc)
        workflow = await collab_service.get_workflow(workflow_id)
        if not workflow:
            raise HTTPException(404, "Workflow not found")
        return workflow.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow: {e}")
        raise HTTPException(500, str(e))

@router.delete("/workflows/{workflow_id}", status_code=204)
async def delete_workflow(workflow_id: str, svc: AgentBuilderService = Depends(get_service)):
    """Delete a workflow"""
    try:
        collab_service = await get_collaboration_service(svc)
        success = await collab_service.delete_workflow(workflow_id)
        if not success:
            raise HTTPException(404, "Workflow not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete workflow: {e}")
        raise HTTPException(500, str(e))

@router.get("/workflows/{workflow_id}/executions")
async def list_workflow_executions(
    workflow_id: str,
    limit: int = Query(50, le=100, description="Maximum number of executions to return"),
    svc: AgentBuilderService = Depends(get_service)
):
    """List execution history for a workflow"""
    try:
        collab_service = await get_collaboration_service(svc)
        executions = await collab_service.list_workflow_executions(workflow_id, limit=limit)
        return {"executions": [e.model_dump() for e in executions], "count": len(executions)}
    except Exception as e:
        logger.error(f"Failed to list workflow executions: {e}")
        raise HTTPException(500, str(e))

@router.get("/workflows/metrics")
async def get_workflow_metrics(svc: AgentBuilderService = Depends(get_service)):
    """Get workflow collaboration metrics"""
    try:
        collab_service = await get_collaboration_service(svc)
        return collab_service.metrics
    except Exception as e:
        logger.error(f"Failed to get workflow metrics: {e}")
        raise HTTPException(500, str(e))

@router.get("/workflows/analytics")
async def get_workflow_analytics(
    workflow_type: Optional[WorkflowType] = Query(None, description="Filter by workflow type"),
    svc: AgentBuilderService = Depends(get_service)
):
    """Get detailed workflow analytics and performance metrics"""
    try:
        collab_service = await get_collaboration_service(svc)
        analytics = await collab_service.get_workflow_analytics(workflow_type=workflow_type)
        return analytics
    except Exception as e:
        logger.error(f"Failed to get workflow analytics: {e}")
        raise HTTPException(500, str(e))

# Models - Get available LLM models
@router.get("/models")
async def get_available_models():
    """Get list of available LLM models (local and cloud)"""
    try:
        import httpx
        from config import Settings

        settings = Settings()
        models = []

        # Add llama.cpp models (GGUF format) - PRIORITY 1
        llamacpp_models = [
            {
                "model_id": "llama_cpp:llama3-8b",
                "name": "Llama 3 8B (llama.cpp)",
                "provider": "llama_cpp",
                "type": "local",
                "description": "Llama 3 8B Instruct - GGUF format (4.8GB)",
                "context_window": 8192,
                "available": True,
                "default_temperature": 0.7,
                "default_max_tokens": 2000
            },
            {
                "model_id": "llama_cpp:llama3-70b",
                "name": "Llama 3 70B (llama.cpp)",
                "provider": "llama_cpp",
                "type": "local",
                "description": "Llama 3 70B Instruct - GGUF format (40GB)",
                "context_window": 8192,
                "available": True,
                "default_temperature": 0.7,
                "default_max_tokens": 2000
            },
            {
                "model_id": "llama_cpp:phi3-mini",
                "name": "Phi-3 Mini (llama.cpp)",
                "provider": "llama_cpp",
                "type": "local",
                "description": "Phi-3 Mini - GGUF format (2.4GB)",
                "context_window": 4096,
                "available": True,
                "default_temperature": 0.7,
                "default_max_tokens": 2000
            },
            {
                "model_id": "llama_cpp:codellama-7b",
                "name": "Code Llama 7B (llama.cpp)",
                "provider": "llama_cpp",
                "type": "local",
                "description": "Code Llama 7B - GGUF format (4.1GB)",
                "context_window": 4096,
                "available": True,
                "default_temperature": 0.7,
                "default_max_tokens": 2000
            },
            {
                "model_id": "llama_cpp:deepseek-coder-6.7b",
                "name": "DeepSeek Coder 6.7B (llama.cpp)",
                "provider": "llama_cpp",
                "type": "local",
                "description": "DeepSeek Coder 6.7B - GGUF format (4.0GB)",
                "context_window": 4096,
                "available": True,
                "default_temperature": 0.7,
                "default_max_tokens": 2000
            }
        ]
        models.extend(llamacpp_models)

        # Try to get Ollama models - PRIORITY 2
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{settings.ollama_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    for model in data.get("models", []):
                        model_name = model.get("name", "")
                        parts = model_name.split(":")
                        base_name = parts[0] if parts else model_name
                        variant = parts[1] if len(parts) > 1 else "latest"

                        models.append({
                            "model_id": f"ollama:{model_name}",
                            "name": f"{base_name.title()} ({variant})",
                            "provider": "ollama",
                            "type": "local",
                            "description": f"Local Ollama model: {model_name}",
                            "context_window": 4096,
                            "available": True,
                            "default_temperature": 0.7,
                            "default_max_tokens": 2000
                        })
        except Exception as e:
            logger.warning(f"Could not fetch Ollama models: {e}")

        # llama.cpp models are always available (added above)
        return {"models": models}

    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        return {"models": []}

# Drafts (Autosave)
@router.post("/drafts", response_model=AgentDraft, status_code=201)
async def create_draft(dc: AgentDraftCreate, svc: AgentBuilderService = Depends(get_service)):
    """Create a new agent draft for autosave"""
    return await svc.create_draft(dc)

@router.get("/drafts/{did}", response_model=AgentDraft)
async def get_draft(did: str, svc: AgentBuilderService = Depends(get_service)):
    """Get a draft by ID"""
    d = await svc.get_draft(did)
    if not d: raise HTTPException(404, "Draft not found")
    return d

@router.patch("/drafts/{did}", response_model=AgentDraft)
async def update_draft(did: str, du: AgentDraftUpdate, svc: AgentBuilderService = Depends(get_service)):
    """Update a draft (autosave)"""
    d = await svc.update_draft(did, du)
    if not d: raise HTTPException(404, "Draft not found")
    return d

@router.delete("/drafts/{did}", status_code=204)
async def delete_draft(did: str, svc: AgentBuilderService = Depends(get_service)):
    """Delete a draft"""
    if not await svc.delete_draft(did): raise HTTPException(404, "Draft not found")

@router.get("/drafts", response_model=List[AgentDraft])
async def list_drafts(limit: int = Query(100, le=1000), svc: AgentBuilderService = Depends(get_service)):
    """List all drafts (not expired)"""
    return await svc.list_drafts(limit)

@router.post("/drafts/{did}/convert", response_model=CustomAgent)
async def convert_draft_to_agent(did: str, svc: AgentBuilderService = Depends(get_service)):
    """Convert a draft to a full agent"""
    a = await svc.convert_draft_to_agent(did)
    if not a: raise HTTPException(404, "Draft not found")
    return a

@router.post("/drafts/cleanup", status_code=200)
async def cleanup_expired_drafts(svc: AgentBuilderService = Depends(get_service)):
    """Clean up expired drafts"""
    count = await svc.cleanup_expired_drafts()
    return {"deleted_count": count}

# Enhanced Validation
@router.post("/validate-enhanced", response_model=EnhancedValidationResult)
async def validate_agent_enhanced(
    agent_id: Optional[str] = Query(None),
    draft_id: Optional[str] = Query(None),
    svc: AgentBuilderService = Depends(get_service)
):
    """Enhanced validation with detailed field-level feedback"""
    return await svc.validate_agent_enhanced(agent_id, draft_id)

# Phase 2, Task 2.4: Comprehensive Validation Endpoints
@router.post("/validate-dependencies", response_model=DependencyValidationResult)
async def validate_dependencies(
    agent_id: Optional[str] = Query(None),
    draft_id: Optional[str] = Query(None),
    svc: AgentBuilderService = Depends(get_service)
):
    """
    Validate all dependencies for an agent configuration.

    Phase 2, Task 2.4: Add Agent Builder Validation
    Checks for required tools, models, API keys, and integrations.
    """
    return await svc.validate_dependencies(agent_id, draft_id)

@router.post("/validate-compatibility", response_model=CompatibilityValidationResult)
async def validate_compatibility(
    agent_id: Optional[str] = Query(None),
    draft_id: Optional[str] = Query(None),
    svc: AgentBuilderService = Depends(get_service)
):
    """
    Validate compatibility between agent components.

    Phase 2, Task 2.4: Add Agent Builder Validation
    Checks for incompatible tool/agent type combinations, model compatibility, etc.
    """
    return await svc.validate_compatibility(agent_id, draft_id)

@router.post("/estimate-performance", response_model=PerformanceEstimate)
async def estimate_performance(
    agent_id: Optional[str] = Query(None),
    draft_id: Optional[str] = Query(None),
    svc: AgentBuilderService = Depends(get_service)
):
    """
    Estimate performance metrics for an agent configuration.

    Phase 2, Task 2.4: Add Agent Builder Validation
    Predicts token usage, response time, cost, and resource requirements.
    """
    return await svc.estimate_performance(agent_id, draft_id)

@router.post("/validate-comprehensive", response_model=ComprehensiveValidationResult)
async def validate_comprehensive(
    agent_id: Optional[str] = Query(None),
    draft_id: Optional[str] = Query(None),
    svc: AgentBuilderService = Depends(get_service)
):
    """
    Perform comprehensive validation combining all validation types.

    Phase 2, Task 2.4: Add Agent Builder Validation
    Aggregates configuration, dependency, compatibility, and performance validation.
    Returns overall score, deployment readiness, and actionable next steps.
    """
    return await svc.validate_agent_comprehensive(agent_id, draft_id)

# Health
@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "module": "Agent Builder",
        "features": [
            "Custom Agent Creation",
            "Template Management",
            "Tool Assignment",
            "Deployment Management",
            "Version Control",
            "Code Generation",
            "Execution Tracking",
            "Statistics",
            "MCP Configuration",
            "Collaboration Workflows (Sequential, Parallel, Hierarchical, Consensus, Pipeline)",
            "Local Model Support",
            "Draft Management (Autosave)",
            "Enhanced Validation",
            "Dependency Validation (Phase 2, Task 2.4)",
            "Compatibility Validation (Phase 2, Task 2.4)",
            "Performance Estimation (Phase 2, Task 2.4)",
            "Comprehensive Validation (Phase 2, Task 2.4)"
        ],
        "endpoints": 48,
        "database": "MongoDB/File Storage"
    }
