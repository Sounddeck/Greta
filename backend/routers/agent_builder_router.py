"""
Agent Builder Router for CPAS4
REST API endpoints for Agent Builder
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
import logging
from services.agent_builder_service import AgentBuilderService
from models.agent_builder_models import *
from database import Database

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/agent-builder", tags=["Agent Builder"])

async def get_service() -> AgentBuilderService:
    db = Database()
    await db.connect()
    return AgentBuilderService(db)

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
async def update_agent(aid: str, au: CustomAgentUpdate, svc: AgentBuilderService = Depends(get_service)):
    a = await svc.update_agent(aid, au)
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
    return await svc.create_version(vc)

@router.get("/agents/{aid}/versions", response_model=List[AgentVersion])
async def list_versions(aid: str, limit: int = Query(50, le=100), svc: AgentBuilderService = Depends(get_service)):
    return await svc.list_versions(aid, limit)

@router.post("/agents/{aid}/rollback", response_model=CustomAgent)
async def rollback_version(aid: str, version_id: str = Query(...), svc: AgentBuilderService = Depends(get_service)):
    a = await svc.rollback_version(aid, version_id)
    if not a: raise HTTPException(404, "Version not found")
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

# Health
@router.get("/health")
async def health_check():
    return {"status": "healthy", "module": "Agent Builder", "features": ["Custom Agent Creation", "Template Management", "Tool Assignment", "Deployment Management", "Version Control", "Code Generation", "Execution Tracking", "Statistics"], "endpoints": 27, "database": "MongoDB/File Storage"}
