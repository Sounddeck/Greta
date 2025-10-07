"""
GRETA All Modules Routers - Consolidated router file for modules 4-12
This file contains all routers for the remaining modules to simplify integration
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import sys
import os

# Add backend directory to path if not already there
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from database import get_database
from services.llm_service import get_llm_service
from services.learning_service import LearningService

# Import all services
from services.agent_management_service import AgentManagementService
from services.autonomous_operations_service import AutonomousOperationsService
from services.agent_builder_service import AgentBuilderService
from services.mcp_hub_service import MCPHubService
from services.osint_service import OSINTService
from services.multimodal_service import MultiModalService
from services.functional_memory_service import FunctionalMemoryService
from services.training_platform_service import TrainingPlatformService
from services.knowledge_base_service import KnowledgeBaseService

logger = logging.getLogger(__name__)

# Create routers for each module
agent_mgmt_router = APIRouter(prefix="/api/v1/agent-management", tags=["Agent Management"])
operations_router = APIRouter(prefix="/api/v1/operations", tags=["Autonomous Operations"])
builder_router = APIRouter(prefix="/api/v1/builder", tags=["Agent Builder"])
mcp_router = APIRouter(prefix="/api/v1/mcp", tags=["MCP Hub"])
osint_router = APIRouter(prefix="/api/v1/osint", tags=["OSINT"])
multimodal_router = APIRouter(prefix="/api/v1/multimodal", tags=["Multi-Modal"])
memory_router = APIRouter(prefix="/api/v1/functional-memory", tags=["Functional Memory"])
training_router = APIRouter(prefix="/api/v1/training", tags=["Training Platform"])
knowledge_router = APIRouter(prefix="/api/v1/knowledge", tags=["Knowledge Base"])

# Global instances
agent_mgmt_instance: Optional[AgentManagementService] = None
operations_instance: Optional[AutonomousOperationsService] = None
builder_instance: Optional[AgentBuilderService] = None
mcp_instance: Optional[MCPHubService] = None
osint_instance: Optional[OSINTService] = None
multimodal_instance: Optional[MultiModalService] = None
memory_instance: Optional[FunctionalMemoryService] = None
training_instance: Optional[TrainingPlatformService] = None
knowledge_instance: Optional[KnowledgeBaseService] = None

# Dependency functions
async def get_agent_mgmt() -> AgentManagementService:
    global agent_mgmt_instance
    if agent_mgmt_instance is None:
        db = await get_database()
        llm_service = await get_llm_service()
        learning_service = LearningService(db.client)
        await learning_service.initialize()
        agent_mgmt_instance = AgentManagementService(db, llm_service, learning_service)
        await agent_mgmt_instance.initialize()
    return agent_mgmt_instance

async def get_operations() -> AutonomousOperationsService:
    global operations_instance
    if operations_instance is None:
        db = await get_database()
        llm_service = await get_llm_service()
        learning_service = LearningService(db.client)
        await learning_service.initialize()
        operations_instance = AutonomousOperationsService(db, llm_service, learning_service)
        await operations_instance.initialize()
    return operations_instance

async def get_builder() -> AgentBuilderService:
    global builder_instance
    if builder_instance is None:
        db = await get_database()
        llm_service = await get_llm_service()
        learning_service = LearningService(db.client)
        await learning_service.initialize()
        builder_instance = AgentBuilderService(db, llm_service, learning_service)
        await builder_instance.initialize()
    return builder_instance

async def get_mcp() -> MCPHubService:
    global mcp_instance
    if mcp_instance is None:
        db = await get_database()
        llm_service = await get_llm_service()
        learning_service = LearningService(db.client)
        await learning_service.initialize()
        mcp_instance = MCPHubService(db, llm_service, learning_service)
        await mcp_instance.initialize()
    return mcp_instance

async def get_osint() -> OSINTService:
    global osint_instance
    if osint_instance is None:
        db = await get_database()
        llm_service = await get_llm_service()
        learning_service = LearningService(db.client)
        await learning_service.initialize()
        osint_instance = OSINTService(db, llm_service, learning_service)
        await osint_instance.initialize()
    return osint_instance

async def get_multimodal() -> MultiModalService:
    global multimodal_instance
    if multimodal_instance is None:
        db = await get_database()
        llm_service = await get_llm_service()
        learning_service = LearningService(db.client)
        await learning_service.initialize()
        multimodal_instance = MultiModalService(db, llm_service, learning_service)
        await multimodal_instance.initialize()
    return multimodal_instance

async def get_memory() -> FunctionalMemoryService:
    global memory_instance
    if memory_instance is None:
        db = await get_database()
        llm_service = await get_llm_service()
        learning_service = LearningService(db.client)
        await learning_service.initialize()
        memory_instance = FunctionalMemoryService(db, llm_service, learning_service)
        await memory_instance.initialize()
    return memory_instance

async def get_training() -> TrainingPlatformService:
    global training_instance
    if training_instance is None:
        db = await get_database()
        llm_service = await get_llm_service()
        learning_service = LearningService(db.client)
        await learning_service.initialize()
        training_instance = TrainingPlatformService(db, llm_service, learning_service)
        await training_instance.initialize()
    return training_instance

async def get_knowledge() -> KnowledgeBaseService:
    global knowledge_instance
    if knowledge_instance is None:
        db = await get_database()
        llm_service = await get_llm_service()
        learning_service = LearningService(db.client)
        await learning_service.initialize()
        knowledge_instance = KnowledgeBaseService(db, llm_service, learning_service)
        await knowledge_instance.initialize()
    return knowledge_instance

# ============================================================================
# AGENT MANAGEMENT ENDPOINTS
# ============================================================================

@agent_mgmt_router.get("/health")
async def agent_mgmt_health():
    return {"status": "healthy", "service": "Agent Management", "timestamp": datetime.utcnow().isoformat()}

@agent_mgmt_router.get("/metrics")
async def agent_mgmt_metrics(service: AgentManagementService = Depends(get_agent_mgmt)):
    return await service.get_metrics()

@agent_mgmt_router.post("/agent")
async def create_agent(config: Dict[str, Any], service: AgentManagementService = Depends(get_agent_mgmt)):
    result = await service.create_agent(config)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result

@agent_mgmt_router.get("/agents")
async def list_agents(status: Optional[str] = None, service: AgentManagementService = Depends(get_agent_mgmt)):
    filters = {"status": status} if status else None
    agents = await service.list_agents(filters)
    return {"success": True, "agents": agents, "count": len(agents)}

# ============================================================================
# AUTONOMOUS OPERATIONS ENDPOINTS
# ============================================================================

@operations_router.get("/health")
async def operations_health():
    return {"status": "healthy", "service": "Autonomous Operations", "timestamp": datetime.utcnow().isoformat()}

@operations_router.get("/metrics")
async def operations_metrics(service: AutonomousOperationsService = Depends(get_operations)):
    return await service.get_metrics()

@operations_router.post("/workflow")
async def create_workflow(config: Dict[str, Any], service: AutonomousOperationsService = Depends(get_operations)):
    result = await service.create_workflow(config)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result

@operations_router.post("/workflow/{workflow_id}/execute")
async def execute_workflow(workflow_id: str, service: AutonomousOperationsService = Depends(get_operations)):
    result = await service.execute_workflow(workflow_id)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result

@operations_router.get("/workflows")
async def list_workflows(status: Optional[str] = None, service: AutonomousOperationsService = Depends(get_operations)):
    filters = {"status": status} if status else None
    workflows = await service.list_workflows(filters)
    return {"success": True, "workflows": workflows, "count": len(workflows)}

# ============================================================================
# AGENT BUILDER ENDPOINTS
# ============================================================================

@builder_router.get("/health")
async def builder_health():
    return {"status": "healthy", "service": "Agent Builder", "timestamp": datetime.utcnow().isoformat()}

@builder_router.get("/metrics")
async def builder_metrics(service: AgentBuilderService = Depends(get_builder)):
    return await service.get_metrics()

@builder_router.post("/project")
async def create_project(config: Dict[str, Any], service: AgentBuilderService = Depends(get_builder)):
    result = await service.create_project(config)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result

@builder_router.post("/project/{project_id}/build")
async def build_project(project_id: str, service: AgentBuilderService = Depends(get_builder)):
    result = await service.build_project(project_id)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result

@builder_router.get("/projects")
async def list_projects(service: AgentBuilderService = Depends(get_builder)):
    projects = await service.list_projects()
    return {"success": True, "projects": projects, "count": len(projects)}

# ============================================================================
# MCP HUB ENDPOINTS
# ============================================================================

@mcp_router.get("/health")
async def mcp_health():
    return {"status": "healthy", "service": "MCP Hub", "timestamp": datetime.utcnow().isoformat()}

@mcp_router.get("/metrics")
async def mcp_metrics(service: MCPHubService = Depends(get_mcp)):
    return await service.get_metrics()

@mcp_router.post("/server")
async def register_server(config: Dict[str, Any], service: MCPHubService = Depends(get_mcp)):
    result = await service.register_server(config)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result

@mcp_router.get("/servers")
async def list_servers(service: MCPHubService = Depends(get_mcp)):
    servers = await service.list_servers()
    return {"success": True, "servers": servers, "count": len(servers)}

# ============================================================================
# OSINT ENDPOINTS
# ============================================================================

@osint_router.get("/health")
async def osint_health():
    return {"status": "healthy", "service": "OSINT Platform", "timestamp": datetime.utcnow().isoformat()}

@osint_router.get("/metrics")
async def osint_metrics(service: OSINTService = Depends(get_osint)):
    return await service.get_metrics()

@osint_router.post("/investigation")
async def create_investigation(config: Dict[str, Any], service: OSINTService = Depends(get_osint)):
    result = await service.create_investigation(config)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result

@osint_router.post("/investigation/{investigation_id}/gather")
async def gather_data(investigation_id: str, source: str, service: OSINTService = Depends(get_osint)):
    result = await service.gather_data(investigation_id, source)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result

@osint_router.get("/investigations")
async def list_investigations(service: OSINTService = Depends(get_osint)):
    investigations = await service.list_investigations()
    return {"success": True, "investigations": investigations, "count": len(investigations)}

# ============================================================================
# MULTI-MODAL ENDPOINTS
# ============================================================================

@multimodal_router.get("/health")
async def multimodal_health():
    return {"status": "healthy", "service": "Multi-Modal Processing", "timestamp": datetime.utcnow().isoformat()}

@multimodal_router.get("/metrics")
async def multimodal_metrics(service: MultiModalService = Depends(get_multimodal)):
    return await service.get_metrics()

@multimodal_router.post("/process")
async def process_media(config: Dict[str, Any], service: MultiModalService = Depends(get_multimodal)):
    result = await service.process_media(config)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result

@multimodal_router.get("/jobs")
async def list_jobs(service: MultiModalService = Depends(get_multimodal)):
    jobs = await service.list_jobs()
    return {"success": True, "jobs": jobs, "count": len(jobs)}

# ============================================================================
# FUNCTIONAL MEMORY ENDPOINTS
# ============================================================================

@memory_router.get("/health")
async def memory_health():
    return {"status": "healthy", "service": "Functional Memory", "timestamp": datetime.utcnow().isoformat()}

@memory_router.get("/metrics")
async def memory_metrics(service: FunctionalMemoryService = Depends(get_memory)):
    return await service.get_metrics()

@memory_router.post("/memory")
async def store_memory(config: Dict[str, Any], service: FunctionalMemoryService = Depends(get_memory)):
    result = await service.store_memory(config)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result

@memory_router.post("/retrieve")
async def retrieve_memory(query: Dict[str, Any], service: FunctionalMemoryService = Depends(get_memory)):
    result = await service.retrieve_memory(query)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result

@memory_router.get("/memories")
async def list_memories(service: FunctionalMemoryService = Depends(get_memory)):
    memories = await service.list_memories()
    return {"success": True, "memories": memories, "count": len(memories)}

# ============================================================================
# TRAINING PLATFORM ENDPOINTS
# ============================================================================

@training_router.get("/health")
async def training_health():
    return {"status": "healthy", "service": "Training Platform", "timestamp": datetime.utcnow().isoformat()}

@training_router.get("/metrics")
async def training_metrics(service: TrainingPlatformService = Depends(get_training)):
    return await service.get_metrics()

@training_router.post("/course")
async def create_course(config: Dict[str, Any], service: TrainingPlatformService = Depends(get_training)):
    result = await service.create_course(config)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result

@training_router.post("/course/{course_id}/enroll")
async def enroll_user(course_id: str, user_id: str, service: TrainingPlatformService = Depends(get_training)):
    result = await service.enroll_user(course_id, user_id)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result

@training_router.get("/courses")
async def list_courses(service: TrainingPlatformService = Depends(get_training)):
    courses = await service.list_courses()
    return {"success": True, "courses": courses, "count": len(courses)}

# ============================================================================
# KNOWLEDGE BASE ENDPOINTS
# ============================================================================

@knowledge_router.get("/health")
async def knowledge_health():
    return {"status": "healthy", "service": "Knowledge Base", "timestamp": datetime.utcnow().isoformat()}

@knowledge_router.get("/metrics")
async def knowledge_metrics(service: KnowledgeBaseService = Depends(get_knowledge)):
    return await service.get_metrics()

@knowledge_router.post("/document")
async def add_document(config: Dict[str, Any], service: KnowledgeBaseService = Depends(get_knowledge)):
    result = await service.add_document(config)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result

@knowledge_router.post("/search")
async def search_documents(query: Dict[str, Any], service: KnowledgeBaseService = Depends(get_knowledge)):
    result = await service.search_documents(query)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result

@knowledge_router.get("/documents")
async def list_documents(service: KnowledgeBaseService = Depends(get_knowledge)):
    documents = await service.list_documents()
    return {"success": True, "documents": documents, "count": len(documents)}

