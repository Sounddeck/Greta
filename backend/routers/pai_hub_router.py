"""
GRETA PAI Hub Router
Complete API endpoints for PAI agent build, deploy, and management
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from database import get_database
from services.llm_service import get_llm_service
from services.learning_service import LearningService
from services.pai_hub_service import PAIHubService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/pai-hub", tags=["PAI Hub"])

# Global PAI Hub instance
pai_hub_instance: Optional[PAIHubService] = None

# Pydantic models
class AgentBuildFromTemplateRequest(BaseModel):
    """Build agent from template request"""
    template_id: str = Field(..., description="Template ID (researcher, engineer, designer, pentester, architect, analyst, writer, teacher)")
    name: Optional[str] = Field(None, description="Custom agent name")
    config: Optional[Dict[str, Any]] = Field(None, description="Custom configuration")

class AgentCustomizeRequest(BaseModel):
    """Agent customization request"""
    agent_id: str = Field(..., description="Agent ID")
    customizations: Dict[str, Any] = Field(..., description="Customizations to apply")

class AgentDeployRequest(BaseModel):
    """Agent deployment request"""
    agent_id: str = Field(..., description="Agent ID to deploy")
    environment: Optional[str] = Field("local", description="Deployment environment (local, staging, production)")
    resources: Optional[Dict[str, Any]] = Field(None, description="Resource allocation")

class AgentUndeployRequest(BaseModel):
    """Agent undeploy request"""
    deployment_id: str = Field(..., description="Deployment ID to undeploy")

async def get_pai_hub() -> PAIHubService:
    """Get or initialize PAI Hub instance"""
    global pai_hub_instance
    
    if pai_hub_instance is None:
        db = await get_database()
        llm_service = await get_llm_service()
        learning_service = LearningService(db.client)
        await learning_service.initialize()
        
        pai_hub_instance = PAIHubService(db, llm_service, learning_service)
        await pai_hub_instance.initialize()
    
    return pai_hub_instance

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "PAI Hub",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/metrics")
async def get_metrics(pai_hub: PAIHubService = Depends(get_pai_hub)):
    """Get PAI Hub metrics"""
    try:
        metrics = await pai_hub.get_metrics()
        return metrics
    except Exception as e:
        logger.error(f"❌ Get metrics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== AGENT BUILDING ENDPOINTS ====================

@router.get("/templates")
async def get_templates(pai_hub: PAIHubService = Depends(get_pai_hub)):
    """Get all agent templates"""
    try:
        result = await pai_hub.get_templates()
        return result
    except Exception as e:
        logger.error(f"Error getting templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agents/build")
async def build_agent_from_template(
    request: AgentBuildFromTemplateRequest,
    pai_hub: PAIHubService = Depends(get_pai_hub)
):
    """Build a new PAI agent from a template"""
    try:
        result = await pai_hub.build_agent_from_template(
            request.template_id,
            request.config
        )

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error building agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agents/customize")
async def customize_agent(
    request: AgentCustomizeRequest,
    pai_hub: PAIHubService = Depends(get_pai_hub)
):
    """Customize an existing agent"""
    try:
        result = await pai_hub.customize_agent(
            request.agent_id,
            request.customizations
        )

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error customizing agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== AGENT DEPLOYMENT ENDPOINTS ====================

@router.post("/agents/deploy")
async def deploy_agent(
    request: AgentDeployRequest,
    pai_hub: PAIHubService = Depends(get_pai_hub)
):
    """Deploy a PAI agent"""
    try:
        deployment_config = {
            "environment": request.environment,
            "resources": request.resources
        }

        result = await pai_hub.deploy_agent(request.agent_id, deployment_config)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deploying agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agents/undeploy")
async def undeploy_agent(
    request: AgentUndeployRequest,
    pai_hub: PAIHubService = Depends(get_pai_hub)
):
    """Undeploy a PAI agent"""
    try:
        result = await pai_hub.undeploy_agent(request.deployment_id)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error undeploying agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== AGENT MANAGEMENT ENDPOINTS ====================

@router.get("/agents")
async def get_all_agents(pai_hub: PAIHubService = Depends(get_pai_hub)):
    """Get all PAI agents"""
    try:
        result = await pai_hub.get_all_agents()
        return result
    except Exception as e:
        logger.error(f"Error getting agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents/{agent_id}")
async def get_agent(
    agent_id: str,
    pai_hub: PAIHubService = Depends(get_pai_hub)
):
    """Get a specific PAI agent"""
    try:
        result = await pai_hub.get_agent(agent_id)

        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/agents/{agent_id}")
async def delete_agent(
    agent_id: str,
    pai_hub: PAIHubService = Depends(get_pai_hub)
):
    """Delete a PAI agent"""
    try:
        result = await pai_hub.delete_agent(agent_id)

        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/deployments")
async def get_all_deployments(pai_hub: PAIHubService = Depends(get_pai_hub)):
    """Get all deployments"""
    try:
        result = await pai_hub.get_all_deployments()
        return result
    except Exception as e:
        logger.error(f"Error getting deployments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/build-history")
async def get_build_history(
    limit: int = 100,
    pai_hub: PAIHubService = Depends(get_pai_hub)
):
    """Get build history"""
    try:
        result = await pai_hub.get_build_history(limit)
        return result
    except Exception as e:
        logger.error(f"Error getting build history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/activity-log")
async def get_activity_log(
    limit: int = 100,
    pai_hub: PAIHubService = Depends(get_pai_hub)
):
    """Get activity log"""
    try:
        result = await pai_hub.get_activity_log(limit)
        return result
    except Exception as e:
        logger.error(f"Error getting activity log: {e}")
        raise HTTPException(status_code=500, detail=str(e))


