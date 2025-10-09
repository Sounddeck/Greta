"""
MCP Hub Router for CPAS4
REST API endpoints for MCP Server Discovery and Management
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
import logging
from services.mcp_service import MCPHubService
from models.mcp_models import *
from database import Database

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/mcp", tags=["MCP Hub"])

async def get_service() -> MCPHubService:
    db = Database()
    await db.connect()
    return MCPHubService(db)

# MCP Server endpoints
@router.post("/servers", response_model=MCPServer, status_code=201)
async def create_server(server: MCPServerCreate, svc: MCPHubService = Depends(get_service)):
    return await svc.create_server(server)

@router.get("/servers/{server_id}", response_model=MCPServer)
async def get_server(server_id: str, svc: MCPHubService = Depends(get_service)):
    srv = await svc.get_server(server_id)
    if not srv: raise HTTPException(404, "MCP Server not found")
    return srv

@router.patch("/servers/{server_id}", response_model=MCPServer)
async def update_server(server_id: str, server_update: MCPServerUpdate, svc: MCPHubService = Depends(get_service)):
    srv = await svc.update_server(server_id, server_update)
    if not srv: raise HTTPException(404, "MCP Server not found")
    return srv

@router.delete("/servers/{server_id}", status_code=204)
async def delete_server(server_id: str, svc: MCPHubService = Depends(get_service)):
    if not await svc.delete_server(server_id): raise HTTPException(404, "MCP Server not found")

@router.get("/servers", response_model=List[MCPServer])
async def list_servers(status: Optional[MCPServerStatus] = None, discovery_source: Optional[DiscoverySource] = None, limit: int = Query(100, le=1000), svc: MCPHubService = Depends(get_service)):
    return await svc.list_servers(status, discovery_source, limit)

# MCP Tool endpoints
@router.post("/servers/{server_id}/tools", response_model=MCPTool, status_code=201)
async def create_tool(server_id: str, tool: MCPToolBase, svc: MCPHubService = Depends(get_service)):
    # Verify server exists
    srv = await svc.get_server(server_id)
    if not srv: raise HTTPException(404, "MCP Server not found")
    return await svc.create_tool(tool, server_id)

@router.get("/tools/{tool_id}", response_model=MCPTool)
async def get_tool(tool_id: str, svc: MCPHubService = Depends(get_service)):
    tool = await svc.get_tool(tool_id)
    if not tool: raise HTTPException(404, "Tool not found")
    return tool

@router.delete("/tools/{tool_id}", status_code=204)
async def delete_tool(tool_id: str, svc: MCPHubService = Depends(get_service)):
    if not await svc.delete_tool(tool_id): raise HTTPException(404, "Tool not found")

@router.get("/tools", response_model=List[MCPTool])
async def list_tools(server_id: Optional[str] = None, tool_type: Optional[MCPToolType] = None, limit: int = Query(100, le=1000), svc: MCPHubService = Depends(get_service)):
    return await svc.list_tools(server_id, tool_type, limit)

# Discovery Config endpoints
@router.post("/discovery/configs", response_model=DiscoveryConfig, status_code=201)
async def create_discovery_config(config: DiscoveryConfigCreate, svc: MCPHubService = Depends(get_service)):
    return await svc.create_discovery_config(config)

@router.get("/discovery/configs/{config_id}", response_model=DiscoveryConfig)
async def get_discovery_config(config_id: str, svc: MCPHubService = Depends(get_service)):
    cfg = await svc.get_discovery_config(config_id)
    if not cfg: raise HTTPException(404, "Discovery config not found")
    return cfg

@router.patch("/discovery/configs/{config_id}", response_model=DiscoveryConfig)
async def update_discovery_config(config_id: str, config_update: DiscoveryConfigUpdate, svc: MCPHubService = Depends(get_service)):
    cfg = await svc.update_discovery_config(config_id, config_update)
    if not cfg: raise HTTPException(404, "Discovery config not found")
    return cfg

@router.delete("/discovery/configs/{config_id}", status_code=204)
async def delete_discovery_config(config_id: str, svc: MCPHubService = Depends(get_service)):
    if not await svc.delete_discovery_config(config_id): raise HTTPException(404, "Discovery config not found")

@router.get("/discovery/configs", response_model=List[DiscoveryConfig])
async def list_discovery_configs(enabled: Optional[bool] = None, limit: int = Query(100, le=1000), svc: MCPHubService = Depends(get_service)):
    return await svc.list_discovery_configs(enabled, limit)

# Discovery Events endpoints
@router.post("/discovery/events", response_model=DiscoveryEvent, status_code=201)
async def create_discovery_event(event: DiscoveryEventBase, svc: MCPHubService = Depends(get_service)):
    return await svc.create_discovery_event(event)

@router.get("/discovery/events", response_model=List[DiscoveryEvent])
async def list_discovery_events(processed: Optional[bool] = None, limit: int = Query(100, le=1000), svc: MCPHubService = Depends(get_service)):
    return await svc.list_discovery_events(processed, limit)

# Health Check endpoints
@router.post("/health-checks", response_model=HealthCheck, status_code=201)
async def create_health_check(check: HealthCheckBase, svc: MCPHubService = Depends(get_service)):
    return await svc.create_health_check(check)

@router.get("/health-checks", response_model=List[HealthCheck])
async def list_health_checks(server_id: Optional[str] = None, limit: int = Query(100, le=1000), svc: MCPHubService = Depends(get_service)):
    return await svc.list_health_checks(server_id, limit)

# Agent Integration endpoints
@router.post("/integrations", response_model=AgentMCPIntegration, status_code=201)
async def create_integration(integration: AgentMCPIntegrationCreate, svc: MCPHubService = Depends(get_service)):
    return await svc.create_integration(integration)

@router.get("/integrations/{integration_id}", response_model=AgentMCPIntegration)
async def get_integration(integration_id: str, svc: MCPHubService = Depends(get_service)):
    intg = await svc.get_integration(integration_id)
    if not intg: raise HTTPException(404, "Integration not found")
    return intg

@router.patch("/integrations/{integration_id}", response_model=AgentMCPIntegration)
async def update_integration(integration_id: str, integration_update: AgentMCPIntegrationUpdate, svc: MCPHubService = Depends(get_service)):
    intg = await svc.update_integration(integration_id, integration_update)
    if not intg: raise HTTPException(404, "Integration not found")
    return intg

@router.delete("/integrations/{integration_id}", status_code=204)
async def delete_integration(integration_id: str, svc: MCPHubService = Depends(get_service)):
    if not await svc.delete_integration(integration_id): raise HTTPException(404, "Integration not found")

@router.get("/integrations", response_model=List[AgentMCPIntegration])
async def list_integrations(agent_id: Optional[str] = None, server_id: Optional[str] = None, limit: int = Query(100, le=1000), svc: MCPHubService = Depends(get_service)):
    return await svc.list_integrations(agent_id, server_id, limit)

# Notifications endpoints
@router.get("/notifications", response_model=List[MCPNotification])
async def list_notifications(read: Optional[bool] = None, limit: int = Query(100, le=1000), svc: MCPHubService = Depends(get_service)):
    return await svc.list_notifications(read, limit)

# Server Catalog
@router.get("/catalog", response_model=List[ServerCatalogEntry])
async def get_server_catalog(limit: int = Query(100, le=1000), svc: MCPHubService = Depends(get_service)):
    return await svc.get_server_catalog(limit)

# Statistics
@router.get("/stats", response_model=MCPHubStats)
async def get_stats(svc: MCPHubService = Depends(get_service)):
    return await svc.get_stats()

# Health check
@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "module": "MCP Hub",
        "features": [
            "MCP Server Management",
            "Tool Discovery",
            "Discovery Automation",
            "Health Monitoring",
            "Agent Integration",
            "Notifications",
            "Server Catalog",
            "Statistics"
        ],
        "endpoints": 30,
        "database": "MongoDB/File Storage"
    }

