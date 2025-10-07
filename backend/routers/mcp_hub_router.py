"""
GRETA MCP Hub Router
Complete API endpoints for MCP server management
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
from services.mcp_hub_service import MCPHubService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/mcp", tags=["MCP Hub"])

# Global instance
mcp_instance: Optional[MCPHubService] = None

# Pydantic models
class MCPServerRegisterRequest(BaseModel):
    """MCP server registration request"""
    name: str = Field(..., description="Server name")
    url: str = Field(..., description="Server URL")
    capabilities: List[str] = Field(..., description="Server capabilities")
    config: Dict[str, Any] = Field(default_factory=dict, description="Server configuration")

class CapabilityExecuteRequest(BaseModel):
    """Capability execution request"""
    capability: str = Field(..., description="Capability to execute")
    params: Dict[str, Any] = Field(default_factory=dict, description="Execution parameters")

async def get_mcp() -> MCPHubService:
    """Get or initialize MCP Hub instance"""
    global mcp_instance
    
    if mcp_instance is None:
        db = database.database
        llm_service = await get_llm_service()
        learning_service = LearningService(db.client if hasattr(db, 'client') else None)
        await learning_service.initialize()
        
        mcp_instance = MCPHubService(db, llm_service, learning_service)
        await mcp_instance.initialize()
    
    return mcp_instance

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "mcp_hub",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/metrics")
async def get_metrics(service: MCPHubService = Depends(get_mcp)):
    """Get service metrics"""
    return await service.get_metrics()

@router.post("/server/register")
async def register_server(
    request: MCPServerRegisterRequest,
    service: MCPHubService = Depends(get_mcp)
):
    """Register a new MCP server"""
    try:
        result = await service.register_server(request.dict())
        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"Error registering server: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/server/{server_id}")
async def get_server(
    server_id: str,
    service: MCPHubService = Depends(get_mcp)
):
    """Get server details"""
    try:
        result = await service.get_server(server_id)
        if not result:
            raise HTTPException(status_code=404, detail="Server not found")
        return {"success": True, "data": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting server: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/servers")
async def list_servers(service: MCPHubService = Depends(get_mcp)):
    """List all MCP servers"""
    try:
        result = await service.get_all_servers()
        return {"success": True, "servers": result, "count": len(result)}
    except Exception as e:
        logger.error(f"Error listing servers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/server/{server_id}")
async def update_server(
    server_id: str,
    updates: Dict[str, Any],
    service: MCPHubService = Depends(get_mcp)
):
    """Update an MCP server"""
    try:
        result = await service.update_server(server_id, updates)
        if not result:
            raise HTTPException(status_code=404, detail="Server not found")
        return {"success": True, "server": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating server: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/server/{server_id}")
async def delete_server(
    server_id: str,
    service: MCPHubService = Depends(get_mcp)
):
    """Delete an MCP server"""
    try:
        result = await service.delete_server(server_id)
        if not result:
            raise HTTPException(status_code=404, detail="Server not found")
        return {"success": True, "message": f"Server {server_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting server: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== TOOL MANAGEMENT ====================

@router.get("/tools")
async def get_tools(service: MCPHubService = Depends(get_mcp)):
    """Get available tools"""
    try:
        return {
            "success": True,
            "tools": service.available_tools,
            "count": len(service.available_tools)
        }
    except Exception as e:
        logger.error(f"Error getting tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tools/{tool_name}")
async def get_tool(
    tool_name: str,
    service: MCPHubService = Depends(get_mcp)
):
    """Get tool details"""
    try:
        tool = service.available_tools.get(tool_name)
        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")
        return {"success": True, "tool": tool}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting tool: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== AGENT TOOL ACCESS ====================

@router.post("/agent/{agent_id}/tools")
async def provide_tools_to_agent(
    agent_id: str,
    agent_type: str,
    service: MCPHubService = Depends(get_mcp)
):
    """Provide tools to an agent"""
    try:
        result = await service.provide_tools_to_agent(agent_id, agent_type)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error providing tools to agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== TOOL EXECUTION ====================

class ToolExecuteRequest(BaseModel):
    """Tool execution request"""
    tool_name: str = Field(..., description="Tool name")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    session_id: Optional[str] = Field(None, description="Session ID")

@router.post("/execute")
async def execute_tool(
    request: ToolExecuteRequest,
    service: MCPHubService = Depends(get_mcp)
):
    """Execute a tool"""
    try:
        result = await service.execute_tool(
            request.tool_name,
            request.parameters,
            request.session_id
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing tool: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== STATUS ====================

@router.get("/status")
async def get_status(service: MCPHubService = Depends(get_mcp)):
    """Get MCP hub status"""
    try:
        result = await service.get_tool_status()
        return result
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

