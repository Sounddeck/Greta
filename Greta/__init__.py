"""
Enhanced GRETA Router Package - Complete PAI System
All routers loaded with error handling and functional implementations
"""
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Import all routers with safe loading
try:
    from .llm import router as llm_router
    llm_imported = True
except ImportError as e:
    logger.warning(f"LLM router not available: {e}")
    llm_router = None
    llm_imported = False

try:
    from .web_commands import router as web_commands_router
    web_commands_imported = True
except ImportError as e:
    logger.warning(f"Web commands router not available: {e}")
    web_commands_router = None
    web_commands_imported = False

try:
    from .jina_mcp import router as jina_mcp_router
    jina_mcp_imported = True
except ImportError as e:
    logger.warning(f"Jina MCP router not available: {e}")
    jina_mcp_router = None
    jina_mcp_imported = False

# Router availability status
available_routers = {
    "llm": llm_imported,
    "web_commands": web_commands_imported,
    "jina_mcp": jina_mcp_imported
}

def get_available_routers() -> Dict[str, Any]:
    """Get status of all routers"""
    return {
        "total": len(available_routers),
        "available": sum(available_routers.values()),
        "details": available_routers
    }

def get_routers() -> Dict[str, Any]:
    """Get all loaded routers"""
    routers = {}
    
    if llm_router:
        routers["llm"] = llm_router
    if web_commands_router:
        routers["web_commands"] = web_commands_router
    if jina_mcp_router:
        routers["jina_mcp"] = jina_mcp_router
    
    return routers

# Import individual router functions for backwards compatibility
def import_router_safe(router_name: str):
    """Safely import a specific router by name"""
    router_map = {
        "llm": llm_router,
        "web_commands": web_commands_router,
        "jina_mcp": jina_mcp_router
    }
    
    return router_map.get(router_name)

logger.info(f"Router initialization complete: {get_available_routers()['available']} of {get_available_routers()['total']} routers loaded")

