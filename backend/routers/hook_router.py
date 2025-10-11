"""
Hook System Router for CPAS4/Greta
REST API endpoints for Hook Service (Week 19-20)
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict, Any, Callable
import logging
from pydantic import BaseModel

# Import hook service
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Greta" / "backend"))

try:
    from services.hook_service import (
        hook_service,
        EventType,
        HookType,
        Hook,
        Plugin,
        Middleware,
        HookExecutionResult,
        HookStats,
    )
    HOOK_SERVICE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Hook service not available: {e}")
    HOOK_SERVICE_AVAILABLE = False

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/hooks", tags=["Hooks"])

# Request/Response Models
class HookCreate(BaseModel):
    hook_id: str
    event_type: str
    hook_type: str = "pre"
    priority: int = 50
    condition: Optional[str] = None

class HookUpdate(BaseModel):
    priority: Optional[int] = None
    condition: Optional[str] = None
    enabled: Optional[bool] = None

class PluginCreate(BaseModel):
    plugin_id: str
    name: str
    description: Optional[str] = None
    enabled: bool = True

class MiddlewareCreate(BaseModel):
    middleware_id: str
    name: str
    priority: int = 50

class EventEmit(BaseModel):
    event_type: str
    event_data: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None

# ============================================================================
# Hook Registration Endpoints
# ============================================================================

@router.post("/register", response_model=Dict[str, Any], status_code=201)
async def register_hook(hook: HookCreate):
    """
    Register a new hook
    
    Args:
        hook_id: Unique hook identifier
        event_type: Event type to hook into (e.g., 'llm.generate.pre')
        hook_type: Hook type ('pre' or 'post')
        priority: Execution priority (0-100, higher = earlier)
        condition: Optional condition for execution
    
    Returns:
        Hook details
    """
    if not HOOK_SERVICE_AVAILABLE:
        raise HTTPException(503, "Hook service not available")
    
    try:
        # Convert to enums
        event = EventType(hook.event_type)
        htype = HookType(hook.hook_type)
        
        # Create a simple callback (in real implementation, this would be loaded from config)
        async def callback(event_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
            logger.info(f"Hook {hook.hook_id} executed for {hook.event_type}")
            return event_data
        
        # Register hook
        result = await hook_service.register_hook(
            hook.hook_id,
            event,
            callback,
            htype,
            hook.priority,
            hook.condition
        )
        
        return {
            "hook_id": result.hook_id,
            "event_type": result.event_type.value,
            "hook_type": result.hook_type.value,
            "priority": result.priority,
            "condition": result.condition,
            "enabled": result.enabled,
            "created_at": result.created_at,
        }
    
    except ValueError as e:
        raise HTTPException(400, f"Invalid event or hook type: {str(e)}")
    except Exception as e:
        logger.error(f"Hook registration error: {e}")
        raise HTTPException(500, f"Hook registration failed: {str(e)}")

@router.delete("/unregister/{hook_id}", status_code=204)
async def unregister_hook(hook_id: str):
    """Unregister a hook"""
    if not HOOK_SERVICE_AVAILABLE:
        raise HTTPException(503, "Hook service not available")
    
    try:
        result = await hook_service.unregister_hook(hook_id)
        if not result:
            raise HTTPException(404, "Hook not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hook unregistration error: {e}")
        raise HTTPException(500, f"Hook unregistration failed: {str(e)}")

@router.get("/list", response_model=List[Dict[str, Any]])
async def list_hooks(
    event_type: Optional[str] = None,
    hook_type: Optional[str] = None,
    enabled_only: bool = False
):
    """
    List all registered hooks
    
    Args:
        event_type: Filter by event type
        hook_type: Filter by hook type ('pre' or 'post')
        enabled_only: Only return enabled hooks
    
    Returns:
        List of hooks
    """
    if not HOOK_SERVICE_AVAILABLE:
        raise HTTPException(503, "Hook service not available")
    
    try:
        # Convert to enums if provided
        event = EventType(event_type) if event_type else None
        htype = HookType(hook_type) if hook_type else None
        
        hooks = await hook_service.list_hooks(event, htype, enabled_only)
        
        return [
            {
                "hook_id": h.hook_id,
                "event_type": h.event_type.value,
                "hook_type": h.hook_type.value,
                "priority": h.priority,
                "condition": h.condition,
                "enabled": h.enabled,
                "created_at": h.created_at,
            }
            for h in hooks
        ]
    
    except ValueError as e:
        raise HTTPException(400, f"Invalid event or hook type: {str(e)}")
    except Exception as e:
        logger.error(f"Hook listing error: {e}")
        raise HTTPException(500, f"Hook listing failed: {str(e)}")

@router.get("/{hook_id}", response_model=Dict[str, Any])
async def get_hook(hook_id: str):
    """Get hook details by ID"""
    if not HOOK_SERVICE_AVAILABLE:
        raise HTTPException(503, "Hook service not available")
    
    try:
        hook = await hook_service.get_hook(hook_id)
        if not hook:
            raise HTTPException(404, "Hook not found")
        
        return {
            "hook_id": hook.hook_id,
            "event_type": hook.event_type.value,
            "hook_type": hook.hook_type.value,
            "priority": hook.priority,
            "condition": hook.condition,
            "enabled": hook.enabled,
            "created_at": hook.created_at,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hook retrieval error: {e}")
        raise HTTPException(500, f"Hook retrieval failed: {str(e)}")

@router.patch("/{hook_id}", response_model=Dict[str, Any])
async def update_hook(hook_id: str, updates: HookUpdate):
    """Update hook configuration"""
    if not HOOK_SERVICE_AVAILABLE:
        raise HTTPException(503, "Hook service not available")
    
    try:
        update_dict = {}
        if updates.priority is not None:
            update_dict["priority"] = updates.priority
        if updates.condition is not None:
            update_dict["condition"] = updates.condition
        if updates.enabled is not None:
            update_dict["enabled"] = updates.enabled
        
        hook = await hook_service.update_hook(hook_id, update_dict)
        if not hook:
            raise HTTPException(404, "Hook not found")
        
        return {
            "hook_id": hook.hook_id,
            "event_type": hook.event_type.value,
            "hook_type": hook.hook_type.value,
            "priority": hook.priority,
            "condition": hook.condition,
            "enabled": hook.enabled,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hook update error: {e}")
        raise HTTPException(500, f"Hook update failed: {str(e)}")

# ============================================================================
# Hook Execution Endpoints
# ============================================================================

@router.post("/emit", response_model=Dict[str, Any])
async def emit_event(event: EventEmit):
    """
    Emit an event and execute all registered hooks
    
    Args:
        event_type: Event type to emit
        event_data: Event data
        context: Optional context
    
    Returns:
        Execution results
    """
    if not HOOK_SERVICE_AVAILABLE:
        raise HTTPException(503, "Hook service not available")
    
    try:
        # Convert to enum
        event_type = EventType(event.event_type)
        
        # Emit event
        result = await hook_service.emit_event(
            event_type,
            event.event_data,
            event.context or {}
        )
        
        return {
            "event_type": event.event_type,
            "hooks_executed": len(result.get("hooks_executed", [])),
            "modified_data": result.get("modified_data"),
            "context": result.get("context"),
        }
    
    except ValueError as e:
        raise HTTPException(400, f"Invalid event type: {str(e)}")
    except Exception as e:
        logger.error(f"Event emission error: {e}")
        raise HTTPException(500, f"Event emission failed: {str(e)}")

@router.post("/test/{hook_id}", response_model=Dict[str, Any])
async def test_hook(hook_id: str, test_data: Dict[str, Any]):
    """Test a hook with sample data"""
    if not HOOK_SERVICE_AVAILABLE:
        raise HTTPException(503, "Hook service not available")
    
    try:
        result = await hook_service.test_hook(hook_id, test_data)
        if not result:
            raise HTTPException(404, "Hook not found")
        
        return {
            "hook_id": hook_id,
            "success": result.get("success", False),
            "output": result.get("output"),
            "execution_time": result.get("execution_time"),
            "error": result.get("error"),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hook test error: {e}")
        raise HTTPException(500, f"Hook test failed: {str(e)}")

# ============================================================================
# Plugin Management Endpoints
# ============================================================================

@router.post("/plugins", response_model=Dict[str, Any], status_code=201)
async def register_plugin(plugin: PluginCreate):
    """Register a new plugin"""
    if not HOOK_SERVICE_AVAILABLE:
        raise HTTPException(503, "Hook service not available")
    
    try:
        # Create empty hooks list (in real implementation, loaded from config)
        result = await hook_service.register_plugin(
            plugin.plugin_id,
            plugin.name,
            [],
            plugin.description
        )
        
        return {
            "plugin_id": result.plugin_id,
            "name": result.name,
            "description": result.description,
            "enabled": result.enabled,
            "hooks_count": len(result.hooks),
            "created_at": result.created_at,
        }
    
    except Exception as e:
        logger.error(f"Plugin registration error: {e}")
        raise HTTPException(500, f"Plugin registration failed: {str(e)}")

@router.get("/plugins", response_model=List[Dict[str, Any]])
async def list_plugins(enabled_only: bool = False):
    """List all registered plugins"""
    if not HOOK_SERVICE_AVAILABLE:
        raise HTTPException(503, "Hook service not available")

    try:
        plugins = await hook_service.list_plugins(enabled_only)

        return [
            {
                "plugin_id": p.plugin_id,
                "name": p.name,
                "description": p.description,
                "enabled": p.enabled,
                "hooks_count": len(p.hooks),
                "created_at": p.created_at,
            }
            for p in plugins
        ]

    except Exception as e:
        logger.error(f"Plugin listing error: {e}")
        raise HTTPException(500, f"Plugin listing failed: {str(e)}")

@router.get("/plugins/{plugin_id}", response_model=Dict[str, Any])
async def get_plugin(plugin_id: str):
    """Get plugin details"""
    if not HOOK_SERVICE_AVAILABLE:
        raise HTTPException(503, "Hook service not available")

    try:
        plugin = await hook_service.get_plugin(plugin_id)
        if not plugin:
            raise HTTPException(404, "Plugin not found")

        return {
            "plugin_id": plugin.plugin_id,
            "name": plugin.name,
            "description": plugin.description,
            "enabled": plugin.enabled,
            "hooks": [h.hook_id for h in plugin.hooks],
            "created_at": plugin.created_at,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Plugin retrieval error: {e}")
        raise HTTPException(500, f"Plugin retrieval failed: {str(e)}")

@router.post("/plugins/{plugin_id}/enable", response_model=Dict[str, bool])
async def enable_plugin(plugin_id: str):
    """Enable a plugin and all its hooks"""
    if not HOOK_SERVICE_AVAILABLE:
        raise HTTPException(503, "Hook service not available")

    try:
        result = await hook_service.enable_plugin(plugin_id)
        if not result:
            raise HTTPException(404, "Plugin not found")

        return {"enabled": True}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Plugin enable error: {e}")
        raise HTTPException(500, f"Plugin enable failed: {str(e)}")

@router.post("/plugins/{plugin_id}/disable", response_model=Dict[str, bool])
async def disable_plugin(plugin_id: str):
    """Disable a plugin and all its hooks"""
    if not HOOK_SERVICE_AVAILABLE:
        raise HTTPException(503, "Hook service not available")

    try:
        result = await hook_service.disable_plugin(plugin_id)
        if not result:
            raise HTTPException(404, "Plugin not found")

        return {"disabled": True}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Plugin disable error: {e}")
        raise HTTPException(500, f"Plugin disable failed: {str(e)}")

@router.delete("/plugins/{plugin_id}", status_code=204)
async def unregister_plugin(plugin_id: str):
    """Unregister a plugin"""
    if not HOOK_SERVICE_AVAILABLE:
        raise HTTPException(503, "Hook service not available")

    try:
        result = await hook_service.unregister_plugin(plugin_id)
        if not result:
            raise HTTPException(404, "Plugin not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Plugin unregistration error: {e}")
        raise HTTPException(500, f"Plugin unregistration failed: {str(e)}")

# ============================================================================
# Middleware Management Endpoints
# ============================================================================

@router.post("/middleware", response_model=Dict[str, Any], status_code=201)
async def add_middleware(middleware: MiddlewareCreate):
    """Add event middleware"""
    if not HOOK_SERVICE_AVAILABLE:
        raise HTTPException(503, "Hook service not available")

    try:
        # Create a simple middleware function
        async def middleware_func(event_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
            logger.info(f"Middleware {middleware.middleware_id} processing event")
            return event_data

        result = await hook_service.add_middleware(
            middleware.middleware_id,
            middleware_func,
            middleware.priority
        )

        return {
            "middleware_id": result.middleware_id,
            "name": middleware.name,
            "priority": result.priority,
            "enabled": result.enabled,
        }

    except Exception as e:
        logger.error(f"Middleware add error: {e}")
        raise HTTPException(500, f"Middleware add failed: {str(e)}")

@router.get("/middleware", response_model=List[Dict[str, Any]])
async def list_middleware():
    """List all middleware"""
    if not HOOK_SERVICE_AVAILABLE:
        raise HTTPException(503, "Hook service not available")

    try:
        middleware_list = await hook_service.list_middleware()

        return [
            {
                "middleware_id": m.middleware_id,
                "priority": m.priority,
                "enabled": m.enabled,
            }
            for m in middleware_list
        ]

    except Exception as e:
        logger.error(f"Middleware listing error: {e}")
        raise HTTPException(500, f"Middleware listing failed: {str(e)}")

@router.delete("/middleware/{middleware_id}", status_code=204)
async def remove_middleware(middleware_id: str):
    """Remove middleware"""
    if not HOOK_SERVICE_AVAILABLE:
        raise HTTPException(503, "Hook service not available")

    try:
        result = await hook_service.remove_middleware(middleware_id)
        if not result:
            raise HTTPException(404, "Middleware not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Middleware removal error: {e}")
        raise HTTPException(500, f"Middleware removal failed: {str(e)}")

# ============================================================================
# Statistics & Monitoring Endpoints
# ============================================================================

@router.get("/stats", response_model=Dict[str, Any])
async def get_hook_stats():
    """Get hook system statistics"""
    if not HOOK_SERVICE_AVAILABLE:
        raise HTTPException(503, "Hook service not available")

    try:
        stats = await hook_service.get_stats()

        return {
            "total_hooks": stats.total_hooks,
            "total_plugins": stats.total_plugins,
            "total_events": stats.total_events,
            "total_executions": stats.total_executions,
            "avg_execution_time": stats.avg_execution_time,
            "hooks_by_event": stats.hooks_by_event,
            "executions_by_event": stats.executions_by_event,
            "failed_executions": stats.failed_executions,
        }

    except Exception as e:
        logger.error(f"Stats retrieval error: {e}")
        raise HTTPException(500, f"Stats retrieval failed: {str(e)}")

@router.post("/stats/reset", status_code=204)
async def reset_hook_stats():
    """Reset hook system statistics"""
    if not HOOK_SERVICE_AVAILABLE:
        raise HTTPException(503, "Hook service not available")

    try:
        await hook_service.reset_stats()
    except Exception as e:
        logger.error(f"Stats reset error: {e}")
        raise HTTPException(500, f"Stats reset failed: {str(e)}")

@router.get("/events", response_model=List[str])
async def get_event_types():
    """Get list of all event types"""
    if not HOOK_SERVICE_AVAILABLE:
        raise HTTPException(503, "Hook service not available")

    try:
        return await hook_service.get_event_types()
    except Exception as e:
        logger.error(f"Event types retrieval error: {e}")
        raise HTTPException(500, f"Event types retrieval failed: {str(e)}")

@router.get("/health", response_model=Dict[str, Any])
async def get_hook_health():
    """Get hook system health status"""
    if not HOOK_SERVICE_AVAILABLE:
        return {"status": "unavailable", "message": "Hook service not available"}

    try:
        health = await hook_service.get_health()
        return {
            "status": "healthy" if health.get("healthy") else "unhealthy",
            "total_hooks": health.get("total_hooks", 0),
            "enabled_hooks": health.get("enabled_hooks", 0),
            "total_plugins": health.get("total_plugins", 0),
            "enabled_plugins": health.get("enabled_plugins", 0),
            "uptime": health.get("uptime", 0),
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {"status": "error", "message": str(e)}

