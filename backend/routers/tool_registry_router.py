"""
Tool Registry Router for CPAS5
MCP-Inspired REST API endpoints for tool discovery, registration, and execution
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from services.tool_registry import (
    ToolRegistry,
    ToolDefinition,
    ToolCategory,
    PermissionLevel,
    ToolPermission,
    ToolExecution
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tools", tags=["Tool Registry"])

# Global tool registry instance
tool_registry_instance: Optional[ToolRegistry] = None


async def get_tool_registry() -> ToolRegistry:
    """Get or initialize tool registry instance"""
    global tool_registry_instance
    
    if tool_registry_instance is None:
        try:
            logger.info("🔧 Initializing Tool Registry for the first time...")
            tool_registry_instance = ToolRegistry()
            await tool_registry_instance.initialize()
            logger.info("✅ Tool Registry initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Tool Registry: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize Tool Registry: {str(e)}")
    
    return tool_registry_instance


# ==================== REQUEST/RESPONSE MODELS ====================

class ToolDiscoveryQuery(BaseModel):
    """Tool discovery query parameters"""
    category: Optional[ToolCategory] = None
    permission_level: Optional[PermissionLevel] = None
    search_query: Optional[str] = None
    tags: Optional[List[str]] = None


class ToolExecuteRequest(BaseModel):
    """Tool execution request"""
    tool_id: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    agent_id: str = "default_agent"


class ToolExecuteResponse(BaseModel):
    """Tool execution response"""
    execution_id: str
    tool_id: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float
    timestamp: datetime


class ToolListResponse(BaseModel):
    """Tool list response"""
    tools: List[ToolDefinition]
    total: int
    category_counts: Dict[str, int]


class ToolStatsResponse(BaseModel):
    """Tool statistics response"""
    total_tools: int
    total_executions: int
    successful_executions: int
    failed_executions: int
    average_execution_time: float
    executions_by_category: Dict[str, int]
    executions_by_agent: Dict[str, int]
    most_used_tools: List[Dict[str, Any]]


class ToolExecutionHistoryResponse(BaseModel):
    """Tool execution history response"""
    executions: List[Dict[str, Any]]
    total: int
    limit: int


class ToolMetricsResponse(BaseModel):
    """Tool metrics response (alias for stats)"""
    metrics: Dict[str, Any]


# ==================== TOOL DISCOVERY ENDPOINTS ====================

@router.get("/discover", response_model=ToolListResponse)
async def discover_tools(
    category: Optional[ToolCategory] = None,
    permission_level: Optional[PermissionLevel] = None,
    search_query: Optional[str] = None,
    limit: int = Query(100, le=1000),
    registry: ToolRegistry = Depends(get_tool_registry)
):
    """
    Discover available tools with filtering
    
    MCP-inspired tool discovery endpoint that allows filtering by:
    - Category (osint, data_processing, web_api, etc.)
    - Permission level (public, restricted, admin)
    - Search query (searches in name and description)
    """
    try:
        # Get all tools
        all_tools = list(registry.tools.values())
        
        # Apply filters
        filtered_tools = all_tools
        
        if category:
            filtered_tools = [t for t in filtered_tools if t.category == category]
        
        if permission_level:
            filtered_tools = [t for t in filtered_tools if t.permission_level == permission_level]
        
        if search_query:
            search_lower = search_query.lower()
            filtered_tools = [
                t for t in filtered_tools
                if search_lower in t.name.lower() or search_lower in t.description.lower()
            ]
        
        # Apply limit
        filtered_tools = filtered_tools[:limit]
        
        # Calculate category counts
        category_counts = {}
        for tool in all_tools:
            category_counts[tool.category.value] = category_counts.get(tool.category.value, 0) + 1
        
        return ToolListResponse(
            tools=filtered_tools,
            total=len(filtered_tools),
            category_counts=category_counts
        )
        
    except Exception as e:
        logger.error(f"Tool discovery failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== TOOL STATISTICS ENDPOINTS ====================

@router.get("/stats", response_model=ToolStatsResponse)
async def get_tool_stats(
    registry: ToolRegistry = Depends(get_tool_registry)
):
    """Get tool registry statistics"""
    try:
        metrics = registry.metrics

        # Calculate most used tools
        tool_usage = {}
        for tool_id, tool in registry.tools.items():
            usage_count = metrics.get("executions_by_category", {}).get(tool.category.value, 0)
            tool_usage[tool_id] = {
                "tool_id": tool_id,
                "name": tool.name,
                "category": tool.category.value,
                "usage_count": usage_count
            }

        most_used = sorted(tool_usage.values(), key=lambda x: x["usage_count"], reverse=True)[:10]

        avg_time = (
            metrics["total_execution_time"] / metrics["total_executions"]
            if metrics["total_executions"] > 0
            else 0.0
        )

        return ToolStatsResponse(
            total_tools=len(registry.tools),
            total_executions=metrics["total_executions"],
            successful_executions=metrics["successful_executions"],
            failed_executions=metrics["failed_executions"],
            average_execution_time=avg_time,
            executions_by_category=metrics["executions_by_category"],
            executions_by_agent=metrics["executions_by_agent"],
            most_used_tools=most_used
        )

    except Exception as e:
        logger.error(f"Get tool stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== HEALTH CHECK ====================

@router.get("/health")
async def health_check(registry: ToolRegistry = Depends(get_tool_registry)):
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "tool_registry",
        "total_tools": len(registry.tools),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/category/{category}", response_model=ToolListResponse)
async def get_tools_by_category(
    category: ToolCategory,
    limit: int = Query(100, le=1000),
    registry: ToolRegistry = Depends(get_tool_registry)
):
    """Get all tools in a specific category"""
    try:
        tools = [t for t in registry.tools.values() if t.category == category]
        tools = tools[:limit]

        category_counts = {category.value: len(tools)}

        return ToolListResponse(
            tools=tools,
            total=len(tools),
            category_counts=category_counts
        )

    except Exception as e:
        logger.error(f"Get tools by category failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== TOOL EXECUTION ENDPOINTS ====================

@router.post("/execute", response_model=ToolExecuteResponse)
async def execute_tool(
    request: ToolExecuteRequest,
    registry: ToolRegistry = Depends(get_tool_registry)
):
    """
    Execute a tool with parameters
    
    MCP-inspired tool execution endpoint with:
    - Permission checking
    - Rate limiting
    - Execution tracking
    - Error handling
    """
    try:
        result = await registry.execute_tool(
            tool_id=request.tool_id,
            parameters=request.parameters,
            agent_id=request.agent_id
        )
        
        return ToolExecuteResponse(
            execution_id=result.get("execution_id", "unknown"),
            tool_id=request.tool_id,
            success=result.get("success", False),
            result=result.get("result"),
            error=result.get("error"),
            execution_time=result.get("execution_time", 0.0),
            timestamp=datetime.utcnow()
        )
        
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== EXECUTION HISTORY ENDPOINTS ====================

@router.get("/executions", response_model=ToolExecutionHistoryResponse)
async def get_execution_history(
    category: Optional[str] = None,
    agent_id: Optional[str] = None,
    limit: int = Query(50, le=500),
    registry: ToolRegistry = Depends(get_tool_registry)
):
    """Get tool execution history"""
    try:
        # Get execution history from registry
        executions = []

        # Filter executions by category if specified
        if category:
            # Get tools in category
            category_tools = [
                tool_id for tool_id, tool in registry.tools.items()
                if tool.category.value == category
            ]
            # Filter executions
            for tool_id in category_tools:
                tool_executions = registry.execution_history.get(tool_id, [])
                executions.extend(tool_executions)
        else:
            # Get all executions
            for tool_id, tool_executions in registry.execution_history.items():
                executions.extend(tool_executions)

        # Filter by agent_id if specified
        if agent_id:
            executions = [e for e in executions if e.get("agent_id") == agent_id]

        # Sort by timestamp (most recent first)
        executions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        # Limit results
        executions = executions[:limit]

        return ToolExecutionHistoryResponse(
            executions=executions,
            total=len(executions),
            limit=limit
        )

    except Exception as e:
        logger.error(f"Get execution history failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=ToolMetricsResponse)
async def get_tool_metrics(
    category: Optional[str] = None,
    registry: ToolRegistry = Depends(get_tool_registry)
):
    """Get tool metrics (alias for stats endpoint)"""
    try:
        metrics = registry.get_metrics()

        # Filter by category if specified
        if category:
            category_tools = [
                tool_id for tool_id, tool in registry.tools.items()
                if tool.category.value == category
            ]

            # Calculate category-specific metrics
            category_executions = sum(
                metrics.get("executions_by_tool", {}).get(tool_id, 0)
                for tool_id in category_tools
            )

            # Get executions for category tools
            total_exec = 0
            success_exec = 0
            failed_exec = 0
            total_time = 0.0

            for tool_id in category_tools:
                tool_metrics = metrics.get("tools", {}).get(tool_id, {})
                total_exec += tool_metrics.get("total_executions", 0)
                success_exec += tool_metrics.get("successful_executions", 0)
                failed_exec += tool_metrics.get("failed_executions", 0)
                total_time += tool_metrics.get("total_execution_time", 0.0)

            avg_time = total_time / total_exec if total_exec > 0 else 0.0

            return ToolMetricsResponse(
                metrics={
                    "total_executions": total_exec,
                    "successful_executions": success_exec,
                    "failed_executions": failed_exec,
                    "avg_execution_time": avg_time,
                    "tools_used": len(category_tools),
                    "executions_by_tool": {
                        tool_id: metrics.get("executions_by_tool", {}).get(tool_id, 0)
                        for tool_id in category_tools
                    },
                    "executions_by_category": {category: category_executions}
                }
            )

        # Return all metrics
        avg_time = (
            metrics["total_execution_time"] / metrics["total_executions"]
            if metrics["total_executions"] > 0
            else 0.0
        )

        return ToolMetricsResponse(
            metrics={
                "total_executions": metrics["total_executions"],
                "successful_executions": metrics["successful_executions"],
                "failed_executions": metrics["failed_executions"],
                "avg_execution_time": avg_time,
                "tools_used": len(registry.tools),
                "executions_by_tool": metrics.get("executions_by_tool", {}),
                "executions_by_category": metrics.get("executions_by_category", {})
            }
        )

    except Exception as e:
        logger.error(f"Get tool metrics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== TOOL DETAILS ENDPOINT (CATCH-ALL) ====================
# NOTE: This must be defined LAST to avoid catching other routes

@router.get("/{tool_id}", response_model=ToolDefinition)
async def get_tool(
    tool_id: str,
    registry: ToolRegistry = Depends(get_tool_registry)
):
    """Get tool details by ID"""
    try:
        tool = registry.tools.get(tool_id)
        if not tool:
            raise HTTPException(status_code=404, detail=f"Tool not found: {tool_id}")

        return tool

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get tool failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
