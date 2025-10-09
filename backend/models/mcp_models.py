"""
MCP Hub Module Data Models for CPAS4
Intelligent MCP server discovery, registration, and monitoring
"""
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum


# Enums
class MCPServerStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DISCOVERING = "discovering"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class MCPToolType(str, Enum):
    FUNCTION = "function"
    RESOURCE = "resource"
    PROMPT = "prompt"
    SAMPLING = "sampling"


class DiscoverySource(str, Enum):
    OFFICIAL_REGISTRY = "official_registry"
    GITHUB = "github"
    NPM = "npm"
    PYPI = "pypi"
    MANUAL = "manual"
    AUTO_DETECTED = "auto_detected"


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


# MCP Server Models
class MCPToolBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    tool_type: MCPToolType
    parameters: Dict[str, Any] = Field(default_factory=dict)
    returns: Optional[Dict[str, Any]] = None
    examples: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MCPTool(MCPToolBase):
    tool_id: str
    server_id: str
    usage_count: int = 0
    last_used: Optional[datetime] = None
    created_at: datetime


class MCPServerBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    version: str = Field(default="1.0.0")
    author: Optional[str] = None
    homepage: Optional[str] = None
    repository: Optional[str] = None
    license: Optional[str] = None
    endpoint: Optional[str] = None
    transport: str = Field(default="stdio")  # stdio, sse, http
    command: Optional[str] = None
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    capabilities: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    discovery_source: DiscoverySource = DiscoverySource.MANUAL
    auto_start: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MCPServerCreate(MCPServerBase):
    pass


class MCPServerUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    version: Optional[str] = None
    status: Optional[MCPServerStatus] = None
    endpoint: Optional[str] = None
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    capabilities: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    auto_start: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None


class MCPServer(MCPServerBase):
    server_id: str
    status: MCPServerStatus = MCPServerStatus.INACTIVE
    health_status: HealthStatus = HealthStatus.UNKNOWN
    tool_count: int = 0
    last_health_check: Optional[datetime] = None
    last_discovery_check: Optional[datetime] = None
    uptime_seconds: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    discovered_at: Optional[datetime] = None


# Discovery Models
class DiscoveryConfigBase(BaseModel):
    source: DiscoverySource
    enabled: bool = True
    check_interval_seconds: int = Field(default=3600, ge=60)  # 1 hour default
    registry_url: Optional[str] = None
    search_keywords: List[str] = Field(default_factory=list)
    filters: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DiscoveryConfigCreate(DiscoveryConfigBase):
    pass


class DiscoveryConfigUpdate(BaseModel):
    enabled: Optional[bool] = None
    check_interval_seconds: Optional[int] = Field(None, ge=60)
    registry_url: Optional[str] = None
    search_keywords: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class DiscoveryConfig(DiscoveryConfigBase):
    config_id: str
    last_check: Optional[datetime] = None
    servers_discovered: int = 0
    created_at: datetime
    updated_at: datetime


# Discovery Event Models
class DiscoveryEventBase(BaseModel):
    source: DiscoverySource
    event_type: str  # new_server, server_updated, server_removed
    server_name: str
    server_data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DiscoveryEvent(DiscoveryEventBase):
    event_id: str
    processed: bool = False
    server_id: Optional[str] = None
    created_at: datetime
    processed_at: Optional[datetime] = None


# Health Check Models
class HealthCheckBase(BaseModel):
    server_id: str
    status: HealthStatus
    response_time_ms: float
    error_message: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class HealthCheck(HealthCheckBase):
    check_id: str
    checked_at: datetime


# Server Catalog Models
class ServerCatalogEntry(BaseModel):
    server_id: str
    name: str
    description: Optional[str] = None
    version: str
    status: MCPServerStatus
    health_status: HealthStatus
    tool_count: int
    capabilities: List[str]
    tags: List[str]
    discovery_source: DiscoverySource
    last_health_check: Optional[datetime] = None
    created_at: datetime


# Integration Models
class AgentMCPIntegrationBase(BaseModel):
    agent_id: str
    server_id: str
    enabled: bool = True
    tool_whitelist: List[str] = Field(default_factory=list)  # Empty = all tools
    tool_blacklist: List[str] = Field(default_factory=list)
    priority: int = Field(default=0, ge=0, le=100)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentMCPIntegrationCreate(AgentMCPIntegrationBase):
    pass


class AgentMCPIntegrationUpdate(BaseModel):
    enabled: Optional[bool] = None
    tool_whitelist: Optional[List[str]] = None
    tool_blacklist: Optional[List[str]] = None
    priority: Optional[int] = Field(None, ge=0, le=100)
    metadata: Optional[Dict[str, Any]] = None


class AgentMCPIntegration(AgentMCPIntegrationBase):
    integration_id: str
    usage_count: int = 0
    last_used: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


# Notification Models
class MCPNotificationBase(BaseModel):
    notification_type: str  # new_server, server_error, health_degraded, etc.
    severity: str = "info"  # info, warning, error, critical
    title: str
    message: str
    server_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MCPNotification(MCPNotificationBase):
    notification_id: str
    read: bool = False
    created_at: datetime
    read_at: Optional[datetime] = None


# Statistics Models
class MCPHubStats(BaseModel):
    total_servers: int
    active_servers: int
    inactive_servers: int
    discovering_servers: int
    error_servers: int
    total_tools: int
    total_integrations: int
    total_discovery_configs: int
    total_health_checks: int
    total_notifications: int
    unread_notifications: int
    servers_by_source: Dict[str, int]
    servers_by_status: Dict[str, int]
    servers_by_health: Dict[str, int]
    tools_by_type: Dict[str, int]
    recent_discoveries: List[Dict[str, Any]]
    top_used_servers: List[Dict[str, Any]]
    health_check_summary: Dict[str, Any]


# Discovery Report Models
class DiscoveryReport(BaseModel):
    report_id: str
    period_start: datetime
    period_end: datetime
    total_checks: int
    new_servers_found: int
    servers_updated: int
    servers_removed: int
    errors_encountered: int
    discovery_by_source: Dict[str, int]
    top_discoveries: List[Dict[str, Any]]
    generated_at: datetime

