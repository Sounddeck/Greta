"""
RBAC (Role-Based Access Control) Models for CPAS4
Pydantic models for roles, permissions, and access control
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# Permission Enums
class ResourceType(str, Enum):
    """Resource types in the system"""
    USER = "user"
    AGENT = "agent"
    TASK = "task"
    WORKFLOW = "workflow"
    INTEGRATION = "integration"
    ANALYTICS = "analytics"
    SYSTEM = "system"
    MCP = "mcp"
    AUDIT = "audit"
    ROLE = "role"


class Action(str, Enum):
    """Actions that can be performed on resources"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    MANAGE = "manage"
    VIEW = "view"
    ADMIN = "admin"


class RoleType(str, Enum):
    """Built-in role types"""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    DEVELOPER = "developer"
    ANALYST = "analyst"
    CUSTOM = "custom"


# Permission Models
class Permission(BaseModel):
    """Permission definition"""
    resource: ResourceType
    action: Action
    description: Optional[str] = None


class PermissionCheck(BaseModel):
    """Permission check request"""
    user_id: str
    resource: ResourceType
    action: Action


# Role Models
class RoleBase(BaseModel):
    """Role base model"""
    name: str
    role_type: RoleType
    description: Optional[str] = None
    permissions: List[Permission] = []
    is_active: bool = True
    metadata: Dict[str, Any] = {}


class RoleCreate(RoleBase):
    """Create role"""
    pass


class RoleUpdate(BaseModel):
    """Update role"""
    name: Optional[str] = None
    description: Optional[str] = None
    permissions: Optional[List[Permission]] = None
    is_active: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None


class Role(RoleBase):
    """Role with ID and timestamps"""
    role_id: str
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str] = None


# User Role Assignment Models
class UserRoleBase(BaseModel):
    """User role assignment base"""
    user_id: str
    role_id: str
    assigned_by: Optional[str] = None
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = {}


class UserRoleCreate(UserRoleBase):
    """Create user role assignment"""
    pass


class UserRole(UserRoleBase):
    """User role assignment with ID and timestamps"""
    assignment_id: str
    created_at: datetime
    is_active: bool = True


# Role Permission Models
class RolePermissions(BaseModel):
    """Role with expanded permissions"""
    role: Role
    permissions: List[Permission]
    permission_count: int


# User Permissions Models
class UserPermissions(BaseModel):
    """User's aggregated permissions from all roles"""
    user_id: str
    roles: List[Role]
    permissions: List[Permission]
    permission_count: int


# Permission Check Result
class PermissionCheckResult(BaseModel):
    """Result of permission check"""
    user_id: str
    resource: ResourceType
    action: Action
    allowed: bool
    reason: Optional[str] = None
    matched_roles: List[str] = []


# Built-in Role Definitions
BUILT_IN_ROLES = {
    RoleType.ADMIN: {
        "name": "Administrator",
        "description": "Full system access with all permissions",
        "permissions": [
            Permission(resource=resource, action=Action.ADMIN)
            for resource in ResourceType
        ]
    },
    RoleType.USER: {
        "name": "User",
        "description": "Standard user with read/write access to own resources",
        "permissions": [
            Permission(resource=ResourceType.AGENT, action=Action.CREATE),
            Permission(resource=ResourceType.AGENT, action=Action.READ),
            Permission(resource=ResourceType.AGENT, action=Action.UPDATE),
            Permission(resource=ResourceType.TASK, action=Action.CREATE),
            Permission(resource=ResourceType.TASK, action=Action.READ),
            Permission(resource=ResourceType.TASK, action=Action.UPDATE),
            Permission(resource=ResourceType.WORKFLOW, action=Action.CREATE),
            Permission(resource=ResourceType.WORKFLOW, action=Action.READ),
            Permission(resource=ResourceType.WORKFLOW, action=Action.EXECUTE),
            Permission(resource=ResourceType.INTEGRATION, action=Action.READ),
            Permission(resource=ResourceType.ANALYTICS, action=Action.VIEW),
            Permission(resource=ResourceType.MCP, action=Action.READ),
        ]
    },
    RoleType.VIEWER: {
        "name": "Viewer",
        "description": "Read-only access to most resources",
        "permissions": [
            Permission(resource=ResourceType.AGENT, action=Action.READ),
            Permission(resource=ResourceType.TASK, action=Action.READ),
            Permission(resource=ResourceType.WORKFLOW, action=Action.READ),
            Permission(resource=ResourceType.INTEGRATION, action=Action.READ),
            Permission(resource=ResourceType.ANALYTICS, action=Action.VIEW),
            Permission(resource=ResourceType.MCP, action=Action.READ),
        ]
    },
    RoleType.DEVELOPER: {
        "name": "Developer",
        "description": "Developer access with MCP and integration management",
        "permissions": [
            Permission(resource=ResourceType.AGENT, action=Action.MANAGE),
            Permission(resource=ResourceType.TASK, action=Action.MANAGE),
            Permission(resource=ResourceType.WORKFLOW, action=Action.MANAGE),
            Permission(resource=ResourceType.INTEGRATION, action=Action.MANAGE),
            Permission(resource=ResourceType.MCP, action=Action.MANAGE),
            Permission(resource=ResourceType.ANALYTICS, action=Action.VIEW),
        ]
    },
    RoleType.ANALYST: {
        "name": "Analyst",
        "description": "Analytics and reporting access",
        "permissions": [
            Permission(resource=ResourceType.AGENT, action=Action.READ),
            Permission(resource=ResourceType.TASK, action=Action.READ),
            Permission(resource=ResourceType.WORKFLOW, action=Action.READ),
            Permission(resource=ResourceType.INTEGRATION, action=Action.READ),
            Permission(resource=ResourceType.ANALYTICS, action=Action.MANAGE),
            Permission(resource=ResourceType.AUDIT, action=Action.READ),
        ]
    }
}


# Role Statistics
class RoleStats(BaseModel):
    """Role usage statistics"""
    role_id: str
    role_name: str
    user_count: int
    active_assignments: int
    total_assignments: int
    created_at: datetime

