"""
RBAC (Role-Based Access Control) Router for CPAS4
REST API endpoints for role and permission management
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
import logging
from services.rbac_service import RBACService
from models.rbac_models import *
from database import Database

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/rbac", tags=["RBAC"])


async def get_service() -> RBACService:
    db = Database()
    await db.connect()
    return RBACService(db)


# Role Management Endpoints
@router.post("/roles", response_model=Role, status_code=201)
async def create_role(
    role: RoleCreate,
    created_by: Optional[str] = None,
    svc: RBACService = Depends(get_service)
):
    """Create new role"""
    return await svc.create_role(role, created_by)


@router.get("/roles/{role_id}", response_model=Role)
async def get_role(
    role_id: str,
    svc: RBACService = Depends(get_service)
):
    """Get role by ID"""
    role = await svc.get_role(role_id)
    if not role:
        raise HTTPException(404, "Role not found")
    return role


@router.patch("/roles/{role_id}", response_model=Role)
async def update_role(
    role_id: str,
    role_update: RoleUpdate,
    svc: RBACService = Depends(get_service)
):
    """Update role"""
    role = await svc.update_role(role_id, role_update)
    if not role:
        raise HTTPException(404, "Role not found")
    return role


@router.delete("/roles/{role_id}", status_code=204)
async def delete_role(
    role_id: str,
    svc: RBACService = Depends(get_service)
):
    """Delete role (soft delete)"""
    success = await svc.delete_role(role_id)
    if not success:
        raise HTTPException(404, "Role not found")
    return None


@router.get("/roles", response_model=List[Role])
async def list_roles(
    active_only: bool = Query(True),
    limit: int = Query(100, le=1000),
    svc: RBACService = Depends(get_service)
):
    """List all roles"""
    return await svc.list_roles(active_only, limit)


# User Role Assignment Endpoints
@router.post("/users/{user_id}/roles/{role_id}", response_model=UserRole, status_code=201)
async def assign_role_to_user(
    user_id: str,
    role_id: str,
    assigned_by: Optional[str] = None,
    expires_at: Optional[str] = None,
    svc: RBACService = Depends(get_service)
):
    """Assign role to user"""
    expires_datetime = None
    if expires_at:
        from datetime import datetime
        expires_datetime = datetime.fromisoformat(expires_at)
    
    try:
        return await svc.assign_role_to_user(user_id, role_id, assigned_by, expires_datetime)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.delete("/users/{user_id}/roles/{role_id}", status_code=204)
async def revoke_role_from_user(
    user_id: str,
    role_id: str,
    svc: RBACService = Depends(get_service)
):
    """Revoke role from user"""
    success = await svc.revoke_role_from_user(user_id, role_id)
    if not success:
        raise HTTPException(404, "Role assignment not found")
    return None


@router.get("/users/{user_id}/roles", response_model=List[Role])
async def get_user_roles(
    user_id: str,
    svc: RBACService = Depends(get_service)
):
    """Get all roles for user"""
    return await svc.get_user_roles(user_id)


@router.get("/users/{user_id}/permissions", response_model=UserPermissions)
async def get_user_permissions(
    user_id: str,
    svc: RBACService = Depends(get_service)
):
    """Get aggregated permissions for user"""
    return await svc.get_user_permissions(user_id)


# Permission Checking Endpoints
@router.post("/check-permission", response_model=PermissionCheckResult)
async def check_permission(
    check: PermissionCheck,
    svc: RBACService = Depends(get_service)
):
    """Check if user has permission"""
    return await svc.check_permission(check.user_id, check.resource, check.action)


@router.get("/users/{user_id}/check", response_model=PermissionCheckResult)
async def check_user_permission(
    user_id: str,
    resource: ResourceType,
    action: Action,
    svc: RBACService = Depends(get_service)
):
    """Check if user has specific permission"""
    return await svc.check_permission(user_id, resource, action)


# Built-in Roles Initialization
@router.post("/initialize-built-in-roles", response_model=List[Role])
async def initialize_built_in_roles(svc: RBACService = Depends(get_service)):
    """Initialize built-in roles (admin, user, viewer, developer, analyst)"""
    return await svc.initialize_built_in_roles()


# Role Statistics
@router.get("/roles/{role_id}/stats", response_model=RoleStats)
async def get_role_stats(
    role_id: str,
    svc: RBACService = Depends(get_service)
):
    """Get role usage statistics"""
    stats = await svc.get_role_stats(role_id)
    if not stats:
        raise HTTPException(404, "Role not found")
    return stats


# Health Check
@router.get("/health")
async def health_check(svc: RBACService = Depends(get_service)):
    """RBAC module health check"""
    roles = await svc.list_roles(active_only=True, limit=10)
    
    return {
        "status": "healthy",
        "module": "rbac",
        "active_roles": len(roles),
        "timestamp": datetime.utcnow().isoformat()
    }


# Module Statistics
@router.get("/stats")
async def get_module_stats(svc: RBACService = Depends(get_service)):
    """Get RBAC module statistics"""
    all_roles = await svc.list_roles(active_only=False, limit=1000)
    active_roles = [r for r in all_roles if r.is_active]
    
    # Count by role type
    by_type = {}
    for role in active_roles:
        type_name = role.role_type.value
        by_type[type_name] = by_type.get(type_name, 0) + 1
    
    return {
        "total_roles": len(all_roles),
        "active_roles": len(active_roles),
        "inactive_roles": len(all_roles) - len(active_roles),
        "by_type": by_type,
        "built_in_roles": len([r for r in active_roles if r.role_type != RoleType.CUSTOM])
    }

