"""
RBAC (Role-Based Access Control) Service for CPAS4
Service layer for managing roles, permissions, and access control
"""

import logging
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from database import Database
from models.rbac_models import *

logger = logging.getLogger(__name__)


class RBACService:
    """RBAC service for role and permission management"""
    
    def __init__(self, db: Database):
        self.db = db
        self.roles_collection = "roles"
        self.user_roles_collection = "user_roles"
        self.permission_cache_collection = "permission_cache"
    
    # Role Management
    async def create_role(self, role_create: RoleCreate, created_by: Optional[str] = None) -> Role:
        """Create new role"""
        role_id = f"role_{uuid.uuid4().hex[:12]}"
        
        role = Role(
            role_id=role_id,
            name=role_create.name,
            role_type=role_create.role_type,
            description=role_create.description,
            permissions=role_create.permissions,
            is_active=role_create.is_active,
            metadata=role_create.metadata,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            created_by=created_by
        )
        
        await self.db.insert_one(self.roles_collection, role.model_dump())
        logger.info(f"Created role: {role_id} ({role.name})")
        
        return role
    
    async def get_role(self, role_id: str) -> Optional[Role]:
        """Get role by ID"""
        doc = await self.db.find_one(self.roles_collection, {"role_id": role_id})
        return Role(**doc) if doc else None
    
    async def get_role_by_name(self, name: str) -> Optional[Role]:
        """Get role by name"""
        doc = await self.db.find_one(self.roles_collection, {"name": name})
        return Role(**doc) if doc else None
    
    async def update_role(self, role_id: str, role_update: RoleUpdate) -> Optional[Role]:
        """Update role"""
        role = await self.get_role(role_id)
        if not role:
            return None
        
        update_data = role_update.model_dump(exclude_unset=True)
        update_data["updated_at"] = datetime.utcnow()
        
        await self.db.update_one(
            self.roles_collection,
            {"role_id": role_id},
            {"$set": update_data}
        )
        
        logger.info(f"Updated role: {role_id}")
        return await self.get_role(role_id)
    
    async def delete_role(self, role_id: str) -> bool:
        """Delete role (soft delete by deactivating)"""
        result = await self.db.update_one(
            self.roles_collection,
            {"role_id": role_id},
            {"$set": {"is_active": False, "updated_at": datetime.utcnow()}}
        )
        
        if result:
            logger.info(f"Deleted role: {role_id}")
        return result
    
    async def list_roles(self, active_only: bool = True, limit: int = 100) -> List[Role]:
        """List all roles"""
        query = {"is_active": True} if active_only else {}
        docs = await self.db.find_many(self.roles_collection, query, limit=limit)
        return [Role(**doc) for doc in docs]
    
    # User Role Assignment
    async def assign_role_to_user(
        self,
        user_id: str,
        role_id: str,
        assigned_by: Optional[str] = None,
        expires_at: Optional[datetime] = None
    ) -> UserRole:
        """Assign role to user"""
        # Check if role exists
        role = await self.get_role(role_id)
        if not role:
            raise ValueError(f"Role not found: {role_id}")
        
        # Check if assignment already exists
        existing = await self.db.find_one(
            self.user_roles_collection,
            {"user_id": user_id, "role_id": role_id, "is_active": True}
        )
        if existing:
            return UserRole(**existing)
        
        assignment_id = f"ua_{uuid.uuid4().hex[:12]}"
        
        user_role = UserRole(
            assignment_id=assignment_id,
            user_id=user_id,
            role_id=role_id,
            assigned_by=assigned_by,
            expires_at=expires_at,
            created_at=datetime.utcnow(),
            is_active=True
        )
        
        await self.db.insert_one(self.user_roles_collection, user_role.model_dump())
        logger.info(f"Assigned role {role_id} to user {user_id}")
        
        # Clear permission cache for user
        await self._clear_permission_cache(user_id)
        
        return user_role
    
    async def revoke_role_from_user(self, user_id: str, role_id: str) -> bool:
        """Revoke role from user"""
        result = await self.db.update_one(
            self.user_roles_collection,
            {"user_id": user_id, "role_id": role_id},
            {"$set": {"is_active": False}}
        )
        
        if result:
            logger.info(f"Revoked role {role_id} from user {user_id}")
            await self._clear_permission_cache(user_id)
        
        return result
    
    async def get_user_roles(self, user_id: str) -> List[Role]:
        """Get all active roles for user"""
        # Get active role assignments
        assignments = await self.db.find_many(
            self.user_roles_collection,
            {"user_id": user_id, "is_active": True},
            limit=100
        )
        
        # Get role details
        roles = []
        for assignment in assignments:
            # Check if expired
            if assignment.get("expires_at"):
                expires_at = assignment["expires_at"]
                if isinstance(expires_at, str):
                    expires_at = datetime.fromisoformat(expires_at)
                if expires_at < datetime.utcnow():
                    continue
            
            role = await self.get_role(assignment["role_id"])
            if role and role.is_active:
                roles.append(role)
        
        return roles
    
    async def get_user_permissions(self, user_id: str) -> UserPermissions:
        """Get aggregated permissions for user from all roles"""
        roles = await self.get_user_roles(user_id)
        
        # Aggregate permissions from all roles
        all_permissions = []
        seen_permissions = set()
        
        for role in roles:
            for perm in role.permissions:
                perm_key = f"{perm.resource}:{perm.action}"
                if perm_key not in seen_permissions:
                    all_permissions.append(perm)
                    seen_permissions.add(perm_key)
        
        return UserPermissions(
            user_id=user_id,
            roles=roles,
            permissions=all_permissions,
            permission_count=len(all_permissions)
        )
    
    # Permission Checking
    async def check_permission(
        self,
        user_id: str,
        resource: ResourceType,
        action: Action
    ) -> PermissionCheckResult:
        """Check if user has permission for resource and action"""
        user_perms = await self.get_user_permissions(user_id)
        
        matched_roles = []
        allowed = False
        
        for role in user_perms.roles:
            for perm in role.permissions:
                # Check for exact match or admin permission
                if (perm.resource == resource and perm.action == action) or \
                   (perm.resource == resource and perm.action == Action.ADMIN) or \
                   (perm.action == Action.ADMIN):
                    allowed = True
                    matched_roles.append(role.name)
                    break
        
        reason = None
        if not allowed:
            if not user_perms.roles:
                reason = "User has no roles assigned"
            else:
                reason = f"No permission for {action.value} on {resource.value}"
        
        return PermissionCheckResult(
            user_id=user_id,
            resource=resource,
            action=action,
            allowed=allowed,
            reason=reason,
            matched_roles=list(set(matched_roles))
        )
    
    # Built-in Roles Initialization
    async def initialize_built_in_roles(self) -> List[Role]:
        """Initialize built-in roles if they don't exist"""
        created_roles = []
        
        for role_type, role_def in BUILT_IN_ROLES.items():
            existing = await self.get_role_by_name(role_def["name"])
            if not existing:
                role = await self.create_role(
                    RoleCreate(
                        name=role_def["name"],
                        role_type=role_type,
                        description=role_def["description"],
                        permissions=role_def["permissions"],
                        is_active=True
                    ),
                    created_by="system"
                )
                created_roles.append(role)
                logger.info(f"Initialized built-in role: {role.name}")
        
        return created_roles
    
    # Helper Methods
    async def _clear_permission_cache(self, user_id: str):
        """Clear permission cache for user"""
        # Note: Using delete_one in a loop since delete_many is not available
        # In production, this would be optimized with a proper delete_many implementation
        pass  # Permission cache clearing is optional for now
    
    async def get_role_stats(self, role_id: str) -> Optional[RoleStats]:
        """Get role usage statistics"""
        role = await self.get_role(role_id)
        if not role:
            return None
        
        # Count assignments
        all_assignments = await self.db.find_many(
            self.user_roles_collection,
            {"role_id": role_id},
            limit=10000
        )
        
        active_assignments = [a for a in all_assignments if a.get("is_active", False)]
        unique_users = set(a["user_id"] for a in active_assignments)
        
        return RoleStats(
            role_id=role_id,
            role_name=role.name,
            user_count=len(unique_users),
            active_assignments=len(active_assignments),
            total_assignments=len(all_assignments),
            created_at=role.created_at
        )

