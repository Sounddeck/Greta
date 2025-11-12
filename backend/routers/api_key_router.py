"""
API Key Router for CPAS4
REST API endpoints for API key management
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from typing import List, Optional
import logging
from services.api_key_service import APIKeyService
from services.audit_service import AuditService
from models.api_key_models import *
from models.audit_models import EventType, SeverityLevel
from database import Database

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/api-keys", tags=["API Keys"])


async def get_service() -> APIKeyService:
    db = Database()
    await db.connect()
    return APIKeyService(db)


async def get_audit_service() -> AuditService:
    db = Database()
    await db.connect()
    return AuditService(db)


def get_client_ip(request: Request) -> str:
    """Extract client IP from request"""
    return request.client.host if request.client else "unknown"


# API Key Management Endpoints
@router.post("", response_model=APIKeyResponse)
async def create_api_key(
    create_request: APIKeyCreate,
    request: Request,
    svc: APIKeyService = Depends(get_service),
    audit_svc: AuditService = Depends(get_audit_service)
):
    """Create new API key
    
    WARNING: The full API key will only be shown once. Save it securely.
    
    Requires RBAC permission: user:update
    """
    try:
        response = await svc.create_api_key(
            user_id=create_request.user_id,
            name=create_request.name,
            scopes=create_request.scopes,
            expires_at=create_request.expires_at
        )
        
        # Log to audit
        await audit_svc.log_event(
            event_type=EventType.API_KEY_CREATED,
            user_id=create_request.user_id,
            severity=SeverityLevel.INFO,
            action="create_api_key",
            ip_address=get_client_ip(request),
            metadata={
                "key_id": response.key_id,
                "key_prefix": response.prefix,
                "scopes": [s.value for s in response.scopes]
            }
        )
        
        return response
    except Exception as e:
        logger.error(f"Failed to create API key: {e}")
        raise HTTPException(500, str(e))


@router.get("", response_model=List[APIKeyList])
async def list_api_keys(
    user_id: str,
    include_revoked: bool = False,
    svc: APIKeyService = Depends(get_service)
):
    """List user's API keys
    
    Requires RBAC permission: user:read
    """
    return await svc.list_api_keys(user_id, include_revoked=include_revoked)


@router.get("/{key_id}", response_model=APIKeyList)
async def get_api_key(
    key_id: str,
    svc: APIKeyService = Depends(get_service)
):
    """Get specific API key details
    
    Requires RBAC permission: user:read
    """
    key = await svc.get_api_key(key_id)
    if not key:
        raise HTTPException(404, "API key not found")
    
    # Determine status
    status = APIKeyStatus.ACTIVE
    if key.revoked_at:
        status = APIKeyStatus.REVOKED
    elif key.expires_at and datetime.utcnow() > key.expires_at:
        status = APIKeyStatus.EXPIRED
    
    return APIKeyList(
        key_id=key.key_id,
        user_id=key.user_id,
        name=key.name,
        prefix=key.prefix,
        scopes=key.scopes,
        enabled=key.enabled,
        created_at=key.created_at,
        last_used_at=key.last_used_at,
        expires_at=key.expires_at,
        revoked_at=key.revoked_at,
        status=status
    )


@router.put("/{key_id}", response_model=APIKeyList)
async def update_api_key(
    key_id: str,
    update: APIKeyUpdate,
    svc: APIKeyService = Depends(get_service)
):
    """Update API key
    
    Requires RBAC permission: user:update
    """
    key = await svc.update_api_key(key_id, update)
    if not key:
        raise HTTPException(404, "API key not found")
    
    # Determine status
    status = APIKeyStatus.ACTIVE
    if key.revoked_at:
        status = APIKeyStatus.REVOKED
    elif key.expires_at and datetime.utcnow() > key.expires_at:
        status = APIKeyStatus.EXPIRED
    
    return APIKeyList(
        key_id=key.key_id,
        user_id=key.user_id,
        name=key.name,
        prefix=key.prefix,
        scopes=key.scopes,
        enabled=key.enabled,
        created_at=key.created_at,
        last_used_at=key.last_used_at,
        expires_at=key.expires_at,
        revoked_at=key.revoked_at,
        status=status
    )


@router.post("/{key_id}/rotate", response_model=APIKeyRotateResponse)
async def rotate_api_key(
    key_id: str,
    revoked_by: Optional[str] = None,
    request: Request = None,
    svc: APIKeyService = Depends(get_service),
    audit_svc: AuditService = Depends(get_audit_service)
):
    """Rotate API key (create new, revoke old)
    
    WARNING: The old key will be revoked immediately. The new key will only be shown once.
    
    Requires RBAC permission: user:update
    """
    response = await svc.rotate_api_key(key_id, revoked_by=revoked_by)
    if not response:
        raise HTTPException(404, "API key not found")
    
    # Get old key for user_id
    old_key = await svc.get_api_key(key_id)
    
    # Log to audit
    if old_key:
        await audit_svc.log_event(
            event_type=EventType.API_KEY_CREATED,
            user_id=old_key.user_id,
            severity=SeverityLevel.INFO,
            action="rotate_api_key",
            ip_address=get_client_ip(request) if request else None,
            metadata={
                "old_key_id": response.old_key_id,
                "new_key_id": response.new_key_id,
                "new_key_prefix": response.prefix
            }
        )
    
    return response


@router.post("/{key_id}/revoke")
async def revoke_api_key(
    key_id: str,
    revoke_request: Optional[APIKeyRevoke] = None,
    request: Request = None,
    svc: APIKeyService = Depends(get_service),
    audit_svc: AuditService = Depends(get_audit_service)
):
    """Revoke API key
    
    Requires RBAC permission: user:update
    """
    # Get key for user_id
    key = await svc.get_api_key(key_id)
    if not key:
        raise HTTPException(404, "API key not found")
    
    reason = revoke_request.reason if revoke_request else None
    revoked_by = revoke_request.revoked_by if revoke_request else None
    
    success = await svc.revoke_api_key(key_id, reason=reason, revoked_by=revoked_by)
    
    if success:
        # Log to audit
        await audit_svc.log_event(
            event_type=EventType.API_KEY_REVOKED,
            user_id=key.user_id,
            severity=SeverityLevel.INFO,
            action="revoke_api_key",
            ip_address=get_client_ip(request) if request else None,
            metadata={
                "key_id": key_id,
                "key_prefix": key.prefix,
                "reason": reason
            }
        )
        
        return {"success": True, "message": "API key revoked"}
    else:
        raise HTTPException(500, "Failed to revoke API key")


@router.delete("/{key_id}")
async def delete_api_key(
    key_id: str,
    svc: APIKeyService = Depends(get_service)
):
    """Delete API key
    
    Requires RBAC permission: user:delete
    """
    success = await svc.delete_api_key(key_id)
    if not success:
        raise HTTPException(404, "API key not found")
    return {"success": True, "message": "API key deleted"}


@router.get("/{key_id}/usage", response_model=APIKeyUsageStats)
async def get_api_key_usage(
    key_id: str,
    svc: APIKeyService = Depends(get_service)
):
    """Get API key usage statistics
    
    Requires RBAC permission: user:read
    """
    return await svc.get_usage_stats(key_id)


# Health Check
@router.get("/health", include_in_schema=False)
async def health_check(svc: APIKeyService = Depends(get_service)):
    """API key module health check"""
    return {
        "status": "healthy",
        "module": "api_keys",
        "timestamp": datetime.utcnow().isoformat()
    }

