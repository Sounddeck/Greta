"""
MFA (Multi-Factor Authentication) Router for CPAS4
REST API endpoints for TOTP, backup codes, and MFA management
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from typing import List
import logging
from services.mfa_service import MFAService
from services.audit_service import AuditService
from models.mfa_models import *
from models.audit_models import EventType, SeverityLevel
from database import Database

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/mfa", tags=["MFA"])


async def get_service() -> MFAService:
    db = Database()
    await db.connect()
    return MFAService(db)


async def get_audit_service() -> AuditService:
    db = Database()
    await db.connect()
    return AuditService(db)


def get_client_ip(request: Request) -> str:
    """Extract client IP from request"""
    return request.client.host if request.client else "unknown"


# TOTP Endpoints
@router.post("/totp/enroll", response_model=TOTPEnrollmentResponse)
async def enroll_totp(
    user_id: str,
    request: Request,
    svc: MFAService = Depends(get_service),
    audit_svc: AuditService = Depends(get_audit_service)
):
    """Start TOTP enrollment
    
    Returns secret key and QR code for authenticator app setup.
    Save the backup codes in a secure location.
    
    Requires RBAC permission: user:update
    """
    try:
        response = await svc.enroll_totp(user_id)
        
        # Log to audit
        await audit_svc.log_event(
            event_type=EventType.MFA_ENABLED,
            user_id=user_id,
            severity=SeverityLevel.INFO,
            action="enroll_totp",
            ip_address=get_client_ip(request),
            metadata={"method": "totp", "enrollment_id": response.enrollment_id}
        )
        
        return response
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/totp/verify-enrollment", response_model=TOTPConfiguration)
async def verify_totp_enrollment(
    verification: TOTPVerificationRequest,
    request: Request,
    svc: MFAService = Depends(get_service),
    audit_svc: AuditService = Depends(get_audit_service)
):
    """Verify TOTP code to complete enrollment
    
    Enter the 6-digit code from your authenticator app.
    
    Requires RBAC permission: user:update
    """
    try:
        config = await svc.verify_and_enable_totp(
            user_id=verification.user_id,
            code=verification.code,
            ip_address=get_client_ip(request)
        )
        
        # Log to audit
        await audit_svc.log_event(
            event_type=EventType.MFA_VERIFIED,
            user_id=verification.user_id,
            severity=SeverityLevel.INFO,
            action="verify_enrollment",
            ip_address=get_client_ip(request),
            metadata={"method": "totp"}
        )
        
        return config
    except ValueError as e:
        # Log failed verification
        await audit_svc.log_event(
            event_type=EventType.MFA_FAILED,
            user_id=verification.user_id,
            severity=SeverityLevel.WARNING,
            action="verify_enrollment",
            ip_address=get_client_ip(request),
            error_message=str(e),
            metadata={"method": "totp"}
        )
        raise HTTPException(400, str(e))


@router.post("/totp/verify")
async def verify_totp_login(
    verification: TOTPVerificationRequest,
    request: Request,
    svc: MFAService = Depends(get_service),
    audit_svc: AuditService = Depends(get_audit_service)
):
    """Verify TOTP code during login
    
    Enter the 6-digit code from your authenticator app.
    
    Requires RBAC permission: user:read
    """
    try:
        is_valid = await svc.verify_totp_login(
            user_id=verification.user_id,
            code=verification.code,
            ip_address=get_client_ip(request)
        )
        
        if is_valid:
            # Log successful verification
            await audit_svc.log_event(
                event_type=EventType.MFA_VERIFIED,
                user_id=verification.user_id,
                severity=SeverityLevel.INFO,
                action="verify_login",
                ip_address=get_client_ip(request),
                metadata={"method": "totp"}
            )
            
            return {"success": True, "message": "TOTP verification successful"}
        else:
            # Log failed verification
            await audit_svc.log_event(
                event_type=EventType.MFA_FAILED,
                user_id=verification.user_id,
                severity=SeverityLevel.WARNING,
                action="verify_login",
                ip_address=get_client_ip(request),
                error_message="Invalid TOTP code",
                metadata={"method": "totp"}
            )
            
            raise HTTPException(401, "Invalid TOTP code")
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/totp/disable")
async def disable_totp(
    user_id: str,
    request: Request,
    svc: MFAService = Depends(get_service),
    audit_svc: AuditService = Depends(get_audit_service)
):
    """Disable TOTP for user
    
    Requires RBAC permission: user:update
    """
    success = await svc.disable_totp(user_id)
    
    if success:
        # Log to audit
        await audit_svc.log_event(
            event_type=EventType.MFA_DISABLED,
            user_id=user_id,
            severity=SeverityLevel.INFO,
            action="disable_totp",
            ip_address=get_client_ip(request),
            metadata={"method": "totp"}
        )
        
        return {"success": True, "message": "TOTP disabled successfully"}
    else:
        raise HTTPException(404, "TOTP not found for user")


# Backup Codes Endpoints
@router.post("/backup-codes/generate", response_model=BackupCodesResponse)
async def generate_backup_codes(
    user_id: str,
    request: Request,
    svc: MFAService = Depends(get_service),
    audit_svc: AuditService = Depends(get_audit_service)
):
    """Generate new backup codes
    
    WARNING: This will invalidate all existing unused backup codes.
    Save these codes in a secure location. They will not be shown again.
    
    Requires RBAC permission: user:update
    """
    response = await svc.generate_backup_codes(user_id)
    
    # Log to audit
    await audit_svc.log_event(
        event_type=EventType.CUSTOM,
        user_id=user_id,
        severity=SeverityLevel.INFO,
        action="generate_backup_codes",
        ip_address=get_client_ip(request),
        metadata={"method": "backup_codes", "count": response.count}
    )
    
    return response


@router.get("/backup-codes", response_model=List[BackupCodeDisplay])
async def get_backup_codes(
    user_id: str,
    svc: MFAService = Depends(get_service)
):
    """Get list of backup codes (masked)
    
    Codes are masked for security. Only the last 4 characters are shown.
    
    Requires RBAC permission: user:read
    """
    return await svc.get_backup_codes(user_id)


@router.post("/backup-codes/verify")
async def verify_backup_code(
    verification: BackupCodeVerificationRequest,
    request: Request,
    svc: MFAService = Depends(get_service),
    audit_svc: AuditService = Depends(get_audit_service)
):
    """Verify backup code
    
    Backup codes can only be used once.
    
    Requires RBAC permission: user:read
    """
    is_valid = await svc.verify_backup_code(
        user_id=verification.user_id,
        code=verification.code,
        ip_address=get_client_ip(request)
    )
    
    if is_valid:
        # Log successful verification
        await audit_svc.log_event(
            event_type=EventType.MFA_VERIFIED,
            user_id=verification.user_id,
            severity=SeverityLevel.INFO,
            action="verify_backup_code",
            ip_address=get_client_ip(request),
            metadata={"method": "backup_code"}
        )
        
        return {"success": True, "message": "Backup code verified successfully"}
    else:
        # Log failed verification
        await audit_svc.log_event(
            event_type=EventType.MFA_FAILED,
            user_id=verification.user_id,
            severity=SeverityLevel.WARNING,
            action="verify_backup_code",
            ip_address=get_client_ip(request),
            error_message="Invalid or already used backup code",
            metadata={"method": "backup_code"}
        )
        
        raise HTTPException(401, "Invalid or already used backup code")


# MFA Status Endpoints
@router.get("/status", response_model=MFAStatusResponse)
async def get_mfa_status(
    user_id: str,
    svc: MFAService = Depends(get_service)
):
    """Get MFA status for user
    
    Requires RBAC permission: user:read
    """
    return await svc.get_mfa_status(user_id)


@router.get("/rate-limit", response_model=MFARateLimitInfo)
async def get_rate_limit_info(
    user_id: str,
    svc: MFAService = Depends(get_service)
):
    """Get rate limit information for user
    
    Requires RBAC permission: user:read
    """
    return await svc.get_rate_limit_info(user_id)


# Health Check
@router.get("/health")
async def health_check(svc: MFAService = Depends(get_service)):
    """MFA module health check"""
    return {
        "status": "healthy",
        "module": "mfa",
        "supported_methods": ["totp", "backup_codes"],
        "timestamp": datetime.utcnow().isoformat()
    }

