"""
Security Management Router for GRETA PAI

This router provides API endpoints for security management including
password validation, IP filtering, request signing, and security auditing.

Endpoints:
    POST /security/validate-password - Validate password against policy
    POST /security/ip/whitelist - Add IP to whitelist
    POST /security/ip/blacklist - Add IP to blacklist
    GET /security/ip/check - Check if IP is allowed
    POST /security/sign-request - Sign request data
    POST /security/verify-signature - Verify request signature
    GET /security/events - Get security events
    GET /security/policy - Get security policy configuration

Example:
    Validate password:
        >>> import requests
        >>> response = requests.post(
        ...     "http://localhost:8000/api/v1/security/validate-password",
        ...     json={"password": "MyP@ssw0rd123", "username": "john"}
        ... )
        >>> print(response.json())

    Add IP to whitelist:
        >>> response = requests.post(
        ...     "http://localhost:8000/api/v1/security/ip/whitelist",
        ...     json={"ip": "192.168.1.100"}
        ... )
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

from models.base_models import StandardResponse
from services.security_service import SecurityService, PasswordPolicy, SecurityHeaders, CORSConfig
from utils.custom_exceptions import InvalidInputError
from database import get_database
from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/security",
    tags=["Security Management"],
    responses={
        404: {"description": "Security resource not found"},
        500: {"description": "Security operation failed"}
    }
)


# Request/Response Models
class PasswordValidationRequest(BaseModel):
    """Password validation request model"""
    password: str = Field(..., description="Password to validate")
    username: Optional[str] = Field(None, description="Username for validation")


class IPAddressRequest(BaseModel):
    """IP address request model"""
    ip: str = Field(..., description="IP address or network (CIDR notation)")


class RequestSigningRequest(BaseModel):
    """Request signing request model"""
    data: Dict[str, Any] = Field(..., description="Data to sign")


class SignatureVerificationRequest(BaseModel):
    """Signature verification request model"""
    data: Dict[str, Any] = Field(..., description="Request data")
    signature: str = Field(..., description="Signature to verify")


# Service instance (will be initialized on startup)
_security_service: Optional[SecurityService] = None


async def get_security_service(db: AsyncIOMotorDatabase = Depends(get_database)) -> SecurityService:
    """Get or create security service instance"""
    global _security_service
    
    if _security_service is None:
        _security_service = SecurityService(database=db)
        await _security_service.initialize()
    
    return _security_service


@router.post("/validate-password", response_model=StandardResponse)
async def validate_password(
    request: PasswordValidationRequest,
    security_service: SecurityService = Depends(get_security_service)
):
    """
    Validate password against security policy.
    
    Checks password against configured security policy including
    length, complexity, and common password requirements.
    
    Args:
        request: Password validation request
        security_service: Security service instance
        
    Returns:
        StandardResponse: Validation results
        
    Example:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/security/validate-password" \\
             -H "Content-Type: application/json" \\
             -d '{"password": "MyP@ssw0rd123", "username": "john"}'
        ```
        
        Response:
        ```json
        {
            "success": true,
            "message": "Password is valid",
            "data": {
                "is_valid": true,
                "errors": []
            }
        }
        ```
    """
    try:
        is_valid, errors = security_service.validate_password(
            request.password,
            request.username
        )
        
        return StandardResponse(
            success=is_valid,
            message="Password is valid" if is_valid else "Password validation failed",
            data={
                "is_valid": is_valid,
                "errors": errors
            }
        )
        
    except Exception as e:
        logger.error(f"Password validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password validation failed"
        )


@router.post("/ip/whitelist", response_model=StandardResponse)
async def add_to_whitelist(
    request: IPAddressRequest,
    security_service: SecurityService = Depends(get_security_service)
):
    """
    Add IP address or network to whitelist.
    
    Args:
        request: IP address request
        security_service: Security service instance
        
    Returns:
        StandardResponse: Operation result
        
    Example:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/security/ip/whitelist" \\
             -H "Content-Type: application/json" \\
             -d '{"ip": "192.168.1.100"}'
        ```
    """
    try:
        await security_service.add_to_whitelist(request.ip)
        
        return StandardResponse(
            success=True,
            message=f"IP {request.ip} added to whitelist",
            data={"ip": request.ip, "list": "whitelist"}
        )
        
    except InvalidInputError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to add IP to whitelist: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add IP to whitelist"
        )


@router.post("/ip/blacklist", response_model=StandardResponse)
async def add_to_blacklist(
    request: IPAddressRequest,
    security_service: SecurityService = Depends(get_security_service)
):
    """
    Add IP address or network to blacklist.
    
    Args:
        request: IP address request
        security_service: Security service instance
        
    Returns:
        StandardResponse: Operation result
        
    Example:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/security/ip/blacklist" \\
             -H "Content-Type: application/json" \\
             -d '{"ip": "10.0.0.50"}'
        ```
    """
    try:
        await security_service.add_to_blacklist(request.ip)
        
        return StandardResponse(
            success=True,
            message=f"IP {request.ip} added to blacklist",
            data={"ip": request.ip, "list": "blacklist"}
        )
        
    except InvalidInputError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to add IP to blacklist: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add IP to blacklist"
        )


@router.get("/ip/check", response_model=StandardResponse)
async def check_ip(
    ip: str = Query(..., description="IP address to check"),
    security_service: SecurityService = Depends(get_security_service)
):
    """
    Check if IP address is allowed.
    
    Args:
        ip: IP address to check
        security_service: Security service instance
        
    Returns:
        StandardResponse: Check result
        
    Example:
        ```bash
        curl "http://localhost:8000/api/v1/security/ip/check?ip=192.168.1.100"
        ```
    """
    try:
        is_allowed = await security_service.is_ip_allowed(ip)
        
        return StandardResponse(
            success=True,
            message=f"IP {ip} is {'allowed' if is_allowed else 'blocked'}",
            data={
                "ip": ip,
                "is_allowed": is_allowed
            }
        )
        
    except Exception as e:
        logger.error(f"IP check error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="IP check failed"
        )


@router.post("/sign-request", response_model=StandardResponse)
async def sign_request(
    request: RequestSigningRequest,
    security_service: SecurityService = Depends(get_security_service)
):
    """
    Sign request data for verification.
    
    Generates HMAC signature for request data to be used in
    X-Request-Signature header.
    
    Args:
        request: Request signing request
        security_service: Security service instance
        
    Returns:
        StandardResponse: Signature
        
    Example:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/security/sign-request" \\
             -H "Content-Type: application/json" \\
             -d '{"data": {"user_id": "123", "action": "delete"}}'
        ```
    """
    try:
        signature = security_service.sign_request(request.data)
        
        return StandardResponse(
            success=True,
            message="Request signed successfully",
            data={
                "signature": signature,
                "data": request.data
            }
        )
        
    except Exception as e:
        logger.error(f"Request signing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Request signing failed"
        )


@router.post("/verify-signature", response_model=StandardResponse)
async def verify_signature(
    request: SignatureVerificationRequest,
    security_service: SecurityService = Depends(get_security_service)
):
    """
    Verify request signature.
    
    Verifies HMAC signature for request data.
    
    Args:
        request: Signature verification request
        security_service: Security service instance
        
    Returns:
        StandardResponse: Verification result
        
    Example:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/security/verify-signature" \\
             -H "Content-Type: application/json" \\
             -d '{"data": {"user_id": "123"}, "signature": "abc123..."}'
        ```
    """
    try:
        is_valid = security_service.verify_request_signature(
            request.data,
            request.signature
        )
        
        return StandardResponse(
            success=is_valid,
            message="Signature is valid" if is_valid else "Signature is invalid",
            data={
                "is_valid": is_valid,
                "signature": request.signature
            }
        )
        
    except Exception as e:
        logger.error(f"Signature verification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Signature verification failed"
        )


@router.get("/events", response_model=StandardResponse)
async def get_security_events(
    limit: int = Query(100, description="Maximum number of events to return", ge=1, le=1000),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    security_service: SecurityService = Depends(get_security_service)
):
    """
    Get security events for audit trail.
    
    Returns recent security events including IP blocks, invalid signatures,
    rate limit violations, etc.
    
    Args:
        limit: Maximum number of events to return
        event_type: Filter by event type (optional)
        security_service: Security service instance
        
    Returns:
        StandardResponse: Security events
        
    Example:
        ```bash
        curl "http://localhost:8000/api/v1/security/events?limit=50&event_type=ip_blocked"
        ```
    """
    try:
        if not security_service.db:
            return StandardResponse(
                success=True,
                message="Security events not available (database not configured)",
                data={"events": []}
            )
        
        # Build query
        query = {}
        if event_type:
            query["event_type"] = event_type
        
        # Get events
        events_cursor = security_service.db.security_events.find(query).sort(
            "timestamp", -1
        ).limit(limit)
        
        events = []
        async for event in events_cursor:
            events.append({
                "event_type": event.get("event_type"),
                "details": event.get("details"),
                "timestamp": event.get("timestamp").isoformat() if event.get("timestamp") else None
            })
        
        return StandardResponse(
            success=True,
            message=f"Found {len(events)} security events",
            data={
                "events": events,
                "total_count": len(events)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get security events: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve security events"
        )


@router.get("/policy", response_model=StandardResponse)
async def get_security_policy(
    security_service: SecurityService = Depends(get_security_service)
):
    """
    Get security policy configuration.
    
    Returns current security policy including password requirements,
    security headers, and CORS configuration.
    
    Returns:
        StandardResponse: Security policy
        
    Example:
        ```bash
        curl "http://localhost:8000/api/v1/security/policy"
        ```
    """
    try:
        policy = security_service.password_policy
        headers = security_service.security_headers
        cors = security_service.cors_config
        
        return StandardResponse(
            success=True,
            message="Security policy retrieved successfully",
            data={
                "password_policy": {
                    "min_length": policy.min_length,
                    "max_length": policy.max_length,
                    "require_uppercase": policy.require_uppercase,
                    "require_lowercase": policy.require_lowercase,
                    "require_digits": policy.require_digits,
                    "require_special": policy.require_special,
                    "special_characters": policy.special_characters,
                    "min_unique_characters": policy.min_unique_characters
                },
                "security_headers": {
                    "hsts": headers.hsts,
                    "csp": headers.csp,
                    "x_frame_options": headers.x_frame_options,
                    "x_content_type_options": headers.x_content_type_options,
                    "x_xss_protection": headers.x_xss_protection,
                    "referrer_policy": headers.referrer_policy,
                    "permissions_policy": headers.permissions_policy
                },
                "cors_config": {
                    "allow_origins": cors.allow_origins,
                    "allow_methods": cors.allow_methods,
                    "allow_credentials": cors.allow_credentials,
                    "max_age": cors.max_age
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get security policy: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve security policy"
        )
