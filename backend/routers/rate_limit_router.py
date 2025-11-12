"""
Rate Limiting Router for CPAS4
REST API endpoints for rate limit management
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from typing import List, Optional
import logging
from services.rate_limit_service import RateLimitService
from models.rate_limit_models import *
from database import Database

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/rate-limit", tags=["Rate Limiting"])


async def get_service() -> RateLimitService:
    db = Database()
    await db.connect()
    return RateLimitService(db)


# Rule Management Endpoints
@router.post("/rules", response_model=RateLimitRule)
async def create_rate_limit_rule(
    rule: RateLimitRuleCreate,
    created_by: Optional[str] = None,
    svc: RateLimitService = Depends(get_service)
):
    """Create new rate limit rule
    
    Requires RBAC permission: system:manage
    """
    return await svc.create_rule(rule, created_by=created_by)


@router.get("/rules", response_model=List[RateLimitRule])
async def list_rate_limit_rules(
    enabled_only: bool = False,
    svc: RateLimitService = Depends(get_service)
):
    """List all rate limit rules
    
    Requires RBAC permission: system:view
    """
    return await svc.list_rules(enabled_only=enabled_only)


@router.get("/rules/{rule_id}", response_model=RateLimitRule)
async def get_rate_limit_rule(
    rule_id: str,
    svc: RateLimitService = Depends(get_service)
):
    """Get specific rate limit rule
    
    Requires RBAC permission: system:view
    """
    rule = await svc.get_rule(rule_id)
    if not rule:
        raise HTTPException(404, "Rate limit rule not found")
    return rule


@router.put("/rules/{rule_id}", response_model=RateLimitRule)
async def update_rate_limit_rule(
    rule_id: str,
    update: RateLimitRuleUpdate,
    svc: RateLimitService = Depends(get_service)
):
    """Update rate limit rule
    
    Requires RBAC permission: system:manage
    """
    rule = await svc.update_rule(rule_id, update)
    if not rule:
        raise HTTPException(404, "Rate limit rule not found")
    return rule


@router.delete("/rules/{rule_id}")
async def delete_rate_limit_rule(
    rule_id: str,
    svc: RateLimitService = Depends(get_service)
):
    """Delete rate limit rule
    
    Requires RBAC permission: system:manage
    """
    success = await svc.delete_rule(rule_id)
    if not success:
        raise HTTPException(404, "Rate limit rule not found")
    return {"success": True, "message": "Rate limit rule deleted"}


# Status Endpoints
@router.get("/status", response_model=List[RateLimitStatus])
async def get_rate_limit_status(
    endpoint: str,
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    request: Request = None,
    svc: RateLimitService = Depends(get_service)
):
    """Get current rate limit status
    
    Requires RBAC permission: user:read
    """
    # Use request IP if not provided
    if not ip_address and request and request.client:
        ip_address = request.client.host
    
    return await svc.get_rate_limit_status(
        endpoint=endpoint,
        user_id=user_id,
        ip_address=ip_address
    )


# Violation Endpoints
@router.get("/violations", response_model=List[RateLimitViolation])
async def get_rate_limit_violations(
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    limit: int = 100,
    svc: RateLimitService = Depends(get_service)
):
    """Get rate limit violations
    
    Requires RBAC permission: system:view
    """
    return await svc.get_violations(
        user_id=user_id,
        ip_address=ip_address,
        limit=limit
    )


# Statistics Endpoints
@router.get("/stats", response_model=RateLimitStats)
async def get_rate_limit_stats(
    svc: RateLimitService = Depends(get_service)
):
    """Get rate limit statistics
    
    Requires RBAC permission: system:view
    """
    return await svc.get_stats()


# Initialization Endpoints
@router.post("/initialize-default-rules", response_model=List[RateLimitRule])
async def initialize_default_rules(
    svc: RateLimitService = Depends(get_service)
):
    """Initialize default rate limit rules
    
    Requires RBAC permission: system:manage
    """
    return await svc.initialize_default_rules()


# Health Check
@router.get("/health")
async def health_check(svc: RateLimitService = Depends(get_service)):
    """Rate limiting module health check"""
    stats = await svc.get_stats()
    
    return {
        "status": "healthy",
        "module": "rate_limiting",
        "total_rules": stats.total_rules,
        "enabled_rules": stats.enabled_rules,
        "timestamp": datetime.utcnow().isoformat()
    }

