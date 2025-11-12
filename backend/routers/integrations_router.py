"""
Third-Party Integrations Router for CPAS4
REST API endpoints for managing third-party integrations
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
import logging
from services.integrations_service import IntegrationsService
from models.integrations_models import *
from database import Database

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/integrations", tags=["Integrations"])


async def get_service() -> IntegrationsService:
    db = Database()
    await db.connect()
    return IntegrationsService(db)


# Integration Management Endpoints
@router.post("", response_model=Integration, status_code=201)
async def create_integration(
    integration: IntegrationCreate,
    svc: IntegrationsService = Depends(get_service)
):
    """Create new integration"""
    return await svc.create_integration(integration)


@router.get("/{integration_id}", response_model=Integration)
async def get_integration(
    integration_id: str,
    svc: IntegrationsService = Depends(get_service)
):
    """Get integration by ID"""
    integration = await svc.get_integration(integration_id)
    if not integration:
        raise HTTPException(404, "Integration not found")
    return integration


@router.patch("/{integration_id}", response_model=Integration)
async def update_integration(
    integration_id: str,
    integration_update: IntegrationUpdate,
    svc: IntegrationsService = Depends(get_service)
):
    """Update integration"""
    integration = await svc.update_integration(integration_id, integration_update)
    if not integration:
        raise HTTPException(404, "Integration not found")
    return integration


@router.delete("/{integration_id}", status_code=204)
async def delete_integration(
    integration_id: str,
    svc: IntegrationsService = Depends(get_service)
):
    """Delete integration"""
    success = await svc.delete_integration(integration_id)
    if not success:
        raise HTTPException(404, "Integration not found")


@router.get("", response_model=List[Integration])
async def list_integrations(
    integration_type: Optional[IntegrationType] = None,
    status: Optional[IntegrationStatus] = None,
    enabled: Optional[bool] = None,
    limit: int = Query(100, le=1000),
    svc: IntegrationsService = Depends(get_service)
):
    """List integrations with filters"""
    return await svc.list_integrations(integration_type, status, enabled, limit)


# Slack Integration Endpoints
@router.post("/slack/register", response_model=Integration, status_code=201)
async def register_slack_integration(
    name: str,
    xoxc_token: Optional[str] = None,
    xoxd_token: Optional[str] = None,
    xoxp_token: Optional[str] = None,
    workspace_url: str = "https://slack.com",
    svc: IntegrationsService = Depends(get_service)
):
    """Register Slack integration via MCP Hub"""
    if not xoxp_token and not (xoxc_token and xoxd_token):
        raise HTTPException(400, "Either xoxp_token or both xoxc_token and xoxd_token are required")
    
    return await svc.register_slack_integration(
        name=name,
        xoxc_token=xoxc_token,
        xoxd_token=xoxd_token,
        xoxp_token=xoxp_token,
        workspace_url=workspace_url
    )


@router.post("/slack/{integration_id}/send-message", response_model=SlackMessage)
async def send_slack_message(
    integration_id: str,
    message: SlackMessageCreate,
    svc: IntegrationsService = Depends(get_service)
):
    """Send message to Slack channel"""
    return await svc.send_slack_message(
        integration_id=integration_id,
        channel_id=message.channel_id,
        text=message.text,
        thread_ts=message.thread_ts
    )


@router.get("/slack/{integration_id}/channels", response_model=List[SlackChannel])
async def get_slack_channels(
    integration_id: str,
    svc: IntegrationsService = Depends(get_service)
):
    """Get Slack channels"""
    return await svc.get_slack_channels(integration_id)


# Stripe Integration Endpoints
@router.post("/stripe/register", response_model=Integration, status_code=201)
async def register_stripe_integration(
    name: str,
    api_key: str,
    webhook_secret: Optional[str] = None,
    svc: IntegrationsService = Depends(get_service)
):
    """Register Stripe integration via MCP Hub"""
    return await svc.register_stripe_integration(
        name=name,
        api_key=api_key,
        webhook_secret=webhook_secret
    )


@router.post("/stripe/{integration_id}/customers", response_model=StripeCustomer, status_code=201)
async def create_stripe_customer(
    integration_id: str,
    email: str,
    name: Optional[str] = None,
    metadata: Optional[dict] = None,
    svc: IntegrationsService = Depends(get_service)
):
    """Create Stripe customer"""
    return await svc.create_stripe_customer(
        integration_id=integration_id,
        email=email,
        name=name,
        metadata=metadata
    )


@router.get("/stripe/{integration_id}/customers", response_model=List[StripeCustomer])
async def list_stripe_customers(
    integration_id: str,
    limit: int = Query(100, le=1000),
    svc: IntegrationsService = Depends(get_service)
):
    """List Stripe customers"""
    return await svc.list_stripe_customers(integration_id, limit)


@router.post("/stripe/{integration_id}/payments", response_model=StripePayment, status_code=201)
async def create_stripe_payment(
    integration_id: str,
    customer_id: str,
    amount: float,
    currency: str = "usd",
    description: Optional[str] = None,
    svc: IntegrationsService = Depends(get_service)
):
    """Create Stripe payment"""
    return await svc.create_stripe_payment(
        integration_id=integration_id,
        customer_id=customer_id,
        amount=amount,
        currency=currency,
        description=description
    )


@router.get("/stripe/{integration_id}/payments", response_model=List[StripePayment])
async def list_stripe_payments(
    integration_id: str,
    customer_id: Optional[str] = None,
    limit: int = Query(100, le=1000),
    svc: IntegrationsService = Depends(get_service)
):
    """List Stripe payments"""
    return await svc.list_stripe_payments(integration_id, customer_id, limit)


# Zapier Integration Endpoints
@router.post("/zapier/register", response_model=Integration, status_code=201)
async def register_zapier_integration(
    name: str,
    webhook_url: str,
    svc: IntegrationsService = Depends(get_service)
):
    """Register Zapier integration"""
    return await svc.register_zapier_integration(
        name=name,
        webhook_url=webhook_url
    )


# Integration Stats & Health Endpoints
@router.get("/{integration_id}/stats", response_model=IntegrationStats)
async def get_integration_stats(
    integration_id: str,
    svc: IntegrationsService = Depends(get_service)
):
    """Get integration statistics"""
    stats = await svc.get_integration_stats(integration_id)
    if not stats:
        raise HTTPException(404, "Integration stats not found")
    return stats


@router.get("/{integration_id}/health", response_model=IntegrationHealth)
async def check_integration_health(
    integration_id: str,
    svc: IntegrationsService = Depends(get_service)
):
    """Check integration health"""
    return await svc.health_check(integration_id)


# Module Health & Stats
@router.get("/health", response_model=dict)
async def get_module_health(svc: IntegrationsService = Depends(get_service)):
    """Get integrations module health"""
    integrations = await svc.list_integrations(limit=1000)
    active_count = sum(1 for i in integrations if i.status == IntegrationStatus.ACTIVE)
    error_count = sum(1 for i in integrations if i.status == IntegrationStatus.ERROR)
    
    return {
        "status": "healthy",
        "total_integrations": len(integrations),
        "active_integrations": active_count,
        "error_integrations": error_count,
        "health_percentage": (active_count / len(integrations) * 100) if integrations else 100
    }


@router.get("/stats", response_model=dict)
async def get_module_stats(svc: IntegrationsService = Depends(get_service)):
    """Get integrations module statistics"""
    integrations = await svc.list_integrations(limit=1000)
    
    # Count by type
    by_type = {}
    for integration in integrations:
        type_name = integration.integration_type.value
        by_type[type_name] = by_type.get(type_name, 0) + 1
    
    # Count by status
    by_status = {}
    for integration in integrations:
        status_name = integration.status.value
        by_status[status_name] = by_status.get(status_name, 0) + 1
    
    return {
        "total_integrations": len(integrations),
        "by_type": by_type,
        "by_status": by_status,
        "enabled_count": sum(1 for i in integrations if i.enabled),
        "disabled_count": sum(1 for i in integrations if not i.enabled)
    }


# Zapier Integration Endpoints
@router.post("/zapier/register", response_model=Integration, status_code=201)
async def register_zapier_integration(
    name: str,
    webhook_url: str,
    mcp_server_url: Optional[str] = None,
    svc: IntegrationsService = Depends(get_service)
):
    """Register Zapier integration (webhook-based or MCP SSE)

    Zapier MCP is cloud-based and uses SSE (Server-Sent Events) endpoints.
    Users create MCP servers via Zapier's web interface.

    Args:
        name: Integration name
        webhook_url: Zapier webhook URL for triggering workflows
        mcp_server_url: Optional Zapier MCP SSE server URL
    """
    return await svc.register_zapier_integration(
        name=name,
        webhook_url=webhook_url,
        mcp_server_url=mcp_server_url
    )


@router.post("/zapier/{integration_id}/trigger", response_model=ZapierWebhook, status_code=201)
async def trigger_zapier_webhook(
    integration_id: str,
    data: dict,
    svc: IntegrationsService = Depends(get_service)
):
    """Trigger Zapier webhook with data

    Args:
        integration_id: Integration ID
        data: Data to send to Zapier webhook
    """
    return await svc.trigger_zapier_webhook(
        integration_id=integration_id,
        data=data
    )


@router.get("/zapier/{integration_id}/webhooks", response_model=List[ZapierWebhook])
async def list_zapier_webhooks(
    integration_id: str,
    limit: int = Query(100, le=1000),
    svc: IntegrationsService = Depends(get_service)
):
    """List Zapier webhooks for integration"""
    return await svc.list_zapier_webhooks(integration_id, limit)

