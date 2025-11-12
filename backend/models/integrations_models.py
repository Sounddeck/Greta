"""
Third-Party Integrations Models for CPAS4
Pydantic models for managing third-party service integrations
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class IntegrationType(str, Enum):
    """Types of integrations"""
    COMMUNICATION = "communication"  # Slack, Discord, Telegram
    PAYMENT = "payment"  # Stripe, PayPal
    AUTOMATION = "automation"  # Zapier, Make
    BUSINESS = "business"  # QuickBooks, Salesforce, HubSpot
    PRODUCTIVITY = "productivity"  # Google Workspace, Microsoft 365
    DEVELOPMENT = "development"  # GitHub, GitLab, Jira
    ANALYTICS = "analytics"  # Google Analytics, Mixpanel
    STORAGE = "storage"  # Dropbox, Box, OneDrive
    OTHER = "other"


class IntegrationStatus(str, Enum):
    """Integration status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PENDING = "pending"
    DISABLED = "disabled"


class AuthType(str, Enum):
    """Authentication types"""
    OAUTH2 = "oauth2"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    CUSTOM = "custom"


# Base Integration Models
class IntegrationBase(BaseModel):
    """Base integration model"""
    name: str
    integration_type: IntegrationType
    description: str
    auth_type: AuthType
    mcp_server_id: Optional[str] = None  # Link to MCP Hub server
    enabled: bool = True
    config: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IntegrationCreate(IntegrationBase):
    """Create integration"""
    pass


class IntegrationUpdate(BaseModel):
    """Update integration"""
    name: Optional[str] = None
    description: Optional[str] = None
    enabled: Optional[bool] = None
    config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class Integration(IntegrationBase):
    """Integration instance"""
    integration_id: str
    status: IntegrationStatus
    mcp_server_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    last_sync: Optional[datetime] = None
    error_message: Optional[str] = None
    usage_count: int = 0


# Slack Integration Models
class SlackChannelBase(BaseModel):
    """Slack channel base"""
    channel_id: str
    name: str
    is_private: bool = False
    topic: Optional[str] = None
    purpose: Optional[str] = None
    member_count: int = 0


class SlackChannel(SlackChannelBase):
    """Slack channel"""
    integration_id: str
    created_at: datetime
    updated_at: datetime


class SlackMessageBase(BaseModel):
    """Slack message base"""
    channel_id: str
    user_id: str
    text: str
    thread_ts: Optional[str] = None
    attachments: List[Dict[str, Any]] = Field(default_factory=list)


class SlackMessageCreate(SlackMessageBase):
    """Create Slack message"""
    pass


class SlackMessage(SlackMessageBase):
    """Slack message"""
    message_id: str
    ts: str
    integration_id: str
    created_at: datetime


# Stripe Integration Models
class StripeCustomerBase(BaseModel):
    """Stripe customer base"""
    customer_id: str
    email: str
    name: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StripeCustomer(StripeCustomerBase):
    """Stripe customer"""
    integration_id: str
    created_at: datetime
    updated_at: datetime


class StripePaymentBase(BaseModel):
    """Stripe payment base"""
    amount: int  # in cents
    currency: str = "usd"
    customer_id: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StripePaymentCreate(StripePaymentBase):
    """Create Stripe payment"""
    pass


class StripePayment(StripePaymentBase):
    """Stripe payment"""
    payment_id: str
    status: str
    integration_id: str
    created_at: datetime


# Zapier Integration Models
class ZapierWebhookBase(BaseModel):
    """Zapier webhook base"""
    webhook_url: str
    event_type: str
    description: Optional[str] = None
    enabled: bool = True


class ZapierWebhookCreate(ZapierWebhookBase):
    """Create Zapier webhook"""
    pass


class ZapierWebhook(ZapierWebhookBase):
    """Zapier webhook"""
    webhook_id: str
    integration_id: str
    created_at: datetime
    updated_at: datetime
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


# Integration Stats
class IntegrationStats(BaseModel):
    """Integration statistics"""
    integration_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    last_request: Optional[datetime] = None
    average_response_time: float = 0.0
    uptime_percentage: float = 100.0


# Integration Health
class IntegrationHealth(BaseModel):
    """Integration health check"""
    integration_id: str
    status: IntegrationStatus
    is_healthy: bool
    last_check: datetime
    response_time_ms: float
    error_message: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


# Integration Event
class IntegrationEventBase(BaseModel):
    """Integration event base"""
    integration_id: str
    event_type: str
    event_data: Dict[str, Any] = Field(default_factory=dict)
    severity: str = "info"  # info, warning, error, critical


class IntegrationEvent(IntegrationEventBase):
    """Integration event"""
    event_id: str
    created_at: datetime
    processed: bool = False
    processed_at: Optional[datetime] = None


# Integration Credentials (encrypted)
class IntegrationCredentialsBase(BaseModel):
    """Integration credentials base"""
    integration_id: str
    auth_type: AuthType
    credentials: Dict[str, str] = Field(default_factory=dict)  # Encrypted
    expires_at: Optional[datetime] = None


class IntegrationCredentialsCreate(IntegrationCredentialsBase):
    """Create integration credentials"""
    pass


class IntegrationCredentials(IntegrationCredentialsBase):
    """Integration credentials"""
    credentials_id: str
    created_at: datetime
    updated_at: datetime
    last_refreshed: Optional[datetime] = None


# Integration Sync
class IntegrationSyncBase(BaseModel):
    """Integration sync base"""
    integration_id: str
    sync_type: str  # full, incremental, manual
    direction: str = "bidirectional"  # inbound, outbound, bidirectional


class IntegrationSyncCreate(IntegrationSyncBase):
    """Create integration sync"""
    pass


class IntegrationSync(IntegrationSyncBase):
    """Integration sync"""
    sync_id: str
    status: str  # pending, running, completed, failed
    started_at: datetime
    completed_at: Optional[datetime] = None
    records_synced: int = 0
    records_failed: int = 0
    error_message: Optional[str] = None


# Integration Mapping (for data transformation)
class IntegrationMappingBase(BaseModel):
    """Integration mapping base"""
    integration_id: str
    source_field: str
    target_field: str
    transformation: Optional[str] = None  # JSON transformation logic
    is_required: bool = False


class IntegrationMappingCreate(IntegrationMappingBase):
    """Create integration mapping"""
    pass


class IntegrationMapping(IntegrationMappingBase):
    """Integration mapping"""
    mapping_id: str
    created_at: datetime
    updated_at: datetime


# Integration Webhook
class IntegrationWebhookBase(BaseModel):
    """Integration webhook base"""
    integration_id: str
    webhook_url: str
    event_types: List[str] = Field(default_factory=list)
    secret: Optional[str] = None
    enabled: bool = True


class IntegrationWebhookCreate(IntegrationWebhookBase):
    """Create integration webhook"""
    pass


class IntegrationWebhook(IntegrationWebhookBase):
    """Integration webhook"""
    webhook_id: str
    created_at: datetime
    updated_at: datetime
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0

