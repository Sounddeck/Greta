"""
Event Models - Type definitions for all system events
Provides type safety and documentation for event-driven communication
"""

from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class EventType(str, Enum):
    """
    All event types in the CPAS5 system
    
    Naming convention: <module>.<entity>.<action>
    Example: agent.created, workflow.completed, lead.scored
    """
    
    # ==================== Agent Events ====================
    AGENT_CREATED = "agent.created"
    AGENT_UPDATED = "agent.updated"
    AGENT_DELETED = "agent.deleted"
    AGENT_EXECUTED = "agent.executed"
    AGENT_EXECUTION_STARTED = "agent.execution.started"
    AGENT_EXECUTION_COMPLETED = "agent.execution.completed"
    AGENT_EXECUTION_FAILED = "agent.execution.failed"
    AGENT_DEPLOYED = "agent.deployed"
    AGENT_ARCHIVED = "agent.archived"
    
    # ==================== Workflow Events ====================
    WORKFLOW_CREATED = "workflow.created"
    WORKFLOW_UPDATED = "workflow.updated"
    WORKFLOW_DELETED = "workflow.deleted"
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_STEP_COMPLETED = "workflow.step.completed"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"
    WORKFLOW_PAUSED = "workflow.paused"
    WORKFLOW_RESUMED = "workflow.resumed"
    
    # ==================== Team Events ====================
    TEAM_CREATED = "team.created"
    TEAM_UPDATED = "team.updated"
    TEAM_DELETED = "team.deleted"
    TEAM_MEMBER_ADDED = "team.member.added"
    TEAM_MEMBER_REMOVED = "team.member.removed"
    
    # ==================== Lead Generation Events ====================
    LEAD_GENERATED = "lead.generated"
    LEAD_SCORED = "lead.scored"
    LEAD_QUALIFIED = "lead.qualified"
    LEAD_DISQUALIFIED = "lead.disqualified"
    LEAD_ENRICHED = "lead.enriched"
    LEAD_BATCH_COMPLETED = "lead.batch.completed"
    
    # ==================== CRM Events ====================
    CRM_CONTACT_CREATED = "crm.contact.created"
    CRM_CONTACT_UPDATED = "crm.contact.updated"
    CRM_CONTACT_DELETED = "crm.contact.deleted"
    CRM_DEAL_CREATED = "crm.deal.created"
    CRM_DEAL_UPDATED = "crm.deal.updated"
    CRM_DEAL_WON = "crm.deal.won"
    CRM_DEAL_LOST = "crm.deal.lost"
    CRM_ACTIVITY_LOGGED = "crm.activity.logged"
    
    # ==================== Knowledge Base Events ====================
    KNOWLEDGE_DOCUMENT_ADDED = "knowledge.document.added"
    KNOWLEDGE_DOCUMENT_UPDATED = "knowledge.document.updated"
    KNOWLEDGE_DOCUMENT_DELETED = "knowledge.document.deleted"
    KNOWLEDGE_IMAGE_GENERATED = "knowledge.image.generated"
    KNOWLEDGE_EMBEDDING_CREATED = "knowledge.embedding.created"
    KNOWLEDGE_SEARCH_PERFORMED = "knowledge.search.performed"
    
    # ==================== Voice Events ====================
    VOICE_PROFILE_CREATED = "voice.profile.created"
    VOICE_PROFILE_UPDATED = "voice.profile.updated"
    VOICE_PROFILE_DELETED = "voice.profile.deleted"
    VOICE_STT_COMPLETED = "voice.stt.completed"
    VOICE_TTS_COMPLETED = "voice.tts.completed"
    VOICE_WAKE_WORD_DETECTED = "voice.wake_word.detected"
    VOICE_WAKE_WORD_STARTED = "voice.wake_word.started"
    VOICE_WAKE_WORD_STOPPED = "voice.wake_word.stopped"
    
    # ==================== Finance Events ====================
    FINANCE_TRANSACTION_CREATED = "finance.transaction.created"
    FINANCE_INVOICE_GENERATED = "finance.invoice.generated"
    FINANCE_PAYMENT_RECEIVED = "finance.payment.received"
    FINANCE_REPORT_GENERATED = "finance.report.generated"
    
    # ==================== ATS Events ====================
    ATS_CANDIDATE_ADDED = "ats.candidate.added"
    ATS_CANDIDATE_UPDATED = "ats.candidate.updated"
    ATS_CANDIDATE_MOVED = "ats.candidate.moved"
    ATS_INTERVIEW_SCHEDULED = "ats.interview.scheduled"
    ATS_OFFER_SENT = "ats.offer.sent"
    
    # ==================== Google Workspace Events ====================
    GOOGLE_EMAIL_SENT = "google.email.sent"
    GOOGLE_EMAIL_RECEIVED = "google.email.received"
    GOOGLE_CALENDAR_EVENT_CREATED = "google.calendar.event.created"
    GOOGLE_DRIVE_FILE_UPLOADED = "google.drive.file.uploaded"
    
    # ==================== MCP Events ====================
    MCP_TOOL_REGISTERED = "mcp.tool.registered"
    MCP_TOOL_EXECUTED = "mcp.tool.executed"
    MCP_SERVER_CONNECTED = "mcp.server.connected"
    MCP_SERVER_DISCONNECTED = "mcp.server.disconnected"
    
    # ==================== System Events ====================
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_ERROR = "system.error"
    SYSTEM_WARNING = "system.warning"
    SYSTEM_HEALTH_CHECK = "system.health_check"
    
    # ==================== User Events ====================
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_LOGGED_IN = "user.logged_in"
    USER_LOGGED_OUT = "user.logged_out"
    USER_PASSWORD_CHANGED = "user.password.changed"


class Event(BaseModel):
    """Base event model"""
    event_id: str = Field(..., description="Unique event identifier")
    type: EventType = Field(..., description="Event type")
    data: Dict[str, Any] = Field(..., description="Event payload")
    source: Optional[str] = Field(None, description="Event source identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "event_id": "550e8400-e29b-41d4-a716-446655440000",
                "type": "lead.generated",
                "data": {
                    "company_name": "Acme Corp",
                    "email": "contact@acme.com",
                    "score": 85
                },
                "source": "lead_generation_service",
                "timestamp": "2025-10-26T12:00:00Z"
            }
        }


# ==================== Specific Event Payloads ====================

class AgentCreatedPayload(BaseModel):
    """Payload for agent.created event"""
    agent_id: str
    name: str
    agent_type: str
    capabilities: list[str]
    created_by: Optional[str] = None


class LeadGeneratedPayload(BaseModel):
    """Payload for lead.generated event"""
    lead_id: Optional[str] = None
    company_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    score: Optional[int] = None
    source: str


class WorkflowCompletedPayload(BaseModel):
    """Payload for workflow.completed event"""
    workflow_id: str
    workflow_name: str
    duration_seconds: float
    steps_completed: int
    result: Dict[str, Any]


class CRMContactCreatedPayload(BaseModel):
    """Payload for crm.contact.created event"""
    contact_id: str
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    source: str


class WakeWordDetectedPayload(BaseModel):
    """Payload for voice.wake_word.detected event"""
    keyword: str
    confidence: float
    timestamp: datetime
    audio_duration_ms: int


# ==================== Event Helpers ====================

def create_event(
    event_type: EventType,
    data: Dict[str, Any],
    source: Optional[str] = None
) -> Event:
    """
    Create a new event
    
    Args:
        event_type: Type of event
        data: Event payload
        source: Event source
        
    Returns:
        Event instance
    """
    import uuid
    return Event(
        event_id=str(uuid.uuid4()),
        type=event_type,
        data=data,
        source=source,
        timestamp=datetime.utcnow()
    )


# ==================== Event Type Groups ====================

AGENT_EVENTS = [
    EventType.AGENT_CREATED,
    EventType.AGENT_UPDATED,
    EventType.AGENT_DELETED,
    EventType.AGENT_EXECUTED,
    EventType.AGENT_EXECUTION_STARTED,
    EventType.AGENT_EXECUTION_COMPLETED,
    EventType.AGENT_EXECUTION_FAILED,
]

WORKFLOW_EVENTS = [
    EventType.WORKFLOW_CREATED,
    EventType.WORKFLOW_STARTED,
    EventType.WORKFLOW_COMPLETED,
    EventType.WORKFLOW_FAILED,
]

LEAD_EVENTS = [
    EventType.LEAD_GENERATED,
    EventType.LEAD_SCORED,
    EventType.LEAD_QUALIFIED,
    EventType.LEAD_ENRICHED,
]

CRM_EVENTS = [
    EventType.CRM_CONTACT_CREATED,
    EventType.CRM_DEAL_CREATED,
    EventType.CRM_DEAL_WON,
    EventType.CRM_DEAL_LOST,
]

VOICE_EVENTS = [
    EventType.VOICE_WAKE_WORD_DETECTED,
    EventType.VOICE_STT_COMPLETED,
    EventType.VOICE_TTS_COMPLETED,
]

