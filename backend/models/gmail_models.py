"""
Gmail Models for Google Workspace Integration

This module defines Pydantic models for Gmail API operations including
messages, threads, labels, and attachments.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, EmailStr


class GmailLabelType(str, Enum):
    """Gmail label types"""
    SYSTEM = "system"
    USER = "user"


class GmailMessageFormat(str, Enum):
    """Gmail message formats"""
    FULL = "full"
    METADATA = "metadata"
    MINIMAL = "minimal"
    RAW = "raw"


# ============================================================================
# Gmail Label Models
# ============================================================================

class GmailLabelBase(BaseModel):
    """Base Gmail label"""
    name: str
    label_list_visibility: Optional[str] = "labelShow"
    message_list_visibility: Optional[str] = "show"


class GmailLabel(GmailLabelBase):
    """Gmail label with metadata"""
    label_id: str
    type: GmailLabelType
    messages_total: Optional[int] = 0
    messages_unread: Optional[int] = 0
    threads_total: Optional[int] = 0
    threads_unread: Optional[int] = 0


class GmailLabelCreate(GmailLabelBase):
    """Create Gmail label"""
    pass


class GmailLabelResponse(BaseModel):
    """Gmail label response"""
    label_id: str
    name: str
    type: GmailLabelType
    messages_total: int
    messages_unread: int


# ============================================================================
# Gmail Message Models
# ============================================================================

class GmailMessageHeader(BaseModel):
    """Gmail message header"""
    name: str
    value: str


class GmailMessagePart(BaseModel):
    """Gmail message part"""
    part_id: Optional[str] = None
    mime_type: str
    filename: Optional[str] = None
    body: Optional[Dict[str, Any]] = None
    parts: Optional[List['GmailMessagePart']] = None


class GmailMessagePayload(BaseModel):
    """Gmail message payload"""
    part_id: Optional[str] = None
    mime_type: str
    filename: Optional[str] = None
    headers: List[GmailMessageHeader]
    body: Optional[Dict[str, Any]] = None
    parts: Optional[List[GmailMessagePart]] = None


class GmailMessageBase(BaseModel):
    """Base Gmail message"""
    thread_id: Optional[str] = None
    label_ids: Optional[List[str]] = None
    snippet: Optional[str] = None


class GmailMessage(GmailMessageBase):
    """Gmail message with full details"""
    message_id: str
    history_id: Optional[str] = None
    internal_date: Optional[str] = None
    payload: Optional[GmailMessagePayload] = None
    size_estimate: Optional[int] = None
    raw: Optional[str] = None


class GmailMessageMetadata(BaseModel):
    """Gmail message metadata"""
    message_id: str
    thread_id: str
    label_ids: List[str]
    snippet: str
    history_id: str
    internal_date: str
    size_estimate: int
    subject: Optional[str] = None
    from_email: Optional[str] = None
    to_email: Optional[str] = None
    date: Optional[str] = None


class GmailMessageListResponse(BaseModel):
    """Gmail message list response"""
    messages: List[GmailMessageMetadata]
    next_page_token: Optional[str] = None
    result_size_estimate: int


# ============================================================================
# Gmail Send Models
# ============================================================================

class GmailSendRequest(BaseModel):
    """Send Gmail message request"""
    to: str = Field(..., description="Recipient email address")
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Email body (plain text or HTML)")
    cc: Optional[str] = Field(None, description="CC email addresses (comma-separated)")
    bcc: Optional[str] = Field(None, description="BCC email addresses (comma-separated)")
    reply_to: Optional[str] = Field(None, description="Reply-to email address")
    html: bool = Field(False, description="Whether body is HTML")
    thread_id: Optional[str] = Field(None, description="Thread ID for reply")


class GmailSendResponse(BaseModel):
    """Send Gmail message response"""
    message_id: str
    thread_id: str
    label_ids: List[str]
    success: bool
    message: str


# ============================================================================
# Gmail Search Models
# ============================================================================

class GmailSearchRequest(BaseModel):
    """Search Gmail messages request"""
    query: str = Field(..., description="Gmail search query (e.g., 'from:user@example.com')")
    max_results: int = Field(10, ge=1, le=500, description="Maximum results to return")
    page_token: Optional[str] = Field(None, description="Page token for pagination")
    include_spam_trash: bool = Field(False, description="Include spam and trash")


class GmailSearchResponse(BaseModel):
    """Search Gmail messages response"""
    messages: List[GmailMessageMetadata]
    next_page_token: Optional[str] = None
    result_size_estimate: int
    query: str


# ============================================================================
# Gmail List Models
# ============================================================================

class GmailListRequest(BaseModel):
    """List Gmail messages request"""
    max_results: int = Field(10, ge=1, le=500, description="Maximum results to return")
    page_token: Optional[str] = Field(None, description="Page token for pagination")
    label_ids: Optional[List[str]] = Field(None, description="Filter by label IDs")
    query: Optional[str] = Field(None, description="Optional search query")
    include_spam_trash: bool = Field(False, description="Include spam and trash")


# ============================================================================
# Gmail Modify Models
# ============================================================================

class GmailModifyRequest(BaseModel):
    """Modify Gmail message request"""
    message_id: str = Field(..., description="Message ID to modify")
    add_label_ids: Optional[List[str]] = Field(None, description="Label IDs to add")
    remove_label_ids: Optional[List[str]] = Field(None, description="Label IDs to remove")


class GmailModifyResponse(BaseModel):
    """Modify Gmail message response"""
    message_id: str
    thread_id: str
    label_ids: List[str]
    success: bool
    message: str


# ============================================================================
# Gmail Attachment Models
# ============================================================================

class GmailAttachment(BaseModel):
    """Gmail attachment"""
    attachment_id: str
    size: int
    filename: str
    mime_type: str


class GmailAttachmentDownloadRequest(BaseModel):
    """Download Gmail attachment request"""
    message_id: str
    attachment_id: str


class GmailAttachmentDownloadResponse(BaseModel):
    """Download Gmail attachment response"""
    attachment_id: str
    filename: str
    mime_type: str
    size: int
    data: str  # Base64 encoded


# ============================================================================
# Gmail Thread Models
# ============================================================================

class GmailThread(BaseModel):
    """Gmail thread"""
    thread_id: str
    history_id: str
    messages: List[GmailMessage]
    snippet: Optional[str] = None


class GmailThreadListResponse(BaseModel):
    """Gmail thread list response"""
    threads: List[Dict[str, str]]  # Simplified thread list
    next_page_token: Optional[str] = None
    result_size_estimate: int


# ============================================================================
# Gmail Batch Operations
# ============================================================================

class GmailBatchModifyRequest(BaseModel):
    """Batch modify Gmail messages request"""
    message_ids: List[str] = Field(..., description="Message IDs to modify")
    add_label_ids: Optional[List[str]] = Field(None, description="Label IDs to add")
    remove_label_ids: Optional[List[str]] = Field(None, description="Label IDs to remove")


class GmailBatchModifyResponse(BaseModel):
    """Batch modify Gmail messages response"""
    success: bool
    modified_count: int
    message: str


class GmailBatchDeleteRequest(BaseModel):
    """Batch delete Gmail messages request"""
    message_ids: List[str] = Field(..., description="Message IDs to delete")


class GmailBatchDeleteResponse(BaseModel):
    """Batch delete Gmail messages response"""
    success: bool
    deleted_count: int
    message: str


# ============================================================================
# Gmail Health Check Models
# ============================================================================

class GmailHealthResponse(BaseModel):
    """Gmail service health response"""
    status: str
    message: str
    timestamp: datetime
    authenticated: bool
    user_email: Optional[str] = None
    quota_remaining: Optional[int] = None

