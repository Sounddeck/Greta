"""
GRETA PAI - Input Validation Models
Security Hardening - Phase 1
Comprehensive request validation for all API endpoints
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum
import re
from datetime import datetime


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class FileType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    PDF = "pdf"
    DOCX = "docx"
    AUDIO = "audio"
    VIDEO = "video"


class SecurityValidatedRequest(BaseModel):
    """Base class for all validated requests with security checks"""

    def __init__(self, **data):
        """Custom init to sanitize input data"""
        for key, value in data.items():
            if isinstance(value, str):
                # Remove potential XSS patterns
                value = re.sub(r'<script[^>]*>.*?</script>', '', value, flags=re.IGNORECASE | re.DOTALL)
                value = re.sub(r'javascript:', '', value, flags=re.IGNORECASE)
                value = re.sub(r'on\w+=["\'][^"\']*["\']', '', value, flags=re.IGNORECASE)
                data[key] = value
        super().__init__(**data)


class ChatRequest(SecurityValidatedRequest):
    """Validated chat message request"""
    message: str = Field(..., min_length=1, max_length=10000,
                        description="User message content")
    conversation_id: Optional[str] = Field(None, regex=r'^conv_[a-f0-9]{8,}$',
                                         description="Optional conversation ID")
    context_window: Optional[int] = Field(1000, ge=1, le=10000,
                                        description="Context window size")
    files: List[str] = Field(default_factory=list, max_items=10,
                           description="File identifiers")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict,
                                             description="Additional metadata")

    class Config:
        schema_extra = {
            "example": {
                "message": "Hello Greta, please analyze this document",
                "conversation_id": None,
                "context_window": 1000,
                "files": [],
                "metadata": {"priority": "high"}
            }
        }


class FileUploadRequest(SecurityValidatedRequest):
    """Validated file upload request"""
    filename: str = Field(..., min_length=1, max_length=255,
                         description="Original filename")
    content_type: str = Field(...,
                            description="MIME content type")
    size_bytes: int = Field(..., gt=0, le=100*1024*1024,  # 100MB max
                           description="File size in bytes")

    @validator('filename')
    def validate_filename(cls, v):
        """Validate filename for security"""
        if '..' in v or '/' in v or '\\' in v:
            raise ValueError('Invalid filename: path traversal detected')
        if not re.match(r'^[\w\-. ]+$', v):
            raise ValueError('Invalid filename: contains unsafe characters')
        return v

    @validator('content_type')
    def validate_content_type(cls, v):
        """Validate content type for security"""
        allowed_types = {
            # Documents
            'text/plain', 'application/pdf', 'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            # Images
            'image/jpeg', 'image/png', 'image/gif', 'image/webp',
            # Audio
            'audio/mpeg', 'audio/wav', 'audio/mp4', 'audio/ogg',
            # Video
            'video/mp4', 'video/webm', 'video/ogg',
        }
        if v.lower() not in allowed_types:
            raise ValueError(f'Unsupported content type: {v}')
        return v


class AgentTaskRequest(SecurityValidatedRequest):
    """Validated agent task request"""
    agent_type: str = Field(..., regex=r'^[a-z_]{2,20}$',
                           description="Agent type identifier")
    task_description: str = Field(..., min_length=10, max_length=2000,
                                description="Task description")
    priority: str = Field("normal", regex=r'^(normal|high|urgent)$',
                         description="Task priority")
    timeout_seconds: int = Field(300, ge=30, le=3600,
                               description="Task timeout in seconds")


class VoiceSynthesisRequest(SecurityValidatedRequest):
    """Validated voice synthesis request"""
    text: str = Field(..., min_length=1, max_length=5000,
                     description="Text to synthesize")
    emotion: str = Field("neutral", regex=r'^(neutral|happy|sad|angry|excited)$',
                        description="Emotional tone")
    language: str = Field("de", regex=r'^(de|en)$',
                         description="Language code")
    speed: float = Field(1.0, ge=0.5, le=2.0,
                        description="Speech speed multiplier")


class LearningFeedbackRequest(SecurityValidatedRequest):
    """Validated learning feedback request"""
    interaction_id: str = Field(..., regex=r'^int_[a-f0-9]{8,}$',
                              description="Interaction identifier")
    rating: float = Field(..., ge=0.0, le=5.0,
                         description="User rating (0-5)")
    notes: Optional[str] = Field(None, max_length=1000,
                               description="Optional feedback notes")
    timestamp: datetime = Field(default_factory=datetime.utcnow,
                              description="Feedback timestamp")


class SystemCommandRequest(SecurityValidatedRequest):
    """Validated system command request"""
    command: str = Field(..., regex=r'^[a-z_][a-z0-9_]*$',
                        description="System command name")
    parameters: Dict[str, Any] = Field(default_factory=dict,
                                     description="Command parameters")

    @validator('parameters')
    def validate_parameters(cls, v):
        """Validate command parameters structure"""
        if len(v) > 10:
            raise ValueError('Too many parameters')
        for key, value in v.items():
            if not re.match(r'^[a-z_][a-z0-9_]{0,20}$', key):
                raise ValueError(f'Invalid parameter name: {key}')
        return v


class BulkOperationRequest(SecurityValidatedRequest):
    """Validated bulk operation request"""
    operation: str = Field(...,
                          description="Bulk operation type")
    items: List[Dict[str, Any]] = Field(..., max_items=100,
                                      description="Items to process")
    batch_size: int = Field(10, ge=1, le=50,
                           description="Processing batch size")


# Response models for consistent API structure
class APIResponse(BaseModel):
    """Standard API response structure"""
    success: bool = Field(..., description="Operation success status")
    data: Optional[Any] = Field(None, description="Response data")
    message: str = Field("Operation completed", description="Response message")
    timestamp: datetime = Field(default_factory=datetime.utcnow,
                              description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Unique request identifier")


class ErrorResponse(BaseModel):
    """Standard error response structure"""
    success: bool = Field(False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code for programmatic handling")
    timestamp: datetime = Field(default_factory=datetime.utcnow,
                              description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


# WebSocket message validation
class WebSocketMessage(BaseModel):
    """Validated WebSocket message"""
    type: str = Field(..., regex=r'^[a-z_]+$', description="Message type")
    payload: Dict[str, Any] = Field(..., description="Message payload")
    message_id: Optional[str] = Field(None, regex=r'^msg_[a-f0-9]{8,}$',
                                    description="Message identifier")
