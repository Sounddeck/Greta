"""
Google Drive Models for CPAS4/GRETA
Pydantic models for Google Drive API integration
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class DriveFileType(str, Enum):
    """Drive file type"""
    FILE = "file"
    FOLDER = "folder"
    DOCUMENT = "document"
    SPREADSHEET = "spreadsheet"
    PRESENTATION = "presentation"
    FORM = "form"
    DRAWING = "drawing"
    PDF = "pdf"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    ARCHIVE = "archive"
    OTHER = "other"


class PermissionRole(str, Enum):
    """Permission role"""
    OWNER = "owner"
    ORGANIZER = "organizer"
    FILE_ORGANIZER = "fileOrganizer"
    WRITER = "writer"
    COMMENTER = "commenter"
    READER = "reader"


class PermissionType(str, Enum):
    """Permission type"""
    USER = "user"
    GROUP = "group"
    DOMAIN = "domain"
    ANYONE = "anyone"


# ============================================================================
# Drive Models
# ============================================================================

class DrivePermission(BaseModel):
    """Drive file permission"""
    permission_id: Optional[str] = Field(None, description="Permission ID")
    email: Optional[str] = Field(None, description="Email address (for user/group)")
    role: PermissionRole = Field(..., description="Permission role")
    type: PermissionType = Field(..., description="Permission type")
    display_name: Optional[str] = Field(None, description="Display name")
    domain: Optional[str] = Field(None, description="Domain (for domain type)")
    allow_file_discovery: bool = Field(False, description="Allow file discovery")


class DriveFile(BaseModel):
    """Drive file metadata"""
    file_id: str = Field(..., description="File ID")
    name: str = Field(..., description="File name")
    mime_type: str = Field(..., description="MIME type")
    file_type: DriveFileType = Field(..., description="File type")
    size: Optional[int] = Field(None, description="File size in bytes")
    created_time: datetime = Field(..., description="Creation time")
    modified_time: datetime = Field(..., description="Last modification time")
    web_view_link: Optional[str] = Field(None, description="Link to view file in browser")
    web_content_link: Optional[str] = Field(None, description="Link to download file")
    parents: List[str] = Field(default_factory=list, description="Parent folder IDs")
    permissions: List[DrivePermission] = Field(default_factory=list, description="File permissions")
    owner_email: Optional[str] = Field(None, description="Owner email")
    shared: bool = Field(False, description="Whether file is shared")
    trashed: bool = Field(False, description="Whether file is in trash")
    starred: bool = Field(False, description="Whether file is starred")
    description: Optional[str] = Field(None, description="File description")
    thumbnail_link: Optional[str] = Field(None, description="Thumbnail link")


class DriveFileUpload(BaseModel):
    """Upload file to Drive request"""
    name: str = Field(..., description="File name")
    mime_type: str = Field(..., description="MIME type")
    content: str = Field(..., description="File content (base64 encoded)")
    parent_folder_id: Optional[str] = Field(None, description="Parent folder ID")
    description: Optional[str] = Field(None, description="File description")


class DriveFileListRequest(BaseModel):
    """List Drive files request"""
    folder_id: Optional[str] = Field(None, description="Folder ID to list files from")
    mime_type: Optional[str] = Field(None, description="Filter by MIME type")
    query: Optional[str] = Field(None, description="Search query")
    max_results: int = Field(10, description="Maximum number of files to return")
    page_token: Optional[str] = Field(None, description="Page token for pagination")
    order_by: str = Field("modifiedTime desc", description="Order by field")
    include_trashed: bool = Field(False, description="Include trashed files")


class DriveFileListResponse(BaseModel):
    """List Drive files response"""
    files: List[DriveFile] = Field(..., description="List of files")
    next_page_token: Optional[str] = Field(None, description="Token for next page")
    total_results: int = Field(..., description="Total number of results")


class DriveFileUploadResponse(BaseModel):
    """Upload file response"""
    file: DriveFile = Field(..., description="Uploaded file metadata")
    success: bool = Field(..., description="Whether upload was successful")
    message: str = Field(..., description="Response message")


class DriveFileDownloadResponse(BaseModel):
    """Download file response"""
    file_id: str = Field(..., description="File ID")
    name: str = Field(..., description="File name")
    mime_type: str = Field(..., description="MIME type")
    content: str = Field(..., description="File content (base64 encoded)")
    size: int = Field(..., description="File size in bytes")
    success: bool = Field(..., description="Whether download was successful")
    message: str = Field(..., description="Response message")


class DriveShareRequest(BaseModel):
    """Share file request"""
    email: str = Field(..., description="Email address to share with")
    role: PermissionRole = Field(..., description="Permission role")
    type: PermissionType = Field(PermissionType.USER, description="Permission type")
    send_notification: bool = Field(True, description="Send email notification")
    message: Optional[str] = Field(None, description="Custom message in notification")


class DriveShareResponse(BaseModel):
    """Share file response"""
    permission: DrivePermission = Field(..., description="Created permission")
    success: bool = Field(..., description="Whether sharing was successful")
    message: str = Field(..., description="Response message")


class DriveFolderCreate(BaseModel):
    """Create folder request"""
    name: str = Field(..., description="Folder name")
    parent_folder_id: Optional[str] = Field(None, description="Parent folder ID")
    description: Optional[str] = Field(None, description="Folder description")


class DriveFolderCreateResponse(BaseModel):
    """Create folder response"""
    folder: DriveFile = Field(..., description="Created folder metadata")
    success: bool = Field(..., description="Whether creation was successful")
    message: str = Field(..., description="Response message")


class DriveDeleteResponse(BaseModel):
    """Delete file response"""
    success: bool = Field(..., description="Whether deletion was successful")
    message: str = Field(..., description="Response message")
    file_id: str = Field(..., description="Deleted file ID")


class DriveHealthResponse(BaseModel):
    """Drive service health response"""
    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Status message")
    timestamp: datetime = Field(..., description="Timestamp")
    authenticated: bool = Field(..., description="Whether user is authenticated")
    user_email: Optional[str] = Field(None, description="User email")
    storage_quota: Optional[dict] = Field(None, description="Storage quota information")


# ============================================================================
# MIME Type Constants
# ============================================================================

class DriveMimeTypes:
    """Common Google Drive MIME types"""
    
    # Google Workspace types
    FOLDER = "application/vnd.google-apps.folder"
    DOCUMENT = "application/vnd.google-apps.document"
    SPREADSHEET = "application/vnd.google-apps.spreadsheet"
    PRESENTATION = "application/vnd.google-apps.presentation"
    FORM = "application/vnd.google-apps.form"
    DRAWING = "application/vnd.google-apps.drawing"
    
    # Common file types
    PDF = "application/pdf"
    TEXT = "text/plain"
    HTML = "text/html"
    CSV = "text/csv"
    JSON = "application/json"
    XML = "application/xml"
    
    # Microsoft Office types
    WORD = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    EXCEL = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    POWERPOINT = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    
    # Image types
    JPEG = "image/jpeg"
    PNG = "image/png"
    GIF = "image/gif"
    SVG = "image/svg+xml"
    
    # Video types
    MP4 = "video/mp4"
    WEBM = "video/webm"
    
    # Audio types
    MP3 = "audio/mpeg"
    WAV = "audio/wav"
    
    # Archive types
    ZIP = "application/zip"
    TAR = "application/x-tar"
    GZIP = "application/gzip"
    
    @classmethod
    def get_file_type(cls, mime_type: str) -> DriveFileType:
        """Get file type from MIME type"""
        if mime_type == cls.FOLDER:
            return DriveFileType.FOLDER
        elif mime_type == cls.DOCUMENT:
            return DriveFileType.DOCUMENT
        elif mime_type == cls.SPREADSHEET:
            return DriveFileType.SPREADSHEET
        elif mime_type == cls.PRESENTATION:
            return DriveFileType.PRESENTATION
        elif mime_type == cls.FORM:
            return DriveFileType.FORM
        elif mime_type == cls.DRAWING:
            return DriveFileType.DRAWING
        elif mime_type == cls.PDF:
            return DriveFileType.PDF
        elif mime_type.startswith("image/"):
            return DriveFileType.IMAGE
        elif mime_type.startswith("video/"):
            return DriveFileType.VIDEO
        elif mime_type.startswith("audio/"):
            return DriveFileType.AUDIO
        elif mime_type in [cls.ZIP, cls.TAR, cls.GZIP]:
            return DriveFileType.ARCHIVE
        else:
            return DriveFileType.OTHER

