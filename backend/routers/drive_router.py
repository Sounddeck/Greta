"""
Google Drive Router for CPAS4/GRETA
API endpoints for Google Drive integration
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from database import Database
from services.drive_service import DriveService
from services.oauth_service import OAuthService
from models.drive_models import (
    DriveFile, DriveFileUpload, DriveFileListRequest, DriveFileListResponse,
    DriveFileUploadResponse, DriveFileDownloadResponse,
    DriveShareRequest, DriveShareResponse,
    DriveFolderCreate, DriveFolderCreateResponse,
    DriveDeleteResponse, DriveHealthResponse
)

# Initialize router
router = APIRouter(prefix="/api/v1/drive", tags=["Google Drive"])

# Initialize database and services
db = Database()
oauth_service = OAuthService(db)
drive_service = DriveService(db, oauth_service)


# ============================================================================
# File Management Endpoints
# ============================================================================

@router.post("/files/upload", response_model=DriveFileUploadResponse)
async def upload_file(
    request: DriveFileUpload,
    user_id: str = Query(..., description="User ID")
):
    """
    Upload file to Google Drive
    
    Uploads a file to Google Drive with optional parent folder.
    
    **Example**:
    ```json
    {
        "name": "document.txt",
        "mime_type": "text/plain",
        "content": "SGVsbG8gV29ybGQh",
        "parent_folder_id": "folder_abc123",
        "description": "Important document"
    }
    ```
    
    **Note**: Content should be base64 encoded.
    """
    try:
        return await drive_service.upload_file(user_id, request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files/{file_id}/download", response_model=DriveFileDownloadResponse)
async def download_file(
    file_id: str,
    user_id: str = Query(..., description="User ID")
):
    """
    Download file from Google Drive
    
    Downloads a file from Google Drive and returns base64 encoded content.
    
    **Path Parameters**:
    - `file_id`: File ID
    
    **Query Parameters**:
    - `user_id`: User ID (required)
    
    **Returns**:
    - File metadata and base64 encoded content
    """
    try:
        return await drive_service.download_file(user_id, file_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/files", response_model=DriveFileListResponse)
async def list_files(
    user_id: str = Query(..., description="User ID"),
    folder_id: Optional[str] = Query(None, description="Folder ID to list files from"),
    mime_type: Optional[str] = Query(None, description="Filter by MIME type"),
    query: Optional[str] = Query(None, description="Search query"),
    max_results: int = Query(10, description="Maximum number of files"),
    page_token: Optional[str] = Query(None, description="Page token for pagination"),
    include_trashed: bool = Query(False, description="Include trashed files")
):
    """
    List files from Google Drive
    
    Lists files from Google Drive with optional filtering and pagination.
    
    **Query Parameters**:
    - `user_id`: User ID (required)
    - `folder_id`: Folder ID to list files from
    - `mime_type`: Filter by MIME type (e.g., "application/pdf")
    - `query`: Search query (e.g., "name contains 'report'")
    - `max_results`: Maximum number of files to return (default: 10)
    - `page_token`: Token for pagination
    - `include_trashed`: Include trashed files (default: false)
    """
    try:
        request = DriveFileListRequest(
            folder_id=folder_id,
            mime_type=mime_type,
            query=query,
            max_results=max_results,
            page_token=page_token,
            include_trashed=include_trashed
        )
        return await drive_service.list_files(user_id, request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files/{file_id}", response_model=DriveFile)
async def get_file(
    file_id: str,
    user_id: str = Query(..., description="User ID")
):
    """
    Get file metadata from Google Drive
    
    Retrieves detailed metadata about a specific file.
    
    **Path Parameters**:
    - `file_id`: File ID
    
    **Query Parameters**:
    - `user_id`: User ID (required)
    
    **Returns**:
    - File metadata (name, size, MIME type, permissions, etc.)
    """
    try:
        return await drive_service.get_file(user_id, file_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/files/{file_id}", response_model=DriveDeleteResponse)
async def delete_file(
    file_id: str,
    user_id: str = Query(..., description="User ID")
):
    """
    Delete file from Google Drive
    
    Permanently deletes a file from Google Drive.
    
    **Path Parameters**:
    - `file_id`: File ID
    
    **Query Parameters**:
    - `user_id`: User ID (required)
    
    **Warning**: This action cannot be undone.
    """
    try:
        return await drive_service.delete_file(user_id, file_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Sharing and Permissions Endpoints
# ============================================================================

@router.post("/files/{file_id}/share", response_model=DriveShareResponse)
async def share_file(
    file_id: str,
    request: DriveShareRequest,
    user_id: str = Query(..., description="User ID")
):
    """
    Share file with permissions
    
    Shares a file with another user or group with specified permissions.
    
    **Path Parameters**:
    - `file_id`: File ID
    
    **Query Parameters**:
    - `user_id`: User ID (required)
    
    **Example**:
    ```json
    {
        "email": "colleague@example.com",
        "role": "writer",
        "type": "user",
        "send_notification": true,
        "message": "Please review this document"
    }
    ```
    
    **Roles**:
    - `owner`: Full ownership
    - `writer`: Can edit
    - `commenter`: Can comment
    - `reader`: Can view only
    """
    try:
        return await drive_service.share_file(user_id, file_id, request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Folder Management Endpoints
# ============================================================================

@router.post("/folders", response_model=DriveFolderCreateResponse)
async def create_folder(
    request: DriveFolderCreate,
    user_id: str = Query(..., description="User ID")
):
    """
    Create folder in Google Drive
    
    Creates a new folder in Google Drive with optional parent folder.
    
    **Example**:
    ```json
    {
        "name": "Project Documents",
        "parent_folder_id": "folder_abc123",
        "description": "All project-related documents"
    }
    ```
    """
    try:
        return await drive_service.create_folder(user_id, request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Health Check Endpoint
# ============================================================================

@router.get("/health", response_model=DriveHealthResponse)
async def drive_health(
    user_id: str = Query(..., description="User ID")
):
    """
    Drive service health check
    
    Checks the health and authentication status of the Drive service.
    
    **Query Parameters**:
    - `user_id`: User ID (required)
    
    **Returns**:
    - Service status
    - Authentication status
    - Storage quota information
    """
    try:
        return await drive_service.get_health(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

