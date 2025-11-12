"""
Gmail Router for Google Workspace Integration

This router provides endpoints for Gmail API operations including
sending emails, reading messages, searching, and managing labels.
"""

from fastapi import APIRouter, HTTPException, Query, Body
from typing import Optional, List

from models.gmail_models import (
    GmailSendRequest,
    GmailSendResponse,
    GmailSearchRequest,
    GmailSearchResponse,
    GmailListRequest,
    GmailMessageListResponse,
    GmailModifyRequest,
    GmailModifyResponse,
    GmailLabelCreate,
    GmailLabelResponse,
    GmailLabel,
    GmailMessage,
    GmailBatchModifyRequest,
    GmailBatchModifyResponse,
    GmailBatchDeleteRequest,
    GmailBatchDeleteResponse,
    GmailHealthResponse,
)
from services.gmail_service import GmailService

router = APIRouter(prefix="/api/v1/gmail", tags=["Gmail"])


# ============================================================================
# Send Email
# ============================================================================

@router.post("/send", response_model=GmailSendResponse)
async def send_email(
    user_id: str = Query(..., description="User ID"),
    request: GmailSendRequest = Body(...)
):
    """
    Send email via Gmail
    
    **Example**:
    ```json
    {
        "to": "recipient@example.com",
        "subject": "Hello from CPAS4",
        "body": "This is a test email",
        "cc": "cc@example.com",
        "bcc": "bcc@example.com",
        "html": false
    }
    ```
    
    **Response**:
    ```json
    {
        "message_id": "msg_abc123",
        "thread_id": "thread_xyz789",
        "label_ids": ["SENT"],
        "success": true,
        "message": "Email sent successfully"
    }
    ```
    """
    service = GmailService()
    return await service.send_email(user_id, request)


# ============================================================================
# List and Search Emails
# ============================================================================

@router.get("/messages", response_model=GmailMessageListResponse)
async def list_emails(
    user_id: str = Query(..., description="User ID"),
    max_results: int = Query(10, ge=1, le=500, description="Maximum results"),
    page_token: Optional[str] = Query(None, description="Page token for pagination"),
    label_ids: Optional[str] = Query(None, description="Comma-separated label IDs"),
    query: Optional[str] = Query(None, description="Optional search query"),
    include_spam_trash: bool = Query(False, description="Include spam and trash")
):
    """
    List emails from Gmail inbox
    
    **Example**:
    ```
    GET /api/v1/gmail/messages?user_id=user123&max_results=10&label_ids=INBOX,UNREAD
    ```
    
    **Response**:
    ```json
    {
        "messages": [
            {
                "message_id": "msg_abc123",
                "thread_id": "thread_xyz789",
                "label_ids": ["INBOX", "UNREAD"],
                "snippet": "This is a preview of the email...",
                "subject": "Email Subject",
                "from_email": "sender@example.com",
                "to_email": "user@example.com",
                "date": "2025-10-09T12:00:00Z"
            }
        ],
        "next_page_token": null,
        "result_size_estimate": 1
    }
    ```
    """
    service = GmailService()
    
    label_list = label_ids.split(",") if label_ids else None
    
    request = GmailListRequest(
        max_results=max_results,
        page_token=page_token,
        label_ids=label_list,
        query=query,
        include_spam_trash=include_spam_trash
    )
    
    return await service.list_emails(user_id, request)


@router.post("/search", response_model=GmailSearchResponse)
async def search_emails(
    user_id: str = Query(..., description="User ID"),
    request: GmailSearchRequest = Body(...)
):
    """
    Search emails in Gmail
    
    Uses Gmail search query syntax.
    
    **Search Query Examples**:
    - `from:user@example.com` - Emails from specific sender
    - `to:user@example.com` - Emails to specific recipient
    - `subject:meeting` - Emails with "meeting" in subject
    - `has:attachment` - Emails with attachments
    - `is:unread` - Unread emails
    - `after:2025/01/01` - Emails after date
    - `before:2025/12/31` - Emails before date
    
    **Example**:
    ```json
    {
        "query": "from:boss@example.com has:attachment",
        "max_results": 20,
        "include_spam_trash": false
    }
    ```
    
    **Response**:
    ```json
    {
        "messages": [...],
        "next_page_token": null,
        "result_size_estimate": 5,
        "query": "from:boss@example.com has:attachment"
    }
    ```
    """
    service = GmailService()
    return await service.search_emails(user_id, request)


@router.get("/messages/{message_id}", response_model=GmailMessage)
async def get_email(
    message_id: str,
    user_id: str = Query(..., description="User ID")
):
    """
    Get email by ID
    
    **Example**:
    ```
    GET /api/v1/gmail/messages/msg_abc123?user_id=user123
    ```
    
    **Response**:
    ```json
    {
        "message_id": "msg_abc123",
        "thread_id": "thread_xyz789",
        "label_ids": ["INBOX"],
        "snippet": "Email preview...",
        "history_id": "12345",
        "internal_date": "1696867200000",
        "size_estimate": 4096
    }
    ```
    """
    service = GmailService()
    
    message = await service.get_email(user_id, message_id)
    
    if not message:
        raise HTTPException(status_code=404, detail="Email not found")
    
    return message


# ============================================================================
# Modify Emails
# ============================================================================

@router.put("/messages/{message_id}/labels", response_model=GmailModifyResponse)
async def modify_email_labels(
    message_id: str,
    user_id: str = Query(..., description="User ID"),
    add_label_ids: Optional[str] = Query(None, description="Comma-separated label IDs to add"),
    remove_label_ids: Optional[str] = Query(None, description="Comma-separated label IDs to remove")
):
    """
    Modify email labels
    
    Add or remove labels from an email.
    
    **Common Label IDs**:
    - `INBOX` - Inbox
    - `SENT` - Sent
    - `DRAFT` - Draft
    - `TRASH` - Trash
    - `SPAM` - Spam
    - `UNREAD` - Unread
    - `STARRED` - Starred
    - `IMPORTANT` - Important
    
    **Example**:
    ```
    PUT /api/v1/gmail/messages/msg_abc123/labels?user_id=user123&add_label_ids=STARRED&remove_label_ids=UNREAD
    ```
    
    **Response**:
    ```json
    {
        "message_id": "msg_abc123",
        "thread_id": "thread_xyz789",
        "label_ids": ["INBOX", "STARRED"],
        "success": true,
        "message": "Email labels modified successfully"
    }
    ```
    """
    service = GmailService()
    
    add_list = add_label_ids.split(",") if add_label_ids else None
    remove_list = remove_label_ids.split(",") if remove_label_ids else None
    
    request = GmailModifyRequest(
        message_id=message_id,
        add_label_ids=add_list,
        remove_label_ids=remove_list
    )
    
    return await service.modify_email(user_id, request)


@router.delete("/messages/{message_id}")
async def delete_email(
    message_id: str,
    user_id: str = Query(..., description="User ID")
):
    """
    Delete email
    
    Permanently deletes an email (cannot be undone).
    
    **Example**:
    ```
    DELETE /api/v1/gmail/messages/msg_abc123?user_id=user123
    ```
    
    **Response**:
    ```json
    {
        "success": true,
        "message": "Email deleted successfully"
    }
    ```
    """
    service = GmailService()
    
    success = await service.delete_email(user_id, message_id)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete email")
    
    return {"success": True, "message": "Email deleted successfully"}


# ============================================================================
# Label Management
# ============================================================================

@router.get("/labels", response_model=List[GmailLabel])
async def list_labels(
    user_id: str = Query(..., description="User ID")
):
    """
    List Gmail labels
    
    **Example**:
    ```
    GET /api/v1/gmail/labels?user_id=user123
    ```
    
    **Response**:
    ```json
    [
        {
            "label_id": "INBOX",
            "name": "INBOX",
            "type": "system",
            "messages_total": 42,
            "messages_unread": 5
        },
        {
            "label_id": "Label_abc123",
            "name": "Work",
            "type": "user",
            "messages_total": 15,
            "messages_unread": 2
        }
    ]
    ```
    """
    service = GmailService()
    return await service.list_labels(user_id)


@router.post("/labels", response_model=GmailLabelResponse)
async def create_label(
    user_id: str = Query(..., description="User ID"),
    request: GmailLabelCreate = Body(...)
):
    """
    Create Gmail label
    
    **Example**:
    ```json
    {
        "name": "Important Projects",
        "label_list_visibility": "labelShow",
        "message_list_visibility": "show"
    }
    ```
    
    **Response**:
    ```json
    {
        "label_id": "Label_abc123",
        "name": "Important Projects",
        "type": "user",
        "messages_total": 0,
        "messages_unread": 0
    }
    ```
    """
    service = GmailService()
    return await service.create_label(user_id, request)


# ============================================================================
# Batch Operations
# ============================================================================

@router.post("/batch/modify", response_model=GmailBatchModifyResponse)
async def batch_modify_emails(
    user_id: str = Query(..., description="User ID"),
    request: GmailBatchModifyRequest = Body(...)
):
    """
    Batch modify multiple emails
    
    **Example**:
    ```json
    {
        "message_ids": ["msg_1", "msg_2", "msg_3"],
        "add_label_ids": ["STARRED"],
        "remove_label_ids": ["UNREAD"]
    }
    ```
    """
    service = GmailService()
    return await service.batch_modify(user_id, request)


@router.post("/batch/delete", response_model=GmailBatchDeleteResponse)
async def batch_delete_emails(
    user_id: str = Query(..., description="User ID"),
    request: GmailBatchDeleteRequest = Body(...)
):
    """
    Batch delete multiple emails
    
    **Example**:
    ```json
    {
        "message_ids": ["msg_1", "msg_2", "msg_3"]
    }
    ```
    """
    service = GmailService()
    return await service.batch_delete(user_id, request)


# ============================================================================
# Health Check
# ============================================================================

@router.get("/health", response_model=GmailHealthResponse)
async def get_gmail_health(
    user_id: str = Query(..., description="User ID")
):
    """
    Get Gmail service health
    
    **Example**:
    ```
    GET /api/v1/gmail/health?user_id=user123
    ```
    """
    service = GmailService()
    return await service.get_health(user_id)

