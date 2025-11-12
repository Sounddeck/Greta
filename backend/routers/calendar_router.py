"""
Google Calendar Router for CPAS4/GRETA
API endpoints for Google Calendar integration
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from datetime import datetime
from database import Database
from services.calendar_service import CalendarService
from services.oauth_service import OAuthService
from models.calendar_models import (
    CalendarEvent, CalendarEventCreate, CalendarEventUpdate,
    CalendarEventListRequest, CalendarEventListResponse,
    Calendar, CalendarListResponse, CalendarHealthResponse,
    CalendarEventResponse, CalendarDeleteResponse
)

# Initialize router
router = APIRouter(prefix="/api/v1/calendar", tags=["Google Calendar"])

# Initialize database and services
db = Database()
oauth_service = OAuthService(db)
calendar_service = CalendarService(db, oauth_service)


# ============================================================================
# Event Management Endpoints
# ============================================================================

@router.post("/events", response_model=CalendarEventResponse)
async def create_event(
    request: CalendarEventCreate,
    user_id: str = Query(..., description="User ID")
):
    """
    Create calendar event
    
    Creates a new event in the specified calendar with attendees and reminders.
    
    **Example**:
    ```json
    {
        "calendar_id": "primary",
        "summary": "Team Meeting",
        "description": "Weekly team sync",
        "location": "Conference Room A",
        "start_time": "2025-10-10T10:00:00Z",
        "end_time": "2025-10-10T11:00:00Z",
        "time_zone": "America/Los_Angeles",
        "attendees": ["alice@example.com", "bob@example.com"],
        "reminders": [
            {"method": "email", "minutes": 1440},
            {"method": "popup", "minutes": 10}
        ]
    }
    ```
    """
    try:
        return await calendar_service.create_event(user_id, request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events", response_model=CalendarEventListResponse)
async def list_events(
    user_id: str = Query(..., description="User ID"),
    calendar_id: str = Query("primary", description="Calendar ID"),
    time_min: Optional[datetime] = Query(None, description="Lower bound for event start time"),
    time_max: Optional[datetime] = Query(None, description="Upper bound for event start time"),
    max_results: int = Query(10, description="Maximum number of events"),
    page_token: Optional[str] = Query(None, description="Page token for pagination")
):
    """
    List calendar events
    
    Lists events from the specified calendar with optional time range filtering.
    
    **Query Parameters**:
    - `user_id`: User ID (required)
    - `calendar_id`: Calendar ID (default: "primary")
    - `time_min`: Lower bound for event start time (ISO 8601)
    - `time_max`: Upper bound for event start time (ISO 8601)
    - `max_results`: Maximum number of events to return (default: 10)
    - `page_token`: Token for pagination
    """
    try:
        request = CalendarEventListRequest(
            calendar_id=calendar_id,
            time_min=time_min,
            time_max=time_max,
            max_results=max_results,
            page_token=page_token
        )
        return await calendar_service.list_events(user_id, request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events/{event_id}", response_model=CalendarEvent)
async def get_event(
    event_id: str,
    user_id: str = Query(..., description="User ID"),
    calendar_id: str = Query("primary", description="Calendar ID")
):
    """
    Get calendar event by ID
    
    Retrieves detailed information about a specific calendar event.
    
    **Path Parameters**:
    - `event_id`: Event ID
    
    **Query Parameters**:
    - `user_id`: User ID (required)
    - `calendar_id`: Calendar ID (default: "primary")
    """
    try:
        return await calendar_service.get_event(user_id, event_id, calendar_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/events/{event_id}", response_model=CalendarEventResponse)
async def update_event(
    event_id: str,
    request: CalendarEventUpdate,
    user_id: str = Query(..., description="User ID"),
    calendar_id: str = Query("primary", description="Calendar ID")
):
    """
    Update calendar event
    
    Updates an existing calendar event. Only provided fields will be updated.
    
    **Path Parameters**:
    - `event_id`: Event ID
    
    **Query Parameters**:
    - `user_id`: User ID (required)
    - `calendar_id`: Calendar ID (default: "primary")
    
    **Example**:
    ```json
    {
        "summary": "Updated Team Meeting",
        "location": "Conference Room B",
        "start_time": "2025-10-10T14:00:00Z",
        "end_time": "2025-10-10T15:00:00Z"
    }
    ```
    """
    try:
        return await calendar_service.update_event(user_id, event_id, request, calendar_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/events/{event_id}", response_model=CalendarDeleteResponse)
async def delete_event(
    event_id: str,
    user_id: str = Query(..., description="User ID"),
    calendar_id: str = Query("primary", description="Calendar ID")
):
    """
    Delete calendar event
    
    Permanently deletes a calendar event.
    
    **Path Parameters**:
    - `event_id`: Event ID
    
    **Query Parameters**:
    - `user_id`: User ID (required)
    - `calendar_id`: Calendar ID (default: "primary")
    """
    try:
        return await calendar_service.delete_event(user_id, event_id, calendar_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Calendar Management Endpoints
# ============================================================================

@router.get("/calendars", response_model=CalendarListResponse)
async def list_calendars(
    user_id: str = Query(..., description="User ID")
):
    """
    List user's calendars
    
    Lists all calendars accessible by the user.
    
    **Query Parameters**:
    - `user_id`: User ID (required)
    
    **Returns**:
    - List of calendars with metadata (name, color, access role, etc.)
    """
    try:
        return await calendar_service.list_calendars(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Health Check Endpoint
# ============================================================================

@router.get("/health", response_model=CalendarHealthResponse)
async def calendar_health(
    user_id: str = Query(..., description="User ID")
):
    """
    Calendar service health check
    
    Checks the health and authentication status of the Calendar service.
    
    **Query Parameters**:
    - `user_id`: User ID (required)
    
    **Returns**:
    - Service status
    - Authentication status
    - Number of accessible calendars
    """
    try:
        return await calendar_service.get_health(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

