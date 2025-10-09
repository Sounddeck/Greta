"""
Google Calendar Models for CPAS4/GRETA
Pydantic models for Google Calendar API integration
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class EventStatus(str, Enum):
    """Calendar event status"""
    CONFIRMED = "confirmed"
    TENTATIVE = "tentative"
    CANCELLED = "cancelled"


class AttendeeResponseStatus(str, Enum):
    """Attendee response status"""
    NEEDS_ACTION = "needsAction"
    DECLINED = "declined"
    TENTATIVE = "tentative"
    ACCEPTED = "accepted"


class ReminderMethod(str, Enum):
    """Reminder method"""
    EMAIL = "email"
    POPUP = "popup"
    SMS = "sms"


# ============================================================================
# Calendar Models
# ============================================================================

class CalendarAttendee(BaseModel):
    """Calendar event attendee"""
    email: str = Field(..., description="Attendee email address")
    display_name: Optional[str] = Field(None, description="Attendee display name")
    response_status: AttendeeResponseStatus = Field(
        AttendeeResponseStatus.NEEDS_ACTION,
        description="Attendee response status"
    )
    optional: bool = Field(False, description="Whether attendance is optional")
    organizer: bool = Field(False, description="Whether attendee is organizer")
    self: bool = Field(False, description="Whether attendee is the calendar owner")


class EventReminder(BaseModel):
    """Event reminder"""
    method: ReminderMethod = Field(..., description="Reminder method")
    minutes: int = Field(..., description="Minutes before event to trigger reminder")


class EventDateTime(BaseModel):
    """Event date/time"""
    date_time: Optional[datetime] = Field(None, description="DateTime for timed events")
    date: Optional[str] = Field(None, description="Date for all-day events (YYYY-MM-DD)")
    time_zone: Optional[str] = Field(None, description="Time zone (e.g., 'America/Los_Angeles')")


class CalendarEvent(BaseModel):
    """Calendar event"""
    event_id: str = Field(..., description="Event ID")
    calendar_id: str = Field(..., description="Calendar ID")
    summary: str = Field(..., description="Event title/summary")
    description: Optional[str] = Field(None, description="Event description")
    location: Optional[str] = Field(None, description="Event location")
    start: EventDateTime = Field(..., description="Event start time")
    end: EventDateTime = Field(..., description="Event end time")
    attendees: List[CalendarAttendee] = Field(default_factory=list, description="Event attendees")
    reminders: List[EventReminder] = Field(default_factory=list, description="Event reminders")
    recurrence: Optional[List[str]] = Field(None, description="Recurrence rules (RRULE)")
    status: EventStatus = Field(EventStatus.CONFIRMED, description="Event status")
    html_link: Optional[str] = Field(None, description="Link to event in Google Calendar")
    created: Optional[datetime] = Field(None, description="Creation time")
    updated: Optional[datetime] = Field(None, description="Last update time")
    creator_email: Optional[str] = Field(None, description="Creator email")
    organizer_email: Optional[str] = Field(None, description="Organizer email")


class CalendarEventCreate(BaseModel):
    """Create calendar event request"""
    calendar_id: str = Field("primary", description="Calendar ID (default: primary)")
    summary: str = Field(..., description="Event title/summary")
    description: Optional[str] = Field(None, description="Event description")
    location: Optional[str] = Field(None, description="Event location")
    start_time: datetime = Field(..., description="Event start time")
    end_time: datetime = Field(..., description="Event end time")
    time_zone: str = Field("UTC", description="Time zone")
    attendees: List[str] = Field(default_factory=list, description="Attendee email addresses")
    reminders: List[EventReminder] = Field(default_factory=list, description="Event reminders")
    recurrence: Optional[List[str]] = Field(None, description="Recurrence rules")
    all_day: bool = Field(False, description="Whether event is all-day")


class CalendarEventUpdate(BaseModel):
    """Update calendar event request"""
    summary: Optional[str] = Field(None, description="Event title/summary")
    description: Optional[str] = Field(None, description="Event description")
    location: Optional[str] = Field(None, description="Event location")
    start_time: Optional[datetime] = Field(None, description="Event start time")
    end_time: Optional[datetime] = Field(None, description="Event end time")
    time_zone: Optional[str] = Field(None, description="Time zone")
    attendees: Optional[List[str]] = Field(None, description="Attendee email addresses")
    reminders: Optional[List[EventReminder]] = Field(None, description="Event reminders")
    recurrence: Optional[List[str]] = Field(None, description="Recurrence rules")
    status: Optional[EventStatus] = Field(None, description="Event status")


class CalendarEventListRequest(BaseModel):
    """List calendar events request"""
    calendar_id: str = Field("primary", description="Calendar ID")
    time_min: Optional[datetime] = Field(None, description="Lower bound for event start time")
    time_max: Optional[datetime] = Field(None, description="Upper bound for event start time")
    max_results: int = Field(10, description="Maximum number of events to return")
    page_token: Optional[str] = Field(None, description="Page token for pagination")
    single_events: bool = Field(True, description="Expand recurring events into instances")
    order_by: str = Field("startTime", description="Order by (startTime or updated)")


class CalendarEventListResponse(BaseModel):
    """List calendar events response"""
    events: List[CalendarEvent] = Field(..., description="List of events")
    next_page_token: Optional[str] = Field(None, description="Token for next page")
    total_results: int = Field(..., description="Total number of results")


class Calendar(BaseModel):
    """Google Calendar"""
    calendar_id: str = Field(..., description="Calendar ID")
    summary: str = Field(..., description="Calendar title")
    description: Optional[str] = Field(None, description="Calendar description")
    location: Optional[str] = Field(None, description="Calendar location")
    time_zone: str = Field(..., description="Calendar time zone")
    primary: bool = Field(False, description="Whether this is the primary calendar")
    access_role: str = Field(..., description="User's access role (owner, writer, reader)")
    background_color: Optional[str] = Field(None, description="Background color")
    foreground_color: Optional[str] = Field(None, description="Foreground color")


class CalendarListResponse(BaseModel):
    """List calendars response"""
    calendars: List[Calendar] = Field(..., description="List of calendars")
    total_results: int = Field(..., description="Total number of calendars")


class CalendarHealthResponse(BaseModel):
    """Calendar service health response"""
    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Status message")
    timestamp: datetime = Field(..., description="Timestamp")
    authenticated: bool = Field(..., description="Whether user is authenticated")
    user_email: Optional[str] = Field(None, description="User email")
    calendars_count: int = Field(0, description="Number of calendars accessible")


class CalendarEventResponse(BaseModel):
    """Calendar event response"""
    event: CalendarEvent = Field(..., description="Calendar event")
    success: bool = Field(..., description="Whether operation was successful")
    message: str = Field(..., description="Response message")


class CalendarDeleteResponse(BaseModel):
    """Calendar event delete response"""
    success: bool = Field(..., description="Whether deletion was successful")
    message: str = Field(..., description="Response message")
    event_id: str = Field(..., description="Deleted event ID")

