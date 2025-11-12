"""
Google Calendar Module Tests
Tests for Calendar models, service, and router
"""

import sys
import os
import asyncio
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import Database
from services.calendar_service import CalendarService
from services.oauth_service import OAuthService
from models.calendar_models import (
    CalendarEventCreate, CalendarEventUpdate, CalendarEventListRequest,
    EventReminder, ReminderMethod
)


class CalendarModuleTests:
    """Calendar module test suite"""
    
    def __init__(self):
        self.db = Database()
        self.oauth_service = OAuthService(self.db)
        self.service = CalendarService(self.db, self.oauth_service)
        self.test_user_id = "test_calendar_user"
    
    async def test_1_service_initialization(self):
        """Test 1: Service initialization"""
        print("\n" + "="*80)
        print("TEST 1: Calendar Service Initialization")
        print("="*80)
        
        assert self.service is not None
        assert self.service.mock_mode is True
        
        print("✅ Calendar service initialized successfully")
        print(f"   - Mock mode: {self.service.mock_mode}")
    
    async def test_2_create_event(self):
        """Test 2: Create calendar event"""
        print("\n" + "="*80)
        print("TEST 2: Create Calendar Event")
        print("="*80)
        
        request = CalendarEventCreate(
            calendar_id="primary",
            summary="Team Meeting",
            description="Weekly team sync",
            location="Conference Room A",
            start_time=datetime.utcnow() + timedelta(days=1),
            end_time=datetime.utcnow() + timedelta(days=1, hours=1),
            time_zone="America/Los_Angeles",
            attendees=["alice@example.com", "bob@example.com"],
            reminders=[
                EventReminder(method=ReminderMethod.EMAIL, minutes=1440),
                EventReminder(method=ReminderMethod.POPUP, minutes=10)
            ]
        )
        
        response = await self.service.create_event(self.test_user_id, request)
        
        assert response.success is True
        assert response.event.summary == "Team Meeting"
        assert len(response.event.attendees) == 2
        assert len(response.event.reminders) == 2
        
        print("✅ Event created successfully")
        print(f"   - Event ID: {response.event.event_id}")
        print(f"   - Summary: {response.event.summary}")
        print(f"   - Attendees: {len(response.event.attendees)}")
        print(f"   - Reminders: {len(response.event.reminders)}")
    
    async def test_3_list_events(self):
        """Test 3: List calendar events"""
        print("\n" + "="*80)
        print("TEST 3: List Calendar Events")
        print("="*80)
        
        request = CalendarEventListRequest(
            calendar_id="primary",
            max_results=5
        )
        
        response = await self.service.list_events(self.test_user_id, request)
        
        assert len(response.events) > 0
        assert response.total_results > 0
        
        print("✅ Events listed successfully")
        print(f"   - Events count: {len(response.events)}")
        print(f"   - Total results: {response.total_results}")
        
        for i, event in enumerate(response.events[:3], 1):
            print(f"\n   Event {i}:")
            print(f"   - ID: {event.event_id}")
            print(f"   - Summary: {event.summary}")
            print(f"   - Start: {event.start.date_time}")
    
    async def test_4_get_event(self):
        """Test 4: Get calendar event by ID"""
        print("\n" + "="*80)
        print("TEST 4: Get Calendar Event by ID")
        print("="*80)
        
        event_id = "test_event_123"
        event = await self.service.get_event(self.test_user_id, event_id)
        
        assert event.event_id == event_id
        assert event.summary is not None
        
        print("✅ Event retrieved successfully")
        print(f"   - Event ID: {event.event_id}")
        print(f"   - Summary: {event.summary}")
        print(f"   - Location: {event.location}")
    
    async def test_5_update_event(self):
        """Test 5: Update calendar event"""
        print("\n" + "="*80)
        print("TEST 5: Update Calendar Event")
        print("="*80)
        
        event_id = "test_event_123"
        request = CalendarEventUpdate(
            summary="Updated Team Meeting",
            location="Conference Room B",
            start_time=datetime.utcnow() + timedelta(days=2),
            end_time=datetime.utcnow() + timedelta(days=2, hours=1)
        )
        
        response = await self.service.update_event(self.test_user_id, event_id, request)
        
        assert response.success is True
        assert response.event.summary == "Updated Team Meeting"
        assert response.event.location == "Conference Room B"
        
        print("✅ Event updated successfully")
        print(f"   - Event ID: {response.event.event_id}")
        print(f"   - New summary: {response.event.summary}")
        print(f"   - New location: {response.event.location}")
    
    async def test_6_delete_event(self):
        """Test 6: Delete calendar event"""
        print("\n" + "="*80)
        print("TEST 6: Delete Calendar Event")
        print("="*80)
        
        event_id = "test_event_to_delete"
        response = await self.service.delete_event(self.test_user_id, event_id)
        
        assert response.success is True
        assert response.event_id == event_id
        
        print("✅ Event deleted successfully")
        print(f"   - Event ID: {response.event_id}")
        print(f"   - Message: {response.message}")
    
    async def test_7_list_calendars(self):
        """Test 7: List user's calendars"""
        print("\n" + "="*80)
        print("TEST 7: List User's Calendars")
        print("="*80)
        
        response = await self.service.list_calendars(self.test_user_id)
        
        assert len(response.calendars) > 0
        assert response.total_results > 0
        
        print("✅ Calendars listed successfully")
        print(f"   - Calendars count: {len(response.calendars)}")
        
        for i, calendar in enumerate(response.calendars, 1):
            print(f"\n   Calendar {i}:")
            print(f"   - ID: {calendar.calendar_id}")
            print(f"   - Summary: {calendar.summary}")
            print(f"   - Primary: {calendar.primary}")
            print(f"   - Access role: {calendar.access_role}")
    
    async def test_8_calendar_health(self):
        """Test 8: Calendar service health check"""
        print("\n" + "="*80)
        print("TEST 8: Calendar Service Health Check")
        print("="*80)
        
        response = await self.service.get_health(self.test_user_id)
        
        assert response.status is not None
        assert response.timestamp is not None
        
        print("✅ Calendar health check successful")
        print(f"   - Status: {response.status}")
        print(f"   - Message: {response.message}")
        print(f"   - Authenticated: {response.authenticated}")
        print(f"   - Calendars count: {response.calendars_count}")
    
    async def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*80)
        print("GOOGLE CALENDAR MODULE TESTS")
        print("="*80)
        
        tests = [
            self.test_1_service_initialization,
            self.test_2_create_event,
            self.test_3_list_events,
            self.test_4_get_event,
            self.test_5_update_event,
            self.test_6_delete_event,
            self.test_7_list_calendars,
            self.test_8_calendar_health
        ]
        
        passed = 0
        failed = 0
        
        for test in tests:
            try:
                await test()
                passed += 1
            except AssertionError as e:
                print(f"❌ Test failed: {e}")
                failed += 1
            except Exception as e:
                print(f"❌ Test error: {e}")
                failed += 1
        
        print("\n" + "="*80)
        print("CALENDAR MODULE TESTS COMPLETE")
        print("="*80)
        print(f"\n{'='*80}")
        if failed == 0:
            print(f"✅ ALL TESTS PASSED ({passed}/{passed + failed})")
        else:
            print(f"❌ SOME TESTS FAILED ({passed}/{passed + failed})")
        print(f"{'='*80}\n")


def main():
    """Main test runner"""
    tests = CalendarModuleTests()
    asyncio.get_event_loop().run_until_complete(tests.run_all_tests())


if __name__ == "__main__":
    main()

