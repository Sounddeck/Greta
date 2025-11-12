"""
Comprehensive tests for event_bus.py module

Tests cover:
- Event publishing and subscription
- Event routing
- Handler registration
- Error handling
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from typing import Dict, Any, List


class TestEventPublishingAndSubscription:
    """Test event publishing and subscription"""
    
    @pytest.mark.asyncio
    async def test_publish_event_success(self):
        """Test successfully publishing an event"""
        from services.event_bus import EventBus
        
        bus = EventBus()
        
        event_id = await bus.publish(
            event_type="test.event",
            data={"key": "value"},
            source="test_source"
        )
        
        assert event_id is not None
        assert len(bus.event_history) == 1
        assert bus.event_history[0]["type"] == "test.event"
        assert bus.event_history[0]["data"] == {"key": "value"}
    
    @pytest.mark.asyncio
    async def test_subscribe_to_event(self):
        """Test subscribing to specific event types"""
        from services.event_bus import EventBus
        
        bus = EventBus()
        callback = AsyncMock()
        
        bus.subscribe("test.event", callback)
        
        assert "test.event" in bus.subscribers
        assert callback in bus.subscribers["test.event"]
    
    @pytest.mark.asyncio
    async def test_multiple_subscribers_same_event(self):
        """Test multiple subscribers to the same event"""
        from services.event_bus import EventBus
        
        bus = EventBus()
        callback1 = AsyncMock()
        callback2 = AsyncMock()
        callback3 = AsyncMock()
        
        bus.subscribe("test.event", callback1)
        bus.subscribe("test.event", callback2)
        bus.subscribe("test.event", callback3)
        
        await bus.publish("test.event", {"data": "test"})
        
        # All callbacks should be called
        callback1.assert_called_once()
        callback2.assert_called_once()
        callback3.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_unsubscribe_from_event(self):
        """Test unsubscribing from events"""
        from services.event_bus import EventBus
        
        bus = EventBus()
        callback = AsyncMock()
        
        bus.subscribe("test.event", callback)
        assert callback in bus.subscribers["test.event"]
        
        bus.unsubscribe("test.event", callback)
        assert callback not in bus.subscribers["test.event"]
    
    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent_callback(self):
        """Test unsubscribing a callback that doesn't exist"""
        from services.event_bus import EventBus
        
        bus = EventBus()
        callback = AsyncMock()
        
        # Should not raise an error
        bus.unsubscribe("test.event", callback)


class TestEventRouting:
    """Test event routing"""
    
    @pytest.mark.asyncio
    async def test_route_to_correct_handlers(self):
        """Test routing events to correct handlers"""
        from services.event_bus import EventBus
        
        bus = EventBus()
        callback_type1 = AsyncMock()
        callback_type2 = AsyncMock()
        
        bus.subscribe("event.type1", callback_type1)
        bus.subscribe("event.type2", callback_type2)
        
        await bus.publish("event.type1", {"data": "test1"})
        
        # Only type1 callback should be called
        callback_type1.assert_called_once()
        callback_type2.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_event_filtering_by_type(self):
        """Test event filtering by type"""
        from services.event_bus import EventBus
        
        bus = EventBus()
        
        await bus.publish("event.type1", {"data": "test1"})
        await bus.publish("event.type2", {"data": "test2"})
        await bus.publish("event.type1", {"data": "test3"})
        
        # Get history filtered by type
        type1_events = bus.get_history(event_type="event.type1")
        
        assert len(type1_events) == 2
        assert all(e["type"] == "event.type1" for e in type1_events)
    
    @pytest.mark.asyncio
    async def test_publish_to_nonexistent_topic(self):
        """Test publishing to topic with no subscribers"""
        from services.event_bus import EventBus
        
        bus = EventBus()
        
        # Should not raise an error
        event_id = await bus.publish("nonexistent.topic", {"data": "test"})
        
        assert event_id is not None
        assert len(bus.event_history) == 1
    
    @pytest.mark.asyncio
    async def test_event_history_limit(self):
        """Test event history respects max limit"""
        from services.event_bus import EventBus
        
        bus = EventBus(max_history=5)
        
        # Publish more events than the limit
        for i in range(10):
            await bus.publish("test.event", {"index": i})
        
        # Should only keep the last 5
        assert len(bus.event_history) == 5
        assert bus.event_history[0]["data"]["index"] == 5
        assert bus.event_history[-1]["data"]["index"] == 9


class TestHandlerRegistration:
    """Test handler registration"""
    
    @pytest.mark.asyncio
    async def test_register_async_handler(self):
        """Test registering async event handlers"""
        from services.event_bus import EventBus
        
        bus = EventBus()
        
        async def async_handler(event):
            pass
        
        bus.subscribe("test.event", async_handler)
        
        assert async_handler in bus.subscribers["test.event"]
    
    @pytest.mark.asyncio
    async def test_register_sync_handler(self):
        """Test registering sync event handlers"""
        from services.event_bus import EventBus
        
        bus = EventBus()
        
        def sync_handler(event):
            pass
        
        bus.subscribe("test.event", sync_handler)
        
        assert sync_handler in bus.subscribers["test.event"]
    
    @pytest.mark.asyncio
    async def test_handler_execution_order(self):
        """Test handler execution order"""
        from services.event_bus import EventBus
        
        bus = EventBus()
        execution_order = []
        
        async def handler1(event):
            execution_order.append(1)
        
        async def handler2(event):
            execution_order.append(2)
        
        async def handler3(event):
            execution_order.append(3)
        
        bus.subscribe("test.event", handler1)
        bus.subscribe("test.event", handler2)
        bus.subscribe("test.event", handler3)
        
        await bus.publish("test.event", {"data": "test"})
        
        # All handlers should execute (order may vary due to concurrent execution)
        assert len(execution_order) == 3
        assert set(execution_order) == {1, 2, 3}
    
    @pytest.mark.asyncio
    async def test_get_subscribers_count(self):
        """Test getting subscriber count"""
        from services.event_bus import EventBus
        
        bus = EventBus()
        
        bus.subscribe("event1", AsyncMock())
        bus.subscribe("event1", AsyncMock())
        bus.subscribe("event2", AsyncMock())
        
        assert bus.get_subscribers_count("event1") == 2
        assert bus.get_subscribers_count("event2") == 1
        assert bus.get_subscribers_count() == 3


class TestErrorHandling:
    """Test error handling"""
    
    @pytest.mark.asyncio
    async def test_handler_error_isolation(self):
        """Test that one handler failure doesn't affect others"""
        from services.event_bus import EventBus
        
        bus = EventBus()
        
        async def failing_handler(event):
            raise Exception("Handler error")
        
        successful_handler = AsyncMock()
        
        bus.subscribe("test.event", failing_handler)
        bus.subscribe("test.event", successful_handler)
        
        # Should not raise an exception
        await bus.publish("test.event", {"data": "test"})
        
        # Successful handler should still be called
        successful_handler.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_sync_handler_error_handling(self):
        """Test error handling for sync handlers"""
        from services.event_bus import EventBus
        
        bus = EventBus()
        
        def failing_sync_handler(event):
            raise Exception("Sync handler error")
        
        successful_handler = AsyncMock()
        
        bus.subscribe("test.event", failing_sync_handler)
        bus.subscribe("test.event", successful_handler)
        
        # Should not raise an exception
        await bus.publish("test.event", {"data": "test"})
        
        # Successful handler should still be called
        successful_handler.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_history_with_filters(self):
        """Test getting event history with filters"""
        from services.event_bus import EventBus
        
        bus = EventBus()
        
        await bus.publish("event1", {"data": "test1"})
        await asyncio.sleep(0.01)  # Small delay
        await bus.publish("event2", {"data": "test2"})
        await bus.publish("event1", {"data": "test3"})
        
        # Filter by event type
        event1_history = bus.get_history(event_type="event1")
        assert len(event1_history) == 2
        
        # Filter by limit
        limited_history = bus.get_history(limit=2)
        assert len(limited_history) == 2
    
    @pytest.mark.asyncio
    async def test_clear_history(self):
        """Test clearing event history"""
        from services.event_bus import EventBus
        
        bus = EventBus()
        
        await bus.publish("test.event", {"data": "test1"})
        await bus.publish("test.event", {"data": "test2"})
        
        assert len(bus.event_history) == 2
        
        bus.clear_history()
        
        assert len(bus.event_history) == 0
    
    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting event bus statistics"""
        from services.event_bus import EventBus
        
        bus = EventBus()
        
        bus.subscribe("event1", AsyncMock())
        bus.subscribe("event1", AsyncMock())
        bus.subscribe("event2", AsyncMock())
        
        await bus.publish("event1", {"data": "test1"})
        await bus.publish("event1", {"data": "test2"})
        await bus.publish("event2", {"data": "test3"})
        
        stats = bus.get_stats()
        
        assert stats["total_events"] == 3
        assert stats["total_subscribers"] == 3
        assert stats["event_types"]["event1"] == 2
        assert stats["event_types"]["event2"] == 1

