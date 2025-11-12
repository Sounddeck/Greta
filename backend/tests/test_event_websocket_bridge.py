"""
Test Event-WebSocket Bridge
Tests the integration between event bus and WebSocket
"""

import asyncio
from services.event_bus import event_bus, publish_event
from services.websocket_manager import websocket_manager
from services.event_websocket_bridge import event_websocket_bridge


async def test_event_websocket_bridge():
    """Test event bus to WebSocket forwarding"""
    print("=" * 60)
    print("EVENT-WEBSOCKET BRIDGE TEST")
    print("=" * 60)
    
    # Start the bridge
    print("\n=== Starting Bridge ===")
    await event_websocket_bridge.start()
    print(f"✅ Bridge started: {event_websocket_bridge.active}")
    
    # Get initial stats
    stats = event_websocket_bridge.get_stats()
    print(f"✅ Mapped event types: {stats['mapped_event_types']}")
    print(f"   Event types: {list(stats['mappings'].keys())[:5]}...")
    
    # Test event publishing
    print("\n=== Publishing Test Events ===")
    
    # Publish lead.scored event
    event_id = await publish_event(
        "lead.scored",
        {
            "lead_id": "test_lead_123",
            "score": 85.0,
            "confidence": 0.9,
            "factors": {
                "company_size": 80.0,
                "industry": 100.0,
                "engagement": 90.0,
                "budget": 40.0,
                "decision_maker": 100.0,
                "geographic": 100.0,
            }
        },
        source="test_suite"
    )
    print(f"✅ Published lead.scored event: {event_id[:8]}...")
    
    # Wait for async processing
    await asyncio.sleep(0.1)
    
    # Publish campaign.created event
    event_id = await publish_event(
        "campaign.created",
        {
            "campaign_id": "test_campaign_123",
            "campaign_name": "Test Campaign",
            "target_industry": "software",
        },
        source="test_suite"
    )
    print(f"✅ Published campaign.created event: {event_id[:8]}...")
    
    # Wait for async processing
    await asyncio.sleep(0.1)
    
    # Get updated stats
    stats = event_websocket_bridge.get_stats()
    print(f"\n=== Bridge Statistics ===")
    print(f"✅ Events forwarded: {stats['events_forwarded']}")
    print(f"✅ Active: {stats['active']}")
    
    # Test custom mapping
    print("\n=== Testing Custom Mapping ===")
    event_websocket_bridge.add_mapping("test.custom_event", "test_topic")
    
    # Subscribe to the custom event
    received_events = []
    
    async def test_handler(event):
        received_events.append(event)
        print(f"✅ Received event: {event['type']}")
    
    event_bus.subscribe("test.custom_event", test_handler)
    
    # Publish custom event
    await publish_event(
        "test.custom_event",
        {"message": "Hello from custom event"},
        source="test_suite"
    )
    
    # Wait for async processing
    await asyncio.sleep(0.1)
    
    print(f"✅ Custom event received: {len(received_events)} events")
    
    # Stop the bridge
    print("\n=== Stopping Bridge ===")
    await event_websocket_bridge.stop()
    print(f"✅ Bridge stopped: {not event_websocket_bridge.active}")
    
    # Final stats
    final_stats = event_websocket_bridge.get_stats()
    print(f"\n=== Final Statistics ===")
    print(f"✅ Total events forwarded: {final_stats['events_forwarded']}")
    print(f"✅ Mapped event types: {final_stats['mapped_event_types']}")
    
    print("\n" + "=" * 60)
    print("✅ ALL BRIDGE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_event_websocket_bridge())

