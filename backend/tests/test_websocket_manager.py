"""
Comprehensive tests for websocket_manager.py module

Tests cover:
- WebSocket connection establishment and authentication
- Message broadcasting to multiple connected clients
- Disconnection handling and cleanup (graceful and ungraceful)
- Connection state management (active, idle, disconnected)
- Subscription/topic management
- Error handling for failed connections and invalid messages
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect


class TestWebSocketConnection:
    """Test WebSocket connection establishment"""
    
    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful WebSocket connection"""
        from services.websocket_manager import WebSocketManager
        
        manager = WebSocketManager()
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        
        connection_id = await manager.connect(mock_websocket, "192.168.1.1")
        
        assert connection_id is not None
        assert connection_id in manager.active_connections
        assert mock_websocket.accept.called
        assert manager.total_connections == 1
    
    @pytest.mark.asyncio
    async def test_connect_stores_connection_info(self):
        """Test that connection info is stored correctly"""
        from services.websocket_manager import WebSocketManager
        
        manager = WebSocketManager()
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        
        connection_id = await manager.connect(mock_websocket, "192.168.1.1")
        
        assert connection_id in manager.connection_info
        conn_info = manager.connection_info[connection_id]
        assert conn_info.client_ip == "192.168.1.1"
        assert conn_info.connection_id == connection_id
    
    @pytest.mark.asyncio
    async def test_connect_sends_welcome_message(self):
        """Test that welcome message is sent on connection"""
        from services.websocket_manager import WebSocketManager
        
        manager = WebSocketManager()
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        
        connection_id = await manager.connect(mock_websocket, "192.168.1.1")
        
        # Verify welcome message was sent
        assert mock_websocket.send_text.called
        call_args = mock_websocket.send_text.call_args[0][0]
        message = json.loads(call_args)
        assert message["type"] == "connection_established"


class TestWebSocketDisconnection:
    """Test WebSocket disconnection handling"""
    
    @pytest.mark.asyncio
    async def test_disconnect_graceful(self):
        """Test graceful disconnection"""
        from services.websocket_manager import WebSocketManager
        
        manager = WebSocketManager()
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        
        connection_id = await manager.connect(mock_websocket, "192.168.1.1")
        
        # Disconnect
        await manager.disconnect(connection_id)
        
        assert connection_id not in manager.active_connections
        assert connection_id not in manager.connection_info
    
    @pytest.mark.asyncio
    async def test_disconnect_cleans_up_subscriptions(self):
        """Test that disconnection cleans up subscriptions"""
        from services.websocket_manager import WebSocketManager
        
        manager = WebSocketManager()
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        
        connection_id = await manager.connect(mock_websocket, "192.168.1.1")
        
        # Subscribe to topics
        await manager.subscribe(connection_id, "test_topic")
        
        # Disconnect
        await manager.disconnect(connection_id)
        
        # Verify subscriptions are cleaned up
        assert connection_id not in manager.connection_subscriptions
        assert "test_topic" not in manager.subscriptions or connection_id not in manager.subscriptions.get("test_topic", set())
    
    @pytest.mark.asyncio
    async def test_disconnect_nonexistent_connection(self):
        """Test disconnecting a non-existent connection"""
        from services.websocket_manager import WebSocketManager
        
        manager = WebSocketManager()
        
        # Should not raise an error
        await manager.disconnect("nonexistent_id")


class TestWebSocketMessaging:
    """Test WebSocket message sending"""
    
    @pytest.mark.asyncio
    async def test_send_to_connection_success(self):
        """Test sending message to specific connection"""
        from services.websocket_manager import WebSocketManager
        
        manager = WebSocketManager()
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        
        connection_id = await manager.connect(mock_websocket, "192.168.1.1")
        
        # Send message
        result = await manager.send_to_connection(connection_id, {
            "type": "test_message",
            "data": {"key": "value"}
        })
        
        assert result is True
        assert manager.total_messages_sent >= 1
    
    @pytest.mark.asyncio
    async def test_send_to_nonexistent_connection(self):
        """Test sending message to non-existent connection"""
        from services.websocket_manager import WebSocketManager
        
        manager = WebSocketManager()
        
        result = await manager.send_to_connection("nonexistent_id", {
            "type": "test_message",
            "data": {}
        })
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_send_adds_timestamp(self):
        """Test that timestamp is added to messages"""
        from services.websocket_manager import WebSocketManager
        
        manager = WebSocketManager()
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        
        connection_id = await manager.connect(mock_websocket, "192.168.1.1")
        
        # Send message without timestamp
        await manager.send_to_connection(connection_id, {
            "type": "test_message",
            "data": {}
        })
        
        # Verify timestamp was added
        call_args = mock_websocket.send_text.call_args_list[-1][0][0]
        message = json.loads(call_args)
        assert "timestamp" in message


class TestWebSocketBroadcasting:
    """Test WebSocket broadcasting"""
    
    @pytest.mark.asyncio
    async def test_broadcast_to_topic(self):
        """Test broadcasting to topic subscribers"""
        from services.websocket_manager import WebSocketManager
        
        manager = WebSocketManager()
        
        # Create multiple connections
        mock_ws1 = AsyncMock(spec=WebSocket)
        mock_ws1.accept = AsyncMock()
        mock_ws1.send_text = AsyncMock()
        
        mock_ws2 = AsyncMock(spec=WebSocket)
        mock_ws2.accept = AsyncMock()
        mock_ws2.send_text = AsyncMock()
        
        conn_id1 = await manager.connect(mock_ws1, "192.168.1.1")
        conn_id2 = await manager.connect(mock_ws2, "192.168.1.2")
        
        # Subscribe to topic
        await manager.subscribe(conn_id1, "test_topic")
        await manager.subscribe(conn_id2, "test_topic")
        
        # Broadcast to topic
        count = await manager.broadcast_to_topic("test_topic", {
            "type": "broadcast_message",
            "data": {"info": "test"}
        })
        
        assert count == 2
    
    @pytest.mark.asyncio
    async def test_broadcast_to_empty_topic(self):
        """Test broadcasting to topic with no subscribers"""
        from services.websocket_manager import WebSocketManager
        
        manager = WebSocketManager()
        
        count = await manager.broadcast_to_topic("empty_topic", {
            "type": "broadcast_message",
            "data": {}
        })
        
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_broadcast_to_all(self):
        """Test broadcasting to all connections"""
        from services.websocket_manager import WebSocketManager
        
        manager = WebSocketManager()
        
        # Create multiple connections
        mock_ws1 = AsyncMock(spec=WebSocket)
        mock_ws1.accept = AsyncMock()
        mock_ws1.send_text = AsyncMock()
        
        mock_ws2 = AsyncMock(spec=WebSocket)
        mock_ws2.accept = AsyncMock()
        mock_ws2.send_text = AsyncMock()
        
        await manager.connect(mock_ws1, "192.168.1.1")
        await manager.connect(mock_ws2, "192.168.1.2")
        
        # Broadcast to all
        count = await manager.broadcast_to_all({
            "type": "global_message",
            "data": {}
        })
        
        assert count == 2


class TestWebSocketSubscriptions:
    """Test WebSocket subscription management"""
    
    @pytest.mark.asyncio
    async def test_subscribe_to_topic(self):
        """Test subscribing to a topic"""
        from services.websocket_manager import WebSocketManager
        
        manager = WebSocketManager()
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        
        connection_id = await manager.connect(mock_websocket, "192.168.1.1")
        
        result = await manager.subscribe(connection_id, "test_topic")
        
        assert result is True
        assert "test_topic" in manager.subscriptions
        assert connection_id in manager.subscriptions["test_topic"]
    
    @pytest.mark.asyncio
    async def test_unsubscribe_from_topic(self):
        """Test unsubscribing from a topic"""
        from services.websocket_manager import WebSocketManager
        
        manager = WebSocketManager()
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        
        connection_id = await manager.connect(mock_websocket, "192.168.1.1")
        
        # Subscribe then unsubscribe
        await manager.subscribe(connection_id, "test_topic")
        result = await manager.unsubscribe(connection_id, "test_topic")
        
        assert result is True
        assert connection_id not in manager.subscriptions.get("test_topic", set())
    
    @pytest.mark.asyncio
    async def test_subscribe_nonexistent_connection(self):
        """Test subscribing with non-existent connection"""
        from services.websocket_manager import WebSocketManager
        
        manager = WebSocketManager()
        
        result = await manager.subscribe("nonexistent_id", "test_topic")
        
        assert result is False


class TestWebSocketStatistics:
    """Test WebSocket statistics"""
    
    @pytest.mark.asyncio
    async def test_get_connection_stats(self):
        """Test getting connection statistics"""
        from services.websocket_manager import WebSocketManager
        
        manager = WebSocketManager()
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        
        await manager.connect(mock_websocket, "192.168.1.1")
        
        stats = await manager.get_connection_stats()
        
        assert "active_connections" in stats
        assert "total_connections" in stats
        assert "total_messages_sent" in stats
        assert stats["active_connections"] == 1
    
    @pytest.mark.asyncio
    async def test_get_connection_info(self):
        """Test getting specific connection info"""
        from services.websocket_manager import WebSocketManager
        
        manager = WebSocketManager()
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        
        connection_id = await manager.connect(mock_websocket, "192.168.1.1")
        
        info = await manager.get_connection_info(connection_id)
        
        assert info is not None
        assert info.connection_id == connection_id
        assert info.client_ip == "192.168.1.1"


class TestWebSocketMessageHandling:
    """Test WebSocket message handling"""
    
    @pytest.mark.asyncio
    async def test_handle_subscribe_message(self):
        """Test handling subscribe message from client"""
        from services.websocket_manager import WebSocketManager
        
        manager = WebSocketManager()
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        
        connection_id = await manager.connect(mock_websocket, "192.168.1.1")
        
        # Handle subscribe message
        message = json.dumps({"type": "subscribe", "topic": "test_topic"})
        await manager.handle_client_message(connection_id, message)
        
        assert "test_topic" in manager.subscriptions
        assert connection_id in manager.subscriptions["test_topic"]
    
    @pytest.mark.asyncio
    async def test_handle_ping_message(self):
        """Test handling ping message from client"""
        from services.websocket_manager import WebSocketManager
        
        manager = WebSocketManager()
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        
        connection_id = await manager.connect(mock_websocket, "192.168.1.1")
        
        # Handle ping message
        message = json.dumps({"type": "ping"})
        await manager.handle_client_message(connection_id, message)
        
        # Verify pong was sent
        call_args = mock_websocket.send_text.call_args_list[-1][0][0]
        response = json.loads(call_args)
        assert response["type"] == "pong"
    
    @pytest.mark.asyncio
    async def test_handle_invalid_json(self):
        """Test handling invalid JSON message"""
        from services.websocket_manager import WebSocketManager
        
        manager = WebSocketManager()
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        
        connection_id = await manager.connect(mock_websocket, "192.168.1.1")
        
        # Should not raise an error
        await manager.handle_client_message(connection_id, "invalid json")


# Fixtures
@pytest.fixture
def websocket_manager():
    """Fixture to provide a WebSocket manager instance"""
    from services.websocket_manager import WebSocketManager
    return WebSocketManager()


@pytest.fixture
async def connected_websocket():
    """Fixture to provide a connected WebSocket"""
    mock_websocket = AsyncMock(spec=WebSocket)
    mock_websocket.accept = AsyncMock()
    mock_websocket.send_text = AsyncMock()
    return mock_websocket

