"""
GRETA WebSocket Router
Real-time communication endpoints for live system updates
"""

import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from typing import Dict, Any

from ..services.websocket_manager import websocket_manager

logger = logging.getLogger(__name__)
router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time communication"""
    connection_id = None
    
    try:
        # Get client IP
        client_ip = websocket.client.host if websocket.client else "unknown"
        
        # Connect client
        connection_id = await websocket_manager.connect(websocket, client_ip)
        
        # Handle messages
        while True:
            try:
                # Receive message from client
                message = await websocket.receive_text()
                
                # Handle the message
                await websocket_manager.handle_client_message(connection_id, message)
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in WebSocket message handling: {e}")
                # Send error to client
                await websocket_manager.send_to_connection(connection_id, {
                    "type": "error",
                    "data": {"message": "Message processing error"},
                    "source": "websocket_router"
                })
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        # Clean up connection
        if connection_id:
            await websocket_manager.disconnect(connection_id)

@router.get("/ws/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics"""
    return await websocket_manager.get_connection_stats()

@router.get("/ws/test")
async def websocket_test_page():
    """Test page for WebSocket functionality"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>GRETA WebSocket Test</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .connected { background-color: #d4edda; color: #155724; }
            .disconnected { background-color: #f8d7da; color: #721c24; }
            .messages { height: 300px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; margin: 10px 0; }
            .controls { margin: 10px 0; }
            button { margin: 5px; padding: 8px 16px; }
            input, select { margin: 5px; padding: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎭 GRETA WebSocket Test Interface</h1>
            
            <div id="status" class="status disconnected">
                Status: Disconnected
            </div>
            
            <div class="controls">
                <button onclick="connect()">Connect</button>
                <button onclick="disconnect()">Disconnect</button>
                <button onclick="ping()">Ping</button>
                <button onclick="getStats()">Get Stats</button>
            </div>
            
            <div class="controls">
                <select id="topicSelect">
                    <option value="system_status">System Status</option>
                    <option value="agent_updates">Agent Updates</option>
                    <option value="llm_responses">LLM Responses</option>
                    <option value="voice_synthesis">Voice Synthesis</option>
                    <option value="memory_updates">Memory Updates</option>
                    <option value="training_progress">Training Progress</option>
                    <option value="osint_results">OSINT Results</option>
                    <option value="reasoning_updates">Reasoning Updates</option>
                    <option value="workflow_status">Workflow Status</option>
                    <option value="orchestrator_commands">Orchestrator Commands</option>
                    <option value="performance_metrics">Performance Metrics</option>
                    <option value="error_notifications">Error Notifications</option>
                </select>
                <button onclick="subscribe()">Subscribe</button>
                <button onclick="unsubscribe()">Unsubscribe</button>
            </div>
            
            <div class="messages" id="messages"></div>
            
            <div class="controls">
                <input type="text" id="messageInput" placeholder="Custom message (JSON)" style="width: 300px;">
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>

        <script>
            let ws = null;
            let connectionId = null;
            
            function updateStatus(message, connected) {
                const status = document.getElementById('status');
                status.textContent = 'Status: ' + message;
                status.className = 'status ' + (connected ? 'connected' : 'disconnected');
            }
            
            function addMessage(message) {
                const messages = document.getElementById('messages');
                const div = document.createElement('div');
                div.innerHTML = '<strong>' + new Date().toLocaleTimeString() + '</strong>: ' + 
                               JSON.stringify(message, null, 2);
                messages.appendChild(div);
                messages.scrollTop = messages.scrollHeight;
            }
            
            function connect() {
                if (ws) {
                    ws.close();
                }
                
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = protocol + '//' + window.location.host + '/api/v1/ws';
                
                ws = new WebSocket(wsUrl);
                
                ws.onopen = function(event) {
                    updateStatus('Connected', true);
                    addMessage({type: 'connection', message: 'Connected to GRETA WebSocket'});
                };
                
                ws.onmessage = function(event) {
                    try {
                        const data = JSON.parse(event.data);
                        addMessage(data);
                        
                        if (data.type === 'connection_established') {
                            connectionId = data.data.connection_id;
                        }
                    } catch (e) {
                        addMessage({type: 'error', message: 'Failed to parse message: ' + event.data});
                    }
                };
                
                ws.onclose = function(event) {
                    updateStatus('Disconnected', false);
                    addMessage({type: 'connection', message: 'Disconnected from WebSocket'});
                    ws = null;
                    connectionId = null;
                };
                
                ws.onerror = function(error) {
                    updateStatus('Error', false);
                    addMessage({type: 'error', message: 'WebSocket error: ' + error});
                };
            }
            
            function disconnect() {
                if (ws) {
                    ws.close();
                }
            }
            
            function ping() {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({type: 'ping'}));
                }
            }
            
            function getStats() {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({type: 'get_stats'}));
                }
            }
            
            function subscribe() {
                const topic = document.getElementById('topicSelect').value;
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({type: 'subscribe', topic: topic}));
                }
            }
            
            function unsubscribe() {
                const topic = document.getElementById('topicSelect').value;
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({type: 'unsubscribe', topic: topic}));
                }
            }
            
            function sendMessage() {
                const input = document.getElementById('messageInput');
                if (ws && ws.readyState === WebSocket.OPEN && input.value) {
                    try {
                        const message = JSON.parse(input.value);
                        ws.send(JSON.stringify(message));
                        input.value = '';
                    } catch (e) {
                        addMessage({type: 'error', message: 'Invalid JSON: ' + e.message});
                    }
                }
            }
            
            // Auto-connect on page load
            window.onload = function() {
                connect();
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@router.post("/ws/broadcast/{topic}")
async def broadcast_to_topic(topic: str, message: Dict[str, Any]):
    """Broadcast message to all subscribers of a topic"""
    count = await websocket_manager.broadcast_to_topic(topic, {
        "type": "broadcast",
        "data": message,
        "source": "api_broadcast"
    })
    
    return {
        "success": True,
        "topic": topic,
        "subscribers_notified": count
    }

@router.post("/ws/broadcast")
async def broadcast_to_all(message: Dict[str, Any]):
    """Broadcast message to all connected clients"""
    count = await websocket_manager.broadcast_to_all({
        "type": "broadcast",
        "data": message,
        "source": "api_broadcast"
    })
    
    return {
        "success": True,
        "clients_notified": count
    }

@router.get("/ws/connections")
async def list_connections():
    """List all active WebSocket connections"""
    connections = []
    
    for conn_id, info in websocket_manager.connection_info.items():
        connections.append({
            "connection_id": conn_id,
            "client_ip": info.client_ip,
            "connected_at": info.connected_at,
            "subscriptions": list(info.subscriptions),
            "last_activity": info.last_activity
        })
    
    return {
        "active_connections": len(connections),
        "connections": connections
    }

@router.get("/ws/topics")
async def list_topics():
    """List all available topics and their subscriber counts"""
    topics = {}
    
    for topic, subscribers in websocket_manager.subscriptions.items():
        topics[topic] = {
            "subscriber_count": len(subscribers),
            "subscribers": list(subscribers)
        }
    
    return {
        "available_topics": websocket_manager._get_available_topics(),
        "active_topics": topics
    }
