#!/usr/bin/env python3
"""
Simple WebSocket test script
"""
import asyncio
import websockets
import json

async def test_websocket():
    """Test WebSocket connection"""
    uri = "ws://localhost:8000/api/v1/ws"
    
    try:
        print(f"Connecting to {uri}...")
        async with websockets.connect(uri) as websocket:
            print("✅ Connected!")
            
            # Receive welcome message
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"Received: {message}")
                data = json.loads(message)
                print(f"Message type: {data.get('type')}")
                print(f"Connection ID: {data.get('data', {}).get('connection_id')}")
            except asyncio.TimeoutError:
                print("❌ Timeout waiting for welcome message")
                return
            
            # Send a test message
            test_msg = json.dumps({"type": "ping"})
            print(f"Sending: {test_msg}")
            await websocket.send(test_msg)
            
            # Receive response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"Received response: {response}")
            except asyncio.TimeoutError:
                print("❌ Timeout waiting for response")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_websocket())

