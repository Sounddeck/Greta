#!/usr/bin/env python3
"""
Script to start the minimal backend and test WebSocket connection
"""
import subprocess
import time
import asyncio
import websockets
import json
import os
import signal
import sys

def kill_port_8000():
    """Kill any process using port 8000"""
    try:
        result = subprocess.run(['lsof', '-i', ':8000'], capture_output=True, text=True, timeout=5)
        lines = result.stdout.strip().split('\n')
        
        if len(lines) > 1:
            # Skip header, get PID from first data line
            parts = lines[1].split()
            pid = int(parts[1])
            print(f"Found process with PID: {pid} on port 8000")
            
            # Kill the process
            os.kill(pid, signal.SIGKILL)
            print(f"✅ Killed process {pid}")
            time.sleep(2)
        else:
            print("✅ No process found on port 8000")
    except Exception as e:
        print(f"⚠️ Error killing port 8000: {e}")

def start_backend():
    """Start the minimal backend"""
    print("\n🚀 Starting minimal backend...")
    try:
        # Change to backend directory
        os.chdir('/Users/macone/Desktop/CPAS5/backend')
        
        # Start the backend process
        process = subprocess.Popen(
            ['python3', 'main_minimal.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print(f"✅ Backend process started with PID: {process.pid}")
        
        # Wait for backend to start
        time.sleep(5)
        
        return process
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return None

async def test_websocket():
    """Test WebSocket connection"""
    print("\n🧪 Testing WebSocket connection...")
    try:
        async with websockets.connect('ws://localhost:8000/api/v1/ws', ping_interval=None) as ws:
            print('✅ Connected to WebSocket!')
            
            # Wait for welcome message
            msg = await asyncio.wait_for(ws.recv(), timeout=3)
            data = json.loads(msg)
            
            print(f'✅ Received message type: {data.get("type")}')
            print(f'✅ Connection ID: {data.get("data", {}).get("connection_id")}')
            print(f'\n✅ Full response:')
            print(json.dumps(data, indent=2))
            
            return True
    except asyncio.TimeoutError:
        print('❌ Timeout waiting for message')
        return False
    except Exception as e:
        print(f'❌ Error: {type(e).__name__}: {e}')
        return False

def main():
    """Main function"""
    print("=" * 60)
    print("CPAS5 Backend WebSocket Test")
    print("=" * 60)
    
    # Kill any existing process on port 8000
    kill_port_8000()
    
    # Start the backend
    backend_process = start_backend()
    
    if backend_process is None:
        print("❌ Failed to start backend")
        sys.exit(1)
    
    try:
        # Test WebSocket connection
        success = asyncio.run(test_websocket())
        
        if success:
            print("\n" + "=" * 60)
            print("✅ WebSocket connection test PASSED!")
            print("=" * 60)
            print("\nThe backend is running and WebSocket is working.")
            print("You can now:")
            print("1. Open http://localhost:5173 in your browser")
            print("2. Check the browser console for WebSocket connection status")
            print("3. The 'Disconnected' status should change to 'Connected'")
        else:
            print("\n" + "=" * 60)
            print("❌ WebSocket connection test FAILED!")
            print("=" * 60)
    finally:
        # Keep backend running
        print("\n🔄 Backend is running. Press Ctrl+C to stop.")
        try:
            backend_process.wait()
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping backend...")
            backend_process.terminate()
            backend_process.wait(timeout=5)
            print("✅ Backend stopped")

if __name__ == "__main__":
    main()

