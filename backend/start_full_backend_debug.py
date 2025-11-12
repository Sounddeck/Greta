#!/usr/bin/env python3
"""
Script to start the full backend with debug output
"""
import subprocess
import time
import os
import signal
import sys
import threading

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

def read_output(pipe, label):
    """Read output from a pipe"""
    for line in iter(pipe.readline, ''):
        if line:
            print(f"[{label}] {line.rstrip()}")

def start_backend():
    """Start the full backend"""
    print("\n🚀 Starting full backend...")
    try:
        # Change to backend directory
        os.chdir('/Users/macone/Desktop/CPAS5/backend')
        
        # Start the backend process
        process = subprocess.Popen(
            ['python3', 'main.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        print(f"✅ Backend process started with PID: {process.pid}")
        
        # Start threads to read output
        stdout_thread = threading.Thread(target=read_output, args=(process.stdout, "STDOUT"))
        stderr_thread = threading.Thread(target=read_output, args=(process.stderr, "STDERR"))
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()
        
        # Wait for backend to start
        print("\n⏳ Waiting for backend to start (10 seconds)...")
        time.sleep(10)
        
        return process
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return None

def main():
    """Main function"""
    print("=" * 60)
    print("CPAS5 Full Backend Debug")
    print("=" * 60)
    
    # Kill any existing process on port 8000
    kill_port_8000()
    
    # Start the backend
    backend_process = start_backend()
    
    if backend_process is None:
        print("❌ Failed to start backend")
        sys.exit(1)
    
    try:
        # Keep backend running
        print("\n🔄 Backend is running. Press Ctrl+C to stop.")
        backend_process.wait()
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping backend...")
        backend_process.terminate()
        try:
            backend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend_process.kill()
        print("✅ Backend stopped")

if __name__ == "__main__":
    main()

