"""
Minimal GRETA Backend - WebSocket Test
"""
import os
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
from loguru import logger

# Configure logging
logger.remove()
logger.add(lambda msg: print(msg, end=""), format="{message}")

# Create FastAPI app
app = FastAPI(
    title="GRETA Backend - Minimal",
    description="Minimal backend for WebSocket testing",
    version="1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket endpoint
@app.websocket("/api/v1/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    logger.info(f"✅ WebSocket connection established")
    
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "connection_established",
            "data": {
                "connection_id": "test-connection",
                "message": "Connected to GRETA Backend"
            }
        })
        
        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received: {data}")
            await websocket.send_json({
                "type": "echo",
                "data": data
            })
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("WebSocket connection closed")

# Health check endpoint
@app.get("/api/v1/system/status")
async def system_status():
    """System status endpoint"""
    return {
        "status": "ok",
        "message": "GRETA Backend is running"
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "GRETA Backend - Minimal",
        "version": "1.0"
    }

if __name__ == "__main__":
    logger.info("🚀 Starting GRETA Backend - Minimal")
    logger.info("Platform: macOS")
    logger.info("✅ All imports successful")
    logger.info("🔄 Starting Uvicorn server on 0.0.0.0:8000")
    
    run(
        "main_minimal:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

