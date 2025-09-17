"""Enhanced Database Manager with MongoDB"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure
from loguru import logger
from typing import Dict, Any
class Database:
    def __init__(self):
        self.client: AsyncIOMotorClient = None
        self.db = None
        self.connected = False
        
    async def connect(self, mongodb_url: str = "mongodb://localhost:27017", database_name: str = "greta_pai"):
        """Connect to MongoDB database"""
        try:
            self.client = AsyncIOMotorClient(mongodb_url)
            await self.client.admin.command('ping')
            self.db = self.client[database_name]
            self.connected = True
            logger.info(f"✅ Connected to MongoDB: {database_name}")
        except ConnectionFailure:
            logger.warning("⚠️ MongoDB not available, using file-based fallback")
            self.connected = False
        except Exception as e:
            logger.error(f"❌ Database error: {e}")
            self.connected = False
    
    async def disconnect(self):
        """Disconnect from MongoDB"""
        if self.client:
            self.client.close()
            self.connected = False
            logger.info("✅ Database disconnected")
    
    async def health_check(self):
        """Check database health"""
        if not self.connected:
            return {"status": "disconnected"}
        try:
            await self.client.admin.command('ping')
            return {"status": "connected", "database": self.db.name}
        except Exception as e:
            return {"status": "error", "error": str(e)}
