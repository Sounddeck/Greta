"""
PAI MongoDB Database Layer - Complete implementation
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from datetime import datetime, timedelta

from config import DatabaseSettings

logger = logging.getLogger(__name__)

class Database:
    """MongoDB-first PAI database implementation"""
    
    def __init__(self, settings: DatabaseSettings):
        self.settings = settings
        self.client: AsyncIOMotorClient = None
        self.db: AsyncIOMotorDatabase = None
        self._connected = False
        
    async def connect(self) -> None:
        """Establish MongoDB connection"""
        try:
            self.client = AsyncIOMotorClient(
                self.settings.mongodb_uri,
                serverSelectionTimeoutMS=self.settings.server_selection_timeout_ms
            )
            # Test connection
            await self.client.admin.command('ping')
            self.db = self.client[self.settings.database_name]
            self._connected = True
            logger.info("✅ MongoDB connected successfully")
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            self._connected = False
            raise
    
    async def disconnect(self) -> None:
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            self._connected = False
            logger.info("🔌 MongoDB disconnected")
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        if not self._connected:
            return {"status": "disconnected", "database": "unavailable"}
        
        try:
            # Test operation
            await self.client.admin.command('ping')
            return {"status": "healthy", "database": "mongodb"}
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    # CONVERSATION MANAGEMENT
    async def store_conversation(self, conversation_data: Dict[str, Any]) -> str:
        """Store conversation with full PAI context"""
        conversation = {
            **conversation_data,
            "_id": None,  # Let MongoDB generate
            "stored_at": datetime.utcnow(),
            "pai_metadata": {
                "intelligence_level": conversation_data.get("intelligence_level", "standard"),
                "reasoning_traces": conversation_data.get("reasoning_traces", []),
                "pattern_matches": conversation_data.get("pattern_matches", [])
            }
        }
        
        result = await self.db[self.settings.conversation_collection].insert_one(conversation)
        conversation["_id"] = result.inserted_id
        
        logger.info(f"Stored conversation {result.inserted_id}")
        return str(result.inserted_id)
    
    async def retrieve_conversations(self, query: Dict[str, Any], limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve conversations with advanced querying"""
        cursor = self.db[self.settings.conversation_collection].find(query).sort("stored_at", -1).limit(limit)
        return list(await cursor.to_list(length=limit))
    
    # USER PROFILE MANAGEMENT
    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user profile with PAI learning data"""
        return await self.db[self.settings.user_profile_collection].find_one({"user_id": user_id})
    
    async def update_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> None:
        """Update user profile with PAI insights"""
        update_data = {
            "$set": {
                **profile_data,
                "last_updated": datetime.utcnow(),
                "pai_adaptation_score": profile_data.get("pai_adaptation_score", 0.0)
            },
            "$inc": {"update_count": 1}
        }
        
        await self.db[self.settings.user_profile_collection].update_one(
            {"user_id": user_id},
            update_data,
            upsert=True
        )
    
    # MEMORY MANAGEMENT
    async def store_memory(self, memory_data: Dict[str, Any]) -> str:
        """Store memory with PAI indexing"""
        memory = {
            **memory_data,
            "stored_at": datetime.utcnow(),
            "access_count": 0,
            "last_accessed": datetime.utcnow(),
            "importance_score": memory_data.get("importance_score", 0.5)
        }
        
        result = await self.db[self.settings.memory_collection].insert_one(memory)
        return str(result.inserted_id)
    
    async def retrieve_memories(self, query: Dict[str, Any], limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve memories with access tracking"""
        # Update access counts for retrieved memories
        await self.db[self.settings.memory_collection].update_many(
            query,
            {"$inc": {"access_count": 1}, "$set": {"last_accessed": datetime.utcnow()}}
        )
        
        # Retrieve memories sorted by relevance
        cursor = self.db[self.settings.memory_collection].find(query).sort("importance_score", -1).limit(limit)
        return list(await cursor.to_list(length=limit))
    
    # INSIGHT MANAGEMENT
    async def store_insight(self, insight_data: Dict[str, Any]) -> str:
        """Store PAI-generated insight"""
        insight = {
            **insight_data,
            "generated_at": datetime.utcnow(),
            "confidence": insight_data.get("confidence", 0.8),
            "usage_count": 0
        }
        
        result = await self.db[self.settings.insights_collection].insert_one(insight)
        return str(result.inserted_id)
    
    async def retrieve_insights(self, query: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve insights sorted by relevance"""
        cursor = self.db[self.settings.insights_collection].find(query).sort([("confidence", -1), ("generated_at", -1)]).limit(limit)
        return list(await cursor.to_list(length=limit))
    
    # PAI-SPECIFIC QUERIES
    async def get_conversation_context(self, conversation_id: str, context_depth: int = 5) -> Dict[str, Any]:
        """Get full context for a conversation including related memories and insights"""
        # Get main conversation
        conversation = await self.db[self.settings.conversation_collection].find_one({"_id": conversation_id})
        if not conversation:
            return {}
        
        # Find related memories
        user_id = conversation.get("user_id")
        topic_keywords = conversation.get("topics", [])
        
        memory_query = {"user_id": user_id} if user_id else {}
        if topic_keywords:
            memory_query["topic_keywords"] = {"$in": topic_keywords}
        
        related_memories = await self.retrieve_memories(memory_query, limit=context_depth)
        
        # Find relevant insights
        insight_query = {"user_id": user_id} if user_id else {}
        if topic_keywords:
            insight_query["topic_keywords"] = {"$in": topic_keywords}
        
        related_insights = await self.retrieve_insights(insight_query, limit=context_depth)
        
        return {
            "conversation": conversation,
            "related_memories": related_memories,
            "relevant_insights": related_insights,
            "context_depth": context_depth,
            "pai_analysis": self._analyze_context_synergy(related_memories, related_insights)
        }
    
    async def get_pai_statistics(self) -> Dict[str, Any]:
        """Get comprehensive PAI system statistics"""
        collections = [
            self.settings.conversation_collection,
            self.settings.memory_collection,
            self.settings.user_profile_collection,
            self.settings.insights_collection
        ]
        
        stats = {}
        for collection_name in collections:
            count = await self.db[collection_name].count_documents({})
            stats[f"{collection_name}_count"] = count
        
        # Add PAI-specific metrics
        stats["pai_intelligence_score"] = await self._calculate_pai_intelligence_score()
        stats["learning_progress"] = await self._get_learning_progress()
        
        return stats
    
    async def _calculate_pai_intelligence_score(self) -> float:
        """Calculate overall PAI intelligence effectiveness"""
        conversations = await self.retrieve_conversations({}, limit=100)
        total_quality = sum(conv.get("pai_quality_score", 0.5) for conv in conversations)
        return total_quality / len(conversations) if conversations else 0.5
    
    async def _get_learning_progress(self) -> Dict[str, Any]:
        """Track PAI learning progress over time"""
        recent_conversations = await self.retrieve_conversations(
            {"stored_at": {"$gte": datetime.utcnow() - timedelta(days=30)}},
            limit=100
        )
        
        avg_quality = sum(conv.get("pai_quality_score", 0.5) for conv in recent_conversations)
        avg_quality = avg_quality / len(recent_conversations) if recent_conversations else 0.5
        
        return {
            "recent_interaction_quality": avg_quality,
            "total_interactions": len(recent_conversations),
            "learning_rate": min(avg_quality * 100, 100)  # Percentage
        }
    
    def _analyze_context_synergy(self, memories: List[Dict[str, Any]], insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how memories and insights work together"""
        memory_weight = len(memories) * 0.1
        insight_weight = len(insights) * 0.15
        synergy_score = min(memory_weight + insight_weight, 1.0)
        
        return {
            "memory_contribution": memory_weight,
            "insight_contribution": insight_weight,
            "synergy_score": synergy_score,
            "pai_context_readiness": synergy_score > 0.6
        }

# Global database instance
pai_database = Database(settings.database)

