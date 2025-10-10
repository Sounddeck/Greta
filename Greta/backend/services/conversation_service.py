"""
Conversation Storage Service
Week 4 - Component 1: Conversation Management with MongoDB

Manages conversation storage, history tracking, and retrieval.
Integrates with AutoGen multi-agent conversations from Week 3.
"""

import os
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
import json


class Message(BaseModel):
    """Message schema"""
    role: str  # "user", "assistant", "system", or agent name
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tokens: Optional[int] = None


class Conversation(BaseModel):
    """Conversation schema"""
    conversation_id: str
    title: Optional[str] = None
    participants: List[str] = Field(default_factory=list)  # List of agent names or "user"
    messages: List[Message] = Field(default_factory=list)
    context_window: int = 4096
    max_context_messages: int = 50  # Max messages to keep in context
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    status: str = "active"  # active, archived, deleted


class ConversationService:
    """
    Conversation Storage Service
    
    Manages conversation storage, history tracking, and retrieval with MongoDB.
    Integrates with AutoGen multi-agent conversations.
    """
    
    def __init__(self, db_client: Optional[AsyncIOMotorClient] = None):
        self.db_client = db_client
        self.collection = None
        self.use_mongodb = False
        
        # JSON fallback path
        self.json_dir = Path.home() / '.cpas4' / 'conversations'
        self.json_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for active conversations
        self.cache: Dict[str, Conversation] = {}
        
        logger.info("Conversation Service initialized")
    
    async def initialize(self, db_client: Optional[AsyncIOMotorClient] = None):
        """Initialize database and load active conversations"""
        if db_client:
            self.db_client = db_client
        
        await self._init_mongodb()
        logger.info("Conversation Service initialized")
    
    async def _init_mongodb(self):
        """Initialize MongoDB collection"""
        if self.db_client:
            try:
                # Get database from client
                db = self.db_client.greta_pai
                self.collection = db.conversations
                
                # Create indexes
                await self.collection.create_index("conversation_id", unique=True)
                await self.collection.create_index("created_at")
                await self.collection.create_index("updated_at")
                await self.collection.create_index("participants")
                await self.collection.create_index("tags")
                await self.collection.create_index("status")
                
                # Text index for search
                await self.collection.create_index([("title", "text"), ("messages.content", "text")])
                
                self.use_mongodb = True
                logger.success("MongoDB collection initialized for conversations")
            except Exception as e:
                logger.error(f"MongoDB initialization failed: {e}")
                logger.warning("Falling back to JSON storage")
                self.use_mongodb = False
        else:
            logger.warning("No MongoDB client provided, using JSON storage")
            self.use_mongodb = False
    
    async def create_conversation(
        self,
        conversation_id: str,
        title: Optional[str] = None,
        participants: Optional[List[str]] = None,
        context_window: int = 4096,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Conversation:
        """Create a new conversation"""
        conversation = Conversation(
            conversation_id=conversation_id,
            title=title or f"Conversation {conversation_id}",
            participants=participants or [],
            context_window=context_window,
            metadata=metadata or {}
        )
        
        # Add to cache
        self.cache[conversation_id] = conversation
        
        # Save to MongoDB
        if self.use_mongodb and self.collection is not None:
            try:
                await self.collection.insert_one(conversation.model_dump())
            except Exception as e:
                logger.error(f"Failed to save conversation to MongoDB: {e}")
        
        # Save to JSON fallback
        await self._save_to_json(conversation)
        
        logger.success(f"Created conversation: {conversation_id}")
        return conversation
    
    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        tokens: Optional[int] = None
    ) -> bool:
        """Add a message to a conversation"""
        # Get or create conversation
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            conversation = await self.create_conversation(conversation_id)
        
        # Create message
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {},
            tokens=tokens
        )
        
        # Add message
        conversation.messages.append(message)
        conversation.updated_at = datetime.utcnow().isoformat()
        
        # Update cache
        self.cache[conversation_id] = conversation
        
        # Save to MongoDB
        if self.use_mongodb and self.collection is not None:
            try:
                await self.collection.update_one(
                    {"conversation_id": conversation_id},
                    {
                        "$push": {"messages": message.model_dump()},
                        "$set": {"updated_at": conversation.updated_at}
                    }
                )
            except Exception as e:
                logger.error(f"Failed to add message to MongoDB: {e}")
        
        # Save to JSON fallback
        await self._save_to_json(conversation)
        
        logger.info(f"Added message to conversation {conversation_id} (role: {role})")
        return True
    
    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID"""
        # Check cache first
        if conversation_id in self.cache:
            return self.cache[conversation_id]
        
        # Try MongoDB
        if self.use_mongodb and self.collection is not None:
            try:
                doc = await self.collection.find_one({"conversation_id": conversation_id})
                if doc:
                    doc.pop('_id', None)
                    conversation = Conversation(**doc)
                    self.cache[conversation_id] = conversation
                    return conversation
            except Exception as e:
                logger.error(f"Failed to get conversation from MongoDB: {e}")
        
        # Try JSON fallback
        json_path = self.json_dir / f"{conversation_id}.json"
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    conversation = Conversation(**data)
                    self.cache[conversation_id] = conversation
                    return conversation
            except Exception as e:
                logger.error(f"Failed to load conversation from JSON: {e}")
        
        return None
    
    async def get_conversation_history(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Message]:
        """Get conversation message history"""
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            return []
        
        messages = conversation.messages[offset:]
        if limit:
            messages = messages[:limit]
        
        return messages
    
    async def get_context_window(
        self,
        conversation_id: str,
        max_tokens: Optional[int] = None,
        max_messages: Optional[int] = None
    ) -> List[Message]:
        """
        Get conversation context within token/message limits.
        Implements sliding window for long conversations.
        """
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            return []
        
        max_tokens = max_tokens or conversation.context_window
        max_messages = max_messages or conversation.max_context_messages
        
        # Start from most recent messages
        messages = list(reversed(conversation.messages))
        context_messages = []
        total_tokens = 0
        
        for message in messages:
            # Estimate tokens if not provided (rough estimate: 1 token ≈ 4 chars)
            msg_tokens = message.tokens or len(message.content) // 4
            
            # Check if adding this message would exceed limits
            if len(context_messages) >= max_messages:
                break
            if total_tokens + msg_tokens > max_tokens:
                break
            
            context_messages.append(message)
            total_tokens += msg_tokens
        
        # Reverse back to chronological order
        return list(reversed(context_messages))
    
    async def search_conversations(
        self,
        query: Optional[str] = None,
        participants: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        status: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Conversation]:
        """Search conversations with filters"""
        conversations = []
        
        if self.use_mongodb and self.collection is not None:
            try:
                # Build query
                mongo_query = {}
                
                if query:
                    mongo_query["$text"] = {"$search": query}
                
                if participants:
                    mongo_query["participants"] = {"$in": participants}
                
                if tags:
                    mongo_query["tags"] = {"$in": tags}
                
                if status:
                    mongo_query["status"] = status
                
                if start_date:
                    mongo_query["created_at"] = {"$gte": start_date}
                
                if end_date:
                    if "created_at" in mongo_query:
                        mongo_query["created_at"]["$lte"] = end_date
                    else:
                        mongo_query["created_at"] = {"$lte": end_date}
                
                # Execute query
                cursor = self.collection.find(mongo_query).skip(offset).limit(limit)
                async for doc in cursor:
                    doc.pop('_id', None)
                    conversations.append(Conversation(**doc))
                
                return conversations
                
            except Exception as e:
                logger.error(f"MongoDB search failed: {e}")
        
        # Fallback to JSON search (simple filtering)
        for json_file in self.json_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    conversation = Conversation(**data)
                    
                    # Apply filters
                    if status and conversation.status != status:
                        continue
                    if participants and not any(p in conversation.participants for p in participants):
                        continue
                    if tags and not any(t in conversation.tags for t in tags):
                        continue
                    
                    conversations.append(conversation)
            except Exception as e:
                logger.error(f"Failed to load conversation from {json_file}: {e}")
        
        return conversations[offset:offset + limit]
    
    async def update_conversation(
        self,
        conversation_id: str,
        **updates
    ) -> bool:
        """Update conversation metadata"""
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            return False
        
        # Update fields
        for key, value in updates.items():
            if hasattr(conversation, key):
                setattr(conversation, key, value)
        
        conversation.updated_at = datetime.utcnow().isoformat()
        
        # Update cache
        self.cache[conversation_id] = conversation
        
        # Update MongoDB
        if self.use_mongodb and self.collection is not None:
            try:
                await self.collection.update_one(
                    {"conversation_id": conversation_id},
                    {"$set": updates}
                )
            except Exception as e:
                logger.error(f"Failed to update conversation in MongoDB: {e}")
        
        # Save to JSON
        await self._save_to_json(conversation)
        
        logger.info(f"Updated conversation: {conversation_id}")
        return True
    
    async def delete_conversation(
        self,
        conversation_id: str,
        hard_delete: bool = False
    ) -> bool:
        """Delete or archive a conversation"""
        if hard_delete:
            # Remove from cache
            self.cache.pop(conversation_id, None)
            
            # Delete from MongoDB
            if self.use_mongodb and self.collection is not None:
                try:
                    await self.collection.delete_one({"conversation_id": conversation_id})
                except Exception as e:
                    logger.error(f"Failed to delete conversation from MongoDB: {e}")
            
            # Delete JSON file
            json_path = self.json_dir / f"{conversation_id}.json"
            if json_path.exists():
                json_path.unlink()
            
            logger.info(f"Hard deleted conversation: {conversation_id}")
        else:
            # Soft delete (archive)
            await self.update_conversation(conversation_id, status="deleted")
            logger.info(f"Archived conversation: {conversation_id}")
        
        return True
    
    async def _save_to_json(self, conversation: Conversation):
        """Save conversation to JSON fallback"""
        try:
            json_path = self.json_dir / f"{conversation.conversation_id}.json"
            with open(json_path, 'w') as f:
                json.dump(conversation.model_dump(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save conversation to JSON: {e}")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        stats = {
            "total_conversations": 0,
            "active_conversations": 0,
            "total_messages": 0,
            "cached_conversations": len(self.cache)
        }
        
        if self.use_mongodb and self.collection is not None:
            try:
                stats["total_conversations"] = await self.collection.count_documents({})
                stats["active_conversations"] = await self.collection.count_documents({"status": "active"})
                
                # Count total messages
                pipeline = [
                    {"$project": {"message_count": {"$size": "$messages"}}},
                    {"$group": {"_id": None, "total": {"$sum": "$message_count"}}}
                ]
                result = await self.collection.aggregate(pipeline).to_list(1)
                if result:
                    stats["total_messages"] = result[0]["total"]
            except Exception as e:
                logger.error(f"Failed to get statistics from MongoDB: {e}")
        
        return stats


# Global instance
conversation_service = ConversationService()
