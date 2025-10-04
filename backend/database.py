"""
GRETA PAI Database Management
MongoDB integration with PAI philosophy - intelligent data orchestration
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from .config import get_settings

# Try to import motor, but provide fallback
try:
    from motor.motor_asyncio import AsyncIOMotorClient
    from motor.core import AgnosticDatabase, AgnosticCollection
    MOTOR_AVAILABLE = True
except ImportError:
    MOTOR_AVAILABLE = False
    logging.warning("⚠️ Motor not available - using file-based storage only")

logger = logging.getLogger(__name__)

class Database:
    """
    GRETA PAI Database - Intelligent data management
    The system's memory exceeds the sum of its parts
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.client = None
        self.db = None
        self.connected = False
        
    async def connect(self) -> bool:
        """Connect to MongoDB with intelligent fallback"""
        try:
            # Attempt MongoDB connection
            self.client = AsyncIOMotorClient(
                self.settings.mongodb_url,
                serverSelectionTimeoutMS=5000
            )
            
            # Test connection
            await self.client.admin.command('ping')
            
            self.db = self.client[self.settings.database_name]
            self.connected = True
            
            logger.info(f"✅ Connected to MongoDB: {self.settings.database_name}")
            
            # Initialize collections
            await self._initialize_collections()
            
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ MongoDB connection failed: {e}")
            logger.info("🔄 Falling back to file-based storage")
            
            # Fallback to file-based storage
            await self._initialize_file_storage()
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from database"""
        if self.client:
            self.client.close()
            self.connected = False
            logger.info("✅ Database disconnected")
    
    async def _initialize_collections(self):
        """Initialize MongoDB collections with PAI intelligence"""
        if not self.db:
            return
            
        collections = [
            "memory",           # System memory and learning
            "interactions",     # User interactions
            "agents",          # Agent states and configurations
            "context",         # PAI context management
            "learning",        # Learning data and insights
            "performance",     # Performance metrics
            "prompts",         # Prompt templates and history
            "models",          # AI model configurations
        ]
        
        for collection_name in collections:
            try:
                # Create collection if it doesn't exist
                if collection_name not in await self.db.list_collection_names():
                    await self.db.create_collection(collection_name)
                    logger.info(f"✅ Created collection: {collection_name}")
                
                # Create indexes for performance
                await self._create_indexes(collection_name)
                
            except Exception as e:
                logger.warning(f"⚠️ Could not create collection {collection_name}: {e}")
    
    async def _create_indexes(self, collection_name: str):
        """Create intelligent indexes for collections"""
        if not self.db:
            return
            
        collection = self.db[collection_name]
        
        # Collection-specific indexes
        index_configs = {
            "memory": [("timestamp", -1), ("agent_id", 1), ("context_type", 1)],
            "interactions": [("timestamp", -1), ("user_id", 1), ("session_id", 1)],
            "agents": [("agent_id", 1), ("status", 1), ("last_active", -1)],
            "context": [("context_id", 1), ("timestamp", -1), ("priority", -1)],
            "learning": [("timestamp", -1), ("learning_type", 1), ("confidence", -1)],
            "performance": [("timestamp", -1), ("metric_type", 1), ("agent_id", 1)],
            "prompts": [("prompt_id", 1), ("category", 1), ("usage_count", -1)],
            "models": [("model_id", 1), ("status", 1), ("last_used", -1)],
        }
        
        if collection_name in index_configs:
            for index_spec in index_configs[collection_name]:
                try:
                    await collection.create_index(index_spec)
                except Exception as e:
                    logger.debug(f"Index creation failed for {collection_name}: {e}")
    
    async def _initialize_file_storage(self):
        """Initialize file-based storage as fallback"""
        import os
        from pathlib import Path
        
        # Create data directory
        data_dir = Path(__file__).parent.parent / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        subdirs = ["memory", "interactions", "agents", "context", "learning"]
        for subdir in subdirs:
            (data_dir / subdir).mkdir(exist_ok=True)
        
        logger.info("✅ File-based storage initialized")
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        if self.connected and self.db:
            try:
                # Test database connection
                await self.db.command('ping')
                
                # Get collection stats
                collections = await self.db.list_collection_names()
                collection_stats = {}
                
                for collection_name in collections:
                    collection = self.db[collection_name]
                    count = await collection.count_documents({})
                    collection_stats[collection_name] = count
                
                return {
                    "status": "healthy",
                    "type": "mongodb",
                    "database": self.settings.database_name,
                    "collections": collections,
                    "stats": collection_stats,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                return {
                    "status": "error",
                    "type": "mongodb",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
        else:
            # File-based storage health check
            import os
            from pathlib import Path
            
            data_dir = Path(__file__).parent.parent / "data"
            if data_dir.exists():
                dirs = [d.name for d in data_dir.iterdir() if d.is_dir()]
                return {
                    "status": "healthy",
                    "type": "file-based",
                    "data_directory": str(data_dir),
                    "directories": dirs,
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                return {
                    "status": "error",
                    "type": "file-based",
                    "error": "Data directory not found",
                    "timestamp": datetime.utcnow().isoformat()
                }
    
    # Collection access methods
    def get_collection(self, name: str):
        """Get collection by name"""
        if self.db and self.connected:
            return self.db[name]
        return None
    
    async def insert_one(self, collection_name: str, document: Dict[str, Any]) -> Optional[str]:
        """Insert document into collection"""
        if self.connected and self.db:
            try:
                collection = self.db[collection_name]
                result = await collection.insert_one(document)
                return str(result.inserted_id)
            except Exception as e:
                logger.error(f"Insert error in {collection_name}: {e}")
                # Fallback to file storage
                return await self._insert_file_fallback(collection_name, document)
        else:
            return await self._insert_file_fallback(collection_name, document)
    
    async def find_one(self, collection_name: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find one document in collection"""
        if self.connected and self.db:
            try:
                collection = self.db[collection_name]
                document = await collection.find_one(query)
                if document:
                    document["_id"] = str(document["_id"])
                return document
            except Exception as e:
                logger.error(f"Find error in {collection_name}: {e}")
                # Fallback to file storage
                return await self._find_file_fallback(collection_name, query)
        else:
            return await self._find_file_fallback(collection_name, query)
    
    async def find_many(self, collection_name: str, query: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """Find multiple documents in collection"""
        if self.connected and self.db:
            try:
                collection = self.db[collection_name]
                cursor = collection.find(query).limit(limit)
                documents = []
                async for document in cursor:
                    document["_id"] = str(document["_id"])
                    documents.append(document)
                return documents
            except Exception as e:
                logger.error(f"Find many error in {collection_name}: {e}")
                # Fallback to file storage
                return await self._find_many_file_fallback(collection_name, query, limit)
        else:
            return await self._find_many_file_fallback(collection_name, query, limit)
    
    async def update_one(self, collection_name: str, query: Dict[str, Any], update: Dict[str, Any]) -> bool:
        """Update one document in collection"""
        if self.connected and self.db:
            try:
                collection = self.db[collection_name]
                result = await collection.update_one(query, update)
                return result.modified_count > 0
            except Exception as e:
                logger.error(f"Update error in {collection_name}: {e}")
                # Fallback to file storage
                return await self._update_file_fallback(collection_name, query, update)
        else:
            return await self._update_file_fallback(collection_name, query, update)
    
    async def delete_one(self, collection_name: str, query: Dict[str, Any]) -> bool:
        """Delete one document from collection"""
        if self.connected and self.db:
            try:
                collection = self.db[collection_name]
                result = await collection.delete_one(query)
                return result.deleted_count > 0
            except Exception as e:
                logger.error(f"Delete error in {collection_name}: {e}")
                # Fallback to file storage
                return await self._delete_file_fallback(collection_name, query)
        else:
            return await self._delete_file_fallback(collection_name, query)
    
    # File-based fallback methods
    async def _insert_file_fallback(self, collection_name: str, document: Dict[str, Any]) -> Optional[str]:
        """File-based insert fallback"""
        import json
        import uuid
        from pathlib import Path
        
        try:
            data_dir = Path(__file__).parent.parent / "data" / collection_name
            data_dir.mkdir(parents=True, exist_ok=True)
            
            doc_id = str(uuid.uuid4())
            document["_id"] = doc_id
            document["timestamp"] = datetime.utcnow().isoformat()
            
            file_path = data_dir / f"{doc_id}.json"
            with open(file_path, 'w') as f:
                json.dump(document, f, indent=2)
            
            return doc_id
        except Exception as e:
            logger.error(f"File insert error: {e}")
            return None
    
    async def _find_file_fallback(self, collection_name: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """File-based find fallback"""
        import json
        from pathlib import Path
        
        try:
            data_dir = Path(__file__).parent.parent / "data" / collection_name
            if not data_dir.exists():
                return None
            
            # Simple query matching (can be enhanced)
            for file_path in data_dir.glob("*.json"):
                with open(file_path, 'r') as f:
                    document = json.load(f)
                
                # Check if document matches query
                match = True
                for key, value in query.items():
                    if key not in document or document[key] != value:
                        match = False
                        break
                
                if match:
                    return document
            
            return None
        except Exception as e:
            logger.error(f"File find error: {e}")
            return None
    
    async def _find_many_file_fallback(self, collection_name: str, query: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """File-based find many fallback"""
        import json
        from pathlib import Path
        
        try:
            data_dir = Path(__file__).parent.parent / "data" / collection_name
            if not data_dir.exists():
                return []
            
            documents = []
            for file_path in data_dir.glob("*.json"):
                if len(documents) >= limit:
                    break
                    
                with open(file_path, 'r') as f:
                    document = json.load(f)
                
                # Check if document matches query
                match = True
                for key, value in query.items():
                    if key not in document or document[key] != value:
                        match = False
                        break
                
                if match:
                    documents.append(document)
            
            return documents
        except Exception as e:
            logger.error(f"File find many error: {e}")
            return []
    
    async def _update_file_fallback(self, collection_name: str, query: Dict[str, Any], update: Dict[str, Any]) -> bool:
        """File-based update fallback"""
        import json
        from pathlib import Path
        
        try:
            data_dir = Path(__file__).parent.parent / "data" / collection_name
            if not data_dir.exists():
                return False
            
            for file_path in data_dir.glob("*.json"):
                with open(file_path, 'r') as f:
                    document = json.load(f)
                
                # Check if document matches query
                match = True
                for key, value in query.items():
                    if key not in document or document[key] != value:
                        match = False
                        break
                
                if match:
                    # Update document
                    document.update(update.get("$set", {}))
                    document["timestamp"] = datetime.utcnow().isoformat()
                    
                    with open(file_path, 'w') as f:
                        json.dump(document, f, indent=2)
                    
                    return True
            
            return False
        except Exception as e:
            logger.error(f"File update error: {e}")
            return False
    
    async def _delete_file_fallback(self, collection_name: str, query: Dict[str, Any]) -> bool:
        """File-based delete fallback"""
        import json
        from pathlib import Path
        
        try:
            data_dir = Path(__file__).parent.parent / "data" / collection_name
            if not data_dir.exists():
                return False
            
            for file_path in data_dir.glob("*.json"):
                with open(file_path, 'r') as f:
                    document = json.load(f)
                
                # Check if document matches query
                match = True
                for key, value in query.items():
                    if key not in document or document[key] != value:
                        match = False
                        break
                
                if match:
                    file_path.unlink()
                    return True
            
            return False
        except Exception as e:
            logger.error(f"File delete error: {e}")
            return False

# Global database instance
database = Database()

async def get_database() -> Database:
    """Get the global database instance"""
    return database
