"""
GRETA PAI Database Management
MongoDB integration with PAI philosophy - intelligent data orchestration

Phase 2, Task 2.1: Database Integrity Improvements
- Transaction support for atomic operations
- Standardized datetime operations
- Enhanced data validation
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from contextlib import asynccontextmanager
from config import get_settings

# Try to import motor, but provide fallback
try:
    from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorClientSession
    from motor.core import AgnosticDatabase, AgnosticCollection
    MOTOR_AVAILABLE = True
except ImportError:
    MOTOR_AVAILABLE = False
    AsyncIOMotorClientSession = None
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
            # Attempt MongoDB connection with connection pooling (Phase 2 Task 2.3)
            # Use asyncio.wait_for to add a timeout to the entire connection process
            async def _connect_mongodb():
                self.client = AsyncIOMotorClient(
                    self.settings.mongodb_url,
                    serverSelectionTimeoutMS=3000,
                    maxPoolSize=50,          # Maximum number of connections in the pool
                    minPoolSize=10,          # Minimum number of connections to maintain
                    maxIdleTimeMS=30000,     # Close connections idle for 30 seconds
                    waitQueueTimeoutMS=5000  # Wait up to 5 seconds for a connection
                )

                # Test connection
                await self.client.admin.command('ping')

                self.db = self.client[self.settings.database_name]
                self.connected = True

                logger.info(f"✅ Connected to MongoDB: {self.settings.database_name}")

                # Initialize collections
                await self._initialize_collections()

                return True

            # Add a 5-second timeout to the entire connection process
            return await asyncio.wait_for(_connect_mongodb(), timeout=5)

        except asyncio.TimeoutError:
            logger.warning("⚠️ MongoDB connection timed out")
            logger.info("🔄 Falling back to file-based storage")

            # Fallback to file-based storage
            await self._initialize_file_storage()
            self.connected = False
            return False
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
        if self.db is None:
            return

        collections = [
            # Core system collections
            "memory",           # System memory and learning
            "interactions",     # User interactions
            "agents",          # Agent states and configurations
            "context",         # PAI context management
            "learning",        # Learning data and insights
            "performance",     # Performance metrics
            "prompts",         # Prompt templates and history
            "models",          # AI model configurations

            # Business collections
            "crm_contacts",    # CRM contacts
            "crm_companies",   # CRM companies
            "crm_deals",       # CRM deals
            "crm_activities",  # CRM activities
            "ats_jobs",        # ATS jobs
            "ats_candidates",  # ATS candidates
            "ats_applications", # ATS applications
            "ats_interviews",  # ATS interviews
            "lead_campaigns",  # Lead generation campaigns
            "leads",           # Generated leads
            "lead_research",   # Lead research reports

            # Agent system collections
            "master_agent_state",  # Master agent state
            "workflows",           # Agent workflows
            "modules",             # System modules
            "learning_data",       # Learning data
            "system_metrics",      # System metrics
            "deployments",         # Agent deployments
            "applications",        # Applications

            # Feature collections
            "training_courses",    # Training courses
            "course_enrollments",  # Course enrollments
            "course_assessments",  # Course assessments
            "course_certificates", # Course certificates
            "builder_projects",    # Builder projects
            "builder_builds",      # Builder builds
            "builder_tests",       # Builder tests
            "managed_agents",      # Managed agents
            "agent_versions",      # Agent versions
            "ab_tests",            # A/B tests
            "osint_investigations", # OSINT investigations
            "osint_pipelines",     # OSINT pipelines
            "functional_memories", # Functional memories
            "multimodal_jobs",     # Multimodal jobs
            "predictions",         # Predictions
            "analytics_datasets",  # Analytics datasets
            "mcp_servers",         # MCP servers
            "knowledge_documents", # Knowledge documents
            "knowledge_graphs",    # Knowledge graphs
            "pai_agents",          # PAI agents
            "pai_deployments",     # PAI deployments

            # OAuth and workspace collections (Phase 2.2)
            "oauth_tokens",        # OAuth tokens for users
            "oauth_states",        # OAuth authorization states
            "workspace_credentials", # Multi-workspace OAuth credentials
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
        """Create intelligent indexes for collections - ENHANCED with comprehensive coverage"""
        if self.db is None:
            return

        collection = self.db[collection_name]

        # Collection-specific indexes with compound indexes for common query patterns
        index_configs = {
            # Core system collections
            "memory": [
                ("timestamp", -1),
                ("agent_id", 1),
                ("context_type", 1),
                [("agent_id", 1), ("timestamp", -1)],  # Compound index
            ],
            "interactions": [
                ("timestamp", -1),
                ("user_id", 1),
                ("session_id", 1),
                [("user_id", 1), ("timestamp", -1)],  # Compound index
            ],
            "agents": [
                ("agent_id", 1),
                ("status", 1),
                ("last_active", -1),
                ("created_at", -1),
                [("status", 1), ("last_active", -1)],  # Compound index
            ],
            "context": [
                ("context_id", 1),
                ("timestamp", -1),
                ("priority", -1),
            ],
            "learning": [
                ("timestamp", -1),
                ("learning_type", 1),
                ("confidence", -1),
            ],
            "performance": [
                ("timestamp", -1),
                ("metric_type", 1),
                ("agent_id", 1),
                [("agent_id", 1), ("timestamp", -1)],  # Compound index
            ],
            "prompts": [
                ("prompt_id", 1),
                ("category", 1),
                ("usage_count", -1),
            ],
            "models": [
                ("model_id", 1),
                ("status", 1),
                ("last_used", -1),
            ],

            # CRM collections
            "crm_contacts": [
                ("contact_id", 1),
                ("user_id", 1),
                ("email", 1),
                ("company", 1),
                ("status", 1),
                ("created_at", -1),
                ("updated_at", -1),
                [("user_id", 1), ("created_at", -1)],  # Compound index
                [("status", 1), ("created_at", -1)],   # Compound index
            ],
            "crm_companies": [
                ("company_id", 1),
                ("user_id", 1),
                ("name", 1),
                ("industry", 1),
                ("created_at", -1),
                ("updated_at", -1),
                [("user_id", 1), ("created_at", -1)],  # Compound index
            ],
            "crm_deals": [
                ("deal_id", 1),
                ("user_id", 1),
                ("contact_id", 1),
                ("company_id", 1),
                ("stage", 1),
                ("created_at", -1),
                ("updated_at", -1),
                ("expected_close_date", -1),
                [("user_id", 1), ("stage", 1)],        # Compound index
                [("stage", 1), ("created_at", -1)],    # Compound index
            ],
            "crm_activities": [
                ("activity_id", 1),
                ("user_id", 1),
                ("contact_id", 1),
                ("deal_id", 1),
                ("activity_type", 1),
                ("created_at", -1),
                [("contact_id", 1), ("created_at", -1)],  # Compound index
                [("deal_id", 1), ("created_at", -1)],     # Compound index
            ],

            # ATS collections
            "ats_jobs": [
                ("job_id", 1),
                ("user_id", 1),
                ("status", 1),
                ("job_type", 1),
                ("department", 1),
                ("created_at", -1),
                ("updated_at", -1),
                [("user_id", 1), ("status", 1)],       # Compound index
                [("status", 1), ("created_at", -1)],   # Compound index
            ],
            "ats_candidates": [
                ("candidate_id", 1),
                ("user_id", 1),
                ("email", 1),
                ("status", 1),
                ("created_at", -1),
                ("updated_at", -1),
                ("last_activity", -1),
                [("user_id", 1), ("status", 1)],       # Compound index
                [("status", 1), ("created_at", -1)],   # Compound index
            ],
            "ats_applications": [
                ("application_id", 1),
                ("user_id", 1),
                ("job_id", 1),
                ("candidate_id", 1),
                ("status", 1),
                ("created_at", -1),
                ("updated_at", -1),
                [("job_id", 1), ("status", 1)],        # Compound index
                [("candidate_id", 1), ("created_at", -1)],  # Compound index
            ],
            "ats_interviews": [
                ("interview_id", 1),
                ("user_id", 1),
                ("application_id", 1),
                ("candidate_id", 1),
                ("status", 1),
                ("scheduled_at", -1),
                ("created_at", -1),
                [("application_id", 1), ("scheduled_at", -1)],  # Compound index
            ],

            # Lead generation collections
            "lead_campaigns": [
                ("campaign_id", 1),
                ("user_id", 1),
                ("status", 1),
                ("created_at", -1),
                ("updated_at", -1),
                [("user_id", 1), ("status", 1)],       # Compound index
            ],
            "leads": [
                ("lead_id", 1),
                ("user_id", 1),
                ("campaign_id", 1),
                ("status", 1),
                ("classification", 1),
                ("created_at", -1),
                ("updated_at", -1),
                [("campaign_id", 1), ("status", 1)],   # Compound index
                [("user_id", 1), ("created_at", -1)],  # Compound index
            ],
            "lead_research": [
                ("research_id", 1),
                ("lead_id", 1),
                ("researched_at", -1),
                [("lead_id", 1), ("researched_at", -1)],  # Compound index
            ],

            # Agent system collections
            "master_agent_state": [
                ("agent_id", 1),
                ("timestamp", -1),
            ],
            "workflows": [
                ("workflow_id", 1),
                ("agent_id", 1),
                ("status", 1),
                ("created_at", -1),
                [("agent_id", 1), ("status", 1)],      # Compound index
            ],
            "modules": [
                ("module_id", 1),
                ("status", 1),
                ("created_at", -1),
            ],
            "learning_data": [
                ("data_id", 1),
                ("agent_id", 1),
                ("timestamp", -1),
                [("agent_id", 1), ("timestamp", -1)],  # Compound index
            ],
            "system_metrics": [
                ("metric_id", 1),
                ("timestamp", -1),
                ("metric_type", 1),
            ],
            "deployments": [
                ("deployment_id", 1),
                ("agent_id", 1),
                ("status", 1),
                ("created_at", -1),
                [("agent_id", 1), ("created_at", -1)],  # Compound index
            ],
            "applications": [
                ("application_id", 1),
                ("agent_id", 1),
                ("status", 1),
                ("created_at", -1),
            ],

            # Feature collections
            "training_courses": [
                ("course_id", 1),
                ("created_at", -1),
            ],
            "course_enrollments": [
                ("enrollment_id", 1),
                ("user_id", 1),
                ("course_id", 1),
                ("created_at", -1),
                [("user_id", 1), ("course_id", 1)],    # Compound index
            ],
            "managed_agents": [
                ("agent_id", 1),
                ("user_id", 1),
                ("status", 1),
                ("created_at", -1),
                [("user_id", 1), ("status", 1)],       # Compound index
            ],
            "osint_investigations": [
                ("investigation_id", 1),
                ("user_id", 1),
                ("status", 1),
                ("created_at", -1),
                [("user_id", 1), ("created_at", -1)],  # Compound index
            ],
            "mcp_servers": [
                ("server_id", 1),
                ("status", 1),
                ("created_at", -1),
            ],
            "knowledge_documents": [
                ("document_id", 1),
                ("user_id", 1),
                ("created_at", -1),
                [("user_id", 1), ("created_at", -1)],  # Compound index
            ],

            # OAuth and workspace collections (Phase 2.2)
            "oauth_tokens": [
                ("token_id", 1),
                ("user_id", 1),
                ("provider", 1),
                ("account_email", 1),
                ("expires_at", -1),
                ("revoked", 1),
                ("created_at", -1),
                [("user_id", 1), ("provider", 1)],     # Compound index
                [("user_id", 1), ("is_primary", -1)],  # Compound index
                [("provider", 1), ("revoked", 1)],     # Compound index
            ],
            "oauth_states": [
                ("state", 1),
                ("user_id", 1),
                ("provider", 1),
                ("created_at", -1),
                ("expires_at", -1),
                [("state", 1), ("provider", 1)],       # Compound index
            ],
            "workspace_credentials": [
                ("credentials_id", 1),
                ("workspace_id", 1),
                ("owner_user_id", 1),
                ("workspace_domain", 1),
                ("is_active", 1),
                ("is_default", 1),
                ("created_at", -1),
                ("last_used_at", -1),
                [("owner_user_id", 1), ("is_active", 1)],      # Compound index
                [("owner_user_id", 1), ("is_default", -1)],    # Compound index
                [("workspace_id", 1), ("is_active", 1)],       # Compound index
                [("authorized_user_ids", 1), ("is_active", 1)], # Compound index for array field
            ],
        }

        if collection_name in index_configs:
            for index_spec in index_configs[collection_name]:
                try:
                    await collection.create_index(index_spec)
                except Exception as e:
                    logger.debug(f"Index creation failed for {collection_name}.{index_spec}: {e}")
    
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
        if self.connected and self.db is not None:
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

    # ========================================================================
    # TRANSACTION SUPPORT (Phase 2, Task 2.1)
    # ========================================================================

    @asynccontextmanager
    async def transaction(self):
        """
        Context manager for database transactions with automatic rollback on failure.

        Usage:
            async with db.transaction() as session:
                await db.insert_one("collection1", doc1, session=session)
                await db.update_one("collection2", query, update, session=session)
                # If any operation fails, all changes are rolled back

        Note: Transactions require MongoDB replica set or sharded cluster.
        Falls back to non-transactional operations if not available.
        """
        if not self.connected or self.db is None or self.client is None:
            # Fallback: No transaction support for file-based storage
            logger.warning("⚠️ Transactions not supported in file-based mode")
            yield None
            return

        session = None
        try:
            # Start a client session
            async with await self.client.start_session() as session:
                # Start transaction
                async with session.start_transaction():
                    logger.debug("✅ Transaction started")
                    yield session
                    # Transaction commits automatically if no exception
                    logger.debug("✅ Transaction committed")
        except Exception as e:
            # Transaction rolls back automatically on exception
            logger.error(f"❌ Transaction rolled back due to error: {e}")
            raise

    async def execute_transaction(self, operations: Callable[[Optional[AsyncIOMotorClientSession]], Any]) -> Any:
        """
        Execute multiple database operations as an atomic transaction.

        Args:
            operations: Async function that takes a session and performs database operations

        Returns:
            Result of the operations function

        Example:
            async def my_operations(session):
                await db.insert_one("contacts", contact_data, session=session)
                await db.update_one("companies", {"company_id": cid}, {"$inc": {"contact_count": 1}}, session=session)
                return {"success": True}

            result = await db.execute_transaction(my_operations)
        """
        if not self.connected or self.db is None or self.client is None:
            # Fallback: Execute without transaction
            logger.warning("⚠️ Executing operations without transaction support")
            return await operations(None)

        try:
            async with await self.client.start_session() as session:
                async with session.start_transaction():
                    result = await operations(session)
                    logger.debug("✅ Transaction executed successfully")
                    return result
        except Exception as e:
            logger.error(f"❌ Transaction failed and rolled back: {e}")
            raise

    # ========================================================================
    # DATETIME UTILITIES (Phase 2, Task 2.1)
    # ========================================================================

    @staticmethod
    def utcnow() -> datetime:
        """
        Standardized UTC datetime for all database operations.
        Use this instead of datetime.utcnow() directly for consistency.
        """
        return datetime.utcnow()

    @staticmethod
    def to_iso(dt: Optional[datetime]) -> Optional[str]:
        """Convert datetime to ISO string for database storage"""
        return dt.isoformat() if dt else None

    @staticmethod
    def from_iso(iso_str: Optional[str]) -> Optional[datetime]:
        """Convert ISO string from database to datetime"""
        if not iso_str:
            return None
        try:
            return datetime.fromisoformat(iso_str)
        except (ValueError, TypeError):
            return None

    # Collection access methods
    def get_collection(self, name: str):
        """Get collection by name"""
        if self.db is not None and self.connected:
            return self.db[name]
        return None
    
    async def insert_one(
        self,
        collection_name: str,
        document: Dict[str, Any],
        session: Optional[AsyncIOMotorClientSession] = None
    ) -> Optional[str]:
        """
        Insert document into collection.

        Args:
            collection_name: Name of the collection
            document: Document to insert
            session: Optional session for transaction support

        Returns:
            Inserted document ID
        """
        if self.connected and self.db is not None:
            try:
                collection = self.db[collection_name]
                result = await collection.insert_one(document, session=session)
                return str(result.inserted_id)
            except Exception as e:
                logger.error(f"Insert error in {collection_name}: {e}")
                # Fallback to file storage (only if not in transaction)
                if session is None:
                    return await self._insert_file_fallback(collection_name, document)
                raise
        else:
            if session is not None:
                raise RuntimeError("Transactions not supported in file-based mode")
            return await self._insert_file_fallback(collection_name, document)
    
    async def find_one(self, collection_name: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find one document in collection"""
        if self.connected and self.db is not None:
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
    
    async def find_many(self, collection_name: str, query: Dict[str, Any], limit: int = 100, sort: Optional[List[tuple]] = None) -> List[Dict[str, Any]]:
        """Find multiple documents in collection with optional sorting"""
        if self.connected and self.db is not None:
            try:
                collection = self.db[collection_name]
                cursor = collection.find(query)

                # Apply sorting if provided
                if sort:
                    cursor = cursor.sort(sort)

                cursor = cursor.limit(limit)
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
    
    async def update_one(
        self,
        collection_name: str,
        query: Dict[str, Any],
        update: Dict[str, Any],
        session: Optional[AsyncIOMotorClientSession] = None
    ) -> bool:
        """
        Update one document in collection.

        Args:
            collection_name: Name of the collection
            query: Query to find document
            update: Update operations
            session: Optional session for transaction support

        Returns:
            True if document was modified
        """
        if self.connected and self.db is not None:
            try:
                collection = self.db[collection_name]
                result = await collection.update_one(query, update, session=session)
                return result.modified_count > 0
            except Exception as e:
                logger.error(f"Update error in {collection_name}: {e}")
                # Fallback to file storage (only if not in transaction)
                if session is None:
                    return await self._update_file_fallback(collection_name, query, update)
                raise
        else:
            if session is not None:
                raise RuntimeError("Transactions not supported in file-based mode")
            return await self._update_file_fallback(collection_name, query, update)

    async def delete_one(
        self,
        collection_name: str,
        query: Dict[str, Any],
        session: Optional[AsyncIOMotorClientSession] = None
    ) -> bool:
        """
        Delete one document from collection.

        Args:
            collection_name: Name of the collection
            query: Query to find document
            session: Optional session for transaction support

        Returns:
            True if document was deleted
        """
        if self.connected and self.db is not None:
            try:
                collection = self.db[collection_name]
                result = await collection.delete_one(query, session=session)
                return result.deleted_count > 0
            except Exception as e:
                logger.error(f"Delete error in {collection_name}: {e}")
                # Fallback to file storage (only if not in transaction)
                if session is None:
                    return await self._delete_file_fallback(collection_name, query)
                raise
        else:
            if session is not None:
                raise RuntimeError("Transactions not supported in file-based mode")
            return await self._delete_file_fallback(collection_name, query)

    async def delete_many(
        self,
        collection_name: str,
        query: Dict[str, Any],
        session: Optional[AsyncIOMotorClientSession] = None
    ) -> int:
        """
        Delete multiple documents from collection.

        Args:
            collection_name: Name of the collection
            query: Query to find documents
            session: Optional session for transaction support

        Returns:
            Number of documents deleted
        """
        if self.connected and self.db is not None:
            try:
                collection = self.db[collection_name]
                result = await collection.delete_many(query, session=session)
                return result.deleted_count
            except Exception as e:
                logger.error(f"Delete many error in {collection_name}: {e}")
                # Fallback to file storage (only if not in transaction)
                if session is None:
                    return await self._delete_many_file_fallback(collection_name, query)
                raise
        else:
            if session is not None:
                raise RuntimeError("Transactions not supported in file-based mode")
            return await self._delete_many_file_fallback(collection_name, query)

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

    async def _delete_many_file_fallback(self, collection_name: str, query: Dict[str, Any]) -> int:
        """File-based delete many fallback"""
        import json
        from pathlib import Path

        try:
            data_dir = Path(__file__).parent.parent / "data" / collection_name
            if not data_dir.exists():
                return 0

            deleted_count = 0
            for file_path in data_dir.glob("*.json"):
                with open(file_path, 'r') as f:
                    document = json.load(f)

                # Check if document matches query (support nested keys like "metadata.test")
                match = True
                for key, value in query.items():
                    if '.' in key:
                        # Handle nested keys
                        keys = key.split('.')
                        current = document
                        for k in keys:
                            if isinstance(current, dict) and k in current:
                                current = current[k]
                            else:
                                match = False
                                break
                        if match and current != value:
                            match = False
                    else:
                        if key not in document or document[key] != value:
                            match = False
                            break

                if match:
                    file_path.unlink()
                    deleted_count += 1

            return deleted_count
        except Exception as e:
            logger.error(f"File delete many error: {e}")
            return 0

# Global database instance
database = Database()

async def get_database() -> Database:
    """Get the global database instance"""
    return database

async def health_check() -> Dict[str, Any]:
    """
    Standalone health check function for the global database instance
    Used by health_router.py
    """
    return await database.health_check()
