"""
🔍 Vector Search Service - Semantic Memory Storage and Retrieval
Inspired by ClaudeFlow's AgentDB for fast semantic search

Features:
- Semantic memory storage using ChromaDB
- Fast vector similarity search (<100ms target)
- Persistent memory across sessions
- Namespace-based organization
- Metadata filtering

Performance Target: <100ms query latency
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

# Try to import ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
    logger.info("✅ ChromaDB available")
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("❌ ChromaDB not available - install with: pip install chromadb==0.4.22")


class VectorSearchService:
    """
    Vector Search Service for semantic memory storage and retrieval
    
    Uses ChromaDB for efficient vector similarity search with persistent storage.
    Provides namespace-based organization for different types of memories.
    """
    
    def __init__(self, persist_directory: str = "./data/vector_db"):
        """
        Initialize Vector Search Service
        
        Args:
            persist_directory: Directory for persistent storage
        """
        self.persist_directory = persist_directory
        self.client = None
        self.collections: Dict[str, Any] = {}
        
        if not CHROMADB_AVAILABLE:
            logger.error("ChromaDB not available - Vector Search Service disabled")
            return
        
        try:
            # Initialize ChromaDB client with persistent storage (ChromaDB 1.3.0 API)
            self.client = chromadb.PersistentClient(path=persist_directory)

            # Use default embedding function (all-MiniLM-L6-v2)
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

            logger.info(f"✅ Vector Search Service initialized (storage: {persist_directory})")

        except Exception as e:
            logger.error(f"Failed to initialize Vector Search Service: {e}")
            self.client = None
    
    def _get_or_create_collection(self, namespace: str) -> Any:
        """
        Get or create a collection for a namespace
        
        Args:
            namespace: Namespace for the collection (e.g., 'conversations', 'code_snippets')
        
        Returns:
            ChromaDB collection
        """
        if not self.client:
            raise RuntimeError("ChromaDB client not initialized")
        
        if namespace in self.collections:
            return self.collections[namespace]
        
        try:
            collection = self.client.get_or_create_collection(
                name=namespace,
                embedding_function=self.embedding_function,
                metadata={"created_at": datetime.utcnow().isoformat()}
            )
            self.collections[namespace] = collection
            logger.debug(f"Collection '{namespace}' ready")
            return collection
            
        except Exception as e:
            logger.error(f"Failed to get/create collection '{namespace}': {e}")
            raise
    
    async def store_memory(
        self,
        content: str,
        namespace: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None
    ) -> str:
        """
        Store a memory in the vector database
        
        Args:
            content: Text content to store
            namespace: Namespace for organization
            metadata: Optional metadata (must be JSON-serializable)
            memory_id: Optional custom ID (auto-generated if not provided)
        
        Returns:
            Memory ID
        """
        if not self.client:
            raise RuntimeError("Vector Search Service not available")
        
        try:
            collection = self._get_or_create_collection(namespace)
            
            # Generate ID if not provided
            if not memory_id:
                memory_id = hashlib.sha256(
                    f"{content}{datetime.utcnow().isoformat()}".encode()
                ).hexdigest()[:16]
            
            # Prepare metadata
            meta = metadata or {}
            meta.update({
                "stored_at": datetime.utcnow().isoformat(),
                "namespace": namespace
            })
            
            # Store in ChromaDB
            collection.add(
                documents=[content],
                metadatas=[meta],
                ids=[memory_id]
            )
            
            logger.debug(f"Stored memory {memory_id} in namespace '{namespace}'")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise
    
    async def search_memories(
        self,
        query: str,
        namespace: str = "default",
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar memories using semantic search
        
        Args:
            query: Search query
            namespace: Namespace to search in
            limit: Maximum number of results
            metadata_filter: Optional metadata filter
        
        Returns:
            List of matching memories with scores
        """
        if not self.client:
            raise RuntimeError("Vector Search Service not available")
        
        try:
            collection = self._get_or_create_collection(namespace)
            
            # Perform similarity search
            results = collection.query(
                query_texts=[query],
                n_results=limit,
                where=metadata_filter
            )
            
            # Format results
            memories = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    memories.append({
                        "id": results['ids'][0][i],
                        "content": doc,
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "distance": results['distances'][0][i] if results['distances'] else None
                    })
            
            logger.debug(f"Found {len(memories)} memories for query in namespace '{namespace}'")
            return memories
            
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            raise
    
    async def get_memory(self, memory_id: str, namespace: str = "default") -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific memory by ID
        
        Args:
            memory_id: Memory ID
            namespace: Namespace
        
        Returns:
            Memory data or None if not found
        """
        if not self.client:
            raise RuntimeError("Vector Search Service not available")
        
        try:
            collection = self._get_or_create_collection(namespace)
            
            result = collection.get(
                ids=[memory_id],
                include=["documents", "metadatas"]
            )
            
            if result['documents'] and result['documents'][0]:
                return {
                    "id": memory_id,
                    "content": result['documents'][0],
                    "metadata": result['metadatas'][0] if result['metadatas'] else {}
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get memory {memory_id}: {e}")
            raise
    
    async def delete_memory(self, memory_id: str, namespace: str = "default") -> bool:
        """
        Delete a memory by ID
        
        Args:
            memory_id: Memory ID
            namespace: Namespace
        
        Returns:
            True if deleted, False if not found
        """
        if not self.client:
            raise RuntimeError("Vector Search Service not available")
        
        try:
            collection = self._get_or_create_collection(namespace)
            collection.delete(ids=[memory_id])
            logger.debug(f"Deleted memory {memory_id} from namespace '{namespace}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False
    
    async def get_collection_stats(self, namespace: str = "default") -> Dict[str, Any]:
        """
        Get statistics for a collection
        
        Args:
            namespace: Namespace
        
        Returns:
            Collection statistics
        """
        if not self.client:
            raise RuntimeError("Vector Search Service not available")
        
        try:
            collection = self._get_or_create_collection(namespace)
            count = collection.count()
            
            return {
                "namespace": namespace,
                "count": count,
                "embedding_function": "all-MiniLM-L6-v2"
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats for namespace '{namespace}': {e}")
            raise


# Global vector search service instance
vector_search_service = VectorSearchService() if CHROMADB_AVAILABLE else None

if vector_search_service:
    logger.info("✅ Vector Search Service ready")
else:
    logger.warning("⚠️ Vector Search Service not available - install ChromaDB")

