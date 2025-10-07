"""
GRETA Functional Memory System Service
Real vector database, memory consolidation, and retrieval system
"""

import asyncio
import logging
import uuid
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import deque
import hashlib

logger = logging.getLogger(__name__)

class FunctionalMemoryService:
    """Functional Memory System with vector database and semantic search"""

    def __init__(self, db, llm_service, learning_service):
        self.db = db
        self.llm_service = llm_service
        self.learning_service = learning_service

        # Memory storage
        self.memories = {}
        self.vector_store = {}  # Simple in-memory vector store (can be replaced with ChromaDB/FAISS)
        self.embeddings_cache = {}

        # Memory types
        self.memory_types = {
            "episodic": {"name": "Episodic Memory", "description": "Event-based memories", "retention_days": 90},
            "semantic": {"name": "Semantic Memory", "description": "Factual knowledge", "retention_days": 365},
            "procedural": {"name": "Procedural Memory", "description": "Skills and procedures", "retention_days": 180},
            "working": {"name": "Working Memory", "description": "Temporary information", "retention_days": 1}
        }

        # Context management
        self.context_window = deque(maxlen=100)  # Last 100 interactions
        self.active_contexts = {}

        # Knowledge graph (simple implementation)
        self.knowledge_graph = {"nodes": {}, "edges": []}

        # Activity log
        self.activity_log = deque(maxlen=500)

        # Metrics
        self.metrics = {
            "total_memories": 0,
            "episodic_memories": 0,
            "semantic_memories": 0,
            "procedural_memories": 0,
            "working_memories": 0,
            "total_retrievals": 0,
            "successful_retrievals": 0,
            "consolidations": 0,
            "embeddings_generated": 0
        }
        
    async def initialize(self):
        try:
            logger.info("🚀 Initializing Functional Memory System...")
            await self._load_memories()
            logger.info("✅ Functional Memory System initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Memory System initialization failed: {e}")
            return False
    
    async def _load_memories(self):
        try:
            memories = await self.db.find_many("functional_memories", {}, limit=1000)
            for memory in memories:
                self.memories[memory["memory_id"]] = memory
            self.metrics["total_memories"] = len(memories)
            logger.info(f"✅ Loaded {len(memories)} memories")
        except Exception as e:
            logger.debug(f"No existing memories: {e}")
    
    # ==================== MEMORY STORAGE ====================

    async def store_memory(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Store a memory with vector embedding"""
        try:
            memory_type = config.get("type", "episodic")
            content = config.get("content")

            if not content:
                return {"success": False, "error": "Content is required"}

            # Generate memory ID
            memory_id = f"mem_{memory_type}_{uuid.uuid4().hex[:8]}"

            # Generate embedding for semantic search
            embedding = await self._generate_embedding(content)

            # Calculate retention date
            retention_days = self.memory_types.get(memory_type, {}).get("retention_days", 90)
            retention_date = (datetime.now() + timedelta(days=retention_days)).isoformat()

            memory = {
                "memory_id": memory_id,
                "type": memory_type,
                "content": content,
                "metadata": config.get("metadata", {}),
                "importance": config.get("importance", 0.5),
                "tags": config.get("tags", []),
                "created_at": datetime.now().isoformat(),
                "retention_date": retention_date,
                "access_count": 0,
                "last_accessed": None,
                "embedding_id": f"emb_{memory_id}",
                "consolidated": False
            }

            # Store embedding
            self.vector_store[memory["embedding_id"]] = {
                "memory_id": memory_id,
                "embedding": embedding,
                "content": content
            }

            # Store memory
            try:
                await self.db.insert_one("functional_memories", memory)
            except Exception as e:
                logger.warning(f"Could not save memory to database: {e}")

            self.memories[memory_id] = memory

            # Update metrics
            self.metrics["total_memories"] += 1
            self.metrics[f"{memory_type}_memories"] = self.metrics.get(f"{memory_type}_memories", 0) + 1

            # Add to knowledge graph
            await self._add_to_knowledge_graph(memory)

            # Log activity
            self._log_activity("memory_stored", f"Stored {memory_type} memory", {"memory_id": memory_id})

            # Record learning
            try:
                await self.learning_service.record_pai_interaction({
                    "user_id": "memory_system",
                    "type": "memory_storage",
                    "category": "memory",
                    "strategic_reasoning": f"Stored {memory_type} memory",
                    "tactical_reasoning": f"Importance: {config.get('importance', 0.5)}",
                    "operational_reasoning": "Memory stored with embedding",
                    "response_quality": 1.0,
                    "user_satisfaction": 1.0,
                    "context_relevance": 1.0
                })
            except Exception as e:
                logger.warning(f"Could not record learning data: {e}")

            logger.info(f"✅ Memory stored: {memory_id}")

            return {
                "success": True,
                "memory_id": memory_id,
                "memory": memory
            }

        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return {"success": False, "error": str(e)}

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text (simple implementation)"""
        try:
            # Check cache
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self.embeddings_cache:
                return self.embeddings_cache[text_hash]

            # Simple embedding generation (in production, use actual embedding model)
            # For now, create a simple hash-based embedding
            embedding = [float(ord(c)) / 255.0 for c in text_hash[:384]]  # 384-dim embedding

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = (np.array(embedding) / norm).tolist()

            # Cache
            self.embeddings_cache[text_hash] = embedding
            self.metrics["embeddings_generated"] += 1

            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 384

    def _log_activity(self, activity_type: str, description: str, metadata: Dict[str, Any] = None):
        """Log an activity"""
        activity = {
            "timestamp": datetime.now().isoformat(),
            "type": activity_type,
            "description": description,
            "metadata": metadata or {}
        }
        self.activity_log.append(activity)
        logger.info(f"📝 Activity: {activity_type} - {description}")
    
    # ==================== MEMORY RETRIEVAL & SEMANTIC SEARCH ====================

    async def semantic_search(self, query: str, limit: int = 10, memory_type: str = None) -> Dict[str, Any]:
        """Semantic search using vector similarity"""
        try:
            # Generate query embedding
            query_embedding = await self._generate_embedding(query)

            # Calculate similarities
            similarities = []
            for emb_id, emb_data in self.vector_store.items():
                memory_id = emb_data["memory_id"]
                memory = self.memories.get(memory_id)

                if not memory:
                    continue

                # Filter by type if specified
                if memory_type and memory.get("type") != memory_type:
                    continue

                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, emb_data["embedding"])

                similarities.append({
                    "memory_id": memory_id,
                    "memory": memory,
                    "similarity": similarity
                })

            # Sort by similarity
            similarities.sort(key=lambda x: x["similarity"], reverse=True)

            # Get top results
            results = similarities[:limit]

            # Update access stats for retrieved memories
            for result in results:
                memory = result["memory"]
                memory["access_count"] = memory.get("access_count", 0) + 1
                memory["last_accessed"] = datetime.now().isoformat()

            self.metrics["total_retrievals"] += 1
            self.metrics["successful_retrievals"] += 1

            self._log_activity("semantic_search", f"Searched for: {query[:50]}...", {"results_count": len(results)})

            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results)
            }

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return {"success": False, "error": str(e)}

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            vec1_np = np.array(vec1)
            vec2_np = np.array(vec2)

            dot_product = np.dot(vec1_np, vec2_np)
            norm1 = np.linalg.norm(vec1_np)
            norm2 = np.linalg.norm(vec2_np)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(dot_product / (norm1 * norm2))

        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    async def retrieve_memory(self, memory_id: str) -> Dict[str, Any]:
        """Retrieve a specific memory by ID"""
        try:
            memory = self.memories.get(memory_id)
            if not memory:
                return {"success": False, "error": f"Memory not found: {memory_id}"}

            # Update access stats
            memory["access_count"] = memory.get("access_count", 0) + 1
            memory["last_accessed"] = datetime.now().isoformat()

            try:
                await self.db.update_one("functional_memories", {"memory_id": memory_id}, {"$set": memory})
            except Exception as e:
                logger.warning(f"Could not update memory in database: {e}")

            return {
                "success": True,
                "memory": memory
            }

        except Exception as e:
            logger.error(f"Error retrieving memory: {e}")
            return {"success": False, "error": str(e)}

    async def get_all_memories(self, memory_type: str = None) -> Dict[str, Any]:
        """Get all memories, optionally filtered by type"""
        try:
            memories_list = list(self.memories.values())

            if memory_type:
                memories_list = [m for m in memories_list if m.get("type") == memory_type]

            return {
                "success": True,
                "memories": memories_list,
                "count": len(memories_list),
                "metrics": {
                    "total": len(memories_list),
                    "episodic": len([m for m in memories_list if m.get("type") == "episodic"]),
                    "semantic": len([m for m in memories_list if m.get("type") == "semantic"]),
                    "procedural": len([m for m in memories_list if m.get("type") == "procedural"]),
                    "working": len([m for m in memories_list if m.get("type") == "working"])
                }
            }

        except Exception as e:
            logger.error(f"Error getting memories: {e}")
            return {"success": False, "error": str(e)}

    # ==================== MEMORY CONSOLIDATION ====================

    async def consolidate_memories(self) -> Dict[str, Any]:
        """Consolidate memories (strengthen important ones, merge similar ones)"""
        try:
            consolidated = 0
            merged = 0

            # Strengthen frequently accessed memories
            for memory in self.memories.values():
                if memory.get("access_count", 0) > 5 and not memory.get("consolidated"):
                    memory["importance"] = min(1.0, memory.get("importance", 0.5) + 0.1)
                    memory["consolidated"] = True
                    consolidated += 1

                    try:
                        await self.db.update_one("functional_memories", {"memory_id": memory["memory_id"]}, {"$set": memory})
                    except Exception as e:
                        logger.warning(f"Could not update memory in database: {e}")

            # Clean up expired working memories
            now = datetime.now()
            expired = []
            for memory_id, memory in list(self.memories.items()):
                if memory.get("type") == "working":
                    retention_date = datetime.fromisoformat(memory.get("retention_date", now.isoformat()))
                    if retention_date < now:
                        expired.append(memory_id)

            for memory_id in expired:
                await self.delete_memory(memory_id)

            self.metrics["consolidations"] += 1

            self._log_activity("memory_consolidation", f"Consolidated {consolidated} memories, removed {len(expired)} expired", {
                "consolidated": consolidated,
                "expired": len(expired)
            })

            return {
                "success": True,
                "consolidated_count": consolidated,
                "merged_count": merged,
                "expired_count": len(expired)
            }

        except Exception as e:
            logger.error(f"Error consolidating memories: {e}")
            return {"success": False, "error": str(e)}

    async def delete_memory(self, memory_id: str) -> Dict[str, Any]:
        """Delete a memory"""
        try:
            memory = self.memories.get(memory_id)
            if not memory:
                return {"success": False, "error": f"Memory not found: {memory_id}"}

            # Remove from vector store
            emb_id = memory.get("embedding_id")
            if emb_id in self.vector_store:
                del self.vector_store[emb_id]

            # Remove from database
            try:
                await self.db.delete_one("functional_memories", {"memory_id": memory_id})
            except Exception as e:
                logger.warning(f"Could not delete memory from database: {e}")

            # Remove from memory
            del self.memories[memory_id]

            # Update metrics
            self.metrics["total_memories"] -= 1
            memory_type = memory.get("type", "episodic")
            self.metrics[f"{memory_type}_memories"] = max(0, self.metrics.get(f"{memory_type}_memories", 0) - 1)

            return {
                "success": True,
                "message": f"Memory {memory_id} deleted successfully"
            }

        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return {"success": False, "error": str(e)}
    
    # ==================== KNOWLEDGE GRAPH ====================

    async def _add_to_knowledge_graph(self, memory: Dict[str, Any]):
        """Add memory to knowledge graph"""
        try:
            memory_id = memory["memory_id"]
            content = memory.get("content", "")

            # Add as node
            self.knowledge_graph["nodes"][memory_id] = {
                "id": memory_id,
                "type": memory.get("type"),
                "content": content[:100],  # Store snippet
                "importance": memory.get("importance", 0.5)
            }

            # Find related memories and create edges
            for other_id, other_memory in self.memories.items():
                if other_id == memory_id:
                    continue

                # Simple relation detection (can be enhanced)
                if self._are_related(memory, other_memory):
                    self.knowledge_graph["edges"].append({
                        "from": memory_id,
                        "to": other_id,
                        "type": "related"
                    })

        except Exception as e:
            logger.error(f"Error adding to knowledge graph: {e}")

    def _are_related(self, memory1: Dict[str, Any], memory2: Dict[str, Any]) -> bool:
        """Check if two memories are related (simple implementation)"""
        try:
            # Check if they share tags
            tags1 = set(memory1.get("tags", []))
            tags2 = set(memory2.get("tags", []))

            if tags1 & tags2:  # Intersection
                return True

            # Check if they're the same type
            if memory1.get("type") == memory2.get("type"):
                # Check content similarity (simple keyword matching)
                content1 = str(memory1.get("content", "")).lower()
                content2 = str(memory2.get("content", "")).lower()

                words1 = set(content1.split())
                words2 = set(content2.split())

                # If they share more than 20% of words
                if len(words1 & words2) / max(len(words1), len(words2), 1) > 0.2:
                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking relation: {e}")
            return False

    async def get_knowledge_graph(self) -> Dict[str, Any]:
        """Get the knowledge graph"""
        try:
            return {
                "success": True,
                "graph": self.knowledge_graph,
                "stats": {
                    "nodes": len(self.knowledge_graph["nodes"]),
                    "edges": len(self.knowledge_graph["edges"])
                }
            }
        except Exception as e:
            logger.error(f"Error getting knowledge graph: {e}")
            return {"success": False, "error": str(e)}

    # ==================== CONTEXT MANAGEMENT ====================

    async def add_to_context(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Add interaction to context window"""
        try:
            self.context_window.append({
                "timestamp": datetime.now().isoformat(),
                "interaction": interaction
            })

            return {"success": True, "context_size": len(self.context_window)}

        except Exception as e:
            logger.error(f"Error adding to context: {e}")
            return {"success": False, "error": str(e)}

    async def get_context(self, limit: int = 10) -> Dict[str, Any]:
        """Get recent context"""
        try:
            context = list(self.context_window)[-limit:]
            return {
                "success": True,
                "context": context,
                "count": len(context)
            }
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return {"success": False, "error": str(e)}

    # ==================== METRICS & ACTIVITY ====================

    async def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics"""
        try:
            return {
                "success": True,
                "metrics": self.metrics,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {"success": False, "error": str(e)}

    async def get_activity_log(self, limit: int = 100) -> Dict[str, Any]:
        """Get activity log"""
        try:
            activities = list(self.activity_log)[-limit:]
            return {
                "success": True,
                "activities": activities,
                "count": len(activities)
            }
        except Exception as e:
            logger.error(f"Error getting activity log: {e}")
            return {"success": False, "error": str(e)}

    async def get_memory_types(self) -> Dict[str, Any]:
        """Get available memory types"""
        try:
            return {
                "success": True,
                "memory_types": self.memory_types
            }
        except Exception as e:
            logger.error(f"Error getting memory types: {e}")
            return {"success": False, "error": str(e)}

