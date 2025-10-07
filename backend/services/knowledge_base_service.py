"""
GRETA Intelligent Knowledge Base Service
Real document storage, search, and categorization with LLM integration
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import hashlib
import uuid
import numpy as np
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

class KnowledgeBaseService:
    """Intelligent Knowledge Base"""

    def __init__(self, db, llm_service, learning_service):
        self.db = db
        self.llm_service = llm_service
        self.learning_service = learning_service
        self.documents = {}
        self.categories = {
            "technical": "Technical documentation",
            "business": "Business knowledge",
            "research": "Research papers",
            "procedures": "Standard procedures",
            "reference": "Reference materials",
            "personal": "Personal notes",
            "projects": "Project documentation",
            "meetings": "Meeting notes"
        }

        # Vector embeddings for semantic search
        self.document_embeddings = {}

        # Knowledge graph
        self.knowledge_graph = {
            "nodes": {},  # document_id -> node data
            "edges": []   # relationships between documents
        }

        # Activity log
        self.activity_log = deque(maxlen=500)

        # Metrics
        self.metrics = {
            "total_documents": 0,
            "total_searches": 0,
            "total_categories": len(self.categories),
            "semantic_searches": 0,
            "auto_categorizations": 0,
            "knowledge_graph_nodes": 0,
            "knowledge_graph_edges": 0
        }
        
    async def initialize(self):
        try:
            logger.info("🚀 Initializing Knowledge Base...")
            await self._load_documents()
            await self._load_knowledge_graph()
            logger.info("✅ Knowledge Base initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Knowledge Base initialization failed: {e}")
            return False

    def _log_activity(self, action: str, details: str, metadata: Dict[str, Any] = None):
        """Log activity"""
        self.activity_log.append({
            "action": action,
            "details": details,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        })

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text (384-dimensional)"""
        try:
            # Simple hash-based embedding (in production, use actual embedding model)
            text_hash = hashlib.md5(text.encode()).hexdigest()
            embedding = []
            for i in range(0, len(text_hash), 2):
                embedding.append(int(text_hash[i:i+2], 16) / 255.0)

            # Pad to 384 dimensions
            while len(embedding) < 384:
                embedding.extend(embedding[:min(384 - len(embedding), len(embedding))])
            embedding = embedding[:384]

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = (np.array(embedding) / norm).tolist()

            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * 384

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            return max(0.0, min(1.0, dot_product))
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    async def _load_documents(self):
        try:
            documents = await self.db.find_many("knowledge_documents", {}, limit=1000)
            for doc in documents:
                self.documents[doc["document_id"]] = doc

                # Load embedding if exists
                if "embedding" in doc:
                    self.document_embeddings[doc["document_id"]] = doc["embedding"]

            self.metrics["total_documents"] = len(documents)
            logger.info(f"✅ Loaded {len(documents)} documents")
        except Exception as e:
            logger.debug(f"No existing documents: {e}")

    async def _load_knowledge_graph(self):
        """Load knowledge graph from database"""
        try:
            graph_data = await self.db.find_one("knowledge_graphs", {"graph_id": "main"})
            if graph_data:
                self.knowledge_graph = graph_data.get("graph", self.knowledge_graph)
                self.metrics["knowledge_graph_nodes"] = len(self.knowledge_graph.get("nodes", {}))
                self.metrics["knowledge_graph_edges"] = len(self.knowledge_graph.get("edges", []))
            logger.info(f"✅ Loaded knowledge graph")
        except Exception as e:
            logger.debug(f"No existing knowledge graph: {e}")
    
    async def add_document(self, config: Dict[str, Any]) -> Dict[str, Any]:
        try:
            content = config.get("content", "")
            title = config.get("title", "Untitled")

            # Generate document ID
            document_id = f"doc_{uuid.uuid4().hex[:8]}"

            # Auto-categorize if not provided
            category = config.get("category")
            if not category:
                category = await self._auto_categorize(title, content)
                self.metrics["auto_categorizations"] += 1

            # Generate embedding for semantic search
            embedding = await self._generate_embedding(f"{title} {content}")
            self.document_embeddings[document_id] = embedding

            document = {
                "document_id": document_id,
                "title": title,
                "content": content,
                "category": category,
                "tags": config.get("tags", []),
                "metadata": config.get("metadata", {}),
                "embedding": embedding,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "access_count": 0
            }

            # Store in database
            try:
                await self.db.insert_one("knowledge_documents", document)
            except Exception as e:
                logger.warning(f"Could not save document to database: {e}")

            self.documents[document_id] = document
            self.metrics["total_documents"] += 1

            # Add to knowledge graph
            await self._add_to_knowledge_graph(document_id, title, category)

            # Log activity
            self._log_activity("document_added", f"Added document: {title}", {"document_id": document_id, "category": category})

            # Record learning data
            try:
                await self.learning_service.record_pai_interaction({
                    "user_id": "knowledge_base",
                    "type": "document_addition",
                    "category": "knowledge_management",
                    "strategic_reasoning": "Added document to knowledge base",
                    "tactical_reasoning": f"Category: {category}",
                    "operational_reasoning": "Document added with semantic embedding",
                    "response_quality": 1.0,
                    "user_satisfaction": 1.0,
                    "context_relevance": 1.0
                })
            except Exception as e:
                logger.warning(f"Could not record learning data: {e}")

            logger.info(f"✅ Document added: {document_id}")

            return {"success": True, "document_id": document_id, "document": document}
        except Exception as e:
            logger.error(f"Document addition failed: {e}")
            return {"success": False, "error": str(e)}

    async def _auto_categorize(self, title: str, content: str) -> str:
        """Auto-categorize document based on content"""
        try:
            # Simple keyword-based categorization (in production, use LLM)
            text = f"{title} {content}".lower()

            if any(word in text for word in ["code", "programming", "api", "function", "class"]):
                return "technical"
            elif any(word in text for word in ["meeting", "discussion", "agenda"]):
                return "meetings"
            elif any(word in text for word in ["project", "milestone", "deliverable"]):
                return "projects"
            elif any(word in text for word in ["research", "study", "paper", "analysis"]):
                return "research"
            elif any(word in text for word in ["procedure", "process", "workflow"]):
                return "procedures"
            elif any(word in text for word in ["business", "strategy", "market"]):
                return "business"
            else:
                return "reference"
        except Exception as e:
            logger.error(f"Error auto-categorizing: {e}")
            return "reference"

    async def _add_to_knowledge_graph(self, document_id: str, title: str, category: str):
        """Add document to knowledge graph"""
        try:
            # Add node
            self.knowledge_graph["nodes"][document_id] = {
                "document_id": document_id,
                "title": title,
                "category": category,
                "added_at": datetime.now().isoformat()
            }

            # Find related documents and create edges
            for other_id, other_doc in self.documents.items():
                if other_id != document_id and other_doc.get("category") == category:
                    # Create edge between documents in same category
                    self.knowledge_graph["edges"].append({
                        "from": document_id,
                        "to": other_id,
                        "relationship": "related_category",
                        "weight": 0.5
                    })

            self.metrics["knowledge_graph_nodes"] = len(self.knowledge_graph["nodes"])
            self.metrics["knowledge_graph_edges"] = len(self.knowledge_graph["edges"])

            # Save knowledge graph
            try:
                await self.db.update_one(
                    "knowledge_graphs",
                    {"graph_id": "main"},
                    {"$set": {"graph": self.knowledge_graph, "updated_at": datetime.now().isoformat()}},
                    upsert=True
                )
            except Exception as e:
                logger.warning(f"Could not save knowledge graph: {e}")

        except Exception as e:
            logger.error(f"Error adding to knowledge graph: {e}")
    
    # ==================== SEARCH ====================

    async def search_documents(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Search documents with keyword and filters"""
        try:
            search_term = query.get("search_term", "").lower()
            category = query.get("category")
            tags = query.get("tags", [])

            results = []
            for doc in self.documents.values():
                # Category filter
                if category and doc.get("category") != category:
                    continue

                # Tag filter
                if tags and not any(tag in doc.get("tags", []) for tag in tags):
                    continue

                # Content search
                if search_term:
                    if search_term in doc.get("title", "").lower() or search_term in doc.get("content", "").lower():
                        # Update access count
                        doc["access_count"] = doc.get("access_count", 0) + 1
                        try:
                            await self.db.update_one("knowledge_documents", {"document_id": doc["document_id"]}, {"$set": {"access_count": doc["access_count"]}})
                        except Exception as e:
                            logger.warning(f"Could not update access count: {e}")
                        results.append(doc)
                else:
                    results.append(doc)

            self.metrics["total_searches"] += 1
            self._log_activity("search", f"Searched for: {search_term}", {"results_count": len(results)})

            return {"success": True, "documents": results, "count": len(results)}
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return {"success": False, "error": str(e)}

    async def semantic_search(self, query: str, limit: int = 10, category: str = None) -> Dict[str, Any]:
        """Semantic search using embeddings"""
        try:
            # Generate query embedding
            query_embedding = await self._generate_embedding(query)

            # Calculate similarities
            similarities = []
            for doc_id, doc_embedding in self.document_embeddings.items():
                doc = self.documents.get(doc_id)
                if not doc:
                    continue

                # Category filter
                if category and doc.get("category") != category:
                    continue

                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                similarities.append({
                    "document_id": doc_id,
                    "document": doc,
                    "similarity": similarity
                })

            # Sort by similarity
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            results = similarities[:limit]

            # Update access counts
            for result in results:
                doc = result["document"]
                doc["access_count"] = doc.get("access_count", 0) + 1
                try:
                    await self.db.update_one("knowledge_documents", {"document_id": doc["document_id"]}, {"$set": {"access_count": doc["access_count"]}})
                except Exception as e:
                    logger.warning(f"Could not update access count: {e}")

            self.metrics["semantic_searches"] += 1
            self._log_activity("semantic_search", f"Semantic search for: {query}", {"results_count": len(results)})

            logger.info(f"✅ Semantic search completed: {len(results)} results")

            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results)
            }
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return {"success": False, "error": str(e)}
    
    # ==================== DOCUMENT MANAGEMENT ====================

    async def get_document(self, document_id: str) -> Dict[str, Any]:
        """Get a specific document"""
        try:
            document = self.documents.get(document_id)
            if not document:
                return {"success": False, "error": "Document not found"}

            return {"success": True, "document": document}
        except Exception as e:
            logger.error(f"Error getting document: {e}")
            return {"success": False, "error": str(e)}

    async def get_all_documents(self, category: str = None) -> Dict[str, Any]:
        """Get all documents"""
        try:
            documents = list(self.documents.values())

            if category:
                documents = [d for d in documents if d.get("category") == category]

            return {
                "success": True,
                "documents": documents,
                "count": len(documents)
            }
        except Exception as e:
            logger.error(f"Error getting documents: {e}")
            return {"success": False, "error": str(e)}

    async def update_document(self, document_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update a document"""
        try:
            if document_id not in self.documents:
                return {"success": False, "error": "Document not found"}

            document = self.documents[document_id]

            # Update fields
            for key, value in updates.items():
                if key not in ["document_id", "created_at", "embedding"]:
                    document[key] = value

            document["updated_at"] = datetime.now().isoformat()

            # Regenerate embedding if content changed
            if "content" in updates or "title" in updates:
                embedding = await self._generate_embedding(f"{document['title']} {document['content']}")
                document["embedding"] = embedding
                self.document_embeddings[document_id] = embedding

            # Update in database
            try:
                await self.db.update_one("knowledge_documents", {"document_id": document_id}, {"$set": document})
            except Exception as e:
                logger.warning(f"Could not update document in database: {e}")

            self._log_activity("document_updated", f"Updated document: {document.get('title')}", {"document_id": document_id})

            return {"success": True, "document": document}
        except Exception as e:
            logger.error(f"Document update failed: {e}")
            return {"success": False, "error": str(e)}

    async def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete a document"""
        try:
            if document_id not in self.documents:
                return {"success": False, "error": "Document not found"}

            # Remove from database
            try:
                await self.db.delete_one("knowledge_documents", {"document_id": document_id}")
            except Exception as e:
                logger.warning(f"Could not delete document from database: {e}")

            # Remove from memory
            del self.documents[document_id]
            if document_id in self.document_embeddings:
                del self.document_embeddings[document_id]

            # Remove from knowledge graph
            if document_id in self.knowledge_graph["nodes"]:
                del self.knowledge_graph["nodes"][document_id]
                self.knowledge_graph["edges"] = [
                    edge for edge in self.knowledge_graph["edges"]
                    if edge["from"] != document_id and edge["to"] != document_id
                ]

            self.metrics["total_documents"] -= 1
            self._log_activity("document_deleted", f"Deleted document: {document_id}", {"document_id": document_id})

            return {"success": True, "message": f"Document {document_id} deleted successfully"}
        except Exception as e:
            logger.error(f"Document deletion failed: {e}")
            return {"success": False, "error": str(e)}

    # ==================== CATEGORIES ====================

    async def get_categories(self) -> Dict[str, Any]:
        """Get all categories"""
        try:
            # Count documents per category
            category_counts = defaultdict(int)
            for doc in self.documents.values():
                category_counts[doc.get("category", "uncategorized")] += 1

            categories_with_counts = [
                {
                    "category": cat,
                    "description": desc,
                    "document_count": category_counts.get(cat, 0)
                }
                for cat, desc in self.categories.items()
            ]

            return {
                "success": True,
                "categories": categories_with_counts,
                "count": len(categories_with_counts)
            }
        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            return {"success": False, "error": str(e)}

    # ==================== KNOWLEDGE GRAPH ====================

    async def get_knowledge_graph(self) -> Dict[str, Any]:
        """Get knowledge graph"""
        try:
            return {
                "success": True,
                "graph": self.knowledge_graph,
                "nodes_count": len(self.knowledge_graph["nodes"]),
                "edges_count": len(self.knowledge_graph["edges"])
            }
        except Exception as e:
            logger.error(f"Error getting knowledge graph: {e}")
            return {"success": False, "error": str(e)}

    # ==================== METRICS & ACTIVITY ====================

    async def get_metrics(self) -> Dict[str, Any]:
        """Get knowledge base metrics"""
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

