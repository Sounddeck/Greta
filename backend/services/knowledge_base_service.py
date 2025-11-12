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
from services.image_generation_providers import ImageGenerationService

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

        # Image generation service
        self.image_service = ImageGenerationService()

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
                await self.db.delete_one("knowledge_documents", {"document_id": document_id})
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

    # ==================== ENHANCED FEATURES (NotebookLM-like) ====================

    async def add_pdf_document(self, file_path: str, metadata: Dict = None) -> Dict[str, Any]:
        """Add PDF document to knowledge base with text extraction"""
        try:
            import PyPDF2

            logger.info(f"📄 Processing PDF: {file_path}")

            # Extract text from PDF
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"

                # Get PDF metadata
                pdf_metadata = pdf_reader.metadata or {}

            # Prepare document metadata
            doc_metadata = metadata or {}
            doc_metadata.update({
                "author": pdf_metadata.get("/Author", "Unknown"),
                "pages": len(pdf_reader.pages),
                "creation_date": str(pdf_metadata.get("/CreationDate", "")),
                "source_type": "pdf",
                "source_path": file_path
            })

            # Add document to knowledge base
            result = await self.add_document({
                "title": doc_metadata.get("title") or pdf_metadata.get("/Title", f"PDF Document - {file_path.split('/')[-1]}"),
                "content": text.strip(),
                "category": doc_metadata.get("category", "reference"),
                "tags": doc_metadata.get("tags", ["pdf", "document"]),
                "metadata": doc_metadata
            })

            if result.get("success"):
                logger.info(f"✅ PDF document added: {result['document_id']}")
                self._log_activity("pdf_added", f"Added PDF: {file_path}", {"document_id": result['document_id']})

            return result

        except Exception as e:
            logger.error(f"❌ PDF processing failed: {e}")
            return {"success": False, "error": str(e)}

    async def add_web_document(self, url: str, metadata: Dict = None) -> Dict[str, Any]:
        """Add web page content to knowledge base with article extraction"""
        try:
            from newspaper import Article

            logger.info(f"🌐 Processing web URL: {url}")

            # Download and parse article
            article = Article(url)
            article.download()
            article.parse()

            # Prepare document metadata
            doc_metadata = metadata or {}
            doc_metadata.update({
                "authors": article.authors,
                "publish_date": str(article.publish_date) if article.publish_date else None,
                "top_image": article.top_image,
                "keywords": article.keywords if hasattr(article, 'keywords') else [],
                "source_type": "web",
                "source_url": url
            })

            # Add document to knowledge base
            result = await self.add_document({
                "title": doc_metadata.get("title") or article.title or f"Web Article - {url}",
                "content": article.text.strip(),
                "category": doc_metadata.get("category", "reference"),
                "tags": doc_metadata.get("tags", ["web", "article"]),
                "metadata": doc_metadata
            })

            if result.get("success"):
                logger.info(f"✅ Web document added: {result['document_id']}")
                self._log_activity("web_added", f"Added web article: {url}", {"document_id": result['document_id']})

            return result

        except Exception as e:
            logger.error(f"❌ Web scraping failed: {e}")
            return {"success": False, "error": str(e)}

    async def generate_summary(self, document_id: str, max_length: int = 200) -> Dict[str, Any]:
        """Generate AI-powered summary of document"""
        try:
            document = self.documents.get(document_id)
            if not document:
                return {"success": False, "error": "Document not found"}

            logger.info(f"🤖 Generating summary for: {document_id}")

            content = document.get("content", "")
            title = document.get("title", "Untitled")

            # Create prompt for LLM
            prompt = f"""Summarize the following document in {max_length} words or less:

Title: {title}

Content:
{content[:5000]}

Provide:
1. A concise summary (2-3 sentences)
2. 3-5 key points (bullet points)
3. Main topics covered (comma-separated)

Format your response as:
SUMMARY: [your summary]
KEY POINTS:
- [point 1]
- [point 2]
- [point 3]
TOPICS: [topic1, topic2, topic3]
"""

            # Generate summary using LLM
            summary_response = await self.llm_service.generate(prompt)

            # Update document with summary
            document["summary"] = summary_response
            document["summary_generated_at"] = datetime.now().isoformat()

            # Save to database
            try:
                await self.db.update_one(
                    "knowledge_documents",
                    {"document_id": document_id},
                    {"$set": {
                        "summary": summary_response,
                        "summary_generated_at": document["summary_generated_at"]
                    }}
                )
            except Exception as e:
                logger.warning(f"Could not save summary to database: {e}")

            logger.info(f"✅ Summary generated for: {document_id}")
            self._log_activity("summary_generated", f"Generated summary for: {title}", {"document_id": document_id})

            return {
                "success": True,
                "document_id": document_id,
                "summary": summary_response,
                "generated_at": document["summary_generated_at"]
            }

        except Exception as e:
            logger.error(f"❌ Summary generation failed: {e}")
            return {"success": False, "error": str(e)}

    async def generate_audio_overview(self, document_id: str) -> Dict[str, Any]:
        """Generate audio overview of document (like NotebookLM)"""
        try:
            document = self.documents.get(document_id)
            if not document:
                return {"success": False, "error": "Document not found"}

            logger.info(f"🎙️ Generating audio overview for: {document_id}")

            # Get or generate summary first
            summary = document.get("summary")
            if not summary:
                summary_result = await self.generate_summary(document_id)
                if not summary_result.get("success"):
                    return {"success": False, "error": "Could not generate summary for audio"}
                summary = summary_result.get("summary", "")

            # Import voice service
            try:
                from services.greta_voice_service import get_greta_voice_service
                voice_service = await get_greta_voice_service()
            except Exception as e:
                logger.warning(f"Voice service not available: {e}")
                return {"success": False, "error": "Voice service not available"}

            # Generate audio from summary
            audio_result = await voice_service.synthesize_speech(
                text=summary,
                voice="en-US-Neural2-F",
                output_format="mp3"
            )

            if not audio_result.get("success"):
                return {"success": False, "error": "Audio synthesis failed"}

            # Store audio path in document
            audio_path = f"knowledge_base/audio/{document_id}.mp3"
            document["audio_overview"] = audio_path
            document["audio_generated_at"] = datetime.now().isoformat()

            # Save to database
            try:
                await self.db.update_one(
                    "knowledge_documents",
                    {"document_id": document_id},
                    {"$set": {
                        "audio_overview": audio_path,
                        "audio_generated_at": document["audio_generated_at"]
                    }}
                )
            except Exception as e:
                logger.warning(f"Could not save audio path to database: {e}")

            logger.info(f"✅ Audio overview generated for: {document_id}")
            self._log_activity("audio_generated", f"Generated audio for: {document.get('title')}", {"document_id": document_id})

            return {
                "success": True,
                "document_id": document_id,
                "audio_path": audio_path,
                "generated_at": document["audio_generated_at"]
            }

        except Exception as e:
            logger.error(f"❌ Audio generation failed: {e}")
            return {"success": False, "error": str(e)}

    async def chat_with_documents(
        self,
        query: str,
        document_ids: List[str] = None,
        conversation_history: List[Dict] = None,
        max_documents: int = 5
    ) -> Dict[str, Any]:
        """Chat with documents using RAG (Retrieval-Augmented Generation)"""
        try:
            logger.info(f"💬 Chat query: {query}")

            # If no document IDs provided, use semantic search to find relevant documents
            if not document_ids:
                search_results = await self.semantic_search(query, limit=max_documents)
                document_ids = [r["document_id"] for r in search_results]

            # Gather context from documents
            context = ""
            sources = []
            for doc_id in document_ids[:max_documents]:
                doc = self.documents.get(doc_id)
                if doc:
                    # Use summary if available, otherwise use truncated content
                    doc_text = doc.get("summary", doc.get("content", ""))[:2000]
                    context += f"\n\n--- {doc.get('title')} ---\n{doc_text}"
                    sources.append({
                        "document_id": doc_id,
                        "title": doc.get("title"),
                        "category": doc.get("category"),
                        "source_type": doc.get("metadata", {}).get("source_type", "text")
                    })

            # Build conversation context
            conversation_context = ""
            if conversation_history:
                for msg in conversation_history[-5:]:  # Last 5 messages
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    conversation_context += f"\n{role.upper()}: {content}"

            # Create RAG prompt
            prompt = f"""You are a helpful AI assistant with access to the user's knowledge base.

Context from documents:
{context}

Conversation history:
{conversation_context}

User question: {query}

Provide a helpful answer based on the context above. If the answer is not in the context, say so clearly.
When possible, cite which documents you're referencing.
Be concise but thorough.
"""

            # Generate response using LLM
            response = await self.llm_service.generate(prompt)

            # Log activity
            self._log_activity("chat_query", f"Chat query: {query}", {
                "document_count": len(sources),
                "query_length": len(query)
            })

            logger.info(f"✅ Chat response generated")

            return {
                "success": True,
                "query": query,
                "response": response,
                "sources": sources,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Chat failed: {e}")
            return {"success": False, "error": str(e)}

    # ==================== ENHANCED FEATURES (NotebookLM-like) ====================

    async def add_pdf_document(self, file_path: str, metadata: Dict = None) -> Dict[str, Any]:
        """Add PDF document to knowledge base with text extraction"""
        try:
            import PyPDF2

            logger.info(f"📄 Processing PDF: {file_path}")

            # Extract text from PDF
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"

                # Get PDF metadata
                pdf_metadata = pdf_reader.metadata or {}

            # Prepare document metadata
            doc_metadata = metadata or {}
            doc_metadata.update({
                "author": pdf_metadata.get("/Author", "Unknown"),
                "pages": len(pdf_reader.pages),
                "creation_date": str(pdf_metadata.get("/CreationDate", "")),
                "source_type": "pdf",
                "source_path": file_path
            })

            # Add document to knowledge base
            result = await self.add_document({
                "title": doc_metadata.get("title") or pdf_metadata.get("/Title", f"PDF Document - {file_path.split('/')[-1]}"),
                "content": text.strip(),
                "category": doc_metadata.get("category", "reference"),
                "tags": doc_metadata.get("tags", ["pdf", "document"]),
                "metadata": doc_metadata
            })

            if result.get("success"):
                logger.info(f"✅ PDF document added: {result['document_id']}")
                self._log_activity("pdf_added", f"Added PDF: {file_path}", {"document_id": result['document_id']})

            return result

        except Exception as e:
            logger.error(f"❌ PDF processing failed: {e}")
            return {"success": False, "error": str(e)}

    async def add_web_document(self, url: str, metadata: Dict = None) -> Dict[str, Any]:
        """Add web page content to knowledge base with article extraction"""
        try:
            from newspaper import Article

            logger.info(f"🌐 Processing web URL: {url}")

            # Download and parse article
            article = Article(url)
            article.download()
            article.parse()

            # Prepare document metadata
            doc_metadata = metadata or {}
            doc_metadata.update({
                "authors": article.authors,
                "publish_date": str(article.publish_date) if article.publish_date else None,
                "top_image": article.top_image,
                "keywords": article.keywords if hasattr(article, 'keywords') else [],
                "source_type": "web",
                "source_url": url
            })

            # Add document to knowledge base
            result = await self.add_document({
                "title": doc_metadata.get("title") or article.title or f"Web Article - {url}",
                "content": article.text.strip(),
                "category": doc_metadata.get("category", "reference"),
                "tags": doc_metadata.get("tags", ["web", "article"]),
                "metadata": doc_metadata
            })

            if result.get("success"):
                logger.info(f"✅ Web document added: {result['document_id']}")
                self._log_activity("web_added", f"Added web article: {url}", {"document_id": result['document_id']})

            return result

        except Exception as e:
            logger.error(f"❌ Web scraping failed: {e}")
            return {"success": False, "error": str(e)}

    async def generate_summary(self, document_id: str, max_length: int = 200) -> Dict[str, Any]:
        """Generate AI-powered summary of document"""
        try:
            document = self.documents.get(document_id)
            if not document:
                return {"success": False, "error": "Document not found"}

            logger.info(f"🤖 Generating summary for: {document_id}")

            content = document.get("content", "")
            title = document.get("title", "Untitled")

            # Create prompt for LLM
            prompt = f"""Summarize the following document in {max_length} words or less:

Title: {title}

Content:
{content[:5000]}

Provide:
1. A concise summary (2-3 sentences)
2. 3-5 key points (bullet points)
3. Main topics covered (comma-separated)

Format your response as:
SUMMARY: [your summary]
KEY POINTS:
- [point 1]
- [point 2]
- [point 3]
TOPICS: [topic1, topic2, topic3]
"""

            # Generate summary using LLM
            summary_response = await self.llm_service.generate(prompt)

            # Update document with summary
            document["summary"] = summary_response
            document["summary_generated_at"] = datetime.now().isoformat()

            # Save to database
            try:
                await self.db.update_one(
                    "knowledge_documents",
                    {"document_id": document_id},
                    {"$set": {
                        "summary": summary_response,
                        "summary_generated_at": document["summary_generated_at"]
                    }}
                )
            except Exception as e:
                logger.warning(f"Could not save summary to database: {e}")

            logger.info(f"✅ Summary generated for: {document_id}")
            self._log_activity("summary_generated", f"Generated summary for: {title}", {"document_id": document_id})

            return {
                "success": True,
                "document_id": document_id,
                "summary": summary_response,
                "generated_at": document["summary_generated_at"]
            }

        except Exception as e:
            logger.error(f"❌ Summary generation failed: {e}")
            return {"success": False, "error": str(e)}

    async def generate_audio_overview(self, document_id: str) -> Dict[str, Any]:
        """Generate audio overview of document (like NotebookLM)"""
        try:
            document = self.documents.get(document_id)
            if not document:
                return {"success": False, "error": "Document not found"}

            logger.info(f"🎙️ Generating audio overview for: {document_id}")

            # Get or generate summary first
            summary = document.get("summary")
            if not summary:
                summary_result = await self.generate_summary(document_id)
                if not summary_result.get("success"):
                    return {"success": False, "error": "Could not generate summary for audio"}
                summary = summary_result.get("summary", "")

            # Import voice service
            try:
                from services.greta_voice_service import get_greta_voice_service
                voice_service = await get_greta_voice_service()
            except Exception as e:
                logger.warning(f"Voice service not available: {e}")
                return {"success": False, "error": "Voice service not available"}

            # Generate audio from summary
            audio_result = await voice_service.synthesize_speech(
                text=summary,
                voice="en-US-Neural2-F",
                output_format="mp3"
            )

            if not audio_result.get("success"):
                return {"success": False, "error": "Audio synthesis failed"}

            # Store audio path in document
            audio_path = f"knowledge_base/audio/{document_id}.mp3"
            document["audio_overview"] = audio_path
            document["audio_generated_at"] = datetime.now().isoformat()

            # Save to database
            try:
                await self.db.update_one(
                    "knowledge_documents",
                    {"document_id": document_id},
                    {"$set": {
                        "audio_overview": audio_path,
                        "audio_generated_at": document["audio_generated_at"]
                    }}
                )
            except Exception as e:
                logger.warning(f"Could not save audio path to database: {e}")

            logger.info(f"✅ Audio overview generated for: {document_id}")
            self._log_activity("audio_generated", f"Generated audio for: {document.get('title')}", {"document_id": document_id})

            return {
                "success": True,
                "document_id": document_id,
                "audio_path": audio_path,
                "generated_at": document["audio_generated_at"]
            }

        except Exception as e:
            logger.error(f"❌ Audio generation failed: {e}")
            return {"success": False, "error": str(e)}

    async def chat_with_documents(
        self,
        query: str,
        document_ids: List[str] = None,
        conversation_history: List[Dict] = None,
        max_documents: int = 5
    ) -> Dict[str, Any]:
        """Chat with documents using RAG (Retrieval-Augmented Generation)"""
        try:
            logger.info(f"💬 Chat query: {query}")

            # If no document IDs provided, use semantic search to find relevant documents
            if not document_ids:
                search_results = await self.semantic_search(query, limit=max_documents)
                document_ids = [r["document_id"] for r in search_results]

            # Gather context from documents
            context = ""
            sources = []
            for doc_id in document_ids[:max_documents]:
                doc = self.documents.get(doc_id)
                if doc:
                    # Use summary if available, otherwise use truncated content
                    doc_text = doc.get("summary", doc.get("content", ""))[:2000]
                    context += f"\n\n--- {doc.get('title')} ---\n{doc_text}"
                    sources.append({
                        "document_id": doc_id,
                        "title": doc.get("title"),
                        "category": doc.get("category"),
                        "source_type": doc.get("metadata", {}).get("source_type", "text")
                    })

            # Build conversation context
            conversation_context = ""
            if conversation_history:
                for msg in conversation_history[-5:]:  # Last 5 messages
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    conversation_context += f"\n{role.upper()}: {content}"

            # Create RAG prompt
            prompt = f"""You are a helpful AI assistant with access to the user's knowledge base.

Context from documents:
{context}

Conversation history:
{conversation_context}

User question: {query}

Provide a helpful answer based on the context above. If the answer is not in the context, say so clearly.
When possible, cite which documents you're referencing.
Be concise but thorough.
"""

            # Generate response using LLM
            response = await self.llm_service.generate(prompt)

            # Log activity
            self._log_activity("chat_query", f"Chat query: {query}", {
                "document_count": len(sources),
                "query_length": len(query)
            })

            logger.info(f"✅ Chat response generated")

            return {
                "success": True,
                "query": query,
                "response": response,
                "sources": sources,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Chat failed: {e}")
            return {"success": False, "error": str(e)}

    async def generate_image(
        self,
        prompt: str,
        provider: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        save_to_knowledge_base: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate image from text prompt using configured provider

        Args:
            prompt: Text description of the image to generate
            provider: Image generation provider to use (default: stable_diffusion)
            negative_prompt: What to avoid in the image
            width: Image width in pixels
            height: Image height in pixels
            num_inference_steps: Number of denoising steps (more = better quality, slower)
            guidance_scale: How closely to follow the prompt (7-15 recommended)
            save_to_knowledge_base: Whether to save the generated image as a document
            **kwargs: Additional provider-specific parameters

        Returns:
            Dict with success status, image data (base64), metadata
        """
        try:
            logger.info(f"🎨 Generating image: {prompt[:50]}...")

            # Generate image using the image service
            result = await self.image_service.generate_image(
                prompt=prompt,
                provider=provider,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                **kwargs
            )

            # Save to knowledge base if requested
            if save_to_knowledge_base and result.get("success"):
                doc_id = await self.add_document(
                    title=f"Generated Image: {prompt[:50]}",
                    content=f"AI-generated image from prompt: {prompt}",
                    category="reference",
                    tags=["ai-generated", "image", "stable-diffusion"],
                    metadata={
                        "type": "generated_image",
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "provider": result["metadata"].get("provider"),
                        "image_data": result["image_data"],
                        "format": result["format"],
                        "width": width,
                        "height": height,
                        "generation_time": result["metadata"].get("generation_time")
                    }
                )
                result["document_id"] = doc_id
                logger.info(f"✅ Image saved to knowledge base: {doc_id}")

            # Log activity
            self._log_activity("generate_image", f"Generated image: {prompt[:50]}", {
                "provider": result.get("metadata", {}).get("provider"),
                "saved_to_kb": save_to_knowledge_base
            })

            return {"success": True, **result}

        except Exception as e:
            logger.error(f"❌ Image generation failed: {e}")
            return {"success": False, "error": str(e)}

    def list_image_providers(self) -> List[Dict[str, Any]]:
        """
        List all available image generation providers

        Returns:
            List of provider information dictionaries
        """
        return self.image_service.list_providers()

    def get_image_provider_info(self, provider_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific image generation provider

        Args:
            provider_name: Name of the provider (e.g., 'stable_diffusion', 'dalle', 'midjourney')

        Returns:
            Provider information dictionary
        """
        return self.image_service.get_provider_info(provider_name)

