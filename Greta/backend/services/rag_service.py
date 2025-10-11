"""
RAG (Retrieval-Augmented Generation) Service for CPAS4/Greta

This service implements vector database integration, document ingestion,
semantic search, and retrieval-augmented generation capabilities.

Week 5-6 Implementation - Part of 23-week plan
Integrates with:
- Week 1: LLM Service for generation
- Week 2: Model Management for embedding models
- Week 4: Conversation Management for context-aware RAG
"""

import asyncio
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from loguru import logger
from pydantic import BaseModel, Field

# Import from existing services
try:
    from backend.services.llm_service import llm_service
    from backend.services.conversation_service import conversation_service
except ImportError:
    logger.warning("Could not import existing services - running in standalone mode")
    llm_service = None
    conversation_service = None


# ============================================================================
# Pydantic Schemas
# ============================================================================

class Document(BaseModel):
    """Document schema for RAG system"""
    document_id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    source: str = Field(..., description="Document source (file path, URL, etc.)")
    doc_type: str = Field(default="text", description="Document type (text, pdf, md, code, etc.)")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    chunk_index: Optional[int] = Field(None, description="Chunk index if document is chunked")
    total_chunks: Optional[int] = Field(None, description="Total chunks if document is chunked")


class DocumentChunk(BaseModel):
    """Document chunk schema for chunking strategy"""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk content")
    chunk_index: int = Field(..., description="Chunk index in document")
    total_chunks: int = Field(..., description="Total chunks in document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    embedding: Optional[List[float]] = Field(None, description="Chunk embedding vector")


class SearchResult(BaseModel):
    """Search result schema"""
    document_id: str
    content: str
    score: float = Field(..., description="Similarity score (0-1)")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_index: Optional[int] = None


class RAGResponse(BaseModel):
    """RAG response schema"""
    query: str
    answer: str
    sources: List[SearchResult]
    context_used: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ============================================================================
# RAG Service Implementation
# ============================================================================

class RAGService:
    """
    RAG Service for document ingestion, semantic search, and retrieval-augmented generation
    
    Features:
    - Vector database integration with ChromaDB
    - Document ingestion with multiple file format support
    - Smart chunking strategy for long documents
    - Semantic search with embedding generation
    - RAG with LLM integration
    - Context-aware retrieval with conversation history
    """
    
    def __init__(
        self,
        persist_directory: str = "./data/chromadb",
        collection_name: str = "greta_documents",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize RAG Service
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            embedding_model: Embedding model to use (default: sentence-transformers)
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        
        # Ensure persist directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "CPAS4/Greta document collection"}
            )
            logger.info(f"Created new collection: {collection_name}")
        
        # Document cache (in-memory for fast access)
        self.document_cache: Dict[str, Document] = {}
        
        logger.info(f"RAG Service initialized with collection: {collection_name}")
    
    # ========================================================================
    # Document Ingestion
    # ========================================================================
    
    def generate_document_id(self, source: str, content: str) -> str:
        """Generate unique document ID from source and content"""
        hash_input = f"{source}:{content[:100]}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def chunk_document(self, content: str, document_id: str) -> List[DocumentChunk]:
        """
        Chunk document into smaller pieces with overlap
        
        Args:
            content: Document content to chunk
            document_id: Parent document ID
            
        Returns:
            List of document chunks
        """
        chunks = []
        content_length = len(content)
        
        if content_length <= self.chunk_size:
            # Document is small enough, no chunking needed
            chunk = DocumentChunk(
                chunk_id=f"{document_id}_chunk_0",
                document_id=document_id,
                content=content,
                chunk_index=0,
                total_chunks=1,
                metadata={"length": content_length}
            )
            chunks.append(chunk)
            return chunks
        
        # Chunk the document with overlap
        start = 0
        chunk_index = 0
        
        while start < content_length:
            end = min(start + self.chunk_size, content_length)
            chunk_content = content[start:end]
            
            chunk = DocumentChunk(
                chunk_id=f"{document_id}_chunk_{chunk_index}",
                document_id=document_id,
                content=chunk_content,
                chunk_index=chunk_index,
                total_chunks=0,  # Will be updated after all chunks are created
                metadata={
                    "start": start,
                    "end": end,
                    "length": len(chunk_content)
                }
            )
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start = end - self.chunk_overlap
            chunk_index += 1
        
        # Update total_chunks for all chunks
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total_chunks
        
        logger.info(f"Chunked document {document_id} into {total_chunks} chunks")
        return chunks
    
    async def ingest_document(
        self,
        content: str,
        source: str,
        doc_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """
        Ingest a document into the RAG system
        
        Args:
            content: Document content
            source: Document source (file path, URL, etc.)
            doc_type: Document type (text, pdf, md, code, etc.)
            metadata: Additional metadata
            
        Returns:
            Ingested document
        """
        # Generate document ID
        document_id = self.generate_document_id(source, content)
        
        # Create document
        document = Document(
            document_id=document_id,
            content=content,
            source=source,
            doc_type=doc_type,
            metadata=metadata or {}
        )
        
        # Chunk document
        chunks = self.chunk_document(content, document_id)
        
        # Add chunks to ChromaDB
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        chunk_contents = [chunk.content for chunk in chunks]
        chunk_metadatas = [
            {
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "source": source,
                "doc_type": doc_type,
                **chunk.metadata
            }
            for chunk in chunks
        ]
        
        # Add to ChromaDB (ChromaDB will generate embeddings automatically)
        self.collection.add(
            ids=chunk_ids,
            documents=chunk_contents,
            metadatas=chunk_metadatas
        )
        
        # Cache document
        self.document_cache[document_id] = document
        
        logger.info(f"Ingested document {document_id} with {len(chunks)} chunks")
        return document
    
    async def ingest_file(self, file_path: str) -> Document:
        """
        Ingest a file into the RAG system
        
        Supports: TXT, MD, PY, JS, JSON, and other text files
        
        Args:
            file_path: Path to file to ingest
            
        Returns:
            Ingested document
        """
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Determine document type from extension
        extension = file_path_obj.suffix.lower()
        doc_type_map = {
            '.txt': 'text',
            '.md': 'markdown',
            '.py': 'python',
            '.js': 'javascript',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.html': 'html',
            '.css': 'css',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
        }
        doc_type = doc_type_map.get(extension, 'text')
        
        # Ingest document
        return await self.ingest_document(
            content=content,
            source=str(file_path),
            doc_type=doc_type,
            metadata={
                "filename": file_path_obj.name,
                "extension": extension,
                "size": len(content)
            }
        )
    
    async def ingest_directory(
        self,
        directory_path: str,
        extensions: Optional[List[str]] = None,
        recursive: bool = True
    ) -> List[Document]:
        """
        Ingest all files in a directory
        
        Args:
            directory_path: Path to directory
            extensions: List of file extensions to include (e.g., ['.txt', '.md'])
            recursive: Whether to search subdirectories
            
        Returns:
            List of ingested documents
        """
        directory = Path(directory_path)
        
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Invalid directory: {directory_path}")
        
        # Default extensions if not specified
        if extensions is None:
            extensions = ['.txt', '.md', '.py', '.js', '.json', '.yaml', '.yml']
        
        # Find all files
        if recursive:
            files = [f for f in directory.rglob('*') if f.suffix in extensions]
        else:
            files = [f for f in directory.glob('*') if f.suffix in extensions]
        
        # Ingest all files
        documents = []
        for file_path in files:
            try:
                doc = await self.ingest_file(str(file_path))
                documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to ingest {file_path}: {e}")
        
        logger.info(f"Ingested {len(documents)} documents from {directory_path}")
        return documents
    
    # ========================================================================
    # Semantic Search
    # ========================================================================
    
    async def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform semantic search on documents
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of search results
        """
        # Query ChromaDB
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filter_metadata
        )
        
        # Parse results
        search_results = []
        
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                result = SearchResult(
                    document_id=results['metadatas'][0][i].get('document_id', ''),
                    content=results['documents'][0][i],
                    score=1.0 - results['distances'][0][i],  # Convert distance to similarity
                    metadata=results['metadatas'][0][i],
                    chunk_index=results['metadatas'][0][i].get('chunk_index')
                )
                search_results.append(result)
        
        logger.info(f"Search for '{query}' returned {len(search_results)} results")
        return search_results
    
    # ========================================================================
    # RAG (Retrieval-Augmented Generation)
    # ========================================================================
    
    async def generate_with_rag(
        self,
        query: str,
        n_results: int = 3,
        conversation_id: Optional[str] = None,
        max_context_length: int = 2000
    ) -> RAGResponse:
        """
        Generate response using RAG (Retrieval-Augmented Generation)
        
        Args:
            query: User query
            n_results: Number of documents to retrieve
            conversation_id: Optional conversation ID for context-aware RAG
            max_context_length: Maximum context length in characters
            
        Returns:
            RAG response with answer and sources
        """
        # Step 1: Retrieve relevant documents
        search_results = await self.search(query, n_results=n_results)
        
        # Step 2: Build context from search results
        context_parts = []
        for i, result in enumerate(search_results):
            context_parts.append(f"[Source {i+1}] {result.content}")
        
        context = "\n\n".join(context_parts)
        
        # Truncate context if too long
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        
        # Step 3: Get conversation history if available
        conversation_context = ""
        if conversation_id and conversation_service:
            try:
                history = await conversation_service.get_context_window(
                    conversation_id,
                    max_tokens=1000,
                    max_messages=5
                )
                if history:
                    conversation_context = "\n\nRecent conversation:\n"
                    for msg in history:
                        conversation_context += f"{msg.role}: {msg.content}\n"
            except Exception as e:
                logger.warning(f"Could not get conversation context: {e}")
        
        # Step 4: Build prompt for LLM
        prompt = f"""You are a helpful AI assistant. Answer the user's question based on the provided context.

Context from knowledge base:
{context}
{conversation_context}

User question: {query}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information, say so."""
        
        # Step 5: Generate response with LLM
        answer = ""
        if llm_service:
            try:
                # Use LLM service to generate response
                response = await llm_service.generate(
                    prompt=prompt,
                    max_tokens=500,
                    temperature=0.7
                )
                answer = response.get('text', 'No response generated')
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                answer = f"Error generating response: {e}"
        else:
            answer = "LLM service not available. Context retrieved successfully."
        
        # Step 6: Create RAG response
        rag_response = RAGResponse(
            query=query,
            answer=answer,
            sources=search_results,
            context_used=context,
            metadata={
                "n_results": n_results,
                "conversation_id": conversation_id,
                "context_length": len(context)
            }
        )
        
        # Step 7: Store in conversation if conversation_id provided
        if conversation_id and conversation_service:
            try:
                await conversation_service.add_message(
                    conversation_id,
                    role="user",
                    content=query
                )
                await conversation_service.add_message(
                    conversation_id,
                    role="assistant",
                    content=answer,
                    metadata={
                        "rag_sources": [s.document_id for s in search_results],
                        "rag_scores": [s.score for s in search_results]
                    }
                )
            except Exception as e:
                logger.warning(f"Could not store in conversation: {e}")
        
        return rag_response
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    async def get_document(self, document_id: str) -> Optional[Document]:
        """Get document by ID"""
        return self.document_cache.get(document_id)
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete document and all its chunks"""
        try:
            # Find all chunks for this document
            results = self.collection.get(
                where={"document_id": document_id}
            )
            
            if results['ids']:
                # Delete all chunks
                self.collection.delete(ids=results['ids'])
                
                # Remove from cache
                if document_id in self.document_cache:
                    del self.document_cache[document_id]
                
                logger.info(f"Deleted document {document_id} and {len(results['ids'])} chunks")
                return True
            
            return False
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection"""
        count = self.collection.count()
        
        return {
            "total_chunks": count,
            "total_documents": len(self.document_cache),
            "collection_name": self.collection_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }
    
    async def reset_collection(self) -> bool:
        """Reset the entire collection (delete all documents)"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "CPAS4/Greta document collection"}
            )
            self.document_cache.clear()
            logger.info(f"Reset collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            return False


# ============================================================================
# Global RAG Service Instance
# ============================================================================

rag_service = RAGService()
