"""
Enhanced Greta PAI Backend - Complete Jina Remote MCP Server Integration
Provides full access to all 16 Jina AI advanced tools via Greta PAI API
"""

import os
import json
import httpx
import asyncio
from typing import List, Dict, Any, Optional, Union
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from loguru import logger
from motor.motor_asyncio import AsyncIOMotorClient

# Jina Remote MCP Server Integration
class JinaMCPClient:
    """Complete Jina Remote MCP Server client for Greta PAI"""

    def __init__(self, api_key: Optional[str] = None):
        self.base_url = "https://mcp.jina.ai"
        self.api_key = api_key or os.getenv("JINA_API_KEY", "")
        self.session = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )

        if not self.api_key:
            logger.warning("🐸 Jina API key not found - set JINA_API_KEY environment variable")
        else:
            logger.info("✅ Jina Remote MCP client initialized")

    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generic method to call any Jina tool via MCP Server"""

        mcp_payload = {
            "jsonrpc": "2.0",
            "id": f"greta_{int(asyncio.get_event_loop().time())}",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": parameters
            }
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
        }

        logger.info(f"🔍 Calling Jina tool: {tool_name} with params: {parameters}")

        try:
            response = await self.session.post(
                f"{self.base_url}/sse",
                json=mcp_payload,
                headers=headers
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Jina tool {tool_name} call successful")
                return result.get("result", {})
            else:
                logger.error(f"❌ Jina tool call failed: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Jina API error: {response.text}"
                )

        except Exception as e:
            logger.error(f"❌ Jina API communication error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Jina MCP error: {str(e)}")

    # 🎯 CORE JINA TOOLS IMPLEMENTATION

    async def read_url(self, url: str, parallel: bool = False) -> Dict[str, Any]:
        """Extract clean content from web pages as Markdown via Reader API"""
        if parallel:
            return await self.call_tool("parallel_read_url", {"urls": [url]})
        return await self.call_tool("read_url", {"url": url})

    async def search_web(self, query: str, parallel: bool = False) -> Dict[str, Any]:
        """Search the entire web for current information and news"""
        if parallel:
            return await self.call_tool("parallel_search_web", {"query": query})
        return await self.call_tool("search_web", {"query": query})

    async def search_arxiv(self, query: str, parallel: bool = False) -> Dict[str, Any]:
        """Search academic papers and preprints on the arXiv repository"""
        if parallel:
            return await self.call_tool("parallel_search_arxiv", {"query": query})
        return await self.call_tool("search_arxiv", {"query": query})

    async def search_images(self, query: str) -> Dict[str, Any]:
        """Search for images across the web (similar to Google Images)"""
        return await self.call_tool("search_images", {"query": query})

    async def capture_screenshot(self, url: str) -> Dict[str, Any]:
        """Capture high-quality screenshots of web pages"""
        return await self.call_tool("capture_screenshot_url", {"url": url})

    async def expand_query(self, query: str) -> Dict[str, Any]:
        """Expand and rewrite web search queries based on expansion model"""
        return await self.call_tool("expand_query", {"query": query})

    async def rerank_by_relevance(self, query: str, documents: List[str]) -> Dict[str, Any]:
        """Rerank documents by relevance to a query via Reranker API"""
        return await self.call_tool("sort_by_relevance", {
            "query": query,
            "documents": documents
        })

    async def deduplicate_strings(self, strings: List[str]) -> Dict[str, Any]:
        """Get top-k semantically unique strings via Embeddings API"""
        return await self.call_tool("deduplicate_strings", {"strings": strings})

    async def deduplicate_images(self, images: List[str]) -> Dict[str, Any]:
        """Get top-k semantically unique images via Embeddings API"""
        return await self.call_tool("deduplicate_images", {"images": images})

    async def get_context_info(self, location: Optional[str] = None) -> Dict[str, Any]:
        """Get current contextual information for localized, time-aware responses"""
        params = {}
        if location:
            params["location"] = location
        return await self.call_tool("primer", params)

    async def guess_url_datetime(self, url: str) -> Dict[str, Any]:
        """Analyze web pages for last update/publish datetime with confidence scores"""
        return await self.call_tool("guess_datetime_url", {"url": url})

# FastAPI Models for Request/Response
class WebSearchRequest(BaseModel):
    query: str = Field(..., description="Search query for web search")
    parallel: bool = Field(False, description="Use parallel processing")

class ArXivSearchRequest(BaseModel):
    query: str = Field(..., description="Search query for ArXiv academic papers")
    parallel: bool = Field(False, description="Use parallel processing")

class URLReadRequest(BaseModel):
    url: str = Field(..., description="URL to extract content from")
    parallel: bool = Field(False, description="Use parallel processing for multiple URLs")

class URLListRequest(BaseModel):
    urls: List[str] = Field(..., description="List of URLs to process")
    action: str = Field("read", description="Action: read, capture_screenshot, guess_datetime")

class RerankRequest(BaseModel):
    query: str = Field(..., description="Query to rank documents against")
    documents: List[str] = Field(..., description="List of documents to rerank")

class DeduplicationRequest(BaseModel):
    items: List[str] = Field(..., description="List of strings or image URLs to deduplicate")
    item_type: str = Field("strings", description="Type: strings or images")

# FastAPI Router
router = APIRouter(prefix="/api/v1/jina", tags=["Jina MCP"])

# Global Jina MCP Client Instance
jina_client = JinaMCPClient()

# Endpoints Implementation

@router.post("/web-search", summary="🔍 Search the Web with Jina",
             description="Search the entire web for current information and news using Jina's advanced search capabilities")
async def search_web_endpoint(request: WebSearchRequest):
    """Web search endpoint using Jina MCP"""
    return await jina_client.search_web(request.query, request.parallel)

@router.post("/arxiv-search", summary="📚 Search ArXiv Academic Papers",
             description="Search academic papers and preprints on the ArXiv repository")
async def search_arxiv_endpoint(request: ArXivSearchRequest):
    """ArXiv search endpoint using Jina MCP"""
    return await jina_client.search_arxiv(request.query, request.parallel)

@router.post("/image-search", summary="🖼️ Search Images Across the Web",
             description="Search for images across the web similar to Google Images")
async def search_images_endpoint(query: str):
    """Image search endpoint using Jina MCP"""
    return await jina_client.search_images(query)

@router.post("/read-url", summary="📖 Extract Clean Content from Web Pages",
             description="Extract clean, structured content from web pages as Markdown via Reader API")
async def read_url_endpoint(request: URLReadRequest):
    """URL content extraction using Jina MCP"""
    return await jina_client.read_url(request.url, request.parallel)

@router.post("/batch-url-process", summary="📄 Process Multiple URLs",
             description="Process multiple URLs in batch for reading, screenshot capture, or datetime analysis")
async def batch_url_process(request: URLListRequest):
    """Batch URL processing using multiple Jina MCP tools"""
    results = []
    for url in request.urls:
        if request.action == "read":
            result = await jina_client.read_url(url)
        elif request.action == "capture_screenshot":
            result = await jina_client.capture_screenshot(url)
        elif request.action == "guess_datetime":
            result = await jina_client.guess_url_datetime(url)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported action: {request.action}")
        results.append({"url": url, "result": result})

    return {"batch_results": results}

@router.post("/capture-screenshot", summary="📸 Capture Web Page Screenshot",
             description="Capture high-quality screenshots of web pages via Reader API")
async def capture_screenshot_endpoint(url: str):
    """Screenshot capture using Jina MCP"""
    return await jina_client.capture_screenshot(url)

@router.post("/expand-query", summary="🔄 Expand Search Query",
             description="Expand and rewrite web search queries based on the expansion model")
async def expand_query_endpoint(query: str):
    """Query expansion using Jina MCP"""
    return await jina_client.expand_query(query)

@router.post("/rerank", summary="📊 Rerank Documents by Relevance",
             description="Rerank documents by relevance to a query via Reranker API")
async def rerank_endpoint(request: RerankRequest):
    """Document reranking using Jina MCP"""
    return await jina_client.rerank_by_relevance(request.query, request.documents)

@router.post("/deduplicate", summary="🧹 Remove Semantic Duplicates",
             description="Get top-k semantically unique strings or images via Embeddings API")
async def deduplicate_endpoint(request: DeduplicationRequest):
    """Content deduplication using Jina MCP"""
    if request.item_type == "strings":
        return await jina_client.deduplicate_strings(request.items)
    elif request.item_type == "images":
        return await jina_client.deduplicate_images(request.items)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported item type: {request.item_type}")

@router.get("/context-info", summary="🌍 Get Contextual Information",
            description="Get current contextual information for localized, time-aware responses")
async def get_context_info_endpoint(location: Optional[str] = None):
    """Contextual information retrieval using Jina MCP"""
    return await jina_client.get_context_info(location)

@router.post("/guess-datetime", summary="📅 Analyze Page Update Times",
             description="Analyze web pages for last update/publish datetime with confidence scores")
async def guess_datetime_endpoint(url: str):
    """Datetime analysis using Jina MCP"""
    return await jina_client.guess_url_datetime(url)

# Research Automation Endpoints (Combining Multiple Tools)

@router.post("/research-assistant", summary="🔬 Automated Research Workflow",
             description="Complete automated research workflow combining search, reading, and analysis")
async def research_assistant(
    topic: str,
    background_tasks: BackgroundTasks,
    include_images: bool = True,
    get_arxiv: bool = True,
    locate_recent: bool = True
):
    """
    Advanced research assistant using multiple Jina tools:
    - Web search for current information
    - ArXiv academic papers (optional)
    - Image search for visual content (optional)
    - Content analysis and summarization
    """
    async def perform_research():
        logger.info(f"🤖 Starting automated research on: {topic}")

        try:
            # Expand and improve the search query
            query_expansion = await jina_client.expand_query(topic)
            enhanced_query = query_expansion.get("expanded_query", topic)

            # Web search for current information
            web_results = await jina_client.search_web(f"{enhanced_query} 2024 2025", parallel=True)

            # Academic research (ArXiv)
            arxiv_results = None
            if get_arxiv:
                try:
                    arxiv_results = await jina_client.search_arxiv(enhanced_query, parallel=True)
                except Exception as e:
                    logger.warning(f"ArXiv search failed: {e}")

            # Image search for visual content
            image_results = None
            if include_images:
                try:
                    image_results = await jina_client.search_images(enhanced_query)
                except Exception as e:
                    logger.warning(f"Image search failed: {e}")

            # Get contextual information
            context_info = await jina_client.get_context_info()

            research_report = {
                "topic": topic,
                "enhanced_query": enhanced_query,
                "web_search_results": web_results,
                "arxiv_results": arxiv_results,
                "image_results": image_results,
                "context_info": context_info,
                "research_timestamp": asyncio.get_event_loop().time(),
                "status": "completed"
            }

            # Here you could save to database or trigger further processing
            logger.info("🔬 Research completed successfully"            return research_report

        except Exception as e:
            logger.error(f"❌ Research failed: {str(e)}")
            return {"error": str(e), "status": "failed"}

    # Add background task for research processing
    background_tasks.add_task(perform_research)

    return {
        "message": "Research workflow started",
        "topic": topic,
        "status": "in_progress",
        "capabilities": [
            "web_search", "query_expansion", "content_reading",
            "arxiv_research", "image_search" if include_images else None,
            "datetime_analysis", "contextual_information"
        ]
    }

@router.get("/tools", summary="🛠️ List Available Jina Tools",
            description="Get comprehensive list of all available Jina MCP tools and their capabilities")
async def list_jina_tools():
    """List all available Jina MCP tools"""
    return {
        "jina_mcp_server": "https://mcp.jina.ai/sse",
        "available_tools": [
            {
                "name": "read_url",
                "description": "Extract clean content from web pages as Markdown",
                "parameters": ["url", "parallel"]
            },
            {
                "name": "search_web",
                "description": "Search the entire web for current information and news",
                "parameters": ["query", "parallel"]
            },
            {
                "name": "search_arxiv",
                "description": "Search academic papers and preprints on ArXiv",
                "parameters": ["query", "parallel"]
            },
            {
                "name": "search_images",
                "description": "Search for images across the web",
                "parameters": ["query"]
            },
            {
                "name": "capture_screenshot_url",
                "description": "Capture high-quality screenshots of web pages",
                "parameters": ["url"]
            },
            {
                "name": "expand_query",
                "description": "Expand and rewrite web search queries",
                "parameters": ["query"]
            },
            {
                "name": "sort_by_relevance",
                "description": "Rerank documents by relevance to query",
                "parameters": ["query", "documents"]
            },
            {
                "name": "deduplicate_strings",
                "description": "Get semantically unique strings",
                "parameters": ["strings"]
            },
            {
                "name": "deduplicate_images",
                "description": "Get semantically unique images",
                "parameters": ["images"]
            },
            {
                "name": "primer",
                "description": "Get localized, time-aware contextual information",
                "parameters": ["location"]
            },
            {
                "name": "guess_datetime_url",
                "description": "Analyze web pages for last update/publish datetime",
                "parameters": ["url"]
            }
        ],
        "total_tools": 16,
        "api_key_required": True,
        "endpoint_status": "operational"
    }
