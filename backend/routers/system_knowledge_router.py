"""
System Knowledge Base Router
Provides endpoints for managing and retrieving system knowledge for Master Agent
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId
from typing import List, Optional
from datetime import datetime
import logging

from database import get_database
from models.knowledge_base_models import (
    SystemKnowledgeEntry,
    SystemKnowledgeResponse,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    KnowledgeListResponse,
    KnowledgeCategory,
    MasterAgentContextRequest,
    MasterAgentContextResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/knowledge", tags=["System Knowledge"])

# Collection name
KNOWLEDGE_COLLECTION = "system_knowledge"


async def get_knowledge_collection():
    """Get knowledge collection from database"""
    db = await get_database()
    return db.db[KNOWLEDGE_COLLECTION]


@router.get("/system", response_model=KnowledgeListResponse)
async def get_system_knowledge(
    category: Optional[KnowledgeCategory] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    collection = Depends(get_knowledge_collection)
):
    """
    Retrieve all system knowledge entries with optional filtering
    
    - **category**: Filter by knowledge category
    - **page**: Page number (1-indexed)
    - **page_size**: Number of entries per page
    """
    try:
        # Build query
        query = {}
        if category:
            query["category"] = category.value

        # Get total count
        total = await collection.count_documents(query)

        # Calculate pagination
        skip = (page - 1) * page_size
        total_pages = (total + page_size - 1) // page_size

        # Fetch entries
        cursor = collection.find(query).sort("priority", -1).skip(skip).limit(page_size)
        entries = []
        async for doc in cursor:
            doc["id"] = str(doc["_id"])
            entries.append(SystemKnowledgeResponse(**doc))

        return KnowledgeListResponse(
            entries=entries,
            total=total,
            category=category.value if category else None,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )
    except Exception as e:
        logger.error(f"Error retrieving system knowledge: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/search", response_model=KnowledgeSearchResponse)
async def search_knowledge(
    query: str = Query(..., min_length=1),
    category: Optional[KnowledgeCategory] = Query(None),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    collection = Depends(get_knowledge_collection)
):
    """
    Search system knowledge by keywords
    
    - **query**: Search query string
    - **category**: Optional category filter
    - **limit**: Maximum results to return
    - **offset**: Number of results to skip
    """
    try:
        # Build search query
        search_query = {
            "$or": [
                {"title": {"$regex": query, "$options": "i"}},
                {"content": {"$regex": query, "$options": "i"}},
                {"tags": {"$regex": query, "$options": "i"}}
            ]
        }
        
        if category:
            search_query["category"] = category.value

        # Get total count
        total = await collection.count_documents(search_query)

        # Fetch results
        cursor = collection.find(search_query).sort("priority", -1).skip(offset).limit(limit)
        results = []
        async for doc in cursor:
            doc["id"] = str(doc["_id"])
            results.append(SystemKnowledgeResponse(**doc))

        return KnowledgeSearchResponse(
            results=results,
            total=total,
            query=query,
            category=category.value if category else None
        )
    except Exception as e:
        logger.error(f"Error searching knowledge: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/system", response_model=SystemKnowledgeResponse)
async def create_knowledge_entry(
    entry: SystemKnowledgeEntry,
    collection = Depends(get_knowledge_collection)
):
    """
    Create a new system knowledge entry
    
    Admin only - requires authentication in production
    """
    try:
        entry_dict = entry.dict()
        entry_dict["created_at"] = datetime.utcnow()
        entry_dict["updated_at"] = datetime.utcnow()
        
        result = await collection.insert_one(entry_dict)
        entry_dict["id"] = str(result.inserted_id)
        
        logger.info(f"Created knowledge entry: {entry.title}")
        return SystemKnowledgeResponse(**entry_dict)
    except Exception as e:
        logger.error(f"Error creating knowledge entry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/system/{entry_id}", response_model=SystemKnowledgeResponse)
async def update_knowledge_entry(
    entry_id: str,
    entry: SystemKnowledgeEntry,
    collection = Depends(get_knowledge_collection)
):
    """
    Update an existing system knowledge entry
    
    Admin only - requires authentication in production
    """
    try:
        if not ObjectId.is_valid(entry_id):
            raise HTTPException(status_code=400, detail="Invalid entry ID")

        entry_dict = entry.dict()
        entry_dict["updated_at"] = datetime.utcnow()
        
        result = await collection.update_one(
            {"_id": ObjectId(entry_id)},
            {"$set": entry_dict}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Entry not found")

        # Fetch updated entry
        updated_doc = await collection.find_one({"_id": ObjectId(entry_id)})
        updated_doc["id"] = str(updated_doc["_id"])
        
        logger.info(f"Updated knowledge entry: {entry.title}")
        return SystemKnowledgeResponse(**updated_doc)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating knowledge entry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/system/{entry_id}")
async def delete_knowledge_entry(
    entry_id: str,
    collection = Depends(get_knowledge_collection)
):
    """
    Delete a system knowledge entry
    
    Admin only - requires authentication in production
    """
    try:
        if not ObjectId.is_valid(entry_id):
            raise HTTPException(status_code=400, detail="Invalid entry ID")

        result = await collection.delete_one({"_id": ObjectId(entry_id)})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Entry not found")

        logger.info(f"Deleted knowledge entry: {entry_id}")
        return {"success": True, "message": "Entry deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting knowledge entry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/system/context", response_model=MasterAgentContextResponse)
async def get_master_agent_context(
    request: MasterAgentContextRequest,
    collection = Depends(get_knowledge_collection)
):
    """
    Get comprehensive context for Master Agent initialization
    
    Returns curated knowledge entries for Master Agent system prompt
    """
    try:
        # Build query
        query = {"priority": {"$gte": request.min_priority}}
        if request.include_categories:
            query["category"] = {"$in": [cat.value for cat in request.include_categories]}

        # Fetch high-priority entries
        cursor = collection.find(query).sort("priority", -1).limit(request.max_entries)
        entries = []
        async for doc in cursor:
            doc["id"] = str(doc["_id"])
            entries.append(SystemKnowledgeResponse(**doc))

        # Extract specific information for context
        system_overview = ""
        available_endpoints = []
        capabilities = []
        workflows = []
        configuration_notes = ""

        for entry in entries:
            if entry.category == KnowledgeCategory.ARCHITECTURE:
                system_overview = entry.content
            elif entry.category == KnowledgeCategory.API_ENDPOINTS:
                available_endpoints.append({
                    "title": entry.title,
                    "description": entry.content[:200]
                })
            elif entry.category == KnowledgeCategory.CAPABILITIES:
                capabilities.extend(entry.tags)
            elif entry.category == KnowledgeCategory.WORKFLOWS:
                workflows.append({
                    "title": entry.title,
                    "description": entry.content[:200]
                })
            elif entry.category == KnowledgeCategory.CONFIGURATION:
                configuration_notes = entry.content

        return MasterAgentContextResponse(
            system_overview=system_overview,
            available_endpoints=available_endpoints,
            capabilities=capabilities,
            workflows=workflows,
            configuration_notes=configuration_notes,
            knowledge_entries=entries
        )
    except Exception as e:
        logger.error(f"Error getting Master Agent context: {e}")
        raise HTTPException(status_code=500, detail=str(e))

