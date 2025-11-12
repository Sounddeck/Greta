"""
Reasoning Bank API Router

Provides REST API endpoints for managing reasoning patterns.
Inspired by ClaudeFlow's pattern storage system.

Endpoints:
- POST /patterns - Create new pattern
- GET /patterns/{pattern_id} - Get pattern by ID
- PUT /patterns/{pattern_id} - Update pattern
- POST /patterns/search - Search for patterns
- POST /patterns/{pattern_id}/use - Record pattern usage
- POST /patterns/{pattern_id}/evolve - Create evolved version
- GET /patterns/stats - Get pattern statistics
"""

import logging
from typing import List
from fastapi import APIRouter, HTTPException, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from database import get_database
from models.reasoning_pattern import (
    ReasoningPattern,
    PatternSearchQuery,
    PatternSearchResult,
    PatternCreateRequest,
    PatternUpdateRequest,
    PatternUsageRecord,
    PatternStats,
    PatternEvolutionRequest
)
from services.reasoning_bank_service import ReasoningBankService
from services.vector_search_service import VectorSearchService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reasoning-bank", tags=["Reasoning Bank"])


# Dependency to get ReasoningBank service
async def get_reasoning_bank_service(
    db: AsyncIOMotorDatabase = Depends(get_database)
) -> ReasoningBankService:
    """Get ReasoningBank service instance"""
    try:
        vector_search = VectorSearchService()
    except Exception as e:
        logger.warning(f"Vector Search not available: {e}")
        vector_search = None
    
    return ReasoningBankService(db=db, vector_search_service=vector_search)


@router.post("/patterns", response_model=ReasoningPattern)
async def create_pattern(
    request: PatternCreateRequest,
    service: ReasoningBankService = Depends(get_reasoning_bank_service)
):
    """
    Create a new reasoning pattern
    
    Stores a successful problem-solving approach for future reuse.
    """
    try:
        pattern = await service.create_pattern(request)
        return pattern
    except Exception as e:
        logger.error(f"Error creating pattern: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns/{pattern_id}", response_model=ReasoningPattern)
async def get_pattern(
    pattern_id: str,
    service: ReasoningBankService = Depends(get_reasoning_bank_service)
):
    """Get a reasoning pattern by ID"""
    pattern = await service.get_pattern(pattern_id)
    if not pattern:
        raise HTTPException(status_code=404, detail="Pattern not found")
    return pattern


@router.put("/patterns/{pattern_id}", response_model=ReasoningPattern)
async def update_pattern(
    pattern_id: str,
    request: PatternUpdateRequest,
    service: ReasoningBankService = Depends(get_reasoning_bank_service)
):
    """Update an existing reasoning pattern"""
    pattern = await service.update_pattern(pattern_id, request)
    if not pattern:
        raise HTTPException(status_code=404, detail="Pattern not found")
    return pattern


@router.post("/patterns/search", response_model=List[PatternSearchResult])
async def search_patterns(
    query: PatternSearchQuery,
    service: ReasoningBankService = Depends(get_reasoning_bank_service)
):
    """
    Search for reasoning patterns
    
    Uses hybrid search (semantic + keyword) to find relevant patterns.
    """
    try:
        results = await service.search_patterns(query)
        return results
    except Exception as e:
        logger.error(f"Error searching patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/patterns/{pattern_id}/use")
async def record_pattern_usage(
    pattern_id: str,
    usage: PatternUsageRecord,
    service: ReasoningBankService = Depends(get_reasoning_bank_service)
):
    """
    Record pattern usage
    
    Tracks effectiveness and updates success metrics.
    """
    # Ensure pattern_id matches
    usage.pattern_id = pattern_id
    
    success = await service.record_usage(usage)
    if not success:
        raise HTTPException(status_code=404, detail="Pattern not found")
    
    return {"success": True, "message": "Usage recorded successfully"}


@router.post("/patterns/{pattern_id}/evolve", response_model=ReasoningPattern)
async def evolve_pattern(
    pattern_id: str,
    request: PatternEvolutionRequest,
    service: ReasoningBankService = Depends(get_reasoning_bank_service)
):
    """
    Create an evolved version of a pattern
    
    Creates a new pattern based on an existing one with improvements.
    """
    # Ensure parent_pattern_id matches
    request.parent_pattern_id = pattern_id
    
    evolved = await service.evolve_pattern(request)
    if not evolved:
        raise HTTPException(status_code=404, detail="Parent pattern not found")
    
    return evolved


@router.get("/patterns/stats", response_model=PatternStats)
async def get_pattern_stats(
    service: ReasoningBankService = Depends(get_reasoning_bank_service)
):
    """
    Get pattern statistics
    
    Returns usage metrics, most popular patterns, etc.
    """
    try:
        stats = await service.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting pattern stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ReasoningBank",
        "version": "1.0.0"
    }

