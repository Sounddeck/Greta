"""
🎯 Skills Router - API endpoints for Skills System
"""

import logging
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
import time

from models.skill_models import (
    SkillResponse,
    SkillListResponse,
    SkillMatchRequest,
    SkillMatchResponse,
    SkillExecutionRequest,
    SkillExecutionResponse
)
from services.skills_service import skills_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/skills", tags=["Skills"])


@router.get("/", response_model=SkillListResponse)
async def list_skills(category: Optional[str] = None):
    """
    List all available skills
    
    Args:
        category: Optional category filter (development, orchestration, analysis, optimization)
    
    Returns:
        List of skills with metadata
    """
    try:
        skills = skills_service.list_skills(category=category)
        
        # Get unique categories
        categories = list(set(s.category for s in skills_service.list_skills()))
        
        return SkillListResponse(
            skills=[
                SkillResponse(
                    name=s.name,
                    display_name=s.display_name,
                    description=s.description,
                    tags=s.tags,
                    category=s.category,
                    activation_keywords=s.activation_keywords,
                    activation_patterns=s.activation_patterns,
                    handler_function=s.handler_function,
                    context_requirements=s.context_requirements,
                    priority=s.priority,
                    examples=s.examples
                )
                for s in skills
            ],
            total=len(skills),
            categories=sorted(categories)
        )
    except Exception as e:
        logger.error(f"Error listing skills: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{skill_name}", response_model=SkillResponse)
async def get_skill(skill_name: str):
    """
    Get details of a specific skill
    
    Args:
        skill_name: Name of the skill
    
    Returns:
        Skill details
    """
    try:
        skill = skills_service.get_skill(skill_name)
        if not skill:
            raise HTTPException(status_code=404, detail=f"Skill not found: {skill_name}")
        
        return SkillResponse(
            name=skill.name,
            display_name=skill.display_name,
            description=skill.description,
            tags=skill.tags,
            category=skill.category,
            activation_keywords=skill.activation_keywords,
            activation_patterns=skill.activation_patterns,
            handler_function=skill.handler_function,
            context_requirements=skill.context_requirements,
            priority=skill.priority,
            examples=skill.examples
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting skill {skill_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/match", response_model=SkillMatchResponse)
async def match_skill(request: SkillMatchRequest):
    """
    Match user input to the most appropriate skill
    
    Args:
        request: Match request with user input and optional context
    
    Returns:
        Matched skill with confidence score
    """
    try:
        match = await skills_service.match_skill(
            user_input=request.user_input,
            context=request.context,
            threshold=request.threshold
        )
        
        if match:
            return SkillMatchResponse(
                matched=True,
                skill=SkillResponse(
                    name=match.skill.name,
                    display_name=match.skill.display_name,
                    description=match.skill.description,
                    tags=match.skill.tags,
                    category=match.skill.category,
                    activation_keywords=match.skill.activation_keywords,
                    activation_patterns=match.skill.activation_patterns,
                    handler_function=match.skill.handler_function,
                    context_requirements=match.skill.context_requirements,
                    priority=match.skill.priority,
                    examples=match.skill.examples
                ),
                confidence=match.confidence,
                matched_keywords=match.matched_keywords,
                matched_patterns=match.matched_patterns
            )
        else:
            # Provide suggestions for common skills
            all_skills = skills_service.list_skills()
            suggestions = [
                f"{s.display_name}: {s.examples[0] if s.examples else s.description}"
                for s in sorted(all_skills, key=lambda x: x.priority, reverse=True)[:3]
            ]
            
            return SkillMatchResponse(
                matched=False,
                suggestions=suggestions
            )
    except Exception as e:
        logger.error(f"Error matching skill: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute", response_model=SkillExecutionResponse)
async def execute_skill(request: SkillExecutionRequest):
    """
    Execute a skill by name
    
    Args:
        request: Execution request with skill name, user input, and context
    
    Returns:
        Execution result
    """
    try:
        start_time = time.time()
        
        result = await skills_service.execute_skill(
            skill_name=request.skill_name,
            user_input=request.user_input,
            context=request.context
        )
        
        execution_time = time.time() - start_time
        
        return SkillExecutionResponse(
            success=result.get("success", False),
            skill=result.get("skill", request.skill_name),
            result=result.get("result"),
            error=result.get("error"),
            execution_time=execution_time
        )
    except Exception as e:
        logger.error(f"Error executing skill: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/categories/list")
async def list_categories():
    """
    List all skill categories
    
    Returns:
        List of unique categories
    """
    try:
        skills = skills_service.list_skills()
        categories = list(set(s.category for s in skills))
        
        return {
            "categories": sorted(categories),
            "total": len(categories)
        }
    except Exception as e:
        logger.error(f"Error listing categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

