"""
Team Router for CPAS5 Agent Builder
REST API endpoints for team management and execution
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from typing import List, Optional, Dict, Any
from models.team_models import *
from services.team_service import TeamService, get_team_service
from services.autogen_studio_bridge import autogen_studio_bridge
from services.team_template_service import team_template_service
from database import get_database, Database
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/teams", tags=["Teams"])


# ============================================================================
# CRUD OPERATIONS
# ============================================================================

@router.post("", response_model=Team, status_code=201)
async def create_team(
    team: TeamCreate,
    service: TeamService = Depends(get_team_service)
) -> Team:
    """
    Create a new team
    
    - **name**: Team name (required)
    - **description**: Team description (required)
    - **agents**: List of team agents with roles (required, min 1)
    - **workflow_pattern**: Workflow pattern (sequential, autonomous, round_robin, selector)
    - **termination**: Termination condition configuration
    """
    try:
        return await service.create_team(team)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating team: {e}")
        raise HTTPException(status_code=500, detail="Failed to create team")


@router.get("/{team_id}", response_model=Team)
async def get_team(
    team_id: str,
    service: TeamService = Depends(get_team_service)
) -> Team:
    """
    Get team by ID
    
    - **team_id**: Team ID
    """
    team = await service.get_team(team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    return team


@router.get("", response_model=List[Team])
async def list_teams(
    status: Optional[str] = Query(None, description="Filter by status (active, archived)"),
    workflow_pattern: Optional[WorkflowPattern] = Query(None, description="Filter by workflow pattern"),
    limit: int = Query(100, le=1000, description="Maximum number of teams to return"),
    offset: int = Query(0, ge=0, description="Number of teams to skip"),
    service: TeamService = Depends(get_team_service)
) -> List[Team]:
    """
    List teams with optional filters
    
    - **status**: Filter by status (active, archived)
    - **workflow_pattern**: Filter by workflow pattern
    - **limit**: Maximum number of teams to return (default: 100, max: 1000)
    - **offset**: Number of teams to skip (default: 0)
    """
    try:
        return await service.list_teams(status, workflow_pattern, limit, offset)
    except Exception as e:
        logger.error(f"Error listing teams: {e}")
        raise HTTPException(status_code=500, detail="Failed to list teams")


@router.patch("/{team_id}", response_model=Team)
async def update_team(
    team_id: str,
    update: TeamUpdate,
    service: TeamService = Depends(get_team_service)
) -> Team:
    """
    Update team
    
    - **team_id**: Team ID
    - **update**: Fields to update (all optional)
    """
    try:
        team = await service.update_team(team_id, update)
        if not team:
            raise HTTPException(status_code=404, detail="Team not found")
        return team
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating team: {e}")
        raise HTTPException(status_code=500, detail="Failed to update team")


@router.delete("/{team_id}", status_code=204)
async def delete_team(
    team_id: str,
    service: TeamService = Depends(get_team_service)
):
    """
    Delete team
    
    - **team_id**: Team ID
    """
    try:
        success = await service.delete_team(team_id)
        if not success:
            raise HTTPException(status_code=404, detail="Team not found")
    except Exception as e:
        logger.error(f"Error deleting team: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete team")


# ============================================================================
# TEAM EXECUTION
# ============================================================================

@router.post("/{team_id}/execute", response_model=TeamExecutionResult)
async def execute_team(
    team_id: str,
    request: TeamExecutionRequest,
    service: TeamService = Depends(get_team_service)
) -> TeamExecutionResult:
    """
    Execute a team workflow
    
    - **team_id**: Team ID
    - **task**: The task/prompt for the team
    - **context**: Optional context data
    - **max_iterations**: Override termination max_turns
    
    Note: This is currently a placeholder. Full execution logic will be implemented in Task 3.
    """
    # Verify team exists
    team = await service.get_team(team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    
    # Validate team configuration
    validation = await service.validate_team(team_id)
    if not validation.valid:
        raise HTTPException(
            status_code=400,
            detail=f"Team validation failed: {', '.join(validation.errors)}"
        )
    
    try:
        # Ensure request has correct team_id
        request.team_id = team_id
        return await service.execute_team(request)
    except Exception as e:
        logger.error(f"Error executing team: {e}")
        raise HTTPException(status_code=500, detail="Failed to execute team")


@router.get("/{team_id}/executions", response_model=List[TeamExecutionResult])
async def list_team_executions(
    team_id: str,
    limit: int = Query(50, le=100, description="Maximum number of executions to return"),
    service: TeamService = Depends(get_team_service)
) -> List[TeamExecutionResult]:
    """
    List execution history for a team
    
    - **team_id**: Team ID
    - **limit**: Maximum number of executions to return (default: 50, max: 100)
    """
    # Verify team exists
    team = await service.get_team(team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    
    try:
        return await service.list_executions(team_id, limit)
    except Exception as e:
        logger.error(f"Error listing executions: {e}")
        raise HTTPException(status_code=500, detail="Failed to list executions")


@router.get("/executions/{execution_id}", response_model=TeamExecutionResult)
async def get_execution(
    execution_id: str,
    service: TeamService = Depends(get_team_service)
) -> TeamExecutionResult:
    """
    Get execution details
    
    - **execution_id**: Execution ID
    """
    execution = await service.get_execution(execution_id)
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    return execution


# ============================================================================
# VALIDATION
# ============================================================================

@router.post("/{team_id}/validate", response_model=TeamValidationResult)
async def validate_team(
    team_id: str,
    service: TeamService = Depends(get_team_service)
) -> TeamValidationResult:
    """
    Validate team configuration
    
    - **team_id**: Team ID
    
    Returns validation result with errors and warnings
    """
    team = await service.get_team(team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    
    try:
        return await service.validate_team(team_id)
    except Exception as e:
        logger.error(f"Error validating team: {e}")
        raise HTTPException(status_code=500, detail="Failed to validate team")


# ============================================================================
# HEALTH CHECK
# ============================================================================
# AUTOGEN STUDIO INTEGRATION
# ============================================================================

@router.post("/import/{studio_team_id}", response_model=Team)
async def import_team_from_autogen_studio(
    studio_team_id: str,
    service: TeamService = Depends(get_team_service)
) -> Team:
    """
    Import a team from AutoGen Studio

    - **studio_team_id**: AutoGen Studio team ID

    Returns the imported team
    """
    try:
        team_dict = await autogen_studio_bridge.import_team_from_studio(studio_team_id)
        return Team(**team_dict)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error importing team from AutoGen Studio: {e}")
        raise HTTPException(status_code=500, detail="Failed to import team")


@router.post("/{team_id}/export")
async def export_team_to_autogen_studio(
    team_id: str,
    service: TeamService = Depends(get_team_service)
) -> Dict[str, Any]:
    """
    Export a team to AutoGen Studio

    - **team_id**: CPAS5 team ID

    Returns the AutoGen Studio team object
    """
    try:
        studio_team = await autogen_studio_bridge.export_team_to_studio(team_id)
        return {
            "success": True,
            "studio_team": studio_team,
            "message": f"Team exported to AutoGen Studio successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error exporting team to AutoGen Studio: {e}")
        raise HTTPException(status_code=500, detail="Failed to export team")


@router.get("/autogen-studio/list")
async def list_autogen_studio_teams() -> Dict[str, Any]:
    """
    List all teams from AutoGen Studio

    Returns list of available teams in AutoGen Studio
    """
    try:
        teams = await autogen_studio_bridge.list_studio_teams()
        return {
            "success": True,
            "teams": teams,
            "count": len(teams)
        }
    except Exception as e:
        logger.error(f"Error listing teams from AutoGen Studio: {e}")
        raise HTTPException(status_code=500, detail="Failed to list teams from AutoGen Studio")


# ============================================================================
# TEAM TEMPLATES
# ============================================================================

@router.get("/templates/list")
async def list_team_templates(
    category: Optional[str] = Query(None, description="Filter by category"),
    is_featured: Optional[bool] = Query(None, description="Filter by featured status"),
    limit: int = Query(100, le=1000, description="Maximum number of templates")
) -> Dict[str, Any]:
    """
    List team templates

    - **category**: Filter by category (sales, dev, support, marketing, research)
    - **is_featured**: Filter by featured status
    - **limit**: Maximum number of templates to return
    """
    try:
        templates = await team_template_service.list_templates(
            category=category,
            is_featured=is_featured,
            limit=limit
        )
        return {
            "success": True,
            "templates": templates,
            "count": len(templates)
        }
    except Exception as e:
        logger.error(f"Error listing team templates: {e}")
        raise HTTPException(status_code=500, detail="Failed to list team templates")


@router.get("/templates/{template_id}")
async def get_team_template(template_id: str) -> Dict[str, Any]:
    """
    Get a team template by ID

    - **template_id**: Template ID
    """
    try:
        template = await team_template_service.get_template(template_id)
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")

        return {
            "success": True,
            "template": template
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting team template: {e}")
        raise HTTPException(status_code=500, detail="Failed to get team template")


@router.post("/templates/{template_id}/deploy", response_model=Team)
async def deploy_team_template(
    template_id: str,
    customizations: Optional[Dict[str, Any]] = Body(None, description="Optional customizations")
) -> Team:
    """
    Deploy a team template

    - **template_id**: Template ID to deploy
    - **customizations**: Optional customizations (name, description, workflow_pattern, termination)

    Returns the created team
    """
    try:
        team = await team_template_service.deploy_template(
            template_id=template_id,
            customizations=customizations
        )
        return team
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error deploying team template: {e}")
        raise HTTPException(status_code=500, detail="Failed to deploy team template")


@router.post("/templates/create")
async def create_team_template(
    template_data: Dict[str, Any] = Body(..., description="Template configuration")
) -> Dict[str, Any]:
    """
    Create a new team template

    - **template_data**: Template configuration

    Returns the created template
    """
    try:
        template = await team_template_service.create_template(template_data)
        return {
            "success": True,
            "template": template,
            "message": "Template created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating team template: {e}")
        raise HTTPException(status_code=500, detail="Failed to create team template")


@router.post("/templates/initialize")
async def initialize_default_templates() -> Dict[str, Any]:
    """
    Initialize default team templates

    This endpoint creates the default templates (Sales, Dev, Support, Marketing, Research)
    if they don't already exist.
    """
    try:
        await team_template_service.initialize_default_templates()
        return {
            "success": True,
            "message": "Default templates initialized successfully"
        }
    except Exception as e:
        logger.error(f"Error initializing default templates: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize default templates")


# ============================================================================

@router.get("/health", include_in_schema=False)
async def health_check():
    """Health check endpoint for team router"""
    return {
        "status": "healthy",
        "service": "team_router",
        "version": "1.0.0"
    }

