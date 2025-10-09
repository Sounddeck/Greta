"""
CPAS4 UI Tasks Router
Provides task execution endpoints for the CPAS4 UI
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from database import get_database
from services.llm_service import get_llm_service
from services.enhanced_master_agent import EnhancedMasterAgent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tasks", tags=["Tasks"])

# Global master agent instance
master_agent_instance: Optional[EnhancedMasterAgent] = None

# Pydantic models
class TaskExecuteRequest(BaseModel):
    """Task execution request from UI"""
    agent_id: str = Field(default="master_agent", description="Agent ID")
    description: str = Field(..., description="Task description/prompt")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Additional input data")
    max_iterations: int = Field(default=3, description="Maximum iterations")

class TaskResponse(BaseModel):
    """Task execution response"""
    id: str
    status: str
    result: Optional[str] = None
    output: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None

async def get_master_agent() -> EnhancedMasterAgent:
    """Get or initialize master agent instance"""
    global master_agent_instance
    
    if master_agent_instance is None:
        try:
            from services.learning_service import LearningService
            
            db = await get_database()
            llm_service = await get_llm_service()
            learning_service = LearningService(db.client)
            await learning_service.initialize()
            
            master_agent_instance = EnhancedMasterAgent(db, llm_service, learning_service)
            await master_agent_instance.initialize()
        except Exception as e:
            logger.error(f"Failed to initialize master agent: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize master agent: {str(e)}")
    
    return master_agent_instance

@router.post("/execute")
async def execute_task(
    request: TaskExecuteRequest,
    master_agent: EnhancedMasterAgent = Depends(get_master_agent)
):
    """
    Execute a task using the master agent
    
    This endpoint is designed for the CPAS4 UI to execute tasks with different agent types.
    """
    try:
        # Extract agent type from input_data
        agent_type = request.input_data.get('agent_type', 'general')
        conversation_history = request.input_data.get('conversation_history', [])
        
        logger.info(f"Executing task for agent type: {agent_type}")
        logger.info(f"Task description: {request.description}")
        
        # Use the LLM service to generate a response
        llm_service = await get_llm_service()
        
        # Build context from conversation history
        context = ""
        if conversation_history:
            context = "\n".join([
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in conversation_history[-5:]  # Last 5 messages
            ])
        
        # Create agent-specific system prompt
        system_prompts = {
            'research': "You are a Research Agent specialized in conducting comprehensive research, gathering information, and analyzing data from multiple sources. Provide detailed, well-researched responses with citations when possible.",
            'coding': "You are a Coding Agent expert in software development, code generation, debugging, and technical problem-solving. Provide clean, well-documented code with explanations.",
            'data_analysis': "You are a Data Analysis Agent specialized in data processing, statistical analysis, visualization, and insights generation. Provide clear analytical insights with data-driven recommendations.",
            'content_creation': "You are a Content Creation Agent expert in creating engaging content, copywriting, and creative writing across various formats. Provide compelling, well-structured content.",
            'task_automation': "You are a Task Automation Agent specialized in automating workflows, creating scripts, and optimizing business processes. Provide practical automation solutions.",
            'customer_support': "You are a Customer Support Agent expert in customer service, issue resolution, and providing helpful support responses. Provide empathetic, solution-oriented responses.",
            'sales': "You are a Sales Agent specialized in sales strategies, lead generation, and customer relationship management. Provide persuasive, value-focused responses.",
            'marketing': "You are a Marketing Agent expert in marketing strategies, campaign planning, and brand development. Provide creative, strategic marketing insights.",
            'finance': "You are a Finance Agent specialized in financial analysis, budgeting, forecasting, and financial planning. Provide accurate, data-driven financial insights.",
            'general': "You are a versatile AI assistant capable of handling a wide range of tasks and queries. Provide helpful, accurate, and comprehensive responses."
        }
        
        system_prompt = system_prompts.get(agent_type, system_prompts['general'])
        
        # Build the full prompt
        full_prompt = f"{system_prompt}\n\n"
        if context:
            full_prompt += f"Previous conversation:\n{context}\n\n"
        full_prompt += f"User request: {request.description}"
        
        # Query the LLM using generate_completion
        try:
            llm_request = {
                "message": full_prompt,
                "max_tokens": 2000,
                "temperature": 0.7,
                "agent_type": agent_type
            }

            response = await llm_service.generate_completion(llm_request)

            result = response.get('response', response.get('text', 'No response generated'))

        except Exception as llm_error:
            logger.error(f"LLM query failed: {llm_error}")
            # Fallback response with agent-specific context
            result = f"I'm the {agent_type.replace('_', ' ').title()} Agent. I received your request: '{request.description}'. However, I'm currently unable to process this request due to a technical issue. Please try again or contact support if the problem persists."
        
        # Create task response
        task_id = f"task_{datetime.utcnow().timestamp()}"
        
        return {
            "id": task_id,
            "status": "completed",
            "result": result,
            "output": result,
            "created_at": datetime.utcnow().isoformat(),
            "completed_at": datetime.utcnow().isoformat(),
            "agent_type": agent_type,
            "description": request.description
        }
        
    except Exception as e:
        logger.error(f"Task execution failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Task execution failed: {str(e)}"
        )

@router.get("/{task_id}")
async def get_task(task_id: str):
    """Get task status and result"""
    # This is a simplified implementation
    # In a real system, you would store tasks in a database
    return {
        "id": task_id,
        "status": "completed",
        "message": "Task retrieval not fully implemented yet"
    }

@router.get("/")
async def list_tasks():
    """List all tasks"""
    # This is a simplified implementation
    return {
        "tasks": [],
        "total": 0,
        "message": "Task listing not fully implemented yet"
    }

@router.post("/{task_id}/cancel")
async def cancel_task(task_id: str):
    """Cancel a running task"""
    # This is a simplified implementation
    return {
        "id": task_id,
        "status": "cancelled",
        "message": "Task cancellation not fully implemented yet"
    }

