"""
GRETA Central Orchestrator Service
The master coordination system that can route commands to any GRETA capability
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import re

# Import all router modules for coordination
try:
    from . import llm, voice, memory, agent, master_agent, training, osint
    from . import reasoning, autonomous, multimodal, predictive, workflows
    from . import system, integrations, web_commands
    ROUTERS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some routers not available for orchestration: {e}")
    ROUTERS_AVAILABLE = False

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic Models
class CommandRequest(BaseModel):
    command: str = Field(..., description="Natural language command or specific instruction")
    mode: str = Field(default="text", description="Command mode: 'text' or 'voice'")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    priority: str = Field(default="normal", description="Priority: 'low', 'normal', 'high', 'urgent'")
    async_execution: bool = Field(default=False, description="Execute asynchronously")

class OrchestrationResponse(BaseModel):
    success: bool
    command_id: str
    routers_invoked: List[str]
    results: Dict[str, Any]
    execution_time: float
    mode: str
    status: str

class SystemStatusResponse(BaseModel):
    orchestrator_status: str
    available_routers: List[str]
    active_commands: int
    total_commands_processed: int
    system_health: str

# Command routing patterns
COMMAND_PATTERNS = {
    # LLM and Text Generation
    r'(generate|create|write|compose|draft).*?(text|content|document|article|email|letter)': 'llm',
    r'(complete|finish|continue).*?(text|sentence|paragraph)': 'llm',
    r'(translate|summarize|analyze).*?(text|document)': 'llm',
    
    # Voice and Audio
    r'(speak|say|announce|voice|audio)': 'voice',
    r'(synthesize|generate).*?(voice|speech|audio)': 'voice',
    
    # Memory and Knowledge
    r'(remember|store|save|recall|retrieve).*?(information|data|knowledge)': 'memory',
    r'(search|find|lookup).*?(memory|knowledge|information)': 'memory',
    
    # Agent Management
    r'(create|build|deploy|manage).*?(agent|bot)': 'agent',
    r'(start|stop|pause|resume).*?(agent|bot)': 'agent',
    
    # Master Agent Control
    r'(orchestrate|coordinate|manage).*?(system|workflow|process)': 'master_agent',
    r'(execute|run).*?(complex|multi-step|workflow)': 'master_agent',
    
    # Training and Learning
    r'(train|learn|teach|educate)': 'training',
    r'(curriculum|lesson|course|skill)': 'training',
    
    # Research and OSINT
    r'(research|investigate|analyze|gather).*?(information|intelligence|data)': 'osint',
    r'(search|find|discover).*?(online|web|internet)': 'osint',
    
    # Reasoning and Analysis
    r'(reason|think|analyze|evaluate|assess)': 'reasoning',
    r'(solve|problem|decision|logic)': 'reasoning',
    
    # Autonomous Operations
    r'(automate|autonomous|self-directed|independent)': 'autonomous',
    r'(schedule|recurring|periodic|automated)': 'autonomous',
    
    # Multi-modal Processing
    r'(image|video|audio|media|file).*?(process|analyze|convert)': 'multimodal',
    r'(upload|download|convert).*?(media|file)': 'multimodal',
    
    # Predictive Analytics
    r'(predict|forecast|trend|future|analytics)': 'predictive',
    r'(data.*analysis|statistics|metrics)': 'predictive',
    
    # Workflows
    r'(workflow|pipeline|automation|process).*?(create|build|run)': 'workflows',
    r'(sequence|steps|procedure)': 'workflows',
    
    # System Management
    r'(system|status|health|monitor|performance)': 'system',
    r'(restart|shutdown|configure|settings)': 'system',
    
    # Integrations
    r'(integrate|connect|api|external|third-party)': 'integrations',
    r'(webhook|callback|notification)': 'integrations',
    
    # Web Commands
    r'(web|website|browser|url|link)': 'web_commands',
    r'(scrape|crawl|fetch).*?(web|site|page)': 'web_commands'
}

# Global state
active_commands = {}
command_counter = 0
total_commands_processed = 0

class GretaOrchestrator:
    """Central orchestration system for all GRETA capabilities"""
    
    def __init__(self):
        self.available_routers = self._discover_routers()
        self.command_history = []
        
    def _discover_routers(self) -> List[str]:
        """Discover available routers dynamically"""
        routers = []
        router_modules = [
            'llm', 'voice', 'memory', 'agent', 'master_agent', 'training',
            'osint', 'reasoning', 'autonomous', 'multimodal', 'predictive',
            'workflows', 'system', 'integrations', 'web_commands'
        ]
        
        for router_name in router_modules:
            try:
                # Check if router module exists and has router attribute
                module = globals().get(router_name)
                if module and hasattr(module, 'router'):
                    routers.append(router_name)
            except Exception as e:
                logger.debug(f"Router {router_name} not available: {e}")
                
        logger.info(f"Orchestrator discovered {len(routers)} available routers: {routers}")
        return routers
    
    def parse_command(self, command: str) -> List[str]:
        """Parse command and determine which routers to invoke"""
        command_lower = command.lower()
        matched_routers = []
        
        # Pattern matching
        for pattern, router_name in COMMAND_PATTERNS.items():
            if re.search(pattern, command_lower) and router_name in self.available_routers:
                if router_name not in matched_routers:
                    matched_routers.append(router_name)
        
        # Default fallback to master_agent for complex commands
        if not matched_routers and 'master_agent' in self.available_routers:
            matched_routers.append('master_agent')
        
        # If still no match, use LLM for text processing
        if not matched_routers and 'llm' in self.available_routers:
            matched_routers.append('llm')
            
        return matched_routers
    
    async def execute_command(self, request: CommandRequest) -> OrchestrationResponse:
        """Execute command by routing to appropriate routers"""
        global command_counter, total_commands_processed
        
        start_time = datetime.now()
        command_counter += 1
        command_id = f"cmd_{command_counter}_{int(start_time.timestamp())}"
        
        try:
            # Parse command to determine routers
            target_routers = self.parse_command(request.command)
            
            if not target_routers:
                raise HTTPException(status_code=400, detail="No suitable router found for command")
            
            logger.info(f"Executing command '{request.command}' via routers: {target_routers}")
            
            # Store active command
            active_commands[command_id] = {
                'command': request.command,
                'routers': target_routers,
                'start_time': start_time,
                'status': 'executing'
            }
            
            # Execute via routers
            results = {}
            for router_name in target_routers:
                try:
                    result = await self._invoke_router(router_name, request)
                    results[router_name] = result
                except Exception as e:
                    logger.error(f"Router {router_name} execution failed: {e}")
                    results[router_name] = {'error': str(e), 'success': False}
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update command status
            active_commands[command_id]['status'] = 'completed'
            active_commands[command_id]['execution_time'] = execution_time
            
            # Clean up completed command
            if command_id in active_commands:
                del active_commands[command_id]
            
            total_commands_processed += 1
            
            return OrchestrationResponse(
                success=True,
                command_id=command_id,
                routers_invoked=target_routers,
                results=results,
                execution_time=execution_time,
                mode=request.mode,
                status='completed'
            )
            
        except Exception as e:
            logger.error(f"Command orchestration failed: {e}")
            if command_id in active_commands:
                active_commands[command_id]['status'] = 'failed'
                active_commands[command_id]['error'] = str(e)
            
            raise HTTPException(status_code=500, detail=f"Orchestration failed: {e}")
    
    async def _invoke_router(self, router_name: str, request: CommandRequest) -> Dict[str, Any]:
        """Invoke specific router with command"""
        # This is a simplified router invocation
        # In a full implementation, each router would have standardized endpoints
        
        router_module = globals().get(router_name)
        if not router_module:
            return {'error': f'Router {router_name} not available', 'success': False}
        
        # For now, return a mock successful response
        # TODO: Implement actual router invocation based on each router's API
        return {
            'success': True,
            'router': router_name,
            'command': request.command,
            'mode': request.mode,
            'message': f'Command processed by {router_name} router',
            'timestamp': datetime.now().isoformat()
        }

# Initialize orchestrator
orchestrator = GretaOrchestrator()

# API Endpoints
@router.post("/orchestrate", response_model=OrchestrationResponse)
async def orchestrate_command(request: CommandRequest, background_tasks: BackgroundTasks):
    """
    Central orchestration endpoint - routes commands to appropriate GRETA capabilities
    """
    if request.async_execution:
        # Execute in background
        background_tasks.add_task(orchestrator.execute_command, request)
        return OrchestrationResponse(
            success=True,
            command_id=f"async_{datetime.now().timestamp()}",
            routers_invoked=[],
            results={'message': 'Command queued for background execution'},
            execution_time=0.0,
            mode=request.mode,
            status='queued'
        )
    else:
        # Execute synchronously
        return await orchestrator.execute_command(request)

@router.get("/status", response_model=SystemStatusResponse)
async def get_orchestrator_status():
    """Get orchestrator system status"""
    return SystemStatusResponse(
        orchestrator_status="active",
        available_routers=orchestrator.available_routers,
        active_commands=len(active_commands),
        total_commands_processed=total_commands_processed,
        system_health="healthy"
    )

@router.get("/routers")
async def list_available_routers():
    """List all available routers and their capabilities"""
    router_info = {}
    for router_name in orchestrator.available_routers:
        router_info[router_name] = {
            'name': router_name,
            'status': 'available',
            'description': f'{router_name.replace("_", " ").title()} capabilities'
        }
    
    return {
        'total_routers': len(orchestrator.available_routers),
        'routers': router_info
    }

@router.post("/voice-command")
async def process_voice_command(command: str, background_tasks: BackgroundTasks):
    """Process voice commands through orchestrator"""
    request = CommandRequest(
        command=command,
        mode="voice",
        priority="normal"
    )
    return await orchestrate_command(request, background_tasks)
