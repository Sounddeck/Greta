"""
GRETA PAI Integration Service
Cohesive integration of Greta and PAI systems with unified orchestration
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from ..config import get_settings
from ..database import get_database
from .local_llm_service import get_local_llm_service
from .greta_voice_service import get_greta_voice_service
from .hrm_integration import get_hrm_manager
from .master_agent_service import get_master_agent_service
from .mcp_hub_service import get_mcp_hub_service

logger = logging.getLogger(__name__)

class GretaPAIIntegration:
    """
    GRETA PAI Integration Service
    Cohesive integration of Greta and PAI systems with unified orchestration
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.db = None
        
        # Service instances
        self.local_llm = None
        self.voice_service = None
        self.hrm_manager = None
        self.master_agent = None
        self.mcp_hub = None
        
        # Integration state
        self.integration_state = {
            "status": "initializing",
            "services_initialized": [],
            "capabilities": {
                "local_processing": False,
                "hrm_reasoning": False,
                "voice_synthesis": False,
                "agent_management": False,
                "tool_access": False,
                "master_control": False
            },
            "orchestration_level": "basic"
        }
        
        # Processing pipeline
        self.processing_pipeline = [
            "input_analysis",
            "context_enrichment",
            "hrm_reasoning",
            "agent_coordination",
            "tool_execution",
            "response_generation",
            "voice_output"
        ]
        
        # Integration metrics
        self.integration_metrics = {
            "total_requests": 0,
            "successful_integrations": 0,
            "average_processing_time": 0,
            "pipeline_efficiency": {},
            "service_usage": {},
            "orchestration_success": 0
        }
        
    async def initialize(self):
        """Initialize Greta-PAI integration"""
        try:
            self.db = await get_database()
            
            # Initialize all services
            await self._initialize_services()
            
            # Establish integration connections
            await self._establish_integration_connections()
            
            # Initialize orchestration
            await self._initialize_orchestration()
            
            self.integration_state["status"] = "active"
            logger.info("✅ Greta-PAI Integration initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Greta-PAI Integration initialization failed: {e}")
            raise
    
    async def _initialize_services(self):
        """Initialize all core services"""
        services = [
            ("local_llm", get_local_llm_service, "local_processing"),
            ("hrm_manager", get_hrm_manager, "hrm_reasoning"),
            ("voice_service", get_greta_voice_service, "voice_synthesis"),
            ("master_agent", get_master_agent_service, "agent_management"),
            ("mcp_hub", get_mcp_hub_service, "tool_access"),
            ("master_control", get_master_agent_service, "master_control")
        ]
        
        for service_name, service_getter, capability in services:
            try:
                service = await service_getter()
                await service.initialize()
                
                # Set service instance
                setattr(self, service_name, service)
                
                # Update integration state
                self.integration_state["services_initialized"].append(service_name)
                self.integration_state["capabilities"][capability] = True
                
                logger.info(f"✅ {service_name} initialized")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize {service_name}: {e}")
    
    async def _establish_integration_connections(self):
        """Establish connections between services"""
        try:
            # Connect HRM with Local LLM
            if self.hrm_manager and self.local_llm:
                await self._connect_hrm_llm()
            
            # Connect Master Agent with all services
            if self.master_agent:
                await self._connect_master_agent()
            
            # Connect MCP Hub with agents
            if self.mcp_hub and self.master_agent:
                await self._connect_mcp_agents()
            
            # Connect Voice with response generation
            if self.voice_service:
                await self._connect_voice_output()
            
            logger.info("✅ Integration connections established")
            
        except Exception as e:
            logger.error(f"❌ Failed to establish integration connections: {e}")
    
    async def _connect_hrm_llm(self):
        """Connect HRM with Local LLM"""
        # HRM enhances LLM reasoning
        self.hrm_llm_connection = {
            "type": "reasoning_enhancement",
            "hrm_provider": self.hrm_manager,
            "llm_provider": self.local_llm,
            "enhancement_method": "hierarchical_reasoning"
        }
        
        logger.info("✅ HRM-LLM connection established")
    
    async def _connect_master_agent(self):
        """Connect Master Agent with all services"""
        self.master_agent_connections = {
            "local_llm": self.local_llm,
            "voice_service": self.voice_service,
            "hrm_manager": self.hrm_manager,
            "mcp_hub": self.mcp_hub
        }
        
        logger.info("✅ Master Agent connections established")
    
    async def _connect_mcp_agents(self):
        """Connect MCP Hub with agents"""
        self.mcp_agent_connection = {
            "master_agent": self.master_agent,
            "tool_hub": self.mcp_hub,
            "coordination_method": "session_based"
        }
        
        logger.info("✅ MCP-Agent connection established")
    
    async def _connect_voice_output(self):
        """Connect Voice with response generation"""
        self.voice_output_connection = {
            "voice_service": self.voice_service,
            "output_method": "synthesis_and_playback",
            "voice_config": {
                "gender": "female",
                "accent": "german",
                "language": "en"
            }
        }
        
        logger.info("✅ Voice output connection established")
    
    async def _initialize_orchestration(self):
        """Initialize unified orchestration"""
        self.integration_state["orchestration_level"] = "advanced"
        
        # Create orchestration rules
        self.orchestration_rules = {
            "simple_requests": ["input_analysis", "hrm_reasoning", "response_generation"],
            "complex_requests": ["input_analysis", "context_enrichment", "hrm_reasoning", "agent_coordination", "tool_execution", "response_generation"],
            "agent_requests": ["input_analysis", "agent_coordination", "tool_execution", "response_generation"],
            "voice_requests": ["input_analysis", "hrm_reasoning", "response_generation", "voice_output"],
            "code_requests": ["input_analysis", "agent_coordination", "tool_execution", "response_generation"]
        }
        
        logger.info("✅ Orchestration initialized")
    
    async def process_integrated_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process request through integrated Greta-PAI system
        """
        start_time = datetime.utcnow()
        
        try:
            self.integration_metrics["total_requests"] += 1
            
            # Analyze request type
            request_analysis = await self._analyze_request(request)
            
            # Determine processing pipeline
            pipeline = await self._determine_pipeline(request_analysis)
            
            # Execute integrated processing
            result = await self._execute_pipeline(pipeline, request, request_analysis)
            
            # Update metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            await self._update_integration_metrics(pipeline, processing_time, result.get("success", False))
            
            return {
                "success": result.get("success", False),
                "result": result,
                "request_analysis": request_analysis,
                "pipeline_used": pipeline,
                "processing_time": processing_time,
                "greta_pai_integrated": True,
                "orchestration_level": self.integration_state["orchestration_level"]
            }
            
        except Exception as e:
            logger.error(f"❌ Integrated request processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "greta_pai_integrated": True,
                "processing_time": (datetime.utcnow() - start_time).total_seconds()
            }
    
    async def _analyze_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze request to determine processing requirements"""
        try:
            message = request.get("message", "").lower()
            request_type = request.get("type", "general")
            voice_request = request.get("voice", False)
            agent_request = request.get("agent", False)
            
            # Determine complexity
            if any(keyword in message for keyword in ["hello", "hi", "simple", "quick"]):
                complexity = "simple"
            elif any(keyword in message for keyword in ["research", "analyze", "investigate", "complex"]):
                complexity = "complex"
            elif any(keyword in message for keyword in ["build", "create", "code", "deploy"]):
                complexity = "agent"
            else:
                complexity = "medium"
            
            # Determine special requirements
            special_requirements = {
                "voice_output": voice_request or "speak" in message,
                "agent_management": agent_request or "build agent" in message,
                "tool_usage": any(keyword in message for keyword in ["search", "find", "look up"]),
                "hrm_reasoning": True,  # Always use HRM for enhanced reasoning
                "code_generation": any(keyword in message for keyword in ["code", "program", "develop"])
            }
            
            return {
                "message": message,
                "request_type": request_type,
                "complexity": complexity,
                "special_requirements": special_requirements,
                "processing_priority": self._determine_priority(complexity, special_requirements)
            }
            
        except Exception as e:
            logger.error(f"❌ Request analysis failed: {e}")
            return {
                "message": request.get("message", ""),
                "request_type": "general",
                "complexity": "medium",
                "special_requirements": {},
                "processing_priority": "medium"
            }
    
    async def _determine_pipeline(self, request_analysis: Dict[str, Any]) -> List[str]:
        """Determine processing pipeline based on request analysis"""
        
        complexity = request_analysis.get("complexity", "medium")
        special_requirements = request_analysis.get("special_requirements", {})
        
        # Base pipeline
        pipeline = ["input_analysis", "hrm_reasoning", "response_generation"]
        
        # Add special requirements
        if special_requirements.get("tool_usage"):
            pipeline.append("tool_execution")
        
        if special_requirements.get("agent_management"):
            pipeline.extend(["agent_coordination", "tool_execution"])
        
        if special_requirements.get("code_generation"):
            pipeline.extend(["agent_coordination", "tool_execution"])
        
        if special_requirements.get("voice_output"):
            pipeline.append("voice_output")
        
        # Add context enrichment for complex requests
        if complexity in ["complex", "agent"]:
            pipeline.insert(1, "context_enrichment")
        
        return pipeline
    
    def _determine_priority(self, complexity: str, special_requirements: Dict[str, Any]) -> str:
        """Determine processing priority"""
        if special_requirements.get("agent_management"):
            return "high"
        elif special_requirements.get("voice_output"):
            return "medium"
        elif complexity == "simple":
            return "low"
        elif complexity == "complex":
            return "high"
        else:
            return "medium"
    
    async def _execute_pipeline(self, pipeline: List[str], request: Dict[str, Any], request_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Execute processing pipeline"""
        
        try:
            pipeline_result = {
                "pipeline": pipeline,
                "stages_completed": [],
                "stages_failed": [],
                "final_result": None,
                "processing_details": {}
            }
            
            # Execute each pipeline stage
            for stage in pipeline:
                try:
                    stage_result = await self._execute_stage(stage, request, request_analysis, pipeline_result)
                    
                    pipeline_result["stages_completed"].append(stage)
                    pipeline_result["processing_details"][stage] = stage_result
                    
                    # If stage failed, check if we can continue
                    if not stage_result.get("success", False):
                        pipeline_result["stages_failed"].append(stage)
                        
                        # Critical stages that must succeed
                        if stage in ["input_analysis", "hrm_reasoning", "response_generation"]:
                            return {
                                "success": False,
                                "error": f"Critical stage {stage} failed",
                                "pipeline_result": pipeline_result
                            }
                        # Non-critical stages can fail gracefully
                        else:
                            continue
                    
                except Exception as e:
                    logger.error(f"❌ Pipeline stage {stage} failed: {e}")
                    pipeline_result["stages_failed"].append(stage)
                    pipeline_result["processing_details"][stage] = {"success": False, "error": str(e)}
                    
                    # Continue for non-critical stages
                    if stage not in ["input_analysis", "hrm_reasoning", "response_generation"]:
                        continue
            
            # Set final result from response generation
            if "response_generation" in pipeline_result["processing_details"]:
                pipeline_result["final_result"] = pipeline_result["processing_details"]["response_generation"]
            else:
                pipeline_result["final_result"] = {"success": False, "error": "No response generated"}
            
            pipeline_result["success"] = (
                len(pipeline_result["stages_failed"]) == 0 and 
                pipeline_result["final_result"].get("success", False)
            )
            
            return pipeline_result
            
        except Exception as e:
            logger.error(f"❌ Pipeline execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "pipeline_result": {
                    "pipeline": pipeline,
                    "stages_completed": [],
                    "stages_failed": pipeline,
                    "final_result": None
                }
            }
    
    async def _execute_stage(self, stage: str, request: Dict[str, Any], request_analysis: Dict[str, Any], pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute specific pipeline stage"""
        
        try:
            if stage == "input_analysis":
                return await self._stage_input_analysis(request, request_analysis)
            elif stage == "context_enrichment":
                return await self._stage_context_enrichment(request, request_analysis, pipeline_result)
            elif stage == "hrm_reasoning":
                return await self._stage_hrm_reasoning(request, request_analysis, pipeline_result)
            elif stage == "agent_coordination":
                return await self._stage_agent_coordination(request, request_analysis, pipeline_result)
            elif stage == "tool_execution":
                return await self._stage_tool_execution(request, request_analysis, pipeline_result)
            elif stage == "response_generation":
                return await self._stage_response_generation(request, request_analysis, pipeline_result)
            elif stage == "voice_output":
                return await self._stage_voice_output(request, request_analysis, pipeline_result)
            else:
                return {
                    "success": False,
                    "error": f"Unknown pipeline stage: {stage}"
                }
                
        except Exception as e:
            logger.error(f"❌ Stage {stage} execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _stage_input_analysis(self, request: Dict[str, Any], request_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Execute input analysis stage"""
        
        return {
            "success": True,
            "stage": "input_analysis",
            "request_analysis": request_analysis,
            "processing_time": 0.01
        }
    
    async def _stage_context_enrichment(self, request: Dict[str, Any], request_analysis: Dict[str, Any], pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute context enrichment stage"""
        
        try:
            # Get context from previous HRM reasoning if available
            context = {}
            if "hrm_reasoning" in pipeline_result["processing_details"]:
                hrm_result = pipeline_result["processing_details"]["hrm_reasoning"]
                if hrm_result.get("success"):
                    context["hrm_reasoning"] = hrm_result.get("reasoning_result", {})
            
            # Add system context
            context["system_capabilities"] = {
                "local_processing": self.integration_state["capabilities"]["local_processing"],
                "voice_synthesis": self.integration_state["capabilities"]["voice_synthesis"],
                "agent_management": self.integration_state["capabilities"]["agent_management"],
                "tool_access": self.integration_state["capabilities"]["tool_access"]
            }
            
            return {
                "success": True,
                "stage": "context_enrichment",
                "context": context,
                "processing_time": 0.05
            }
            
        except Exception as e:
            logger.error(f"❌ Context enrichment failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "stage": "context_enrichment"
            }
    
    async def _stage_hrm_reasoning(self, request: Dict[str, Any], request_analysis: Dict[str, Any], pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute HRM reasoning stage"""
        
        try:
            if self.hrm_manager:
                # Get context from previous stage
                context = pipeline_result["processing_details"].get("context_enrichment", {}).get("context", {})
                
                hrm_request = {
                    "message": request.get("message", ""),
                    "context": context,
                    "complexity": request_analysis.get("complexity", "medium")
                }
                
                hrm_result = await self.hrm_manager.reason_hierarchically(hrm_request)
                return {
                    "success": True,
                    "stage": "hrm_reasoning",
                    "hrm_result": hrm_result,
                    "processing_time": hrm_result.get("reasoning_time", 0.5)
                }
            else:
                return {
                    "success": False,
                    "error": "HRM manager not available",
                    "stage": "hrm_reasoning"
                }
                
        except Exception as e:
            logger.error(f"❌ HRM reasoning failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "stage": "hrm_reasoning"
            }
    
    async def _stage_agent_coordination(self, request: Dict[str, Any], request_analysis: Dict[str, Any], pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent coordination stage"""
        
        try:
            if self.master_agent:
                # Determine agent action based on request
                agent_type = self._determine_required_agent_type(request_analysis)
                
                if agent_type:
                    agent_request = {
                        "type": "manage_agents",
                        "action": "list"
                    }
                    
                    agent_result = await self.master_agent.process_master_request(agent_request)
                    return {
                        "success": True,
                        "stage": "agent_coordination",
                        "agent_type": agent_type,
                        "agent_result": agent_result,
                        "processing_time": 0.2
                    }
                else:
                    return {
                        "success": True,
                        "stage": "agent_coordination",
                        "message": "No agent coordination required",
                        "processing_time": 0.05
                    }
            else:
                return {
                    "success": False,
                    "error": "Master agent not available",
                    "stage": "agent_coordination"
                }
                
        except Exception as e:
            logger.error(f"❌ Agent coordination failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "stage": "agent_coordination"
            }
    
    def _determine_required_agent_type(self, request_analysis: Dict[str, Any]) -> Optional[str]:
        """Determine required agent type"""
        
        special_requirements = request_analysis.get("special_requirements", {})
        
        if special_requirements.get("code_generation"):
            return "code_agent"
        elif special_requirements.get("agent_management"):
            return "master_agent"
        elif request_analysis.get("message", "").lower().startswith("research"):
            return "research_agent"
        elif request_analysis.get("message", "").lower().startswith("create"):
            return "creative_agent"
        elif request_analysis.get("message", "").lower().startswith("analyze data"):
            return "data_agent"
        
        return None
    
    async def _stage_tool_execution(self, request: Dict[str, Any], request_analysis: Dict[str, Any], pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool execution stage"""
        
        try:
            if self.mcp_hub:
                # Create a temporary session for tool execution
                session_result = await self.mcp_hub.provide_tools_to_agent(
                    "integration_agent",
                    "master_agent",
                    {"request": request}
                )
                
                if session_result.get("success"):
                    session_id = session_result["session_id"]
                    
                    # Execute web search as example tool
                    tool_result = await self.mcp_hub.execute_tool(
                        session_id,
                        "web_search",
                        {"query": request.get("message", ""), "max_results": 5}
                    )
                    
                    return {
                        "success": True,
                        "stage": "tool_execution",
                        "tool_result": tool_result,
                        "session_id": session_id,
                        "processing_time": tool_result.get("execution_time", 1.0)
                    }
                else:
                    return {
                        "success": False,
                        "error": "Failed to create tool session",
                        "stage": "tool_execution"
                    }
            else:
                return {
                    "success": False,
                    "error": "MCP hub not available",
                    "stage": "tool_execution"
                }
                
        except Exception as e:
            logger.error(f"❌ Tool execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "stage": "tool_execution"
            }
    
    async def _stage_response_generation(self, request: Dict[str, Any], request_analysis: Dict[str, Any], pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute response generation stage"""
        
        try:
            if self.local_llm:
                # Prepare enhanced request with context
                context = {}
                
                # Add HRM reasoning context
                if "hrm_reasoning" in pipeline_result["processing_details"]:
                    hrm_result = pipeline_result["processing_details"]["hrm_reasoning"]
                    if hrm_result.get("success"):
                        context["hrm_enhancement"] = hrm_result.get("reasoning_result", {})
                
                # Add tool results context
                if "tool_execution" in pipeline_result["processing_details"]:
                    tool_result = pipeline_result["processing_details"]["tool_execution"]
                    if tool_result.get("success"):
                        context["tool_results"] = tool_result.get("result", {})
                
                # Generate response
                llm_request = {
                    "message": request.get("message", ""),
                    "use_hrm": True,
                    "context": context
                }
                
                llm_response = await self.local_llm.generate_completion(llm_request)
                
                if llm_response.get("success"):
                    return {
                        "success": True,
                        "stage": "response_generation",
                        "response": llm_response.get("response"),
                        "model_used": llm_response.get("model_used"),
                        "hrm_enhanced": llm_response.get("hrm_enhanced", False),
                        "local_processing": llm_response.get("local_processing", False),
                        "processing_time": llm_response.get("response_time", 0.5)
                    }
                else:
                    return {
                        "success": False,
                        "error": "LLM generation failed",
                        "stage": "response_generation"
                    }
            else:
                return {
                    "success": False,
                    "error": "Local LLM not available",
                    "stage": "response_generation"
                }
                
        except Exception as e:
            logger.error(f"❌ Response generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "stage": "response_generation"
            }
    
    async def _stage_voice_output(self, request: Dict[str, Any], request_analysis: Dict[str, Any], pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute voice output stage"""
        
        try:
            if self.voice_service:
                # Get response from previous stage
                response_text = ""
                if "response_generation" in pipeline_result["processing_details"]:
                    response_data = pipeline_result["processing_details"]["response_generation"]
                    response_text = response_data.get("response", "")
                
                if response_text:
                    # Synthesize and speak
                    voice_result = await self.voice_service.speak_text(response_text)
                    
                    return {
                        "success": True,
                        "stage": "voice_output",
                        "voice_result": voice_result,
                        "text_spoken": response_text,
                        "processing_time": voice_result.get("synthesis_result", {}).get("synthesis_time", 1.0) + voice_result.get("play_result", {}).get("played", 0)
                    }
                else:
                    return {
                        "success": False,
                        "error": "No text to speak",
                        "stage": "voice_output"
                    }
            else:
                return {
                    "success": False,
                    "error": "Voice service not available",
                    "stage": "voice_output"
                }
                
        except Exception as e:
            logger.error(f"❌ Voice output failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "stage": "voice_output"
            }
    
    async def _update_integration_metrics(self, pipeline: List[str], processing_time: float, success: bool):
        """Update integration metrics"""
        
        # Update pipeline efficiency
        pipeline_key = "->".join(pipeline)
        if pipeline_key not in self.integration_metrics["pipeline_efficiency"]:
            self.integration_metrics["pipeline_efficiency"][pipeline_key] = {
                "uses": 0,
                "total_time": 0,
                "successes": 0
            }
        
        pipeline_metrics = self.integration_metrics["pipeline_efficiency"][pipeline_key]
        pipeline_metrics["uses"] += 1
        pipeline_metrics["total_time"] += processing_time
        
        if success:
            pipeline_metrics["successes"] += 1
            self.integration_metrics["orchestration_success"] += 1
        
        # Update average processing time
        total_requests = self.integration_metrics["total_requests"]
        current_avg = self.integration_metrics["average_processing_time"]
        self.integration_metrics["average_processing_time"] = (
            (current_avg * (total_requests - 1) + processing_time) / total_requests
        )
        
        # Update successful integrations
        if success:
            self.integration_metrics["successful_integrations"] += 1
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """Get integration status"""
        
        return {
            "status": self.integration_state["status"],
            "services_initialized": self.integration_state["services_initialized"],
            "capabilities": self.integration_state["capabilities"],
            "orchestration_level": self.integration_state["orchestration_level"],
            "integration_metrics": self.integration_metrics,
            "processing_pipeline": self.processing_pipeline,
            "orchestration_rules": self.orchestration_rules,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Greta-PAI integration health check"""
        
        # Check all critical services
        service_health = {
            "local_llm": getattr(self.local_llm, None) is not None,
            "hrm_manager": getattr(self.hrm_manager, None) is not None,
            "voice_service": getattr(self.voice_service, None) is not None,
            "master_agent": getattr(self.master_agent, None) is not None,
            "mcp_hub": getattr(self.mcp_hub, None) is not None
        }
        
        all_healthy = all(service_health.values())
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "service": "greta_pai_integration",
            "service_health": service_health,
            "integration_state": self.integration_state,
            "integration_metrics": self.integration_metrics,
            "orchestration_success_rate": (
                self.integration_metrics["orchestration_success"] / 
                max(self.integration_metrics["total_requests"], 1)
            ) * 100,
            "timestamp": datetime.utcnow().isoformat()
        }

# Global integration instance
greta_pai_integration = GretaPAIIntegration()

async def get_greta_pai_integration() -> GretaPAIIntegration:
    """Get the global Greta-PAI integration instance"""
    return greta_pai_integration
