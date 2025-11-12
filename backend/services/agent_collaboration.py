"""
Agent Collaboration Service for CPAS5
Implements Sequential, Parallel, and Hierarchical collaboration patterns
"""

import logging
import uuid
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field
from database import Database, get_database
from models.agent_builder_models import AgentExecutionCreate, ExecutionStatus

logger = logging.getLogger(__name__)


class WorkflowType(str, Enum):
    """Workflow collaboration types"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"


class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowDefinition(BaseModel):
    """Workflow definition model"""
    workflow_id: str
    name: str
    description: str
    workflow_type: WorkflowType
    agents: List[str]  # Agent IDs
    configuration: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class WorkflowExecution(BaseModel):
    """Workflow execution record"""
    execution_id: str
    workflow_id: str
    workflow_type: WorkflowType
    status: WorkflowStatus
    agents: List[str]
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    agent_results: List[Dict[str, Any]] = Field(default_factory=list)
    started_at: datetime
    completed_at: Optional[datetime] = None
    execution_time: float = 0.0
    error: Optional[str] = None


class SequentialWorkflow:
    """
    Sequential Workflow: Agent A → Agent B → Agent C
    Each agent processes the output of the previous agent
    """
    
    def __init__(self, agent_builder_service):
        self.agent_builder = agent_builder_service
    
    async def execute(
        self,
        agents: List[str],
        task: Dict[str, Any],
        configuration: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute sequential workflow"""
        try:
            logger.info(f"🔄 Starting sequential workflow with {len(agents)} agents")
            
            results = []
            current_input = task
            
            for i, agent_id in enumerate(agents):
                logger.info(f"📍 Step {i+1}/{len(agents)}: Executing agent {agent_id}")
                
                # Execute agent
                execution = await self.agent_builder.execute_agent(
                    AgentExecutionCreate(
                        agent_id=agent_id,
                        input_data=current_input,
                        configuration=configuration or {}
                    )
                )
                
                # Check if execution succeeded
                if execution.status == ExecutionStatus.FAILED:
                    logger.error(f"❌ Agent {agent_id} failed: {execution.error}")
                    return {
                        "success": False,
                        "error": f"Agent {agent_id} failed: {execution.error}",
                        "failed_at_step": i + 1,
                        "results": results
                    }
                
                # Store result
                result = {
                    "agent_id": agent_id,
                    "step": i + 1,
                    "execution_id": execution.execution_id,
                    "output": execution.output_data,
                    "execution_time": execution.execution_time
                }
                results.append(result)
                
                # Use output as input for next agent
                if execution.output_data:
                    current_input = {
                        "prompt": execution.output_data.get("response", ""),
                        "previous_step": result
                    }
            
            logger.info(f"✅ Sequential workflow completed successfully")
            
            return {
                "success": True,
                "workflow_type": "sequential",
                "steps_completed": len(results),
                "results": results,
                "final_output": results[-1]["output"] if results else None
            }
            
        except Exception as e:
            logger.error(f"❌ Sequential workflow failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": results
            }


class ParallelExecution:
    """
    Parallel Execution: Multiple agents execute simultaneously
    Results are aggregated at the end
    """
    
    def __init__(self, agent_builder_service):
        self.agent_builder = agent_builder_service
    
    async def execute(
        self,
        agents: List[str],
        task: Dict[str, Any],
        configuration: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute parallel workflow"""
        try:
            logger.info(f"⚡ Starting parallel execution with {len(agents)} agents")
            
            # Create execution tasks for all agents
            tasks = []
            for agent_id in agents:
                task_coro = self.agent_builder.execute_agent(
                    AgentExecutionCreate(
                        agent_id=agent_id,
                        input_data=task,
                        configuration=configuration or {}
                    )
                )
                tasks.append(task_coro)
            
            # Execute all agents in parallel
            executions = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            results = []
            successful = 0
            failed = 0
            
            for i, execution in enumerate(executions):
                agent_id = agents[i]
                
                if isinstance(execution, Exception):
                    logger.error(f"❌ Agent {agent_id} raised exception: {execution}")
                    results.append({
                        "agent_id": agent_id,
                        "success": False,
                        "error": str(execution)
                    })
                    failed += 1
                elif execution.status == ExecutionStatus.FAILED:
                    logger.warning(f"⚠️ Agent {agent_id} failed: {execution.error}")
                    results.append({
                        "agent_id": agent_id,
                        "success": False,
                        "execution_id": execution.execution_id,
                        "error": execution.error
                    })
                    failed += 1
                else:
                    logger.info(f"✅ Agent {agent_id} completed successfully")
                    results.append({
                        "agent_id": agent_id,
                        "success": True,
                        "execution_id": execution.execution_id,
                        "output": execution.output_data,
                        "execution_time": execution.execution_time
                    })
                    successful += 1
            
            # Aggregate results
            aggregated_output = self._aggregate_results(results)
            
            logger.info(f"✅ Parallel execution completed: {successful} successful, {failed} failed")
            
            return {
                "success": successful > 0,
                "workflow_type": "parallel",
                "agents_executed": len(agents),
                "successful": successful,
                "failed": failed,
                "results": results,
                "aggregated_output": aggregated_output
            }
            
        except Exception as e:
            logger.error(f"❌ Parallel execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from parallel execution"""
        successful_results = [r for r in results if r.get("success")]
        
        if not successful_results:
            return {"message": "No successful results to aggregate"}
        
        # Combine all responses
        combined_responses = []
        for result in successful_results:
            output = result.get("output", {})
            response = output.get("response", "")
            if response:
                combined_responses.append({
                    "agent_id": result["agent_id"],
                    "response": response
                })
        
        return {
            "combined_responses": combined_responses,
            "total_successful": len(successful_results)
        }


class HierarchicalTeam:
    """
    Hierarchical Team: Master → Team Leads → Specialists
    Multi-level coordination with task decomposition
    """
    
    def __init__(self, agent_builder_service):
        self.agent_builder = agent_builder_service
    
    async def execute(
        self,
        master_agent_id: str,
        team_leads: List[str],
        specialists: Dict[str, List[str]],  # team_lead_id -> [specialist_ids]
        task: Dict[str, Any],
        configuration: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute hierarchical workflow"""
        try:
            logger.info(f"🏢 Starting hierarchical workflow")
            logger.info(f"   Master: {master_agent_id}")
            logger.info(f"   Team Leads: {len(team_leads)}")
            logger.info(f"   Specialists: {sum(len(s) for s in specialists.values())}")
            
            # Step 1: Master agent decomposes task
            logger.info("📋 Step 1: Master agent decomposing task...")
            
            master_execution = await self.agent_builder.execute_agent(
                AgentExecutionCreate(
                    agent_id=master_agent_id,
                    input_data={
                        "prompt": f"Decompose this task into subtasks for {len(team_leads)} team leads:\n{task.get('prompt', '')}",
                        "mode": "task_decomposition"
                    },
                    configuration=configuration or {}
                )
            )
            
            if master_execution.status == ExecutionStatus.FAILED:
                return {
                    "success": False,
                    "error": f"Master agent failed: {master_execution.error}"
                }
            
            # Step 2: Team leads coordinate specialists
            logger.info("👥 Step 2: Team leads coordinating specialists...")
            
            team_results = []
            
            for i, team_lead_id in enumerate(team_leads):
                # Get specialists for this team lead
                team_specialists = specialists.get(team_lead_id, [])
                
                if not team_specialists:
                    logger.warning(f"⚠️ No specialists for team lead {team_lead_id}")
                    continue
                
                # Team lead executes with specialists in parallel
                specialist_tasks = []
                for specialist_id in team_specialists:
                    specialist_task = self.agent_builder.execute_agent(
                        AgentExecutionCreate(
                            agent_id=specialist_id,
                            input_data=task,
                            configuration=configuration or {}
                        )
                    )
                    specialist_tasks.append(specialist_task)
                
                # Execute specialists in parallel
                specialist_executions = await asyncio.gather(*specialist_tasks, return_exceptions=True)
                
                # Collect specialist results
                specialist_results = []
                for j, execution in enumerate(specialist_executions):
                    specialist_id = team_specialists[j]
                    
                    if isinstance(execution, Exception) or execution.status == ExecutionStatus.FAILED:
                        specialist_results.append({
                            "specialist_id": specialist_id,
                            "success": False,
                            "error": str(execution) if isinstance(execution, Exception) else execution.error
                        })
                    else:
                        specialist_results.append({
                            "specialist_id": specialist_id,
                            "success": True,
                            "output": execution.output_data
                        })
                
                team_results.append({
                    "team_lead_id": team_lead_id,
                    "specialists": specialist_results
                })
            
            # Step 3: Master agent aggregates results
            logger.info("🎯 Step 3: Master agent aggregating results...")
            
            aggregation_prompt = f"Aggregate these team results:\n{team_results}"
            
            final_execution = await self.agent_builder.execute_agent(
                AgentExecutionCreate(
                    agent_id=master_agent_id,
                    input_data={
                        "prompt": aggregation_prompt,
                        "mode": "result_aggregation"
                    },
                    configuration=configuration or {}
                )
            )
            
            logger.info(f"✅ Hierarchical workflow completed")
            
            return {
                "success": True,
                "workflow_type": "hierarchical",
                "master_decomposition": master_execution.output_data,
                "team_results": team_results,
                "final_aggregation": final_execution.output_data
            }
            
        except Exception as e:
            logger.error(f"❌ Hierarchical workflow failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


class AgentCollaborationService:
    """
    Main service for managing agent collaboration workflows
    """

    def __init__(self, agent_builder_service):
        self.agent_builder = agent_builder_service
        self.db: Optional[Database] = None

        # Collaboration pattern implementations
        self.sequential = SequentialWorkflow(agent_builder_service)
        self.parallel = ParallelExecution(agent_builder_service)
        self.hierarchical = HierarchicalTeam(agent_builder_service)

        # Collections
        self.workflows_collection = "workflows"
        self.workflow_executions_collection = "workflow_executions"

        # Metrics
        self.metrics = {
            "total_workflows": 0,
            "successful_workflows": 0,
            "failed_workflows": 0,
            "workflows_by_type": {},
            "average_execution_time": 0.0
        }

    async def initialize(self):
        """Initialize collaboration service"""
        try:
            self.db = await get_database()
            logger.info("✅ Agent Collaboration Service initialized")
        except Exception as e:
            logger.error(f"❌ Collaboration service initialization failed: {e}")
            raise

    async def create_workflow(
        self,
        name: str,
        description: str,
        workflow_type: WorkflowType,
        agents: List[str],
        configuration: Optional[Dict[str, Any]] = None
    ) -> WorkflowDefinition:
        """Create a new workflow definition"""
        try:
            workflow_id = f"wf_{uuid.uuid4().hex[:12]}"

            workflow = WorkflowDefinition(
                workflow_id=workflow_id,
                name=name,
                description=description,
                workflow_type=workflow_type,
                agents=agents,
                configuration=configuration or {},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )

            # Save to database
            workflow_dict = workflow.model_dump()
            workflow_dict["created_at"] = workflow_dict["created_at"].isoformat()
            workflow_dict["updated_at"] = workflow_dict["updated_at"].isoformat()

            await self.db.insert_one(self.workflows_collection, workflow_dict)

            logger.info(f"✅ Workflow created: {name} ({workflow_id})")

            return workflow

        except Exception as e:
            logger.error(f"❌ Workflow creation failed: {e}")
            raise

    async def execute_workflow(
        self,
        workflow_id: str,
        input_data: Dict[str, Any]
    ) -> WorkflowExecution:
        """Execute a workflow"""
        execution_id = f"wfexec_{uuid.uuid4().hex[:12]}"
        start_time = datetime.utcnow()

        try:
            # Get workflow definition
            workflow_dict = await self.db.find_one(
                self.workflows_collection,
                {"workflow_id": workflow_id}
            )

            if not workflow_dict:
                raise ValueError(f"Workflow not found: {workflow_id}")

            # Convert datetime strings
            for k in ["created_at", "updated_at"]:
                if k in workflow_dict and isinstance(workflow_dict[k], str):
                    workflow_dict[k] = datetime.fromisoformat(workflow_dict[k])

            workflow = WorkflowDefinition(**workflow_dict)

            # Create execution record
            execution = WorkflowExecution(
                execution_id=execution_id,
                workflow_id=workflow_id,
                workflow_type=workflow.workflow_type,
                status=WorkflowStatus.RUNNING,
                agents=workflow.agents,
                input_data=input_data,
                started_at=start_time
            )

            # Save initial execution record
            exec_dict = execution.model_dump()
            exec_dict["started_at"] = exec_dict["started_at"].isoformat()
            await self.db.insert_one(self.workflow_executions_collection, exec_dict)

            # Execute based on workflow type
            logger.info(f"🚀 Executing {workflow.workflow_type} workflow: {workflow.name}")

            if workflow.workflow_type == WorkflowType.SEQUENTIAL:
                result = await self.sequential.execute(
                    workflow.agents,
                    input_data,
                    workflow.configuration
                )
            elif workflow.workflow_type == WorkflowType.PARALLEL:
                result = await self.parallel.execute(
                    workflow.agents,
                    input_data,
                    workflow.configuration
                )
            elif workflow.workflow_type == WorkflowType.HIERARCHICAL:
                # For hierarchical, configuration should specify master, leads, specialists
                master = workflow.configuration.get("master_agent")
                leads = workflow.configuration.get("team_leads", [])
                specialists = workflow.configuration.get("specialists", {})

                result = await self.hierarchical.execute(
                    master,
                    leads,
                    specialists,
                    input_data,
                    workflow.configuration
                )
            else:
                raise ValueError(f"Unknown workflow type: {workflow.workflow_type}")

            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()

            # Update execution record
            status = WorkflowStatus.COMPLETED if result.get("success") else WorkflowStatus.FAILED

            await self.db.update_one(
                self.workflow_executions_collection,
                {"execution_id": execution_id},
                {
                    "$set": {
                        "status": status,
                        "output_data": result,
                        "agent_results": result.get("results", []),
                        "completed_at": datetime.utcnow().isoformat(),
                        "execution_time": execution_time,
                        "error": result.get("error")
                    }
                }
            )

            # Update metrics
            self._update_metrics(workflow.workflow_type, execution_time, result.get("success", False))

            logger.info(f"✅ Workflow execution completed in {execution_time:.2f}s")

            # Return updated execution
            return await self.get_workflow_execution(execution_id)

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            error_message = str(e)

            logger.error(f"❌ Workflow execution failed: {error_message}")

            # Update execution record with failure
            await self.db.update_one(
                self.workflow_executions_collection,
                {"execution_id": execution_id},
                {
                    "$set": {
                        "status": WorkflowStatus.FAILED,
                        "error": error_message,
                        "completed_at": datetime.utcnow().isoformat(),
                        "execution_time": execution_time
                    }
                }
            )

            return await self.get_workflow_execution(execution_id)

    async def get_workflow_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution by ID"""
        try:
            exec_dict = await self.db.find_one(
                self.workflow_executions_collection,
                {"execution_id": execution_id}
            )

            if not exec_dict:
                return None

            # Convert datetime strings
            for k in ["started_at", "completed_at"]:
                if k in exec_dict and exec_dict[k] and isinstance(exec_dict[k], str):
                    exec_dict[k] = datetime.fromisoformat(exec_dict[k])

            return WorkflowExecution(**exec_dict)

        except Exception as e:
            logger.error(f"Failed to get workflow execution: {e}")
            return None

    def _update_metrics(self, workflow_type: WorkflowType, execution_time: float, success: bool):
        """Update workflow metrics"""
        self.metrics["total_workflows"] += 1

        if success:
            self.metrics["successful_workflows"] += 1
        else:
            self.metrics["failed_workflows"] += 1

        # By type
        type_key = workflow_type.value
        if type_key not in self.metrics["workflows_by_type"]:
            self.metrics["workflows_by_type"][type_key] = 0
        self.metrics["workflows_by_type"][type_key] += 1

        # Average execution time
        total = self.metrics["total_workflows"]
        current_avg = self.metrics["average_execution_time"]
        self.metrics["average_execution_time"] = (current_avg * (total - 1) + execution_time) / total


# Global collaboration service instance
collaboration_service = None


async def get_collaboration_service(agent_builder_service):
    """Get or create collaboration service instance"""
    global collaboration_service

    if collaboration_service is None:
        collaboration_service = AgentCollaborationService(agent_builder_service)
        await collaboration_service.initialize()

    return collaboration_service

