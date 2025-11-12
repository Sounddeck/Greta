"""
Team Service for CPAS5 Agent Builder
Service layer for managing teams and team executions
"""

import uuid
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from models.team_models import *
from database import Database

logger = logging.getLogger(__name__)


class TeamService:
    """Service for managing teams and team executions"""
    
    def __init__(self, db: Database):
        self.db = db
        self.teams_collection = "teams"
        self.executions_collection = "team_executions"
        self.messages_collection = "team_messages"
    
    async def create_team(self, team_create: TeamCreate, created_by: Optional[str] = None) -> Team:
        """Create a new team"""
        team_id = f"team-{uuid.uuid4().hex[:12]}"
        
        # Validate agents exist
        for team_agent in team_create.agents:
            agent = await self.db.find_one(
                "custom_agents",
                {"agent_id": team_agent.agent_id}
            )
            if not agent:
                raise ValueError(f"Agent {team_agent.agent_id} not found")
        
        # Validate sequential pattern has order
        if team_create.workflow_pattern == WorkflowPattern.SEQUENTIAL:
            for team_agent in team_create.agents:
                if team_agent.order is None:
                    raise ValueError(f"Agent {team_agent.role} missing order for sequential pattern")
        
        # Create team document
        now = datetime.utcnow()
        team_doc = {
            "team_id": team_id,
            "name": team_create.name,
            "description": team_create.description,
            "agents": [agent.model_dump() for agent in team_create.agents],
            "workflow_pattern": team_create.workflow_pattern.value,
            "termination": team_create.termination.model_dump(),
            "metadata": team_create.metadata or {},
            "status": TeamStatus.ACTIVE.value,
            "created_at": now,
            "updated_at": now,
            "created_by": created_by
        }
        
        # Insert into database
        await self.db.insert_one(self.teams_collection, team_doc)
        
        logger.info(f"Created team: {team_id} - {team_create.name}")
        
        return await self.get_team(team_id)
    
    async def get_team(self, team_id: str) -> Optional[Team]:
        """Get team by ID"""
        team_doc = await self.db.find_one(
            self.teams_collection,
            {"team_id": team_id}
        )
        
        if not team_doc:
            return None
        
        # Convert to Team model
        return self._doc_to_team(team_doc)
    
    async def list_teams(
        self,
        status: Optional[str] = None,
        workflow_pattern: Optional[WorkflowPattern] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Team]:
        """List teams with filters"""
        query = {}
        
        if status:
            query["status"] = status
        
        if workflow_pattern:
            query["workflow_pattern"] = workflow_pattern.value
        
        # Get teams from database
        team_docs = await self.db.find_many(
            self.teams_collection,
            query,
            limit=limit
        )
        
        # Convert to Team models
        teams = [self._doc_to_team(doc) for doc in team_docs]
        
        # Apply offset (MongoDB find_many doesn't support offset directly in our implementation)
        if offset > 0:
            teams = teams[offset:]
        
        return teams
    
    async def update_team(self, team_id: str, update: TeamUpdate) -> Optional[Team]:
        """Update team"""
        # Check if team exists
        existing_team = await self.get_team(team_id)
        if not existing_team:
            return None
        
        # Build update document
        update_doc = {"$set": {"updated_at": datetime.utcnow()}}
        
        if update.name is not None:
            update_doc["$set"]["name"] = update.name
        
        if update.description is not None:
            update_doc["$set"]["description"] = update.description
        
        if update.workflow_pattern is not None:
            update_doc["$set"]["workflow_pattern"] = update.workflow_pattern.value
        
        if update.termination is not None:
            update_doc["$set"]["termination"] = update.termination.model_dump()
        
        if update.metadata is not None:
            update_doc["$set"]["metadata"] = update.metadata
        
        if update.status is not None:
            update_doc["$set"]["status"] = update.status.value
        
        if update.agents is not None:
            # Validate agents exist
            for team_agent in update.agents:
                agent = await self.db.find_one(
                    "custom_agents",
                    {"agent_id": team_agent.agent_id}
                )
                if not agent:
                    raise ValueError(f"Agent {team_agent.agent_id} not found")
            
            update_doc["$set"]["agents"] = [agent.model_dump() for agent in update.agents]
        
        # Update in database
        await self.db.update_one(
            self.teams_collection,
            {"team_id": team_id},
            update_doc
        )
        
        logger.info(f"Updated team: {team_id}")
        
        return await self.get_team(team_id)
    
    async def delete_team(self, team_id: str) -> bool:
        """Delete team"""
        result = await self.db.delete_one(
            self.teams_collection,
            {"team_id": team_id}
        )
        
        if result:
            logger.info(f"Deleted team: {team_id}")
        
        return result
    
    async def validate_team(self, team_id: str) -> TeamValidationResult:
        """Validate team configuration"""
        team = await self.get_team(team_id)
        if not team:
            return TeamValidationResult(
                valid=False,
                errors=["Team not found"]
            )
        
        errors = []
        warnings = []
        
        # Check if team has agents
        if not team.agents:
            errors.append("Team must have at least one agent")
        
        # Validate agents exist and are active
        for team_agent in team.agents:
            agent = await self.db.find_one(
                "custom_agents",
                {"agent_id": team_agent.agent_id}
            )
            if not agent:
                errors.append(f"Agent {team_agent.agent_id} not found")
            elif agent.get("status") != "active":
                warnings.append(f"Agent {team_agent.agent_id} ({team_agent.role}) is not active")
        
        # Validate sequential pattern has order
        if team.workflow_pattern == WorkflowPattern.SEQUENTIAL:
            for team_agent in team.agents:
                if team_agent.order is None:
                    errors.append(f"Agent {team_agent.role} missing order for sequential pattern")
        
        # Validate termination condition
        if team.termination.type == TerminationType.MAX_TURNS:
            try:
                max_turns = int(team.termination.value)
                if max_turns <= 0:
                    errors.append("max_turns must be positive")
            except (ValueError, TypeError):
                errors.append("max_turns must be a number")
        
        return TeamValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    async def execute_team(self, request: TeamExecutionRequest) -> TeamExecutionResult:
        """
        Execute a team workflow with full sequential chat implementation

        This method implements the Sequential Chat pattern where agents execute
        in a predefined order, passing messages and context between each other.
        """
        from services.agent_builder_service import AgentBuilderService
        from services.llm_service import LLMService
        from models.agent_builder_models import AgentExecutionCreate

        execution_id = f"exec-{uuid.uuid4().hex[:12]}"
        start_time = datetime.utcnow()

        try:
            # 1. Get team configuration
            team = await self.get_team(request.team_id)
            if not team:
                raise ValueError(f"Team not found: {request.team_id}")

            # 2. Validate team before execution
            validation = await self.validate_team(request.team_id)
            if not validation.is_valid:
                raise ValueError(f"Team validation failed: {', '.join(validation.errors)}")

            # 3. Create execution document
            execution_doc = {
                "execution_id": execution_id,
                "team_id": request.team_id,
                "task": request.task,
                "context": request.context or {},
                "status": ExecutionStatus.RUNNING.value,
                "messages": [],
                "final_output": None,
                "metadata": {
                    "total_turns": 0,
                    "total_tokens": 0,
                    "duration_ms": 0,
                    "termination_reason": "not_started",
                    "workflow_pattern": team.workflow_pattern.value,
                    "agents_executed": []
                },
                "created_at": start_time,
                "completed_at": None
            }

            await self.db.insert_one(self.executions_collection, execution_doc)

            logger.info(f"Starting team execution: {execution_id} for team {request.team_id}")

            # 4. Execute workflow based on pattern
            if team.workflow_pattern == WorkflowPattern.SEQUENTIAL:
                result = await self._execute_sequential(team, request, execution_id, execution_doc)
            elif team.workflow_pattern == WorkflowPattern.AUTONOMOUS:
                result = await self._execute_autonomous(team, request, execution_id, execution_doc)
            elif team.workflow_pattern == WorkflowPattern.ROUND_ROBIN:
                result = await self._execute_round_robin(team, request, execution_id, execution_doc)
            elif team.workflow_pattern == WorkflowPattern.SELECTOR:
                result = await self._execute_selector(team, request, execution_id, execution_doc)
            else:
                raise ValueError(f"Unsupported workflow pattern: {team.workflow_pattern}")

            return result

        except Exception as e:
            logger.error(f"Team execution failed: {e}")

            # Update execution status to FAILED
            await self.db.update_one(
                self.executions_collection,
                {"execution_id": execution_id},
                {
                    "$set": {
                        "status": ExecutionStatus.FAILED.value,
                        "metadata.termination_reason": f"error: {str(e)}",
                        "completed_at": datetime.utcnow()
                    }
                }
            )

            # Return failed result
            return TeamExecutionResult(
                execution_id=execution_id,
                team_id=request.team_id,
                status=ExecutionStatus.FAILED,
                messages=[],
                final_output=None,
                metadata={
                    "total_turns": 0,
                    "total_tokens": 0,
                    "duration_ms": 0,
                    "termination_reason": f"error: {str(e)}",
                    "workflow_pattern": team.workflow_pattern.value if team else "unknown",
                    "agents_executed": []
                },
                created_at=start_time,
                completed_at=datetime.utcnow()
            )

    async def _execute_sequential(
        self,
        team: Team,
        request: TeamExecutionRequest,
        execution_id: str,
        execution_doc: Dict[str, Any]
    ) -> TeamExecutionResult:
        """
        Execute Sequential Chat pattern

        Agents execute in predefined order based on TeamAgent.order field.
        Each agent receives the conversation history and adds their response.
        """
        from services.agent_builder_service import AgentBuilderService
        from models.agent_builder_models import AgentExecutionCreate

        messages = []
        conversation_history = []
        total_turns = 0
        total_tokens = 0

        # Sort agents by order
        sorted_agents = sorted(team.agents, key=lambda a: a.order or 0)

        # Initialize agent builder service
        agent_service = AgentBuilderService(self.db)

        # Build initial context
        current_context = {
            "task": request.task,
            "team_context": request.context or {},
            "conversation_history": []
        }

        logger.info(f"Sequential execution: {len(sorted_agents)} agents in order")

        # Execute agents in sequence
        for idx, team_agent in enumerate(sorted_agents):
            turn_number = idx + 1

            # Check termination condition before executing next agent
            should_terminate, reason = await self._check_termination(
                team.termination,
                messages,
                total_turns,
                conversation_history
            )

            if should_terminate:
                logger.info(f"Termination condition met: {reason}")
                break

            try:
                # Get agent configuration
                agent = await agent_service.get_agent(team_agent.agent_id)
                if not agent:
                    logger.warning(f"Agent not found: {team_agent.agent_id}, skipping")
                    continue

                logger.info(f"Turn {turn_number}: Executing agent {agent.name} ({team_agent.role})")

                # Build prompt for this agent
                agent_prompt = await self._build_agent_prompt(
                    team_agent,
                    agent,
                    current_context,
                    conversation_history
                )

                # Execute agent
                execution_request = AgentExecutionCreate(
                    agent_id=team_agent.agent_id,
                    input_data={"prompt": agent_prompt, "context": current_context},
                    configuration={}
                )

                agent_execution = await agent_service.execute_agent(execution_request)

                # Extract response
                agent_response = agent_execution.output_data.get("response", "") if agent_execution.output_data else ""

                # Create message
                message = TeamMessage(
                    role=team_agent.role,
                    agent_id=team_agent.agent_id,
                    agent_name=agent.name,
                    content=agent_response,
                    turn_number=turn_number,
                    timestamp=datetime.utcnow(),
                    metadata={
                        "execution_id": agent_execution.execution_id,
                        "tokens_used": agent_execution.metadata.get("tokens_used", 0) if agent_execution.metadata else 0
                    }
                )

                messages.append(message)
                conversation_history.append({
                    "role": team_agent.role,
                    "content": agent_response
                })

                # Update context for next agent
                current_context["conversation_history"] = conversation_history
                current_context["last_speaker"] = team_agent.role
                current_context["last_message"] = agent_response

                # Update metrics
                total_turns += 1
                total_tokens += message.metadata.get("tokens_used", 0)

                # Update execution document
                await self.db.update_one(
                    self.executions_collection,
                    {"execution_id": execution_id},
                    {
                        "$push": {"messages": message.model_dump()},
                        "$set": {
                            "metadata.total_turns": total_turns,
                            "metadata.total_tokens": total_tokens
                        },
                        "$addToSet": {"metadata.agents_executed": team_agent.agent_id}
                    }
                )

                logger.info(f"Turn {turn_number} complete: {len(agent_response)} chars")

            except Exception as e:
                logger.error(f"Error executing agent {team_agent.agent_id}: {e}")
                # Continue with next agent
                continue

        # Determine final status and termination reason
        should_terminate, termination_reason = await self._check_termination(
            team.termination,
            messages,
            total_turns,
            conversation_history
        )

        if not should_terminate:
            termination_reason = "all_agents_executed"

        # Calculate duration
        end_time = datetime.utcnow()
        duration_ms = int((end_time - execution_doc["created_at"]).total_seconds() * 1000)

        # Generate final output
        final_output = await self._generate_final_output(messages, conversation_history)

        # Update execution document
        await self.db.update_one(
            self.executions_collection,
            {"execution_id": execution_id},
            {
                "$set": {
                    "status": ExecutionStatus.COMPLETED.value,
                    "final_output": final_output,
                    "metadata.duration_ms": duration_ms,
                    "metadata.termination_reason": termination_reason,
                    "completed_at": end_time
                }
            }
        )

        logger.info(f"Sequential execution complete: {total_turns} turns, {total_tokens} tokens")

        return TeamExecutionResult(
            execution_id=execution_id,
            team_id=team.team_id,
            status=ExecutionStatus.COMPLETED,
            messages=messages,
            final_output=final_output,
            metadata={
                "total_turns": total_turns,
                "total_tokens": total_tokens,
                "duration_ms": duration_ms,
                "termination_reason": termination_reason,
                "workflow_pattern": team.workflow_pattern.value,
                "agents_executed": [a.agent_id for a in sorted_agents[:total_turns]]
            },
            created_at=execution_doc["created_at"],
            completed_at=end_time
        )

    async def get_execution(self, execution_id: str) -> Optional[TeamExecutionResult]:
        """Get execution by ID"""
        exec_doc = await self.db.find_one(
            self.executions_collection,
            {"execution_id": execution_id}
        )
        
        if not exec_doc:
            return None
        
        return self._doc_to_execution(exec_doc)
    
    async def list_executions(self, team_id: str, limit: int = 50) -> List[TeamExecutionResult]:
        """List execution history for a team"""
        exec_docs = await self.db.find_many(
            self.executions_collection,
            {"team_id": team_id},
            limit=limit
        )
        
        return [self._doc_to_execution(doc) for doc in exec_docs]
    
    # Helper methods for execution

    async def _check_termination(
        self,
        termination: TerminationCondition,
        messages: List[TeamMessage],
        total_turns: int,
        conversation_history: List[Dict[str, str]]
    ) -> tuple[bool, str]:
        """Check if termination condition is met"""

        if termination.type == TerminationType.MAX_TURNS:
            max_turns = termination.value
            if total_turns >= max_turns:
                return True, f"max_turns_reached ({max_turns})"

        elif termination.type == TerminationType.KEYWORD:
            keyword = termination.value.lower()
            for msg in messages:
                if keyword in msg.content.lower():
                    return True, f"keyword_detected ({keyword})"

        elif termination.type == TerminationType.MAX_CONSECUTIVE:
            max_consecutive = termination.value
            if len(messages) >= max_consecutive:
                # Check if last N messages are from same agent
                last_n_agents = [msg.agent_id for msg in messages[-max_consecutive:]]
                if len(set(last_n_agents)) == 1:
                    return True, f"max_consecutive_reached ({max_consecutive})"

        elif termination.type == TerminationType.FUNCTION:
            # Custom termination function (placeholder for now)
            # Could be implemented with eval() or custom function registry
            logger.warning("Custom termination functions not yet implemented")

        return False, "not_terminated"

    async def _build_agent_prompt(
        self,
        team_agent: TeamAgent,
        agent: Any,  # CustomAgent
        context: Dict[str, Any],
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """Build prompt for agent based on role and conversation history"""

        # Start with agent's system message
        prompt_parts = []

        # Add role context
        prompt_parts.append(f"You are {team_agent.role}.")
        if team_agent.description:
            prompt_parts.append(team_agent.description)

        # Add task
        prompt_parts.append(f"\nTask: {context['task']}")

        # Add conversation history
        if conversation_history:
            prompt_parts.append("\nConversation so far:")
            for msg in conversation_history:
                prompt_parts.append(f"{msg['role']}: {msg['content']}")

        # Add instructions
        prompt_parts.append(f"\nAs {team_agent.role}, provide your response:")

        return "\n".join(prompt_parts)

    async def _generate_final_output(
        self,
        messages: List[TeamMessage],
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """Generate final output summary from conversation"""

        if not messages:
            return "No messages generated"

        # Simple summary: return last message or concatenate all
        if len(messages) == 1:
            return messages[0].content

        # For multiple messages, create a summary
        summary_parts = [
            "Team Conversation Summary:",
            f"Total turns: {len(messages)}",
            "",
            "Conversation:"
        ]

        for msg in messages:
            summary_parts.append(f"\n{msg.role} ({msg.agent_name}):")
            summary_parts.append(msg.content[:500] + "..." if len(msg.content) > 500 else msg.content)

        return "\n".join(summary_parts)

    async def _execute_autonomous(
        self,
        team: Team,
        request: TeamExecutionRequest,
        execution_id: str,
        execution_doc: Dict[str, Any]
    ) -> TeamExecutionResult:
        """Execute Autonomous Chat pattern (placeholder)"""
        logger.warning("Autonomous pattern not yet implemented, using sequential")
        return await self._execute_sequential(team, request, execution_id, execution_doc)

    async def _execute_round_robin(
        self,
        team: Team,
        request: TeamExecutionRequest,
        execution_id: str,
        execution_doc: Dict[str, Any]
    ) -> TeamExecutionResult:
        """Execute Round Robin pattern (placeholder)"""
        logger.warning("Round Robin pattern not yet implemented, using sequential")
        return await self._execute_sequential(team, request, execution_id, execution_doc)

    async def _execute_selector(
        self,
        team: Team,
        request: TeamExecutionRequest,
        execution_id: str,
        execution_doc: Dict[str, Any]
    ) -> TeamExecutionResult:
        """Execute Selector pattern (placeholder)"""
        logger.warning("Selector pattern not yet implemented, using sequential")
        return await self._execute_sequential(team, request, execution_id, execution_doc)

    # Helper methods for data conversion

    def _doc_to_team(self, doc: Dict[str, Any]) -> Team:
        """Convert MongoDB document to Team model"""
        return Team(
            team_id=doc["team_id"],
            name=doc["name"],
            description=doc["description"],
            agents=[TeamAgent(**agent) for agent in doc["agents"]],
            workflow_pattern=WorkflowPattern(doc["workflow_pattern"]),
            termination=TerminationCondition(**doc["termination"]),
            metadata=doc.get("metadata", {}),
            status=TeamStatus(doc.get("status", "active")),
            created_at=doc["created_at"],
            updated_at=doc["updated_at"],
            created_by=doc.get("created_by")
        )
    
    def _doc_to_execution(self, doc: Dict[str, Any]) -> TeamExecutionResult:
        """Convert MongoDB document to TeamExecutionResult model"""
        messages = [TeamMessage(**msg) for msg in doc.get("messages", [])]
        
        return TeamExecutionResult(
            execution_id=doc["execution_id"],
            team_id=doc["team_id"],
            status=ExecutionStatus(doc["status"]),
            messages=messages,
            final_output=doc.get("final_output"),
            metadata=doc.get("metadata", {}),
            created_at=doc["created_at"],
            completed_at=doc.get("completed_at")
        )


# Dependency injection
async def get_team_service(db: Database = None) -> TeamService:
    """Get team service instance"""
    if db is None:
        from database import get_database
        db = await get_database()
    return TeamService(db)

