"""
Multi-Agent Collaboration Service

Enables multiple agents to work together on complex tasks through:
- Agent team formation based on task requirements
- Inter-agent communication protocol
- Parallel execution coordination
- Consensus mechanism for decision-making
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import uuid4
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)


class AgentRole(BaseModel):
    """Agent role in a team"""
    role_id: str = Field(default_factory=lambda: str(uuid4()))
    role_name: str
    capabilities: List[str]
    priority: int = 1  # Higher priority agents make final decisions


class AgentTeamMember(BaseModel):
    """Team member definition"""
    agent_id: str
    agent_name: str
    role: AgentRole
    status: str = "available"  # available, busy, offline
    current_task: Optional[str] = None


class AgentMessage(BaseModel):
    """Inter-agent communication message"""
    message_id: str = Field(default_factory=lambda: str(uuid4()))
    from_agent: str
    to_agent: Optional[str] = None  # None = broadcast
    message_type: str  # request, response, broadcast, vote
    content: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    requires_response: bool = False


class CollaborationTask(BaseModel):
    """Task for agent collaboration"""
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    description: str
    required_capabilities: List[str]
    assigned_agents: List[str] = Field(default_factory=list)
    status: str = "pending"  # pending, in_progress, completed, failed
    results: Dict[str, Any] = Field(default_factory=dict)
    consensus_required: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class ConsensusVote(BaseModel):
    """Vote for consensus decision"""
    vote_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str
    agent_id: str
    decision: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MultiAgentCollaborationService:
    """Service for multi-agent collaboration"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.teams: Dict[str, List[AgentTeamMember]] = {}
        self.message_queue: List[AgentMessage] = []
        self.active_tasks: Dict[str, CollaborationTask] = {}
        self.consensus_votes: Dict[str, List[ConsensusVote]] = {}
        
        logger.info("✅ Multi-Agent Collaboration Service initialized")
    
    async def form_team(
        self,
        task_description: str,
        required_capabilities: List[str],
        max_team_size: int = 5
    ) -> Dict[str, Any]:
        """
        Form an agent team based on task requirements
        
        Args:
            task_description: Description of the task
            required_capabilities: List of required capabilities
            max_team_size: Maximum number of agents in team
            
        Returns:
            Team formation result with selected agents
        """
        try:
            # Get available agents from database
            agents_collection = self.db.get_collection("agents")
            available_agents = await agents_collection.find(
                {"status": "available"}
            ).to_list(length=100)
            
            # Score agents based on capability match
            scored_agents = []
            for agent in available_agents:
                agent_capabilities = agent.get("capabilities", [])
                match_score = len(set(required_capabilities) & set(agent_capabilities))
                if match_score > 0:
                    scored_agents.append({
                        "agent": agent,
                        "score": match_score
                    })
            
            # Sort by score and select top agents
            scored_agents.sort(key=lambda x: x["score"], reverse=True)
            selected_agents = scored_agents[:max_team_size]
            
            # Create team members
            team_id = str(uuid4())
            team_members = []
            
            for item in selected_agents:
                agent = item["agent"]
                role = AgentRole(
                    role_name=agent.get("role", "contributor"),
                    capabilities=agent.get("capabilities", []),
                    priority=item["score"]
                )
                
                member = AgentTeamMember(
                    agent_id=agent["_id"],
                    agent_name=agent.get("name", "Unknown"),
                    role=role
                )
                team_members.append(member)
            
            self.teams[team_id] = team_members
            
            logger.info(f"✅ Formed team {team_id} with {len(team_members)} agents")
            
            return {
                "success": True,
                "team_id": team_id,
                "team_size": len(team_members),
                "members": [m.dict() for m in team_members],
                "coverage": len(set(required_capabilities) & 
                              set(cap for m in team_members for cap in m.role.capabilities))
            }
            
        except Exception as e:
            logger.error(f"❌ Team formation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def send_message(
        self,
        from_agent: str,
        message_type: str,
        content: Dict[str, Any],
        to_agent: Optional[str] = None,
        requires_response: bool = False
    ) -> AgentMessage:
        """
        Send message between agents
        
        Args:
            from_agent: Sender agent ID
            message_type: Type of message
            content: Message content
            to_agent: Recipient agent ID (None for broadcast)
            requires_response: Whether response is required
            
        Returns:
            Created message
        """
        message = AgentMessage(
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=message_type,
            content=content,
            requires_response=requires_response
        )
        
        self.message_queue.append(message)
        
        # Store in database
        messages_collection = self.db.get_collection("agent_messages")
        await messages_collection.insert_one(message.dict())
        
        logger.info(f"📨 Message sent from {from_agent} to {to_agent or 'all'}")
        
        return message
    
    async def execute_parallel_tasks(
        self,
        team_id: str,
        tasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute multiple tasks in parallel across team
        
        Args:
            team_id: Team ID
            tasks: List of tasks to execute
            
        Returns:
            Execution results
        """
        if team_id not in self.teams:
            return {"success": False, "error": "Team not found"}
        
        team = self.teams[team_id]
        
        # Assign tasks to agents
        task_assignments = []
        for i, task in enumerate(tasks):
            agent = team[i % len(team)]  # Round-robin assignment
            task_assignments.append({
                "task": task,
                "agent": agent.agent_id
            })
        
        # Execute tasks in parallel
        async def execute_task(assignment):
            task = assignment["task"]
            agent_id = assignment["agent"]
            
            # Simulate task execution
            await asyncio.sleep(0.1)  # Placeholder for actual execution
            
            return {
                "task_id": task.get("id", str(uuid4())),
                "agent_id": agent_id,
                "status": "completed",
                "result": {"success": True}
            }
        
        results = await asyncio.gather(*[execute_task(a) for a in task_assignments])
        
        logger.info(f"✅ Executed {len(results)} parallel tasks for team {team_id}")
        
        return {
            "success": True,
            "team_id": team_id,
            "tasks_executed": len(results),
            "results": results
        }
    
    async def reach_consensus(
        self,
        task_id: str,
        team_id: str,
        decision_options: List[str],
        timeout_seconds: int = 30
    ) -> Dict[str, Any]:
        """
        Reach consensus among team members
        
        Args:
            task_id: Task ID
            team_id: Team ID
            decision_options: Available decision options
            timeout_seconds: Timeout for consensus
            
        Returns:
            Consensus result
        """
        if team_id not in self.teams:
            return {"success": False, "error": "Team not found"}
        
        team = self.teams[team_id]
        
        # Collect votes from each agent
        votes = []
        for member in team:
            # Simulate voting (in real implementation, would call agent's decision function)
            vote = ConsensusVote(
                task_id=task_id,
                agent_id=member.agent_id,
                decision=decision_options[0],  # Placeholder
                confidence=0.8,
                reasoning="Automated vote for demonstration"
            )
            votes.append(vote)
        
        self.consensus_votes[task_id] = votes
        
        # Calculate consensus
        vote_counts = {}
        weighted_votes = {}
        
        for vote in votes:
            # Count votes
            vote_counts[vote.decision] = vote_counts.get(vote.decision, 0) + 1
            
            # Weight by confidence and agent priority
            agent = next((m for m in team if m.agent_id == vote.agent_id), None)
            weight = vote.confidence * (agent.role.priority if agent else 1)
            weighted_votes[vote.decision] = weighted_votes.get(vote.decision, 0) + weight
        
        # Determine consensus
        if not weighted_votes:
            # No votes, default to first option
            consensus_decision = decision_options[0] if decision_options else "no_decision"
            consensus_confidence = 0.0
        else:
            consensus_decision = max(weighted_votes, key=weighted_votes.get)
            consensus_confidence = weighted_votes[consensus_decision] / sum(weighted_votes.values())
        
        logger.info(f"✅ Reached consensus for task {task_id}: {consensus_decision}")
        
        return {
            "success": True,
            "task_id": task_id,
            "consensus_decision": consensus_decision,
            "consensus_confidence": consensus_confidence,
            "vote_distribution": vote_counts,
            "total_votes": len(votes)
        }
    
    async def get_team_status(self, team_id: str) -> Dict[str, Any]:
        """Get status of a team"""
        if team_id not in self.teams:
            return {"success": False, "error": "Team not found"}
        
        team = self.teams[team_id]
        
        return {
            "success": True,
            "team_id": team_id,
            "team_size": len(team),
            "members": [m.dict() for m in team],
            "active_tasks": len([t for t in self.active_tasks.values() 
                               if team_id in t.assigned_agents])
        }

