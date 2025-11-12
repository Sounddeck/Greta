"""
AutoGen Studio Bridge Service
Connects CPAS5 with AutoGen Studio for workflow import/export and agent synchronization
"""

import json
import requests
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AutoGenStudioBridge:
    """
    Bridge between CPAS5 and AutoGen Studio
    
    Features:
    - Import workflows from AutoGen Studio to CPAS5
    - Export CPAS5 agents to AutoGen Studio
    - Bidirectional synchronization
    - Format conversion between systems
    """
    
    def __init__(
        self,
        studio_url: str = "http://localhost:8080",
        workspace_path: str = "./autogen_studio_workspace"
    ):
        """
        Initialize AutoGen Studio bridge
        
        Args:
            studio_url: AutoGen Studio server URL
            workspace_path: Path to AutoGen Studio workspace
        """
        self.studio_url = studio_url
        self.workspace_path = Path(workspace_path)
        self.workflows_path = self.workspace_path / "workflows"
        self.agents_path = self.workspace_path / "agents"
        
        # Create directories if they don't exist
        self.workflows_path.mkdir(parents=True, exist_ok=True)
        self.agents_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🌉 AutoGen Studio Bridge initialized (URL: {studio_url})")
    
    # ==================== Workflow Operations ====================
    
    async def import_workflow_from_studio(
        self,
        workflow_id: str
    ) -> Dict[str, Any]:
        """
        Import workflow from AutoGen Studio to CPAS5
        
        Args:
            workflow_id: AutoGen Studio workflow ID
            
        Returns:
            CPAS5 workflow object
        """
        try:
            # 1. Fetch workflow from Studio
            response = requests.get(
                f"{self.studio_url}/api/workflows/{workflow_id}",
                timeout=10
            )
            response.raise_for_status()
            studio_workflow = response.json()
            
            # 2. Convert to CPAS5 format
            cpas5_workflow = self._convert_studio_to_cpas5_workflow(
                studio_workflow
            )
            
            # 3. Save to CPAS5 database
            from services.workflow_runtime import WorkflowRuntime
            runtime = WorkflowRuntime()
            saved_workflow = await runtime.create_workflow(cpas5_workflow)
            
            logger.info(f"✅ Imported workflow from Studio: {workflow_id}")
            return saved_workflow
            
        except requests.RequestException as e:
            logger.error(f"❌ Failed to import workflow from Studio: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to process workflow: {e}")
            raise
    
    async def export_workflow_to_studio(
        self,
        workflow_id: str
    ) -> Dict[str, Any]:
        """
        Export CPAS5 workflow to AutoGen Studio
        
        Args:
            workflow_id: CPAS5 workflow ID
            
        Returns:
            AutoGen Studio workflow object
        """
        try:
            # 1. Fetch workflow from CPAS5
            from services.workflow_runtime import WorkflowRuntime
            runtime = WorkflowRuntime()
            cpas5_workflow = await runtime.get_workflow(workflow_id)
            
            # 2. Convert to Studio format
            studio_workflow = self._convert_cpas5_to_studio_workflow(
                cpas5_workflow
            )
            
            # 3. Send to Studio
            response = requests.post(
                f"{self.studio_url}/api/workflows",
                json=studio_workflow,
                timeout=10
            )
            response.raise_for_status()
            
            logger.info(f"✅ Exported workflow to Studio: {workflow_id}")
            return response.json()
            
        except requests.RequestException as e:
            logger.error(f"❌ Failed to export workflow to Studio: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to process workflow: {e}")
            raise
    
    # ==================== Team Operations ====================

    async def import_team_from_studio(
        self,
        team_id: str
    ) -> Dict[str, Any]:
        """
        Import team from AutoGen Studio to CPAS5

        Args:
            team_id: AutoGen Studio team ID

        Returns:
            CPAS5 team object
        """
        try:
            # 1. Fetch team from Studio
            response = requests.get(
                f"{self.studio_url}/api/teams/{team_id}",
                timeout=10
            )
            response.raise_for_status()
            studio_team = response.json()

            # 2. Convert to CPAS5 format
            cpas5_team = self._convert_studio_to_cpas5_team(studio_team)

            # 3. Save to CPAS5 database
            from services.team_service import TeamService
            from models.team_models import TeamCreate

            team_service = TeamService()
            team_create = TeamCreate(**cpas5_team)
            saved_team = await team_service.create_team(team_create)

            # 4. Record sync metadata
            await self._record_sync_metadata(
                entity_type="team",
                cpas5_id=saved_team.team_id,
                studio_id=team_id,
                sync_direction="import"
            )

            logger.info(f"✅ Imported team from Studio: {team_id}")
            return saved_team.model_dump()

        except requests.RequestException as e:
            logger.error(f"❌ Failed to import team from Studio: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to process team: {e}")
            raise

    async def export_team_to_studio(
        self,
        team_id: str
    ) -> Dict[str, Any]:
        """
        Export CPAS5 team to AutoGen Studio

        Args:
            team_id: CPAS5 team ID

        Returns:
            AutoGen Studio team object
        """
        try:
            # 1. Fetch team from CPAS5
            from services.team_service import TeamService

            team_service = TeamService()
            cpas5_team = await team_service.get_team(team_id)

            if not cpas5_team:
                raise ValueError(f"Team not found: {team_id}")

            # 2. Convert to Studio format
            studio_team = self._convert_cpas5_to_studio_team(
                cpas5_team.model_dump()
            )

            # 3. Send to Studio
            response = requests.post(
                f"{self.studio_url}/api/teams",
                json=studio_team,
                timeout=10
            )
            response.raise_for_status()
            studio_response = response.json()

            # 4. Record sync metadata
            await self._record_sync_metadata(
                entity_type="team",
                cpas5_id=team_id,
                studio_id=studio_response.get("id"),
                sync_direction="export"
            )

            logger.info(f"✅ Exported team to Studio: {team_id}")
            return studio_response

        except requests.RequestException as e:
            logger.error(f"❌ Failed to export team to Studio: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to process team: {e}")
            raise

    async def list_studio_teams(self) -> List[Dict[str, Any]]:
        """
        List all teams from AutoGen Studio

        Returns:
            List of AutoGen Studio teams
        """
        try:
            response = requests.get(
                f"{self.studio_url}/api/teams",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"❌ Failed to list teams from Studio: {e}")
            raise

    # ==================== Agent Operations ====================

    async def import_agent_from_studio(
        self,
        agent_id: str
    ) -> Dict[str, Any]:
        """
        Import agent from AutoGen Studio to CPAS5
        
        Args:
            agent_id: AutoGen Studio agent ID
            
        Returns:
            CPAS5 agent object
        """
        try:
            # Fetch from Studio
            response = requests.get(
                f"{self.studio_url}/api/agents/{agent_id}",
                timeout=10
            )
            response.raise_for_status()
            studio_agent = response.json()
            
            # Convert to CPAS5 format
            cpas5_agent = self._convert_studio_to_cpas5_agent(studio_agent)
            
            # Save to CPAS5
            from services.agent_builder_service import AgentBuilderService
            builder = AgentBuilderService()
            saved_agent = await builder.create_agent(cpas5_agent)
            
            logger.info(f"✅ Imported agent from Studio: {agent_id}")
            return saved_agent
            
        except Exception as e:
            logger.error(f"❌ Failed to import agent: {e}")
            raise
    
    async def export_agent_to_studio(
        self,
        agent_id: str
    ) -> Dict[str, Any]:
        """
        Export CPAS5 agent to AutoGen Studio
        
        Args:
            agent_id: CPAS5 agent ID
            
        Returns:
            AutoGen Studio agent object
        """
        try:
            # Fetch from CPAS5
            from services.agent_builder_service import AgentBuilderService
            builder = AgentBuilderService()
            cpas5_agent = await builder.get_agent(agent_id)
            
            # Convert to Studio format
            studio_agent = self._convert_cpas5_to_studio_agent(cpas5_agent)
            
            # Send to Studio
            response = requests.post(
                f"{self.studio_url}/api/agents",
                json=studio_agent,
                timeout=10
            )
            response.raise_for_status()
            
            logger.info(f"✅ Exported agent to Studio: {agent_id}")
            return response.json()
            
        except Exception as e:
            logger.error(f"❌ Failed to export agent: {e}")
            raise
    
    # ==================== Format Conversion ====================

    def _convert_studio_to_cpas5_team(
        self,
        studio_team: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert AutoGen Studio team to CPAS5 format"""
        return {
            "name": studio_team.get("name"),
            "description": studio_team.get("description"),
            "agents": [
                {
                    "role": agent.get("role", f"agent_{idx+1}"),
                    "agent_id": agent.get("id"),
                    "order": agent.get("order", idx + 1),
                    "description": agent.get("description", "")
                }
                for idx, agent in enumerate(studio_team.get("agents", []))
            ],
            "workflow_pattern": self._map_workflow_type(
                studio_team.get("workflow_type", "sequential")
            ),
            "termination": studio_team.get("termination", {
                "type": "max_turns",
                "value": 10
            }),
            "metadata": {
                "studio_metadata": studio_team,  # Keep original
                "imported_from_studio": True,
                "studio_id": studio_team.get("id")
            }
        }

    def _convert_cpas5_to_studio_team(
        self,
        cpas5_team: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert CPAS5 team to AutoGen Studio format"""
        return {
            "name": cpas5_team.get("name"),
            "description": cpas5_team.get("description"),
            "agents": [
                {
                    "id": agent.get("agent_id"),
                    "role": agent.get("role"),
                    "order": agent.get("order"),
                    "description": agent.get("description", "")
                }
                for agent in cpas5_team.get("agents", [])
            ],
            "workflow_type": self._map_workflow_type_reverse(
                cpas5_team.get("workflow_pattern", "sequential")
            ),
            "termination": cpas5_team.get("termination", {
                "type": "max_turns",
                "value": 10
            }),
            "metadata": {
                "cpas5_metadata": cpas5_team.get("metadata", {}),
                "exported_from_cpas5": True,
                "cpas5_id": cpas5_team.get("team_id")
            }
        }

    def _convert_studio_to_cpas5_workflow(
        self,
        studio_workflow: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert AutoGen Studio workflow to CPAS5 format"""
        return {
            "name": studio_workflow.get("name"),
            "description": studio_workflow.get("description"),
            "workflow_type": self._map_workflow_type(
                studio_workflow.get("type")
            ),
            "agents": [
                agent["id"] for agent in studio_workflow.get("agents", [])
            ],
            "configuration": {
                "max_rounds": studio_workflow.get("max_rounds", 10),
                "termination": studio_workflow.get("termination", {}),
                "studio_metadata": studio_workflow  # Keep original
            }
        }
    
    def _convert_cpas5_to_studio_workflow(
        self,
        cpas5_workflow: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert CPAS5 workflow to AutoGen Studio format"""
        return {
            "name": cpas5_workflow.get("name"),
            "description": cpas5_workflow.get("description"),
            "type": self._map_workflow_type_reverse(
                cpas5_workflow.get("workflow_type")
            ),
            "agents": [
                {"id": agent_id} 
                for agent_id in cpas5_workflow.get("agents", [])
            ],
            "max_rounds": cpas5_workflow.get("configuration", {}).get(
                "max_rounds", 10
            ),
            "termination": cpas5_workflow.get("configuration", {}).get(
                "termination", {}
            )
        }
    
    def _convert_studio_to_cpas5_agent(
        self,
        studio_agent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert AutoGen Studio agent to CPAS5 format"""
        return {
            "name": studio_agent.get("name"),
            "description": studio_agent.get("description"),
            "agent_type": studio_agent.get("type", "assistant"),
            "capabilities": studio_agent.get("capabilities", []),
            "tools": studio_agent.get("tools", []),
            "configuration": {
                "system_message": studio_agent.get("system_message"),
                "llm_config": studio_agent.get("llm_config", {}),
                "studio_metadata": studio_agent  # Keep original
            }
        }
    
    def _convert_cpas5_to_studio_agent(
        self,
        cpas5_agent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert CPAS5 agent to AutoGen Studio format"""
        return {
            "name": cpas5_agent.get("name"),
            "description": cpas5_agent.get("description"),
            "type": cpas5_agent.get("agent_type", "assistant"),
            "capabilities": cpas5_agent.get("capabilities", []),
            "tools": cpas5_agent.get("tools", []),
            "system_message": cpas5_agent.get("configuration", {}).get(
                "system_message"
            ),
            "llm_config": cpas5_agent.get("configuration", {}).get(
                "llm_config", {}
            )
        }
    
    def _map_workflow_type(self, studio_type: str) -> str:
        """Map Studio workflow type to CPAS5 type"""
        mapping = {
            "sequential": "sequential",
            "parallel": "parallel",
            "hierarchical": "hierarchical",
            "group_chat": "autonomous",  # Map to autonomous
            "round_robin": "round_robin",
            "selector": "selector"
        }
        return mapping.get(studio_type, "sequential")

    def _map_workflow_type_reverse(self, cpas5_type: str) -> str:
        """Map CPAS5 workflow type to Studio type"""
        mapping = {
            "sequential": "sequential",
            "parallel": "parallel",
            "hierarchical": "hierarchical",
            "autonomous": "group_chat",
            "round_robin": "round_robin",
            "selector": "selector"
        }
        return mapping.get(cpas5_type, "sequential")

    async def _record_sync_metadata(
        self,
        entity_type: str,
        cpas5_id: str,
        studio_id: str,
        sync_direction: str
    ) -> None:
        """
        Record synchronization metadata for tracking

        Args:
            entity_type: Type of entity (team, agent, workflow)
            cpas5_id: CPAS5 entity ID
            studio_id: AutoGen Studio entity ID
            sync_direction: Direction of sync (import, export, bidirectional)
        """
        try:
            from datetime import datetime
            from utils.database import Database
            import uuid

            db = Database()
            sync_doc = {
                "sync_id": f"sync-{uuid.uuid4().hex[:12]}",
                "entity_type": entity_type,
                "cpas5_id": cpas5_id,
                "studio_id": studio_id,
                "studio_url": self.studio_url,
                "last_sync_at": datetime.utcnow(),
                "sync_direction": sync_direction,
                "sync_status": "synced",
                "sync_errors": [],
                "metadata": {
                    "cpas5_version": "1.0.0",
                    "studio_version": "0.4.1",
                    "format_version": "1.0"
                },
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }

            # Upsert sync metadata
            await db.update_one(
                "autogen_studio_sync",
                {"cpas5_id": cpas5_id, "entity_type": entity_type},
                {"$set": sync_doc},
                upsert=True
            )

            logger.info(f"📝 Recorded sync metadata: {entity_type} {cpas5_id} ↔ {studio_id}")

        except Exception as e:
            logger.warning(f"⚠️  Failed to record sync metadata: {e}")
            # Don't fail the main operation if metadata recording fails
    
    # ==================== Health Check ====================
    
    def check_studio_connection(self) -> bool:
        """
        Check if AutoGen Studio is accessible
        
        Returns:
            True if connected, False otherwise
        """
        try:
            response = requests.get(
                f"{self.studio_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False


# Global bridge instance
autogen_studio_bridge = AutoGenStudioBridge()

