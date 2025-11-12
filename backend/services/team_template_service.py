"""
Team Template Service
Manages pre-built team templates for quick deployment
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from utils.database import Database
from models.team_models import TeamCreate, Team
from services.team_service import TeamService

logger = logging.getLogger(__name__)


class TeamTemplateService:
    """
    Service for managing team templates
    
    Features:
    - Pre-built team templates (Sales, Dev, Support, Marketing, Research)
    - Template CRUD operations
    - Template deployment
    - Template customization
    """
    
    def __init__(self):
        self.db = Database()
        self.templates_collection = "team_templates"
        self.team_service = TeamService()
        
        logger.info("🎨 Team Template Service initialized")
    
    async def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get a team template by ID"""
        template = await self.db.find_one(
            self.templates_collection,
            {"template_id": template_id}
        )
        return template
    
    async def list_templates(
        self,
        category: Optional[str] = None,
        is_featured: Optional[bool] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List team templates with optional filters
        
        Args:
            category: Filter by category (sales, dev, support, marketing, research)
            is_featured: Filter by featured status
            limit: Maximum number of templates to return
        """
        query = {}
        
        if category:
            query["category"] = category
        
        if is_featured is not None:
            query["is_featured"] = is_featured
        
        templates = await self.db.find(
            self.templates_collection,
            query,
            limit=limit
        )
        
        return list(templates)
    
    async def deploy_template(
        self,
        template_id: str,
        customizations: Optional[Dict[str, Any]] = None
    ) -> Team:
        """
        Deploy a team template
        
        Args:
            template_id: Template ID to deploy
            customizations: Optional customizations (name, description, etc.)
            
        Returns:
            Created team
        """
        # Get template
        template = await self.get_template(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")
        
        # Build team creation payload
        team_data = {
            "name": customizations.get("name", template["name"]) if customizations else template["name"],
            "description": customizations.get("description", template["description"]) if customizations else template["description"],
            "agents": template["agents"],
            "workflow_pattern": template["workflow_pattern"],
            "termination": template["termination"],
            "metadata": {
                **template.get("metadata", {}),
                "deployed_from_template": template_id,
                "template_version": template.get("version", "1.0.0")
            }
        }
        
        # Apply customizations
        if customizations:
            if "workflow_pattern" in customizations:
                team_data["workflow_pattern"] = customizations["workflow_pattern"]
            if "termination" in customizations:
                team_data["termination"] = customizations["termination"]
        
        # Create team
        team_create = TeamCreate(**team_data)
        team = await self.team_service.create_team(team_create)
        
        # Update template deployment count
        await self.db.update_one(
            self.templates_collection,
            {"template_id": template_id},
            {"$inc": {"metadata.total_deployments": 1}}
        )
        
        logger.info(f"✅ Deployed template: {template_id} → {team.team_id}")
        return team
    
    async def create_template(
        self,
        template_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a new team template
        
        Args:
            template_data: Template configuration
            
        Returns:
            Created template
        """
        template_id = f"tmpl-{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow()
        
        template_doc = {
            "template_id": template_id,
            "name": template_data["name"],
            "description": template_data["description"],
            "category": template_data.get("category", "custom"),
            "icon": template_data.get("icon", "📋"),
            "agents": template_data["agents"],
            "workflow_pattern": template_data["workflow_pattern"],
            "termination": template_data["termination"],
            "metadata": {
                "estimated_execution_time": template_data.get("estimated_execution_time", "Unknown"),
                "success_rate": template_data.get("success_rate", 0.0),
                "total_deployments": 0,
                "average_rating": 0.0,
                "tags": template_data.get("tags", []),
                "difficulty": template_data.get("difficulty", "intermediate")
            },
            "created_at": now,
            "updated_at": now,
            "created_by": None,  # Single-user system
            "is_public": template_data.get("is_public", True),
            "is_featured": template_data.get("is_featured", False),
            "version": template_data.get("version", "1.0.0")
        }
        
        await self.db.insert_one(self.templates_collection, template_doc)
        
        logger.info(f"✅ Created template: {template_id} - {template_data['name']}")
        return template_doc
    
    async def initialize_default_templates(self) -> None:
        """Initialize default team templates"""
        
        # Check if templates already exist
        existing = await self.db.count_documents(self.templates_collection, {})
        if existing > 0:
            logger.info(f"📋 {existing} templates already exist, skipping initialization")
            return
        
        logger.info("🎨 Initializing default team templates...")
        
        # Sales Team Template
        await self.create_template({
            "name": "Sales Team",
            "description": "Automated lead generation, qualification, and outreach team",
            "category": "sales",
            "icon": "💼",
            "agents": [
                {
                    "role": "lead_generator",
                    "agent_template": "researcher",
                    "order": 1,
                    "description": "Find and qualify leads",
                    "tools": ["web_search", "linkedin_search", "company_research"],
                    "system_prompt_template": "You are a lead generation specialist. Your role is to find and qualify potential leads based on the given criteria."
                },
                {
                    "role": "lead_qualifier",
                    "agent_template": "analyst",
                    "order": 2,
                    "description": "Score and prioritize leads",
                    "tools": ["lead_scoring", "crm_lookup"],
                    "system_prompt_template": "You are a lead qualification expert. Analyze leads and assign scores based on fit, budget, and timing."
                },
                {
                    "role": "outreach_specialist",
                    "agent_template": "writer",
                    "order": 3,
                    "description": "Craft personalized emails",
                    "tools": ["email_composer", "template_library"],
                    "system_prompt_template": "You are an outreach specialist. Create personalized, compelling emails that resonate with prospects."
                },
                {
                    "role": "account_manager",
                    "agent_template": "coordinator",
                    "order": 4,
                    "description": "Track and follow up",
                    "tools": ["crm_update", "calendar_schedule"],
                    "system_prompt_template": "You are an account manager. Track interactions, schedule follow-ups, and maintain relationships."
                }
            ],
            "workflow_pattern": "sequential",
            "termination": {"type": "max_turns", "value": 15},
            "estimated_execution_time": "5-10 minutes",
            "success_rate": 0.92,
            "tags": ["sales", "lead-gen", "automation"],
            "difficulty": "beginner",
            "is_featured": True
        })
        
        # Dev Team Template
        await self.create_template({
            "name": "Dev Team",
            "description": "Full development lifecycle team from design to deployment",
            "category": "dev",
            "icon": "💻",
            "agents": [
                {
                    "role": "architect",
                    "agent_template": "designer",
                    "order": 1,
                    "description": "Design system architecture",
                    "tools": ["diagram_generator", "code_analyzer"],
                    "system_prompt_template": "You are a software architect. Design scalable, maintainable system architectures."
                },
                {
                    "role": "engineer",
                    "agent_template": "coder",
                    "order": 2,
                    "description": "Implement features",
                    "tools": ["code_generator", "git_operations"],
                    "system_prompt_template": "You are a software engineer. Write clean, efficient, well-tested code."
                },
                {
                    "role": "tester",
                    "agent_template": "qa",
                    "order": 3,
                    "description": "Test and validate",
                    "tools": ["test_runner", "bug_tracker"],
                    "system_prompt_template": "You are a QA engineer. Thoroughly test features and identify bugs."
                },
                {
                    "role": "devops",
                    "agent_template": "deployer",
                    "order": 4,
                    "description": "Deploy and monitor",
                    "tools": ["docker", "kubernetes", "monitoring"],
                    "system_prompt_template": "You are a DevOps engineer. Deploy applications reliably and monitor performance."
                },
                {
                    "role": "reviewer",
                    "agent_template": "validator",
                    "order": 5,
                    "description": "Code review and approval",
                    "tools": ["code_review", "quality_check"],
                    "system_prompt_template": "You are a code reviewer. Ensure code quality, security, and best practices."
                }
            ],
            "workflow_pattern": "autonomous",
            "termination": {"type": "keyword", "value": "APPROVED"},
            "estimated_execution_time": "10-20 minutes",
            "success_rate": 0.88,
            "tags": ["dev", "software", "automation"],
            "difficulty": "intermediate",
            "is_featured": True
        })
        
        # Support Team Template
        await self.create_template({
            "name": "Support Team",
            "description": "Customer support escalation team with tiered response",
            "category": "support",
            "icon": "🎧",
            "agents": [
                {
                    "role": "tier1_support",
                    "agent_template": "first_responder",
                    "order": 1,
                    "description": "Handle common issues",
                    "tools": ["knowledge_base", "ticket_system"],
                    "system_prompt_template": "You are a Tier 1 support agent. Handle common customer issues quickly and efficiently."
                },
                {
                    "role": "tier2_support",
                    "agent_template": "specialist",
                    "order": 2,
                    "description": "Handle complex issues",
                    "tools": ["debugging_tools", "system_access"],
                    "system_prompt_template": "You are a Tier 2 support specialist. Resolve complex technical issues."
                },
                {
                    "role": "escalation_manager",
                    "agent_template": "coordinator",
                    "order": 3,
                    "description": "Escalate critical issues",
                    "tools": ["notification_system", "incident_management"],
                    "system_prompt_template": "You are an escalation manager. Handle critical issues and coordinate responses."
                }
            ],
            "workflow_pattern": "round_robin",
            "termination": {"type": "keyword", "value": "RESOLVED"},
            "estimated_execution_time": "3-8 minutes",
            "success_rate": 0.95,
            "tags": ["support", "customer-service", "escalation"],
            "difficulty": "beginner",
            "is_featured": True
        })

        # Marketing Team Template
        await self.create_template({
            "name": "Marketing Team",
            "description": "Content creation, SEO, and social media marketing team",
            "category": "marketing",
            "icon": "📢",
            "agents": [
                {
                    "role": "content_creator",
                    "agent_template": "writer",
                    "order": 1,
                    "description": "Create marketing content",
                    "tools": ["content_generator", "image_generator"],
                    "system_prompt_template": "You are a content creator. Create engaging, high-quality marketing content."
                },
                {
                    "role": "seo_specialist",
                    "agent_template": "optimizer",
                    "order": 2,
                    "description": "Optimize for search",
                    "tools": ["keyword_research", "seo_analyzer"],
                    "system_prompt_template": "You are an SEO specialist. Optimize content for search engines and user engagement."
                },
                {
                    "role": "social_media_manager",
                    "agent_template": "publisher",
                    "order": 3,
                    "description": "Publish and engage",
                    "tools": ["social_media_api", "analytics"],
                    "system_prompt_template": "You are a social media manager. Publish content and engage with audiences."
                },
                {
                    "role": "analytics_specialist",
                    "agent_template": "analyst",
                    "order": 4,
                    "description": "Track performance",
                    "tools": ["analytics_dashboard", "reporting"],
                    "system_prompt_template": "You are an analytics specialist. Track campaign performance and provide insights."
                }
            ],
            "workflow_pattern": "selector",
            "termination": {"type": "max_turns", "value": 20},
            "estimated_execution_time": "8-15 minutes",
            "success_rate": 0.85,
            "tags": ["marketing", "content", "seo", "social-media"],
            "difficulty": "intermediate",
            "is_featured": True
        })

        # Research Team Template
        await self.create_template({
            "name": "Research Team",
            "description": "Multi-agent research and analysis team",
            "category": "research",
            "icon": "🔬",
            "agents": [
                {
                    "role": "researcher",
                    "agent_template": "investigator",
                    "order": 1,
                    "description": "Gather information",
                    "tools": ["web_search", "academic_search", "news_api"],
                    "system_prompt_template": "You are a researcher. Gather comprehensive, accurate information from multiple sources."
                },
                {
                    "role": "analyst",
                    "agent_template": "processor",
                    "order": 2,
                    "description": "Analyze findings",
                    "tools": ["data_analysis", "visualization"],
                    "system_prompt_template": "You are an analyst. Analyze research findings and identify key insights."
                },
                {
                    "role": "writer",
                    "agent_template": "documenter",
                    "order": 3,
                    "description": "Create research report",
                    "tools": ["document_generator", "citation_manager"],
                    "system_prompt_template": "You are a research writer. Create clear, well-structured research reports."
                }
            ],
            "workflow_pattern": "sequential",
            "termination": {"type": "max_turns", "value": 10},
            "estimated_execution_time": "5-12 minutes",
            "success_rate": 0.90,
            "tags": ["research", "analysis", "reporting"],
            "difficulty": "beginner",
            "is_featured": True
        })

        logger.info("✅ Default templates initialized (5 templates)")


# Global instance
team_template_service = TeamTemplateService()

