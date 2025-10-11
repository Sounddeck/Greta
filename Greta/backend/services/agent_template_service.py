"""
Agent Template Service - Week 11-12 Implementation

Manages pre-configured agent templates for quick agent deployment.
Integrates with Week 3 AutoGen and Week 7-8 Advanced AutoGen services.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field


# ============================================================================
# Pydantic Schemas
# ============================================================================


class AgentCapability(BaseModel):
    """Agent capability definition"""

    name: str
    description: str
    enabled: bool = True
    config: Dict[str, Any] = Field(default_factory=dict)


class AgentPersonality(BaseModel):
    """Agent personality traits"""

    tone: str = "professional"  # professional, friendly, casual, formal
    verbosity: str = "balanced"  # concise, balanced, detailed
    creativity: float = 0.7  # 0.0 (conservative) to 1.0 (creative)
    formality: float = 0.5  # 0.0 (casual) to 1.0 (formal)
    traits: List[str] = Field(default_factory=list)  # helpful, analytical, creative, etc.


class AgentToolConfig(BaseModel):
    """Agent tool configuration"""

    tool_name: str
    enabled: bool = True
    config: Dict[str, Any] = Field(default_factory=dict)
    priority: int = 0  # Higher priority tools are preferred


class AgentTemplate(BaseModel):
    """Complete agent template definition"""

    # Identity
    template_id: str
    name: str
    description: str
    version: str = "1.0.0"
    author: str = "system"
    category: str = "general"  # general, coding, research, writing, analysis, etc.
    tags: List[str] = Field(default_factory=list)

    # Agent Configuration
    agent_type: str = "assistant"  # assistant, researcher, coder, critic, etc.
    system_message: str = ""
    model_preference: Optional[str] = None  # Preferred model name
    max_tokens: int = 2048
    temperature: float = 0.7

    # Personality
    personality: AgentPersonality = Field(default_factory=AgentPersonality)

    # Capabilities
    capabilities: List[AgentCapability] = Field(default_factory=list)

    # Tools
    tools: List[AgentToolConfig] = Field(default_factory=list)

    # Advanced Features
    use_rag: bool = False
    rag_collection: Optional[str] = None
    use_memory: bool = False
    memory_config: Dict[str, Any] = Field(default_factory=dict)
    use_code_execution: bool = False

    # Metadata
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    usage_count: int = 0
    rating: float = 0.0
    is_public: bool = True
    parent_template_id: Optional[str] = None  # For template inheritance
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TemplateStats(BaseModel):
    """Template registry statistics"""

    total_templates: int
    templates_by_category: Dict[str, int]
    templates_by_author: Dict[str, int]
    most_used: List[str]
    highest_rated: List[str]


# ============================================================================
# Agent Template Service
# ============================================================================


class AgentTemplateService:
    """
    Agent Template Service

    Features:
    - Template storage and versioning
    - Template CRUD operations
    - Template search and filtering
    - Template customization
    - Template marketplace/registry
    - Template instantiation
    - Import/export functionality
    """

    def __init__(
        self,
        templates_dir: str = "./data/templates",
        registry_file: str = "./data/templates/registry.json",
    ):
        self.templates_dir = Path(templates_dir)
        self.registry_file = Path(registry_file)

        # Create directories
        self.templates_dir.mkdir(parents=True, exist_ok=True)

        # Template registry (in-memory)
        self.templates: Dict[str, AgentTemplate] = {}

        # Load existing templates
        self._load_registry()

        # Load default templates if registry is empty
        if not self.templates:
            self._create_default_templates()

        logger.info(f"Agent Template Service initialized")
        logger.info(f"  Directory: {self.templates_dir}")
        logger.info(f"  Templates: {len(self.templates)}")

    def _load_registry(self):
        """Load template registry from disk"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r") as f:
                    data = json.load(f)

                for template_id, template_data in data.items():
                    self.templates[template_id] = AgentTemplate(**template_data)

                logger.info(f"Loaded {len(self.templates)} templates from registry")

            except Exception as e:
                logger.error(f"Error loading registry: {e}")
        else:
            logger.info("No existing registry found, will create default templates")

    def _save_registry(self):
        """Save template registry to disk"""
        try:
            data = {
                template_id: template.model_dump()
                for template_id, template in self.templates.items()
            }

            with open(self.registry_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved registry with {len(self.templates)} templates")

        except Exception as e:
            logger.error(f"Error saving registry: {e}")

    def _create_default_templates(self):
        """Create default template library"""
        logger.info("Creating default template library...")

        # 1. General Assistant Template
        self.templates["assistant_general"] = AgentTemplate(
            template_id="assistant_general",
            name="General Assistant",
            description="A helpful general-purpose assistant for various tasks",
            category="general",
            agent_type="assistant",
            system_message="You are a helpful AI assistant. Provide clear, accurate, and helpful responses to user queries.",
            tags=["general", "assistant", "helpful"],
            personality=AgentPersonality(
                tone="friendly",
                verbosity="balanced",
                creativity=0.7,
                formality=0.5,
                traits=["helpful", "patient", "clear"],
            ),
            capabilities=[
                AgentCapability(
                    name="general_knowledge",
                    description="Answer general knowledge questions",
                ),
                AgentCapability(
                    name="task_assistance", description="Help with various tasks"
                ),
            ],
        )

        # 2. Code Assistant Template
        self.templates["coder_python"] = AgentTemplate(
            template_id="coder_python",
            name="Python Coding Assistant",
            description="Expert Python developer for coding tasks, debugging, and code review",
            category="coding",
            agent_type="coder",
            system_message="You are an expert Python developer. Write clean, efficient, well-documented code. Follow PEP 8 style guidelines.",
            tags=["coding", "python", "development"],
            personality=AgentPersonality(
                tone="professional",
                verbosity="detailed",
                creativity=0.6,
                formality=0.7,
                traits=["precise", "analytical", "thorough"],
            ),
            capabilities=[
                AgentCapability(name="code_writing", description="Write Python code"),
                AgentCapability(name="debugging", description="Debug Python code"),
                AgentCapability(name="code_review", description="Review code quality"),
            ],
            use_code_execution=True,
        )

        # 3. Research Assistant Template
        self.templates["researcher_academic"] = AgentTemplate(
            template_id="researcher_academic",
            name="Academic Researcher",
            description="Research assistant for academic and scientific inquiries",
            category="research",
            agent_type="researcher",
            system_message="You are an academic research assistant. Provide well-researched, cited, and thorough responses. Focus on accuracy and depth.",
            tags=["research", "academic", "analysis"],
            personality=AgentPersonality(
                tone="formal",
                verbosity="detailed",
                creativity=0.5,
                formality=0.9,
                traits=["thorough", "analytical", "precise", "scholarly"],
            ),
            capabilities=[
                AgentCapability(
                    name="literature_review", description="Review academic literature"
                ),
                AgentCapability(
                    name="data_analysis", description="Analyze research data"
                ),
                AgentCapability(
                    name="citation", description="Provide proper citations"
                ),
            ],
            use_rag=True,
        )

        # 4. Creative Writer Template
        self.templates["writer_creative"] = AgentTemplate(
            template_id="writer_creative",
            name="Creative Writer",
            description="Creative writing assistant for stories, articles, and content",
            category="writing",
            agent_type="assistant",
            system_message="You are a creative writing assistant. Help users craft engaging, well-written content with vivid descriptions and compelling narratives.",
            tags=["writing", "creative", "content"],
            personality=AgentPersonality(
                tone="friendly",
                verbosity="detailed",
                creativity=0.9,
                formality=0.4,
                traits=["creative", "imaginative", "expressive", "engaging"],
            ),
            capabilities=[
                AgentCapability(
                    name="story_writing", description="Write creative stories"
                ),
                AgentCapability(
                    name="content_creation", description="Create engaging content"
                ),
                AgentCapability(name="editing", description="Edit and improve writing"),
            ],
        )

        # 5. Data Analyst Template
        self.templates["analyst_data"] = AgentTemplate(
            template_id="analyst_data",
            name="Data Analyst",
            description="Data analysis expert for insights, visualization, and reporting",
            category="analysis",
            agent_type="assistant",
            system_message="You are a data analyst. Analyze data, identify patterns, and provide actionable insights. Use clear visualizations and statistical methods.",
            tags=["analysis", "data", "statistics"],
            personality=AgentPersonality(
                tone="professional",
                verbosity="balanced",
                creativity=0.6,
                formality=0.7,
                traits=["analytical", "detail-oriented", "logical", "insightful"],
            ),
            capabilities=[
                AgentCapability(
                    name="data_analysis", description="Analyze datasets"
                ),
                AgentCapability(
                    name="visualization", description="Create data visualizations"
                ),
                AgentCapability(
                    name="reporting", description="Generate analysis reports"
                ),
            ],
            use_code_execution=True,
        )

        # 6. Technical Critic Template
        self.templates["critic_technical"] = AgentTemplate(
            template_id="critic_technical",
            name="Technical Critic",
            description="Critical reviewer for technical work, code, and documentation",
            category="review",
            agent_type="critic",
            system_message="You are a technical critic. Provide constructive, detailed feedback on technical work. Identify issues, suggest improvements, and ensure quality.",
            tags=["review", "critic", "quality"],
            personality=AgentPersonality(
                tone="professional",
                verbosity="detailed",
                creativity=0.5,
                formality=0.8,
                traits=["critical", "thorough", "constructive", "detail-oriented"],
            ),
            capabilities=[
                AgentCapability(name="code_review", description="Review code quality"),
                AgentCapability(
                    name="documentation_review", description="Review documentation"
                ),
                AgentCapability(
                    name="architecture_review", description="Review system architecture"
                ),
            ],
        )

        # 7. Customer Support Template
        self.templates["support_customer"] = AgentTemplate(
            template_id="support_customer",
            name="Customer Support Agent",
            description="Friendly customer support agent for handling inquiries and issues",
            category="support",
            agent_type="assistant",
            system_message="You are a customer support agent. Be friendly, patient, and helpful. Resolve customer issues efficiently and professionally.",
            tags=["support", "customer-service", "help"],
            personality=AgentPersonality(
                tone="friendly",
                verbosity="balanced",
                creativity=0.6,
                formality=0.6,
                traits=["patient", "empathetic", "helpful", "professional"],
            ),
            capabilities=[
                AgentCapability(
                    name="issue_resolution", description="Resolve customer issues"
                ),
                AgentCapability(
                    name="product_knowledge", description="Provide product information"
                ),
                AgentCapability(
                    name="escalation", description="Escalate complex issues"
                ),
            ],
            use_rag=True,
        )

        # 8. Tutor/Educator Template
        self.templates["tutor_general"] = AgentTemplate(
            template_id="tutor_general",
            name="Educational Tutor",
            description="Patient tutor for teaching and explaining concepts",
            category="education",
            agent_type="assistant",
            system_message="You are an educational tutor. Explain concepts clearly, use examples, and adapt to the student's level. Be patient and encouraging.",
            tags=["education", "tutor", "teaching"],
            personality=AgentPersonality(
                tone="friendly",
                verbosity="detailed",
                creativity=0.7,
                formality=0.5,
                traits=["patient", "clear", "encouraging", "adaptive"],
            ),
            capabilities=[
                AgentCapability(
                    name="concept_explanation", description="Explain complex concepts"
                ),
                AgentCapability(
                    name="example_generation", description="Generate helpful examples"
                ),
                AgentCapability(
                    name="assessment", description="Assess understanding"
                ),
            ],
        )

        self._save_registry()
        logger.info(f"Created {len(self.templates)} default templates")

    async def create_template(
        self,
        template_id: str,
        name: str,
        description: str,
        agent_type: str = "assistant",
        category: str = "general",
        system_message: str = "",
        **kwargs,
    ) -> AgentTemplate:
        """
        Create a new agent template

        Args:
            template_id: Unique template ID
            name: Template name
            description: Template description
            agent_type: Agent type (assistant, researcher, coder, etc.)
            category: Template category
            system_message: Agent system message
            **kwargs: Additional template fields

        Returns:
            AgentTemplate
        """
        template = AgentTemplate(
            template_id=template_id,
            name=name,
            description=description,
            agent_type=agent_type,
            category=category,
            system_message=system_message,
            **kwargs,
        )

        self.templates[template_id] = template
        self._save_registry()

        logger.info(f"Created template: {template_id} ({name})")
        return template

    async def get_template(self, template_id: str) -> Optional[AgentTemplate]:
        """Get template by ID"""
        template = self.templates.get(template_id)

        if template:
            # Update usage stats
            template.usage_count += 1
            template.updated_at = datetime.utcnow().isoformat()
            self._save_registry()

        return template

    async def list_templates(
        self,
        category: Optional[str] = None,
        agent_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        author: Optional[str] = None,
        is_public: Optional[bool] = None,
        limit: int = 100,
    ) -> List[AgentTemplate]:
        """
        List templates with optional filtering

        Args:
            category: Filter by category
            agent_type: Filter by agent type
            tags: Filter by tags (any match)
            author: Filter by author
            is_public: Filter by public/private
            limit: Maximum results

        Returns:
            List of AgentTemplate
        """
        templates = list(self.templates.values())

        # Apply filters
        if category:
            templates = [t for t in templates if t.category == category]

        if agent_type:
            templates = [t for t in templates if t.agent_type == agent_type]

        if tags:
            templates = [t for t in templates if any(tag in t.tags for tag in tags)]

        if author:
            templates = [t for t in templates if t.author == author]

        if is_public is not None:
            templates = [t for t in templates if t.is_public == is_public]

        # Sort by usage count (most used first)
        templates.sort(key=lambda t: t.usage_count, reverse=True)

        return templates[:limit]

    async def update_template(
        self, template_id: str, updates: Dict[str, Any]
    ) -> Optional[AgentTemplate]:
        """Update template fields"""
        if template_id not in self.templates:
            return None

        template = self.templates[template_id]

        # Update fields
        for key, value in updates.items():
            if hasattr(template, key):
                setattr(template, key, value)

        template.updated_at = datetime.utcnow().isoformat()
        self._save_registry()

        logger.info(f"Updated template: {template_id}")
        return template

    async def delete_template(self, template_id: str) -> bool:
        """Delete template from registry"""
        if template_id not in self.templates:
            logger.warning(f"Template not found: {template_id}")
            return False

        del self.templates[template_id]
        self._save_registry()

        logger.info(f"Deleted template: {template_id}")
        return True

    async def search_templates(self, query: str, limit: int = 10) -> List[AgentTemplate]:
        """
        Search templates by name, description, or tags

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching templates
        """
        query_lower = query.lower()
        results = []

        for template in self.templates.values():
            # Search in name, description, tags, category
            if (
                query_lower in template.name.lower()
                or query_lower in template.description.lower()
                or any(query_lower in tag.lower() for tag in template.tags)
                or query_lower in template.category.lower()
            ):
                results.append(template)

        # Sort by usage count
        results.sort(key=lambda t: t.usage_count, reverse=True)

        return results[:limit]

    async def customize_template(
        self,
        base_template_id: str,
        new_template_id: str,
        name: str,
        customizations: Dict[str, Any],
    ) -> Optional[AgentTemplate]:
        """
        Create a customized version of an existing template

        Args:
            base_template_id: ID of template to customize
            new_template_id: ID for new customized template
            name: Name for customized template
            customizations: Fields to customize

        Returns:
            New customized AgentTemplate or None if base not found
        """
        base_template = self.templates.get(base_template_id)
        if not base_template:
            logger.error(f"Base template not found: {base_template_id}")
            return None

        # Create copy of base template
        template_data = base_template.model_dump()
        template_data["template_id"] = new_template_id
        template_data["name"] = name
        template_data["parent_template_id"] = base_template_id
        template_data["created_at"] = datetime.utcnow().isoformat()
        template_data["updated_at"] = datetime.utcnow().isoformat()
        template_data["usage_count"] = 0

        # Apply customizations
        template_data.update(customizations)

        # Create new template
        new_template = AgentTemplate(**template_data)
        self.templates[new_template_id] = new_template
        self._save_registry()

        logger.info(f"Customized template {base_template_id} -> {new_template_id}")
        return new_template

    async def clone_template(
        self, template_id: str, new_template_id: str, new_name: str
    ) -> Optional[AgentTemplate]:
        """Clone an existing template"""
        return await self.customize_template(
            template_id, new_template_id, new_name, {}
        )

    async def get_stats(self) -> TemplateStats:
        """Get template registry statistics"""
        templates = list(self.templates.values())

        # Count by category
        by_category = {}
        for template in templates:
            by_category[template.category] = by_category.get(template.category, 0) + 1

        # Count by author
        by_author = {}
        for template in templates:
            by_author[template.author] = by_author.get(template.author, 0) + 1

        # Most used
        sorted_by_usage = sorted(templates, key=lambda t: t.usage_count, reverse=True)
        most_used = [t.template_id for t in sorted_by_usage[:5]]

        # Highest rated
        sorted_by_rating = sorted(templates, key=lambda t: t.rating, reverse=True)
        highest_rated = [t.template_id for t in sorted_by_rating[:5]]

        return TemplateStats(
            total_templates=len(templates),
            templates_by_category=by_category,
            templates_by_author=by_author,
            most_used=most_used,
            highest_rated=highest_rated,
        )

    async def export_template(
        self, template_id: str, format: str = "json"
    ) -> Optional[str]:
        """
        Export template to JSON or YAML

        Args:
            template_id: Template ID
            format: Export format (json or yaml)

        Returns:
            File path or None if template not found
        """
        template = self.templates.get(template_id)
        if not template:
            return None

        export_dir = self.templates_dir / "exports"
        export_dir.mkdir(exist_ok=True)

        if format == "json":
            file_path = export_dir / f"{template_id}.json"
            with open(file_path, "w") as f:
                json.dump(template.model_dump(), f, indent=2)
        elif format == "yaml":
            file_path = export_dir / f"{template_id}.yaml"
            # YAML export would require pyyaml
            # For now, use JSON
            with open(file_path, "w") as f:
                json.dump(template.model_dump(), f, indent=2)
        else:
            logger.error(f"Unsupported format: {format}")
            return None

        logger.info(f"Exported template {template_id} to {file_path}")
        return str(file_path)

    async def import_template(self, file_path: str) -> Optional[AgentTemplate]:
        """
        Import template from JSON or YAML file

        Args:
            file_path: Path to template file

        Returns:
            Imported AgentTemplate or None if import failed
        """
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            template = AgentTemplate(**data)
            self.templates[template.template_id] = template
            self._save_registry()

            logger.info(f"Imported template: {template.template_id}")
            return template

        except Exception as e:
            logger.error(f"Error importing template: {e}")
            return None

    async def rate_template(self, template_id: str, rating: float) -> bool:
        """
        Rate a template (0.0 to 5.0)

        Args:
            template_id: Template ID
            rating: Rating value (0.0 to 5.0)

        Returns:
            True if successful
        """
        template = self.templates.get(template_id)
        if not template:
            return False

        # Simple average (in production, would track individual ratings)
        if template.rating == 0.0:
            template.rating = rating
        else:
            template.rating = (template.rating + rating) / 2

        template.updated_at = datetime.utcnow().isoformat()
        self._save_registry()

        logger.info(f"Rated template {template_id}: {rating}")
        return True

    async def instantiate_template(
        self, template_id: str, agent_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Instantiate a template into an agent configuration

        This creates an agent configuration dict that can be used with
        Week 3 AutoGen service or Week 7-8 Advanced AutoGen service.

        Args:
            template_id: Template ID
            agent_name: Optional custom agent name

        Returns:
            Agent configuration dict or None if template not found
        """
        template = await self.get_template(template_id)
        if not template:
            return None

        # Create agent configuration
        agent_config = {
            "name": agent_name or template.name,
            "agent_type": template.agent_type,
            "system_message": template.system_message,
            "llm_config": {
                "model": template.model_preference or "llama-2-7b-chat",
                "max_tokens": template.max_tokens,
                "temperature": template.temperature,
            },
            "personality": template.personality.model_dump(),
            "capabilities": [cap.model_dump() for cap in template.capabilities],
            "tools": [tool.model_dump() for tool in template.tools],
            "use_rag": template.use_rag,
            "rag_collection": template.rag_collection,
            "use_memory": template.use_memory,
            "memory_config": template.memory_config,
            "use_code_execution": template.use_code_execution,
            "template_id": template_id,
            "template_version": template.version,
        }

        logger.info(f"Instantiated template {template_id} as agent")
        return agent_config

    async def get_template_categories(self) -> List[str]:
        """Get list of all template categories"""
        categories = set()
        for template in self.templates.values():
            categories.add(template.category)
        return sorted(list(categories))

    async def get_templates_by_capability(
        self, capability_name: str
    ) -> List[AgentTemplate]:
        """Get templates that have a specific capability"""
        results = []
        for template in self.templates.values():
            for cap in template.capabilities:
                if cap.name == capability_name:
                    results.append(template)
                    break
        return results

    async def validate_template(self, template: AgentTemplate) -> tuple[bool, List[str]]:
        """
        Validate template configuration

        Args:
            template: Template to validate

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check required fields
        if not template.template_id:
            errors.append("template_id is required")
        if not template.name:
            errors.append("name is required")
        if not template.description:
            errors.append("description is required")

        # Check temperature range
        if not 0.0 <= template.temperature <= 2.0:
            errors.append("temperature must be between 0.0 and 2.0")

        # Check personality creativity
        if not 0.0 <= template.personality.creativity <= 1.0:
            errors.append("personality.creativity must be between 0.0 and 1.0")

        # Check personality formality
        if not 0.0 <= template.personality.formality <= 1.0:
            errors.append("personality.formality must be between 0.0 and 1.0")

        # Check rating
        if not 0.0 <= template.rating <= 5.0:
            errors.append("rating must be between 0.0 and 5.0")

        return (len(errors) == 0, errors)


# ============================================================================
# Singleton Instance
# ============================================================================

agent_template_service = AgentTemplateService()

