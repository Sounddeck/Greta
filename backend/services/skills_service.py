"""
🎯 Skills Service - Natural Language Task Routing System
Inspired by ClaudeFlow's skills system for intelligent task activation

Features:
- Natural language skill activation (no slash commands)
- Progressive disclosure for context efficiency
- Semantic matching for skill selection
- 5 core skills: SPARC, Pair Programming, Swarm Orchestration, Code Review, Performance Analysis
- PHASE 1 ENHANCEMENT: Auto-save skills from successful execution patterns
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field
from datetime import datetime
import re
import uuid

logger = logging.getLogger(__name__)


class Skill(BaseModel):
    """Skill definition with metadata and activation patterns"""
    name: str
    display_name: str
    description: str
    tags: List[str] = Field(default_factory=list)
    category: str
    activation_keywords: List[str] = Field(default_factory=list)
    activation_patterns: List[str] = Field(default_factory=list)  # Regex patterns
    handler_function: str  # Name of the handler method
    context_requirements: Dict[str, Any] = Field(default_factory=dict)
    examples: List[str] = Field(default_factory=list)
    priority: int = 0  # Higher priority skills matched first

    # PHASE 1 ENHANCEMENT: Auto-save metadata
    skill_id: str = Field(default_factory=lambda: f"skill_{uuid.uuid4().hex[:12]}")
    is_auto_saved: bool = False  # True if auto-saved from execution pattern
    source_pattern_id: Optional[str] = None  # ID of the pattern this skill was created from
    usage_count: int = 0  # Number of times this skill has been used
    success_rate: float = 0.0  # Success rate (0.0 to 1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class SkillMatch(BaseModel):
    """Result of skill matching"""
    skill: Skill
    confidence: float  # 0.0 to 1.0
    matched_keywords: List[str] = Field(default_factory=list)
    matched_patterns: List[str] = Field(default_factory=list)


class SkillsService:
    """
    Natural Language Skills System
    
    Provides intelligent task routing based on natural language input,
    eliminating the need for memorizing commands or slash commands.
    """
    
    def __init__(self):
        self.skills: Dict[str, Skill] = {}
        self.skill_handlers: Dict[str, Callable] = {}
        self._load_core_skills()
        logger.info("✅ Skills Service initialized with 5 core skills")
    
    def _load_core_skills(self):
        """Load the 5 core skills inspired by ClaudeFlow"""
        
        # Skill 1: SPARC Methodology
        self.register_skill(Skill(
            name="sparc_methodology",
            display_name="SPARC Development Methodology",
            description="Systematic feature development using SPARC (Specification, Pseudocode, Architecture, Refinement, Completion)",
            tags=["development", "tdd", "methodology", "systematic"],
            category="development",
            activation_keywords=[
                "build feature", "develop feature", "create feature",
                "tdd", "test driven", "systematic development",
                "sparc", "specification", "architecture"
            ],
            activation_patterns=[
                r"build\s+(a\s+)?feature",
                r"develop\s+.*\s+with\s+tests",
                r"create\s+.*\s+systematically",
                r"use\s+sparc",
                r"test[\s-]driven\s+development"
            ],
            handler_function="handle_sparc_methodology",
            context_requirements={
                "requires_specification": True,
                "requires_tests": True
            },
            examples=[
                "Build a user authentication feature with tests",
                "Develop a payment processing system using TDD",
                "Create a notification service systematically"
            ],
            priority=10
        ))
        
        # Skill 2: Pair Programming
        self.register_skill(Skill(
            name="pair_programming",
            display_name="AI Pair Programming",
            description="Collaborative coding with AI assistance, code review, and real-time suggestions",
            tags=["coding", "collaboration", "review", "assistance"],
            category="development",
            activation_keywords=[
                "pair program", "code together", "help me code",
                "review code", "code review", "debug together",
                "collaborate on code"
            ],
            activation_patterns=[
                r"pair\s+program",
                r"code\s+together",
                r"help\s+(me\s+)?(write|code|debug)",
                r"review\s+(my\s+)?code",
                r"collaborate\s+on"
            ],
            handler_function="handle_pair_programming",
            context_requirements={
                "requires_code_context": True
            },
            examples=[
                "Help me code a REST API endpoint",
                "Review my authentication logic",
                "Debug this database query together"
            ],
            priority=9
        ))
        
        # Skill 3: Swarm Orchestration
        self.register_skill(Skill(
            name="swarm_orchestration",
            display_name="Multi-Agent Swarm Orchestration",
            description="Coordinate multiple agents for complex tasks using swarm intelligence",
            tags=["orchestration", "multi-agent", "swarm", "coordination"],
            category="orchestration",
            activation_keywords=[
                "orchestrate agents", "coordinate agents", "multi-agent",
                "swarm", "team of agents", "agent collaboration",
                "parallel execution"
            ],
            activation_patterns=[
                r"orchestrate\s+(multiple\s+)?agents",
                r"coordinate\s+.*\s+agents",
                r"multi[\s-]agent",
                r"swarm\s+(of\s+)?agents",
                r"team\s+of\s+agents",
                r"parallel\s+execution"
            ],
            handler_function="handle_swarm_orchestration",
            context_requirements={
                "requires_task_decomposition": True,
                "requires_agent_selection": True
            },
            examples=[
                "Orchestrate multiple agents to analyze this codebase",
                "Coordinate a team of agents for data processing",
                "Use swarm intelligence to optimize this workflow"
            ],
            priority=8
        ))
        
        # Skill 4: Code Review & Analysis
        self.register_skill(Skill(
            name="code_review",
            display_name="Comprehensive Code Review",
            description="Deep code analysis covering security, performance, best practices, and maintainability",
            tags=["review", "analysis", "security", "performance", "quality"],
            category="analysis",
            activation_keywords=[
                "code review", "analyze code", "review this code",
                "security audit", "performance review", "code quality",
                "best practices", "code analysis"
            ],
            activation_patterns=[
                r"(code\s+)?review",
                r"analyze\s+(this\s+)?code",
                r"security\s+audit",
                r"performance\s+(review|analysis)",
                r"code\s+quality",
                r"best\s+practices"
            ],
            handler_function="handle_code_review",
            context_requirements={
                "requires_code_input": True
            },
            examples=[
                "Review this authentication module for security issues",
                "Analyze the performance of this database query",
                "Check this code for best practices violations"
            ],
            priority=7
        ))
        
        # Skill 5: Performance Analysis & Optimization
        self.register_skill(Skill(
            name="performance_optimization",
            display_name="Performance Analysis & Optimization",
            description="Analyze system performance, identify bottlenecks, and suggest optimizations",
            tags=["performance", "optimization", "profiling", "benchmarking"],
            category="optimization",
            activation_keywords=[
                "optimize performance", "performance analysis", "find bottlenecks",
                "speed up", "improve performance", "benchmark",
                "profiling", "performance tuning"
            ],
            activation_patterns=[
                r"optimize\s+performance",
                r"performance\s+(analysis|optimization)",
                r"find\s+bottlenecks",
                r"speed\s+up",
                r"improve\s+performance",
                r"benchmark",
                r"profil(e|ing)"
            ],
            handler_function="handle_performance_optimization",
            context_requirements={
                "requires_performance_data": False  # Can work with or without data
            },
            examples=[
                "Optimize the performance of this API endpoint",
                "Find bottlenecks in the database queries",
                "Benchmark this algorithm and suggest improvements"
            ],
            priority=6
        ))
        
        logger.info(f"✅ Loaded {len(self.skills)} core skills")
    
    def register_skill(self, skill: Skill):
        """Register a new skill"""
        self.skills[skill.name] = skill
        logger.debug(f"Registered skill: {skill.display_name}")
    
    def get_skill(self, name: str) -> Optional[Skill]:
        """Get skill by name"""
        return self.skills.get(name)
    
    def list_skills(self, category: Optional[str] = None) -> List[Skill]:
        """List all skills, optionally filtered by category"""
        if category:
            return [s for s in self.skills.values() if s.category == category]
        return list(self.skills.values())
    
    async def match_skill(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        threshold: float = 0.3
    ) -> Optional[SkillMatch]:
        """
        Match user input to the most appropriate skill
        
        Args:
            user_input: Natural language input from user
            context: Optional context for matching
            threshold: Minimum confidence threshold (0.0 to 1.0)
        
        Returns:
            SkillMatch if a skill matches above threshold, None otherwise
        """
        user_input_lower = user_input.lower()
        matches: List[SkillMatch] = []
        
        # Sort skills by priority (highest first)
        sorted_skills = sorted(self.skills.values(), key=lambda s: s.priority, reverse=True)
        
        for skill in sorted_skills:
            confidence = 0.0
            matched_keywords = []
            matched_patterns = []

            # Keyword matching (40% weight)
            # If ANY keyword matches, give full keyword score
            keyword_score = 0.0
            for keyword in skill.activation_keywords:
                if keyword.lower() in user_input_lower:
                    matched_keywords.append(keyword)

            if matched_keywords:
                keyword_score = 0.4  # Full keyword weight if any keyword matches

            confidence += keyword_score

            # Pattern matching (60% weight)
            # If ANY pattern matches, give full pattern score
            pattern_score = 0.0
            for pattern in skill.activation_patterns:
                if re.search(pattern, user_input_lower):
                    matched_patterns.append(pattern)

            if matched_patterns:
                pattern_score = 0.6  # Full pattern weight if any pattern matches

            confidence += pattern_score

            # Priority boost (up to 10% based on priority)
            priority_boost = (skill.priority / 100.0) * 0.1
            confidence = min(confidence + priority_boost, 1.0)
            
            if confidence >= threshold:
                matches.append(SkillMatch(
                    skill=skill,
                    confidence=confidence,
                    matched_keywords=matched_keywords,
                    matched_patterns=matched_patterns
                ))
        
        # Return highest confidence match
        if matches:
            best_match = max(matches, key=lambda m: m.confidence)
            logger.info(f"✅ Matched skill: {best_match.skill.display_name} (confidence: {best_match.confidence:.2f})")
            return best_match
        
        logger.debug(f"No skill matched for input: {user_input[:100]}")
        return None
    
    def register_handler(self, skill_name: str, handler: Callable):
        """Register a handler function for a skill"""
        self.skill_handlers[skill_name] = handler
        logger.debug(f"Registered handler for skill: {skill_name}")
    
    async def execute_skill(
        self,
        skill_name: str,
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a skill by name
        
        Args:
            skill_name: Name of the skill to execute
            user_input: User's input
            context: Optional execution context
        
        Returns:
            Execution result dictionary
        """
        skill = self.get_skill(skill_name)
        if not skill:
            return {
                "success": False,
                "error": f"Skill not found: {skill_name}"
            }
        
        handler = self.skill_handlers.get(skill_name)
        if not handler:
            return {
                "success": False,
                "error": f"No handler registered for skill: {skill_name}",
                "skill": skill.dict()
            }
        
        try:
            result = await handler(user_input, context or {})
            return {
                "success": True,
                "skill": skill.display_name,
                "result": result
            }
        except Exception as e:
            logger.error(f"Error executing skill {skill_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "skill": skill.display_name
            }

    def get_categories(self) -> List[str]:
        """Get list of all skill categories"""
        categories = set(skill.category for skill in self.skills.values())
        return sorted(list(categories))

    # ============================================================================
    # PHASE 1 ENHANCEMENT: AUTO-SAVE SKILLS FROM EXECUTION PATTERNS
    # ============================================================================

    async def auto_save_skill_from_pattern(
        self,
        pattern: Dict[str, Any],
        execution_data: Dict[str, Any]
    ) -> Optional[Skill]:
        """
        PHASE 1 ENHANCEMENT: Automatically create a skill from a successful execution pattern

        This method analyzes a successful execution pattern and creates a reusable skill
        that can be applied to similar future tasks.

        Args:
            pattern: Reasoning pattern from ReasoningBank
            execution_data: Execution data including tools used, response, etc.

        Returns:
            Created Skill if successful, None otherwise
        """
        try:
            # Extract pattern information
            pattern_id = pattern.get("pattern_id", "unknown")
            pattern_type = pattern.get("pattern_type", "general")
            title = pattern.get("title", "Untitled Pattern")
            problem_description = pattern.get("problem_description", "")
            solution_summary = pattern.get("solution_summary", "")
            tags = pattern.get("tags", [])

            # Determine skill category based on pattern type
            category_mapping = {
                "research": "research",
                "debugging": "development",
                "optimization": "optimization",
                "agent_execution": "general",
                "coding": "development",
                "data_analysis": "analysis"
            }
            category = category_mapping.get(pattern_type, "general")

            # Generate skill name from title
            skill_name = self._generate_skill_name(title)

            # Check if skill already exists
            if skill_name in self.skills:
                logger.debug(f"Skill already exists: {skill_name}")
                return self.skills[skill_name]

            # Extract activation keywords from problem description and tags
            activation_keywords = self._extract_keywords(problem_description, tags)

            # Generate activation patterns from keywords
            activation_patterns = self._generate_patterns(activation_keywords)

            # Extract examples from execution data
            examples = self._extract_examples(execution_data)

            # Determine priority based on pattern complexity and success
            priority = self._calculate_skill_priority(pattern, execution_data)

            # Create the skill
            skill = Skill(
                name=skill_name,
                display_name=title,
                description=solution_summary or problem_description[:200],
                tags=tags,
                category=category,
                activation_keywords=activation_keywords,
                activation_patterns=activation_patterns,
                handler_function=f"handle_auto_saved_{skill_name}",
                context_requirements={
                    "auto_saved": True,
                    "pattern_type": pattern_type
                },
                examples=examples,
                priority=priority,
                is_auto_saved=True,
                source_pattern_id=pattern_id,
                usage_count=0,
                success_rate=1.0  # Start with 100% since it's from a successful execution
            )

            # Register the skill
            self.register_skill(skill)

            # Create a dynamic handler for this skill
            self._create_dynamic_handler(skill, pattern, execution_data)

            logger.info(f"✅ Auto-saved skill from pattern: {skill.display_name} ({skill_name})")
            return skill

        except Exception as e:
            logger.error(f"❌ Failed to auto-save skill from pattern: {e}")
            return None

    def _generate_skill_name(self, title: str) -> str:
        """Generate a valid skill name from title"""
        # Convert to lowercase, replace spaces with underscores, remove special chars
        name = title.lower()
        name = re.sub(r'[^a-z0-9\s_]', '', name)
        name = re.sub(r'\s+', '_', name)
        name = name[:50]  # Limit length
        return name or f"skill_{uuid.uuid4().hex[:8]}"

    def _extract_keywords(self, problem_description: str, tags: List[str]) -> List[str]:
        """Extract activation keywords from problem description and tags"""
        keywords = []

        # Add tags as keywords
        keywords.extend(tags)

        # Extract key phrases from problem description
        if problem_description:
            # Simple keyword extraction - look for important words
            words = problem_description.lower().split()
            # Filter out common words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            important_words = [w for w in words if len(w) > 3 and w not in stop_words]
            keywords.extend(important_words[:10])  # Top 10 words

        # Remove duplicates and return
        return list(set(keywords))[:15]  # Limit to 15 keywords

    def _generate_patterns(self, keywords: List[str]) -> List[str]:
        """Generate regex patterns from keywords"""
        patterns = []
        for keyword in keywords[:10]:  # Limit to 10 patterns
            # Create a simple pattern that matches the keyword with word boundaries
            pattern = rf"\b{re.escape(keyword)}\b"
            patterns.append(pattern)
        return patterns

    def _extract_examples(self, execution_data: Dict[str, Any]) -> List[str]:
        """Extract example usage from execution data"""
        examples = []

        # Get the input prompt as an example
        input_data = execution_data.get("input_data", {})
        prompt = input_data.get("prompt", input_data.get("message", ""))
        if prompt:
            examples.append(prompt[:200])  # Limit length

        return examples[:3]  # Limit to 3 examples

    def _calculate_skill_priority(self, pattern: Dict[str, Any], execution_data: Dict[str, Any]) -> int:
        """Calculate skill priority based on pattern complexity and success"""
        priority = 5  # Base priority

        # Increase priority based on complexity
        complexity = pattern.get("complexity", "low")
        if complexity == "high":
            priority += 3
        elif complexity == "medium":
            priority += 2

        # Increase priority if tools were used
        tools_used = execution_data.get("output_data", {}).get("tool_calls", 0)
        if tools_used > 0:
            priority += min(tools_used, 3)  # Max +3 for tool usage

        return min(priority, 10)  # Cap at 10

    def _create_dynamic_handler(self, skill: Skill, pattern: Dict[str, Any], execution_data: Dict[str, Any]):
        """Create a dynamic handler for the auto-saved skill"""
        async def dynamic_handler(user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
            """Dynamic handler for auto-saved skill"""
            return {
                "mode": "auto_saved_skill",
                "skill_name": skill.name,
                "pattern_id": skill.source_pattern_id,
                "pattern_type": pattern.get("pattern_type", "general"),
                "guidance": f"Applying learned pattern: {skill.display_name}",
                "solution_approach": pattern.get("solution_summary", ""),
                "context_enrichment": {
                    "auto_saved_skill": True,
                    "pattern_based": True,
                    "learned_from_execution": True,
                    "pattern_id": skill.source_pattern_id
                }
            }

        # Register the handler
        self.register_handler(skill.name, dynamic_handler)

    async def update_skill_metrics(self, skill_name: str, success: bool):
        """
        Update skill usage metrics

        Args:
            skill_name: Name of the skill
            success: Whether the skill execution was successful
        """
        skill = self.skills.get(skill_name)
        if not skill:
            return

        # Update usage count
        skill.usage_count += 1

        # Update success rate (exponential moving average)
        alpha = 0.2  # Weight for new observation
        if success:
            skill.success_rate = skill.success_rate * (1 - alpha) + alpha
        else:
            skill.success_rate = skill.success_rate * (1 - alpha)

        # Update timestamp
        skill.updated_at = datetime.utcnow()

        logger.debug(f"Updated metrics for skill {skill_name}: usage={skill.usage_count}, success_rate={skill.success_rate:.2f}")

    def get_auto_saved_skills(self) -> List[Skill]:
        """Get all auto-saved skills"""
        return [s for s in self.skills.values() if s.is_auto_saved]

    def get_skill_stats(self) -> Dict[str, Any]:
        """Get statistics about skills"""
        auto_saved = self.get_auto_saved_skills()
        return {
            "total_skills": len(self.skills),
            "core_skills": len(self.skills) - len(auto_saved),
            "auto_saved_skills": len(auto_saved),
            "categories": self.get_categories(),
            "total_usage": sum(s.usage_count for s in self.skills.values()),
            "average_success_rate": sum(s.success_rate for s in self.skills.values()) / len(self.skills) if self.skills else 0.0
        }


# ============================================================================
# Skill Handler Functions
# ============================================================================

async def handle_sparc_methodology(user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handler for SPARC Development Methodology skill

    Guides systematic feature development through:
    - Specification
    - Pseudocode
    - Architecture
    - Refinement
    - Completion
    """
    return {
        "mode": "sparc_development",
        "phases": ["specification", "pseudocode", "architecture", "refinement", "completion"],
        "current_phase": "specification",
        "requirements": {
            "specification": True,
            "tests": True,
            "documentation": True,
            "systematic_approach": True
        },
        "guidance": "Starting SPARC methodology. First, let's define the specification...",
        "context_enrichment": {
            "development_mode": True,
            "requires_tests": True,
            "systematic_approach": True
        }
    }


async def handle_pair_programming(user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handler for AI Pair Programming skill

    Provides collaborative coding assistance with:
    - Real-time code suggestions
    - Code review
    - Debugging help
    - Best practices guidance
    """
    return {
        "mode": "pair_programming",
        "features": ["code_suggestions", "review", "debugging", "best_practices"],
        "interaction_style": "collaborative",
        "guidance": "I'm ready to pair program with you. Let's work on this together...",
        "context_enrichment": {
            "collaborative_mode": True,
            "requires_code_context": True,
            "interactive": True
        }
    }


async def handle_swarm_orchestration(user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handler for Multi-Agent Swarm Orchestration skill

    Coordinates multiple agents for complex tasks:
    - Task decomposition
    - Agent selection
    - Parallel execution
    - Result aggregation
    """
    return {
        "mode": "swarm_orchestration",
        "orchestration_type": "multi_agent",
        "capabilities": ["task_decomposition", "agent_selection", "parallel_execution", "result_aggregation"],
        "guidance": "Initiating swarm orchestration. Analyzing task for agent distribution...",
        "context_enrichment": {
            "force_multi_agent": True,
            "orchestration_mode": "swarm",
            "requires_task_decomposition": True,
            "parallel_execution": True
        }
    }


async def handle_code_review(user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handler for Comprehensive Code Review skill

    Performs deep code analysis covering:
    - Security vulnerabilities
    - Performance issues
    - Best practices violations
    - Maintainability concerns
    """
    return {
        "mode": "code_review",
        "review_aspects": ["security", "performance", "best_practices", "maintainability", "testing"],
        "depth": "comprehensive",
        "guidance": "Starting comprehensive code review. Analyzing for security, performance, and best practices...",
        "context_enrichment": {
            "analysis_mode": True,
            "deep_inspection": True,
            "requires_code_input": True,
            "review_categories": ["security", "performance", "quality"]
        }
    }


async def handle_performance_optimization(user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handler for Performance Analysis & Optimization skill

    Analyzes and optimizes system performance:
    - Bottleneck identification
    - Performance profiling
    - Optimization recommendations
    - Benchmarking
    """
    return {
        "mode": "performance_optimization",
        "analysis_types": ["bottleneck_detection", "profiling", "benchmarking", "optimization"],
        "focus_areas": ["cpu", "memory", "io", "network", "database"],
        "guidance": "Starting performance analysis. Identifying bottlenecks and optimization opportunities...",
        "context_enrichment": {
            "optimization_mode": True,
            "performance_focus": True,
            "requires_profiling": False,  # Can work without existing profiling data
            "analysis_depth": "deep"
        }
    }


# Global skills service instance
skills_service = SkillsService()

# Register skill handlers
skills_service.register_handler("sparc_methodology", handle_sparc_methodology)
skills_service.register_handler("pair_programming", handle_pair_programming)
skills_service.register_handler("swarm_orchestration", handle_swarm_orchestration)
skills_service.register_handler("code_review", handle_code_review)
skills_service.register_handler("performance_optimization", handle_performance_optimization)

logger.info("✅ All skill handlers registered")

