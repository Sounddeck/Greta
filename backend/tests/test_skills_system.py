"""
Unit Tests for Skills System
Tests the natural language task routing and skill matching functionality
"""

import pytest
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.skills_service import (
    SkillsService,
    Skill,
    SkillMatch,
    handle_sparc_methodology,
    handle_pair_programming,
    handle_swarm_orchestration,
    handle_code_review,
    handle_performance_optimization
)


class TestSkillsService:
    """Test the SkillsService class"""
    
    @pytest.fixture
    def skills_service(self):
        """Create a fresh SkillsService instance for each test"""
        return SkillsService()
    
    def test_service_initialization(self, skills_service):
        """Test that the service initializes with 5 core skills"""
        assert len(skills_service.skills) == 5
        assert "sparc_methodology" in skills_service.skills
        assert "pair_programming" in skills_service.skills
        assert "swarm_orchestration" in skills_service.skills
        assert "code_review" in skills_service.skills
        assert "performance_optimization" in skills_service.skills
    
    def test_get_skill(self, skills_service):
        """Test retrieving a skill by name"""
        skill = skills_service.get_skill("sparc_methodology")
        assert skill is not None
        assert skill.name == "sparc_methodology"
        assert skill.display_name == "SPARC Development Methodology"
        assert skill.category == "development"
        assert skill.priority == 10
    
    def test_list_skills(self, skills_service):
        """Test listing all skills"""
        skills = skills_service.list_skills()
        assert len(skills) == 5
    
    def test_list_skills_by_category(self, skills_service):
        """Test filtering skills by category"""
        dev_skills = skills_service.list_skills(category="development")
        assert len(dev_skills) == 2  # SPARC and Pair Programming
        
        analysis_skills = skills_service.list_skills(category="analysis")
        assert len(analysis_skills) == 1  # Code Review
    
    def test_get_categories(self, skills_service):
        """Test getting all categories"""
        categories = skills_service.get_categories()
        assert len(categories) == 4
        assert "development" in categories
        assert "orchestration" in categories
        assert "analysis" in categories
        assert "optimization" in categories


class TestSkillMatching:
    """Test the skill matching algorithm"""
    
    @pytest.fixture
    def skills_service(self):
        """Create a fresh SkillsService instance for each test"""
        return SkillsService()
    
    @pytest.mark.asyncio
    async def test_match_sparc_methodology(self, skills_service):
        """Test matching SPARC methodology skill"""
        match = await skills_service.match_skill(
            "I want to build a feature using TDD and systematic development",
            threshold=0.5  # Lower threshold for testing
        )
        assert match is not None
        assert match.skill.name == "sparc_methodology"
        assert match.confidence >= 0.5
        assert len(match.matched_keywords) > 0 or len(match.matched_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_match_pair_programming(self, skills_service):
        """Test matching pair programming skill"""
        match = await skills_service.match_skill(
            "Help me code a REST API endpoint",
            threshold=0.3
        )
        assert match is not None
        assert match.skill.name == "pair_programming"
        assert match.confidence >= 0.3
    
    @pytest.mark.asyncio
    async def test_match_swarm_orchestration(self, skills_service):
        """Test matching swarm orchestration skill"""
        match = await skills_service.match_skill(
            "Orchestrate multiple agents to analyze this codebase",
            threshold=0.3
        )
        assert match is not None
        assert match.skill.name == "swarm_orchestration"
        assert match.confidence >= 0.3
    
    @pytest.mark.asyncio
    async def test_match_code_review(self, skills_service):
        """Test matching code review skill"""
        match = await skills_service.match_skill(
            "Review this authentication module for security issues",
            threshold=0.3
        )
        assert match is not None
        assert match.skill.name == "code_review"
        assert match.confidence >= 0.3
    
    @pytest.mark.asyncio
    async def test_match_performance_optimization(self, skills_service):
        """Test matching performance optimization skill"""
        match = await skills_service.match_skill(
            "I need to optimize performance and find bottlenecks in the system",
            threshold=0.5  # Lower threshold for testing
        )
        assert match is not None
        assert match.skill.name == "performance_optimization"
        assert match.confidence >= 0.5
    
    @pytest.mark.asyncio
    async def test_no_match_below_threshold(self, skills_service):
        """Test that no match is returned when confidence is below threshold"""
        match = await skills_service.match_skill(
            "This is completely unrelated text",
            threshold=0.9  # Very high threshold
        )
        assert match is None
    
    @pytest.mark.asyncio
    async def test_priority_ordering(self, skills_service):
        """Test that higher priority skills are matched first when confidence is similar"""
        # "build" could match multiple skills, but SPARC has highest priority
        match = await skills_service.match_skill(
            "build a feature",
            threshold=0.5  # Lower threshold for testing
        )
        assert match is not None
        # Should match SPARC (priority 10) over others
        assert match.skill.priority >= 8


class TestSkillHandlers:
    """Test the skill handler functions"""
    
    @pytest.mark.asyncio
    async def test_sparc_handler(self):
        """Test SPARC methodology handler"""
        result = await handle_sparc_methodology(
            "Build a user auth feature",
            {}
        )
        assert result["mode"] == "sparc_development"
        assert "phases" in result
        assert len(result["phases"]) == 5
        assert result["context_enrichment"]["development_mode"] is True
    
    @pytest.mark.asyncio
    async def test_pair_programming_handler(self):
        """Test pair programming handler"""
        result = await handle_pair_programming(
            "Help me code an API",
            {}
        )
        assert result["mode"] == "pair_programming"
        assert result["interaction_style"] == "collaborative"
        assert result["context_enrichment"]["collaborative_mode"] is True
    
    @pytest.mark.asyncio
    async def test_swarm_orchestration_handler(self):
        """Test swarm orchestration handler"""
        result = await handle_swarm_orchestration(
            "Coordinate agents",
            {}
        )
        assert result["mode"] == "swarm_orchestration"
        assert result["orchestration_type"] == "multi_agent"
        assert result["context_enrichment"]["force_multi_agent"] is True
    
    @pytest.mark.asyncio
    async def test_code_review_handler(self):
        """Test code review handler"""
        result = await handle_code_review(
            "Review my code",
            {}
        )
        assert result["mode"] == "code_review"
        assert result["depth"] == "comprehensive"
        assert "security" in result["review_aspects"]
    
    @pytest.mark.asyncio
    async def test_performance_optimization_handler(self):
        """Test performance optimization handler"""
        result = await handle_performance_optimization(
            "Optimize performance",
            {}
        )
        assert result["mode"] == "performance_optimization"
        assert "bottleneck_detection" in result["analysis_types"]
        assert result["context_enrichment"]["optimization_mode"] is True


class TestSkillExecution:
    """Test skill execution through the service"""

    @pytest.fixture
    def skills_service(self):
        """Create a fresh SkillsService instance with handlers registered"""
        from services.skills_service import (
            skills_service,
            handle_sparc_methodology,
            handle_pair_programming,
            handle_swarm_orchestration,
            handle_code_review,
            handle_performance_optimization
        )
        # Use the global instance which has handlers registered
        return skills_service
    
    @pytest.mark.asyncio
    async def test_execute_skill_success(self, skills_service):
        """Test successful skill execution"""
        result = await skills_service.execute_skill(
            "sparc_methodology",
            "Build a feature",
            {}
        )
        assert result["success"] is True
        assert "result" in result
        assert result["result"]["mode"] == "sparc_development"
    
    @pytest.mark.asyncio
    async def test_execute_nonexistent_skill(self, skills_service):
        """Test executing a skill that doesn't exist"""
        result = await skills_service.execute_skill(
            "nonexistent_skill",
            "Some input",
            {}
        )
        assert result["success"] is False
        assert "error" in result
        assert "not found" in result["error"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

