"""
Unit Tests for Chain-of-Thought Prompt Enhancer

Tests the CoT enhancement functionality including:
- Task type detection
- CoT template application
- Enhancement for different task types
- Integration with prompt orchestrator
"""

import pytest
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.cot_prompt_enhancer import CoTPromptEnhancer, TaskType


class TestCoTPromptEnhancer:
    """Test suite for CoTPromptEnhancer"""
    
    @pytest.fixture
    def enhancer(self):
        """Create a CoTPromptEnhancer instance for testing"""
        return CoTPromptEnhancer()
    
    # Task Type Detection Tests
    
    def test_detect_reasoning_task(self, enhancer):
        """Test detection of reasoning tasks"""
        prompt = "Why does the sky appear blue? Explain the reasoning behind this phenomenon."
        task_type = enhancer.detect_task_type(prompt)
        assert task_type == TaskType.REASONING
    
    def test_detect_analysis_task(self, enhancer):
        """Test detection of analysis tasks"""
        prompt = "Analyze the performance metrics of our application and compare them to industry standards."
        task_type = enhancer.detect_task_type(prompt)
        assert task_type == TaskType.ANALYSIS
    
    def test_detect_problem_solving_task(self, enhancer):
        """Test detection of problem-solving tasks"""
        prompt = "How can I solve the issue with slow database queries in my application?"
        task_type = enhancer.detect_task_type(prompt)
        assert task_type == TaskType.PROBLEM_SOLVING
    
    def test_detect_creative_task(self, enhancer):
        """Test detection of creative tasks"""
        prompt = "Create an innovative design for a mobile app that helps people track their fitness goals."
        task_type = enhancer.detect_task_type(prompt)
        assert task_type == TaskType.CREATIVE
    
    def test_detect_research_task(self, enhancer):
        """Test detection of research tasks"""
        prompt = "Research the latest trends in artificial intelligence and machine learning."
        task_type = enhancer.detect_task_type(prompt)
        assert task_type == TaskType.RESEARCH
    
    def test_detect_coding_task(self, enhancer):
        """Test detection of coding tasks"""
        prompt = "Write a Python function to implement binary search on a sorted array."
        task_type = enhancer.detect_task_type(prompt)
        assert task_type == TaskType.CODING
    
    def test_detect_planning_task(self, enhancer):
        """Test detection of planning tasks"""
        prompt = "Plan a comprehensive strategy for launching our new product in the market."
        task_type = enhancer.detect_task_type(prompt)
        assert task_type == TaskType.PLANNING
    
    def test_detect_decision_making_task(self, enhancer):
        """Test detection of decision-making tasks"""
        prompt = "Which cloud provider should we choose for our infrastructure: AWS, Azure, or GCP?"
        task_type = enhancer.detect_task_type(prompt)
        assert task_type == TaskType.DECISION_MAKING
    
    def test_detect_general_task(self, enhancer):
        """Test detection of general tasks (no specific keywords)"""
        prompt = "Tell me about the weather today."
        task_type = enhancer.detect_task_type(prompt)
        assert task_type == TaskType.GENERAL
    
    # CoT Enhancement Tests
    
    def test_enhance_with_auto_detection(self, enhancer):
        """Test enhancement with automatic task type detection"""
        prompt = "How can I optimize my Python code for better performance?"
        result = enhancer.enhance_with_cot(prompt)
        
        assert result["success"] is True
        assert result["enhancement_applied"] is True
        assert len(result["enhanced_prompt"]) > len(prompt)
        assert prompt in result["enhanced_prompt"]
        assert result["task_type"] in [t.value for t in TaskType]
    
    def test_enhance_with_explicit_task_type(self, enhancer):
        """Test enhancement with explicitly specified task type"""
        prompt = "What is the best approach?"
        result = enhancer.enhance_with_cot(prompt, task_type=TaskType.PROBLEM_SOLVING)
        
        assert result["success"] is True
        assert result["task_type"] == TaskType.PROBLEM_SOLVING.value
        assert result["cot_template_used"] == TaskType.PROBLEM_SOLVING.value
        assert "Phase 1: Problem Understanding" in result["enhanced_prompt"]
    
    def test_enhance_reasoning_task(self, enhancer):
        """Test enhancement for reasoning tasks"""
        prompt = "Why is the Earth round?"
        result = enhancer.enhance_with_cot(prompt, task_type=TaskType.REASONING)
        
        assert result["success"] is True
        assert "step-by-step" in result["enhanced_prompt"].lower()
        assert "reasoning" in result["enhanced_prompt"].lower()
    
    def test_enhance_analysis_task(self, enhancer):
        """Test enhancement for analysis tasks"""
        prompt = "Analyze the market trends."
        result = enhancer.enhance_with_cot(prompt, task_type=TaskType.ANALYSIS)
        
        assert result["success"] is True
        assert "analyze" in result["enhanced_prompt"].lower()
        assert "observation" in result["enhanced_prompt"].lower()
    
    def test_enhance_coding_task(self, enhancer):
        """Test enhancement for coding tasks"""
        prompt = "Implement a sorting algorithm."
        result = enhancer.enhance_with_cot(prompt, task_type=TaskType.CODING)
        
        assert result["success"] is True
        assert "requirements" in result["enhanced_prompt"].lower()
        assert "implementation" in result["enhanced_prompt"].lower()
        assert "testing" in result["enhanced_prompt"].lower()
    
    def test_enhance_with_custom_template(self, enhancer):
        """Test enhancement with custom CoT template"""
        prompt = "Solve this problem."
        custom_template = "Custom Step 1: Do this\nCustom Step 2: Do that"
        result = enhancer.enhance_with_cot(prompt, custom_cot_template=custom_template)
        
        assert result["success"] is True
        assert "Custom Step 1" in result["enhanced_prompt"]
        assert "Custom Step 2" in result["enhanced_prompt"]
    
    # Metadata Tests
    
    def test_enhancement_metadata(self, enhancer):
        """Test that enhancement includes proper metadata"""
        prompt = "Test prompt"
        result = enhancer.enhance_with_cot(prompt)
        
        assert "metadata" in result
        assert "original_length" in result["metadata"]
        assert "enhanced_length" in result["metadata"]
        assert "cot_overhead" in result["metadata"]
        assert "enhancement_ratio" in result["metadata"]
        
        assert result["metadata"]["original_length"] == len(prompt)
        assert result["metadata"]["enhanced_length"] == len(result["enhanced_prompt"])
        assert result["metadata"]["enhancement_ratio"] > 1.0  # Enhanced should be longer
    
    def test_original_prompt_preserved(self, enhancer):
        """Test that original prompt is preserved in result"""
        prompt = "Original test prompt"
        result = enhancer.enhance_with_cot(prompt)
        
        assert result["original_prompt"] == prompt
        assert prompt in result["enhanced_prompt"]
    
    # Utility Methods Tests
    
    def test_get_available_task_types(self, enhancer):
        """Test getting list of available task types"""
        task_types = enhancer.get_available_task_types()
        
        assert isinstance(task_types, list)
        assert len(task_types) > 0
        assert "reasoning" in task_types
        assert "analysis" in task_types
        assert "problem_solving" in task_types
        assert "coding" in task_types
    
    def test_get_cot_template(self, enhancer):
        """Test getting CoT template for specific task type"""
        template = enhancer.get_cot_template(TaskType.REASONING)
        
        assert isinstance(template, str)
        assert len(template) > 0
        assert "step" in template.lower()
    
    def test_get_all_templates(self, enhancer):
        """Test that all task types have templates"""
        for task_type in TaskType:
            template = enhancer.get_cot_template(task_type)
            assert isinstance(template, str)
            assert len(template) > 0
    
    # Edge Cases
    
    def test_empty_prompt(self, enhancer):
        """Test enhancement with empty prompt"""
        prompt = ""
        result = enhancer.enhance_with_cot(prompt)
        
        # Should still succeed but with minimal enhancement
        assert result["success"] is True
        assert len(result["enhanced_prompt"]) > 0
    
    def test_very_long_prompt(self, enhancer):
        """Test enhancement with very long prompt"""
        prompt = "Test prompt. " * 1000  # Very long prompt
        result = enhancer.enhance_with_cot(prompt)
        
        assert result["success"] is True
        assert prompt in result["enhanced_prompt"]
    
    def test_special_characters_in_prompt(self, enhancer):
        """Test enhancement with special characters"""
        prompt = "Test with special chars: @#$%^&*(){}[]|\\<>?/~`"
        result = enhancer.enhance_with_cot(prompt)
        
        assert result["success"] is True
        assert prompt in result["enhanced_prompt"]
    
    def test_unicode_in_prompt(self, enhancer):
        """Test enhancement with Unicode characters"""
        prompt = "Test with Unicode: 你好 مرحبا Привет 🚀"
        result = enhancer.enhance_with_cot(prompt)
        
        assert result["success"] is True
        assert prompt in result["enhanced_prompt"]
    
    # Multiple Task Type Keywords
    
    def test_multiple_task_type_keywords(self, enhancer):
        """Test detection when prompt has multiple task type keywords"""
        prompt = "Analyze this problem and solve it by writing code to implement a solution."
        # Should detect the most prominent task type
        task_type = enhancer.detect_task_type(prompt)
        
        # Should be one of the mentioned types
        assert task_type in [TaskType.ANALYSIS, TaskType.PROBLEM_SOLVING, TaskType.CODING]
    
    # Integration Tests
    
    def test_enhancement_preserves_prompt_intent(self, enhancer):
        """Test that enhancement doesn't change the original prompt's intent"""
        prompt = "Calculate the sum of 1 + 2 + 3"
        result = enhancer.enhance_with_cot(prompt)
        
        # Original prompt should be intact in the enhanced version
        assert prompt in result["enhanced_prompt"]
        # Should not have modified the actual question
        assert "1 + 2 + 3" in result["enhanced_prompt"]
    
    def test_enhancement_adds_structure(self, enhancer):
        """Test that enhancement adds structured reasoning"""
        prompt = "How do I fix this bug?"
        result = enhancer.enhance_with_cot(prompt)
        
        # Should add structured steps
        enhanced = result["enhanced_prompt"]
        assert any(word in enhanced.lower() for word in ["step", "phase", "approach"])
    
    def test_consistent_enhancement(self, enhancer):
        """Test that same prompt gets consistent enhancement"""
        prompt = "Test consistency"
        result1 = enhancer.enhance_with_cot(prompt, task_type=TaskType.GENERAL)
        result2 = enhancer.enhance_with_cot(prompt, task_type=TaskType.GENERAL)
        
        # Should get same enhancement for same inputs
        assert result1["enhanced_prompt"] == result2["enhanced_prompt"]
        assert result1["task_type"] == result2["task_type"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

