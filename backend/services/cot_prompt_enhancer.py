"""
Chain-of-Thought (CoT) Prompt Enhancer

Enhances prompts with step-by-step reasoning patterns to improve LLM performance
on complex tasks. Implements various CoT strategies based on task type.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Types of tasks that benefit from different CoT strategies"""
    REASONING = "reasoning"
    ANALYSIS = "analysis"
    PROBLEM_SOLVING = "problem_solving"
    CREATIVE = "creative"
    RESEARCH = "research"
    CODING = "coding"
    PLANNING = "planning"
    DECISION_MAKING = "decision_making"
    GENERAL = "general"


class CoTPromptEnhancer:
    """
    Chain-of-Thought Prompt Enhancement Service
    
    Adds structured reasoning patterns to prompts to improve LLM performance.
    Uses different CoT templates based on task type for optimal results.
    """
    
    # CoT Templates for different task types
    COT_TEMPLATES = {
        TaskType.REASONING: """
Let's approach this step-by-step with careful reasoning:

1. **Understand the Question**: First, identify what is being asked and what information is provided.
2. **Break Down the Problem**: Decompose the question into smaller, manageable parts.
3. **Apply Logic**: Work through each part systematically, applying relevant principles.
4. **Verify Reasoning**: Check each step for logical consistency.
5. **Synthesize Answer**: Combine insights to form a complete, well-reasoned response.

Now, let's work through this:""",
        
        TaskType.ANALYSIS: """
To analyze this thoroughly, I'll follow a structured approach:

**Step 1: Observation** - What are the key facts and data points?
**Step 2: Pattern Recognition** - What patterns, trends, or relationships exist?
**Step 3: Context Evaluation** - What is the broader context and significance?
**Step 4: Critical Assessment** - What are the strengths, weaknesses, and implications?
**Step 5: Conclusion** - What insights can we draw from this analysis?

Let me analyze this systematically:""",
        
        TaskType.PROBLEM_SOLVING: """
Let's solve this problem methodically:

**Phase 1: Problem Understanding**
- What exactly is the problem?
- What are the constraints and requirements?
- What is the desired outcome?

**Phase 2: Solution Planning**
- What approaches could work?
- What are the pros and cons of each?
- Which approach is most promising?

**Phase 3: Solution Execution**
- Implement the chosen approach step-by-step
- Monitor progress and adjust as needed

**Phase 4: Verification**
- Does the solution meet all requirements?
- Are there edge cases to consider?
- Can we optimize further?

Let's work through this:""",
        
        TaskType.CREATIVE: """
Let's approach this creatively with structured ideation:

1. **Explore the Space**: What are the possibilities and constraints?
2. **Generate Ideas**: Brainstorm multiple creative approaches without judgment.
3. **Evaluate Options**: Consider feasibility, impact, and originality.
4. **Refine Concept**: Develop the most promising idea with details.
5. **Present Solution**: Articulate the creative solution clearly.

Let me think creatively about this:""",
        
        TaskType.RESEARCH: """
Let's conduct thorough research using a systematic approach:

**Research Phase 1: Define Scope**
- What specific information do we need?
- What are the key questions to answer?

**Research Phase 2: Information Gathering**
- What sources are most relevant?
- What data points are critical?

**Research Phase 3: Analysis & Synthesis**
- How does the information connect?
- What patterns emerge?

**Research Phase 4: Conclusions**
- What are the key findings?
- What recommendations follow?

Let me research this systematically:""",
        
        TaskType.CODING: """
Let's approach this coding task with engineering discipline:

**Step 1: Requirements Analysis**
- What functionality is needed?
- What are the inputs, outputs, and constraints?

**Step 2: Design**
- What's the overall architecture?
- What data structures and algorithms are appropriate?

**Step 3: Implementation**
- Write clean, maintainable code
- Follow best practices and conventions

**Step 4: Testing & Validation**
- Does it handle edge cases?
- Is it performant and correct?

Let me code this step-by-step:""",
        
        TaskType.PLANNING: """
Let's create a comprehensive plan:

**Planning Step 1: Goal Definition**
- What are we trying to achieve?
- What does success look like?

**Planning Step 2: Current State Assessment**
- Where are we now?
- What resources are available?

**Planning Step 3: Strategy Development**
- What steps will get us to the goal?
- What's the optimal sequence?

**Planning Step 4: Risk & Contingency**
- What could go wrong?
- What are the backup plans?

**Planning Step 5: Execution Roadmap**
- What are the milestones?
- How do we track progress?

Let me plan this systematically:""",
        
        TaskType.DECISION_MAKING: """
Let's make this decision using a structured framework:

**Decision Framework:**

1. **Define the Decision**: What exactly needs to be decided?
2. **Identify Options**: What are all possible choices?
3. **Establish Criteria**: What factors matter most?
4. **Evaluate Each Option**: 
   - Pros and cons
   - Alignment with criteria
   - Risks and benefits
5. **Make Recommendation**: Which option is best and why?
6. **Consider Implementation**: How would we execute this decision?

Let me work through this decision:""",
        
        TaskType.GENERAL: """
Let's think through this step-by-step:

1. First, let me understand what's being asked
2. Then, I'll identify the key components
3. Next, I'll work through each part systematically
4. Finally, I'll synthesize a complete answer

Here's my reasoning:"""
    }
    
    # Keywords that indicate different task types
    TASK_TYPE_KEYWORDS = {
        TaskType.REASONING: ["why", "explain", "reason", "justify", "logic", "deduce", "infer"],
        TaskType.ANALYSIS: ["analyze", "examine", "evaluate", "assess", "compare", "contrast", "review"],
        TaskType.PROBLEM_SOLVING: ["solve", "fix", "resolve", "troubleshoot", "debug", "issue", "problem"],
        TaskType.CREATIVE: ["create", "design", "imagine", "brainstorm", "innovate", "invent", "generate"],
        TaskType.RESEARCH: ["research", "investigate", "find", "discover", "explore", "study", "learn about"],
        TaskType.CODING: ["code", "implement", "program", "develop", "write", "function", "class", "algorithm"],
        TaskType.PLANNING: ["plan", "strategy", "roadmap", "schedule", "organize", "coordinate", "timeline"],
        TaskType.DECISION_MAKING: ["decide", "choose", "select", "recommend", "should", "which", "best option"],
    }
    
    def __init__(self):
        """Initialize the CoT Prompt Enhancer"""
        logger.info("🧠 CoT Prompt Enhancer initialized")
    
    def detect_task_type(self, prompt: str) -> TaskType:
        """
        Detect the task type from the prompt content
        
        Args:
            prompt: The user's prompt
            
        Returns:
            TaskType enum value
        """
        prompt_lower = prompt.lower()
        
        # Count keyword matches for each task type
        task_scores = {}
        for task_type, keywords in self.TASK_TYPE_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in prompt_lower)
            if score > 0:
                task_scores[task_type] = score
        
        # Return task type with highest score, or GENERAL if no matches
        if task_scores:
            best_task_type = max(task_scores.items(), key=lambda x: x[1])[0]
            logger.info(f"🎯 Detected task type: {best_task_type.value} (score: {task_scores[best_task_type]})")
            return best_task_type
        
        logger.info("🎯 No specific task type detected, using GENERAL")
        return TaskType.GENERAL
    
    def enhance_with_cot(
        self, 
        prompt: str, 
        task_type: Optional[TaskType] = None,
        custom_cot_template: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enhance a prompt with Chain-of-Thought reasoning
        
        Args:
            prompt: The original prompt to enhance
            task_type: Optional explicit task type (auto-detected if not provided)
            custom_cot_template: Optional custom CoT template to use
            
        Returns:
            Dictionary with enhanced_prompt, original_prompt, task_type, and metadata
        """
        try:
            # Auto-detect task type if not provided
            if task_type is None:
                task_type = self.detect_task_type(prompt)
            
            # Get CoT template
            if custom_cot_template:
                cot_template = custom_cot_template
            else:
                cot_template = self.COT_TEMPLATES.get(task_type, self.COT_TEMPLATES[TaskType.GENERAL])
            
            # Combine prompt with CoT template
            enhanced_prompt = f"""{cot_template}

{prompt}"""
            
            logger.info(f"✅ Enhanced prompt with {task_type.value} CoT template")
            
            return {
                "success": True,
                "enhanced_prompt": enhanced_prompt,
                "original_prompt": prompt,
                "task_type": task_type.value,
                "cot_template_used": task_type.value,
                "enhancement_applied": True,
                "metadata": {
                    "original_length": len(prompt),
                    "enhanced_length": len(enhanced_prompt),
                    "cot_overhead": len(cot_template),
                    "enhancement_ratio": len(enhanced_prompt) / len(prompt) if len(prompt) > 0 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to enhance prompt with CoT: {e}")
            return {
                "success": False,
                "enhanced_prompt": prompt,  # Return original on failure
                "original_prompt": prompt,
                "task_type": "unknown",
                "error": str(e),
                "enhancement_applied": False
            }
    
    def get_available_task_types(self) -> List[str]:
        """Get list of available task types"""
        return [task_type.value for task_type in TaskType]
    
    def get_cot_template(self, task_type: TaskType) -> str:
        """Get the CoT template for a specific task type"""
        return self.COT_TEMPLATES.get(task_type, self.COT_TEMPLATES[TaskType.GENERAL])


