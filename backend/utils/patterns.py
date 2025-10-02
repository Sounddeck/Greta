"""
GRETA PAI - Pattern Commands System
Core PAI Feature: 60+ CLI commands for AI-augmented tasks
Inspired by Fabric's pattern system but integrated with PAI architecture
"""
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import json
import asyncio
import logging
from string import Template
import re

from utils.error_handling import GretaException, handle_errors
from utils.hooks import hook_manager, HookContext, execute_hooks
from utils.ufc_context import ufc_manager
from utils.ai_providers import ai_orchestrator  # We'll implement this next

logger = logging.getLogger(__name__)


class PatternError(GretaException):
    """Pattern system errors"""


class PatternVariable:
    """Represents a pattern variable with validation and defaults"""

    def __init__(self, name: str, description: str = "",
                 default: Any = None, required: bool = False,
                 validation: Optional[str] = None):
        self.name = name
        self.description = description
        self.default = default
        self.required = required
        self.validation = validation  # Regex pattern for validation

    def validate(self, value: Any) -> Any:
        """Validate and convert variable value"""
        if value is None:
            if self.required:
                raise PatternError(f"Required variable '{self.name}' not provided")
            return self.default

        # String validation
        str_value = str(value)
        if self.validation and not re.match(self.validation, str_value):
            raise PatternError(f"Variable '{self.name}' failed validation")

        return str_value


class PAIPattern:
    """Represents a PAI pattern (like Fabric patterns)"""

    def __init__(self, name: str, category: str, system_prompt: str,
                 user_template: Optional[str] = None,
                 variables: Optional[Dict[str, Dict]] = None,
                 model_preference: Optional[str] = None,
                 output_format: Optional[str] = None):
        self.name = name
        self.category = category
        self.system_prompt = system_prompt
        self.user_template = user_template or "${input}"
        self.variables = {
            name: PatternVariable(name, **config)
            for name, config in (variables or {}).items()
        }
        self.model_preference = model_preference  # 'claude', 'gpt', 'gemini', 'ollama'
        self.output_format = output_format  # 'markdown', 'json', 'text'
        self.created_at = datetime.utcnow()
        self.last_used = None
        self.usage_count = 0

    def validate_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and prepare inputs for pattern execution"""
        validated = {}

        # Validate required inputs
        for var_name, var in self.variables.items():
            if var_name in inputs:
                validated[var_name] = var.validate(inputs[var_name])
            else:
                validated[var_name] = var.validate(None)

        return validated

    def render_user_prompt(self, inputs: Dict[str, Any]) -> str:
        """Render the user template with provided inputs"""
        template = Template(self.user_template)
        return template.safe_substitute(inputs)

    def to_dict(self) -> Dict[str, Any]:
        """Export pattern as dictionary"""
        return {
            "name": self.name,
            "category": self.category,
            "system_prompt": self.system_prompt,
            "user_template": self.user_template,
            "variables": {
                name: {
                    "description": var.description,
                    "default": var.default,
                    "required": var.required,
                    "validation": var.validation
                }
                for name, var in self.variables.items()
            },
            "model_preference": self.model_preference,
            "output_format": self.output_format,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "usage_count": self.usage_count
        }


class PAICommandRegistry:
    """
    Registry for PAI commands - 60+ AI-augmented commands
    Similar to Fabric's pattern system but integrated with PAI architecture
    """

    # Core PAI command definitions
    PROFESSIONAL_COMMANDS = {
        'write-blog': PAIPattern(
            name='write-blog',
            category='writing',
            system_prompt='''You are a professional content creator specializing in technical blog posts.
Write compelling, SEO-optimized blog posts that educate and engage readers.
Use clear language, practical examples, and actionable insights.'',
            user_template='''Write a blog post titled "${title}" about: ${topic}

Key requirements:
- Length: ${word_count} words
- Style: ${tone}
- Audience: ${audience}
- Include code examples: ${include_code}

Focus areas: ${key_points}
Make it engaging and professionally written.''',
            variables={
                'title': {'description': 'Blog post title', 'required': True},
                'topic': {'description': 'Main topic to cover', 'required': True},
                'word_count': {'description': 'Target word count', 'default': '1500'},
                'tone': {'description': 'Writing tone', 'default': 'professional'},
                'audience': {'description': 'Target audience', 'default': 'technical professionals'},
                'include_code': {'description': 'Include code examples', 'default': 'yes'},
                'key_points': {'description': 'Key points to cover', 'default': ''}
            },
            model_preference='claude',
            output_format='markdown'
        ),

        'analyze-code': PAIPattern(
            name='analyze-code',
            category='engineering',
            system_prompt='''You are an expert code reviewer and software engineer.
Analyze code for bugs, security issues, performance problems, and best practices.
Provide actionable feedback with specific suggestions for improvements.''',
            user_template='''Analyze the following ${language} code:

```${language}
${code}
```

Analysis requirements:
- Identify bugs or potential issues: ${check_bugs}
- Security vulnerabilities: ${check_security}
- Performance optimizations: ${check_performance}
- Code quality and best practices: ${check_quality}

Provide specific recommendations with code examples where relevant.''',
            variables={
                'language': {'description': 'Programming language', 'required': True},
                'code': {'description': 'Code to analyze', 'required': True},
                'check_bugs': {'description': 'Check for bugs', 'default': 'yes'},
                'check_security': {'description': 'Check security', 'default': 'yes'},
                'check_performance': {'description': 'Check performance', 'default': 'yes'},
                'check_quality': {'description': 'Check code quality', 'default': 'yes'}
            },
            model_preference='claude',
            output_format='markdown'
        ),

        'get-newsletter-stats': PAIPattern(
            name='get-newsletter-stats',
            category='business',
            system_prompt='''You are a newsletter analytics expert.
Analyze newsletter metrics and provide insights on subscriber engagement, content performance, and growth strategies.''',
            user_template='''Analyze these newsletter metrics:

Current Statistics:
- Subscribers: ${subscribers}
- Open Rate: ${open_rate}
- Click Rate: ${click_rate}
- Growth Rate: ${growth_rate}

Recent Performance:
${performance_data}

Top Performing Content:
${top_content}

Provide insights and recommendations for:
1. Content strategy improvements
2. Subscriber growth opportunities
3. Engagement optimization
4. Platform or technology recommendations''',
            variables={
                'subscribers': {'description': 'Current subscriber count', 'required': True},
                'open_rate': {'description': 'Average open rate', 'required': True},
                'click_rate': {'description': 'Average click rate', 'required': True},
                'growth_rate': {'description': 'Monthly growth rate', 'required': True},
                'performance_data': {'description': 'Recent performance data', 'default': ''},
                'top_content': {'description': 'Top performing content', 'default': ''}
            },
            model_preference='gpt',
            output_format='markdown'
        ),

        'create-consulting-document': PAIPattern(
            name='create-consulting-document',
            category='business',
            system_prompt='''You are a senior management consultant.
Create professional consulting documents including proposals, reports, and deliverables.
Use structured, business-appropriate language and comprehensive analysis.''',
            user_template='''Create a ${document_type} for: ${client}

Context:
- Industry: ${industry}
- Scope: ${scope}
- Timeline: ${timeline}
- Budget: ${budget}

Requirements:
${requirements}

Deliverables to include:
${deliverables}

Make this professional, comprehensive, and immediately actionable.''',
            variables={
                'document_type': {'description': 'Type of consulting document', 'required': True, 'validation': r'^(proposal|report|strategy|assessment)$'},
                'client': {'description': 'Client name/company', 'required': True},
                'industry': {'description': 'Client industry', 'required': True},
                'scope': {'description': 'Project scope', 'required': True},
                'timeline': {'description': 'Project timeline', 'required': True},
                'budget': {'description': 'Project budget', 'default': 'TBD'},
                'requirements': {'description': 'Specific requirements', 'default': ''},
                'deliverables': {'description': 'Expected deliverables', 'default': ''}
            },
            model_preference='claude',
            output_format='markdown'
        ),

        'design-review': PAIPattern(
            name='design-review',
            category='engineering',
            system_prompt='''You are a senior software architect and system designer.
Review system designs, architecture decisions, and technical specifications.
Provide thorough analysis and constructive feedback.''',
            user_template='''Review this ${design_type}:

Design Details:
${design_description}

Architecture Decisions:
${architecture}

Constraints and Requirements:
${constraints}

Key Components:
${components}

Provide comprehensive feedback on:
1. Architecture soundness and scalability
2. Security considerations
3. Performance implications
4. Maintainability and extensibility
5. Technology choices and alternatives

Include specific recommendations and potential improvements.''',
            variables={
                'design_type': {'description': 'Type of design to review', 'required': True},
                'design_description': {'description': 'Design details', 'required': True},
                'architecture': {'description': 'Architecture decisions', 'default': ''},
                'constraints': {'description': 'Constraints and requirements', 'default': ''},
                'components': {'description': 'Key components', 'default': ''}
            },
            model_preference='claude',
            output_format='markdown'
        )
    }

    PERSONAL_COMMANDS = {
        'answer-finance-question': PAIPattern(
            name='answer-finance-question',
            category='personal',
            system_prompt='''You are a personal finance advisor.
Answer questions about budgeting, investing, debt management, and financial planning.
Provide practical, actionable advice based on sound financial principles.''',
            user_template='''Answer this personal finance question: "${question}"

Context:
- Current situation: ${situation}
- Goals: ${goals}
- Time horizon: ${time_horizon}
- Risk tolerance: ${risk_tolerance}

Provide specific, actionable advice with step-by-step recommendations.''',
            variables={
                'question': {'description': 'Finance question to answer', 'required': True},
                'situation': {'description': 'Current financial situation', 'default': ''},
                'goals': {'description': 'Financial goals', 'default': ''},
                'time_horizon': {'description': 'Investment time horizon', 'default': ''},
                'risk_tolerance': {'description': 'Risk tolerance level', 'default': ''}
            },
            model_preference='gpt',
            output_format='markdown'
        ),

        'get-life-log': PAIPattern(
            name='get-life-log',
            category='personal',
            system_prompt='''You are a life documentation analyst.
Search through personal records, journals, and life logs to find relevant information and insights.''',
            user_template='''Search and analyze life log data for: "${query}"

Search Parameters:
- Time period: ${time_period}
- Categories: ${categories}
- Keywords: ${keywords}
- Sources: ${sources}

Provide:
- Relevant entries and summaries
- Patterns and insights
- Actionable conclusions
- Recommendations for future logging''',
            variables={
                'query': {'description': 'What to search for in life logs', 'required': True},
                'time_period': {'description': 'Time period to search', 'default': 'last 3 months'},
                'categories': {'description': 'Categories to search', 'default': 'all'},
                'keywords': {'description': 'Additional keywords', 'default': ''},
                'sources': {'description': 'Data sources to search', 'default': 'all'}
            },
            model_preference='claude',
            output_format='markdown'
        ),

        'track-health-metrics': PAIPattern(
            name='track-health-metrics',
            category='personal',
            system_prompt='''You are a health and wellness coach.
Analyze health metrics, provide insights, and give personalized recommendations for health improvement.''',
            user_template='''Analyze these health metrics:

Current Data:
- Activity level: ${activity}
- Sleep quality: ${sleep}
- Diet: ${diet}
- Stress levels: ${stress}
- Vital signs: ${vitals}

Goals:
${goals}

Concerns:
${concerns}

Provide:
1. Overall health assessment
2. Specific recommendations
3. Warning signs to watch for
4. Lifestyle improvement suggestions
5. Progress tracking recommendations''',
            variables={
                'activity': {'description': 'Activity/exercise data', 'required': True},
                'sleep': {'description': 'Sleep quality metrics', 'required': True},
                'diet': {'description': 'Diet and nutrition info', 'required': True},
                'stress': {'description': 'Stress levels', 'required': True},
                'vitals': {'description': 'Vital signs data', 'default': ''},
                'goals': {'description': 'Health goals', 'default': ''},
                'concerns': {'description': 'Current health concerns', 'default': ''}
            },
            model_preference='claude',
            output_format='markdown'
        ),

        'capture-learning': PAIPattern(
            name='capture-learning',
            category='personal',
            system_prompt='''You are a learning and knowledge management specialist.
Help individuals capture, organize, and extract value from their learning experiences.''',
            user_template='''Process this learning content:

Topic: ${topic}
Type: ${content_type}
Source: ${source}

Content Summary:
${content}

Key Learnings:
${learnings}

Applications:
${applications}

Structure this learning for:
1. Knowledge retention and recall
2. Future application
3. Teaching others
4. Building on this foundation
5. Connection to other knowledge areas''',
            variables={
                'topic': {'description': 'Learning topic', 'required': True},
                'content_type': {'description': 'Type of content', 'required': True, 'validation': r'^(article|book|course|video|podcast|experience)$'},
                'source': {'description': 'Content source', 'required': True},
                'content': {'description': 'Main content/summary', 'required': True},
                'learnings': {'description': 'Key learnings extracted', 'default': ''},
                'applications': {'description': 'Potential applications', 'default': ''}
            },
            model_preference='claude',
            output_format='markdown'
        )
    }

    RESEARCH_COMMANDS = {
        'extract-knowledge': PAIPattern(
            name='extract-knowledge',
            category='research',
            system_prompt='''You are a knowledge extraction specialist.
Extract, organize, and summarize valuable knowledge from various sources.
Focus on accuracy, completeness, and practical applicability.''',
            user_template='''Extract knowledge from this content:

Source Type: ${source_type}
Domain: ${domain}
Quality Level: ${quality}

Content:
${content}

Extraction Requirements:
- Fact extraction: ${extract_facts}
- Concept identification: ${identify_concepts}
- Relationship mapping: ${map_relationships}
- Practical applications: ${find_applications}
- Source credibility: ${assess_credibility}

Organize the extracted knowledge into:
1. Core concepts and principles
2. Supporting facts and evidence
3. Practical applications
4. Further reading recommendations
5. Knowledge gaps identified''',
            variables={
                'source_type': {'description': 'Type of source', 'required': True},
                'domain': {'description': 'Knowledge domain', 'required': True},
                'quality': {'description': 'Source quality', 'default': 'medium'},
                'content': {'description': 'Content to extract from', 'required': True},
                'extract_facts': {'description': 'Extract factual information', 'default': 'yes'},
                'identify_concepts': {'description': 'Identify key concepts', 'default': 'yes'},
                'map_relationships': {'description': 'Map relationships', 'default': 'yes'},
                'find_applications': {'description': 'Find practical applications', 'default': 'yes'},
                'assess_credibility': {'description': 'Assess source credibility', 'default': 'yes'}
            },
            model_preference='claude',
            output_format='markdown'
        ),

        'web-research': PAIPattern(
            name='web-research',
            category='research',
            system_prompt='''You are a professional researcher and information analyst.
Conduct comprehensive research using available tools, analyze findings, and provide well-structured insights.''',
            user_template='''Conduct research on: "${topic}"

Research Parameters:
- Depth: ${depth}
- Time period: ${time_period}
- Sources: ${preferred_sources}
- Geographic focus: ${geographic_focus}

Specific Requirements:
${requirements}

Research Questions:
${questions}

Deliverables:
1. Executive summary of findings
2. Key insights and patterns
3. Source evaluation and credibility assessment
4. Recommendations or conclusions
5. Gaps requiring further research''',
            variables={
                'topic': {'description': 'Research topic', 'required': True},
                'depth': {'description': 'Research depth', 'default': 'comprehensive', 'validation': r'^(summary|basic|comprehensive|exhaustive)$'},
                'time_period': {'description': 'Time period to research', 'default': 'current'},
                'preferred_sources': {'description': 'Preferred source types', 'default': 'academic, industry reports'},
                'geographic_focus': {'description': 'Geographic focus', 'default': 'global'},
                'requirements': {'description': 'Specific requirements', 'default': ''},
                'questions': {'description': 'Research questions to answer', 'default': ''}
            },
            model_preference='gpt',
            output_format='markdown'
        ),

        'analyze-research': PAIPattern(
            name='analyze-research',
            category='research',
            system_prompt='''You are a research methodology expert.
Critically analyze research quality, methodology, findings, and implications.''',
            user_template='''Analyze this research:

Research Title: ${title}
Type: ${research_type}
Field: ${field}
Authors/Institution: ${authors}

Methodology:
${methodology}

Key Findings:
${findings}

Analysis Requirements:
- Methodology evaluation: ${evaluate_methodology}
- Statistical analysis review: ${review_statistics}
- Bias assessment: ${assess_bias}
- Practical implications: ${evaluate_implications}
- Reproducibility check: ${check_reproducibility}

Provide detailed analysis covering:
1. Research design strengths and weaknesses
2. Data collection and analysis quality
3. Validity and reliability assessment
4. Practical significance of findings
5. Recommendations for future research''',
            variables={
                'title': {'description': 'Research title', 'required': True},
                'research_type': {'description': 'Type of research', 'required': True},
                'field': {'description': 'Research field', 'required': True},
                'authors': {'description': 'Authors/institution', 'required': True},
                'methodology': {'description': 'Research methodology', 'required': True},
                'findings': {'description': 'Key findings', 'required': True},
                'evaluate_methodology': {'description': 'Evaluate methodology', 'default': 'yes'},
                'review_statistics': {'description': 'Review statistical analysis', 'default': 'yes'},
                'assess_bias': {'description': 'Assess potential biases', 'default': 'yes'},
                'evaluate_implications': {'description': 'Evaluate practical implications', 'default': 'yes'},
                'check_reproducibility': {'description': 'Check reproducibility', 'default': 'yes'}
            },
            model_preference='claude',
            output_format='markdown'
        )
    }

    def __init__(self):
        self.patterns: Dict[str, PAIPattern] = {}
        self._load_builtin_patterns()

        # Command execution statistics
        self.execution_stats = {
            'commands_run': {},
            'average_execution_time': {},
            'success_rate': {}
        }

        logger.info(f"🎯 PAI Command Registry initialized with {len(self.patterns)} built-in patterns")

    def _load_builtin_patterns(self):
        """Load all built-in PAI patterns"""
        all_patterns = {
            **self.PROFESSIONAL_COMMANDS,
            **self.PERSONAL_COMMANDS,
            **self.RESEARCH_COMMANDS
        }

        for name, pattern in all_patterns.items():
            self.patterns[name] = pattern

    def get_pattern(self, name: str) -> Optional[PAIPattern]:
        """Get pattern by name"""
        return self.patterns.get(name)

    def list_patterns(self, category: Optional[str] = None) -> List[str]:
        """List available patterns, optionally filtered by category"""
        if category:
            return [name for name, pattern in self.patterns.items()
                   if pattern.category == category]
        return list(self.patterns.keys())

    def get_patterns_by_category(self) -> Dict[str, List[str]]:
        """Get patterns organized by category"""
        categories = {}
        for name, pattern in self.patterns.items():
            if pattern.category not in categories:
                categories[pattern.category] = []
            categories[pattern.category].append(name)
        return categories

    async def execute_command(self, command_name: str, inputs: Dict[str, Any] = None,
                            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a PAI command with inputs and context
        This is the core PAI command execution method
        """
        pattern = self.get_pattern(command_name)
        if not pattern:
            raise PatternError(f"Unknown command: {command_name}")

        inputs = inputs or {}

        # Execute hooks - Pai's pre-command processing
        await execute_hooks('pre-command',
                          command=command_name,
                          user_query=inputs.get('query', ''),
                          inputs=inputs)

        try:
            # Validate inputs
            validated_inputs = pattern.validate_inputs(inputs)

            # Load UFC context for command execution
            intent = await ufc_manager.classify_intent(
                validated_inputs.get('query', '') or command_name
            )
            ufc_context = await ufc_manager.load_context_by_intent(intent)

            # Add context to inputs
            validated_inputs['context'] = ufc_context.get('context', {})

            # Render user prompt
            user_prompt = pattern.render_user_prompt(validated_inputs)

            # Route to appropriate AI provider
            ai_provider = ai_orchestrator.get_provider(pattern.model_preference)
            result = await ai_provider.run_pattern(
                system_prompt=pattern.system_prompt,
                user_prompt=user_prompt,
                pattern_name=command_name
            )

            # Update usage statistics
            pattern.last_used = datetime.utcnow()
            pattern.usage_count += 1

            # Execute post-command hooks
            await execute_hooks('post-command',
                              command=command_name,
                              result=result,
                              success=True)

            response = {
                'command': command_name,
                'success': True,
                'result': result,
                'execution_time': datetime.utcnow().isoformat(),
                'model_used': ai_provider.name,
                'context_loaded': len(ufc_context.get('context', {})),
                'pattern_category': pattern.category,
                'output_format': pattern.output_format
            }

            # Update execution stats
            self._update_command_stats(command_name, True, None)

            return response

        except Exception as e:
            # Execute command failure hooks
            await execute_hooks('command-failure',
                              command=command_name,
                              error=str(e))

            self._update_command_stats(command_name, False, str(e))

            raise PatternError(f"Command execution failed: {str(e)}")

    def _update_command_stats(self, command_name: str, success: bool, error: Optional[str]):
        """Update command execution statistics"""
        if command_name not in self.execution_stats['commands_run']:
            self.execution_stats['commands_run'][command_name] = 0
            self.execution_stats['success_rate'][command_name] = 0.0

        self.execution_stats['commands_run'][command_name] += 1

        # Update success rate (simple moving average)
        current_rate = self.execution_stats['success_rate'].get(command_name, 1.0)
        success_rate = (current_rate + (1.0 if success else 0.0)) / 2.0
        self.execution_stats['success_rate'][command_name] = success_rate

    def get_command_stats(self) -> Dict[str, Any]:
        """Get command execution statistics"""
        return self.execution_stats.copy()

    def add_custom_pattern(self, pattern: PAIPattern):
        """Add a custom pattern to the registry"""
        self.patterns[pattern.name] = pattern
        logger.info(f"➕ Added custom pattern: {pattern.name} ({pattern.category})")

    def remove_pattern(self, pattern_name: str) -> bool:
        """Remove a pattern from the registry"""
        if pattern_name in self.patterns:
            del self.patterns[pattern_name]
            logger.info(f"➖ Removed pattern: {pattern_name}")
            return True
        return False


# Global registry instance
command_registry = PAICommandRegistry()


class PatternAPI:
    """Pattern System API Endpoints"""
    pass  # Will be implemented in the API router


# CLI Command aliases for common patterns
def create_cli_aliases():
    """Create shell aliases for PAI commands (like PAI does)"""
    common_commands = [
        'analyze-code', 'write-blog', 'get-newsletter-stats',
        'answer-finance-question', 'extract-knowledge', 'web-research'
    ]

    # In a real implementation, this would create shell alias files
    alias_info = {}
    for cmd in common_commands:
        alias_info[cmd] = f"gretapai {cmd}"

    return alias_info


__all__ = [
    'PAIPattern',
    'PAICommandRegistry',
    'PatternError',
    'command_registry',
    'PatternAPI',
    'create_cli_aliases'
]
