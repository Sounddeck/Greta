"""
GRETA PAI - Specialized Agent System
Core PAI Feature: Tools-enabled agents with parallelization
Inspired by PAI's agent ecosystem with specialized capabilities
"""
from typing import Dict, List, Any, Optional, Union, Callable
import asyncio
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from dataclasses import dataclass, field
import time
import re

from utils.error_handling import GretaException, handle_errors
from utils.hooks import hook_manager, HookContext, execute_hooks
from utils.ufc_context import ufc_manager
from utils.ai_providers import ai_orchestrator
from utils.mcp_servers import mcp_orchestrator
from utils.patterns import command_registry

logger = logging.getLogger(__name__)


class AgentError(GretaException):
    """Agent system errors"""


@dataclass
class AgentTool:
    """Represents a tool that an agent can use"""
    name: str
    description: str
    function: Callable
    async_mode: bool = False
    requires_mcp: Optional[str] = None  # MCP service name if needed
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTask:
    """Represents a task being executed by an agent"""
    task_id: str
    agent_name: str
    task_description: str
    status: str = "pending"  # pending, running, completed, failed
    priority: int = 5  # 1-10, higher = more important
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    tools_used: List[str] = field(default_factory=list)


class SpecializedAgent:
    """
    PAI's Specialized Agent with tools and parallel execution capabilities
    Inspired by PAI's researcher, engineer, designer agent system
    """

    def __init__(self, name: str, specialty: str, role_description: str,
                 tools: List[AgentTool] = None, voice_enabled: bool = True):
        self.name = name
        self.specialty = specialty
        self.role_description = role_description
        self.tools = tools or []
        self.voice_enabled = voice_enabled

        # Agent statistics and capabilities
        self.tasks_completed = 0
        self.success_rate = 1.0
        self.specialized_prompt = self._create_specialized_prompt()
        self.capabilities = self._extract_capabilities()

        # Parallel execution management
        self.max_concurrent_tasks = 3
        self._active_tasks: Dict[str, AgentTask] = {}
        self._task_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix=f"{name}-executor")

        logger.info(f"🤖 Specialized Agent '{name}' initialized - {specialty} expert")

    def _create_specialized_prompt(self) -> str:
        """Create the specialized system prompt for this agent"""
        tool_descriptions = "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in self.tools
        ])

        return f"""You are {self.name}, a specialized {self.specialty} expert in the GRETA PAI system.

{self.role_description}

You have access to the following specialized tools:
{tool_descriptions}

When executing tasks:
1. Analyze the request thoroughly
2. Use your specialized tools strategically
3. Provide detailed, actionable results
4. Collaborate with other PAI agents when needed
5. Maintain high standards of quality and precision

Your specialty: {self.specialty}
Expertise level: Advanced professional

Always provide outputs that demonstrate deep understanding of the domain."""

    def _extract_capabilities(self) -> List[str]:
        """Extract capabilities from tools and specialty"""
        capabilities = []

        # Add specialty-based capabilities
        if "research" in self.specialty.lower():
            capabilities.extend([
                "deep_web_research", "data_analysis", "information_synthesis",
                "source_evaluation", "trend_identification"
            ])
        elif "engineer" in self.specialty.lower():
            capabilities.extend([
                "code_analysis", "system_design", "debugging", "optimization",
                "architecture_review", "testing_strategy"
            ])
        elif "designer" in self.specialty.lower():
            capabilities.extend([
                "ui_ux_design", "user_research", "prototyping", "visual_design",
                "design_systems", "accessibility_audit"
            ])
        elif "pentester" in self.specialty.lower():
            capabilities.extend([
                "security_assessment", "vulnerability_scanning", "risk_analysis",
                "compliance_check", "threat_modeling"
            ])
        elif "architect" in self.specialty.lower():
            capabilities.extend([
                "system_architecture", "scalability_planning", "technology_evaluation",
                "infrastructure_design", "performance_optimization"
            ])

        # Add tool-based capabilities
        for tool in self.tools:
            capabilities.append(tool.name.replace('_', ' '))

        return capabilities

    def add_tool(self, tool: AgentTool):
        """Add a tool to the agent's toolkit"""
        self.tools.append(tool)
        # Regenerate system prompt with new tool
        self.specialized_prompt = self._create_specialized_prompt()
        self.capabilities = self._extract_capabilities()
        logger.info(f"🔧 Agent {self.name} added tool: {tool.name}")

    async def execute_task(self, task_description: str, context: Dict[str, Any] = None,
                          priority: int = 5) -> Dict[str, Any]:
        """
        Execute a specialized task with full PAI agent capabilities
        """
        task_id = f"{self.name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hash(task_description) % 1000}"
        context = context or {}

        task = AgentTask(
            task_id=task_id,
            agent_name=self.name,
            task_description=task_description,
            priority=priority,
            context=context
        )

        # Check concurrent task limits
        if len(self._active_tasks) >= self.max_concurrent_tasks:
            await self._wait_for_task_slot()

        self._active_tasks[task_id] = task

        try:
            # Execute pre-agent hooks
            await execute_hooks('agent-selected',
                              agent_name=self.name,
                              task=task_description,
                              specialty=self.specialty)

            await execute_hooks('agent-started',
                              task_id=task_id,
                              agent_name=self.name)

            # Execute the task
            task.status = "running"
            task.started_at = datetime.utcnow()

            result = await self._execute_specialized_task(task)

            # Complete the task
            task.status = "completed"
            task.completed_at = datetime.utcnow()
            task.result = result
            self.tasks_completed += 1

            # Execute completion hooks
            await execute_hooks('agent-completed',
                              task_id=task_id,
                              agent_name=self.name,
                              result=result)

            # Voice notification (PAI-style)
            if self.voice_enabled:
                await self._announce_completion(task, result)

            response = {
                'task_id': task_id,
                'agent': self.name,
                'status': 'completed',
                'result': result,
                'execution_time': (task.completed_at - task.started_at).total_seconds() if task.completed_at and task.started_at else None,
                'tools_used': task.tools_used,
                'specialty': self.specialty
            }

            return response

        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            task.completed_at = datetime.utcnow()

            await execute_hooks('agent-error',
                              task_id=task_id,
                              agent_name=self.name,
                              error=str(e))

            logger.error(f"Agent {self.name} task {task_id} failed: {e}")

            return {
                'task_id': task_id,
                'agent': self.name,
                'status': 'failed',
                'error': str(e),
                'specialty': self.specialty
            }

        finally:
            # Clean up completed task
            if task_id in self._active_tasks:
                del self._active_tasks[task_id]

    async def _execute_specialized_task(self, task: AgentTask) -> Dict[str, Any]:
        """
        Execute the specialized task using agent AI and tools
        """
        # Load relevant UFC context
        intent = await ufc_manager.classify_intent(task.task_description)
        context_data = await ufc_manager.load_context_by_intent(intent)

        # Create comprehensive task prompt
        task_context = {
            'specialty_instruction': self.specialized_prompt,
            'task_description': task.task_description,
            'agent_specialty': self.specialty,
            'available_tools': [tool.name for tool in self.tools],
            'context_data': context_data.get('context', {}),
            'relevant_files': context_data.get('relevant_files', []),
            'capabilities': self.capabilities
        }

        # Get preferred AI provider for this specialty
        provider_preference = self._get_provider_preference()
        provider = ai_orchestrator.get_provider(provider_preference)

        # Execute with provider
        system_prompt = task_context['specialty_instruction']
        user_prompt = self._create_task_prompt(task_context)

        result = await provider.run_pattern(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            pattern_name=f"agent_{self.name}_{task.task_id}"
        )

        # Check if result wants to use tools
        tool_results = await self._execute_tool_calls(result, task)

        return {
            'ai_response': result,
            'tool_results': tool_results,
            'context_used': len(context_data.get('context', {})),
            'tools_executed': len(tool_results),
            'specialty_insights': self._extract_specialty_insights(result)
        }

    def _get_provider_preference(self) -> str:
        """Get preferred AI provider for this specialty"""
        # PAI-style provider mapping
        provider_map = {
            'research': 'claude',      # Best for research and analysis
            'engineering': 'claude',   # Best for coding and logic
            'design': 'gemini',        # Best for creative visual tasks
            'security': 'claude',      # Best for detailed analysis
            'architecture': 'gpt',     # Best for business/complex planning
            'personal': 'ollama',      # Privacy-first for personal
        }

        return provider_map.get(self.specialty.lower(), 'claude')

    def _create_task_prompt(self, task_context: Dict[str, Any]) -> str:
        """Create the user prompt for the agent task"""
        return f"""Task: {task_context['task_description']}

Your Specialty: {task_context['agent_specialty']}
Your Capabilities: {', '.join(task_context['capabilities'])}
Available Tools: {', '.join(task_context['available_tools'])}

Relevant Context:
{json.dumps(task_context['context_data'], indent=2)}

Task Instructions:
1. Apply your specialized {task_context['agent_specialty']} expertise
2. Use available tools when they would help (specify tool calls clearly)
3. Provide detailed, actionable results
4. Reference relevant context when applicable
5. Maintain professional standards appropriate to your specialty

Execute this task to the best of your specialized capabilities."""

    async def _execute_tool_calls(self, ai_response: str, task: AgentTask) -> List[Dict[str, Any]]:
        """Execute any tool calls found in the AI response"""
        tool_results = []

        # Simple tool call detection (in production would use more sophisticated parsing)
        tool_call_patterns = {
            'web_research': r'web_research\((.*?)\)',
            'code_analysis': r'code_analysis\((.*?)\)',
            'financial_data': r'financial_data\((.*?)\)',
            'communication': r'communication\((.*?)\)',
        }

        for tool_name, pattern in tool_call_patterns.items():
            matches = re.findall(pattern, ai_response, re.IGNORECASE)
            for match in matches:
                try:
                    # Parse tool parameters (simplified)
                    params = self._parse_tool_params(match)

                    # Execute tool
                    result = await self._execute_tool(tool_name, params)

                    tool_results.append({
                        'tool': tool_name,
                        'parameters': params,
                        'result': result,
                        'success': True
                    })

                    task.tools_used.append(tool_name)

                except Exception as e:
                    logger.warning(f"Tool {tool_name} execution failed: {e}")
                    tool_results.append({
                        'tool': tool_name,
                        'parameters': params,
                        'error': str(e),
                        'success': False
                    })

        return tool_results

    async def _execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Execute a specific tool"""
        # Find the tool
        tool = next((t for t in self.tools if t.name == tool_name), None)
        if not tool:
            raise AgentError(f"Tool {tool_name} not found in agent toolkit")

        task.tools_used.append(tool_name)

        # Execute via MCP if required
        if tool.requires_mcp:
            return await mcp_orchestrator.execute_service_task(
                tool.requires_mcp, tool_name, **params
            )

        # Execute directly
        if tool.async_mode:
            return await tool.function(**params)
        else:
            # Run in thread pool to not block
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self._task_executor, tool.function, **params)

    def _parse_tool_params(self, param_string: str) -> Dict[str, Any]:
        """Parse tool parameters from string (simplified)"""
        # In production would use proper JSON parsing or argument parsing
        params = {}
        try:
            # Simple split and assign (placeholder)
            pairs = param_string.split(',')
            for pair in pairs:
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    params[key.strip()] = value.strip().strip('"\'')

        except Exception:
            # Fallback: treat as single parameter
            params['query'] = param_string

        return params

    def _extract_specialty_insights(self, ai_response: str) -> List[str]:
        """Extract specialty-specific insights from response"""
        insights = []

        # Specialty-specific insight patterns
        specialty_patterns = {
            'research': ['key findings', 'important trends', 'recommendations'],
            'engineering': ['code issues', 'performance improvements', 'architecture suggestions'],
            'design': ['user experience', 'visual improvements', 'usability enhancements'],
            'security': ['vulnerabilities', 'risk levels', 'security recommendations'],
            'architecture': ['scalability issues', 'design improvements', 'technology recommendations']
        }

        patterns = specialty_patterns.get(self.specialty.lower(), [])
        for pattern in patterns:
            if pattern.lower() in ai_response.lower():
                insights.append(f"Contains {pattern} recommendations")

        return insights

    async def _announce_completion(self, task: AgentTask, result: Any):
        """Announce task completion using voice (PAI-style)"""
        try:
            completion_message = f"Task completed by {self.name} the {self.specialty} specialist"

            # Use PAI communication service for voice
            voice_result = await mcp_orchestrator.execute_service_task(
                'communication',
                'elevenlabs_tts',
                text=completion_message,
                voice='Kore'
            )

            if voice_result and voice_result.get('status') == 'generated':
                logger.info(f"🔊 Voice notification for agent {self.name} task completion")

        except Exception as e:
            logger.debug(f"Voice notification failed (non-critical): {e}")
            # Don't fail the task if voice fails

    async def _wait_for_task_slot(self):
        """Wait for an active task to complete to free up a slot"""
        while len(self._active_tasks) >= self.max_concurrent_tasks:
            await asyncio.sleep(0.1)

            # Clean up any completed tasks
            completed_tasks = [tid for tid, task in self._active_tasks.items()
                             if task.status in ['completed', 'failed']]
            for tid in completed_tasks:
                del self._active_tasks[tid]

    def get_status(self) -> Dict[str, Any]:
        """Get agent status and statistics"""
        return {
            'name': self.name,
            'specialty': self.specialty,
            'active_tasks': len(self._active_tasks),
            'tasks_completed': self.tasks_completed,
            'success_rate': self.success_rate,
            'tools_available': len(self.tools),
            'capabilities': self.capabilities,
            'voice_enabled': self.voice_enabled
        }


class ResearcherAgent(SpecializedAgent):
    """PAI Researcher Agent - Deep web research and information analysis"""

    def __init__(self):
        tools = [
            AgentTool(
                name='web_research',
                description='Conduct deep web research using multiple sources',
                function=self._web_research_tool,
                requires_mcp='web_browser',
                async_mode=True
            ),
            AgentTool(
                name='data_analysis',
                description='Analyze and synthesize research data',
                function=self._analyze_research_data,
                async_mode=True
            ),
            AgentTool(
                name='source_evaluation',
                description='Evaluate credibility and quality of information sources',
                function=self._evaluate_sources,
                async_mode=False
            )
        ]

        super().__init__(
            name='Researcher',
            specialty='research',
            role_description='''You are Dr. Elena Vasquez, a world-renowned research scientist and information analyst.
You possess expertise in deep research methodologies, data synthesis, and source evaluation.
Your mission is to uncover truth, identify patterns, and provide comprehensive analysis.''',
            tools=tools,
            voice_enabled=True
        )


class EngineerAgent(SpecializedAgent):
    """PAI Engineer Agent - Code analysis, debugging, and system design"""

    def __init__(self):
        tools = [
            AgentTool(
                name='code_analysis',
                description='Analyze code for bugs, performance issues, and best practices',
                function=self._code_analysis_tool,
                async_mode=True
            ),
            AgentTool(
                name='debugging_assistant',
                description='Help identify and resolve software bugs',
                function=self._debugging_tool,
                async_mode=False
            ),
            AgentTool(
                name='performance_optimizer',
                description='Optimize code and system performance',
                function=self._performance_tool,
                async_mode=True
            )
        ]

        super().__init__(
            name='Engineer',
            specialty='engineering',
            role_description='''You are Marcus Chen, a senior software engineer and system architect.
You have 15+ years of experience in software development, system design, and optimization.
Your expertise covers multiple programming languages, frameworks, and architectures.''',
            tools=tools,
            voice_enabled=True
        )


class DesignerAgent(SpecializedAgent):
    """PAI Designer Agent - UI/UX design and user experience"""

    def __init__(self):
        tools = [
            AgentTool(
                name='ui_design_assessment',
                description='Evaluate user interface design quality',
                function=self._ui_design_tool,
                async_mode=False
            ),
            AgentTool(
                name='user_research',
                description='Conduct user experience research and analysis',
                function=self._user_research_tool,
                async_mode=True
            ),
            AgentTool(
                name='prototyping_assistant',
                description='Help design and iterate on prototypes',
                function=self._prototyping_tool,
                async_mode=False
            )
        ]

        super().__init__(
            name='Designer',
            specialty='design',
            role_description='''You are Sofia Rodriguez, a senior UX/UI designer and design researcher.
You specialize in human-centered design, user experience optimization, and visual design excellence.
Your work follows Nielsen Norman Group guidelines and incorporates cognitive psychology principles.''',
            tools=tools,
            voice_enabled=True
        )


class PentesterAgent(SpecializedAgent):
    """PAI Pentester Agent - Security testing and vulnerability assessment"""

    def __init__(self):
        tools = [
            AgentTool(
                name='security_assessment',
                description='Conduct comprehensive security assessments',
                function=self._security_assessment_tool,
                async_mode=True
            ),
            AgentTool(
                name='vulnerability_scan',
                description='Scan for potential security vulnerabilities',
                function=self._vulnerability_scan_tool,
                async_mode=True
            ),
            AgentTool(
                name='risk_analysis',
                description='Analyze security risks and impact',
                function=self._risk_analysis_tool,
                async_mode=False
            )
        ]

        super().__init__(
            name='Pentester',
            specialty='security',
            role_description='''You are Agent Zero, a certified ethical hacker and cybersecurity expert.
You have extensive experience in penetration testing, vulnerability assessment, and security architecture.
Your methodologies follow OWASP guidelines and industry best practices.''',
            tools=tools,
            voice_enabled=True
        )


class ArchitectAgent(SpecializedAgent):
    """PAI Architect Agent - System architecture and technical planning"""

    def __init__(self):
        tools = [
            AgentTool(
                name='architecture_review',
                description='Review and optimize system architectures',
                function=self._architecture_review_tool,
                async_mode=False
            ),
            AgentTool(
                name='scalability_planning',
                description='Design scalable system architectures',
                function=self._scalability_planning_tool,
                async_mode=True
            ),
            AgentTool(
                name='technology_evaluation',
                description='Evaluate and recommend technologies',
                function=self._technology_eval_tool,
                async_mode=True
            )
        ]

        super().__init__(
            name='Architect',
            specialty='architecture',
            role_description='''You are Dr. James Harrington, a systems architect and technology strategist.
You have 20+ years of experience in enterprise architecture, distributed systems, and technology planning.
Your designs prioritize scalability, reliability, maintainability, and cost-effectiveness.''',
            tools=tools,
            voice_enabled=True
        )


class AgentOrchestrator:
    """
    PAI Agent Orchestrator - Manages specialized agents and parallel execution
    Core of PAI's multi-agent workflow system
    """

    def __init__(self):
        self.agents: Dict[str, SpecializedAgent] = {}
        self._initialize_agents()

        # Workflow management
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.workflow_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="workflow-executor")

        logger.info(f"🎭 Agent Orchestrator initialized with {len(self.agents)} specialized agents")

    def _initialize_agents(self):
        """Initialize all PAI specialized agents"""
        self.agents = {
            'researcher': ResearcherAgent(),
            'engineer': EngineerAgent(),
            'designer': DesignerAgent(),
            'pentester': PentesterAgent(),
            'architect': ArchitectAgent()
        }

    def get_agent(self, agent_type: str) -> SpecializedAgent:
        """Get a specialized agent by type"""
        agent_key = agent_type.lower()
        if agent_key not in self.agents:
            available = ", ".join(self.agents.keys())
            raise AgentError(f"Agent type '{agent_type}' not found. Available: {available}")

        return self.agents[agent_key]

    async def execute_agent_task(self, agent_type: str, task_description: str,
                               context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a task with a specific specialized agent
        """
        agent = self.get_agent(agent_type)

        logger.info(f"🤖 Executing {agent_type} task: {task_description[:50]}...")

        # Execute pre-agent hooks
        await execute_hooks('agent-selected',
                          agent_type=agent_type,
                          agent_name=agent.name,
                          task=task_description)

        result = await agent.execute_task(task_description, context)

        logger.info(f"✅ Agent {agent_type} completed task with status: {result['status']}")
        return result

    async def execute_parallel_workflow(self, workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a complex workflow with multiple agents working in parallel
        PAI's advanced multi-agent coordination
        """
        workflow_id = f"workflow_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        self.active_workflows[workflow_id] = {
            'status': 'running',
            'start_time': datetime.utcnow(),
            'tasks': workflow_config.get('tasks', []),
            'results': {},
            'agent_assignments': workflow_config.get('agent_assignments', {})
        }

        try:
            # Analyze workflow to determine execution strategy
            execution_plan = await self._analyze_workflow(workflow_config)

            # Execute tasks based on dependencies and parallelization
            results = await self._execute_workflow_tasks(execution_plan, workflow_id)

            # Synthesize final result
            final_result = await self._synthesize_workflow_results(results, workflow_config)

            self.active_workflows[workflow_id].update({
                'status': 'completed',
                'end_time': datetime.utcnow(),
                'final_result': final_result
            })

            return {
                'workflow_id': workflow_id,
                'status': 'completed',
                'result': final_result,
                'task_results': results,
                'execution_time': (datetime.utcnow() - self.active_workflows[workflow_id]['start_time']).total_seconds()
            }

        except Exception as e:
            self.active_workflows[workflow_id]['status'] = 'failed'
            logger.error(f"Workflow {workflow_id} failed: {e}")

            return {
                'workflow_id': workflow_id,
                'status': 'failed',
                'error': str(e)
            }

    async def _analyze_workflow(self, workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze workflow to determine execution strategy"""
        tasks = workflow_config.get('tasks', [])
        agent_assignments = workflow_config.get('agent_assignments', {})

        # Determine which tasks can run in parallel
        parallel_tasks = []
        sequential_tasks = []

        task_dependencies = {}
        for task in tasks:
            deps = task.get('dependencies', [])
            task_dependencies[task['id']] = deps

            if not deps:
                parallel_tasks.append(task)
            else:
                sequential_tasks.append(task)

        return {
            'parallel_tasks': parallel_tasks,
            'sequential_tasks': sequential_tasks,
            'dependencies': task_dependencies,
            'agent_assignments': agent_assignments
        }

    async def _execute_workflow_tasks(self, execution_plan: Dict[str, Any],
                                    workflow_id: str) -> Dict[str, Any]:
        """Execute workflow tasks with parallelization"""
        results = {}

        # Execute parallel tasks first
        if execution_plan['parallel_tasks']:
            parallel_results = await asyncio.gather(*[
                self.execute_agent_task(
                    execution_plan['agent_assignments'].get(task['id'], 'engineer'),
                    task['description'],
                    task.get('context', {})
                )
                for task in execution_plan['parallel_tasks']
            ], return_exceptions=True)

            for i, result in enumerate(parallel_results):
                task_id = execution_plan['parallel_tasks'][i]['id']
                if isinstance(result, Exception):
                    results[task_id] = {'status': 'failed', 'error': str(result)}
                else:
                    results[task_id] = result

        # Execute sequential tasks
        for task in execution_plan['sequential_tasks']:
            # Check if dependencies are satisfied
            deps = execution_plan['dependencies'][task['id']]
            deps_satisfied = all(results.get(dep, {}).get('status') == 'completed' for dep in deps)

            if not deps_satisfied:
                results[task['id']] = {
                    'status': 'skipped',
                    'reason': 'Dependencies not satisfied'
                }
                continue

            agent_type = execution_plan['agent_assignments'].get(task['id'], 'engineer')
            result = await self.execute_agent_task(agent_type, task['description'], task.get('context', {}))
            results[task['id']] = result

        return results

    async def _synthesize_workflow_results(self, task_results: Dict[str, Any],
                                         workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize final workflow result from all task results"""
        # Use a synthesis agent to combine results
        synthesis_prompt = f"""
        Synthesize the following workflow results into a comprehensive final output:

        Workflow: {workflow_config.get('name', 'Multi-agent collaboration')}

        Task Results:
        {json.dumps(task_results, indent=2)}

        Provide:
        1. Executive summary of all findings
        2. Key insights and conclusions
        3. Actionable recommendations
        4. Any conflicts or areas needing clarification
        """

        synthesis_agent = self.get_agent('researcher')  # Good at synthesis
        synthesis_result = await synthesis_agent.execute_task(
            synthesis_prompt,
            context={'synthesis': True, 'workflow_config': workflow_config}
        )

        return {
            'synthesis': synthesis_result.get('result', {}).get('ai_response', 'Synthesis failed'),
            'raw_results': task_results,
            'workflow_type': workflow_config.get('type', 'unknown')
        }

    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return {
            agent_name: agent.get_status()
            for agent_name, agent in self.agents.items()
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall agent system status"""
        total_tasks = sum(agent.tasks_completed for agent in self.agents.values())
        avg_success_rate = sum(agent.success_rate for agent in self.agents.values()) / len(self.agents)

        return {
            'total_agents': len(self.agents),
            'agent_types': list(self.agents.keys()),
            'total_tasks_completed': total_tasks,
            'average_success_rate': avg_success_rate,
            'active_workflows': len(self.active_workflows),
            'system_health': 'excellent' if avg_success_rate > 0.95 else 'good' if avg_success_rate > 0.8 else 'needs_attention'
        }


# Global PAI Agent Orchestrator instance
agent_orchestrator = AgentOrchestrator()


# Agent tool implementations (simplified - would be full implementations in production)
async def _web_research_tool(self, query: str, sources: int = 5) -> Dict[str, Any]:
    """PAI Web Research Tool"""
    return await mcp_orchestrator.execute_service_task('web_browser', 'page_content', url=query)

async def _code_analysis_tool(self, language: str, code: str) -> Dict[str, Any]:
    """PAI Code Analysis Tool"""
    return await command_registry.execute_command('analyze-code', {'language': language, 'code': code})

async def _security_assessment_tool(self, target: str) -> Dict[str, Any]:
    """PAI Security Assessment Tool"""
    # Would integrate with security scanning tools
    return {'status': 'assessment_completed', 'target': target, 'risk_level': 'medium'}


__all__ = [
    'SpecializedAgent',
    'ResearcherAgent',
    'EngineerAgent',
    'DesignerAgent',
    'PentesterAgent',
    'ArchitectAgent',
    'AgentOrchestrator',
    'agent_orchestrator'
]
