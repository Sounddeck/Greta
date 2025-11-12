"""
Executive Team Service for Multi-Business Management

Creates and manages AI executive teams with elite business advisor capabilities.
Integrates with Agent Builder system for deployment and execution.
"""

import logging
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime

from models.business_models import (
    AgentConfig, DecisionAuthority, ExecutiveTeam,
    ExecutiveDecision, BusinessStatus
)
from models.agent_builder_models import (
    CustomAgent, CustomAgentCreate, AgentType, AgentStatus,
    ToolAssignment, AgentExecutionCreate
)
from database import Database
from services.agent_builder_service import AgentBuilderService

logger = logging.getLogger(__name__)


class ExecutiveTeamService:
    """Service for managing AI executive teams"""
    
    def __init__(self, db: Database):
        self.db = db
        self.agent_builder = AgentBuilderService(db)
        self.collection_name = "executive_teams"
    
    # ============================================================================
    # EXECUTIVE AGENT TEMPLATES
    # ============================================================================
    
    def get_elite_business_advisor_template(self) -> Dict[str, Any]:
        """
        Elite Business Advisor Template
        
        Equivalent to Harvard MBA or Wharton MBA graduate with expertise in:
        - Strategic planning and business development
        - Financial analysis and forecasting
        - Operations management and optimization
        - Technology strategy and digital transformation
        - Market analysis and competitive intelligence
        """
        return {
            "name": "Elite Business Advisor",
            "role": "Chief Business Advisor",
            "description": "Elite AI business advisor with MBA-level expertise across all business functions",
            "capabilities": [
                # Strategic Planning
                "strategic_planning",
                "business_model_design",
                "market_analysis",
                "competitive_intelligence",
                "growth_strategy",
                "risk_assessment",
                
                # Financial Management
                "financial_analysis",
                "budgeting_forecasting",
                "investment_analysis",
                "cash_flow_management",
                "financial_modeling",
                "roi_calculation",
                
                # Operations
                "process_optimization",
                "resource_allocation",
                "supply_chain_management",
                "quality_management",
                "performance_metrics",
                
                # Technology & Innovation
                "digital_transformation",
                "technology_strategy",
                "innovation_management",
                "automation_opportunities",
                "data_analytics",
                
                # Leadership & Management
                "organizational_design",
                "talent_management",
                "change_management",
                "stakeholder_management",
                "decision_making",
            ],
            "tools": [
                "web_search",
                "data_analysis",
                "financial_calculator",
                "market_research",
                "document_analysis",
                "report_generation",
                "email_communication",
                "calendar_management",
                "crm_integration",
                "ats_integration",
            ],
            "personality": """You are an elite business advisor with the combined expertise of a Harvard MBA and Wharton MBA graduate. 

Your approach is:
- **Strategic**: Always think long-term and consider multiple scenarios
- **Data-Driven**: Base recommendations on solid analysis and metrics
- **Pragmatic**: Balance theory with practical implementation
- **Holistic**: Consider all aspects of the business (finance, operations, technology, people)
- **Proactive**: Anticipate challenges and opportunities before they arise
- **Clear**: Communicate complex concepts in simple, actionable terms

You excel at:
1. Analyzing complex business situations from multiple angles
2. Developing comprehensive strategies with clear action plans
3. Identifying risks and opportunities others might miss
4. Making data-driven recommendations with confidence intervals
5. Balancing short-term needs with long-term vision
6. Translating strategy into executable tactics

Your decision-making framework:
- Gather relevant data and context
- Analyze from financial, operational, strategic, and risk perspectives
- Consider multiple options with pros/cons
- Recommend the optimal path with clear rationale
- Define success metrics and monitoring approach
- Plan for contingencies

You communicate in a professional, confident manner while remaining humble and open to feedback.""",
            "decision_authority": DecisionAuthority.HIGH,
            "requires_approval_for": [
                "major_investments",  # > $50,000
                "strategic_pivots",
                "major_partnerships",
                "significant_hiring",  # > 5 people
                "major_technology_changes",
                "legal_commitments",
            ],
            "system_prompt": """You are an Elite Business Advisor AI with MBA-level expertise across all business functions.

CORE COMPETENCIES:
1. Strategic Planning & Business Development
2. Financial Analysis & Management
3. Operations & Process Optimization
4. Technology Strategy & Digital Transformation
5. Market Analysis & Competitive Intelligence
6. Leadership & Organizational Management

DECISION-MAKING FRAMEWORK:
1. **Analyze**: Gather data, understand context, identify key factors
2. **Evaluate**: Consider multiple options, assess risks/opportunities
3. **Recommend**: Provide clear recommendation with rationale
4. **Plan**: Define action steps, metrics, and contingencies
5. **Monitor**: Track progress and adjust as needed

COMMUNICATION STYLE:
- Clear, concise, and actionable
- Data-driven with specific metrics
- Strategic yet practical
- Confident but open to feedback
- Professional and respectful

APPROVAL REQUIREMENTS:
You have HIGH decision authority but require human approval for:
- Major investments (>$50K)
- Strategic pivots
- Major partnerships
- Significant hiring (>5 people)
- Major technology changes
- Legal commitments

For all other decisions, you can act autonomously within your authority level.

When making recommendations:
1. State the situation clearly
2. Present your analysis with supporting data
3. Recommend a specific course of action
4. Explain the rationale and expected outcomes
5. Identify risks and mitigation strategies
6. Define success metrics
7. Request approval if required

Always prioritize the business's long-term success while addressing immediate needs."""
        }
    
    def get_ceo_template(self) -> Dict[str, Any]:
        """CEO Agent Template - Strategic Leadership"""
        return {
            "name": "AI CEO",
            "role": "Chief Executive Officer",
            "description": "Strategic leader focused on vision, growth, and stakeholder management",
            "capabilities": [
                "strategic_vision",
                "business_development",
                "stakeholder_management",
                "fundraising",
                "partnership_development",
                "brand_strategy",
                "crisis_management",
                "board_communication",
            ],
            "tools": [
                "web_search",
                "email_communication",
                "calendar_management",
                "document_analysis",
                "report_generation",
            ],
            "personality": "Visionary leader focused on growth and strategic direction",
            "decision_authority": DecisionAuthority.HIGH,
            "requires_approval_for": [
                "major_strategic_decisions",
                "fundraising",
                "major_partnerships",
                "company_pivots",
            ],
        }
    
    def get_cfo_template(self) -> Dict[str, Any]:
        """CFO Agent Template - Financial Management"""
        return {
            "name": "AI CFO",
            "role": "Chief Financial Officer",
            "description": "Financial expert managing budgets, forecasts, and financial strategy",
            "capabilities": [
                "financial_planning",
                "budgeting",
                "forecasting",
                "financial_reporting",
                "cash_flow_management",
                "investment_analysis",
                "risk_management",
                "compliance",
            ],
            "tools": [
                "financial_calculator",
                "data_analysis",
                "report_generation",
                "spreadsheet_integration",
            ],
            "personality": "Detail-oriented financial expert focused on fiscal responsibility",
            "decision_authority": DecisionAuthority.MEDIUM,
            "requires_approval_for": [
                "major_expenditures",
                "investment_decisions",
                "financial_restructuring",
            ],
        }
    
    def get_coo_template(self) -> Dict[str, Any]:
        """COO Agent Template - Operations Management"""
        return {
            "name": "AI COO",
            "role": "Chief Operating Officer",
            "description": "Operations expert optimizing processes and resource allocation",
            "capabilities": [
                "process_optimization",
                "resource_management",
                "supply_chain",
                "quality_control",
                "performance_monitoring",
                "vendor_management",
                "operational_efficiency",
            ],
            "tools": [
                "data_analysis",
                "process_automation",
                "project_management",
                "crm_integration",
            ],
            "personality": "Efficiency-focused operations expert driving execution excellence",
            "decision_authority": DecisionAuthority.MEDIUM,
            "requires_approval_for": [
                "major_process_changes",
                "significant_vendor_contracts",
                "operational_restructuring",
            ],
        }
    
    def get_cto_template(self) -> Dict[str, Any]:
        """CTO Agent Template - Technology Strategy"""
        return {
            "name": "AI CTO",
            "role": "Chief Technology Officer",
            "description": "Technology leader managing tech stack, architecture, and innovation",
            "capabilities": [
                "technology_strategy",
                "architecture_design",
                "technical_leadership",
                "innovation_management",
                "security_management",
                "technical_debt_management",
                "vendor_evaluation",
            ],
            "tools": [
                "code_analysis",
                "technical_documentation",
                "api_integration",
                "security_scanning",
            ],
            "personality": "Innovative technology leader balancing cutting-edge with stability",
            "decision_authority": DecisionAuthority.MEDIUM,
            "requires_approval_for": [
                "major_architecture_changes",
                "significant_technology_investments",
                "security_critical_decisions",
            ],
        }

    def get_sales_template(self) -> Dict[str, Any]:
        """Sales Agent Template - Revenue Generation & Customer Acquisition"""
        return {
            "name": "AI Sales Director",
            "role": "Sales Director",
            "description": "Sales expert focused on revenue generation, customer acquisition, and deal closing",
            "capabilities": [
                "sales_strategy",
                "lead_generation",
                "customer_relationship_management",
                "deal_negotiation",
                "pipeline_management",
                "sales_forecasting",
                "account_management",
            ],
            "tools": [
                "crm_integration",
                "email_communication",
                "lead_scoring",
                "sales_analytics",
            ],
            "personality": "Results-driven sales leader focused on building relationships and closing deals",
            "decision_authority": DecisionAuthority.MEDIUM,
            "requires_approval_for": [
                "major_pricing_changes",
                "large_contract_negotiations",
                "strategic_partnership_deals",
            ],
        }

    def get_marketing_template(self) -> Dict[str, Any]:
        """Marketing Agent Template - Brand & Growth Marketing"""
        return {
            "name": "AI Marketing Director",
            "role": "Marketing Director",
            "description": "Marketing expert managing brand strategy, campaigns, and customer acquisition",
            "capabilities": [
                "marketing_strategy",
                "brand_management",
                "campaign_planning",
                "content_strategy",
                "digital_marketing",
                "market_research",
                "customer_segmentation",
            ],
            "tools": [
                "analytics_integration",
                "content_creation",
                "social_media_management",
                "email_marketing",
            ],
            "personality": "Creative marketing leader balancing brand vision with data-driven growth",
            "decision_authority": DecisionAuthority.MEDIUM,
            "requires_approval_for": [
                "major_brand_changes",
                "large_marketing_campaigns",
                "significant_marketing_budget_allocation",
            ],
        }

    def get_accounting_template(self) -> Dict[str, Any]:
        """Accounting Agent Template - Financial Reporting & Compliance"""
        return {
            "name": "AI Accounting Manager",
            "role": "Accounting Manager",
            "description": "Accounting expert managing financial records, reporting, and compliance",
            "capabilities": [
                "financial_reporting",
                "bookkeeping",
                "tax_compliance",
                "audit_preparation",
                "accounts_payable_receivable",
                "financial_reconciliation",
                "expense_management",
            ],
            "tools": [
                "accounting_software_integration",
                "financial_reporting",
                "tax_calculation",
                "invoice_management",
            ],
            "personality": "Detail-oriented accounting professional ensuring accuracy and compliance",
            "decision_authority": DecisionAuthority.LOW,
            "requires_approval_for": [
                "major_accounting_policy_changes",
                "tax_strategy_decisions",
                "audit_responses",
            ],
        }

    # ============================================================================
    # TEAM CREATION & MANAGEMENT
    # ============================================================================

    async def create_elite_advisor(self, business_id: str, business_name: str) -> AgentConfig:
        """
        Create an elite business advisor agent for a business

        Args:
            business_id: ID of the business
            business_name: Name of the business

        Returns:
            AgentConfig for the elite advisor
        """
        template = self.get_elite_business_advisor_template()

        # Create agent in Agent Builder system
        agent_create = CustomAgentCreate(
            name=f"{business_name} - Elite Business Advisor",
            description=template["description"],
            agent_type=AgentType.GENERAL,
            capabilities=template["capabilities"],
            system_prompt=template["system_prompt"],
            tools=[
                ToolAssignment(tool_id=tool, is_enabled=True, configuration={}, permissions=[])
                for tool in template["tools"]
            ],
            configuration={
                "business_id": business_id,
                "business_name": business_name,
                "role": template["role"],
                "personality": template["personality"],
                "decision_authority": template["decision_authority"],
                "requires_approval_for": template["requires_approval_for"],
            },
            tags=["executive", "business_advisor", "elite", business_id],
            metadata={
                "business_id": business_id,
                "executive_role": "elite_advisor",
                "created_by": "executive_team_service",
            }
        )

        # Create agent
        agent = await self.agent_builder.create_agent(agent_create)

        # Deploy agent
        from models.agent_builder_models import DeploymentCreate, DeploymentEnvironment
        deployment = await self.agent_builder.deploy_agent(
            DeploymentCreate(
                agent_id=agent.agent_id,
                environment=DeploymentEnvironment.PRODUCTION,
                configuration={
                    "business_id": business_id,
                    "auto_start": True,
                }
            )
        )

        logger.info(f"Created and deployed elite advisor for business {business_id}: {agent.agent_id}")

        # Return AgentConfig
        return AgentConfig(
            name=template["name"],
            role=template["role"],
            capabilities=template["capabilities"],
            tools=template["tools"],
            personality=template["personality"],
            decision_authority=template["decision_authority"],
            requires_approval_for=template["requires_approval_for"],
            agent_id=agent.agent_id,
        )

    async def create_executive_team(
        self,
        business_id: str,
        business_name: str,
        team_type: str = "elite_advisor"  # "elite_advisor" or "full_team"
    ) -> ExecutiveTeam:
        """
        Create an executive team for a business

        Args:
            business_id: ID of the business
            business_name: Name of the business
            team_type: Type of team ("elite_advisor" for single advisor, "full_team" for all executives)

        Returns:
            ExecutiveTeam configuration
        """
        if team_type == "elite_advisor":
            # Create single elite advisor
            advisor = await self.create_elite_advisor(business_id, business_name)
            return ExecutiveTeam(business_advisor=advisor)

        elif team_type == "full_team":
            # Create full executive team (CEO, CFO, COO, CTO, Sales, Marketing, Accounting)
            team = ExecutiveTeam()

            # Create each executive
            templates = {
                "ceo": self.get_ceo_template(),
                "cfo": self.get_cfo_template(),
                "coo": self.get_coo_template(),
                "cto": self.get_cto_template(),
                "sales": self.get_sales_template(),
                "marketing": self.get_marketing_template(),
                "accounting": self.get_accounting_template(),
            }

            for role, template in templates.items():
                agent_create = CustomAgentCreate(
                    name=f"{business_name} - {template['name']}",
                    description=template["description"],
                    agent_type=AgentType.ROLE_BASED,
                    capabilities=template["capabilities"],
                    system_prompt=self._generate_executive_prompt(template),
                    tools=[
                        ToolAssignment(tool_id=tool, enabled=True, configuration={})
                        for tool in template["tools"]
                    ],
                    configuration={
                        "business_id": business_id,
                        "business_name": business_name,
                        "role": template["role"],
                        "personality": template["personality"],
                        "decision_authority": template["decision_authority"],
                        "requires_approval_for": template["requires_approval_for"],
                    },
                    tags=["executive", role, business_id],
                    metadata={
                        "business_id": business_id,
                        "executive_role": role,
                        "created_by": "executive_team_service",
                    }
                )

                agent = await self.agent_builder.create_agent(agent_create)

                # Deploy agent
                from models.agent_builder_models import DeploymentCreate, DeploymentEnvironment
                await self.agent_builder.deploy_agent(
                    DeploymentCreate(
                        agent_id=agent.agent_id,
                        environment=DeploymentEnvironment.PRODUCTION,
                        configuration={"business_id": business_id, "auto_start": True}
                    )
                )

                # Add to team
                agent_config = AgentConfig(
                    name=template["name"],
                    role=template["role"],
                    capabilities=template["capabilities"],
                    tools=template["tools"],
                    personality=template["personality"],
                    decision_authority=template["decision_authority"],
                    requires_approval_for=template["requires_approval_for"],
                    agent_id=agent.agent_id,
                )

                setattr(team, role, agent_config)

            logger.info(f"Created full executive team for business {business_id}")
            return team

        else:
            raise ValueError(f"Invalid team_type: {team_type}")

    def _generate_executive_prompt(self, template: Dict[str, Any]) -> str:
        """Generate system prompt for an executive agent"""
        return f"""You are an {template['name']} AI agent with expertise in {template['role']}.

ROLE: {template['role']}
PERSONALITY: {template['personality']}

CAPABILITIES:
{chr(10).join(f"- {cap}" for cap in template['capabilities'])}

DECISION AUTHORITY: {template['decision_authority']}

APPROVAL REQUIREMENTS:
You require human approval for:
{chr(10).join(f"- {req}" for req in template['requires_approval_for'])}

Your primary responsibility is to provide expert guidance and make decisions within your authority level.
Always consider the business's best interests and collaborate with other executives when needed.

When making recommendations:
1. Analyze the situation from your area of expertise
2. Consider risks and opportunities
3. Provide clear, actionable recommendations
4. Request approval when required
5. Coordinate with other executives as needed
"""

    # ============================================================================
    # TEAM COORDINATION & DECISION MAKING
    # ============================================================================

    async def execute_advisor_consultation(
        self,
        business_id: str,
        agent_id: str,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a consultation with the elite business advisor

        Args:
            business_id: ID of the business
            agent_id: ID of the advisor agent
            query: Question or situation to analyze
            context: Additional context (financial data, market data, etc.)

        Returns:
            Advisor's analysis and recommendations
        """
        execution_input = {
            "query": query,
            "business_id": business_id,
            "context": context or {},
            "timestamp": datetime.utcnow().isoformat(),
        }

        execution = await self.agent_builder.execute_agent(
            AgentExecutionCreate(
                agent_id=agent_id,
                input_data=execution_input,
                configuration={
                    "max_iterations": 5,
                    "require_approval": False,  # Advisor can provide recommendations without approval
                }
            )
        )

        return {
            "execution_id": execution.execution_id,
            "status": execution.status,
            "recommendations": execution.output_data,
            "requires_approval": self._check_approval_required(execution.output_data),
            "timestamp": execution.started_at,
        }

    async def coordinate_team_decision(
        self,
        business_id: str,
        executive_team: ExecutiveTeam,
        decision_topic: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Coordinate a decision across multiple executives

        Args:
            business_id: ID of the business
            executive_team: The executive team
            decision_topic: Topic requiring decision
            context: Context and data for the decision

        Returns:
            Coordinated decision with input from all relevant executives
        """
        # Determine which executives should be involved
        involved_executives = self._determine_involved_executives(decision_topic, executive_team)

        # Collect input from each executive
        executive_inputs = []
        for exec_config in involved_executives:
            if exec_config.agent_id:
                execution = await self.agent_builder.execute_agent(
                    AgentExecutionCreate(
                        agent_id=exec_config.agent_id,
                        input_data={
                            "decision_topic": decision_topic,
                            "context": context,
                            "role": exec_config.role,
                        },
                        configuration={"max_iterations": 3}
                    )
                )

                executive_inputs.append({
                    "role": exec_config.role,
                    "name": exec_config.name,
                    "recommendation": execution.output_data,
                    "authority_level": exec_config.decision_authority,
                })

        # Synthesize recommendations
        coordinated_decision = self._synthesize_recommendations(
            decision_topic,
            executive_inputs,
            context
        )

        return coordinated_decision

    def _check_approval_required(self, output_data: Dict[str, Any]) -> bool:
        """Check if the recommendation requires human approval"""
        # Check for approval keywords or high-impact decisions
        approval_keywords = [
            "major_investment",
            "strategic_pivot",
            "significant_hiring",
            "major_partnership",
            "legal_commitment",
        ]

        output_str = str(output_data).lower()
        return any(keyword in output_str for keyword in approval_keywords)

    def _determine_involved_executives(
        self,
        decision_topic: str,
        team: ExecutiveTeam
    ) -> List[AgentConfig]:
        """Determine which executives should be involved in a decision"""
        topic_lower = decision_topic.lower()
        involved = []

        # Elite advisor is always involved
        if team.business_advisor:
            involved.append(team.business_advisor)

        # Add specific executives based on topic
        if any(word in topic_lower for word in ["financial", "budget", "investment", "revenue"]):
            if team.cfo:
                involved.append(team.cfo)

        if any(word in topic_lower for word in ["operations", "process", "efficiency", "supply"]):
            if team.coo:
                involved.append(team.coo)

        if any(word in topic_lower for word in ["technology", "technical", "software", "infrastructure"]):
            if team.cto:
                involved.append(team.cto)

        if any(word in topic_lower for word in ["strategic", "growth", "partnership", "vision"]):
            if team.ceo:
                involved.append(team.ceo)

        return involved

    def _synthesize_recommendations(
        self,
        decision_topic: str,
        executive_inputs: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize recommendations from multiple executives"""
        return {
            "decision_topic": decision_topic,
            "executive_inputs": executive_inputs,
            "synthesis": {
                "consensus_points": self._find_consensus(executive_inputs),
                "conflicting_views": self._find_conflicts(executive_inputs),
                "recommended_action": self._determine_recommended_action(executive_inputs),
                "risk_assessment": self._assess_risks(executive_inputs),
                "approval_required": any(
                    inp.get("authority_level") == DecisionAuthority.HIGH
                    for inp in executive_inputs
                ),
            },
            "context": context,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _find_consensus(self, inputs: List[Dict[str, Any]]) -> List[str]:
        """Find consensus points across executive recommendations"""
        # Simplified consensus detection
        return ["All executives agree on the strategic importance of this decision"]

    def _find_conflicts(self, inputs: List[Dict[str, Any]]) -> List[str]:
        """Find conflicting views across executive recommendations"""
        # Simplified conflict detection
        return []

    def _determine_recommended_action(self, inputs: List[Dict[str, Any]]) -> str:
        """Determine the recommended action based on executive input"""
        # Simplified - in production, this would use more sophisticated logic
        if inputs:
            return f"Proceed with caution based on input from {len(inputs)} executives"
        return "Insufficient input for recommendation"

    def _assess_risks(self, inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess risks based on executive input"""
        return {
            "overall_risk_level": "medium",
            "key_risks": ["Market uncertainty", "Resource constraints"],
            "mitigation_strategies": ["Phased approach", "Regular monitoring"],
        }

    # ============================================================================
    # DECISION LOGGING & APPROVAL
    # ============================================================================

    async def create_decision_record(
        self,
        business_id: str,
        agent_role: str,
        decision_type: str,
        description: str,
        amount: Optional[float] = None,
        requires_approval: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ExecutiveDecision:
        """Create a decision record for tracking and approval"""
        decision_id = f"decision_{uuid.uuid4().hex[:12]}"

        decision = ExecutiveDecision(
            id=decision_id,
            business_id=business_id,
            agent_role=agent_role,
            decision_type=decision_type,
            description=description,
            amount=amount,
            requires_approval=requires_approval,
            metadata=metadata or {},
        )

        # Store in database
        decision_dict = decision.model_dump()
        decision_dict["created_at"] = decision_dict["created_at"].isoformat()
        await self.db.insert_one("executive_decisions", decision_dict)

        logger.info(f"Created decision record {decision_id} for business {business_id}")
        return decision

    async def approve_decision(
        self,
        decision_id: str,
        approved_by: str,
        approved: bool
    ) -> ExecutiveDecision:
        """Approve or reject a decision"""
        await self.db.update_one(
            "executive_decisions",
            {"id": decision_id},
            {
                "$set": {
                    "approved": approved,
                    "approved_by": approved_by,
                    "approved_at": datetime.utcnow().isoformat(),
                }
            }
        )

        # Retrieve updated decision
        decision_dict = await self.db.find_one("executive_decisions", {"id": decision_id})
        if decision_dict:
            return ExecutiveDecision(**decision_dict)

        raise ValueError(f"Decision not found: {decision_id}")

    async def get_pending_decisions(self, business_id: str) -> List[ExecutiveDecision]:
        """Get all pending decisions for a business"""
        decisions = await self.db.find_many(
            "executive_decisions",
            {
                "business_id": business_id,
                "requires_approval": True,
                "approved": None,
            },
            limit=100
        )

        return [ExecutiveDecision(**d) for d in decisions]

