"""
Business Service for Multi-Business Management

Handles CRUD operations and business logic for managing multiple autonomous businesses.
Implements multi-tenant data isolation and business lifecycle management.
"""

import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from database import Database
from models.business_models import (
    Business,
    BusinessCreate,
    BusinessUpdate,
    BusinessStatus,
    ExecutiveTeam,
    BusinessStats,
    ExecutiveDecision,
    AgentConfig,
    DecisionAuthority
)

logger = logging.getLogger(__name__)


class BusinessService:
    """Service for managing businesses in multi-tenant architecture"""
    
    def __init__(self, database: Database):
        self.db = database
        self.businesses_collection = "businesses"
        self.decisions_collection = "executive_decisions"
        logger.info("✅ Business Service initialized with real database")
    
    async def create_business(self, business_data: BusinessCreate, owner_id: str) -> Business:
        """
        Create a new business with initial configuration
        
        Args:
            business_data: Business creation data
            owner_id: ID of the user creating the business
            
        Returns:
            Created Business object
        """
        try:
            # Generate workspace path if not provided
            workspace_path = business_data.workspace_path
            if not workspace_path:
                safe_name = business_data.name.lower().replace(" ", "-").replace("/", "-")
                workspace_path = f"/businesses/{safe_name}"
            
            # Create business document
            business = Business(
                name=business_data.name,
                description=business_data.description,
                industry=business_data.industry,
                owner_id=owner_id,
                workspace_path=workspace_path,
                isolated_data=business_data.isolated_data,
                tags=business_data.tags,
                status=BusinessStatus.ACTIVE
            )
            
            # Note: Executive team creation is handled separately via the executive team service
            # The frontend should call POST /api/v1/executive-teams/businesses/{business_id}/create-team
            # after the business is created if enable_executive_team is true

            # Set autonomous mode
            business.approval_workflow.autonomous_mode = business_data.autonomous_mode
            
            # Convert to dict for database insertion
            business_dict = business.model_dump(by_alias=True, exclude={"id"})
            
            # Convert datetime objects to ISO strings
            if "created_at" in business_dict:
                business_dict["created_at"] = business_dict["created_at"].isoformat()
            if "updated_at" in business_dict:
                business_dict["updated_at"] = business_dict["updated_at"].isoformat()
            
            # Insert into database
            business_id = await self.db.insert_one(self.businesses_collection, business_dict)
            business.id = business_id
            
            # Create workspace directory
            try:
                os.makedirs(workspace_path, exist_ok=True)
                logger.info(f"✅ Created workspace directory: {workspace_path}")
            except Exception as e:
                logger.warning(f"⚠️ Could not create workspace directory: {e}")
            
            logger.info(f"✅ Business created: {business.name} ({business_id})")
            return business
            
        except Exception as e:
            logger.error(f"❌ Error creating business: {e}")
            raise
    
    async def get_business(self, business_id: str) -> Optional[Business]:
        """Get a business by ID"""
        try:
            business_dict = await self.db.find_one(
                self.businesses_collection,
                {"_id": business_id}
            )
            
            if not business_dict:
                return None
            
            # Convert ISO strings back to datetime
            if "created_at" in business_dict and isinstance(business_dict["created_at"], str):
                business_dict["created_at"] = datetime.fromisoformat(business_dict["created_at"])
            if "updated_at" in business_dict and isinstance(business_dict["updated_at"], str):
                business_dict["updated_at"] = datetime.fromisoformat(business_dict["updated_at"])
            
            return Business(**business_dict)
            
        except Exception as e:
            logger.error(f"❌ Error getting business {business_id}: {e}")
            return None
    
    async def list_businesses(
        self,
        owner_id: Optional[str] = None,
        status: Optional[BusinessStatus] = None,
        skip: int = 0,
        limit: int = 50
    ) -> tuple[List[Business], int]:
        """
        List businesses with optional filtering
        
        Returns:
            Tuple of (businesses list, total count)
        """
        try:
            query = {}
            if owner_id:
                query["owner_id"] = owner_id
            if status:
                query["status"] = status
            
            # Get businesses
            business_dicts = await self.db.find_many(
                self.businesses_collection,
                query,
                limit=limit
            )
            
            businesses = []
            for business_dict in business_dicts:
                # Convert ISO strings back to datetime
                if "created_at" in business_dict and isinstance(business_dict["created_at"], str):
                    business_dict["created_at"] = datetime.fromisoformat(business_dict["created_at"])
                if "updated_at" in business_dict and isinstance(business_dict["updated_at"], str):
                    business_dict["updated_at"] = datetime.fromisoformat(business_dict["updated_at"])
                
                businesses.append(Business(**business_dict))
            
            total = len(businesses)
            return businesses, total
            
        except Exception as e:
            logger.error(f"❌ Error listing businesses: {e}")
            return [], 0
    
    async def update_business(self, business_id: str, update_data: BusinessUpdate) -> Optional[Business]:
        """Update a business"""
        try:
            # Build update document
            update_dict = update_data.model_dump(exclude_unset=True, exclude_none=True)
            if not update_dict:
                return await self.get_business(business_id)
            
            update_dict["updated_at"] = datetime.utcnow().isoformat()
            
            # Update in database
            success = await self.db.update_one(
                self.businesses_collection,
                {"_id": business_id},
                update_dict
            )
            
            if success:
                return await self.get_business(business_id)
            return None
            
        except Exception as e:
            logger.error(f"❌ Error updating business {business_id}: {e}")
            return None
    
    async def delete_business(self, business_id: str) -> bool:
        """Delete a business (soft delete by setting status to archived)"""
        try:
            success = await self.db.update_one(
                self.businesses_collection,
                {"_id": business_id},
                {
                    "status": BusinessStatus.ARCHIVED,
                    "updated_at": datetime.utcnow().isoformat()
                }
            )
            
            if success:
                logger.info(f"✅ Business archived: {business_id}")
            return success
            
        except Exception as e:
            logger.error(f"❌ Error deleting business {business_id}: {e}")
            return False
    
    async def get_business_stats(self, business_id: str) -> BusinessStats:
        """Get statistics for a business"""
        try:
            business = await self.get_business(business_id)
            if not business:
                return BusinessStats()
            
            stats = BusinessStats()
            
            # Get CRM stats if CRM is configured
            if business.infrastructure.crm_id:
                # TODO: Query CRM database for stats
                pass
            
            # Get ATS stats if ATS is configured
            if business.infrastructure.ats_id:
                # TODO: Query ATS database for stats
                pass
            
            # Get executive decisions stats
            decisions = await self.db.find_many(
                self.decisions_collection,
                {"business_id": business_id},
                limit=1000
            )
            
            stats.decisions_made = len(decisions)
            stats.decisions_pending_approval = len([
                d for d in decisions
                if d.get("requires_approval") and d.get("approved") is None
            ])
            
            # Infrastructure status
            stats.google_workspace_active = (
                business.infrastructure.google_workspace is not None and
                business.infrastructure.google_workspace.setup_completed
            )
            stats.crm_active = business.infrastructure.crm_id is not None
            stats.ats_active = business.infrastructure.ats_id is not None
            
            return stats

        except Exception as e:
            logger.error(f"❌ Error getting business stats {business_id}: {e}")
            return BusinessStats()

    async def create_executive_decision(
        self,
        business_id: str,
        agent_role: str,
        decision_type: str,
        description: str,
        amount: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ExecutiveDecision:
        """Create a new executive decision that may require approval"""
        try:
            business = await self.get_business(business_id)
            if not business:
                raise ValueError(f"Business {business_id} not found")

            # Check if approval is required
            requires_approval = (
                not business.approval_workflow.autonomous_mode or
                decision_type in business.approval_workflow.approval_required_for or
                (amount and business.approval_workflow.approval_threshold_amount and
                 amount > business.approval_workflow.approval_threshold_amount)
            )

            decision = ExecutiveDecision(
                business_id=business_id,
                agent_role=agent_role,
                decision_type=decision_type,
                description=description,
                amount=amount,
                requires_approval=requires_approval,
                metadata=metadata or {}
            )

            decision_dict = decision.model_dump(by_alias=True, exclude={"id"})

            # Convert datetime objects to ISO strings
            if "created_at" in decision_dict:
                decision_dict["created_at"] = decision_dict["created_at"].isoformat()
            if "approved_at" in decision_dict and decision_dict["approved_at"]:
                decision_dict["approved_at"] = decision_dict["approved_at"].isoformat()

            decision_id = await self.db.insert_one(self.decisions_collection, decision_dict)
            decision.id = decision_id

            logger.info(f"✅ Executive decision created: {decision_id}")
            return decision

        except Exception as e:
            logger.error(f"❌ Error creating executive decision: {e}")
            raise

    async def approve_decision(
        self,
        decision_id: str,
        approved_by: str,
        approved: bool
    ) -> Optional[ExecutiveDecision]:
        """Approve or reject an executive decision"""
        try:
            success = await self.db.update_one(
                self.decisions_collection,
                {"_id": decision_id},
                {
                    "approved": approved,
                    "approved_by": approved_by,
                    "approved_at": datetime.utcnow().isoformat()
                }
            )

            if success:
                decision_dict = await self.db.find_one(
                    self.decisions_collection,
                    {"_id": decision_id}
                )

                if decision_dict:
                    # Convert ISO strings back to datetime
                    if "created_at" in decision_dict and isinstance(decision_dict["created_at"], str):
                        decision_dict["created_at"] = datetime.fromisoformat(decision_dict["created_at"])
                    if "approved_at" in decision_dict and decision_dict["approved_at"]:
                        decision_dict["approved_at"] = datetime.fromisoformat(decision_dict["approved_at"])

                    logger.info(f"✅ Decision {'approved' if approved else 'rejected'}: {decision_id}")
                    return ExecutiveDecision(**decision_dict)

            return None

        except Exception as e:
            logger.error(f"❌ Error approving decision {decision_id}: {e}")
            return None

    async def get_pending_decisions(self, business_id: str) -> List[ExecutiveDecision]:
        """Get all pending decisions for a business"""
        try:
            decision_dicts = await self.db.find_many(
                self.decisions_collection,
                {
                    "business_id": business_id,
                    "requires_approval": True,
                    "approved": None
                },
                limit=100
            )

            decisions = []
            for decision_dict in decision_dicts:
                # Convert ISO strings back to datetime
                if "created_at" in decision_dict and isinstance(decision_dict["created_at"], str):
                    decision_dict["created_at"] = datetime.fromisoformat(decision_dict["created_at"])
                if "approved_at" in decision_dict and decision_dict["approved_at"]:
                    decision_dict["approved_at"] = datetime.fromisoformat(decision_dict["approved_at"])

                decisions.append(ExecutiveDecision(**decision_dict))

            return decisions

        except Exception as e:
            logger.error(f"❌ Error getting pending decisions for business {business_id}: {e}")
            return []

    def _create_default_executive_team(self) -> ExecutiveTeam:
        """Create default executive team configuration"""
        return ExecutiveTeam(
            business_advisor=AgentConfig(
                name="Business Advisor",
                role="Strategic advisor and mentor",
                capabilities=["market_analysis", "competitive_intelligence", "strategic_planning", "risk_assessment"],
                tools=["tavily_search", "exa_search", "osint_tools", "financial_analysis"],
                personality="Wise, experienced, strategic thinker",
                decision_authority=DecisionAuthority.ADVISORY_ONLY
            ),
            ceo=AgentConfig(
                name="CEO",
                role="Chief Executive Officer",
                capabilities=["strategic_decision_making", "team_coordination", "stakeholder_communication", "vision_setting"],
                tools=["gmail", "calendar", "drive", "crm", "all_research_tools"],
                personality="Visionary, decisive, inspiring leader",
                decision_authority=DecisionAuthority.HIGH,
                requires_approval_for=["major_investments", "strategic_pivots"]
            ),
            cfo=AgentConfig(
                name="CFO",
                role="Chief Financial Officer",
                capabilities=["financial_planning", "budgeting", "financial_reporting", "investment_analysis"],
                tools=["google_sheets", "financial_apis", "accounting_software"],
                personality="Analytical, detail-oriented, risk-aware",
                decision_authority=DecisionAuthority.MEDIUM,
                requires_approval_for=["large_expenditures", "financial_commitments"]
            ),
            coo=AgentConfig(
                name="COO",
                role="Chief Operating Officer",
                capabilities=["operations_management", "process_optimization", "resource_allocation", "performance_monitoring"],
                tools=["project_management", "automation_tools", "analytics"],
                personality="Efficient, systematic, results-driven",
                decision_authority=DecisionAuthority.MEDIUM
            ),
            cto=AgentConfig(
                name="CTO",
                role="Chief Technology Officer",
                capabilities=["technology_strategy", "architecture_design", "technical_leadership", "innovation"],
                tools=["github", "coding_tools", "tech_research", "deployment_tools"],
                personality="Innovative, technical, forward-thinking",
                decision_authority=DecisionAuthority.MEDIUM,
                requires_approval_for=["major_tech_decisions", "architecture_changes"]
            )
        )

