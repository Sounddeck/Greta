"""
CRM Service for CPAS4 - Real Database Integration
Comprehensive Customer Relationship Management with AI features
"""

import logging
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
from database import Database
from models.crm_models import (
    Contact, ContactCreate, ContactUpdate,
    Company, CompanyCreate, CompanyUpdate,
    Deal, DealCreate, DealUpdate,
    Activity, ActivityCreate, ActivityUpdate,
    LeadScore, NextAction, CRMStats,
    ContactStatus, DealStage, ActivityType, Priority
)

logger = logging.getLogger(__name__)


class CRMService:
    """
    CRM Service with Real Database Integration
    Manages contacts, companies, deals, and activities
    """
    
    def __init__(self, database: Database):
        self.db = database
        self.contacts_collection = "crm_contacts"
        self.companies_collection = "crm_companies"
        self.deals_collection = "crm_deals"
        self.activities_collection = "crm_activities"
        logger.info("✅ CRM Service initialized with real database")
    
    # ========================================================================
    # CONTACT OPERATIONS
    # ========================================================================
    
    async def create_contact(self, contact_create: ContactCreate) -> Contact:
        """Create a new contact"""
        try:
            contact_id = f"contact_{uuid.uuid4().hex[:12]}"
            
            contact = Contact(
                contact_id=contact_id,
                **contact_create.model_dump(),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Store in database
            contact_dict = contact.model_dump()
            contact_dict["created_at"] = contact.created_at.isoformat()
            contact_dict["updated_at"] = contact.updated_at.isoformat()
            if contact_dict.get("last_contacted"):
                contact_dict["last_contacted"] = contact.last_contacted.isoformat()
            
            await self.db.insert_one(self.contacts_collection, contact_dict)
            
            logger.info(f"✅ Contact created: {contact_id}")
            return contact
            
        except Exception as e:
            logger.error(f"❌ Error creating contact: {e}")
            raise
    
    async def get_contact(self, contact_id: str) -> Optional[Contact]:
        """Get contact by ID"""
        try:
            contact_dict = await self.db.find_one(
                self.contacts_collection,
                {"contact_id": contact_id}
            )
            
            if not contact_dict:
                return None
            
            # Convert ISO strings back to datetime
            if "created_at" in contact_dict and isinstance(contact_dict["created_at"], str):
                contact_dict["created_at"] = datetime.fromisoformat(contact_dict["created_at"])
            if "updated_at" in contact_dict and isinstance(contact_dict["updated_at"], str):
                contact_dict["updated_at"] = datetime.fromisoformat(contact_dict["updated_at"])
            if "last_contacted" in contact_dict and contact_dict["last_contacted"]:
                contact_dict["last_contacted"] = datetime.fromisoformat(contact_dict["last_contacted"])
            
            return Contact(**contact_dict)
            
        except Exception as e:
            logger.error(f"❌ Error getting contact {contact_id}: {e}")
            return None
    
    async def update_contact(self, contact_id: str, contact_update: ContactUpdate) -> Optional[Contact]:
        """Update contact"""
        try:
            update_data = contact_update.model_dump(exclude_unset=True)
            update_data["updated_at"] = datetime.utcnow().isoformat()
            
            success = await self.db.update_one(
                self.contacts_collection,
                {"contact_id": contact_id},
                {"$set": update_data}
            )
            
            if success:
                logger.info(f"✅ Contact updated: {contact_id}")
                return await self.get_contact(contact_id)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error updating contact {contact_id}: {e}")
            return None
    
    async def delete_contact(self, contact_id: str) -> bool:
        """Delete contact"""
        try:
            success = await self.db.delete_one(
                self.contacts_collection,
                {"contact_id": contact_id}
            )
            
            if success:
                logger.info(f"✅ Contact deleted: {contact_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error deleting contact {contact_id}: {e}")
            return False
    
    async def list_contacts(
        self,
        status: Optional[ContactStatus] = None,
        company_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Contact]:
        """List contacts with filters"""
        try:
            filter_dict = {}
            
            if status:
                filter_dict["status"] = status
            if company_id:
                filter_dict["company_id"] = company_id
            if tags:
                filter_dict["tags"] = {"$in": tags}
            
            contact_dicts = await self.db.find_many(
                self.contacts_collection,
                filter_dict,
                limit=limit
            )
            
            contacts = []
            for contact_dict in contact_dicts:
                # Convert ISO strings back to datetime
                if "created_at" in contact_dict and isinstance(contact_dict["created_at"], str):
                    contact_dict["created_at"] = datetime.fromisoformat(contact_dict["created_at"])
                if "updated_at" in contact_dict and isinstance(contact_dict["updated_at"], str):
                    contact_dict["updated_at"] = datetime.fromisoformat(contact_dict["updated_at"])
                if "last_contacted" in contact_dict and contact_dict["last_contacted"]:
                    contact_dict["last_contacted"] = datetime.fromisoformat(contact_dict["last_contacted"])
                
                contacts.append(Contact(**contact_dict))
            
            return contacts
            
        except Exception as e:
            logger.error(f"❌ Error listing contacts: {e}")
            return []
    
    # ========================================================================
    # COMPANY OPERATIONS
    # ========================================================================
    
    async def create_company(self, company_create: CompanyCreate) -> Company:
        """Create a new company"""
        try:
            company_id = f"company_{uuid.uuid4().hex[:12]}"
            
            company = Company(
                company_id=company_id,
                **company_create.model_dump(),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Store in database
            company_dict = company.model_dump()
            company_dict["created_at"] = company.created_at.isoformat()
            company_dict["updated_at"] = company.updated_at.isoformat()
            
            await self.db.insert_one(self.companies_collection, company_dict)
            
            logger.info(f"✅ Company created: {company_id}")
            return company
            
        except Exception as e:
            logger.error(f"❌ Error creating company: {e}")
            raise
    
    async def get_company(self, company_id: str) -> Optional[Company]:
        """Get company by ID"""
        try:
            company_dict = await self.db.find_one(
                self.companies_collection,
                {"company_id": company_id}
            )
            
            if not company_dict:
                return None
            
            # Convert ISO strings back to datetime
            if "created_at" in company_dict and isinstance(company_dict["created_at"], str):
                company_dict["created_at"] = datetime.fromisoformat(company_dict["created_at"])
            if "updated_at" in company_dict and isinstance(company_dict["updated_at"], str):
                company_dict["updated_at"] = datetime.fromisoformat(company_dict["updated_at"])
            
            return Company(**company_dict)
            
        except Exception as e:
            logger.error(f"❌ Error getting company {company_id}: {e}")
            return None
    
    async def update_company(self, company_id: str, company_update: CompanyUpdate) -> Optional[Company]:
        """Update company"""
        try:
            update_data = company_update.model_dump(exclude_unset=True)
            update_data["updated_at"] = datetime.utcnow().isoformat()
            
            success = await self.db.update_one(
                self.companies_collection,
                {"company_id": company_id},
                {"$set": update_data}
            )
            
            if success:
                logger.info(f"✅ Company updated: {company_id}")
                return await self.get_company(company_id)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error updating company {company_id}: {e}")
            return None
    
    async def delete_company(self, company_id: str) -> bool:
        """Delete company"""
        try:
            success = await self.db.delete_one(
                self.companies_collection,
                {"company_id": company_id}
            )
            
            if success:
                logger.info(f"✅ Company deleted: {company_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error deleting company {company_id}: {e}")
            return False
    
    async def list_companies(
        self,
        industry: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Company]:
        """List companies with filters"""
        try:
            filter_dict = {}
            
            if industry:
                filter_dict["industry"] = industry
            if tags:
                filter_dict["tags"] = {"$in": tags}
            
            company_dicts = await self.db.find_many(
                self.companies_collection,
                filter_dict,
                limit=limit
            )
            
            companies = []
            for company_dict in company_dicts:
                # Convert ISO strings back to datetime
                if "created_at" in company_dict and isinstance(company_dict["created_at"], str):
                    company_dict["created_at"] = datetime.fromisoformat(company_dict["created_at"])
                if "updated_at" in company_dict and isinstance(company_dict["updated_at"], str):
                    company_dict["updated_at"] = datetime.fromisoformat(company_dict["updated_at"])
                
                companies.append(Company(**company_dict))
            
            return companies

        except Exception as e:
            logger.error(f"❌ Error listing companies: {e}")
            return []

    # ========================================================================
    # DEAL OPERATIONS
    # ========================================================================

    async def create_deal(self, deal_create: DealCreate) -> Deal:
        """Create a new deal"""
        try:
            deal_id = f"deal_{uuid.uuid4().hex[:12]}"

            deal = Deal(
                deal_id=deal_id,
                **deal_create.model_dump(),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )

            # Store in database
            deal_dict = deal.model_dump()
            deal_dict["created_at"] = deal.created_at.isoformat()
            deal_dict["updated_at"] = deal.updated_at.isoformat()
            if deal_dict.get("expected_close_date"):
                deal_dict["expected_close_date"] = deal.expected_close_date.isoformat()
            if deal_dict.get("actual_close_date"):
                deal_dict["actual_close_date"] = deal.actual_close_date.isoformat()

            await self.db.insert_one(self.deals_collection, deal_dict)

            logger.info(f"✅ Deal created: {deal_id}")
            return deal

        except Exception as e:
            logger.error(f"❌ Error creating deal: {e}")
            raise

    async def get_deal(self, deal_id: str) -> Optional[Deal]:
        """Get deal by ID"""
        try:
            deal_dict = await self.db.find_one(
                self.deals_collection,
                {"deal_id": deal_id}
            )

            if not deal_dict:
                return None

            # Convert ISO strings back to datetime
            if "created_at" in deal_dict and isinstance(deal_dict["created_at"], str):
                deal_dict["created_at"] = datetime.fromisoformat(deal_dict["created_at"])
            if "updated_at" in deal_dict and isinstance(deal_dict["updated_at"], str):
                deal_dict["updated_at"] = datetime.fromisoformat(deal_dict["updated_at"])
            if "expected_close_date" in deal_dict and deal_dict["expected_close_date"]:
                deal_dict["expected_close_date"] = datetime.fromisoformat(deal_dict["expected_close_date"])
            if "actual_close_date" in deal_dict and deal_dict["actual_close_date"]:
                deal_dict["actual_close_date"] = datetime.fromisoformat(deal_dict["actual_close_date"])

            return Deal(**deal_dict)

        except Exception as e:
            logger.error(f"❌ Error getting deal {deal_id}: {e}")
            return None

    async def update_deal(self, deal_id: str, deal_update: DealUpdate) -> Optional[Deal]:
        """Update deal"""
        try:
            update_data = deal_update.model_dump(exclude_unset=True)
            update_data["updated_at"] = datetime.utcnow().isoformat()

            # Handle datetime fields
            if "expected_close_date" in update_data and update_data["expected_close_date"]:
                update_data["expected_close_date"] = update_data["expected_close_date"].isoformat()
            if "actual_close_date" in update_data and update_data["actual_close_date"]:
                update_data["actual_close_date"] = update_data["actual_close_date"].isoformat()

            success = await self.db.update_one(
                self.deals_collection,
                {"deal_id": deal_id},
                {"$set": update_data}
            )

            if success:
                logger.info(f"✅ Deal updated: {deal_id}")
                return await self.get_deal(deal_id)

            return None

        except Exception as e:
            logger.error(f"❌ Error updating deal {deal_id}: {e}")
            return None

    async def delete_deal(self, deal_id: str) -> bool:
        """Delete deal"""
        try:
            success = await self.db.delete_one(
                self.deals_collection,
                {"deal_id": deal_id}
            )

            if success:
                logger.info(f"✅ Deal deleted: {deal_id}")

            return success

        except Exception as e:
            logger.error(f"❌ Error deleting deal {deal_id}: {e}")
            return False

    async def list_deals(
        self,
        stage: Optional[DealStage] = None,
        company_id: Optional[str] = None,
        contact_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Deal]:
        """List deals with filters"""
        try:
            filter_dict = {}

            if stage:
                filter_dict["stage"] = stage
            if company_id:
                filter_dict["company_id"] = company_id
            if contact_id:
                filter_dict["contact_id"] = contact_id

            deal_dicts = await self.db.find_many(
                self.deals_collection,
                filter_dict,
                limit=limit
            )

            deals = []
            for deal_dict in deal_dicts:
                # Convert ISO strings back to datetime
                if "created_at" in deal_dict and isinstance(deal_dict["created_at"], str):
                    deal_dict["created_at"] = datetime.fromisoformat(deal_dict["created_at"])
                if "updated_at" in deal_dict and isinstance(deal_dict["updated_at"], str):
                    deal_dict["updated_at"] = datetime.fromisoformat(deal_dict["updated_at"])
                if "expected_close_date" in deal_dict and deal_dict["expected_close_date"]:
                    deal_dict["expected_close_date"] = datetime.fromisoformat(deal_dict["expected_close_date"])
                if "actual_close_date" in deal_dict and deal_dict["actual_close_date"]:
                    deal_dict["actual_close_date"] = datetime.fromisoformat(deal_dict["actual_close_date"])

                deals.append(Deal(**deal_dict))

            return deals

        except Exception as e:
            logger.error(f"❌ Error listing deals: {e}")
            return []

    # ========================================================================
    # ACTIVITY OPERATIONS
    # ========================================================================

    async def create_activity(self, activity_create: ActivityCreate) -> Activity:
        """Create a new activity"""
        try:
            activity_id = f"activity_{uuid.uuid4().hex[:12]}"

            activity = Activity(
                activity_id=activity_id,
                **activity_create.model_dump(),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )

            # Store in database
            activity_dict = activity.model_dump()
            activity_dict["created_at"] = activity.created_at.isoformat()
            activity_dict["updated_at"] = activity.updated_at.isoformat()
            if activity_dict.get("due_date"):
                activity_dict["due_date"] = activity.due_date.isoformat()
            if activity_dict.get("completed_at"):
                activity_dict["completed_at"] = activity.completed_at.isoformat()

            await self.db.insert_one(self.activities_collection, activity_dict)

            logger.info(f"✅ Activity created: {activity_id}")
            return activity

        except Exception as e:
            logger.error(f"❌ Error creating activity: {e}")
            raise

    async def get_activity(self, activity_id: str) -> Optional[Activity]:
        """Get activity by ID"""
        try:
            activity_dict = await self.db.find_one(
                self.activities_collection,
                {"activity_id": activity_id}
            )

            if not activity_dict:
                return None

            # Convert ISO strings back to datetime
            if "created_at" in activity_dict and isinstance(activity_dict["created_at"], str):
                activity_dict["created_at"] = datetime.fromisoformat(activity_dict["created_at"])
            if "updated_at" in activity_dict and isinstance(activity_dict["updated_at"], str):
                activity_dict["updated_at"] = datetime.fromisoformat(activity_dict["updated_at"])
            if "due_date" in activity_dict and activity_dict["due_date"]:
                activity_dict["due_date"] = datetime.fromisoformat(activity_dict["due_date"])
            if "completed_at" in activity_dict and activity_dict["completed_at"]:
                activity_dict["completed_at"] = datetime.fromisoformat(activity_dict["completed_at"])

            return Activity(**activity_dict)

        except Exception as e:
            logger.error(f"❌ Error getting activity {activity_id}: {e}")
            return None

    async def update_activity(self, activity_id: str, activity_update: ActivityUpdate) -> Optional[Activity]:
        """Update activity"""
        try:
            update_data = activity_update.model_dump(exclude_unset=True)
            update_data["updated_at"] = datetime.utcnow().isoformat()

            # Handle datetime fields
            if "due_date" in update_data and update_data["due_date"]:
                update_data["due_date"] = update_data["due_date"].isoformat()
            if "completed_at" in update_data and update_data["completed_at"]:
                update_data["completed_at"] = update_data["completed_at"].isoformat()

            # Auto-set completed_at when marking as completed
            if "completed" in update_data and update_data["completed"] and "completed_at" not in update_data:
                update_data["completed_at"] = datetime.utcnow().isoformat()

            success = await self.db.update_one(
                self.activities_collection,
                {"activity_id": activity_id},
                {"$set": update_data}
            )

            if success:
                logger.info(f"✅ Activity updated: {activity_id}")
                return await self.get_activity(activity_id)

            return None

        except Exception as e:
            logger.error(f"❌ Error updating activity {activity_id}: {e}")
            return None

    async def delete_activity(self, activity_id: str) -> bool:
        """Delete activity"""
        try:
            success = await self.db.delete_one(
                self.activities_collection,
                {"activity_id": activity_id}
            )

            if success:
                logger.info(f"✅ Activity deleted: {activity_id}")

            return success

        except Exception as e:
            logger.error(f"❌ Error deleting activity {activity_id}: {e}")
            return False

    async def list_activities(
        self,
        activity_type: Optional[ActivityType] = None,
        contact_id: Optional[str] = None,
        company_id: Optional[str] = None,
        deal_id: Optional[str] = None,
        completed: Optional[bool] = None,
        limit: int = 100
    ) -> List[Activity]:
        """List activities with filters"""
        try:
            filter_dict = {}

            if activity_type:
                filter_dict["activity_type"] = activity_type
            if contact_id:
                filter_dict["contact_id"] = contact_id
            if company_id:
                filter_dict["company_id"] = company_id
            if deal_id:
                filter_dict["deal_id"] = deal_id
            if completed is not None:
                filter_dict["completed"] = completed

            activity_dicts = await self.db.find_many(
                self.activities_collection,
                filter_dict,
                limit=limit
            )

            activities = []
            for activity_dict in activity_dicts:
                # Convert ISO strings back to datetime
                if "created_at" in activity_dict and isinstance(activity_dict["created_at"], str):
                    activity_dict["created_at"] = datetime.fromisoformat(activity_dict["created_at"])
                if "updated_at" in activity_dict and isinstance(activity_dict["updated_at"], str):
                    activity_dict["updated_at"] = datetime.fromisoformat(activity_dict["updated_at"])
                if "due_date" in activity_dict and activity_dict["due_date"]:
                    activity_dict["due_date"] = datetime.fromisoformat(activity_dict["due_date"])
                if "completed_at" in activity_dict and activity_dict["completed_at"]:
                    activity_dict["completed_at"] = datetime.fromisoformat(activity_dict["completed_at"])

                activities.append(Activity(**activity_dict))

            return activities

        except Exception as e:
            logger.error(f"❌ Error listing activities: {e}")
            return []

    # ========================================================================
    # AI-POWERED FEATURES
    # ========================================================================

    async def calculate_lead_score(self, contact_id: str) -> LeadScore:
        """Calculate AI-powered lead score for a contact"""
        try:
            contact = await self.get_contact(contact_id)
            if not contact:
                raise ValueError(f"Contact {contact_id} not found")

            # Get related data
            activities = await self.list_activities(contact_id=contact_id, limit=50)
            deals = await self.list_deals(contact_id=contact_id, limit=10)

            # Calculate score factors
            factors = {}

            # Factor 1: Contact completeness (0-20 points)
            completeness = 0
            if contact.email:
                completeness += 5
            if contact.phone:
                completeness += 5
            if contact.title:
                completeness += 5
            if contact.company_id:
                completeness += 5
            factors["profile_completeness"] = completeness

            # Factor 2: Engagement level (0-30 points)
            engagement = min(len(activities) * 3, 30)
            factors["engagement_level"] = engagement

            # Factor 3: Deal value (0-25 points)
            total_deal_value = sum(deal.value for deal in deals if deal.stage not in [DealStage.CLOSED_LOST])
            deal_score = min(total_deal_value / 10000 * 25, 25)
            factors["deal_potential"] = deal_score

            # Factor 4: Recency (0-15 points)
            recency = 0
            if contact.last_contacted:
                days_since_contact = (datetime.utcnow() - contact.last_contacted).days
                if days_since_contact < 7:
                    recency = 15
                elif days_since_contact < 30:
                    recency = 10
                elif days_since_contact < 90:
                    recency = 5
            factors["recency"] = recency

            # Factor 5: Status (0-10 points)
            status_scores = {
                ContactStatus.CUSTOMER: 10,
                ContactStatus.PROSPECT: 7,
                ContactStatus.LEAD: 5,
                ContactStatus.INACTIVE: 2,
                ContactStatus.LOST: 0
            }
            factors["status_value"] = status_scores.get(contact.status, 0)

            # Calculate total score
            total_score = sum(factors.values())

            # Generate recommendations
            recommendations = []
            if factors["recency"] < 10:
                recommendations.append("Schedule follow-up call - no recent contact")
            if factors["engagement_level"] < 15:
                recommendations.append("Increase engagement with personalized content")
            if factors["deal_potential"] > 15 and contact.status == ContactStatus.LEAD:
                recommendations.append("Upgrade to prospect status - high deal potential")
            if factors["profile_completeness"] < 15:
                recommendations.append("Complete contact profile for better insights")

            # Update contact with new score
            await self.update_contact(contact_id, ContactUpdate(lead_score=total_score))

            return LeadScore(
                contact_id=contact_id,
                score=total_score,
                factors=factors,
                confidence=0.85,
                recommendations=recommendations,
                calculated_at=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"❌ Error calculating lead score for {contact_id}: {e}")
            raise

    async def suggest_next_action(self, entity_type: str, entity_id: str) -> NextAction:
        """Suggest AI-powered next action for an entity"""
        try:
            if entity_type == "contact":
                contact = await self.get_contact(entity_id)
                if not contact:
                    raise ValueError(f"Contact {entity_id} not found")

                # Get recent activities
                activities = await self.list_activities(contact_id=entity_id, limit=10)
                recent_activities = [a for a in activities if (datetime.utcnow() - a.created_at).days < 30]

                # Determine next action
                if not recent_activities:
                    return NextAction(
                        entity_type="contact",
                        entity_id=entity_id,
                        action_type=ActivityType.CALL,
                        subject=f"Initial outreach to {contact.first_name} {contact.last_name}",
                        description="No recent activity detected. Schedule an introductory call to establish relationship.",
                        priority=Priority.HIGH,
                        reasoning="Contact has no recent engagement. Initial outreach is critical.",
                        confidence=0.9
                    )
                elif contact.status == ContactStatus.LEAD and len(recent_activities) > 3:
                    return NextAction(
                        entity_type="contact",
                        entity_id=entity_id,
                        action_type=ActivityType.MEETING,
                        subject=f"Product demo with {contact.first_name} {contact.last_name}",
                        description="Contact shows high engagement. Schedule product demonstration.",
                        priority=Priority.HIGH,
                        reasoning="Multiple recent interactions indicate strong interest. Time for demo.",
                        confidence=0.85
                    )
                else:
                    return NextAction(
                        entity_type="contact",
                        entity_id=entity_id,
                        action_type=ActivityType.EMAIL,
                        subject=f"Follow-up with {contact.first_name} {contact.last_name}",
                        description="Send personalized follow-up email with relevant content.",
                        priority=Priority.MEDIUM,
                        reasoning="Maintain engagement with regular touchpoints.",
                        confidence=0.75
                    )

            elif entity_type == "deal":
                deal = await self.get_deal(entity_id)
                if not deal:
                    raise ValueError(f"Deal {entity_id} not found")

                # Suggest action based on stage
                if deal.stage == DealStage.LEAD:
                    return NextAction(
                        entity_type="deal",
                        entity_id=entity_id,
                        action_type=ActivityType.CALL,
                        subject=f"Qualify deal: {deal.title}",
                        description="Schedule qualification call to assess fit and budget.",
                        priority=Priority.HIGH,
                        reasoning="New deal needs qualification to move forward.",
                        confidence=0.9
                    )
                elif deal.stage == DealStage.PROPOSAL:
                    return NextAction(
                        entity_type="deal",
                        entity_id=entity_id,
                        action_type=ActivityType.MEETING,
                        subject=f"Proposal review: {deal.title}",
                        description="Schedule meeting to review and discuss proposal details.",
                        priority=Priority.URGENT,
                        reasoning="Proposal stage requires active engagement to close.",
                        confidence=0.95
                    )
                else:
                    return NextAction(
                        entity_type="deal",
                        entity_id=entity_id,
                        action_type=ActivityType.EMAIL,
                        subject=f"Update on {deal.title}",
                        description="Send status update and next steps.",
                        priority=Priority.MEDIUM,
                        reasoning="Regular communication maintains deal momentum.",
                        confidence=0.7
                    )

            else:
                raise ValueError(f"Unsupported entity type: {entity_type}")

        except Exception as e:
            logger.error(f"❌ Error suggesting next action for {entity_type} {entity_id}: {e}")
            raise

    # ========================================================================
    # STATISTICS
    # ========================================================================

    async def get_crm_stats(self) -> CRMStats:
        """Get comprehensive CRM statistics"""
        try:
            # Get all data
            all_contacts = await self.list_contacts(limit=10000)
            all_companies = await self.list_companies(limit=10000)
            all_deals = await self.list_deals(limit=10000)
            all_activities = await self.list_activities(limit=10000)

            # Count by status
            contacts_by_status = {}
            for status in ContactStatus:
                contacts_by_status[status.value] = sum(1 for c in all_contacts if c.status == status)

            # Count deals by stage
            deals_by_stage = {}
            for stage in DealStage:
                deals_by_stage[stage.value] = sum(1 for d in all_deals if d.stage == stage)

            # Calculate pipeline value
            total_pipeline_value = sum(
                d.value for d in all_deals
                if d.stage not in [DealStage.CLOSED_WON, DealStage.CLOSED_LOST]
            )

            # Calculate won value
            total_won_value = sum(
                d.value for d in all_deals
                if d.stage == DealStage.CLOSED_WON
            )

            # Calculate average deal size
            won_deals = [d for d in all_deals if d.stage == DealStage.CLOSED_WON]
            average_deal_size = total_won_value / len(won_deals) if won_deals else 0.0

            # Calculate win rate
            closed_deals = [d for d in all_deals if d.stage in [DealStage.CLOSED_WON, DealStage.CLOSED_LOST]]
            win_rate = (len(won_deals) / len(closed_deals) * 100) if closed_deals else 0.0

            # Count activities by type
            activities_by_type = {}
            for activity_type in ActivityType:
                activities_by_type[activity_type.value] = sum(
                    1 for a in all_activities if a.activity_type == activity_type
                )

            return CRMStats(
                total_contacts=len(all_contacts),
                total_companies=len(all_companies),
                total_deals=len(all_deals),
                total_activities=len(all_activities),
                contacts_by_status=contacts_by_status,
                deals_by_stage=deals_by_stage,
                total_pipeline_value=total_pipeline_value,
                total_won_value=total_won_value,
                average_deal_size=average_deal_size,
                win_rate=win_rate,
                activities_by_type=activities_by_type
            )

        except Exception as e:
            logger.error(f"❌ Error getting CRM stats: {e}")
            raise

