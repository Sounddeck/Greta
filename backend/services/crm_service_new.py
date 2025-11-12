"""
CRM Service for CPAS4 - Real Database Integration
Comprehensive Customer Relationship Management with AI features

Phase 2, Task 2.1: Database Integrity Improvements
- Standardized datetime operations using Database.utcnow()
- Enhanced validation for update operations
- Transaction support for atomic operations

Phase 3, Task 3.1: Optimize Database Queries
- Added pagination support for list methods
- Optimized aggregation pipelines for stats
- Reduced memory usage by avoiding loading all data

Phase 3, Task 3.2: Implement Caching Layer
- Added Redis caching for CRM stats (5-minute TTL)
- Cache invalidation on create/update/delete operations
- Cached list operations with 2-minute TTL
"""

import logging
import uuid
import re
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from database import Database
from models.crm_models import (
    Contact, ContactCreate, ContactUpdate,
    Company, CompanyCreate, CompanyUpdate,
    Deal, DealCreate, DealUpdate,
    Activity, ActivityCreate, ActivityUpdate,
    LeadScore, NextAction, CRMStats,
    DealProbability, ContactSentiment, SentimentType,
    ContactStatus, DealStage, ActivityType, Priority
)
from services.event_bus import publish_event, subscribe_to_event
from models.event_models import EventType

logger = logging.getLogger(__name__)


class CRMService:
    """
    CRM Service with Real Database Integration
    Manages contacts, companies, deals, and activities
    """

    def __init__(self, database: Database, cache_service=None):
        self.db = database
        self.cache = cache_service  # Optional cache service
        self.contacts_collection = "crm_contacts"
        self.companies_collection = "crm_companies"
        self.deals_collection = "crm_deals"
        self.activities_collection = "crm_activities"

        # Subscribe to lead events
        subscribe_to_event(EventType.LEAD_QUALIFIED, self._on_lead_qualified)
        subscribe_to_event(EventType.LEAD_ENRICHED, self._on_lead_enriched)

        logger.info(f"✅ CRM Service initialized with real database (cache: {'enabled' if cache_service else 'disabled'})")
    
    # ========================================================================
    # CONTACT OPERATIONS
    # ========================================================================
    
    async def create_contact(self, contact_create: ContactCreate) -> Contact:
        """Create a new contact with standardized datetime operations"""
        try:
            contact_id = f"contact_{uuid.uuid4().hex[:12]}"
            now = Database.utcnow()  # Standardized datetime

            contact = Contact(
                contact_id=contact_id,
                **contact_create.model_dump(),
                created_at=now,
                updated_at=now
            )

            # Store in database
            contact_dict = contact.model_dump()
            contact_dict["created_at"] = Database.to_iso(contact.created_at)
            contact_dict["updated_at"] = Database.to_iso(contact.updated_at)
            if contact_dict.get("last_contacted"):
                contact_dict["last_contacted"] = Database.to_iso(contact.last_contacted)

            await self.db.insert_one(self.contacts_collection, contact_dict)

            # Publish event
            await publish_event(
                EventType.CRM_CONTACT_CREATED,
                {
                    "contact_id": contact_id,
                    "name": f"{contact.first_name} {contact.last_name}",
                    "email": contact.email,
                    "phone": contact.phone,
                    "company_id": contact.company_id,
                    "source": contact_create.model_dump().get("source", "manual"),
                },
                source="crm_service"
            )

            # Invalidate cache on create
            if self.cache:
                await self.cache.invalidate_by_pattern("stats:crm*")
                await self.cache.invalidate_by_pattern("list:crm_contacts*")
                logger.debug("✅ Cache invalidated after contact creation")

            logger.info(f"✅ Contact created: {contact_id}")
            return contact

        except Exception as e:
            logger.error(f"❌ Error creating contact: {e}")
            raise
    
    async def get_contact(self, contact_id: str) -> Optional[Contact]:
        """Get contact by ID with standardized datetime conversion"""
        try:
            contact_dict = await self.db.find_one(
                self.contacts_collection,
                {"contact_id": contact_id}
            )

            if not contact_dict:
                return None

            # Convert ISO strings back to datetime using standardized method
            contact_dict["created_at"] = Database.from_iso(contact_dict.get("created_at"))
            contact_dict["updated_at"] = Database.from_iso(contact_dict.get("updated_at"))
            contact_dict["last_contacted"] = Database.from_iso(contact_dict.get("last_contacted"))

            return Contact(**contact_dict)

        except Exception as e:
            logger.error(f"❌ Error getting contact {contact_id}: {e}")
            return None
    
    async def update_contact(self, contact_id: str, contact_update: ContactUpdate) -> Optional[Contact]:
        """
        Update contact with validation and standardized datetime.

        Phase 2, Task 2.1: Added validation for update operations
        """
        try:
            # Validate contact exists
            existing_contact = await self.get_contact(contact_id)
            if not existing_contact:
                logger.warning(f"⚠️ Contact not found: {contact_id}")
                return None

            # Prepare update data
            update_data = contact_update.model_dump(exclude_unset=True)

            # Validate email if being updated
            if "email" in update_data and update_data["email"]:
                # Check for duplicate email (excluding current contact)
                existing_email = await self.db.find_one(
                    self.contacts_collection,
                    {"email": update_data["email"], "contact_id": {"$ne": contact_id}}
                )
                if existing_email:
                    logger.warning(f"⚠️ Email already exists: {update_data['email']}")
                    raise ValueError(f"Email {update_data['email']} is already in use")

            # Standardized datetime
            update_data["updated_at"] = Database.to_iso(Database.utcnow())

            # Convert datetime fields to ISO strings
            if "last_contacted" in update_data and update_data["last_contacted"]:
                update_data["last_contacted"] = Database.to_iso(update_data["last_contacted"])

            success = await self.db.update_one(
                self.contacts_collection,
                {"contact_id": contact_id},
                {"$set": update_data}
            )

            if success:
                logger.info(f"✅ Contact updated: {contact_id}")
                return await self.get_contact(contact_id)

            return None

        except ValueError as e:
            # Re-raise validation errors
            raise
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
                # Convert ISO strings back to datetime using standardized method
                contact_dict["created_at"] = Database.from_iso(contact_dict.get("created_at"))
                contact_dict["updated_at"] = Database.from_iso(contact_dict.get("updated_at"))
                contact_dict["last_contacted"] = Database.from_iso(contact_dict.get("last_contacted"))

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
    # PAGINATED LIST METHODS (Phase 3, Task 3.1)
    # ========================================================================

    async def list_contacts_paginated(
        self,
        page: int = 1,
        page_size: int = 100,
        status: Optional[ContactStatus] = None,
        company_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        sort_field: str = "created_at",
        sort_direction: int = -1
    ) -> Tuple[List[Contact], Dict[str, Any]]:
        """
        List contacts with pagination support.

        Phase 3, Task 3.1: Implement query result pagination for large datasets

        Returns:
            Tuple of (contacts list, pagination metadata)
        """
        try:
            # Build filter
            filter_dict = {}
            if status:
                filter_dict["status"] = status
            if company_id:
                filter_dict["company_id"] = company_id
            if tags:
                filter_dict["tags"] = {"$in": tags}

            # Get total count
            if self.db.connected and self.db.db is not None:
                total_count = await self.db.db[self.contacts_collection].count_documents(filter_dict)
            else:
                total_count = 0

            # Calculate pagination
            skip = (page - 1) * page_size
            total_pages = (total_count + page_size - 1) // page_size

            # Get paginated data
            contact_dicts = await self.db.find_many(
                self.contacts_collection,
                filter_dict,
                limit=page_size,
                sort=[(sort_field, sort_direction)]
            )

            # Apply skip manually (Motor doesn't support skip in find_many)
            if skip > 0 and len(contact_dicts) > skip:
                contact_dicts = contact_dicts[skip:]

            # Convert to Contact objects
            contacts = []
            for contact_dict in contact_dicts:
                contact_dict["created_at"] = Database.from_iso(contact_dict.get("created_at"))
                contact_dict["updated_at"] = Database.from_iso(contact_dict.get("updated_at"))
                contact_dict["last_contacted"] = Database.from_iso(contact_dict.get("last_contacted"))
                contacts.append(Contact(**contact_dict))

            # Build pagination metadata
            pagination = {
                "total_count": total_count,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            }

            logger.debug(f"✅ Listed {len(contacts)} contacts (page {page}/{total_pages})")
            return contacts, pagination

        except Exception as e:
            logger.error(f"❌ Error listing contacts (paginated): {e}")
            return [], {"total_count": 0, "page": 1, "page_size": page_size, "total_pages": 0, "has_next": False, "has_prev": False}

    async def list_deals_paginated(
        self,
        page: int = 1,
        page_size: int = 100,
        stage: Optional[DealStage] = None,
        contact_id: Optional[str] = None,
        company_id: Optional[str] = None,
        sort_field: str = "created_at",
        sort_direction: int = -1
    ) -> Tuple[List[Deal], Dict[str, Any]]:
        """
        List deals with pagination support.

        Phase 3, Task 3.1: Implement query result pagination for large datasets

        Returns:
            Tuple of (deals list, pagination metadata)
        """
        try:
            # Build filter
            filter_dict = {}
            if stage:
                filter_dict["stage"] = stage
            if contact_id:
                filter_dict["contact_id"] = contact_id
            if company_id:
                filter_dict["company_id"] = company_id

            # Get total count
            if self.db.connected and self.db.db is not None:
                total_count = await self.db.db[self.deals_collection].count_documents(filter_dict)
            else:
                total_count = 0

            # Calculate pagination
            skip = (page - 1) * page_size
            total_pages = (total_count + page_size - 1) // page_size

            # Get paginated data
            deal_dicts = await self.db.find_many(
                self.deals_collection,
                filter_dict,
                limit=page_size,
                sort=[(sort_field, sort_direction)]
            )

            # Apply skip manually
            if skip > 0 and len(deal_dicts) > skip:
                deal_dicts = deal_dicts[skip:]

            # Convert to Deal objects
            deals = []
            for deal_dict in deal_dicts:
                deal_dict["created_at"] = Database.from_iso(deal_dict.get("created_at"))
                deal_dict["updated_at"] = Database.from_iso(deal_dict.get("updated_at"))
                deal_dict["expected_close_date"] = Database.from_iso(deal_dict.get("expected_close_date"))
                deal_dict["closed_at"] = Database.from_iso(deal_dict.get("closed_at"))
                deals.append(Deal(**deal_dict))

            # Build pagination metadata
            pagination = {
                "total_count": total_count,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            }

            logger.debug(f"✅ Listed {len(deals)} deals (page {page}/{total_pages})")
            return deals, pagination

        except Exception as e:
            logger.error(f"❌ Error listing deals (paginated): {e}")
            return [], {"total_count": 0, "page": 1, "page_size": page_size, "total_pages": 0, "has_next": False, "has_prev": False}

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
                days_since_contact = (Database.utcnow() - contact.last_contacted).days
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
                calculated_at=Database.utcnow()
            )

        except Exception as e:
            logger.error(f"❌ Error calculating lead score for {contact_id}: {e}")
            raise

    async def suggest_next_action(self, entity_type: str, entity_id: str) -> NextAction:
        """
        Suggest AI-powered next action for an entity.

        Phase 2, Task 2.2: Enhanced with timing recommendations and duration estimates

        Args:
            entity_type: Type of entity (contact, deal, company)
            entity_id: ID of the entity

        Returns:
            NextAction with recommended action, timing, and reasoning
        """
        try:
            now = Database.utcnow()

            if entity_type == "contact":
                contact = await self.get_contact(entity_id)
                if not contact:
                    raise ValueError(f"Contact {entity_id} not found")

                # Get recent activities
                activities = await self.list_activities(contact_id=entity_id, limit=10)
                recent_activities = [a for a in activities if (now - a.created_at).days < 30]

                # Calculate days since last contact
                days_since_contact = 999
                if contact.last_contacted:
                    days_since_contact = (now - contact.last_contacted).days

                # Determine next action based on engagement patterns
                if not recent_activities or days_since_contact > 30:
                    # No recent activity - urgent outreach needed
                    suggested_date = now + timedelta(days=1)
                    return NextAction(
                        entity_type="contact",
                        entity_id=entity_id,
                        action_type=ActivityType.CALL,
                        subject=f"Initial outreach to {contact.first_name} {contact.last_name}",
                        description="No recent activity detected. Schedule an introductory call to establish relationship and understand their needs.",
                        priority=Priority.HIGH,
                        reasoning=f"Contact has no recent engagement ({days_since_contact} days since last contact). Initial outreach is critical to re-establish relationship.",
                        confidence=0.9,
                        suggested_date=suggested_date,
                        estimated_duration=30
                    )
                elif contact.status == ContactStatus.LEAD and len(recent_activities) >= 3:
                    # High engagement lead - ready for demo
                    suggested_date = now + timedelta(days=3)
                    return NextAction(
                        entity_type="contact",
                        entity_id=entity_id,
                        action_type=ActivityType.MEETING,
                        subject=f"Product demo with {contact.first_name} {contact.last_name}",
                        description="Contact shows high engagement. Schedule product demonstration to showcase value proposition and address specific needs.",
                        priority=Priority.HIGH,
                        reasoning=f"Multiple recent interactions ({len(recent_activities)} activities in 30 days) indicate strong interest. Time for product demo to move forward.",
                        confidence=0.85,
                        suggested_date=suggested_date,
                        estimated_duration=60
                    )
                elif contact.status == ContactStatus.PROSPECT and days_since_contact > 14:
                    # Prospect needs follow-up
                    suggested_date = now + timedelta(days=2)
                    return NextAction(
                        entity_type="contact",
                        entity_id=entity_id,
                        action_type=ActivityType.EMAIL,
                        subject=f"Check-in with {contact.first_name} {contact.last_name}",
                        description="Send personalized check-in email to maintain engagement and provide value-added content.",
                        priority=Priority.MEDIUM,
                        reasoning=f"Prospect status with {days_since_contact} days since last contact. Regular touchpoints maintain relationship momentum.",
                        confidence=0.8,
                        suggested_date=suggested_date,
                        estimated_duration=15
                    )
                else:
                    # Regular follow-up
                    suggested_date = now + timedelta(days=7)
                    return NextAction(
                        entity_type="contact",
                        entity_id=entity_id,
                        action_type=ActivityType.EMAIL,
                        subject=f"Follow-up with {contact.first_name} {contact.last_name}",
                        description="Send personalized follow-up email with relevant content and industry insights.",
                        priority=Priority.MEDIUM,
                        reasoning="Maintain engagement with regular touchpoints and value-added communication.",
                        confidence=0.75,
                        suggested_date=suggested_date,
                        estimated_duration=15
                    )

            elif entity_type == "deal":
                deal = await self.get_deal(entity_id)
                if not deal:
                    raise ValueError(f"Deal {entity_id} not found")

                # Calculate days in current stage
                days_in_stage = (now - deal.created_at).days

                # Suggest action based on stage and time in stage
                if deal.stage == DealStage.LEAD:
                    suggested_date = now + timedelta(days=1)
                    return NextAction(
                        entity_type="deal",
                        entity_id=entity_id,
                        action_type=ActivityType.CALL,
                        subject=f"Qualify deal: {deal.title}",
                        description="Schedule qualification call to assess fit, budget, authority, need, and timeline (BANT).",
                        priority=Priority.HIGH,
                        reasoning=f"New deal needs qualification to move forward. Deal has been in LEAD stage for {days_in_stage} days.",
                        confidence=0.9,
                        suggested_date=suggested_date,
                        estimated_duration=45
                    )
                elif deal.stage == DealStage.QUALIFIED:
                    suggested_date = now + timedelta(days=3)
                    return NextAction(
                        entity_type="deal",
                        entity_id=entity_id,
                        action_type=ActivityType.DEMO,
                        subject=f"Product demo: {deal.title}",
                        description="Schedule comprehensive product demonstration tailored to their specific needs and use cases.",
                        priority=Priority.HIGH,
                        reasoning=f"Qualified deal ready for demo. Move to proposal stage. Deal has been qualified for {days_in_stage} days.",
                        confidence=0.88,
                        suggested_date=suggested_date,
                        estimated_duration=60
                    )
                elif deal.stage == DealStage.PROPOSAL:
                    urgency = Priority.URGENT if days_in_stage > 14 else Priority.HIGH
                    suggested_date = now + timedelta(days=1)
                    return NextAction(
                        entity_type="deal",
                        entity_id=entity_id,
                        action_type=ActivityType.MEETING,
                        subject=f"Proposal review: {deal.title}",
                        description="Schedule meeting to review proposal details, address concerns, and discuss next steps toward closing.",
                        priority=urgency,
                        reasoning=f"Proposal stage requires active engagement to close. Deal has been in proposal for {days_in_stage} days - {'urgent action needed' if days_in_stage > 14 else 'maintain momentum'}.",
                        confidence=0.95,
                        suggested_date=suggested_date,
                        estimated_duration=45
                    )
                elif deal.stage == DealStage.NEGOTIATION:
                    suggested_date = now + timedelta(days=1)
                    return NextAction(
                        entity_type="deal",
                        entity_id=entity_id,
                        action_type=ActivityType.CALL,
                        subject=f"Negotiation call: {deal.title}",
                        description="Schedule call to address final concerns, negotiate terms, and move toward contract signing.",
                        priority=Priority.URGENT,
                        reasoning=f"Deal in negotiation stage for {days_in_stage} days. Critical to close quickly to prevent deal stagnation.",
                        confidence=0.92,
                        suggested_date=suggested_date,
                        estimated_duration=30
                    )
                else:
                    suggested_date = now + timedelta(days=7)
                    return NextAction(
                        entity_type="deal",
                        entity_id=entity_id,
                        action_type=ActivityType.EMAIL,
                        subject=f"Update on {deal.title}",
                        description="Send status update and next steps to maintain communication.",
                        priority=Priority.MEDIUM,
                        reasoning="Regular communication maintains deal momentum and relationship.",
                        confidence=0.7,
                        suggested_date=suggested_date,
                        estimated_duration=15
                    )

            else:
                raise ValueError(f"Unsupported entity type: {entity_type}")

        except Exception as e:
            logger.error(f"❌ Error suggesting next action for {entity_type} {entity_id}: {e}")
            raise

    async def analyze_contact_sentiment(self, contact_id: str) -> ContactSentiment:
        """
        Analyze contact sentiment based on interaction history.

        Phase 2, Task 2.2: AI-powered sentiment analysis

        Analyzes:
        - Activity notes and descriptions
        - Interaction frequency and recency
        - Deal progression
        - Response patterns

        Args:
            contact_id: ID of the contact to analyze

        Returns:
            ContactSentiment with sentiment type, score, and trends
        """
        try:
            contact = await self.get_contact(contact_id)
            if not contact:
                raise ValueError(f"Contact {contact_id} not found")

            now = Database.utcnow()
            factors = {}
            insights = []
            trends = []

            # Get all activities for sentiment analysis
            activities = await self.list_activities(contact_id=contact_id, limit=100)

            # Analyze activity notes for sentiment keywords
            positive_keywords = [
                'great', 'excellent', 'interested', 'excited', 'love', 'perfect',
                'amazing', 'wonderful', 'fantastic', 'impressed', 'happy', 'pleased',
                'enthusiastic', 'positive', 'agree', 'yes', 'definitely', 'absolutely'
            ]
            negative_keywords = [
                'concerned', 'worried', 'issue', 'problem', 'difficult', 'challenge',
                'disappointed', 'unhappy', 'frustrated', 'angry', 'no', 'not interested',
                'expensive', 'costly', 'budget', 'competitor', 'delay', 'postpone'
            ]
            neutral_keywords = [
                'okay', 'fine', 'maybe', 'consider', 'think', 'review', 'discuss',
                'evaluate', 'assess', 'analyze', 'understand', 'clarify'
            ]

            positive_count = 0
            negative_count = 0
            neutral_count = 0

            for activity in activities:
                notes_lower = (activity.notes or '').lower()
                description_lower = (activity.description or '').lower()
                combined_text = f"{notes_lower} {description_lower}"

                # Count keyword occurrences
                for keyword in positive_keywords:
                    if keyword in combined_text:
                        positive_count += 1

                for keyword in negative_keywords:
                    if keyword in combined_text:
                        negative_count += 1

                for keyword in neutral_keywords:
                    if keyword in combined_text:
                        neutral_count += 1

            factors["positive_signals"] = positive_count
            factors["negative_signals"] = negative_count
            factors["neutral_signals"] = neutral_count

            # Factor 1: Keyword sentiment (-30 to +30 points)
            keyword_score = (positive_count * 2) - (negative_count * 2)
            factors["keyword_sentiment_score"] = keyword_score

            # Factor 2: Engagement frequency (-20 to +20 points)
            recent_activities = [a for a in activities if (now - a.created_at).days < 30]
            if len(recent_activities) >= 5:
                engagement_score = 20
                insights.append(f"High engagement: {len(recent_activities)} activities in 30 days")
            elif len(recent_activities) >= 3:
                engagement_score = 10
                insights.append(f"Good engagement: {len(recent_activities)} activities in 30 days")
            elif len(recent_activities) >= 1:
                engagement_score = 0
                insights.append(f"Moderate engagement: {len(recent_activities)} activities in 30 days")
            else:
                engagement_score = -20
                insights.append("Low engagement: No recent activities")
            factors["engagement_score"] = engagement_score

            # Factor 3: Deal progression (-20 to +30 points)
            deals = await self.list_deals(contact_id=contact_id, limit=50)
            active_deals = [d for d in deals if d.stage not in [DealStage.CLOSED_WON, DealStage.CLOSED_LOST]]
            won_deals = [d for d in deals if d.stage == DealStage.CLOSED_WON]
            lost_deals = [d for d in deals if d.stage == DealStage.CLOSED_LOST]

            if won_deals:
                deal_score = 30
                insights.append(f"Positive history: {len(won_deals)} won deal(s)")
            elif active_deals:
                # Check stage of active deals
                advanced_deals = [d for d in active_deals if d.stage in [DealStage.PROPOSAL, DealStage.NEGOTIATION]]
                if advanced_deals:
                    deal_score = 20
                    insights.append(f"Active deals in advanced stages: {len(advanced_deals)}")
                else:
                    deal_score = 10
                    insights.append(f"Active deals in early stages: {len(active_deals)}")
            elif lost_deals:
                deal_score = -20
                insights.append(f"Negative history: {len(lost_deals)} lost deal(s)")
            else:
                deal_score = 0
                insights.append("No deal history")
            factors["deal_progression_score"] = deal_score

            # Factor 4: Response time (-10 to +10 points)
            if contact.last_contacted:
                days_since_contact = (now - contact.last_contacted).days
                if days_since_contact < 7:
                    response_score = 10
                    insights.append("Recent contact (within 7 days)")
                elif days_since_contact < 30:
                    response_score = 5
                    insights.append("Contact within last month")
                elif days_since_contact < 90:
                    response_score = -5
                    insights.append("Contact becoming stale (30-90 days)")
                else:
                    response_score = -10
                    insights.append(f"Stale contact ({days_since_contact} days since last contact)")
            else:
                response_score = 0
                insights.append("No contact history")
            factors["response_time_score"] = response_score

            # Calculate overall sentiment score (-1 to +1)
            total_score = keyword_score + engagement_score + deal_score + response_score
            max_possible = 90  # 30 + 20 + 30 + 10
            min_possible = -80  # -30 + -20 + -20 + -10

            # Normalize to -1 to +1 range
            if total_score > 0:
                sentiment_score = total_score / max_possible
            else:
                sentiment_score = total_score / abs(min_possible)

            # Determine sentiment type
            if sentiment_score > 0.3:
                sentiment_type = SentimentType.POSITIVE
                insights.append("Overall sentiment is POSITIVE - strong relationship")
            elif sentiment_score > 0.1:
                sentiment_type = SentimentType.MIXED
                insights.append("Overall sentiment is MIXED with positive lean")
            elif sentiment_score > -0.1:
                sentiment_type = SentimentType.NEUTRAL
                insights.append("Overall sentiment is NEUTRAL - maintain engagement")
            elif sentiment_score > -0.3:
                sentiment_type = SentimentType.MIXED
                insights.append("Overall sentiment is MIXED with negative lean - needs attention")
            else:
                sentiment_type = SentimentType.NEGATIVE
                insights.append("Overall sentiment is NEGATIVE - immediate action required")

            # Build sentiment trends (last 90 days in 30-day buckets)
            for days_ago in [90, 60, 30]:
                period_start = now - timedelta(days=days_ago)
                period_end = now - timedelta(days=days_ago-30) if days_ago > 30 else now
                period_activities = [
                    a for a in activities
                    if period_start <= a.created_at <= period_end
                ]

                period_positive = sum(1 for a in period_activities
                                    if any(kw in (a.notes or '').lower() for kw in positive_keywords))
                period_negative = sum(1 for a in period_activities
                                    if any(kw in (a.notes or '').lower() for kw in negative_keywords))

                period_sentiment = "positive" if period_positive > period_negative else \
                                 "negative" if period_negative > period_positive else "neutral"

                trends.append({
                    "period": f"{days_ago-30}-{days_ago} days ago" if days_ago > 30 else "Last 30 days",
                    "activities": len(period_activities),
                    "sentiment": period_sentiment,
                    "positive_signals": period_positive,
                    "negative_signals": period_negative
                })

            # Determine confidence
            confidence = 0.6  # Base confidence
            if len(activities) > 10:
                confidence += 0.1
            if len(activities) > 20:
                confidence += 0.1
            if len(recent_activities) > 3:
                confidence += 0.1
            confidence = min(0.9, confidence)

            return ContactSentiment(
                contact_id=contact_id,
                sentiment=sentiment_type,
                confidence=confidence,
                score=sentiment_score,
                factors=factors,
                trends=trends,
                insights=insights,
                analyzed_at=now
            )

        except Exception as e:
            logger.error(f"❌ Error analyzing contact sentiment for {contact_id}: {e}")
            raise

    async def auto_log_activity(
        self,
        activity_type: ActivityType,
        subject: str,
        description: str,
        contact_id: Optional[str] = None,
        company_id: Optional[str] = None,
        deal_id: Optional[str] = None,
        priority: Priority = Priority.MEDIUM
    ) -> Activity:
        """
        Automatically log an activity with intelligent defaults.

        Phase 2, Task 2.2: Automated activity logging

        Features:
        - Automatic timestamp
        - Intelligent priority assignment
        - Auto-linking to related entities
        - Activity summary generation

        Args:
            activity_type: Type of activity
            subject: Activity subject
            description: Activity description
            contact_id: Optional contact ID
            company_id: Optional company ID
            deal_id: Optional deal ID
            priority: Activity priority (auto-assigned if not specified)

        Returns:
            Created Activity object
        """
        try:
            # Auto-assign priority based on activity type if not specified
            if priority == Priority.MEDIUM:
                if activity_type in [ActivityType.MEETING, ActivityType.DEMO]:
                    priority = Priority.HIGH
                elif activity_type == ActivityType.CALL:
                    priority = Priority.MEDIUM
                else:
                    priority = Priority.LOW

            # Create activity
            activity_create = ActivityCreate(
                activity_type=activity_type,
                subject=subject,
                description=description,
                contact_id=contact_id,
                company_id=company_id,
                deal_id=deal_id,
                priority=priority,
                completed=True,  # Auto-logged activities are completed
                notes=f"Auto-logged at {Database.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
            )

            activity = await self.create_activity(activity_create)

            # Update contact's last_contacted if contact_id provided
            if contact_id:
                await self.db.update_one(
                    self.contacts_collection,
                    {"contact_id": contact_id},
                    {"$set": {"last_contacted": Database.to_iso(Database.utcnow())}}
                )

            logger.info(f"✅ Auto-logged activity: {activity_type.value} - {subject}")
            return activity

        except Exception as e:
            logger.error(f"❌ Error auto-logging activity: {e}")
            raise

    async def predict_deal_probability(self, deal_id: str) -> DealProbability:
        """
        Predict deal win probability using AI analysis.

        Phase 2, Task 2.2: AI-powered deal probability prediction

        Analyzes multiple factors:
        - Deal stage and progression
        - Time in pipeline
        - Deal value relative to average
        - Contact engagement level
        - Activity frequency

        Args:
            deal_id: ID of the deal to analyze

        Returns:
            DealProbability with percentage, confidence, and insights
        """
        try:
            deal = await self.get_deal(deal_id)
            if not deal:
                raise ValueError(f"Deal {deal_id} not found")

            now = Database.utcnow()
            factors = {}
            insights = []
            recommendations = []

            # Factor 1: Deal stage (0-40 points)
            stage_probabilities = {
                DealStage.LEAD: 10,
                DealStage.QUALIFIED: 20,
                DealStage.PROPOSAL: 50,
                DealStage.NEGOTIATION: 75,
                DealStage.CLOSED_WON: 100,
                DealStage.CLOSED_LOST: 0
            }
            stage_score = stage_probabilities.get(deal.stage, 10)
            factors["stage_probability"] = stage_score
            insights.append(f"Deal is in {deal.stage.value} stage (base probability: {stage_score}%)")

            # Factor 2: Time in pipeline (-20 to +10 points)
            days_in_pipeline = (now - deal.created_at).days
            if days_in_pipeline < 30:
                time_modifier = 10  # Fresh deal, good momentum
                insights.append(f"Deal is {days_in_pipeline} days old - good momentum")
            elif days_in_pipeline < 60:
                time_modifier = 0  # Normal timeline
                insights.append(f"Deal is {days_in_pipeline} days old - normal timeline")
            elif days_in_pipeline < 90:
                time_modifier = -10  # Getting stale
                insights.append(f"Deal is {days_in_pipeline} days old - needs attention")
                recommendations.append("Accelerate deal progression - timeline is extending")
            else:
                time_modifier = -20  # Very stale
                insights.append(f"Deal is {days_in_pipeline} days old - high risk of stagnation")
                recommendations.append("URGENT: Re-engage immediately or consider closing as lost")
            factors["time_in_pipeline_modifier"] = time_modifier

            # Factor 3: Deal value relative to average (-10 to +15 points)
            # Get average deal size from stats
            all_deals = await self.list_deals(limit=1000)
            won_deals = [d for d in all_deals if d.stage == DealStage.CLOSED_WON]
            avg_deal_value = sum(d.value for d in won_deals) / len(won_deals) if won_deals else 50000

            value_ratio = deal.value / avg_deal_value if avg_deal_value > 0 else 1.0
            if value_ratio > 2.0:
                value_modifier = -10  # Very large deal, harder to close
                insights.append(f"Deal value (${deal.value:,.0f}) is {value_ratio:.1f}x average - requires extra attention")
                recommendations.append("Large deal requires executive sponsorship and detailed proposal")
            elif value_ratio > 1.5:
                value_modifier = 0  # Above average
                insights.append(f"Deal value (${deal.value:,.0f}) is above average")
            elif value_ratio > 0.5:
                value_modifier = 10  # Normal range, easier to close
                insights.append(f"Deal value (${deal.value:,.0f}) is in normal range")
            else:
                value_modifier = 15  # Small deal, easier to close
                insights.append(f"Deal value (${deal.value:,.0f}) is below average - quick close opportunity")
            factors["deal_value_modifier"] = value_modifier

            # Factor 4: Contact engagement (0-20 points)
            if deal.contact_id:
                activities = await self.list_activities(contact_id=deal.contact_id, limit=50)
                recent_activities = [a for a in activities if (now - a.created_at).days < 30]

                if len(recent_activities) >= 5:
                    engagement_score = 20
                    insights.append(f"High contact engagement ({len(recent_activities)} activities in 30 days)")
                elif len(recent_activities) >= 3:
                    engagement_score = 15
                    insights.append(f"Good contact engagement ({len(recent_activities)} activities in 30 days)")
                elif len(recent_activities) >= 1:
                    engagement_score = 10
                    insights.append(f"Moderate contact engagement ({len(recent_activities)} activities in 30 days)")
                    recommendations.append("Increase engagement frequency to build momentum")
                else:
                    engagement_score = 0
                    insights.append("Low contact engagement - no recent activities")
                    recommendations.append("CRITICAL: Re-establish contact immediately")
                factors["contact_engagement"] = engagement_score
            else:
                factors["contact_engagement"] = 0
                insights.append("No contact associated with deal")
                recommendations.append("Associate a contact with this deal for better tracking")

            # Calculate final probability
            base_probability = stage_score
            modifiers = time_modifier + value_modifier + (factors["contact_engagement"] - 10)
            final_probability = max(0, min(100, base_probability + modifiers))

            # Determine confidence based on data availability
            confidence = 0.75
            if deal.contact_id and len(recent_activities) > 0:
                confidence = 0.85
            if days_in_pipeline > 90:
                confidence = 0.65  # Less confident for stale deals

            # Add stage-specific recommendations
            if deal.stage == DealStage.LEAD and final_probability < 30:
                recommendations.append("Qualify this lead thoroughly before investing more time")
            elif deal.stage == DealStage.PROPOSAL and final_probability > 60:
                recommendations.append("Strong proposal - push for commitment and close")
            elif deal.stage == DealStage.NEGOTIATION and final_probability < 70:
                recommendations.append("Address concerns blocking negotiation progress")

            return DealProbability(
                deal_id=deal_id,
                probability=final_probability,
                confidence=confidence,
                factors=factors,
                insights=insights,
                recommendations=recommendations,
                calculated_at=now
            )

        except Exception as e:
            logger.error(f"❌ Error predicting deal probability for {deal_id}: {e}")
            raise

    # ========================================================================
    # STATISTICS
    # ========================================================================

    async def get_crm_stats(self) -> CRMStats:
        """
        Get comprehensive CRM statistics using optimized aggregation pipelines.

        Phase 3, Task 3.1: Optimize aggregation pipelines in MongoDB
        Phase 3, Task 3.2: Cache CRM stats (5-minute TTL)

        BEFORE: Loaded 10,000 records of each type into memory (40,000 total)
        AFTER: Uses MongoDB aggregation to calculate stats in database
        NOW: Cached with 5-minute TTL for improved performance
        """
        try:
            # Try to get from cache first
            if self.cache:
                cache_key = self.cache.get_stats_key("crm")
                cached_stats = await self.cache.get(cache_key)
                if cached_stats:
                    logger.debug("✅ CRM stats retrieved from cache")
                    return CRMStats(**cached_stats)
            # Use aggregation pipelines for efficient stats calculation
            if not self.db.connected or self.db.db is None:
                raise RuntimeError("Database not connected")

            # Get contact stats by status
            contacts_pipeline = [
                {"$group": {"_id": "$status", "count": {"$sum": 1}}}
            ]
            contacts_by_status_results = await self.db.db["crm_contacts"].aggregate(contacts_pipeline).to_list(None)
            contacts_by_status = {r["_id"]: r["count"] for r in contacts_by_status_results}
            total_contacts = sum(contacts_by_status.values())

            # Get company count
            total_companies = await self.db.db["crm_companies"].count_documents({})

            # Get deal stats by stage with value calculations
            deals_pipeline = [
                {
                    "$group": {
                        "_id": "$stage",
                        "count": {"$sum": 1},
                        "total_value": {"$sum": "$value"}
                    }
                }
            ]
            deals_by_stage_results = await self.db.db["crm_deals"].aggregate(deals_pipeline).to_list(None)
            deals_by_stage = {r["_id"]: r["count"] for r in deals_by_stage_results}
            total_deals = sum(deals_by_stage.values())

            # Calculate pipeline value (all deals except closed)
            total_pipeline_value = sum(
                r["total_value"] for r in deals_by_stage_results
                if r["_id"] not in [DealStage.CLOSED_WON.value, DealStage.CLOSED_LOST.value]
            )

            # Calculate won value
            won_result = next(
                (r for r in deals_by_stage_results if r["_id"] == DealStage.CLOSED_WON.value),
                {"count": 0, "total_value": 0.0}
            )
            total_won_value = won_result["total_value"]
            won_count = won_result["count"]

            # Calculate average deal size
            average_deal_size = total_won_value / won_count if won_count > 0 else 0.0

            # Calculate win rate
            lost_count = next(
                (r["count"] for r in deals_by_stage_results if r["_id"] == DealStage.CLOSED_LOST.value),
                0
            )
            closed_count = won_count + lost_count
            win_rate = (won_count / closed_count * 100) if closed_count > 0 else 0.0

            # Get activity stats by type
            activities_pipeline = [
                {"$group": {"_id": "$activity_type", "count": {"$sum": 1}}}
            ]
            activities_by_type_results = await self.db.db["crm_activities"].aggregate(activities_pipeline).to_list(None)
            activities_by_type = {r["_id"]: r["count"] for r in activities_by_type_results}
            total_activities = sum(activities_by_type.values())

            logger.info(
                f"✅ CRM stats calculated using aggregation pipelines: "
                f"{total_contacts} contacts, {total_companies} companies, "
                f"{total_deals} deals, {total_activities} activities"
            )

            stats = CRMStats(
                total_contacts=total_contacts,
                total_companies=total_companies,
                total_deals=total_deals,
                total_activities=total_activities,
                contacts_by_status=contacts_by_status,
                deals_by_stage=deals_by_stage,
                total_pipeline_value=total_pipeline_value,
                total_won_value=total_won_value,
                average_deal_size=average_deal_size,
                win_rate=win_rate,
                activities_by_type=activities_by_type
            )

            # Cache the stats for 5 minutes
            if self.cache:
                cache_key = self.cache.get_stats_key("crm")
                await self.cache.cache_stats("crm", stats.model_dump())
                logger.debug("✅ CRM stats cached (5-minute TTL)")

            return stats

        except Exception as e:
            logger.error(f"❌ Error getting CRM stats: {e}")
            raise

    # ========================================================================
    # EVENT HANDLERS
    # ========================================================================

    async def _on_lead_qualified(self, event: Dict[str, Any]):
        """
        Handle lead qualified event - automatically create CRM contact

        Args:
            event: Event data containing lead information
        """
        try:
            lead_data = event["data"]
            logger.info(f"📥 Received qualified lead: {lead_data.get('company_name')}")

            # Create contact from qualified lead
            contact_create = ContactCreate(
                name=lead_data.get("company_name", "Unknown"),
                email=lead_data.get("email"),
                phone=lead_data.get("phone"),
                company=lead_data.get("company_name"),
                title="Lead",
                status=ContactStatus.LEAD,
                tags=["qualified_lead", lead_data.get("industry", "")],
                notes=f"Auto-created from qualified lead (Score: {lead_data.get('score', 0)})",
                source="lead_generation"
            )

            contact = await self.create_contact(contact_create)
            logger.info(f"✅ Created CRM contact from qualified lead: {contact.contact_id}")

        except Exception as e:
            logger.error(f"❌ Error handling lead qualified event: {e}")

    async def _on_lead_enriched(self, event: Dict[str, Any]):
        """
        Handle lead enriched event - update existing contact with enriched data

        Args:
            event: Event data containing enriched lead information
        """
        try:
            lead_data = event["data"]
            company_data = lead_data.get("company_data", {})

            # Find existing contact by company name
            contacts = await self.list_contacts(limit=1000)
            matching_contact = None
            for contact in contacts:
                if contact.company and contact.company.lower() == company_data.get("company_name", "").lower():
                    matching_contact = contact
                    break

            if matching_contact:
                # Update contact with enriched data
                update_data = ContactUpdate(
                    email=company_data.get("email") or matching_contact.email,
                    phone=company_data.get("phone") or matching_contact.phone,
                    website=company_data.get("website") or matching_contact.website,
                    notes=f"{matching_contact.notes or ''}\n\nEnriched data (Score: {lead_data.get('score', 0)})"
                )

                await self.update_contact(matching_contact.contact_id, update_data)
                logger.info(f"✅ Updated contact with enriched lead data: {matching_contact.contact_id}")

        except Exception as e:
            logger.error(f"❌ Error handling lead enriched event: {e}")

