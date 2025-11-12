"""
Lead Generation Service

Automated lead generation and research service that leverages OSINT tools
and research automation to discover, qualify, and monitor leads.
"""

import secrets
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging

from database import Database
from models.lead_generation_models import (
    LeadCampaign,
    LeadCampaignConfig,
    Lead,
    LeadStatus,
    LeadClassification,
    ResearchReport,
    LeadScore,
    CompanyData,
    KeyPerson,
    NewsItem,
    SocialActivity,
    TechnologyStack,
    Opportunity,
    ResearchDepth,
    MonitoringFrequency,
    LeadUpdate,
)
from services.osint_service import OSINTService
from services.research_automation import ResearchAutomationEngine
from services.event_bus import publish_event
from models.event_models import EventType

logger = logging.getLogger(__name__)


class LeadGenerationService:
    """Automated lead generation and research service"""
    
    def __init__(self, db: Optional[Database] = None):
        """Initialize lead generation service"""
        self.db = db or Database()
        self.campaigns_collection = "lead_campaigns"
        self.leads_collection = "leads"
        self.research_collection = "lead_research"
        
        # Initialize dependent services
        self.osint_service = OSINTService(db)
        self.research_service = ResearchAutomationEngine(db)
        
        # Mock mode for testing
        self.mock_mode = True
        if self.mock_mode:
            logger.info("⚠️  Lead Generation Service running in MOCK MODE")
    
    # ========================================================================
    # Campaign Management
    # ========================================================================
    
    async def create_campaign(
        self,
        user_id: str,
        config: LeadCampaignConfig
    ) -> str:
        """
        Create a new lead generation campaign
        
        Args:
            user_id: User ID
            config: Campaign configuration
        
        Returns:
            campaign_id
        """
        campaign_id = f"campaign_{secrets.token_urlsafe(16)}"
        
        campaign = LeadCampaign(
            campaign_id=campaign_id,
            user_id=user_id,
            config=config,
            status="active",
            leads_generated=0,
            leads_qualified=0,
        )
        
        # Store in database
        campaign_data = campaign.model_dump()
        campaign_data["created_at"] = campaign.created_at.isoformat()
        campaign_data["updated_at"] = campaign.updated_at.isoformat()
        campaign_data["config"] = config.model_dump()
        
        await self.db.insert_one(self.campaigns_collection, campaign_data)
        
        logger.info(f"Created campaign {campaign_id} for user {user_id}")
        return campaign_id
    
    async def get_campaign(self, campaign_id: str) -> Optional[LeadCampaign]:
        """Get campaign by ID"""
        campaign_data = await self.db.find_one(
            self.campaigns_collection,
            {"campaign_id": campaign_id}
        )
        
        if not campaign_data:
            return None
        
        # Convert ISO strings back to datetime
        if isinstance(campaign_data.get("created_at"), str):
            campaign_data["created_at"] = datetime.fromisoformat(campaign_data["created_at"])
        if isinstance(campaign_data.get("updated_at"), str):
            campaign_data["updated_at"] = datetime.fromisoformat(campaign_data["updated_at"])
        
        # Convert config dict to LeadCampaignConfig
        if isinstance(campaign_data.get("config"), dict):
            campaign_data["config"] = LeadCampaignConfig(**campaign_data["config"])
        
        return LeadCampaign(**campaign_data)
    
    async def list_campaigns(self, user_id: str) -> List[LeadCampaign]:
        """List all campaigns for a user"""
        campaigns_data = await self.db.find_many(
            self.campaigns_collection,
            {"user_id": user_id}
        )
        
        if not campaigns_data:
            return []
        
        campaigns = []
        for campaign_data in campaigns_data:
            # Convert ISO strings
            if isinstance(campaign_data.get("created_at"), str):
                campaign_data["created_at"] = datetime.fromisoformat(campaign_data["created_at"])
            if isinstance(campaign_data.get("updated_at"), str):
                campaign_data["updated_at"] = datetime.fromisoformat(campaign_data["updated_at"])
            
            # Convert config
            if isinstance(campaign_data.get("config"), dict):
                campaign_data["config"] = LeadCampaignConfig(**campaign_data["config"])
            
            campaigns.append(LeadCampaign(**campaign_data))
        
        return campaigns
    
    # ========================================================================
    # Lead Management
    # ========================================================================
    
    async def create_lead(
        self,
        campaign_id: str,
        user_id: str,
        company_name: str,
        industry: str,
        location: str,
        **kwargs
    ) -> str:
        """
        Create a new lead
        
        Args:
            campaign_id: Campaign ID
            user_id: User ID
            company_name: Company name
            industry: Industry
            location: Location
            **kwargs: Additional lead data
        
        Returns:
            lead_id
        """
        lead_id = f"lead_{secrets.token_urlsafe(16)}"
        
        lead = Lead(
            lead_id=lead_id,
            campaign_id=campaign_id,
            user_id=user_id,
            company_name=company_name,
            industry=industry,
            location=location,
            **kwargs
        )
        
        # Store in database
        lead_data = lead.model_dump()
        lead_data["created_at"] = lead.created_at.isoformat()
        lead_data["updated_at"] = lead.updated_at.isoformat()
        if lead.next_research_date:
            lead_data["next_research_date"] = lead.next_research_date.isoformat()
        
        await self.db.insert_one(self.leads_collection, lead_data)
        
        # Update campaign stats
        await self.db.update_one(
            self.campaigns_collection,
            {"campaign_id": campaign_id},
            {"$inc": {"leads_generated": 1}}
        )

        # Publish event
        await publish_event(
            EventType.LEAD_GENERATED,
            {
                "lead_id": lead_id,
                "campaign_id": campaign_id,
                "company_name": company_name,
                "industry": industry,
                "location": location,
                "email": kwargs.get("email"),
                "phone": kwargs.get("phone"),
                "website": kwargs.get("website"),
            },
            source="lead_generation_service"
        )

        logger.info(f"Created lead {lead_id} for campaign {campaign_id}")
        return lead_id
    
    async def get_lead(self, lead_id: str) -> Optional[Lead]:
        """Get lead by ID"""
        lead_data = await self.db.find_one(
            self.leads_collection,
            {"lead_id": lead_id}
        )
        
        if not lead_data:
            return None
        
        # Convert ISO strings back to datetime
        if isinstance(lead_data.get("created_at"), str):
            lead_data["created_at"] = datetime.fromisoformat(lead_data["created_at"])
        if isinstance(lead_data.get("updated_at"), str):
            lead_data["updated_at"] = datetime.fromisoformat(lead_data["updated_at"])
        if isinstance(lead_data.get("next_research_date"), str):
            lead_data["next_research_date"] = datetime.fromisoformat(lead_data["next_research_date"])
        
        return Lead(**lead_data)
    
    async def list_leads(
        self,
        campaign_id: Optional[str] = None,
        user_id: Optional[str] = None,
        status: Optional[LeadStatus] = None,
        page: int = 1,
        page_size: int = 20
    ) -> tuple[List[Lead], int]:
        """
        List leads with filtering and pagination
        
        Returns:
            (leads, total_count)
        """
        query = {}
        if campaign_id:
            query["campaign_id"] = campaign_id
        if user_id:
            query["user_id"] = user_id
        if status:
            query["status"] = status
        
        # Get total count
        all_leads = await self.db.find_many(self.leads_collection, query)
        total = len(all_leads) if all_leads else 0
        
        # Paginate
        skip = (page - 1) * page_size
        leads_data = all_leads[skip:skip + page_size] if all_leads else []
        
        leads = []
        for lead_data in leads_data:
            # Convert ISO strings
            if isinstance(lead_data.get("created_at"), str):
                lead_data["created_at"] = datetime.fromisoformat(lead_data["created_at"])
            if isinstance(lead_data.get("updated_at"), str):
                lead_data["updated_at"] = datetime.fromisoformat(lead_data["updated_at"])
            if isinstance(lead_data.get("next_research_date"), str):
                lead_data["next_research_date"] = datetime.fromisoformat(lead_data["next_research_date"])
            
            leads.append(Lead(**lead_data))
        
        return leads, total
    
    async def update_lead_status(
        self,
        lead_id: str,
        status: LeadStatus
    ) -> bool:
        """Update lead status"""
        result = await self.db.update_one(
            self.leads_collection,
            {"lead_id": lead_id},
            {
                "$set": {
                    "status": status,
                    "updated_at": datetime.utcnow().isoformat()
                }
            }
        )

        # Update campaign stats if qualified
        if status == LeadStatus.QUALIFIED:
            lead = await self.get_lead(lead_id)
            if lead:
                await self.db.update_one(
                    self.campaigns_collection,
                    {"campaign_id": lead.campaign_id},
                    {"$inc": {"leads_qualified": 1}}
                )

        return result is not None

    async def update_lead_score(
        self,
        lead_id: str,
        score: float
    ) -> bool:
        """Update lead score"""
        result = await self.db.update_one(
            self.leads_collection,
            {"lead_id": lead_id},
            {
                "$set": {
                    "lead_score": score,
                    "updated_at": datetime.utcnow().isoformat()
                }
            }
        )

        return result is not None

    async def classify_lead(
        self,
        lead_id: str,
        classification: LeadClassification
    ) -> Dict[str, Any]:
        """
        Classify a lead and automatically create ATS candidate or CRM contact

        Args:
            lead_id: Lead ID
            classification: Lead classification (candidate, client, both, unclassified)

        Returns:
            Dictionary with classification results including created IDs
        """
        # Get the lead
        lead = await self.get_lead(lead_id)
        if not lead:
            raise ValueError(f"Lead {lead_id} not found")

        result = {
            "lead_id": lead_id,
            "classification": classification,
            "ats_candidate_id": None,
            "crm_contact_id": None,
            "message": ""
        }

        # Update lead classification
        update_data = {
            "classification": classification,
            "updated_at": datetime.utcnow().isoformat()
        }

        # Create ATS candidate if classified as candidate or both
        if classification in [LeadClassification.CANDIDATE, LeadClassification.BOTH]:
            try:
                # Import ATS service
                from services.ats_service_real import ATSService
                ats_service = ATSService(self.db)

                # Extract contact info from key_people
                first_name = "Unknown"
                last_name = ""
                email = None
                phone = None

                if lead.key_people and len(lead.key_people) > 0:
                    person = lead.key_people[0]
                    name_parts = person.name.split(" ", 1)
                    first_name = name_parts[0]
                    last_name = name_parts[1] if len(name_parts) > 1 else ""
                    email = person.email
                    phone = person.phone

                # Create candidate
                from models.ats_models import CandidateCreate
                candidate_create = CandidateCreate(
                    first_name=first_name,
                    last_name=last_name,
                    email=email or f"{lead.company_name.lower().replace(' ', '.')}@example.com",
                    phone=phone,
                    location=lead.location,
                    current_company=lead.company_name,
                    skills=[],
                    tags=["lead_generation", lead.industry],
                    custom_fields={
                        "source_lead_id": lead_id,
                        "company_website": lead.website,
                        "lead_score": lead.lead_score
                    }
                )

                candidate = await ats_service.create_candidate(candidate_create)
                update_data["ats_candidate_id"] = candidate.candidate_id
                result["ats_candidate_id"] = candidate.candidate_id

                logger.info(f"✅ Created ATS candidate {candidate.candidate_id} from lead {lead_id}")

                # Publish event
                await publish_event(
                    EventType.LEAD_QUALIFIED,
                    {
                        "lead_id": lead_id,
                        "candidate_id": candidate.candidate_id,
                        "company_name": lead.company_name,
                        "classification": "candidate"
                    },
                    source="lead_generation_service"
                )

            except Exception as e:
                logger.error(f"Failed to create ATS candidate from lead {lead_id}: {e}")
                result["message"] += f"ATS candidate creation failed: {str(e)}. "

        # Create CRM contact if classified as client or both
        if classification in [LeadClassification.CLIENT, LeadClassification.BOTH]:
            try:
                # Import CRM service
                from services.crm_service_new import CRMService
                crm_service = CRMService(self.db)

                # Extract contact info
                name = lead.company_name
                email = None
                phone = None
                title = "Lead"

                if lead.key_people and len(lead.key_people) > 0:
                    person = lead.key_people[0]
                    name = person.name
                    email = person.email
                    phone = person.phone
                    title = person.title or "Lead"

                # Create contact
                from models.crm_models import ContactCreate, ContactStatus
                contact_create = ContactCreate(
                    first_name=name.split(" ")[0] if " " in name else name,
                    last_name=" ".join(name.split(" ")[1:]) if " " in name else "",
                    email=email,
                    phone=phone,
                    title=title,
                    company_name=lead.company_name,
                    status=ContactStatus.LEAD,
                    tags=["lead_generation", lead.industry],
                    notes=f"Auto-created from lead generation (Score: {lead.lead_score})",
                    source="lead_generation",
                    custom_fields={
                        "source_lead_id": lead_id,
                        "website": lead.website,
                        "location": lead.location
                    }
                )

                contact = await crm_service.create_contact(contact_create)
                update_data["crm_contact_id"] = contact.contact_id
                result["crm_contact_id"] = contact.contact_id

                logger.info(f"✅ Created CRM contact {contact.contact_id} from lead {lead_id}")

                # Publish event
                await publish_event(
                    EventType.LEAD_QUALIFIED,
                    {
                        "lead_id": lead_id,
                        "contact_id": contact.contact_id,
                        "company_name": lead.company_name,
                        "classification": "client"
                    },
                    source="lead_generation_service"
                )

            except Exception as e:
                logger.error(f"Failed to create CRM contact from lead {lead_id}: {e}")
                result["message"] += f"CRM contact creation failed: {str(e)}. "

        # Update lead in database
        await self.db.update_one(
            self.leads_collection,
            {"lead_id": lead_id},
            {"$set": update_data}
        )

        # Set success message
        if not result["message"]:
            if classification == LeadClassification.CANDIDATE:
                result["message"] = "Lead classified as candidate and ATS record created"
            elif classification == LeadClassification.CLIENT:
                result["message"] = "Lead classified as client and CRM record created"
            elif classification == LeadClassification.BOTH:
                result["message"] = "Lead classified as both and ATS + CRM records created"
            else:
                result["message"] = "Lead classification updated"

        logger.info(f"Classified lead {lead_id} as {classification}")
        return result

    # ========================================================================
    # Lead Research
    # ========================================================================

    async def research_lead(
        self,
        lead_id: str,
        research_depth: ResearchDepth = ResearchDepth.MEDIUM
    ) -> str:
        """
        Perform comprehensive research on a lead

        Args:
            lead_id: Lead ID
            research_depth: Depth of research (shallow, medium, deep)

        Returns:
            research_id
        """
        lead = await self.get_lead(lead_id)
        if not lead:
            raise ValueError(f"Lead {lead_id} not found")

        # Update lead status
        await self.update_lead_status(lead_id, LeadStatus.RESEARCHING)

        # Execute research workflow
        research_report = await self._execute_research_workflow(lead, research_depth)

        # Store research report
        research_id = f"research_{secrets.token_urlsafe(16)}"
        research_report.research_id = research_id

        research_data = research_report.model_dump()
        research_data["researched_at"] = research_report.researched_at.isoformat()
        research_data["lead_score"]["calculated_at"] = research_report.lead_score.calculated_at.isoformat()
        research_data["company_data"] = research_report.company_data.model_dump()

        await self.db.insert_one(self.research_collection, research_data)

        # Update lead with research findings
        is_qualified = research_report.lead_score.score >= 70
        await self.db.update_one(
            self.leads_collection,
            {"lead_id": lead_id},
            {
                "$set": {
                    "lead_score": research_report.lead_score.score,
                    "status": LeadStatus.QUALIFIED if is_qualified else LeadStatus.NEW,
                    "updated_at": datetime.utcnow().isoformat()
                },
                "$push": {"research_history": research_id}
            }
        )

        # Publish events
        await publish_event(
            EventType.LEAD_ENRICHED,
            {
                "lead_id": lead_id,
                "research_id": research_id,
                "score": research_report.lead_score.score,
                "research_depth": research_depth.value,
                "company_data": research_report.company_data.model_dump(),
            },
            source="lead_generation_service"
        )

        if is_qualified:
            await publish_event(
                EventType.LEAD_QUALIFIED,
                {
                    "lead_id": lead_id,
                    "score": research_report.lead_score.score,
                    "company_name": lead.company_name,
                    "industry": lead.industry,
                },
                source="lead_generation_service"
            )

        logger.info(f"Completed research {research_id} for lead {lead_id}")
        return research_id

    async def _execute_research_workflow(
        self,
        lead: Lead,
        research_depth: ResearchDepth
    ) -> ResearchReport:
        """
        Execute multi-stage research workflow

        Stages:
        1. Company Discovery (OSINT tools)
        2. Personnel Identification (LinkedIn, company website)
        3. News & Events Monitoring (RSS feeds, news APIs)
        4. Social Media Analysis (Twitter, LinkedIn activity)
        5. Technology Stack Detection (BuiltWith, Wappalyzer)
        6. Competitive Landscape (market research)
        7. Contact Information Enrichment
        8. Engagement Opportunity Identification
        """
        # Stage 1: Company Discovery
        company_data = await self._discover_company(lead)

        # Stage 2: Personnel Identification
        key_people = await self._identify_decision_makers(company_data, research_depth)

        # Stage 3: News Monitoring
        recent_news = await self._monitor_news(lead.company_name, research_depth)

        # Stage 4: Social Analysis
        social_activity = await self._analyze_social_media(company_data, research_depth)

        # Stage 5: Tech Stack (only for deep research)
        tech_stack = None
        if research_depth == ResearchDepth.DEEP and company_data.website:
            tech_stack = await self._detect_technology(company_data.website)

        # Stage 6: Competitive Analysis (only for medium/deep)
        competitors = []
        if research_depth in [ResearchDepth.MEDIUM, ResearchDepth.DEEP]:
            competitors = await self._analyze_competitors(company_data)

        # Stage 7: Contact Enrichment (only for deep)
        if research_depth == ResearchDepth.DEEP:
            key_people = await self._enrich_contacts(key_people)

        # Stage 8: Opportunities
        opportunities = await self._identify_opportunities(
            company_data, recent_news, social_activity
        )

        # Calculate lead score
        lead_score = await self._calculate_lead_score(
            company_data, key_people, recent_news, social_activity, opportunities
        )

        return ResearchReport(
            research_id="",  # Will be set by caller
            lead_id=lead.lead_id,
            company_data=company_data,
            key_people=key_people,
            recent_news=recent_news,
            social_activity=social_activity,
            tech_stack=tech_stack,
            competitors=competitors,
            opportunities=opportunities,
            lead_score=lead_score,
            research_depth=research_depth,
        )

    async def _discover_company(self, lead: Lead) -> CompanyData:
        """Discover company information using OSINT tools"""
        if self.mock_mode:
            return CompanyData(
                company_name=lead.company_name,
                website=f"https://{lead.company_name.lower().replace(' ', '')}.com",
                industry=lead.industry,
                location=lead.location,
                company_size=lead.company_size or "51-200",
                revenue="$10M-$50M",
                funding="Series A",
                description=f"{lead.company_name} is a leading company in the {lead.industry} industry.",
                founded_year=2018,
            )

        # Real implementation would use OSINT tools
        # investigation_id = await self.osint_service.create_investigation(...)
        # company_data = await self.osint_service.collect_data(...)
        return CompanyData(
            company_name=lead.company_name,
            industry=lead.industry,
            location=lead.location,
        )

    async def _identify_decision_makers(
        self,
        company_data: CompanyData,
        research_depth: ResearchDepth
    ) -> List[KeyPerson]:
        """Identify key decision makers"""
        if self.mock_mode:
            people = [
                KeyPerson(
                    name="John Smith",
                    title="CEO",
                    linkedin=f"https://linkedin.com/in/johnsmith",
                    email="john@company.com" if research_depth == ResearchDepth.DEEP else None,
                ),
                KeyPerson(
                    name="Jane Doe",
                    title="CTO",
                    linkedin=f"https://linkedin.com/in/janedoe",
                ),
            ]

            if research_depth == ResearchDepth.DEEP:
                people.append(
                    KeyPerson(
                        name="Bob Johnson",
                        title="VP of Sales",
                        linkedin=f"https://linkedin.com/in/bobjohnson",
                        email="bob@company.com",
                    )
                )

            return people

        # Real implementation would scrape LinkedIn, company website
        return []

    async def _monitor_news(
        self,
        company_name: str,
        research_depth: ResearchDepth
    ) -> List[NewsItem]:
        """Monitor recent news and press releases"""
        if self.mock_mode:
            news_count = 1 if research_depth == ResearchDepth.SHALLOW else 3
            return [
                NewsItem(
                    title=f"{company_name} Announces New Product Launch",
                    url=f"https://news.example.com/article{i}",
                    source="TechCrunch",
                    published_date=(datetime.utcnow() - timedelta(days=i*7)).isoformat(),
                    summary=f"Recent news about {company_name}",
                )
                for i in range(news_count)
            ]

        # Real implementation would use news APIs, RSS feeds
        return []

    async def _analyze_social_media(
        self,
        company_data: CompanyData,
        research_depth: ResearchDepth
    ) -> List[SocialActivity]:
        """Analyze social media presence and activity"""
        if self.mock_mode:
            platforms = ["linkedin", "twitter"]
            if research_depth == ResearchDepth.DEEP:
                platforms.append("facebook")

            return [
                SocialActivity(
                    platform=platform,
                    followers=5000 + (i * 1000),
                    engagement_rate=0.05 + (i * 0.01),
                    recent_posts=[
                        f"Recent post {j} on {platform}"
                        for j in range(2 if research_depth == ResearchDepth.DEEP else 1)
                    ],
                    activity_level="medium",
                )
                for i, platform in enumerate(platforms)
            ]

        # Real implementation would use social media APIs
        return []

    async def _detect_technology(self, website: str) -> TechnologyStack:
        """Detect technology stack"""
        if self.mock_mode:
            return TechnologyStack(
                technologies=["React", "Node.js", "PostgreSQL"],
                frameworks=["Next.js", "Express"],
                platforms=["AWS", "Vercel"],
                tools=["GitHub", "Stripe", "SendGrid"],
            )

        # Real implementation would use BuiltWith, Wappalyzer APIs
        return TechnologyStack()

    async def _analyze_competitors(self, company_data: CompanyData) -> List[str]:
        """Analyze competitive landscape"""
        if self.mock_mode:
            return [
                f"Competitor A in {company_data.industry}",
                f"Competitor B in {company_data.industry}",
                f"Competitor C in {company_data.industry}",
            ]

        # Real implementation would use market research APIs
        return []

    async def _enrich_contacts(self, key_people: List[KeyPerson]) -> List[KeyPerson]:
        """Enrich contact information"""
        if self.mock_mode:
            # Add email/phone to contacts that don't have them
            for person in key_people:
                if not person.email:
                    person.email = f"{person.name.lower().replace(' ', '.')}@company.com"
                if not person.phone:
                    person.phone = "+1-555-0100"

        # Real implementation would use contact enrichment APIs
        return key_people

    async def _identify_opportunities(
        self,
        company_data: CompanyData,
        recent_news: List[NewsItem],
        social_activity: List[SocialActivity]
    ) -> List[Opportunity]:
        """Identify engagement opportunities"""
        opportunities = []

        if self.mock_mode:
            # Analyze news for opportunities
            for news in recent_news:
                if "product launch" in news.title.lower():
                    opportunities.append(
                        Opportunity(
                            opportunity_type="product_launch",
                            description=f"Recent product launch: {news.title}",
                            confidence=0.9,
                            source=news.url,
                        )
                    )
                elif "funding" in news.title.lower():
                    opportunities.append(
                        Opportunity(
                            opportunity_type="funding",
                            description=f"Recent funding round: {news.title}",
                            confidence=0.85,
                            source=news.url,
                        )
                    )

            # Analyze social activity
            high_engagement = [s for s in social_activity if s.engagement_rate and s.engagement_rate > 0.05]
            if high_engagement:
                opportunities.append(
                    Opportunity(
                        opportunity_type="social_engagement",
                        description=f"High social media engagement on {', '.join([s.platform for s in high_engagement])}",
                        confidence=0.75,
                        source="social_media_analysis",
                    )
                )

            # Default opportunity if none found
            if not opportunities:
                opportunities.append(
                    Opportunity(
                        opportunity_type="general_outreach",
                        description="Company is active and growing, good time for outreach",
                        confidence=0.6,
                        source="general_analysis",
                    )
                )

        return opportunities

    async def _calculate_lead_score(
        self,
        company_data: CompanyData,
        key_people: List[KeyPerson],
        recent_news: List[NewsItem],
        social_activity: List[SocialActivity],
        opportunities: List[Opportunity]
    ) -> LeadScore:
        """Calculate lead score based on research findings"""
        factors = {}

        # Company fit (0-30 points)
        company_fit = 0.0
        if company_data.company_size in ["51-200", "201-1000"]:
            company_fit += 15.0
        if company_data.revenue:
            company_fit += 10.0
        if company_data.funding:
            company_fit += 5.0
        factors["company_fit"] = company_fit

        # Decision maker accessibility (0-25 points)
        decision_maker_score = 0.0
        if key_people:
            decision_maker_score += min(len(key_people) * 5.0, 15.0)
        contacts_with_email = [p for p in key_people if p.email]
        if contacts_with_email:
            decision_maker_score += 10.0
        factors["decision_maker_accessibility"] = decision_maker_score

        # Engagement potential (0-25 points)
        engagement_score = 0.0
        if recent_news:
            engagement_score += min(len(recent_news) * 5.0, 15.0)
        avg_engagement = sum([s.engagement_rate or 0 for s in social_activity]) / len(social_activity) if social_activity else 0
        engagement_score += avg_engagement * 100  # Convert to 0-10 scale
        factors["engagement_potential"] = min(engagement_score, 25.0)

        # Timing signals (0-20 points)
        timing_score = 0.0
        for opp in opportunities:
            if opp.opportunity_type in ["funding", "expansion", "hiring"]:
                timing_score += opp.confidence * 10.0
        factors["timing_signals"] = min(timing_score, 20.0)

        # Calculate total score
        total_score = sum(factors.values())

        # Generate recommendations
        recommendations = []
        if total_score >= 80:
            recommendations.append("High-priority lead - immediate outreach recommended")
        elif total_score >= 60:
            recommendations.append("Qualified lead - schedule outreach within 1 week")
        else:
            recommendations.append("Monitor lead for better timing signals")

        if contacts_with_email:
            recommendations.append(f"Direct email contact available for {len(contacts_with_email)} decision makers")

        if opportunities:
            top_opp = max(opportunities, key=lambda o: o.confidence)
            recommendations.append(f"Best opportunity: {top_opp.description}")

        return LeadScore(
            score=total_score,
            factors=factors,
            confidence=0.8,  # Confidence in the scoring
            recommendations=recommendations,
        )

    # ========================================================================
    # Lead Monitoring
    # ========================================================================

    async def start_monitoring(
        self,
        lead_id: str,
        frequency: MonitoringFrequency = MonitoringFrequency.WEEKLY
    ) -> bool:
        """Start continuous monitoring of a lead"""
        # Calculate next check time
        now = datetime.utcnow()
        if frequency == MonitoringFrequency.DAILY:
            next_check = now + timedelta(days=1)
        elif frequency == MonitoringFrequency.WEEKLY:
            next_check = now + timedelta(weeks=1)
        else:  # MONTHLY
            next_check = now + timedelta(days=30)

        result = await self.db.update_one(
            self.leads_collection,
            {"lead_id": lead_id},
            {
                "$set": {
                    "monitoring_active": True,
                    "next_research_date": next_check.isoformat(),
                    "updated_at": now.isoformat()
                }
            }
        )

        logger.info(f"Started monitoring for lead {lead_id} with {frequency} frequency")
        return result is not None

    async def stop_monitoring(self, lead_id: str) -> bool:
        """Stop monitoring a lead"""
        result = await self.db.update_one(
            self.leads_collection,
            {"lead_id": lead_id},
            {
                "$set": {
                    "monitoring_active": False,
                    "next_research_date": None,
                    "updated_at": datetime.utcnow().isoformat()
                }
            }
        )

        logger.info(f"Stopped monitoring for lead {lead_id}")
        return result is not None

    async def check_for_updates(self, lead_id: str) -> List[LeadUpdate]:
        """Check for updates on a lead"""
        lead = await self.get_lead(lead_id)
        if not lead:
            return []

        updates = []

        if self.mock_mode:
            # Simulate finding updates
            updates.append(
                LeadUpdate(
                    update_type="news",
                    description=f"New article published about {lead.company_name}",
                    source="https://news.example.com/article",
                )
            )

        # Real implementation would check news, social media, etc.
        return updates

    # ========================================================================
    # Statistics
    # ========================================================================

    async def get_campaign_stats(self, campaign_id: str) -> Dict[str, Any]:
        """Get campaign statistics"""
        campaign = await self.get_campaign(campaign_id)
        if not campaign:
            return {}

        leads, total = await self.list_leads(campaign_id=campaign_id, page_size=1000)

        qualified_leads = [l for l in leads if l.status == LeadStatus.QUALIFIED]
        contacted_leads = [l for l in leads if l.status == LeadStatus.CONTACTED]
        converted_leads = [l for l in leads if l.status == LeadStatus.CONVERTED]

        avg_score = sum([l.lead_score for l in leads]) / len(leads) if leads else 0.0

        # Get top opportunities
        all_opportunities = []
        for lead in leads:
            all_opportunities.extend(lead.opportunities)

        top_opportunities = sorted(
            all_opportunities,
            key=lambda o: o.confidence,
            reverse=True
        )[:5]

        return {
            "campaign_id": campaign_id,
            "total_leads": total,
            "qualified_leads": len(qualified_leads),
            "contacted_leads": len(contacted_leads),
            "converted_leads": len(converted_leads),
            "avg_lead_score": round(avg_score, 2),
            "top_opportunities": top_opportunities,
        }

