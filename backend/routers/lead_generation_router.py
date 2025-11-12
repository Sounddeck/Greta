"""
Lead Generation Router

API endpoints for automated lead generation and research.
"""

from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Optional
from datetime import datetime

from database import Database, get_database
from services.lead_generation_service import LeadGenerationService
from services.lead_scoring_service import LeadScoringService
from models.lead_generation_models import (
    CreateCampaignRequest,
    CreateCampaignResponse,
    LeadCampaign,
    Lead,
    LeadStatus,
    LeadClassification,
    ResearchLeadRequest,
    ResearchLeadResponse,
    LeadListResponse,
    LeadScoreResponse,
    CampaignStatsResponse,
    MonitoringStatusResponse,
    MonitoringFrequency,
    LeadGenerationHealthResponse,
    ClassifyLeadRequest,
    ClassifyLeadResponse,
)

router = APIRouter(prefix="/api/v1/leads", tags=["Lead Generation"])


# ============================================================================
# Campaign Endpoints
# ============================================================================

@router.post("/campaigns", response_model=CreateCampaignResponse)
async def create_campaign(
    request: CreateCampaignRequest,
    db: Database = Depends(get_database)
):
    """
    Create a new lead generation campaign
    
    Creates an automated lead generation campaign with specified targeting criteria.
    The campaign will continuously discover and research leads based on the configuration.
    """
    service = LeadGenerationService(db)
    campaign_id = await service.create_campaign(request.user_id, request.config)
    
    return CreateCampaignResponse(
        campaign_id=campaign_id,
        message=f"Campaign '{request.config.campaign_name}' created successfully"
    )


@router.get("/campaigns/{campaign_id}", response_model=LeadCampaign)
async def get_campaign(
    campaign_id: str,
    db: Database = Depends(get_database)
):
    """Get campaign details"""
    service = LeadGenerationService(db)
    campaign = await service.get_campaign(campaign_id)
    
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    return campaign


@router.get("/campaigns", response_model=list[LeadCampaign])
async def list_campaigns(
    user_id: str = Query(..., description="User ID"),
    db: Database = Depends(get_database)
):
    """List all campaigns for a user"""
    service = LeadGenerationService(db)
    campaigns = await service.list_campaigns(user_id)
    return campaigns


@router.get("/campaigns/{campaign_id}/stats", response_model=CampaignStatsResponse)
async def get_campaign_stats(
    campaign_id: str,
    db: Database = Depends(get_database)
):
    """Get campaign statistics"""
    service = LeadGenerationService(db)
    stats = await service.get_campaign_stats(campaign_id)
    
    if not stats:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    return CampaignStatsResponse(**stats)


# ============================================================================
# Lead Endpoints
# ============================================================================

@router.post("/", response_model=dict)
async def create_lead(
    campaign_id: str = Query(..., description="Campaign ID"),
    user_id: str = Query(..., description="User ID"),
    company_name: str = Query(..., description="Company name"),
    industry: str = Query(..., description="Industry"),
    location: str = Query(..., description="Location"),
    company_size: Optional[str] = Query(None, description="Company size"),
    website: Optional[str] = Query(None, description="Website URL"),
    db: Database = Depends(get_database)
):
    """
    Create a new lead
    
    Manually add a lead to a campaign for research and tracking.
    """
    service = LeadGenerationService(db)
    lead_id = await service.create_lead(
        campaign_id=campaign_id,
        user_id=user_id,
        company_name=company_name,
        industry=industry,
        location=location,
        company_size=company_size,
        website=website,
    )
    
    return {
        "lead_id": lead_id,
        "message": f"Lead '{company_name}' created successfully"
    }


@router.get("/{lead_id}", response_model=Lead)
async def get_lead(
    lead_id: str,
    db: Database = Depends(get_database)
):
    """Get lead details"""
    service = LeadGenerationService(db)
    lead = await service.get_lead(lead_id)
    
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    return lead


@router.get("/", response_model=LeadListResponse)
async def list_leads(
    campaign_id: Optional[str] = Query(None, description="Filter by campaign ID"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    status: Optional[LeadStatus] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    db: Database = Depends(get_database)
):
    """
    List leads with filtering and pagination
    
    Returns a paginated list of leads with optional filtering by campaign, user, or status.
    """
    service = LeadGenerationService(db)
    leads, total = await service.list_leads(
        campaign_id=campaign_id,
        user_id=user_id,
        status=status,
        page=page,
        page_size=page_size
    )
    
    return LeadListResponse(
        leads=leads,
        total=total,
        page=page,
        page_size=page_size
    )


@router.patch("/{lead_id}/status")
async def update_lead_status(
    lead_id: str,
    status: LeadStatus = Query(..., description="New status"),
    db: Database = Depends(get_database)
):
    """Update lead status"""
    service = LeadGenerationService(db)
    success = await service.update_lead_status(lead_id, status)

    if not success:
        raise HTTPException(status_code=404, detail="Lead not found")

    return {"message": f"Lead status updated to {status}"}


@router.post("/{lead_id}/score", response_model=LeadScoreResponse)
async def score_lead(
    lead_id: str,
    engagement_opens: int = Query(0, description="Email opens count"),
    engagement_visits: int = Query(0, description="Website visits count"),
    engagement_downloads: int = Query(0, description="Content downloads count"),
    db: Database = Depends(get_database)
):
    """
    Calculate lead score based on multiple factors

    Scoring factors (0-100 each):
    - Company Size (25%): Based on employee count
    - Industry Match (20%): Alignment with target industries
    - Engagement (20%): Email opens, website visits, content downloads
    - Budget Indicators (15%): Funding, revenue signals
    - Decision Maker (10%): Contact level (C-suite, VP, Manager)
    - Geographic Fit (10%): Location preference match

    Returns weighted score (0-100), factor breakdown, confidence, and recommendations.
    """
    # Get lead
    lead_service = LeadGenerationService(db)
    lead = await lead_service.get_lead(lead_id)

    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")

    # Calculate score
    scoring_service = LeadScoringService(db)
    engagement_data = {
        "email_opens": engagement_opens,
        "website_visits": engagement_visits,
        "content_downloads": engagement_downloads,
    }

    score = scoring_service.calculate_score(lead, engagement_data=engagement_data)

    # Save score to database
    await scoring_service.save_score(lead_id, score)

    # Update lead's score field
    await lead_service.update_lead_score(lead_id, score.score)

    return LeadScoreResponse(
        lead_id=lead_id,
        score=score.score,
        factors=score.factors,
        confidence=score.confidence,
        recommendations=score.recommendations,
        calculated_at=score.calculated_at
    )


@router.post("/{lead_id}/classify", response_model=ClassifyLeadResponse)
async def classify_lead(
    lead_id: str,
    request: ClassifyLeadRequest,
    db: Database = Depends(get_database)
):
    """
    Classify a lead and automatically create ATS candidate or CRM contact

    Classification options:
    - **candidate**: Creates an ATS candidate record for job recruitment
    - **client**: Creates a CRM contact record for sales/customer management
    - **both**: Creates both ATS candidate and CRM contact records
    - **unclassified**: Removes classification (no records created)

    When a lead is classified:
    - The lead's classification field is updated
    - Appropriate records are created in ATS and/or CRM systems
    - The lead is linked to the created records via ats_candidate_id and/or crm_contact_id
    - Events are published for downstream processing

    Returns the classification result including any created record IDs.
    """
    service = LeadGenerationService(db)

    try:
        result = await service.classify_lead(lead_id, request.classification)

        return ClassifyLeadResponse(
            lead_id=result["lead_id"],
            classification=result["classification"],
            ats_candidate_id=result.get("ats_candidate_id"),
            crm_contact_id=result.get("crm_contact_id"),
            message=result["message"]
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


# ============================================================================
# Research Endpoints
# ============================================================================

@router.post("/{lead_id}/research", response_model=ResearchLeadResponse)
async def research_lead(
    lead_id: str,
    request: ResearchLeadRequest,
    db: Database = Depends(get_database)
):
    """
    Perform comprehensive research on a lead
    
    Executes a multi-stage research workflow to gather intelligence about the lead:
    - Company information and background
    - Key personnel and decision makers
    - Recent news and press releases
    - Social media presence and activity
    - Technology stack (deep research only)
    - Competitive landscape
    - Contact information enrichment
    - Engagement opportunities
    
    The research depth determines how comprehensive the research will be:
    - shallow: Basic company info and 1-2 key people
    - medium: Company info, key people, news, social media
    - deep: Comprehensive research including tech stack and contact enrichment
    """
    service = LeadGenerationService(db)
    
    try:
        research_id = await service.research_lead(lead_id, request.research_depth)
        
        return ResearchLeadResponse(
            research_id=research_id,
            lead_id=lead_id,
            status="completed",
            message=f"Research completed with {request.research_depth} depth"
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}")


# ============================================================================
# Monitoring Endpoints
# ============================================================================

@router.post("/{lead_id}/monitoring/start")
async def start_monitoring(
    lead_id: str,
    frequency: MonitoringFrequency = Query(
        MonitoringFrequency.WEEKLY,
        description="Monitoring frequency"
    ),
    db: Database = Depends(get_database)
):
    """
    Start continuous monitoring of a lead
    
    Enables automated monitoring to track changes and updates:
    - Company news and announcements
    - Personnel changes
    - Funding rounds
    - Product launches
    - Social media activity
    - Website changes
    """
    service = LeadGenerationService(db)
    success = await service.start_monitoring(lead_id, frequency)
    
    if not success:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    return {"message": f"Monitoring started with {frequency} frequency"}


@router.post("/{lead_id}/monitoring/stop")
async def stop_monitoring(
    lead_id: str,
    db: Database = Depends(get_database)
):
    """Stop monitoring a lead"""
    service = LeadGenerationService(db)
    success = await service.stop_monitoring(lead_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    return {"message": "Monitoring stopped"}


@router.get("/{lead_id}/monitoring/status", response_model=MonitoringStatusResponse)
async def get_monitoring_status(
    lead_id: str,
    db: Database = Depends(get_database)
):
    """Get monitoring status for a lead"""
    service = LeadGenerationService(db)
    lead = await service.get_lead(lead_id)
    
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    # Check for updates
    updates = await service.check_for_updates(lead_id)
    
    return MonitoringStatusResponse(
        lead_id=lead_id,
        monitoring_active=lead.monitoring_active,
        frequency=MonitoringFrequency.WEEKLY,  # TODO: Store frequency in lead
        last_check=lead.updated_at,
        next_check=lead.next_research_date,
        updates_found=len(updates)
    )


# ============================================================================
# Health Check
# ============================================================================

@router.get("/health", response_model=LeadGenerationHealthResponse)
async def health_check(db: Database = Depends(get_database)):
    """Lead generation service health check"""
    service = LeadGenerationService(db)
    
    # Get counts
    campaigns = await service.list_campaigns("all")  # Mock user_id
    all_leads, total_leads = await service.list_leads(page_size=1000)
    
    return LeadGenerationHealthResponse(
        status="healthy",
        message="Lead generation service is operational",
        timestamp=datetime.utcnow(),
        active_campaigns=len([c for c in campaigns if c.status == "active"]),
        total_leads=total_leads,
        researches_today=0,  # TODO: Track daily research count
    )

