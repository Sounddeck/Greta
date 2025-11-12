"""
Integration Tests for Lead Generation Module

Tests the complete lead generation workflow including:
- Campaign creation and management
- Lead creation and management
- Multi-stage research workflow
- Lead scoring and qualification
- Automated monitoring
"""

import pytest
import sys
import os
from datetime import datetime, timedelta
from typing import List

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from database import Database
from services.lead_generation_service import LeadGenerationService
from models.lead_generation_models import (
    LeadCampaignConfig,
    LeadStatus,
    ResearchDepth,
    MonitoringFrequency,
    CompanySize,
)


@pytest.fixture
async def db():
    """Create test database connection"""
    database = Database()
    await database.connect()
    yield database
    # Cleanup after tests
    await database.close()


@pytest.fixture
async def lead_service(db):
    """Create lead generation service"""
    return LeadGenerationService(db)


@pytest.fixture
def sample_campaign_config():
    """Sample campaign configuration"""
    return LeadCampaignConfig(
        campaign_name="Test AI Startups Campaign",
        target_industry="Artificial Intelligence",
        target_location="San Francisco, CA",
        company_size=CompanySize.SMALL,
        keywords=["AI", "machine learning", "LLM", "automation"],
        research_depth=ResearchDepth.MEDIUM,
        monitoring_frequency=MonitoringFrequency.WEEKLY,
        max_leads=50,
        auto_research=True
    )


@pytest.mark.integration
class TestCampaignManagement:
    """Test campaign creation and management"""
    
    @pytest.mark.asyncio
    async def test_create_campaign(self, lead_service, sample_campaign_config):
        """Test creating a new campaign"""
        campaign_id = await lead_service.create_campaign(
            user_id="test_user_123",
            config=sample_campaign_config
        )
        
        assert campaign_id is not None
        assert campaign_id.startswith("campaign_")
        
        # Verify campaign was created
        campaign = await lead_service.get_campaign(campaign_id)
        assert campaign is not None
        assert campaign.campaign_id == campaign_id
        assert campaign.user_id == "test_user_123"
        assert campaign.config.campaign_name == "Test AI Startups Campaign"
        assert campaign.status == "active"
    
    @pytest.mark.asyncio
    async def test_list_campaigns(self, lead_service, sample_campaign_config):
        """Test listing campaigns for a user"""
        # Create multiple campaigns
        campaign_id_1 = await lead_service.create_campaign(
            user_id="test_user_456",
            config=sample_campaign_config
        )
        
        config_2 = sample_campaign_config.model_copy()
        config_2.campaign_name = "Second Campaign"
        campaign_id_2 = await lead_service.create_campaign(
            user_id="test_user_456",
            config=config_2
        )
        
        # List campaigns
        campaigns = await lead_service.list_campaigns("test_user_456")
        
        assert len(campaigns) >= 2
        campaign_ids = [c.campaign_id for c in campaigns]
        assert campaign_id_1 in campaign_ids
        assert campaign_id_2 in campaign_ids
    
    @pytest.mark.asyncio
    async def test_get_campaign(self, lead_service, sample_campaign_config):
        """Test retrieving a specific campaign"""
        campaign_id = await lead_service.create_campaign(
            user_id="test_user_789",
            config=sample_campaign_config
        )
        
        campaign = await lead_service.get_campaign(campaign_id)
        
        assert campaign is not None
        assert campaign.campaign_id == campaign_id
        assert campaign.config.target_industry == "Artificial Intelligence"


@pytest.mark.integration
class TestLeadManagement:
    """Test lead creation and management"""
    
    @pytest.mark.asyncio
    async def test_create_lead(self, lead_service, sample_campaign_config):
        """Test creating a new lead"""
        # Create campaign first
        campaign_id = await lead_service.create_campaign(
            user_id="test_user_lead",
            config=sample_campaign_config
        )
        
        # Create lead
        lead_id = await lead_service.create_lead(
            campaign_id=campaign_id,
            user_id="test_user_lead",
            company_name="Acme AI Corp",
            industry="Artificial Intelligence",
            location="San Francisco, CA",
            website="https://acmeai.com",
            company_size="50-200"
        )
        
        assert lead_id is not None
        assert lead_id.startswith("lead_")
        
        # Verify lead was created
        lead = await lead_service.get_lead(lead_id)
        assert lead is not None
        assert lead.lead_id == lead_id
        assert lead.company_name == "Acme AI Corp"
        assert lead.status == LeadStatus.NEW
    
    @pytest.mark.asyncio
    async def test_list_leads(self, lead_service, sample_campaign_config):
        """Test listing leads with filtering"""
        # Create campaign
        campaign_id = await lead_service.create_campaign(
            user_id="test_user_list",
            config=sample_campaign_config
        )
        
        # Create multiple leads
        lead_ids = []
        for i in range(5):
            lead_id = await lead_service.create_lead(
                campaign_id=campaign_id,
                user_id="test_user_list",
                company_name=f"Company {i}",
                industry="AI",
                location="SF"
            )
            lead_ids.append(lead_id)
        
        # List all leads
        leads, total = await lead_service.list_leads(page_size=100)
        assert total >= 5
        
        # Filter by campaign
        campaign_leads, campaign_total = await lead_service.list_leads(
            campaign_id=campaign_id,
            page_size=100
        )
        assert campaign_total >= 5
        
        # Filter by status
        new_leads, new_total = await lead_service.list_leads(
            status=LeadStatus.NEW,
            page_size=100
        )
        assert new_total >= 5
    
    @pytest.mark.asyncio
    async def test_update_lead_status(self, lead_service, sample_campaign_config):
        """Test updating lead status"""
        # Create campaign and lead
        campaign_id = await lead_service.create_campaign(
            user_id="test_user_status",
            config=sample_campaign_config
        )
        
        lead_id = await lead_service.create_lead(
            campaign_id=campaign_id,
            user_id="test_user_status",
            company_name="Status Test Corp",
            industry="AI",
            location="SF"
        )
        
        # Update status
        await lead_service.update_lead_status(lead_id, LeadStatus.CONTACTED)
        
        # Verify update
        lead = await lead_service.get_lead(lead_id)
        assert lead.status == LeadStatus.CONTACTED


@pytest.mark.integration
class TestResearchWorkflow:
    """Test multi-stage research workflow"""
    
    @pytest.mark.asyncio
    async def test_research_lead_shallow(self, lead_service, sample_campaign_config):
        """Test shallow research workflow"""
        # Create campaign and lead
        campaign_id = await lead_service.create_campaign(
            user_id="test_user_research",
            config=sample_campaign_config
        )
        
        lead_id = await lead_service.create_lead(
            campaign_id=campaign_id,
            user_id="test_user_research",
            company_name="Research Test Inc",
            industry="AI",
            location="SF",
            website="https://researchtest.com"
        )
        
        # Execute shallow research
        research_id = await lead_service.research_lead(
            lead_id=lead_id,
            research_depth=ResearchDepth.SHALLOW
        )
        
        assert research_id is not None
        assert research_id.startswith("research_")
        
        # Verify lead was updated with research
        lead = await lead_service.get_lead(lead_id)
        assert research_id in lead.research_history
        assert lead.lead_score > 0  # Should have been scored
    
    @pytest.mark.asyncio
    async def test_research_lead_medium(self, lead_service, sample_campaign_config):
        """Test medium depth research workflow"""
        # Create campaign and lead
        campaign_id = await lead_service.create_campaign(
            user_id="test_user_medium",
            config=sample_campaign_config
        )
        
        lead_id = await lead_service.create_lead(
            campaign_id=campaign_id,
            user_id="test_user_medium",
            company_name="Medium Research Corp",
            industry="AI",
            location="SF",
            website="https://mediumresearch.com"
        )
        
        # Execute medium research
        research_id = await lead_service.research_lead(
            lead_id=lead_id,
            research_depth=ResearchDepth.MEDIUM
        )
        
        assert research_id is not None
        
        # Verify comprehensive data was gathered
        lead = await lead_service.get_lead(lead_id)
        assert len(lead.key_people) > 0  # Should have identified key people
        assert len(lead.recent_news) > 0  # Should have news items
    
    @pytest.mark.asyncio
    async def test_research_lead_deep(self, lead_service, sample_campaign_config):
        """Test deep research workflow"""
        # Create campaign and lead
        campaign_id = await lead_service.create_campaign(
            user_id="test_user_deep",
            config=sample_campaign_config
        )
        
        lead_id = await lead_service.create_lead(
            campaign_id=campaign_id,
            user_id="test_user_deep",
            company_name="Deep Research LLC",
            industry="AI",
            location="SF",
            website="https://deepresearch.com"
        )
        
        # Execute deep research
        research_id = await lead_service.research_lead(
            lead_id=lead_id,
            research_depth=ResearchDepth.DEEP
        )
        
        assert research_id is not None
        
        # Verify comprehensive data including tech stack
        lead = await lead_service.get_lead(lead_id)
        assert len(lead.key_people) > 0
        assert len(lead.recent_news) > 0
        # In mock mode, tech stack should be populated
        # assert lead.tech_stack is not None


@pytest.mark.integration
class TestMonitoring:
    """Test automated monitoring functionality"""
    
    @pytest.mark.asyncio
    async def test_start_monitoring(self, lead_service, sample_campaign_config):
        """Test starting automated monitoring"""
        # Create campaign and lead
        campaign_id = await lead_service.create_campaign(
            user_id="test_user_monitor",
            config=sample_campaign_config
        )
        
        lead_id = await lead_service.create_lead(
            campaign_id=campaign_id,
            user_id="test_user_monitor",
            company_name="Monitor Test Corp",
            industry="AI",
            location="SF"
        )
        
        # Start monitoring
        await lead_service.start_monitoring(
            lead_id=lead_id,
            frequency=MonitoringFrequency.DAILY
        )
        
        # Verify monitoring is active
        lead = await lead_service.get_lead(lead_id)
        assert lead.monitoring_active is True
        assert lead.next_research_date is not None
    
    @pytest.mark.asyncio
    async def test_stop_monitoring(self, lead_service, sample_campaign_config):
        """Test stopping automated monitoring"""
        # Create campaign and lead
        campaign_id = await lead_service.create_campaign(
            user_id="test_user_stop",
            config=sample_campaign_config
        )
        
        lead_id = await lead_service.create_lead(
            campaign_id=campaign_id,
            user_id="test_user_stop",
            company_name="Stop Monitor Corp",
            industry="AI",
            location="SF"
        )
        
        # Start then stop monitoring
        await lead_service.start_monitoring(lead_id, MonitoringFrequency.WEEKLY)
        await lead_service.stop_monitoring(lead_id)
        
        # Verify monitoring is stopped
        lead = await lead_service.get_lead(lead_id)
        assert lead.monitoring_active is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])

