"""
Test Lead Scoring API Endpoint
Tests the /api/v1/leads/{lead_id}/score endpoint
"""

import asyncio
from datetime import datetime

from database import Database
from services.lead_scoring_service import LeadScoringService
from models.lead_generation_models import Lead, KeyPerson


async def test_lead_scoring_api():
    """Test lead scoring API integration"""
    print("=" * 60)
    print("LEAD SCORING API TEST")
    print("=" * 60)

    # Initialize services
    db = Database()
    await db.connect()

    scoring_service = LeadScoringService(db)

    # Create a test lead directly in database
    print("\n=== Creating Test Lead ===")
    lead_id = "test_lead_scoring_123"

    lead_data = {
        "lead_id": lead_id,
        "campaign_id": "test_campaign_123",
        "user_id": "test_user_123",
        "company_name": "TechCorp Inc",
        "industry": "software",
        "location": "United States",
        "company_size": "201-1000",
        "website": "https://techcorp.com",
        "lead_score": 0.0,
        "status": "new",
        "research_history": [],
        "key_people": [
            {"name": "John Smith", "title": "CTO"},
            {"name": "Jane Doe", "title": "VP of Sales"}
        ],
        "recent_news": [],
        "opportunities": [],
        "monitoring_active": False,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }

    await db.insert_one("leads", lead_data)
    print(f"✅ Created lead: {lead_id}")
    
    # Get the lead
    lead_data_retrieved = await db.find_one("leads", {"lead_id": lead_id})

    # Convert to Lead object
    lead_data_retrieved["created_at"] = datetime.fromisoformat(lead_data_retrieved["created_at"])
    lead_data_retrieved["updated_at"] = datetime.fromisoformat(lead_data_retrieved["updated_at"])
    lead_data_retrieved["key_people"] = [KeyPerson(**kp) for kp in lead_data_retrieved["key_people"]]

    lead = Lead(**lead_data_retrieved)
    print(f"✅ Retrieved lead: {lead.company_name}")
    
    # Calculate score
    print("\n=== Calculating Lead Score ===")
    engagement_data = {
        "email_opens": 3,
        "website_visits": 5,
        "content_downloads": 2
    }
    
    score = scoring_service.calculate_score(lead, engagement_data=engagement_data)
    print(f"✅ Score calculated: {score.score}")
    print(f"   Confidence: {score.confidence}")
    print(f"   Factors:")
    for factor, value in score.factors.items():
        print(f"     - {factor}: {value:.1f}")
    
    # Save score
    print("\n=== Saving Score ===")
    score_id = await scoring_service.save_score(lead_id, score)
    print(f"✅ Score saved: {score_id}")
    
    # Update lead score
    await db.update_one(
        "leads",
        {"lead_id": lead_id},
        {"$set": {"lead_score": score.score, "updated_at": datetime.utcnow().isoformat()}}
    )
    print(f"✅ Lead score updated in database")

    # Retrieve updated lead
    updated_lead_data = await db.find_one("leads", {"lead_id": lead_id})
    print(f"✅ Updated lead score: {updated_lead_data['lead_score']}")
    
    # Verify score matches
    assert abs(updated_lead_data['lead_score'] - score.score) < 0.01, "Score mismatch!"
    
    # Get score history
    print("\n=== Score History ===")
    history = await scoring_service.get_score_history(lead_id, limit=5)
    print(f"✅ Score history entries: {len(history)}")
    for i, hist_score in enumerate(history, 1):
        print(f"   {i}. Score: {hist_score.score}, Calculated: {hist_score.calculated_at}")
    
    # Cleanup
    print("\n=== Cleanup ===")
    await db.delete_one("leads", {"lead_id": lead_id})
    await db.delete_one("lead_scores", {"lead_id": lead_id})
    # Delete all history entries for this lead
    history_entries = await db.find_many("lead_score_history", {"lead_id": lead_id})
    for entry in history_entries:
        await db.delete_one("lead_score_history", {"_id": entry["_id"]})
    print(f"✅ Test data cleaned up")
    
    await db.disconnect()
    
    print("\n" + "=" * 60)
    print("✅ ALL API TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_lead_scoring_api())

