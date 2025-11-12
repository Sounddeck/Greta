"""
Tests for Lead Scoring Service
Validates scoring algorithm, edge cases, and performance
"""

import time
from datetime import datetime

from models.lead_generation_models import Lead, KeyPerson, Opportunity
from services.lead_scoring_service import LeadScoringService


def create_test_lead(**kwargs) -> Lead:
    """Create a test lead with default values"""
    defaults = {
        "lead_id": "test_lead_123",
        "campaign_id": "test_campaign_123",
        "user_id": "test_user_123",
        "company_name": "Test Company",
        "industry": "technology",
        "location": "United States",
        "company_size": "51-200",
        "website": "https://testcompany.com",
        "lead_score": 0.0,
        "status": "new",
        "research_history": [],
        "key_people": [],
        "recent_news": [],
        "opportunities": [],
        "monitoring_active": False,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    defaults.update(kwargs)
    return Lead(**defaults)


def test_company_size_scoring():
    """Test company size scoring"""
    print("\n=== Test 1: Company Size Scoring ===")
    
    service = LeadScoringService()
    
    test_cases = [
        ("1-10", 20.0),
        ("11-50", 40.0),
        ("51-200", 60.0),
        ("201-1000", 80.0),
        ("1000+", 100.0),
        ("Enterprise", 100.0),
        (None, 50.0),  # Missing data
    ]
    
    for company_size, expected_score in test_cases:
        lead = create_test_lead(company_size=company_size)
        score = service._score_company_size(lead)
        assert score == expected_score, f"Company size '{company_size}' should score {expected_score}, got {score}"
        print(f"✅ Company size '{company_size}': {score} points")
    
    print("✅ All company size tests passed")


def test_industry_scoring():
    """Test industry scoring"""
    print("\n=== Test 2: Industry Scoring ===")

    service = LeadScoringService()

    test_cases = [
        ("technology", 100.0),  # Target industry
        ("software", 100.0),  # Target industry
        ("fintech", 100.0),  # Target industry
        ("tech services", 60.0),  # Related industry
        ("digital marketing", 60.0),  # Related industry
        ("manufacturing", 20.0),  # Other industry
        ("retail", 20.0),  # Other industry
        ("", 20.0),  # Missing data (empty string)
    ]
    
    for industry, expected_score in test_cases:
        lead = create_test_lead(industry=industry)
        score = service._score_industry(lead)
        assert score == expected_score, f"Industry '{industry}' should score {expected_score}, got {score}"
        print(f"✅ Industry '{industry}': {score} points")
    
    print("✅ All industry tests passed")


def test_decision_maker_scoring():
    """Test decision maker scoring"""
    print("\n=== Test 3: Decision Maker Scoring ===")
    
    service = LeadScoringService()
    
    test_cases = [
        ([KeyPerson(name="John Doe", title="CEO")], 100.0),
        ([KeyPerson(name="Jane Smith", title="CTO")], 100.0),
        ([KeyPerson(name="Bob Johnson", title="VP of Sales")], 70.0),
        ([KeyPerson(name="Alice Brown", title="Director of Marketing")], 70.0),
        ([KeyPerson(name="Charlie Wilson", title="Sales Manager")], 40.0),
        ([KeyPerson(name="David Lee", title="Software Engineer")], 20.0),
        ([], 50.0),  # No key people
    ]
    
    for key_people, expected_score in test_cases:
        lead = create_test_lead(key_people=key_people)
        score = service._score_decision_maker(lead)
        assert score == expected_score, f"Key people {[p.title for p in key_people]} should score {expected_score}, got {score}"
        title = key_people[0].title if key_people else "None"
        print(f"✅ Decision maker '{title}': {score} points")
    
    print("✅ All decision maker tests passed")


def test_geographic_scoring():
    """Test geographic scoring"""
    print("\n=== Test 4: Geographic Scoring ===")

    service = LeadScoringService()

    test_cases = [
        ("United States", 100.0),  # Target region
        ("Canada", 100.0),  # Target region
        ("United Kingdom", 100.0),  # Target region
        ("Germany", 100.0),  # Target region
        ("Asia", 60.0),  # Adjacent region
        ("Australia", 60.0),  # Adjacent region
        ("Africa", 30.0),  # Other region
        ("", 30.0),  # Missing data (empty string)
    ]
    
    for location, expected_score in test_cases:
        lead = create_test_lead(location=location)
        score = service._score_geographic(lead)
        assert score == expected_score, f"Location '{location}' should score {expected_score}, got {score}"
        print(f"✅ Location '{location}': {score} points")
    
    print("✅ All geographic tests passed")


def test_engagement_scoring():
    """Test engagement scoring"""
    print("\n=== Test 5: Engagement Scoring ===")
    
    service = LeadScoringService()
    
    test_cases = [
        ({"email_opens": 0, "website_visits": 0, "content_downloads": 0}, 0.0),
        ({"email_opens": 1, "website_visits": 1, "content_downloads": 1}, 28.0),
        ({"email_opens": 4, "website_visits": 5, "content_downloads": 2}, 100.0),
        ({"email_opens": 10, "website_visits": 10, "content_downloads": 10}, 100.0),  # Capped at 100
    ]
    
    for engagement_data, expected_score in test_cases:
        lead = create_test_lead()
        score = service._score_engagement(lead, engagement_data)
        assert score == expected_score, f"Engagement {engagement_data} should score {expected_score}, got {score}"
        print(f"✅ Engagement {engagement_data}: {score} points")
    
    print("✅ All engagement tests passed")


def test_full_score_calculation():
    """Test full score calculation with all factors"""
    print("\n=== Test 6: Full Score Calculation ===")
    
    service = LeadScoringService()
    
    # High-quality lead
    high_quality_lead = create_test_lead(
        company_size="1000+",  # 100 points * 0.25 = 25
        industry="software",  # 100 points * 0.20 = 20
        location="United States",  # 100 points * 0.10 = 10
        key_people=[KeyPerson(name="John Doe", title="CEO")],  # 100 points * 0.10 = 10
    )
    
    engagement_data = {
        "email_opens": 4,
        "website_visits": 5,
        "content_downloads": 2
    }  # 100 points * 0.20 = 20
    
    score = service.calculate_score(high_quality_lead, engagement_data=engagement_data)
    
    print(f"\n✅ High-Quality Lead Score: {score.score}")
    print(f"   Breakdown:")
    for factor, value in score.factors.items():
        weight = service.WEIGHTS[factor]
        contribution = value * weight
        print(f"   - {factor}: {value:.1f} points (weight: {weight:.0%}, contribution: {contribution:.1f})")
    print(f"   Confidence: {score.confidence:.2f}")
    print(f"   Recommendations: {len(score.recommendations)}")
    
    # Verify score is in valid range
    assert 0 <= score.score <= 100, f"Score should be 0-100, got {score.score}"
    assert score.score >= 70, f"High-quality lead should score >= 70, got {score.score}"
    
    # Low-quality lead
    low_quality_lead = create_test_lead(
        company_size="1-10",  # 20 points * 0.25 = 5
        industry="retail",  # 20 points * 0.20 = 4
        location="Africa",  # 30 points * 0.10 = 3
        key_people=[],  # 50 points * 0.10 = 5
    )
    
    score = service.calculate_score(low_quality_lead, engagement_data={})
    
    print(f"\n✅ Low-Quality Lead Score: {score.score}")
    print(f"   Breakdown:")
    for factor, value in score.factors.items():
        weight = service.WEIGHTS[factor]
        contribution = value * weight
        print(f"   - {factor}: {value:.1f} points (weight: {weight:.0%}, contribution: {contribution:.1f})")
    
    assert score.score < 50, f"Low-quality lead should score < 50, got {score.score}"
    
    print("\n✅ Full score calculation tests passed")


def test_score_range():
    """Test that scores are always in 0-100 range"""
    print("\n=== Test 7: Score Range Validation ===")

    service = LeadScoringService()

    # Test 100 random leads
    for i in range(100):
        lead = create_test_lead(
            company_size=["1-10", "11-50", "51-200", "201-1000", "1000+", ""][i % 6],
            industry=["technology", "software", "retail", "manufacturing", ""][i % 5],
            location=["United States", "Canada", "Asia", "Africa", ""][i % 5],
        )

        score = service.calculate_score(lead)
        assert 0 <= score.score <= 100, f"Score out of range: {score.score}"

    print(f"✅ All 100 test leads scored within 0-100 range")


def test_performance():
    """Test scoring performance"""
    print("\n=== Test 8: Performance Test ===")
    
    service = LeadScoringService()
    
    # Create test lead
    lead = create_test_lead(
        company_size="201-1000",
        industry="technology",
        location="United States",
        key_people=[KeyPerson(name="John Doe", title="CTO")],
    )
    
    # Measure time for 100 score calculations
    num_calculations = 100
    start_time = time.time()
    
    for _ in range(num_calculations):
        score = service.calculate_score(lead)
    
    end_time = time.time()
    total_time_ms = (end_time - start_time) * 1000
    avg_time_ms = total_time_ms / num_calculations
    
    print(f"✅ Performance test completed:")
    print(f"   Total time: {total_time_ms:.2f}ms for {num_calculations} calculations")
    print(f"   Average time per calculation: {avg_time_ms:.2f}ms")
    
    # Check if performance is acceptable (< 100ms per calculation)
    assert avg_time_ms < 100, f"Performance too slow: {avg_time_ms:.2f}ms per calculation"
    print(f"   ✅ Performance acceptable (< 100ms per calculation)")


def test_missing_data_handling():
    """Test handling of missing data"""
    print("\n=== Test 9: Missing Data Handling ===")

    service = LeadScoringService()

    # Lead with minimal data
    minimal_lead = create_test_lead(
        company_size="",
        industry="",
        location="",
        website="",
        key_people=[],
        opportunities=[],
    )
    
    # Should not raise exception
    score = service.calculate_score(minimal_lead)
    
    print(f"✅ Minimal lead scored: {score.score}")
    print(f"   Confidence: {score.confidence:.2f}")
    assert 0 <= score.score <= 100, "Score should be in valid range"
    assert score.confidence < 0.5, "Confidence should be low for minimal data"
    
    print("✅ Missing data handled gracefully")


def run_all_tests():
    """Run all lead scoring tests"""
    print("=" * 60)
    print("LEAD SCORING TESTS")
    print("=" * 60)
    
    try:
        test_company_size_scoring()
        test_industry_scoring()
        test_decision_maker_scoring()
        test_geographic_scoring()
        test_engagement_scoring()
        test_full_score_calculation()
        test_score_range()
        test_performance()
        test_missing_data_handling()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    run_all_tests()

