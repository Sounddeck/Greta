"""
Lead Scoring Service for GRETA PAI
Intelligent lead scoring based on multiple factors with weighted algorithm
"""

from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger

from database import Database
from models.lead_generation_models import Lead, LeadScore
from services.event_bus import publish_event


class LeadScoringService:
    """
    Lead Scoring Service - Intelligent lead prioritization
    
    Scoring Factors (0-100 each):
    - Company Size (25%): Based on employee count
    - Industry Match (20%): Alignment with target industries
    - Engagement (20%): Email opens, website visits, content downloads
    - Budget Indicators (15%): Funding, revenue signals
    - Decision Maker (10%): Contact level (C-suite, VP, Manager)
    - Geographic Fit (10%): Location preference match
    
    Final Score: Weighted sum of all factors (0-100)
    """
    
    # Scoring weights
    WEIGHTS = {
        "company_size": 0.25,
        "industry": 0.20,
        "engagement": 0.20,
        "budget": 0.15,
        "decision_maker": 0.10,
        "geographic": 0.10,
    }
    
    # Target industries (can be configured per campaign)
    TARGET_INDUSTRIES = [
        "technology",
        "software",
        "saas",
        "fintech",
        "healthcare tech",
        "ai/ml",
        "data analytics",
    ]
    
    # Target regions (can be configured per campaign)
    TARGET_REGIONS = [
        "north america",
        "united states",
        "usa",
        "canada",
        "europe",
        "uk",
        "united kingdom",
        "britain",
        "germany",
    ]
    
    def __init__(self, db: Optional[Database] = None):
        """Initialize lead scoring service"""
        self.db = db or Database()
        self.scores_collection = "lead_scores"
        self.score_history_collection = "lead_score_history"
    
    def calculate_score(self, lead: Lead, **context) -> LeadScore:
        """
        Calculate comprehensive lead score
        
        Args:
            lead: Lead object to score
            **context: Additional context (engagement_data, etc.)
            
        Returns:
            LeadScore object with breakdown
        """
        # Calculate individual factor scores
        scores = {
            "company_size": self._score_company_size(lead),
            "industry": self._score_industry(lead),
            "engagement": self._score_engagement(lead, context.get("engagement_data", {})),
            "budget": self._score_budget(lead),
            "decision_maker": self._score_decision_maker(lead),
            "geographic": self._score_geographic(lead),
        }
        
        # Calculate weighted total score
        weighted_score = sum(
            scores[factor] * self.WEIGHTS[factor]
            for factor in scores
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(scores, lead)
        
        # Calculate confidence based on data completeness
        confidence = self._calculate_confidence(lead)
        
        return LeadScore(
            score=round(weighted_score, 1),
            factors=scores,
            confidence=confidence,
            recommendations=recommendations,
            calculated_at=datetime.utcnow()
        )
    
    def _score_company_size(self, lead: Lead) -> float:
        """
        Score based on company size (employee count)

        Scoring:
        - 1-10 employees: 20 points
        - 11-50 employees: 40 points
        - 51-200 employees: 60 points
        - 201-1000 employees: 80 points
        - 1000+ employees: 100 points
        - Unknown: 50 points (neutral)
        """
        if not lead.company_size:
            return 50.0  # Neutral score for missing data

        size_lower = lead.company_size.lower()

        # Check for enterprise/large company keywords first
        if "1000+" in size_lower or "1001+" in size_lower or "enterprise" in size_lower:
            return 100.0

        # Parse employee count ranges - check larger ranges first
        if "201-1000" in size_lower or "201 to 1000" in size_lower:
            return 80.0
        elif "51-200" in size_lower or "51 to 200" in size_lower:
            return 60.0
        elif "11-50" in size_lower or "11 to 50" in size_lower:
            return 40.0
        elif "1-10" in size_lower or "1 to 10" in size_lower:
            return 20.0

        # Try to extract number
        try:
            # Look for numbers in the string
            import re
            numbers = re.findall(r'\d+', size_lower)
            if numbers:
                count = int(numbers[0])
                if count <= 10:
                    return 20.0
                elif count <= 50:
                    return 40.0
                elif count <= 200:
                    return 60.0
                elif count <= 1000:
                    return 80.0
                else:
                    return 100.0
        except:
            pass

        return 50.0  # Default neutral score
    
    def _score_industry(self, lead: Lead) -> float:
        """
        Score based on industry match
        
        Scoring:
        - Target industry: 100 points
        - Related industry: 60 points
        - Other industry: 20 points
        """
        if not lead.industry:
            return 20.0  # Low score for missing data
        
        industry_lower = lead.industry.lower()
        
        # Check for exact match
        for target in self.TARGET_INDUSTRIES:
            if target in industry_lower:
                return 100.0
        
        # Check for related industries
        related_keywords = [
            "tech", "digital", "software", "cloud", "internet",
            "mobile", "web", "app", "platform", "service"
        ]
        for keyword in related_keywords:
            if keyword in industry_lower:
                return 60.0
        
        return 20.0  # Other industry
    
    def _score_engagement(self, lead: Lead, engagement_data: Dict) -> float:
        """
        Score based on engagement metrics
        
        Factors:
        - Email opens: 0-40 points
        - Website visits: 0-40 points
        - Content downloads: 0-20 points
        """
        score = 0.0
        
        # Email engagement
        email_opens = engagement_data.get("email_opens", 0)
        score += min(email_opens * 10, 40)  # Max 40 points
        
        # Website visits
        website_visits = engagement_data.get("website_visits", 0)
        score += min(website_visits * 8, 40)  # Max 40 points
        
        # Content downloads
        downloads = engagement_data.get("content_downloads", 0)
        score += min(downloads * 10, 20)  # Max 20 points
        
        return min(score, 100.0)
    
    def _score_budget(self, lead: Lead) -> float:
        """
        Score based on budget indicators
        
        Factors:
        - Funding raised: 0-50 points
        - Revenue estimate: 0-50 points
        """
        score = 0.0
        
        # Check for funding information in opportunities
        for opportunity in lead.opportunities:
            if opportunity.opportunity_type == "funding":
                score += 50.0 * opportunity.confidence
                break
        
        # Check company size as revenue proxy
        if lead.company_size:
            size_lower = lead.company_size.lower()
            if "1000+" in size_lower or "enterprise" in size_lower:
                score += 50.0
            elif "201-1000" in size_lower:
                score += 40.0
            elif "51-200" in size_lower:
                score += 30.0
            elif "11-50" in size_lower:
                score += 20.0
            else:
                score += 10.0
        
        return min(score, 100.0)
    
    def _score_decision_maker(self, lead: Lead) -> float:
        """
        Score based on decision maker level

        Scoring:
        - C-Suite (CEO, CTO, CFO, etc.): 100 points
        - VP/Director: 70 points
        - Manager: 40 points
        - Individual Contributor: 20 points
        - Unknown: 50 points
        """
        if not lead.key_people:
            return 50.0  # Neutral for missing data

        # Check highest-level contact
        max_score = 0.0

        for person in lead.key_people:
            title_lower = person.title.lower() if person.title else ""

            # C-Suite - use word boundaries to avoid false matches (e.g., "cto" in "director")
            import re
            c_suite_pattern = r'\b(ceo|cto|cfo|coo|cmo|chief|president|founder)\b'
            if re.search(c_suite_pattern, title_lower):
                max_score = max(max_score, 100.0)
            # VP/Director
            elif re.search(r'\b(vp|vice president)\b', title_lower):
                max_score = max(max_score, 70.0)
            elif re.search(r'\b(director|head of)\b', title_lower):
                max_score = max(max_score, 70.0)
            # Manager
            elif re.search(r'\bmanager\b', title_lower):
                max_score = max(max_score, 40.0)
            # Individual
            else:
                max_score = max(max_score, 20.0)

        return max_score if max_score > 0 else 50.0
    
    def _score_geographic(self, lead: Lead) -> float:
        """
        Score based on geographic fit
        
        Scoring:
        - Target region: 100 points
        - Adjacent region: 60 points
        - Other region: 30 points
        """
        if not lead.location:
            return 30.0  # Low score for missing data
        
        location_lower = lead.location.lower()
        
        # Check for target regions
        for region in self.TARGET_REGIONS:
            if region in location_lower:
                return 100.0
        
        # Check for adjacent regions
        adjacent_regions = ["asia", "australia", "south america", "mexico"]
        for region in adjacent_regions:
            if region in location_lower:
                return 60.0
        
        return 30.0  # Other region
    
    def _calculate_confidence(self, lead: Lead) -> float:
        """
        Calculate confidence score based on data completeness
        
        Returns:
            Confidence score (0-1)
        """
        data_points = 0
        total_points = 6
        
        if lead.company_size:
            data_points += 1
        if lead.industry:
            data_points += 1
        if lead.location:
            data_points += 1
        if lead.key_people:
            data_points += 1
        if lead.opportunities:
            data_points += 1
        if lead.website:
            data_points += 1
        
        return round(data_points / total_points, 2)
    
    def _generate_recommendations(self, scores: Dict[str, float], lead: Lead) -> List[str]:
        """
        Generate actionable recommendations based on scores
        
        Args:
            scores: Individual factor scores
            lead: Lead object
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Low company size score
        if scores["company_size"] < 50:
            recommendations.append("Consider targeting larger companies for better ROI")
        
        # Low industry match
        if scores["industry"] < 60:
            recommendations.append("Industry may not be ideal fit - verify alignment")
        
        # Low engagement
        if scores["engagement"] < 40:
            recommendations.append("Increase engagement through targeted content and outreach")
        
        # Low budget indicators
        if scores["budget"] < 50:
            recommendations.append("Verify budget availability before heavy investment")
        
        # Low decision maker score
        if scores["decision_maker"] < 60:
            recommendations.append("Try to connect with higher-level decision makers")
        
        # Low geographic fit
        if scores["geographic"] < 60:
            recommendations.append("Geographic location may present challenges - consider timezone/language")
        
        # High overall score
        total_score = sum(scores[f] * self.WEIGHTS[f] for f in scores)
        if total_score >= 70:
            recommendations.append("High-priority lead - prioritize for immediate outreach")
        elif total_score >= 50:
            recommendations.append("Medium-priority lead - include in nurture campaign")
        else:
            recommendations.append("Low-priority lead - consider automated nurture or disqualify")
        
        return recommendations
    
    async def save_score(self, lead_id: str, score: LeadScore) -> str:
        """
        Save lead score to database

        Args:
            lead_id: Lead ID
            score: LeadScore object

        Returns:
            Score ID
        """
        score_data = {
            "lead_id": lead_id,
            "score": score.score,
            "factors": score.factors,
            "confidence": score.confidence,
            "recommendations": score.recommendations,
            "calculated_at": score.calculated_at.isoformat(),
        }

        # Save to scores collection (current score) - upsert manually
        existing = await self.db.find_one(self.scores_collection, {"lead_id": lead_id})
        if existing:
            await self.db.update_one(
                self.scores_collection,
                {"lead_id": lead_id},
                {"$set": score_data}
            )
        else:
            await self.db.insert_one(self.scores_collection, score_data)

        # Save to history collection
        score_id = await self.db.insert_one(self.score_history_collection, score_data)

        logger.info(f"Saved lead score for {lead_id}: {score.score}")

        # Publish event for real-time updates
        await publish_event(
            "lead.scored",
            {
                "lead_id": lead_id,
                "score": score.score,
                "factors": score.factors,
                "confidence": score.confidence,
                "recommendations": score.recommendations,
                "calculated_at": score.calculated_at.isoformat(),
            },
            source="lead_scoring_service"
        )

        return score_id
    
    async def get_score(self, lead_id: str) -> Optional[LeadScore]:
        """Get current lead score"""
        score_data = await self.db.find_one(
            self.scores_collection,
            {"lead_id": lead_id}
        )
        
        if not score_data:
            return None
        
        # Convert ISO string back to datetime
        if isinstance(score_data.get("calculated_at"), str):
            score_data["calculated_at"] = datetime.fromisoformat(score_data["calculated_at"])
        
        return LeadScore(**score_data)
    
    async def get_score_history(self, lead_id: str, limit: int = 10) -> List[LeadScore]:
        """Get lead score history"""
        # Get all scores for this lead
        all_scores = await self.db.find_many(
            self.score_history_collection,
            {"lead_id": lead_id},
            limit=1000  # Get up to 1000 scores
        )

        if not all_scores:
            return []

        # Sort by calculated_at descending
        all_scores.sort(
            key=lambda x: x.get("calculated_at", ""),
            reverse=True
        )

        # Limit results
        limited_scores = all_scores[:limit]

        # Convert to LeadScore objects
        scores = []
        for score_data in limited_scores:
            if isinstance(score_data.get("calculated_at"), str):
                score_data["calculated_at"] = datetime.fromisoformat(score_data["calculated_at"])
            scores.append(LeadScore(**score_data))

        return scores

