"""
GRETA Research & Reporting Automation System
Automated deep research with scheduled reports and intelligence gathering
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
from pydantic import BaseModel, Field
import uuid
import hashlib

# Scheduling dependencies
try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
    SCHEDULING_AVAILABLE = True
except ImportError:
    logging.warning("Scheduling dependencies not available. Install: pip install apscheduler")
    SCHEDULING_AVAILABLE = False

# Research dependencies
try:
    import requests
    from bs4 import BeautifulSoup
    import feedparser
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    logging.warning("Web scraping dependencies not available. Install: pip install requests beautifulsoup4 feedparser")
    WEB_SCRAPING_AVAILABLE = False

logger = logging.getLogger(__name__)

class ResearchRequest(BaseModel):
    topic: str = Field(..., description="Research topic or query")
    research_type: str = Field(default="comprehensive", description="Type of research")
    sources: List[str] = Field(default_factory=list, description="Specific sources to include")
    depth: str = Field(default="medium", description="Research depth: shallow, medium, deep")
    output_format: str = Field(default="report", description="Output format")
    schedule: Optional[Dict[str, Any]] = Field(None, description="Scheduling configuration")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ResearchResult(BaseModel):
    research_id: str
    topic: str
    status: str
    findings: List[Dict[str, Any]]
    sources_used: List[str]
    confidence_score: float
    research_time: float
    report_path: Optional[str]
    next_update: Optional[str]

class ScheduledResearch(BaseModel):
    schedule_id: str
    topic: str
    frequency: str
    next_run: str
    last_run: Optional[str]
    total_runs: int
    status: str

class ResearchAutomationEngine:
    """Engine for automated research and reporting"""
    
    def __init__(self):
        self.reports_dir = Path("research_reports")
        self.reports_dir.mkdir(exist_ok=True)
        
        self.data_dir = Path("research_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize scheduler
        if SCHEDULING_AVAILABLE:
            self.scheduler = AsyncIOScheduler()
            self.scheduler.start()
        else:
            self.scheduler = None
        
        # Research sources configuration
        self.research_sources = {
            "news": {
                "rss_feeds": [
                    "https://feeds.reuters.com/reuters/topNews",
                    "https://rss.cnn.com/rss/edition.rss",
                    "https://feeds.bbci.co.uk/news/rss.xml"
                ],
                "apis": []
            },
            "academic": {
                "sources": ["arxiv", "pubmed", "google_scholar"],
                "apis": []
            },
            "business": {
                "sources": ["sec_filings", "earnings_calls", "market_data"],
                "apis": []
            },
            "social": {
                "sources": ["twitter", "reddit", "linkedin"],
                "apis": []
            },
            "web": {
                "search_engines": ["google", "bing", "duckduckgo"],
                "apis": []
            }
        }
        
        # Active research jobs
        self.active_research = {}
        self.scheduled_research = {}
        
        # Research templates
        self.report_templates = {
            "executive_summary": """
# Executive Summary: {topic}

## Key Findings
{key_findings}

## Recommendations
{recommendations}

## Risk Assessment
{risk_assessment}

## Next Steps
{next_steps}
            """,
            
            "market_analysis": """
# Market Analysis: {topic}

## Market Size & Growth
{market_size}

## Competitive Landscape
{competitive_landscape}

## Trends & Opportunities
{trends}

## Threats & Challenges
{threats}

## Investment Outlook
{investment_outlook}
            """,
            
            "technical_report": """
# Technical Report: {topic}

## Overview
{overview}

## Technical Analysis
{technical_analysis}

## Implementation Considerations
{implementation}

## Performance Metrics
{performance}

## Conclusions
{conclusions}
            """
        }
    
    async def conduct_research(self, request: ResearchRequest) -> ResearchResult:
        """Conduct comprehensive research on a topic"""
        start_time = datetime.now()
        research_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Starting research: {request.topic} (ID: {research_id})")
            
            # Store active research
            self.active_research[research_id] = {
                "topic": request.topic,
                "status": "in_progress",
                "start_time": start_time,
                "request": request
            }
            
            # Gather data from multiple sources
            findings = await self._gather_research_data(request)
            
            # Analyze and synthesize findings
            analysis = await self._analyze_findings(findings, request)
            
            # Generate report
            report_path = await self._generate_report(research_id, request, analysis)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(findings, analysis)
            
            research_time = (datetime.now() - start_time).total_seconds()
            
            # Schedule next update if requested
            next_update = None
            if request.schedule:
                next_update = await self._schedule_research_update(research_id, request)
            
            # Clean up active research
            if research_id in self.active_research:
                del self.active_research[research_id]
            
            result = ResearchResult(
                research_id=research_id,
                topic=request.topic,
                status="completed",
                findings=analysis.get("findings", []),
                sources_used=analysis.get("sources_used", []),
                confidence_score=confidence_score,
                research_time=research_time,
                report_path=report_path,
                next_update=next_update
            )
            
            logger.info(f"✅ Research completed: {research_id} ({research_time:.2f}s)")
            return result
            
        except Exception as e:
            logger.error(f"Research failed: {e}")
            
            # Update active research status
            if research_id in self.active_research:
                self.active_research[research_id]["status"] = "failed"
                self.active_research[research_id]["error"] = str(e)
            
            return ResearchResult(
                research_id=research_id,
                topic=request.topic,
                status="failed",
                findings=[],
                sources_used=[],
                confidence_score=0.0,
                research_time=(datetime.now() - start_time).total_seconds(),
                report_path=None,
                next_update=None
            )
    
    async def _gather_research_data(self, request: ResearchRequest) -> List[Dict[str, Any]]:
        """Gather data from multiple research sources"""
        findings = []
        
        # Determine sources based on research type
        if request.research_type == "news":
            findings.extend(await self._gather_news_data(request.topic))
        elif request.research_type == "academic":
            findings.extend(await self._gather_academic_data(request.topic))
        elif request.research_type == "business":
            findings.extend(await self._gather_business_data(request.topic))
        elif request.research_type == "comprehensive":
            # Gather from all sources
            findings.extend(await self._gather_news_data(request.topic))
            findings.extend(await self._gather_web_data(request.topic))
            findings.extend(await self._gather_social_data(request.topic))
        
        # Add custom sources if specified
        for source in request.sources:
            custom_data = await self._gather_custom_source_data(source, request.topic)
            findings.extend(custom_data)
        
        return findings
    
    async def _gather_news_data(self, topic: str) -> List[Dict[str, Any]]:
        """Gather news data from RSS feeds and news APIs"""
        findings = []
        
        if not WEB_SCRAPING_AVAILABLE:
            return findings
        
        try:
            # RSS feeds
            for feed_url in self.research_sources["news"]["rss_feeds"]:
                try:
                    feed = feedparser.parse(feed_url)
                    for entry in feed.entries[:5]:  # Limit to 5 entries per feed
                        if topic.lower() in entry.title.lower() or topic.lower() in entry.summary.lower():
                            findings.append({
                                "type": "news",
                                "source": feed.feed.title if hasattr(feed.feed, 'title') else feed_url,
                                "title": entry.title,
                                "content": entry.summary,
                                "url": entry.link,
                                "published": entry.published if hasattr(entry, 'published') else None,
                                "relevance_score": self._calculate_relevance(topic, entry.title + " " + entry.summary)
                            })
                except Exception as e:
                    logger.warning(f"Failed to parse RSS feed {feed_url}: {e}")
            
        except Exception as e:
            logger.error(f"News data gathering failed: {e}")
        
        return findings
    
    async def _gather_web_data(self, topic: str) -> List[Dict[str, Any]]:
        """Gather web data through search and scraping"""
        findings = []
        
        # Mock web search results (in production, would use actual search APIs)
        mock_results = [
            {
                "type": "web",
                "source": "example.com",
                "title": f"Comprehensive guide to {topic}",
                "content": f"This article covers everything you need to know about {topic}...",
                "url": f"https://example.com/{topic.replace(' ', '-')}",
                "relevance_score": 0.85
            },
            {
                "type": "web", 
                "source": "research.org",
                "title": f"Latest developments in {topic}",
                "content": f"Recent research shows significant progress in {topic} field...",
                "url": f"https://research.org/studies/{topic.replace(' ', '-')}",
                "relevance_score": 0.78
            }
        ]
        
        findings.extend(mock_results)
        return findings
    
    async def _gather_academic_data(self, topic: str) -> List[Dict[str, Any]]:
        """Gather academic research data"""
        findings = []
        
        # Mock academic results
        mock_academic = [
            {
                "type": "academic",
                "source": "arXiv",
                "title": f"A Survey of {topic} Methodologies",
                "content": f"This paper presents a comprehensive survey of current {topic} approaches...",
                "authors": ["Dr. Smith", "Dr. Johnson"],
                "published": "2024-01-15",
                "citations": 42,
                "relevance_score": 0.92
            }
        ]
        
        findings.extend(mock_academic)
        return findings
    
    async def _gather_business_data(self, topic: str) -> List[Dict[str, Any]]:
        """Gather business and financial data"""
        findings = []
        
        # Mock business data
        mock_business = [
            {
                "type": "business",
                "source": "Market Research Inc",
                "title": f"{topic} Market Analysis 2024",
                "content": f"The {topic} market is expected to grow by 15% annually...",
                "market_size": "$2.5B",
                "growth_rate": "15%",
                "relevance_score": 0.88
            }
        ]
        
        findings.extend(mock_business)
        return findings
    
    async def _gather_social_data(self, topic: str) -> List[Dict[str, Any]]:
        """Gather social media and community data"""
        findings = []
        
        # Mock social data
        mock_social = [
            {
                "type": "social",
                "source": "Twitter",
                "content": f"Trending discussions about {topic} show positive sentiment...",
                "sentiment": "positive",
                "engagement": 1250,
                "relevance_score": 0.65
            }
        ]
        
        findings.extend(mock_social)
        return findings
    
    async def _gather_custom_source_data(self, source: str, topic: str) -> List[Dict[str, Any]]:
        """Gather data from custom sources"""
        # Mock custom source data
        return [{
            "type": "custom",
            "source": source,
            "content": f"Custom data about {topic} from {source}",
            "relevance_score": 0.70
        }]
    
    async def _analyze_findings(self, findings: List[Dict[str, Any]], request: ResearchRequest) -> Dict[str, Any]:
        """Analyze and synthesize research findings"""
        
        # Sort findings by relevance
        findings.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        # Extract key insights
        key_findings = []
        sources_used = []
        
        for finding in findings[:10]:  # Top 10 most relevant
            key_findings.append({
                "insight": finding.get("title", ""),
                "source": finding.get("source", ""),
                "confidence": finding.get("relevance_score", 0),
                "type": finding.get("type", "")
            })
            
            if finding.get("source") not in sources_used:
                sources_used.append(finding.get("source"))
        
        # Generate summary and recommendations
        analysis = {
            "findings": key_findings,
            "sources_used": sources_used,
            "summary": f"Research on {request.topic} reveals {len(key_findings)} key insights from {len(sources_used)} sources.",
            "recommendations": [
                f"Continue monitoring {request.topic} developments",
                "Consider deeper analysis of top findings",
                "Expand research to additional sources"
            ],
            "confidence_level": "medium",
            "data_quality": "good"
        }
        
        return analysis
    
    async def _generate_report(self, research_id: str, request: ResearchRequest, analysis: Dict[str, Any]) -> str:
        """Generate research report"""
        
        # Select appropriate template
        template_name = request.metadata.get("template", "executive_summary")
        template = self.report_templates.get(template_name, self.report_templates["executive_summary"])
        
        # Generate report content
        report_content = f"""
# Research Report: {request.topic}

**Research ID:** {research_id}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Research Type:** {request.research_type}
**Depth:** {request.depth}

## Executive Summary
{analysis.get('summary', 'No summary available')}

## Key Findings
"""
        
        for i, finding in enumerate(analysis.get('findings', []), 1):
            report_content += f"\n{i}. **{finding.get('insight', '')}**\n"
            report_content += f"   - Source: {finding.get('source', '')}\n"
            report_content += f"   - Confidence: {finding.get('confidence', 0):.2f}\n"
            report_content += f"   - Type: {finding.get('type', '')}\n"
        
        report_content += f"""

## Sources Used
{', '.join(analysis.get('sources_used', []))}

## Recommendations
"""
        
        for i, rec in enumerate(analysis.get('recommendations', []), 1):
            report_content += f"{i}. {rec}\n"
        
        report_content += f"""

## Metadata
- **Confidence Level:** {analysis.get('confidence_level', 'unknown')}
- **Data Quality:** {analysis.get('data_quality', 'unknown')}
- **Total Sources:** {len(analysis.get('sources_used', []))}
- **Research Duration:** {request.metadata.get('duration', 'unknown')}

---
*Generated by GRETA Research Automation Engine*
        """
        
        # Save report
        report_path = self.reports_dir / f"research_report_{research_id}.md"
        report_path.write_text(report_content.strip())
        
        return str(report_path)
    
    def _calculate_relevance(self, topic: str, content: str) -> float:
        """Calculate relevance score between topic and content"""
        topic_words = set(topic.lower().split())
        content_words = set(content.lower().split())
        
        if not topic_words:
            return 0.0
        
        intersection = topic_words.intersection(content_words)
        return len(intersection) / len(topic_words)
    
    def _calculate_confidence_score(self, findings: List[Dict[str, Any]], analysis: Dict[str, Any]) -> float:
        """Calculate overall confidence score for research"""
        if not findings:
            return 0.0
        
        # Average relevance scores
        relevance_scores = [f.get("relevance_score", 0) for f in findings]
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        
        # Factor in number of sources
        source_factor = min(len(analysis.get("sources_used", [])) / 5, 1.0)
        
        # Factor in data quality
        quality_factor = 0.8 if analysis.get("data_quality") == "good" else 0.6
        
        return avg_relevance * source_factor * quality_factor
    
    async def _schedule_research_update(self, research_id: str, request: ResearchRequest) -> str:
        """Schedule recurring research updates"""
        if not self.scheduler or not request.schedule:
            return None
        
        schedule_config = request.schedule
        frequency = schedule_config.get("frequency", "daily")
        
        # Calculate next run time
        if frequency == "hourly":
            next_run = datetime.now() + timedelta(hours=1)
            trigger = IntervalTrigger(hours=1)
        elif frequency == "daily":
            next_run = datetime.now() + timedelta(days=1)
            trigger = IntervalTrigger(days=1)
        elif frequency == "weekly":
            next_run = datetime.now() + timedelta(weeks=1)
            trigger = IntervalTrigger(weeks=1)
        elif frequency == "monthly":
            next_run = datetime.now() + timedelta(days=30)
            trigger = IntervalTrigger(days=30)
        else:
            return None
        
        # Schedule the job
        job_id = f"research_update_{research_id}"
        self.scheduler.add_job(
            self._execute_scheduled_research,
            trigger,
            args=[request],
            id=job_id,
            replace_existing=True
        )
        
        # Store scheduled research info
        self.scheduled_research[job_id] = ScheduledResearch(
            schedule_id=job_id,
            topic=request.topic,
            frequency=frequency,
            next_run=next_run.isoformat(),
            last_run=None,
            total_runs=0,
            status="active"
        )
        
        return next_run.isoformat()
    
    async def _execute_scheduled_research(self, request: ResearchRequest):
        """Execute scheduled research update"""
        try:
            logger.info(f"Executing scheduled research: {request.topic}")
            result = await self.conduct_research(request)
            logger.info(f"Scheduled research completed: {result.research_id}")
        except Exception as e:
            logger.error(f"Scheduled research failed: {e}")
    
    async def get_research_status(self, research_id: str) -> Optional[Dict[str, Any]]:
        """Get status of active or completed research"""
        if research_id in self.active_research:
            return self.active_research[research_id]
        
        # Check for completed research report
        report_path = self.reports_dir / f"research_report_{research_id}.md"
        if report_path.exists():
            return {
                "research_id": research_id,
                "status": "completed",
                "report_path": str(report_path),
                "completed_at": report_path.stat().st_mtime
            }
        
        return None
    
    async def list_scheduled_research(self) -> List[ScheduledResearch]:
        """List all scheduled research jobs"""
        return list(self.scheduled_research.values())
    
    async def cancel_scheduled_research(self, schedule_id: str) -> bool:
        """Cancel scheduled research"""
        if self.scheduler and schedule_id in self.scheduled_research:
            try:
                self.scheduler.remove_job(schedule_id)
                del self.scheduled_research[schedule_id]
                return True
            except Exception as e:
                logger.error(f"Failed to cancel scheduled research: {e}")
        
        return False

# Global service instance
research_automation_engine = ResearchAutomationEngine()
