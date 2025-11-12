"""
OSINT Workflows for CPAS5
Reusable workflows for lead generation, competitive intelligence, and contact enrichment
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class OSINTWorkflows:
    """
    Reusable OSINT workflows for AI agents
    """
    
    def __init__(self, tool_registry, db):
        """
        Initialize OSINT workflows
        
        Args:
            tool_registry: ToolRegistry instance for executing tools
            db: MongoDB database instance for storing results
        """
        self.tool_registry = tool_registry
        self.db = db
    
    # ==================== WORKFLOW 1: LEAD GENERATION ====================
    
    async def lead_generation_workflow(
        self,
        company_name: str,
        agent_id: str = "lead_gen_agent"
    ) -> Dict[str, Any]:
        """
        Complete lead generation workflow for a target company
        
        Args:
            company_name: Name of the target company
            agent_id: ID of the agent executing the workflow
            
        Returns:
            Dict with workflow results including contacts and leads
        """
        workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
        logger.info(f"Starting lead generation workflow {workflow_id} for {company_name}")
        
        results = {
            "workflow_id": workflow_id,
            "workflow_type": "lead_generation",
            "company_name": company_name,
            "agent_id": agent_id,
            "started_at": datetime.utcnow().isoformat(),
            "steps_completed": [],
            "contacts_found": [],
            "leads_created": [],
            "errors": []
        }
        
        try:
            # Step 1: Find company website
            logger.info(f"Step 1: Finding website for {company_name}")
            search_result = await self.tool_registry.execute_tool(
                tool_id="web_search",
                parameters={"query": f"{company_name} official website", "num_results": 5},
                agent_id=agent_id
            )
            
            if not search_result.get("success"):
                results["errors"].append("Failed to find company website")
                return results
            
            # Extract domain from first result
            domain = self._extract_domain(search_result.get("results", [{}])[0].get("url", ""))
            if not domain:
                results["errors"].append("Could not extract domain from search results")
                return results
            
            results["domain"] = domain
            results["steps_completed"].append("find_website")
            
            # Step 2: Find emails for domain (Hunter.io)
            logger.info(f"Step 2: Finding emails for {domain}")
            email_result = await self.tool_registry.execute_tool(
                tool_id="osint_hunter_domain_search",
                parameters={"domain": domain, "limit": 20},
                agent_id=agent_id
            )
            
            if email_result.get("success"):
                emails = email_result.get("result", {}).get("emails", [])
                results["steps_completed"].append("find_emails")
                
                # Step 3: Verify each email
                logger.info(f"Step 3: Verifying {len(emails)} emails")
                verified_emails = []
                for email_data in emails:
                    if email_data.get("confidence", 0) > 90:
                        verify_result = await self.tool_registry.execute_tool(
                            tool_id="osint_hunter_email_verifier",
                            parameters={"email": email_data["email"]},
                            agent_id=agent_id
                        )
                        
                        if verify_result.get("success") and verify_result.get("result", {}).get("score", 0) > 80:
                            verified_emails.append({
                                **email_data,
                                "verification_score": verify_result.get("result", {}).get("score")
                            })
                
                results["steps_completed"].append("verify_emails")
                results["verified_emails_count"] = len(verified_emails)
            else:
                results["errors"].append(f"Email search failed: {email_result.get('error')}")
                verified_emails = []
            
            # Step 4: Scrape company website for additional contacts
            logger.info(f"Step 4: Scraping website for contacts")
            scrape_result = await self.tool_registry.execute_tool(
                tool_id="osint_contact_scraper",
                parameters={"url": f"https://{domain}/about"},
                agent_id=agent_id
            )
            
            additional_contacts = {}
            if scrape_result.get("success"):
                scrape_data = scrape_result.get("result", {})
                additional_contacts = {
                    "emails": scrape_data.get("emails", []),
                    "phones": scrape_data.get("phone_numbers", []),
                    "social_links": scrape_data.get("social_links", {})
                }
                results["steps_completed"].append("scrape_website")
            
            # Step 5: Analyze phone numbers
            logger.info(f"Step 5: Analyzing phone numbers")
            validated_phones = []
            for phone in additional_contacts.get("phones", []):
                phone_result = await self.tool_registry.execute_tool(
                    tool_id="osint_phone_lookup",
                    parameters={"phone_number": phone, "region": "US"},
                    agent_id=agent_id
                )
                
                if phone_result.get("success") and phone_result.get("result", {}).get("is_valid"):
                    validated_phones.append(phone_result.get("result"))
            
            results["steps_completed"].append("analyze_phones")
            results["validated_phones_count"] = len(validated_phones)
            
            # Step 6: Store contacts in database
            logger.info(f"Step 6: Storing contacts in database")
            for email_data in verified_emails:
                contact = {
                    "contact_id": f"contact_{uuid.uuid4().hex[:12]}",
                    "email": email_data.get("email"),
                    "first_name": email_data.get("first_name"),
                    "last_name": email_data.get("last_name"),
                    "full_name": f"{email_data.get('first_name', '')} {email_data.get('last_name', '')}".strip(),
                    "company": company_name,
                    "domain": domain,
                    "position": email_data.get("position"),
                    "department": email_data.get("department"),
                    "source": "hunter_io",
                    "source_tool": "osint_hunter_domain_search",
                    "confidence_score": email_data.get("verification_score", 0) / 100,
                    "verified": True,
                    "status": "verified",
                    "tags": ["lead", "osint_generated"],
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
                
                # Add phone if available
                if validated_phones:
                    contact["phone"] = validated_phones[0].get("phone_number")
                    contact["phone_type"] = validated_phones[0].get("number_type")
                
                # Add social profiles
                if additional_contacts.get("social_links"):
                    contact["social_profiles"] = [
                        {"platform": platform, "url": url}
                        for platform, urls in additional_contacts["social_links"].items()
                        for url in urls
                    ]
                
                # Save to database
                await self.db.osint_contacts.insert_one(contact)
                results["contacts_found"].append(contact)
            
            results["steps_completed"].append("store_contacts")
            
            # Step 7: Create leads
            logger.info(f"Step 7: Creating leads")
            for contact in results["contacts_found"]:
                # Calculate lead score
                lead_score = self._calculate_lead_score(contact)
                
                lead = {
                    "lead_id": f"lead_{uuid.uuid4().hex[:12]}",
                    "contact_id": contact["contact_id"],
                    "lead_score": lead_score,
                    "lead_status": "new",
                    "qualified": lead_score >= 70,
                    "qualification_criteria": {
                        "has_email": bool(contact.get("email")),
                        "has_phone": bool(contact.get("phone")),
                        "decision_maker": "ceo" in contact.get("position", "").lower() or "vp" in contact.get("position", "").lower(),
                        "verified_contact": contact.get("verified", False)
                    },
                    "contacted": False,
                    "contact_attempts": 0,
                    "source_campaign": f"OSINT_{datetime.utcnow().strftime('%Y%m')}",
                    "source_channel": "osint_research",
                    "tags": ["osint_generated", "high_priority" if lead_score >= 80 else "medium_priority"],
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
                
                # Save to database
                await self.db.osint_leads.insert_one(lead)
                results["leads_created"].append(lead)
            
            results["steps_completed"].append("create_leads")
            
        except Exception as e:
            logger.error(f"Workflow error: {e}")
            results["errors"].append(str(e))
        
        results["completed_at"] = datetime.utcnow().isoformat()
        results["success"] = len(results["errors"]) == 0
        results["total_contacts"] = len(results["contacts_found"])
        results["total_leads"] = len(results["leads_created"])
        
        logger.info(f"Workflow {workflow_id} completed: {results['total_contacts']} contacts, {results['total_leads']} leads")
        
        return results
    
    # ==================== WORKFLOW 2: COMPETITIVE INTELLIGENCE ====================
    
    async def competitive_intelligence_workflow(
        self,
        competitor_domain: str,
        agent_id: str = "research_agent"
    ) -> Dict[str, Any]:
        """
        Gather comprehensive intelligence about a competitor
        
        Args:
            competitor_domain: Domain of the competitor
            agent_id: ID of the agent executing the workflow
            
        Returns:
            Dict with intelligence report
        """
        workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
        logger.info(f"Starting competitive intelligence workflow {workflow_id} for {competitor_domain}")
        
        report = {
            "workflow_id": workflow_id,
            "workflow_type": "competitive_intelligence",
            "competitor_domain": competitor_domain,
            "agent_id": agent_id,
            "started_at": datetime.utcnow().isoformat(),
            "intelligence": {},
            "errors": []
        }
        
        try:
            # Step 1: WHOIS lookup
            whois_result = await self.tool_registry.execute_tool(
                tool_id="osint_whois_lookup",
                parameters={"domain": competitor_domain},
                agent_id=agent_id
            )
            
            if whois_result.get("success"):
                report["intelligence"]["domain_registration"] = whois_result.get("result")
            
            # Step 2: DNS lookup
            dns_result = await self.tool_registry.execute_tool(
                tool_id="osint_dns_lookup",
                parameters={"domain": competitor_domain, "record_type": "ALL"},
                agent_id=agent_id
            )
            
            if dns_result.get("success"):
                report["intelligence"]["dns_records"] = dns_result.get("result")
            
            # Step 3: Find key employees
            employees_result = await self.tool_registry.execute_tool(
                tool_id="osint_hunter_domain_search",
                parameters={"domain": competitor_domain, "limit": 50},
                agent_id=agent_id
            )
            
            if employees_result.get("success"):
                report["intelligence"]["employees"] = employees_result.get("result")
            
            # Step 4: Scrape website
            scrape_result = await self.tool_registry.execute_tool(
                tool_id="osint_web_scraper",
                parameters={"url": f"https://{competitor_domain}"},
                agent_id=agent_id
            )
            
            if scrape_result.get("success"):
                report["intelligence"]["website_data"] = scrape_result.get("result")
            
            # Step 5: Extract social profiles
            social_result = await self.tool_registry.execute_tool(
                tool_id="osint_contact_scraper",
                parameters={"url": f"https://{competitor_domain}/about"},
                agent_id=agent_id
            )
            
            if social_result.get("success"):
                report["intelligence"]["social_presence"] = social_result.get("result", {}).get("social_links")
            
        except Exception as e:
            logger.error(f"Workflow error: {e}")
            report["errors"].append(str(e))
        
        report["completed_at"] = datetime.utcnow().isoformat()
        report["success"] = len(report["errors"]) == 0
        
        return report
    
    # ==================== HELPER METHODS ====================
    
    def _extract_domain(self, url: str) -> Optional[str]:
        """Extract domain from URL"""
        from urllib.parse import urlparse
        try:
            parsed = urlparse(url)
            domain = parsed.netloc or parsed.path
            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except:
            return None
    
    def _calculate_lead_score(self, contact: Dict[str, Any]) -> int:
        """Calculate lead score based on contact data"""
        score = 50  # Base score
        
        # Email verified
        if contact.get("verified"):
            score += 20
        
        # Has phone
        if contact.get("phone"):
            score += 10
        
        # Decision maker
        position = contact.get("position", "").lower()
        if any(title in position for title in ["ceo", "cto", "cfo", "vp", "director"]):
            score += 20
        
        # High confidence
        if contact.get("confidence_score", 0) > 0.9:
            score += 10
        
        return min(score, 100)

