"""
Enhanced OSINT Tools for CPAS5
Email/Contact Discovery, Advanced Web Scraping, and Intelligence Gathering
"""

import asyncio
import logging
import os
import re
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import aiohttp
import phonenumbers
from phonenumbers import geocoder, carrier, timezone
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

logger = logging.getLogger(__name__)


class OSINTEnhancedTools:
    """
    Enhanced OSINT tools for contact discovery and intelligence gathering
    """
    
    def __init__(self):
        self.config = {
            "hunter_io_api_key": os.getenv("HUNTER_IO_API_KEY"),
            "hunter_io_base_url": "https://api.hunter.io/v2",
            "request_timeout": 30,
            "max_retries": 3,
            "user_agent": "CPAS5-OSINT/1.0",
        }

        # Expose API key for easy access
        self.hunter_api_key = self.config["hunter_io_api_key"]

        # Results cache
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour

        # Metrics
        self.metrics = {
            "hunter_calls": 0,
            "phone_lookups": 0,
            "web_scrapes": 0,
            "contacts_found": 0,
            "api_errors": 0,
        }
    
    # ==================== HUNTER.IO INTEGRATION ====================
    
    async def hunter_domain_search(self, domain: str, limit: int = 10) -> Dict[str, Any]:
        """
        Search for email addresses associated with a domain using Hunter.io
        
        Args:
            domain: Domain name to search (e.g., "example.com")
            limit: Maximum number of results (default: 10, max: 100)
            
        Returns:
            Dict with email addresses, names, positions, and confidence scores
        """
        try:
            if not self.config["hunter_io_api_key"]:
                return {
                    "success": False,
                    "error": "Hunter.io API key not configured",
                    "requires_setup": True
                }
            
            self.metrics["hunter_calls"] += 1
            
            # Check cache
            cache_key = f"hunter_domain:{domain}:{limit}"
            if cache_key in self.cache:
                cached_result = self.cache[cache_key]
                if (datetime.utcnow() - cached_result["timestamp"]).seconds < self.cache_ttl:
                    return cached_result["data"]
            
            # Make API request
            url = f"{self.config['hunter_io_base_url']}/domain-search"
            params = {
                "domain": domain,
                "limit": min(limit, 100),
                "api_key": self.config["hunter_io_api_key"]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Extract and format results
                        emails = []
                        for email_data in data.get("data", {}).get("emails", []):
                            emails.append({
                                "email": email_data.get("value"),
                                "first_name": email_data.get("first_name"),
                                "last_name": email_data.get("last_name"),
                                "position": email_data.get("position"),
                                "department": email_data.get("department"),
                                "confidence": email_data.get("confidence"),
                                "sources": email_data.get("sources", []),
                                "verified": email_data.get("verification", {}).get("status") == "valid"
                            })
                        
                        self.metrics["contacts_found"] += len(emails)
                        
                        result = {
                            "success": True,
                            "domain": domain,
                            "emails": emails,
                            "total_found": len(emails),
                            "organization": data.get("data", {}).get("organization"),
                            "pattern": data.get("data", {}).get("pattern"),
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        
                        # Cache result
                        self.cache[cache_key] = {
                            "data": result,
                            "timestamp": datetime.utcnow()
                        }
                        
                        return result
                    else:
                        error_data = await response.json()
                        self.metrics["api_errors"] += 1
                        return {
                            "success": False,
                            "error": error_data.get("errors", [{}])[0].get("details", "API request failed"),
                            "status_code": response.status
                        }
        
        except Exception as e:
            logger.error(f"Hunter.io domain search failed: {e}")
            self.metrics["api_errors"] += 1
            return {
                "success": False,
                "error": str(e)
            }
    
    async def hunter_email_finder(self, domain: str, first_name: str, last_name: str) -> Dict[str, Any]:
        """
        Find a specific person's email address using Hunter.io
        
        Args:
            domain: Company domain
            first_name: Person's first name
            last_name: Person's last name
            
        Returns:
            Dict with email address and confidence score
        """
        try:
            if not self.config["hunter_io_api_key"]:
                return {
                    "success": False,
                    "error": "Hunter.io API key not configured"
                }
            
            self.metrics["hunter_calls"] += 1
            
            url = f"{self.config['hunter_io_base_url']}/email-finder"
            params = {
                "domain": domain,
                "first_name": first_name,
                "last_name": last_name,
                "api_key": self.config["hunter_io_api_key"]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        data = await response.json()
                        email_data = data.get("data", {})
                        
                        return {
                            "success": True,
                            "email": email_data.get("email"),
                            "first_name": email_data.get("first_name"),
                            "last_name": email_data.get("last_name"),
                            "position": email_data.get("position"),
                            "confidence": email_data.get("score"),
                            "sources": email_data.get("sources", []),
                            "verified": email_data.get("verification", {}).get("status") == "valid"
                        }
                    else:
                        self.metrics["api_errors"] += 1
                        return {
                            "success": False,
                            "error": "Email not found or API error",
                            "status_code": response.status
                        }
        
        except Exception as e:
            logger.error(f"Hunter.io email finder failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def hunter_email_verifier(self, email: str) -> Dict[str, Any]:
        """
        Verify an email address using Hunter.io
        
        Args:
            email: Email address to verify
            
        Returns:
            Dict with verification status and details
        """
        try:
            if not self.config["hunter_io_api_key"]:
                return {
                    "success": False,
                    "error": "Hunter.io API key not configured"
                }
            
            self.metrics["hunter_calls"] += 1
            
            url = f"{self.config['hunter_io_base_url']}/email-verifier"
            params = {
                "email": email,
                "api_key": self.config["hunter_io_api_key"]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        data = await response.json()
                        verification = data.get("data", {})
                        
                        return {
                            "success": True,
                            "email": email,
                            "status": verification.get("status"),  # valid, invalid, accept_all, unknown
                            "score": verification.get("score"),
                            "regexp": verification.get("regexp"),
                            "gibberish": verification.get("gibberish"),
                            "disposable": verification.get("disposable"),
                            "webmail": verification.get("webmail"),
                            "mx_records": verification.get("mx_records"),
                            "smtp_server": verification.get("smtp_server"),
                            "smtp_check": verification.get("smtp_check"),
                            "accept_all": verification.get("accept_all"),
                            "block": verification.get("block")
                        }
                    else:
                        self.metrics["api_errors"] += 1
                        return {
                            "success": False,
                            "error": "Verification failed",
                            "status_code": response.status
                        }
        
        except Exception as e:
            logger.error(f"Hunter.io email verifier failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    # ==================== PHONE NUMBER INTELLIGENCE ====================
    
    async def phone_number_lookup(self, phone_number: str, region: str = "US") -> Dict[str, Any]:
        """
        Analyze and extract information from a phone number
        
        Args:
            phone_number: Phone number to analyze
            region: Default region code (e.g., "US", "GB", "FR")
            
        Returns:
            Dict with phone number details, carrier, location, etc.
        """
        try:
            self.metrics["phone_lookups"] += 1
            
            # Parse phone number
            try:
                parsed = phonenumbers.parse(phone_number, region)
            except phonenumbers.NumberParseException as e:
                return {
                    "success": False,
                    "error": f"Invalid phone number: {str(e)}"
                }
            
            # Validate
            is_valid = phonenumbers.is_valid_number(parsed)
            is_possible = phonenumbers.is_possible_number(parsed)
            
            # Get details
            number_type = phonenumbers.number_type(parsed)
            type_names = {
                0: "FIXED_LINE",
                1: "MOBILE",
                2: "FIXED_LINE_OR_MOBILE",
                3: "TOLL_FREE",
                4: "PREMIUM_RATE",
                5: "SHARED_COST",
                6: "VOIP",
                7: "PERSONAL_NUMBER",
                8: "PAGER",
                9: "UAN",
                10: "VOICEMAIL",
                -1: "UNKNOWN"
            }
            
            return {
                "success": True,
                "phone_number": phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164),
                "international_format": phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL),
                "national_format": phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.NATIONAL),
                "country_code": parsed.country_code,
                "national_number": parsed.national_number,
                "is_valid": is_valid,
                "is_possible": is_possible,
                "number_type": type_names.get(number_type, "UNKNOWN"),
                "location": geocoder.description_for_number(parsed, "en"),
                "carrier": carrier.name_for_number(parsed, "en"),
                "timezones": timezone.time_zones_for_number(parsed)
            }
        
        except Exception as e:
            logger.error(f"Phone number lookup failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    # ==================== ADVANCED WEB SCRAPING ====================
    
    async def scrape_contact_info(self, url: str) -> Dict[str, Any]:
        """
        Scrape contact information from a webpage
        
        Args:
            url: URL to scrape
            
        Returns:
            Dict with extracted emails, phone numbers, and social links
        """
        try:
            self.metrics["web_scrapes"] += 1

            # Create SSL context that doesn't verify certificates (for testing)
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            connector = aiohttp.TCPConnector(ssl=ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                headers = {"User-Agent": self.config["user_agent"]}
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        return {
                            "success": False,
                            "error": f"HTTP {response.status}"
                        }

                    html = await response.text()
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract emails
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = list(set(re.findall(email_pattern, html)))
            
            # Extract phone numbers
            phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
            phones = list(set(re.findall(phone_pattern, html)))
            
            # Extract social media links
            social_links = {
                "linkedin": [],
                "twitter": [],
                "facebook": [],
                "instagram": [],
                "github": []
            }
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                if 'linkedin.com' in href:
                    social_links["linkedin"].append(href)
                elif 'twitter.com' in href or 'x.com' in href:
                    social_links["twitter"].append(href)
                elif 'facebook.com' in href:
                    social_links["facebook"].append(href)
                elif 'instagram.com' in href:
                    social_links["instagram"].append(href)
                elif 'github.com' in href:
                    social_links["github"].append(href)
            
            # Remove duplicates
            for platform in social_links:
                social_links[platform] = list(set(social_links[platform]))
            
            self.metrics["contacts_found"] += len(emails)
            
            return {
                "success": True,
                "url": url,
                "emails": emails,
                "phone_numbers": phones,
                "social_links": social_links,
                "total_contacts": len(emails) + len(phones),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Contact scraping failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics"""
        return self.metrics.copy()

