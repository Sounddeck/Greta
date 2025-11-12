"""
GRETA PAI OSINT Free Tools Service
Free OSINT tools that don't require API keys
"""

import asyncio
import logging
import socket
import ssl
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from urllib.parse import urlparse
import re

# Third-party imports
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class OSINTFreeToolsService:
    """
    Free OSINT tools service - no API keys required
    Implements DNS, WHOIS, web scraping, and SSL certificate analysis
    """
    
    def __init__(self):
        self.config = {
            "request_timeout": 10,
            "max_retries": 3,
            "user_agent": "GRETA-PAI-OSINT/1.0 (Educational/Research)",
            "respect_robots": True,
            "rate_limit_delay": 1.0,
        }
        
        # Results cache
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
        
        # Metrics
        self.metrics = {
            "dns_lookups": 0,
            "whois_lookups": 0,
            "web_scrapes": 0,
            "ssl_checks": 0,
            "cache_hits": 0,
            "errors": 0
        }
    
    async def initialize(self):
        """Initialize the service"""
        try:
            logger.info("✅ OSINT Free Tools Service initialized")
            return True
        except Exception as e:
            logger.error(f"❌ OSINT Free Tools initialization failed: {e}")
            return False
    
    # ==================== DNS LOOKUPS ====================
    
    async def dns_lookup(self, domain: str) -> Dict[str, Any]:
        """
        Perform DNS lookup for a domain
        
        Args:
            domain: Domain name to lookup
            
        Returns:
            Dict with DNS information
        """
        try:
            self.metrics["dns_lookups"] += 1
            
            # Check cache
            cache_key = f"dns:{domain}"
            if cache_key in self.cache:
                self.metrics["cache_hits"] += 1
                return self.cache[cache_key]
            
            # Remove protocol if present
            domain = domain.replace('http://', '').replace('https://', '').split('/')[0]
            
            # Forward DNS lookup
            try:
                ip_address = socket.gethostbyname(domain)
            except socket.gaierror as e:
                return {
                    "success": False,
                    "error": f"DNS lookup failed: {str(e)}",
                    "domain": domain
                }
            
            # Reverse DNS lookup
            try:
                hostname = socket.gethostbyaddr(ip_address)[0]
            except (socket.herror, socket.gaierror):
                hostname = None
            
            # Get all IP addresses
            try:
                addr_info = socket.getaddrinfo(domain, None)
                all_ips = list(set([addr[4][0] for addr in addr_info]))
            except socket.gaierror:
                all_ips = [ip_address]
            
            result = {
                "success": True,
                "domain": domain,
                "ip_address": ip_address,
                "all_ip_addresses": all_ips,
                "reverse_dns": hostname,
                "timestamp": datetime.now().isoformat(),
                "tool": "dns_lookup"
            }
            
            # Cache result
            self.cache[cache_key] = result
            
            logger.info(f"✅ DNS lookup successful for {domain}: {ip_address}")
            return result
            
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"❌ DNS lookup error for {domain}: {e}")
            return {
                "success": False,
                "error": str(e),
                "domain": domain
            }
    
    # ==================== WHOIS LOOKUPS ====================
    
    async def whois_lookup(self, domain: str) -> Dict[str, Any]:
        """
        Perform WHOIS lookup for a domain
        Note: This is a simplified version using whois command
        For production, use python-whois library
        
        Args:
            domain: Domain name to lookup
            
        Returns:
            Dict with WHOIS information
        """
        try:
            self.metrics["whois_lookups"] += 1
            
            # Check cache
            cache_key = f"whois:{domain}"
            if cache_key in self.cache:
                self.metrics["cache_hits"] += 1
                return self.cache[cache_key]
            
            # Remove protocol if present
            domain = domain.replace('http://', '').replace('https://', '').split('/')[0]
            
            # Try to use python-whois if available
            try:
                import whois
                w = whois.whois(domain)
                
                result = {
                    "success": True,
                    "domain": domain,
                    "registrar": w.registrar if hasattr(w, 'registrar') else None,
                    "creation_date": str(w.creation_date) if hasattr(w, 'creation_date') else None,
                    "expiration_date": str(w.expiration_date) if hasattr(w, 'expiration_date') else None,
                    "updated_date": str(w.updated_date) if hasattr(w, 'updated_date') else None,
                    "name_servers": w.name_servers if hasattr(w, 'name_servers') else None,
                    "status": w.status if hasattr(w, 'status') else None,
                    "emails": w.emails if hasattr(w, 'emails') else None,
                    "org": w.org if hasattr(w, 'org') else None,
                    "timestamp": datetime.now().isoformat(),
                    "tool": "whois_lookup"
                }
                
            except ImportError:
                # Fallback: Return mock data with installation instructions
                result = {
                    "success": False,
                    "error": "python-whois not installed",
                    "message": "Install with: pip install python-whois",
                    "domain": domain,
                    "mock_mode": True,
                    "mock_data": {
                        "registrar": "Mock Registrar (install python-whois for real data)",
                        "creation_date": "2020-01-01",
                        "expiration_date": "2025-01-01",
                        "name_servers": ["ns1.example.com", "ns2.example.com"],
                        "status": ["clientTransferProhibited"]
                    }
                }
            
            # Cache result
            self.cache[cache_key] = result
            
            logger.info(f"✅ WHOIS lookup completed for {domain}")
            return result
            
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"❌ WHOIS lookup error for {domain}: {e}")
            return {
                "success": False,
                "error": str(e),
                "domain": domain
            }
    
    # ==================== WEB SCRAPING ====================
    
    async def scrape_website(self, url: str, extract_contacts: bool = True) -> Dict[str, Any]:
        """
        Scrape basic information from a website
        
        Args:
            url: URL to scrape
            extract_contacts: Whether to extract contact information
            
        Returns:
            Dict with scraped information
        """
        try:
            self.metrics["web_scrapes"] += 1
            
            # Check cache
            cache_key = f"scrape:{url}"
            if cache_key in self.cache:
                self.metrics["cache_hits"] += 1
                return self.cache[cache_key]
            
            # Add protocol if missing
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            # Make request
            headers = {"User-Agent": self.config["user_agent"]}
            response = requests.get(url, headers=headers, timeout=self.config["request_timeout"])
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract basic information
            title = soup.find('title')
            title_text = title.get_text().strip() if title else None
            
            # Extract meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            description = meta_desc.get('content') if meta_desc else None
            
            # Extract emails if requested
            emails = []
            if extract_contacts:
                email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                emails = list(set(re.findall(email_pattern, response.text)))
            
            # Extract phone numbers (basic pattern)
            phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
            phones = list(set(re.findall(phone_pattern, response.text)))
            
            # Extract social media links
            social_links = {}
            social_patterns = {
                'linkedin': r'linkedin\.com/(?:company|in)/([^/\s"\']+)',
                'twitter': r'twitter\.com/([^/\s"\']+)',
                'facebook': r'facebook\.com/([^/\s"\']+)',
                'instagram': r'instagram\.com/([^/\s"\']+)'
            }
            
            for platform, pattern in social_patterns.items():
                matches = re.findall(pattern, response.text, re.IGNORECASE)
                if matches:
                    social_links[platform] = list(set(matches))[:3]  # Limit to 3
            
            result = {
                "success": True,
                "url": url,
                "title": title_text,
                "description": description,
                "emails": emails[:10],  # Limit to 10
                "phones": phones[:10],  # Limit to 10
                "social_links": social_links,
                "status_code": response.status_code,
                "timestamp": datetime.now().isoformat(),
                "tool": "web_scraper"
            }
            
            # Cache result
            self.cache[cache_key] = result
            
            logger.info(f"✅ Web scraping successful for {url}")
            return result
            
        except requests.RequestException as e:
            self.metrics["errors"] += 1
            logger.error(f"❌ Web scraping error for {url}: {e}")
            return {
                "success": False,
                "error": f"Request failed: {str(e)}",
                "url": url
            }
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"❌ Web scraping error for {url}: {e}")
            return {
                "success": False,
                "error": str(e),
                "url": url
            }
    
    # ==================== SSL CERTIFICATE INFO ====================
    
    async def get_ssl_info(self, domain: str, port: int = 443) -> Dict[str, Any]:
        """
        Get SSL certificate information for a domain
        
        Args:
            domain: Domain name
            port: SSL port (default: 443)
            
        Returns:
            Dict with SSL certificate information
        """
        try:
            self.metrics["ssl_checks"] += 1
            
            # Check cache
            cache_key = f"ssl:{domain}:{port}"
            if cache_key in self.cache:
                self.metrics["cache_hits"] += 1
                return self.cache[cache_key]
            
            # Remove protocol if present
            domain = domain.replace('http://', '').replace('https://', '').split('/')[0]
            
            # Create SSL context
            context = ssl.create_default_context()
            
            # Connect and get certificate
            with socket.create_connection((domain, port), timeout=self.config["request_timeout"]) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert = ssock.getpeercert()
            
            # Parse certificate information
            result = {
                "success": True,
                "domain": domain,
                "port": port,
                "subject": dict(x[0] for x in cert.get('subject', [])),
                "issuer": dict(x[0] for x in cert.get('issuer', [])),
                "version": cert.get('version'),
                "serial_number": cert.get('serialNumber'),
                "not_before": cert.get('notBefore'),
                "not_after": cert.get('notAfter'),
                "subject_alt_names": [x[1] for x in cert.get('subjectAltName', [])],
                "timestamp": datetime.now().isoformat(),
                "tool": "ssl_checker"
            }
            
            # Cache result
            self.cache[cache_key] = result
            
            logger.info(f"✅ SSL check successful for {domain}")
            return result
            
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"❌ SSL check error for {domain}: {e}")
            return {
                "success": False,
                "error": str(e),
                "domain": domain,
                "port": port
            }
    
    # ==================== UTILITY METHODS ====================
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics"""
        return {
            **self.metrics,
            "cache_size": len(self.cache),
            "timestamp": datetime.now().isoformat()
        }
    
    def clear_cache(self):
        """Clear the results cache"""
        self.cache.clear()
        logger.info("✅ Cache cleared")

