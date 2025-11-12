"""
GRETA PAI OSINT Free Tools Router
API endpoints for free OSINT tools (no API keys required)
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging

from services.osint_free_tools import OSINTFreeToolsService
from middleware.rate_limit import limiter, get_rate_limit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/osint/free", tags=["OSINT Free Tools"])

# Global service instance
osint_service: Optional[OSINTFreeToolsService] = None


def get_osint_service() -> OSINTFreeToolsService:
    """Get OSINT service instance"""
    global osint_service
    if osint_service is None:
        osint_service = OSINTFreeToolsService()
    return osint_service


# ==================== REQUEST/RESPONSE MODELS ====================

class DNSLookupRequest(BaseModel):
    domain: str = Field(..., description="Domain name to lookup")


class WHOISLookupRequest(BaseModel):
    domain: str = Field(..., description="Domain name to lookup")


class WebScrapeRequest(BaseModel):
    url: str = Field(..., description="URL to scrape")
    extract_contacts: bool = Field(True, description="Extract contact information")


class SSLCheckRequest(BaseModel):
    domain: str = Field(..., description="Domain name to check")
    port: int = Field(443, description="SSL port (default: 443)")


class OSINTResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ==================== API ENDPOINTS ====================

@router.post("/dns/lookup", response_model=OSINTResponse)
@limiter.limit(get_rate_limit("osint_free"))
async def dns_lookup(request: DNSLookupRequest):
    """
    Perform DNS lookup for a domain
    
    **Free Tool** - No API key required
    
    Returns:
    - IP address
    - All IP addresses
    - Reverse DNS
    """
    try:
        service = get_osint_service()
        result = await service.dns_lookup(request.domain)
        
        if result.get("success"):
            return OSINTResponse(success=True, data=result)
        else:
            return OSINTResponse(
                success=False,
                error=result.get("error", "DNS lookup failed")
            )
            
    except Exception as e:
        logger.error(f"DNS lookup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/whois/lookup", response_model=OSINTResponse)
@limiter.limit(get_rate_limit("osint_free"))
async def whois_lookup(request: WHOISLookupRequest):
    """
    Perform WHOIS lookup for a domain
    
    **Free Tool** - No API key required
    
    Returns:
    - Registrar information
    - Creation/expiration dates
    - Name servers
    - Contact information
    """
    try:
        service = get_osint_service()
        result = await service.whois_lookup(request.domain)
        
        if result.get("success") or result.get("mock_mode"):
            return OSINTResponse(success=True, data=result)
        else:
            return OSINTResponse(
                success=False,
                error=result.get("error", "WHOIS lookup failed")
            )
            
    except Exception as e:
        logger.error(f"WHOIS lookup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/web/scrape", response_model=OSINTResponse)
@limiter.limit(get_rate_limit("osint_free"))
async def web_scrape(request: WebScrapeRequest):
    """
    Scrape basic information from a website
    
    **Free Tool** - No API key required
    
    Returns:
    - Page title and description
    - Email addresses
    - Phone numbers
    - Social media links
    """
    try:
        service = get_osint_service()
        result = await service.scrape_website(
            request.url,
            extract_contacts=request.extract_contacts
        )
        
        if result.get("success"):
            return OSINTResponse(success=True, data=result)
        else:
            return OSINTResponse(
                success=False,
                error=result.get("error", "Web scraping failed")
            )
            
    except Exception as e:
        logger.error(f"Web scraping error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ssl/check", response_model=OSINTResponse)
@limiter.limit(get_rate_limit("osint_free"))
async def ssl_check(request: SSLCheckRequest):
    """
    Get SSL certificate information for a domain
    
    **Free Tool** - No API key required
    
    Returns:
    - Certificate subject and issuer
    - Validity dates
    - Subject alternative names
    - Certificate version and serial number
    """
    try:
        service = get_osint_service()
        result = await service.get_ssl_info(request.domain, request.port)
        
        if result.get("success"):
            return OSINTResponse(success=True, data=result)
        else:
            return OSINTResponse(
                success=False,
                error=result.get("error", "SSL check failed")
            )
            
    except Exception as e:
        logger.error(f"SSL check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_metrics():
    """
    Get OSINT free tools metrics
    
    Returns usage statistics and cache information
    """
    try:
        service = get_osint_service()
        metrics = service.get_metrics()
        return {"success": True, "metrics": metrics}
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/clear")
async def clear_cache():
    """
    Clear the OSINT tools cache
    
    Useful for forcing fresh lookups
    """
    try:
        service = get_osint_service()
        service.clear_cache()
        return {"success": True, "message": "Cache cleared"}
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "OSINT Free Tools",
        "tools": ["DNS Lookup", "WHOIS Lookup", "Web Scraping", "SSL Certificate Check"]
    }


# ==================== INITIALIZATION ====================

@router.on_event("startup")
async def startup_event():
    """Initialize OSINT service on startup"""
    global osint_service
    osint_service = OSINTFreeToolsService()
    await osint_service.initialize()
    logger.info("✅ OSINT Free Tools Router initialized")

